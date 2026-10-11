"""Causal PTS decoding, answer-free prompts and independently bounded views."""
from dataclasses import dataclass, replace
from pathlib import Path
import json
import math
import numpy as np
import torch
from PIL import Image, ImageOps
from .budget import chunk_intervals, sample_times, token_budget, allocate_feasible, budget_id, VIEWS
from .visual import VideoChunkMetadata
from .compression import with_priority
from .memory import BundleMemory, recency_scores, commit, verify_commit
from .sparse import DenseVideoPrompt, encode_bundle
from .utils import packed_bundle, unpack_bundle
from .inputs import find_question_start

@dataclass(frozen=True)
class SampledChunk:
    frames: torch.Tensor
    times: tuple[float, ...]
    start: float
    end: float

class VideoReader:

    def __init__(self, path):
        import av
        self.container = av.open(str(path))
        self.stream = self.container.streams.video[0]
        self.origin = float((self.stream.start_time or 0) * self.stream.time_base)
        self.fps = float(self.stream.average_rate)
        self.duration = float(self.stream.duration * self.stream.time_base) if self.stream.duration is not None else float(self.container.duration / av.time_base)
        self.width = self.stream.codec_context.width
        self.height = self.stream.codec_context.height

    def close(self):
        self.container.close()

    def frame_before(self, target):
        self.container.seek(int((self.origin + target) / self.stream.time_base), stream=self.stream, backward=True, any_frame=False)
        chosen = None
        time = None
        for frame in self.container.decode(self.stream):
            if frame.pts is None:
                continue
            actual = float(frame.pts * self.stream.time_base) - self.origin
            if actual > target + 1e-08:
                break
            if actual >= -1e-08:
                chosen = frame
                time = max(0.0, actual)
        if chosen is None:
            raise ValueError(f'no causal frame at {target}')
        image = ImageOps.pad(chosen.to_image().convert('RGB'), (128, 128), method=Image.Resampling.BICUBIC, color=(0, 0, 0), centering=(0.5, 0.5))
        tensor = torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).contiguous()
        return (tensor, time)

    def chunk(self, start, end):
        if end > self.duration + 0.05:
            raise ValueError('query extends beyond source duration')
        images = []
        times = []
        for target in sample_times(start, end):
            image, actual = self.frame_before(target)
            images.append(image)
            times.append(actual)
        if not all((t <= end and t <= requested + 1e-08 for t, requested in zip(times, sample_times(start, end)))):
            raise ValueError('future sample')
        return SampledChunk(torch.stack(images), tuple(times), start, end)

    def prefix(self, timestamp):
        return tuple((self.chunk(a, b) for a, b in chunk_intervals(timestamp)))

def processor_batch(processor, chunks, fps, query, device):
    content = []
    for c in chunks:
        content.extend(({'type': 'text', 'text': f'Video interval {c['start']:.3f} to {c['end']:.3f} seconds: '}, {'type': 'video'}))
    content.append({'type': 'text', 'text': query})
    text = processor.apply_chat_template([{'role': 'user', 'content': content}], tokenize=False, add_generation_prompt=True)
    metadata = [dict(total_num_frames=max(4, int(c['end'] * fps) + 1), fps=fps, duration=c['end'], frames_indices=[t * fps for t in c['times']]) for c in chunks]
    batch = dict(processor(text=[text], videos=[c['frames'] for c in chunks], video_metadata=metadata, do_sample_frames=False, do_resize=False, cap_pixels_per_frame=False, return_tensors='pt'))
    return {k: v.to(device) for k, v in batch.items()}

def positions(model, batch):
    from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.rope import get_rope_index
    p, _ = get_rope_index(model.config.spatial_merge_size, model.image_token_id, model.video_token_id, model.vision_start_token_id, batch['input_ids'], video_grid_thw=batch['video_grid_thw'], attention_mask=batch['attention_mask'])
    return p

def question_text(record):
    options = '; '.join((f'{chr(65 + i)}. {v}' for i, v in enumerate(record['options']))) + ';'
    return f'Question: {record['question']}\nOptions:\n{options}\n\nRespond only with the letter corresponding to your chosen option (e.g., A, B, C).\nDo not include any additional text or explanation in your response.'

def parse_record(value, base, *, require_answer=True):
    """Normalize the documented four-choice schema without putting labels in prompts."""
    if not isinstance(value, dict):
        raise ValueError('manifest records must be objects')
    video = value.get('video')
    choices = value.get('choices')
    question = value.get('question')
    timestamp = value.get('timestamp')
    if not isinstance(video, str) or not video.strip():
        raise ValueError('video path required')
    if not isinstance(question, str) or not question.strip():
        raise ValueError('nonempty question required')
    if not isinstance(choices, list) or len(choices) != 4 or any((not isinstance(x, str) or not x.strip() for x in choices)):
        raise ValueError('exactly four nonempty choices required')
    if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)) or (not math.isfinite(timestamp)) or (timestamp <= 0):
        raise ValueError('positive finite causal timestamp required')
    identifier = value.get('id')
    if isinstance(identifier, bool) or not isinstance(identifier, (str, int)) or (not str(identifier)):
        raise ValueError('a unique string or integer id is required')
    path = Path(video).expanduser()
    if not path.is_absolute():
        path = Path(base) / path
    result = dict(id=identifier, video=str(path.resolve()), source_id=value.get('source_id', video), question=question, options=list(choices), realtime=float(timestamp))
    if not isinstance(result['source_id'], str) or not result['source_id']:
        raise ValueError('source_id must be a nonempty string')
    gt = value.get('answer')
    if require_answer or gt is not None:
        if isinstance(gt, bool) or not isinstance(gt, int) or (not 0 <= gt < 4):
            raise ValueError('answer must be an index from 0 to 3')
        result['gt'] = gt
    return result

def load_manifest(path, *, require_answer=True):
    """Read a JSON array. Relative video paths resolve against the manifest directory."""
    path = Path(path)
    values = json.loads(path.read_text())
    if not isinstance(values, list) or not values:
        raise ValueError('manifest must be a nonempty JSON array')
    rows = [parse_record(v, path.parent, require_answer=require_answer) for v in values]
    if len({str(r['id']) for r in rows}) != len(rows):
        raise ValueError('duplicate QA IDs')
    return rows

def query_only(record):
    """Only question and options may reach the multimodal prompt."""
    return {k: record[k] for k in ('question', 'options')}

@torch.no_grad()
def build_context(model, processor, record, ratios):
    """Replay the verified frozen vision/Recency logic with no recoverable hidden archive.

    Each requested view owns its independently bounded memory. Inference requests
    one view. Training also requests FULL for the frozen teacher and all four views.
    Only scalar timestamp metadata survive independently of these memories.
    """
    ratios = tuple(ratios)
    if not ratios or len(set(ratios)) != len(ratios):
        raise ValueError('distinct requested budgets required')
    count = len(chunk_intervals(record['realtime']))
    memories = {VIEWS[budget_id(r)]: BundleMemory((), token_budget(count, r)) for r in ratios}
    history = {v: [] for v in memories}
    prompt_metadata = []
    device = next(model.parameters()).device
    reader = VideoReader(record['video'])
    try:
        if record['realtime'] > reader.duration + 0.05:
            raise ValueError('query exceeds actual video duration')
        for i, (start, end) in enumerate(chunk_intervals(record['realtime'])):
            sampled = reader.chunk(start, end)
            chunk = dict(frames=sampled.frames, times=sampled.times, start=start, end=end)
            batch = processor_batch(processor, [chunk], reader.fps, 'Describe this video interval.', device)
            batch['position_ids'] = positions(model, batch)
            _, bundle = encode_bundle(model, batch, VideoChunkMetadata(record['source_id'], i, reader.fps, tuple((round(t * reader.fps) for t in chunk['times']))))
            if batch['video_grid_thw'].tolist() != [[2, 8, 8]] or len(bundle.final_features) != 32:
                raise ValueError('the qualified 128-square/four-frame geometry requires 32 tokens per chunk')
            times = torch.tensor(chunk['times'], device=device, dtype=torch.float32).view(2, 2).mean(1).repeat_interleave(16)
            bundle = with_priority(replace(bundle, original_token_index=bundle.original_token_index + 32 * i, timestamps=times))
            current = unpack_bundle(packed_bundle(bundle), 'cpu')
            for view, memory in memories.items():
                active = memory.active(current, end)
                costs = allocate_feasible(recency_scores(active), active, memory.budget, i)
                updated = commit(memory, current, end, costs)
                verify_commit(memory, current, updated, costs)
                memories[view] = updated
                history[view].append(dict(arrival=i, spend=sum(costs), cap=memory.budget, costs=costs, retained=[c.original_token_index.tolist() for c in updated.chunks]))
            prompt_metadata.append(dict(times=list(chunk['times']), start=start, end=end))
            del batch, bundle, current, chunk, sampled
        prompt_chunks = [dict(**m, frames=torch.zeros(4, 3, 128, 128, dtype=torch.uint8)) for m in prompt_metadata]
        text = question_text(query_only(record))
        batch = processor_batch(processor, prompt_chunks, reader.fps, text, device)
        if batch['video_grid_thw'].tolist() != [[2, 8, 8]] * count:
            raise ValueError('dense prompt grid changed')
        p = positions(model, batch)
        dense = DenseVideoPrompt(*(batch[k].cpu() for k in ('input_ids', 'attention_mask', 'mm_token_type_ids')), p.cpu(), model.video_token_id)
        visual = (dense.input_ids[0] == model.video_token_id).nonzero().flatten()
        if len(visual) != 32 * count:
            raise ValueError('original visual placeholders changed')
        query_ids = torch.tensor(processor.tokenizer.encode(text, add_special_tokens=False))
        boundary = find_question_start(dense.input_ids, query_ids, int(visual[-1]))
        return dict(dense=dense, views={v: m.chunks for v, m in memories.items()}, question_start=boundary, metadata=prompt_metadata, audit={v: dict(cap=m.budget, spend=sum((len(c.final_features) for c in m.chunks)), history=history[v], irreversible=True, violations=0) for v, m in memories.items()})
    finally:
        reader.close()
