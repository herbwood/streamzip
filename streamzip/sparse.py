"""Original sparse sequence positions and aligned native DeepStack insertion."""
from dataclasses import dataclass, replace
import torch
from torch import Tensor
from .visual import VisualTokenBundle

@dataclass(frozen=True)
class DenseVideoPrompt:
    input_ids: Tensor
    attention_mask: Tensor
    mm_token_type_ids: Tensor
    position_ids: Tensor
    video_token_id: int

@dataclass(frozen=True)
class SparsePrefill:
    input_ids: Tensor
    attention_mask: Tensor
    mm_token_type_ids: Tensor
    position_ids: Tensor
    inputs_embeds: Tensor
    visual_pos_masks: Tensor
    original_prompt_positions: Tensor
    next_mrope_position: int
    bundle: VisualTokenBundle

def build_sparse_prefill(model, dense: DenseVideoPrompt, bundle: VisualTokenBundle) -> SparsePrefill:
    bundle.validate()
    if dense.input_ids.shape[0] != 1 or not dense.attention_mask.bool().all():
        raise ValueError('sparse prefill supports one unpadded prompt')
    original_visual = dense.input_ids[0] == dense.video_token_id
    if (bundle.prompt_positions >= dense.input_ids.shape[1]).any() or not original_visual[bundle.prompt_positions].all():
        raise ValueError('bundle points outside original video placeholders')
    if not torch.equal(bundle.mrope_coordinates, dense.position_ids[:, 0, bundle.prompt_positions].T):
        raise ValueError('bundle M-RoPE provenance disagrees with original prompt')
    keep = ~original_visual
    keep = keep.clone()
    keep[bundle.prompt_positions] = True
    indices = keep.nonzero(as_tuple=True)[0]
    ids = dense.input_ids[:, indices].clone()
    mask = dense.attention_mask[:, indices].clone()
    modality = dense.mm_token_type_ids[:, indices].clone()
    positions = dense.position_ids[:, :, indices].clone()
    visual_mask = ids == dense.video_token_id
    if int(visual_mask.sum()) != bundle.final_features.shape[0] or not torch.equal(modality == 2, visual_mask):
        raise ValueError('placeholder/modality count or order mismatch')
    embeddings = model.get_input_embeddings()(ids).clone()
    embeddings[visual_mask] = bundle.final_features
    return SparsePrefill(ids, mask, modality, positions, embeddings, visual_mask, indices, int(positions.max()) + 1, bundle)

class _EmbeddingView:

    def __init__(self, language_model):
        self.language_model = language_model

    def get_input_embeddings(self):

        def embed(ids):
            return self.language_model.embedding(input_ids=ids, position_ids=None).transpose(0, 1).contiguous()
        return embed

def encode_bundle(model, batch, metadata):
    """Run frozen native vision once; explicit original positions are required.

    Grid metadata is used only for the original dense encoding/provenance.
    Sparse language execution never reconstructs a rectangular grid.
    """
    cfg = model.config
    if cfg.sequence_parallel or cfg.context_parallel_size != 1 or cfg.pipeline_model_parallel_size != 1:
        raise ValueError('StreamZip supports CP=PP=1 and SP=false only')
    ids = batch['input_ids']
    grid = batch['video_grid_thw']
    mask = batch.get('attention_mask', torch.ones_like(ids))
    if ids.shape[0] != 1 or not mask.bool().all() or grid.shape != (1, 3) or ('pixel_values' in batch):
        raise ValueError('one unpadded video and batch one required')
    positions = batch['position_ids']
    if positions.shape != (3, 1, ids.shape[1]):
        raise ValueError('explicit original three-axis positions required')
    prompt_positions = (ids[0] == model.video_token_id).nonzero(as_tuple=True)[0]
    with torch.no_grad():
        final, companions = model.vision_model(hidden_states=batch['pixel_values_videos'], grid_thw=grid)
    t, h, w = map(int, grid[0].tolist())
    merge = cfg.spatial_merge_size
    rows, cols = (h // merge, w // merge)
    if h % merge or w % merge or final.shape != (t * rows * cols, cfg.hidden_size):
        raise ValueError('native grid/merger/width disagree')
    device = final.device
    temporal = torch.arange(t, device=device).repeat_interleave(rows * cols)
    spatial = torch.stack(torch.meshgrid(torch.arange(rows, device=device), torch.arange(cols, device=device), indexing='ij'), -1).reshape(-1, 2).repeat(t, 1)
    frames = metadata.frame_indices
    if len(frames) % t:
        raise ValueError('source frames must match temporal grouping')
    patch = len(frames) // t
    times = torch.tensor([(frames[i * patch] + frames[(i + 1) * patch - 1]) / (2 * metadata.fps) for i in range(t)], device=device, dtype=torch.float32)
    bundle = VisualTokenBundle(final.detach().clone(), tuple((x.detach().clone() for x in companions)), temporal, spatial, times[temporal], positions[:, 0, prompt_positions].T.contiguous(), prompt_positions, torch.arange(final.shape[0], device=device), metadata)
    bundle.validate()
    dense = DenseVideoPrompt(ids.clone(), mask.clone(), batch['mm_token_type_ids'].clone(), positions.clone(), model.video_token_id)
    return (dense, bundle)

def sparse_prefill(language_model, dense, bundle):
    """Reuse the approved width-independent HF sparse sequence/provenance logic."""
    return build_sparse_prefill(_EmbeddingView(language_model), dense, bundle)

def language_kwargs(sparse, **extra):
    """Core uses [L,B,D] decoder inputs and [B,L] visual masks.

    Unpadded causal attention uses None, matching native outer behavior; the
    retained 2D keep-mask is still validated by sparse sequence construction.
    """
    return dict(input_ids=sparse.input_ids, position_ids=sparse.position_ids, attention_mask=None, decoder_input=sparse.inputs_embeds.transpose(0, 1).contiguous(), visual_pos_masks=sparse.visual_pos_masks, deepstack_visual_embeds=list(sparse.bundle.deepstack_features), **extra)

def combine(chunks, dense):
    """Reanchor compact original token indices to a new question's dense prompt."""
    if not chunks:
        raise ValueError('current-positive policy requires a nonempty bundle')
    fields = ('final_features', 'temporal_coordinates', 'spatial_coordinates', 'timestamps', 'original_token_index')
    d = {f: torch.cat([getattr(c, f) for c in chunks]) for f in fields}
    d['deepstack_features'] = tuple((torch.cat([c.deepstack_features[i] for c in chunks]) for i in range(3)))
    placeholders = (dense.input_ids[0] == dense.video_token_id).nonzero(as_tuple=True)[0]
    d['prompt_positions'] = placeholders[d['original_token_index']]
    d['mrope_coordinates'] = dense.position_ids[:, 0, d['prompt_positions']].T.contiguous()
    first = chunks[0]
    d['metadata'] = replace(first.metadata, chunk_id=-1)
    out = VisualTokenBundle(**d)
    out.validate()
    return out
