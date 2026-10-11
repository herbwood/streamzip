"""Exercise the public causal orchestration with frozen synthetic feature arrivals."""
from dataclasses import replace
import torch
from streamzip import data
from streamzip.model import StreamZipModel

def test_context_only_exposes_requested_bounded_views(monkeypatch, arrivals):

    class Reader:
        duration = 24.0
        fps = 1.0

        def __init__(self, path):
            self.closed = False

        def chunk(self, start, end):
            return data.SampledChunk(torch.zeros(4, 3, 128, 128, dtype=torch.uint8), tuple(data.sample_times(start, end)), start, end)

        def close(self):
            self.closed = True

    class Tokenizer:

        def encode(self, text, add_special_tokens=False):
            return [4, 5, 6]

    class Processor:
        tokenizer = Tokenizer()

        def apply_chat_template(self, messages, **kwargs):
            return ''.join((c.get('text', '') for c in messages[0]['content']))

        def __call__(self, *, text, videos, video_metadata, **kwargs):
            assert not kwargs['do_sample_frames'] and (not kwargs['do_resize'])
            n = len(videos)
            ids = torch.tensor([[1] + [100] * (32 * n) + [4, 5, 6]])
            return dict(input_ids=ids, attention_mask=torch.ones_like(ids), mm_token_type_ids=torch.where(ids == 100, 2, 0), video_grid_thw=torch.tensor([[2, 8, 8]] * n))

    class Model:
        video_token_id = 100

        def parameters(self):
            return iter([torch.zeros(1)])
    calls = []

    def encode(model, batch, metadata):
        calls.append(metadata.chunk_id)
        c = arrivals[metadata.chunk_id]
        return (None, replace(c, original_token_index=c.original_token_index - 32 * metadata.chunk_id))
    monkeypatch.setattr(data, 'VideoReader', Reader)
    monkeypatch.setattr(data, 'encode_bundle', encode)
    monkeypatch.setattr(data, 'positions', lambda model, batch: torch.arange(batch['input_ids'].shape[1]).reshape(1, 1, -1).repeat(3, 1, 1))
    record = dict(id='synthetic', video='unused.mp4', source_id='synthetic-video', realtime=24.0, question='Which object?', options=['a', 'b', 'c', 'd'])
    context = data.build_context(Model(), Processor(), record, (0.25,))
    assert calls == [0, 1, 2]
    assert set(context['views']) == {'25'}
    assert sum((len(c.final_features) for c in context['views']['25'])) == 24
    assert not any((k in context for k in ('arrivals', 'archive', 'discarded_features')))
    assert context['question_start'] == 97
    model = object.__new__(StreamZipModel)
    model.attach_context(record, context)
    assert set(model.views) == {'25'}
    model.clear_context()
    assert model.bank_id is None and (not hasattr(model, 'views'))
