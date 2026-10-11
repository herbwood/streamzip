import torch
from torch import nn
import pytest
from streamzip.memory import build_views
from streamzip.sparse import DenseVideoPrompt, build_sparse_prefill, combine
from streamzip.inputs import BudgetEmbedding, insert_budget, find_question_start
from streamzip.data import parse_record, query_only, question_text, load_manifest, VideoReader

def fixture(arrivals):
    ids = torch.tensor([[1] + [100] * 96 + [4, 5, 6]])
    p = torch.arange(100).reshape(1, 1, -1).repeat(3, 1, 1)
    for c in arrivals:
        p[:, 0, c.prompt_positions] = c.mrope_coordinates.T
    dense = DenseVideoPrompt(ids, torch.ones_like(ids), torch.where(ids == 100, 2, 0), p, 100)
    views, _ = build_views(arrivals, [8.0, 16.0, 24.0])

    class Model:

        def get_input_embeddings(self):
            return nn.Embedding(128, 8)
    return (build_sparse_prefill(Model(), dense, combine(views['25'], dense)), dense)

def test_sparse_positions_budget_token_shift_and_gradients(arrivals):
    sparse, dense = fixture(arrivals)
    budget = BudgetEmbedding(8, seed=3)
    a, at = insert_budget(sparse, budget, 0, 97, 7)
    b, _ = insert_budget(sparse, budget, 1, 97, 7)
    assert torch.equal(a.input_ids, b.input_ids) and (not torch.equal(a.inputs_embeds[0, at], b.inputs_embeds[0, at]))
    assert torch.equal(a.position_ids[:, :, at + 1:], sparse.position_ids[:, :, at:] + 1)
    assert torch.equal(a.position_ids[:, 0, a.visual_pos_masks[0]].T, a.bundle.mrope_coordinates)
    assert torch.equal(a.inputs_embeds[0, a.visual_pos_masks[0]], a.bundle.final_features)
    a.inputs_embeds.sum().backward()
    assert budget.weight.grad[0].abs().sum() > 0 and budget.weight.grad[1:].count_nonzero() == 0
    assert find_question_start(dense.input_ids, torch.tensor([4, 5, 6]), 96) == 97
    with pytest.raises(ValueError):
        insert_budget(sparse, budget, 0, 1, 100)

def test_manifest_labels_never_enter_prompt(tmp_path):
    import json
    raw = dict(id='demo', video='clip.mp4', question='Which object?', choices=['red', 'blue', 'green', 'yellow'], answer=1, timestamp=9.0)
    p = tmp_path / 'manifest.json'
    p.write_text(json.dumps([raw]))
    record = load_manifest(p)[0]
    assert record['video'] == str(tmp_path / 'clip.mp4')
    assert query_only(record) == {'question': raw['question'], 'options': raw['choices']}
    assert question_text(record) == question_text(dict(record, gt=3))
    with pytest.raises(ValueError):
        parse_record(dict(raw, answer=True), tmp_path)
    with pytest.raises(ValueError):
        parse_record(dict(raw, timestamp=0), tmp_path)
    p.write_text(json.dumps([raw, raw]))
    with pytest.raises(ValueError):
        load_manifest(p)

def test_actual_decoder_uses_display_pts_before_target(tmp_path):
    import av
    import numpy as np
    path = tmp_path / 'synthetic.mkv'
    with av.open(str(path), 'w') as output:
        stream = output.add_stream('ffv1', rate=4)
        stream.width = 16
        stream.height = 16
        stream.pix_fmt = 'bgr0'
        for i in range(16):
            frame = av.VideoFrame.from_ndarray(np.full((16, 16, 3), i * 8, dtype=np.uint8), format='rgb24')
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    reader = VideoReader(path)
    try:
        pixels, t = reader.frame_before(2.125)
        assert t == 2.0 and pixels.shape == (3, 128, 128)
        chunk = reader.chunk(0.0, 4.0)
        assert list(chunk.times) == [0.5, 1.5, 2.5, 3.5] and max(chunk.times) <= 4.0
    finally:
        reader.close()
