import torch
import pytest
from streamzip.memory import build_views, BundleMemory, commit
from streamzip.compression import FeatureCoresetSelector, coreset_order

def test_all_views_irreversible_and_streams_aligned(arrivals):
    views, audits = build_views(arrivals, [8.0, 16.0, 24.0])
    assert [sum((len(c.final_features) for c in views[v])) for v in ('FULL', '50', '25', '12_5')] == [96, 48, 24, 16]
    full = torch.cat([c.final_features for c in arrivals])
    for chunks in views.values():
        for c in chunks:
            torch.testing.assert_close(c.final_features, full[c.original_token_index], rtol=0, atol=0)
            for i, z in enumerate(c.deepstack_features):
                torch.testing.assert_close(z, full[c.original_token_index] + 10000 * (i + 1), rtol=0, atol=0)
            assert c.final_features.untyped_storage().nbytes() == c.final_features.numel() * c.final_features.element_size()
    for audit in audits.values():
        previous = set()
        for h in audit['history']:
            retained = {i for ids in h['retained'] for i in ids}
            current = set(range(h['arrival'] * 32, (h['arrival'] + 1) * 32))
            assert retained <= previous | current
            assert h['spend'] == sum(h['costs']) <= h['cap'] and h['costs'][-1] >= 8
            previous = retained

def test_deterministic_temporal_balance_and_nested_subsets(arrivals):
    selector = FeatureCoresetSelector()
    original = arrivals[0]
    small = selector.select(original, 8)
    middle = selector.select(original, 16)
    assert torch.equal(small.original_token_index, selector.select(original, 8).original_token_index)
    assert set(small.original_token_index.tolist()) <= set(middle.original_token_index.tolist())
    assert torch.equal(small.original_token_index, selector.select(middle, 8).original_token_index)
    assert [(small.temporal_coordinates == t).sum().item() for t in (0, 1)] == [4, 4]
    assert torch.equal(coreset_order(original.final_features, original.temporal_coordinates), coreset_order(original.final_features, original.temporal_coordinates))

def test_no_expansion_future_or_current_eviction(arrivals):
    memory = BundleMemory((), 8)
    with pytest.raises(ValueError):
        memory.active(arrivals[0], 1.0)
    with pytest.raises(ValueError):
        commit(memory, arrivals[0], 8.0, [0])
    memory = commit(memory, arrivals[0], 8.0, [8])
    with pytest.raises(ValueError):
        commit(memory, arrivals[1], 16.0, [16, 8])
    with pytest.raises(ValueError):
        memory.active(arrivals[2], 24.0)
