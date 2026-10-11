import pytest
from streamzip.budget import token_budget, budget_id, chunk_intervals, sample_times

@pytest.mark.parametrize('chunks,expected', [(1, [32, 16, 8, 8]), (3, [96, 48, 24, 16]), (15, [480, 240, 120, 64])])
def test_exact_nominal_caps_and_discrete_rounding(chunks, expected):
    assert [token_budget(chunks, r) for r in (1.0, 0.5, 0.25, 0.125)] == expected

@pytest.mark.parametrize('bad', [True, 0.1, 0.0, 2.0, None])
def test_unsupported_budget_rejected(bad):
    with pytest.raises(ValueError):
        budget_id(bad)

def test_budget_vector_order_and_causal_sampling():
    assert [budget_id(r) for r in (1.0, 0.5, 0.25, 0.125)] == list(range(4))
    assert chunk_intervals(17.0) == [(0.0, 8.0), (8.0, 16.0), (16.0, 17.0)]
    assert sample_times(16.0, 17.0) == [16.125, 16.375, 16.625, 16.875]
    with pytest.raises(ValueError):
        chunk_intervals(float('nan'))
