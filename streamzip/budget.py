"""Frozen StreamZip settings and attainable visual-token budgets."""
import math
SEED = 20261006
BASE_MODEL_SEED = 20261003
CHUNK_SECONDS = 8.0
CANDIDATES = (0, 8, 16, 32)
VIEWS = ('FULL', '50', '25', '12_5')
RATIOS = (1.0, 0.5, 0.25, 0.125)
LAYERS = tuple(range(28, 36))
TARGETS = tuple((f'*.language_model.decoder.layers.{i}.{part}' for i in LAYERS for part in ('self_attention.linear_q', 'self_attention.linear_k', 'self_attention.linear_v', 'self_attention.linear_proj', 'mlp.linear_fc1', 'mlp.linear_fc2')))

def budget_id(ratio):
    """Map the four supported visual-token ratios to learned vectors."""
    if isinstance(ratio, bool) or ratio not in RATIOS:
        raise ValueError('budget must be 1.0, 0.5, 0.25 or 0.125')
    return RATIOS.index(ratio)

def chunk_intervals(timestamp):
    if not math.isfinite(timestamp) or timestamp <= 0:
        raise ValueError('positive finite query time required')
    return [(i * CHUNK_SECONDS, min((i + 1) * CHUNK_SECONDS, timestamp)) for i in range(math.ceil(timestamp / CHUNK_SECONDS))]

def sample_times(start, end):
    if not 0 <= start < end:
        raise ValueError('invalid chunk interval')
    return [start + (end - start) * (i + 0.5) / 4 for i in range(4)]

def token_budget(chunks, ratio):
    if isinstance(chunks, bool) or not isinstance(chunks, int) or chunks < 1:
        raise ValueError('positive integer chunk count required')
    if ratio == 1.0 and (not isinstance(ratio, bool)):
        return 32 * chunks
    if isinstance(ratio, bool) or ratio not in RATIOS[1:]:
        raise ValueError('unsupported relative budget')
    return min(32 * chunks, max(8, 8 * math.floor(32 * chunks * ratio / 8 + 0.5)))

def allocate_feasible(scores, chunks, budget, current_id):
    """Largest attainable equal spend; leave the original exact DP unchanged.

    One 32-token arrival cannot realize 24 tokens, and two cannot realize 56.
    Such early prefixes use the closest feasible value below the fixed cap.
    """
    from .memory import feasibility, allocate_scores
    mask = feasibility(chunks, current_id)
    reachable = {0}
    for row in mask:
        reachable = {previous + k for previous in reachable for k, ok in zip(CANDIDATES, row.tolist()) if ok and previous + k <= budget}
    if not reachable:
        raise ValueError('no feasible current-positive allocation')
    target = max(reachable)
    return allocate_scores(scores, chunks, target, current_id)

def approved_parameter(name):
    import re
    if name.endswith('language_model.streamzip_budget.weight'):
        return True
    match = re.search('language_model.decoder.layers\\.(\\d+)\\.(self_attention|mlp)\\.(linear_qkv|linear_proj|linear_fc1|linear_fc2)\\.adapter\\.(.+)', name)
    if match is None or int(match[1]) not in LAYERS:
        return False
    branch, linear, tail = match.group(2, 3, 4)
    if linear == 'linear_qkv':
        return branch == 'self_attention' and bool(re.fullmatch('adapter_[qkv]\\.(linear_in|linear_out)\\.weight', tail))
    if (branch, linear) not in (('self_attention', 'linear_proj'), ('mlp', 'linear_fc1'), ('mlp', 'linear_fc2')):
        return False
    return tail in ('linear_in.weight', 'linear_out.weight')
