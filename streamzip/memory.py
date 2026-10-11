"""Bounded irreversible Recency memory; discarded features cannot reappear."""
from dataclasses import dataclass
from functools import lru_cache
from types import SimpleNamespace
import math
import torch
from ._allocation import allocate
from .compression import FeatureCoresetSelector
from .budget import CANDIDATES, VIEWS, RATIOS, token_budget, allocate_feasible

def recency_scores(active):
    """Qualified fixed Recency score: (chunk ID + 1) times token cost."""
    device = active[0].final_features.device
    cost = torch.tensor(CANDIDATES, device=device, dtype=torch.float64)
    return torch.tensor([c.metadata.chunk_id + 1 for c in active], device=device, dtype=torch.float64)[:, None] * cost

@dataclass(frozen=True)
class BundleMemory:
    chunks: tuple
    budget: int
    now: float = -1.0
    next_id: int = 0

    def validate(self):
        if not isinstance(self.budget, int) or self.budget < 0 or (not isinstance(self.next_id, int)) or (self.next_id < 0) or (not math.isfinite(self.now)):
            raise ValueError('invalid memory header')
        ids = [c.metadata.chunk_id for c in self.chunks]
        if ids != sorted(set(ids)) or any((i < 0 or i >= self.next_id for i in ids)):
            raise ValueError('invalid chunk identities')
        if sum((c.final_features.shape[0] for c in self.chunks)) > self.budget:
            raise ValueError('memory exceeds B')
        for c in self.chunks:
            c.validate()
            if c.timestamps.max() > self.now:
                raise ValueError('future token timestamp')
            if any((x.requires_grad for x in (c.final_features, *c.deepstack_features))):
                raise ValueError('state cannot retain autograd')
            for x in (c.final_features, *c.deepstack_features):
                if x.untyped_storage().nbytes() != x.numel() * x.element_size():
                    raise ValueError('state must own compact feature storage')

    def active(self, current, now):
        self.validate()
        current.validate()
        if not math.isfinite(now) or current.timestamps.max() > now:
            raise ValueError('invalid arrival clock')
        if current.metadata.chunk_id != self.next_id or now < self.now:
            raise ValueError('noncausal arrival')
        return (*self.chunks, current)

def feasibility(chunks, current_id, device='cpu'):
    candidates = torch.tensor(CANDIDATES, device=device)
    mask = torch.stack([candidates <= c.final_features.shape[0] for c in chunks])
    for i, c in enumerate(chunks):
        if c.metadata.chunk_id == current_id:
            mask[i, 0] = False
    return mask

def allocate_scores(scores, chunks, budget, current_id, exact=True):
    """Unchanged core <=B DP; exact realized spend is an evaluation control."""
    scores = scores.detach().double().cpu()
    costs = torch.tensor(CANDIDATES)
    feasible = feasibility(chunks, current_id)
    attainable = min(budget, sum((c.final_features.shape[0] for c in chunks)))
    if exact:
        scores = scores + (2 * float(scores.abs().sum()) + 1) * costs.double()
    allocation = allocate(scores[None], costs, feasible[None], torch.ones(1, len(chunks), dtype=torch.bool), torch.tensor([attainable]))
    if exact and int(allocation.total_cost) != attainable:
        raise ValueError('equal realized spend infeasible')
    return allocation.costs[0].tolist()

def commit(memory, current, now, costs):
    chunks = memory.active(current, now)
    if len(costs) != len(chunks) or costs[-1] < 8:
        raise ValueError('current chunk requires >=8 tokens')
    selected = []
    for c, k in zip(chunks, costs):
        if k not in CANDIDATES or k > c.final_features.shape[0]:
            raise ValueError('invalid expansion or cost')
        if k:
            selected.append(FeatureCoresetSelector().select(c, k))
    out = BundleMemory(tuple(selected), memory.budget, now, memory.next_id + 1)
    out.validate()
    return out

def verify_commit(previous, current, new, costs):
    new.validate()
    assert sum(costs) == sum((len(c.final_features) for c in new.chunks)) <= new.budget
    old = {c.metadata.chunk_id: set(c.original_token_index.tolist()) for c in previous.chunks}
    old[current.metadata.chunk_id] = set(current.original_token_index.tolist())
    for c in new.chunks:
        assert set(c.original_token_index.tolist()) <= old[c.metadata.chunk_id], 'historical recovery'
        n = len(c.final_features)
        assert all((len(x) == n for x in c.deepstack_features))
        assert c.mrope_coordinates.shape == (n, 3)

@lru_cache(maxsize=None)
def allocation_schedule(count, ratio):
    """Exact DP depends only on Recency IDs and recoverable token counts.

    Memoize integer trajectories, never visual features. This preserves the
    qualified exact solver and ties while sharing outcome-independent work
    across source videos with identical fixture geometry.
    """
    cap = token_budget(count, ratio)
    retained = []
    history = []
    for i in range(count):
        active = [SimpleNamespace(final_features=torch.empty(k, 0), metadata=SimpleNamespace(chunk_id=c)) for c, k in [*retained, (i, 32)]]
        costs = allocate_feasible(recency_scores(active), active, cap, i)
        history.append(dict(active=[(c.metadata.chunk_id, len(c.final_features)) for c in active], costs=costs))
        retained = [(c.metadata.chunk_id, k) for c, k in zip(active, costs) if k]
    return history

def build_views(arrivals, end_times):
    if not arrivals or len(arrivals) != len(end_times):
        raise ValueError('positive causal chunk sequence required')
    out = {'FULL': tuple(arrivals)}
    audit = {}
    for name, ratio in zip(VIEWS[1:], RATIOS[1:]):
        cap = token_budget(len(arrivals), ratio)
        memory = BundleMemory((), cap)
        history = []
        schedule = allocation_schedule(len(arrivals), ratio)
        for i, (current, now) in enumerate(zip(arrivals, end_times)):
            active = memory.active(current, now)
            assert [(c.metadata.chunk_id, len(c.final_features)) for c in active] == schedule[i]['active']
            costs = schedule[i]['costs']
            updated = commit(memory, current, now, costs)
            verify_commit(memory, current, updated, costs)
            assert costs[-1] >= 8 and sum(costs) <= cap
            history.append(dict(arrival=current.metadata.chunk_id, spend=sum(costs), cap=cap, costs=costs, retained=[c.original_token_index.tolist() for c in updated.chunks]))
            memory = updated
        out[name] = memory.chunks
        audit[name] = dict(cap=cap, spend=sum((len(c.final_features) for c in memory.chunks)), history=history, irreversible=True, violations=0)
    assert tuple(out) == VIEWS
    return (out, audit)
