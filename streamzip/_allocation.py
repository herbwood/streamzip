"""Exact multiple-choice knapsack. No concavity or greedy assumptions."""
import math
import torch
from torch import Tensor
from ._types import Allocation, check_candidates

def allocate(utility: Tensor, candidates: Tensor, candidate_valid: Tensor, chunk_valid: Tensor, budget: Tensor) -> Allocation:
    check_candidates(candidates)
    if utility.ndim != 3 or not utility.is_floating_point() or utility.shape[-1] != candidates.numel():
        raise ValueError('utility must be floating [S,A,J]')
    s, a, _ = utility.shape
    if candidate_valid.shape != utility.shape or candidate_valid.dtype != torch.bool or chunk_valid.shape != (s, a) or (chunk_valid.dtype != torch.bool):
        raise ValueError('allocation mask shapes/dtypes disagree')
    if budget.shape != (s,) or budget.dtype != torch.int64 or (budget < 0).any():
        raise ValueError('budget must be nonnegative int64 [S]')
    if any((x.device != utility.device for x in (candidates, candidate_valid, chunk_valid, budget))):
        raise ValueError('allocator inputs must share a device')
    if not torch.isfinite(utility[candidate_valid & chunk_valid.unsqueeze(-1)]).all():
        raise ValueError('feasible utilities must be finite')
    values = utility.detach().double().cpu()
    feasible = candidate_valid.cpu()
    active = chunk_valid.cpu()
    costs_list = candidates.tolist()
    unit = math.gcd(*costs_list) or 1
    costs = torch.zeros((s, a), dtype=torch.int64, device=utility.device)
    indices = torch.full_like(costs, -1)
    totals = torch.zeros(s, dtype=torch.int64, device=utility.device)
    objectives = utility.new_zeros(s)
    for b in range(s):
        cap = int(budget[b]) // unit
        dp = {0: (0.0, 0)}
        history = []
        active_ids = active[b].nonzero(as_tuple=True)[0].tolist()
        for i in active_ids:
            transitions = {}
            choices = feasible[b, i].nonzero(as_tuple=True)[0].tolist()
            for cost, (score, rank) in dp.items():
                for j in choices:
                    new_cost = cost + costs_list[j] // unit
                    if new_cost > cap:
                        continue
                    entry = (score + float(values[b, i, j]), rank, j, cost)
                    previous = transitions.get(new_cost)
                    if previous is None or entry[0] > previous[0] or (entry[0] == previous[0] and entry[1:3] < previous[1:3]):
                        transitions[new_cost] = entry
            if not transitions:
                raise ValueError(f'infeasible allocation for batch {b}')
            ordered = sorted({entry[1:3] for entry in transitions.values()})
            ranks = {key: rank for rank, key in enumerate(ordered)}
            dp = {cost: (entry[0], ranks[entry[1:3]]) for cost, entry in transitions.items()}
            history.append({cost: (entry[3], entry[2]) for cost, entry in transitions.items()})
        cost, (score, _) = min(dp.items(), key=lambda item: (-item[1][0], item[0], item[1][1]))
        back_cost = cost
        for i, layer in zip(reversed(active_ids), reversed(history)):
            back_cost, j = layer[back_cost]
            indices[b, i] = j
            costs[b, i] = costs_list[j]
        totals[b] = cost * unit
        objectives[b] = score
    return Allocation(costs, indices, totals, objectives.detach())
