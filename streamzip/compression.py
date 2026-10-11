"""Deterministic temporal-balanced cosine-feature coreset."""
from dataclasses import dataclass, fields
import torch
from torch.nn.functional import normalize
from .visual import VisualTokenBundle, StreamZipTokenSelector
from ._types import CompressedTokens, check_tokens

def coreset_order(features, temporal):
    """FP32 cosine distances, stable source-index tie breaks, round-robin time."""
    if features.ndim != 2 or temporal.shape != features.shape[:1]:
        raise ValueError('features [N,D] and temporal [N] required')
    if not torch.isfinite(features).all():
        raise ValueError('nonfinite features')
    orders = []
    for t in temporal.unique(sorted=True):
        indices = (temporal == t).nonzero(as_tuple=True)[0]
        z = normalize(features[indices].float(), dim=-1)
        centroid = normalize(z.mean(0), dim=0)
        first = int((z @ centroid).argmax())
        selected = [first]
        distance = (1 - z @ z[first]).clamp_min(0)
        for _ in range(len(indices) - 1):
            distance[selected] = -1
            nxt = int(distance.argmax())
            selected.append(nxt)
            distance = torch.minimum(distance, (1 - z @ z[nxt]).clamp_min(0))
        orders.append(indices[torch.tensor(selected, device=indices.device)])
    order = [row[j] for j in range(max(map(len, orders))) for row in orders if j < len(row)]
    return torch.stack(order)

@dataclass(frozen=True)
class RankedBundle(VisualTokenBundle):
    selection_priority: torch.Tensor

    def validate(self):
        super().validate()
        ranks = self.selection_priority
        if ranks.shape != self.temporal_coordinates.shape or ranks.dtype != torch.int64 or ranks.device != self.final_features.device:
            raise ValueError('invalid coreset provenance')
        if (ranks < 0).any() or ranks.unique().numel() != ranks.numel():
            raise ValueError('priority must be unique and nonnegative')

    def select(self, indices):
        base = super().select(indices)
        return RankedBundle(**{f.name: getattr(base, f.name) for f in fields(VisualTokenBundle)}, selection_priority=self.selection_priority.index_select(0, indices).detach())

def with_priority(bundle):
    if isinstance(bundle, RankedBundle):
        return bundle
    order = coreset_order(bundle.final_features, bundle.temporal_coordinates)
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(len(order), device=order.device)
    out = RankedBundle(**{f.name: getattr(bundle, f.name) for f in fields(VisualTokenBundle)}, selection_priority=inverse)
    out.validate()
    return out

class PriorityCompressor:
    """Compact token selection bound only to retained integer priorities."""

    def __init__(self, ranks):
        self.ranks = ranks

    def compress(self, tokens, valid, requested_k):
        check_tokens(tokens, valid)
        if self.ranks.shape != valid.shape or self.ranks.device != tokens.device:
            raise ValueError('rank shape/device mismatch')
        if requested_k.shape != (tokens.shape[0],) or requested_k.dtype != torch.int64 or requested_k.device != tokens.device:
            raise ValueError('requested_k must be int64 [B]')
        if (requested_k < 0).any() or (requested_k > valid.sum(-1)).any():
            raise ValueError('cannot recover discarded tokens')
        capacity = int(requested_k.max())
        out = tokens.new_zeros(tokens.shape[0], capacity, tokens.shape[-1])
        mask = torch.zeros(tokens.shape[0], capacity, dtype=torch.bool, device=tokens.device)
        sources = torch.full(mask.shape, -1, dtype=torch.int64, device=tokens.device)
        for b, k in enumerate(requested_k.tolist()):
            available = valid[b].nonzero(as_tuple=True)[0]
            indices = available[self.ranks[b, available].argsort(stable=True)[:k]].sort().values
            out[b, :k] = tokens[b, indices]
            mask[b, :k] = True
            sources[b, :k] = indices
        return CompressedTokens(out, mask, sources)

class FeatureCoresetSelector:

    def select(self, bundle, k):
        bundle = with_priority(bundle)
        return StreamZipTokenSelector(PriorityCompressor(bundle.selection_priority[None])).select(bundle, k)
