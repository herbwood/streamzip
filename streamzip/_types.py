"""Tensor contracts for deterministic retention."""
from dataclasses import dataclass
import torch
from torch import Tensor

def check_tokens(tokens: Tensor, valid: Tensor, name: str='tokens') -> None:
    if tokens.ndim != 3 or not tokens.is_floating_point():
        raise ValueError(f'{name} must be floating [S,N,D]')
    if tokens.shape[-1] == 0 or valid.shape != tokens.shape[:2] or valid.dtype != torch.bool:
        raise ValueError(f'{name} requires nonzero D and boolean [S,N] valid mask')
    if valid.device != tokens.device:
        raise ValueError('features and masks must share a device')
    if not torch.isfinite(tokens[valid]).all():
        raise ValueError(f'{name} contains nonfinite valid features')

def check_candidates(candidates: Tensor) -> None:
    if candidates.ndim != 1 or candidates.dtype != torch.int64 or candidates.numel() == 0:
        raise ValueError('candidate costs must be nonempty int64 [J]')
    if (candidates < 0).any() or (candidates[1:] <= candidates[:-1]).any():
        raise ValueError('candidate costs must be sorted, unique and nonnegative')

@dataclass(frozen=True)
class Allocation:
    costs: Tensor
    choice_index: Tensor
    total_cost: Tensor
    objective: Tensor

@dataclass(frozen=True)
class CompressedTokens:
    tokens: Tensor
    valid: Tensor
    source_index: Tensor
