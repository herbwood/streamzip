"""Aligned final/DeepStack features and original token provenance."""
from dataclasses import dataclass
import torch
from torch import Tensor

@dataclass(frozen=True)
class VideoChunkMetadata:
    video_id: str
    chunk_id: int
    fps: float
    frame_indices: tuple[int, ...]

@dataclass(frozen=True)
class VisualTokenBundle:
    final_features: Tensor
    deepstack_features: tuple[Tensor, ...]
    temporal_coordinates: Tensor
    spatial_coordinates: Tensor
    timestamps: Tensor
    mrope_coordinates: Tensor
    prompt_positions: Tensor
    original_token_index: Tensor
    metadata: VideoChunkMetadata

    def validate(self):
        final = self.final_features
        if final.ndim != 2 or not final.is_floating_point() or min(final.shape) < 1:
            raise ValueError('queried bundle requires positive [N,Dl] features')
        n, _ = final.shape
        if len(self.deepstack_features) != 3:
            raise ValueError('StreamZip requires three aligned DeepStack streams')
        for features in (final, *self.deepstack_features):
            if features.shape != final.shape or features.dtype != final.dtype or features.device != final.device:
                raise ValueError('feature streams must have identical shape/dtype/device')
            if not torch.isfinite(features).all():
                raise ValueError('nonfinite visual feature')
        integers = ((self.temporal_coordinates, (n,)), (self.spatial_coordinates, (n, 2)), (self.mrope_coordinates, (n, 3)), (self.prompt_positions, (n,)), (self.original_token_index, (n,)))
        for value, shape in integers:
            if value.shape != shape or value.dtype != torch.int64 or value.device != final.device or (value < 0).any():
                raise ValueError('invalid integer provenance shape/dtype/device/value')
        for values in (self.prompt_positions, self.original_token_index):
            if (values[1:] <= values[:-1]).any():
                raise ValueError('source order must be unique and strictly increasing')
        if self.timestamps.shape != (n,) or self.timestamps.dtype != torch.float32 or self.timestamps.device != final.device:
            raise ValueError('timestamps must be float32 [N] on the feature device')
        if not torch.isfinite(self.timestamps).all() or (self.timestamps < 0).any():
            raise ValueError('invalid timestamp')
        if self.metadata.fps <= 0 or not self.metadata.frame_indices:
            raise ValueError('source FPS and frame indices are required')

    def select(self, indices: Tensor) -> 'VisualTokenBundle':
        self.validate()
        if indices.ndim != 1 or indices.dtype != torch.int64 or indices.device != self.final_features.device or (not indices.numel()):
            raise ValueError('selection requires positive int64 [K] on the feature device')
        if (indices < 0).any() or (indices >= self.final_features.shape[0]).any() or (indices[1:] <= indices[:-1]).any():
            raise ValueError('selection must be in-range, unique and preserve source order')

        def take(value):
            return value.index_select(0, indices).detach()
        selected = VisualTokenBundle(take(self.final_features), tuple((take(x) for x in self.deepstack_features)), take(self.temporal_coordinates), take(self.spatial_coordinates), take(self.timestamps), take(self.mrope_coordinates), take(self.prompt_positions), take(self.original_token_index), self.metadata)
        selected.validate()
        return selected

class StreamZipTokenSelector:
    """Apply explicit retained indices without merging visual features."""

    def __init__(self, compressor):
        self.compressor = compressor

    def select(self, bundle: VisualTokenBundle, k: int) -> VisualTokenBundle:
        bundle.validate()
        if not 1 <= k <= bundle.final_features.shape[0]:
            raise ValueError('current queried bundle cannot be evicted or expanded')
        tokens = bundle.final_features.unsqueeze(0)
        valid = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
        output = self.compressor.compress(tokens, valid, torch.tensor([k], device=tokens.device))
        if output.valid.shape != (1, k) or output.source_index.shape != (1, k) or (not output.valid.all()):
            raise ValueError('compressor must realize exactly K selected indices')
        selected = bundle.select(output.source_index[0])
        if output.tokens.shape != (1, k, tokens.shape[-1]) or not torch.equal(output.tokens[0], selected.final_features):
            raise ValueError('StreamZip supports selection only, not merging/pooling')
        return selected
