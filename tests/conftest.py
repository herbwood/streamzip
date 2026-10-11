"""Small synthetic aligned-feature fixtures; no model download or CUDA."""
import pytest
import torch
from streamzip.visual import VisualTokenBundle, VideoChunkMetadata
from streamzip.compression import with_priority

@pytest.fixture
def arrivals():
    generator = torch.Generator().manual_seed(17)
    result = []
    for chunk in range(3):
        final = torch.randn(32, 8, generator=generator)
        temporal = torch.arange(2).repeat_interleave(16)
        spatial = torch.stack(torch.meshgrid(torch.arange(4), torch.arange(4), indexing='ij'), -1).reshape(-1, 2).repeat(2, 1)
        positions = torch.arange(chunk * 32 + 1, (chunk + 1) * 32 + 1)
        rope = torch.stack((temporal + chunk * 2, spatial[:, 0], spatial[:, 1]), -1)
        bundle = VisualTokenBundle(final, tuple((final + 10000 * (i + 1) for i in range(3))), temporal, spatial, temporal.float() * 4 + chunk * 8 + 2, rope, positions, torch.arange(chunk * 32, (chunk + 1) * 32), VideoChunkMetadata('synthetic-video', chunk, 1.0, (chunk * 8 + 1, chunk * 8 + 3, chunk * 8 + 5, chunk * 8 + 7)))
        result.append(with_priority(bundle))
    return tuple(result)
