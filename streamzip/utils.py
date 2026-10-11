"""Serialization, parameter checksums and native local QKV packing."""
from dataclasses import asdict, fields
import hashlib
import json
from pathlib import Path
import torch
from .visual import VisualTokenBundle, VideoChunkMetadata
from .compression import RankedBundle

def packed_bundle(bundle):
    return {f.name: asdict(bundle.metadata) if f.name == 'metadata' else tuple((x.detach().cpu() for x in getattr(bundle, f.name))) if f.name == 'deepstack_features' else getattr(bundle, f.name).detach().cpu() for f in fields(bundle)}

def unpack_bundle(data, device):
    values = {k: VideoChunkMetadata(**v) if k == 'metadata' else tuple((x.to(device) for x in v)) if k == 'deepstack_features' else v.to(device) for k, v in data.items()}
    return RankedBundle(**values) if 'selection_priority' in values else VisualTokenBundle(**values)

def tensor_sha(value):
    v = value.detach().cpu().contiguous()
    h = hashlib.sha256(str(v.dtype).encode() + str(tuple(v.shape)).encode())
    h.update(v.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()

def checksum(named):
    h = hashlib.sha256()
    for name, p in named:
        h.update(name.encode())
        h.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()

def interleave_local_qkv(query, key, value, head_size):
    """Native grouped packed order on one TP shard; pure tensor contract."""
    if query.shape[:-1] != key.shape[:-1] or key.shape != value.shape:
        raise ValueError('aligned Q/K/V leading dimensions required')
    heads = query.shape[-1] // head_size
    groups = key.shape[-1] // head_size
    if not groups or query.shape[-1] % head_size or key.shape[-1] % head_size or heads % groups:
        raise ValueError('local grouped attention dimensions disagree')
    lead = query.shape[:-1]
    q = query.reshape(*lead, groups, heads // groups, head_size)
    k = key.reshape(*lead, groups, 1, head_size)
    v = value.reshape(*lead, groups, 1, head_size)
    return torch.cat((q, k, v), dim=-2).reshape(*lead, -1)

def write_json(path, value):
    """Write a new output atomically, preserving readable JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temp.replace(path)
