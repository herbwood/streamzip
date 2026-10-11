"""Readable configs with fixed scientific settings and user-owned paths."""
from pathlib import Path
import yaml
FIXED = {'streamzip': dict(chunk_seconds=8, budgets=[1.0, 0.5, 0.25, 0.125], minimum_current_tokens=8), 'lora': dict(rank=8, alpha=16, dropout=0, top_layers=8), 'loss': dict(margin_weight=0.5, margin=0.2, distill_weight=0.3, temperature=1.0), 'training': dict(epochs=2, lora_lr=0.0001, budget_lr=0.0003, lora_weight_decay=0.0001, budget_weight_decay=0, grad_clip=1.0), 'runtime': dict(tensor_parallel_size=2, data_parallel_size=1)}
ALLOWED = {'model': {'name_or_path', 'native_checkpoint', 'adapter_checkpoint', 'local_files_only'}, 'streamzip': set(FIXED['streamzip']), 'lora': set(FIXED['lora']), 'loss': set(FIXED['loss']), 'training': set(FIXED['training']) | {'manifest', 'output_dir'}, 'evaluation': {'output_dir'}, 'runtime': set(FIXED['runtime']) | {'master_port'}}

def load_config(path, *, training=False):
    """Reject unsupported scientific variations instead of silently ignoring them."""
    config = yaml.safe_load(Path(path).read_text())
    if not isinstance(config, dict) or set(config) - set(ALLOWED):
        raise ValueError('unknown configuration section')
    for section, values in config.items():
        if not isinstance(values, dict) or set(values) - ALLOWED[section]:
            raise ValueError(f'unknown fields in {section}')
    for section in ('streamzip', 'lora', 'loss', 'runtime') + (('training',) if training else ()):
        if section not in config:
            raise ValueError(f'missing {section} configuration')
        for name, expected in FIXED[section].items():
            actual = config[section].get(name)
            if isinstance(actual, bool) or actual != expected:
                raise ValueError(f'{section}.{name} must match the released setting {expected!r}')
    model = config.get('model', {})
    for field in ('name_or_path', 'native_checkpoint'):
        if not isinstance(model.get(field), str) or not model[field].strip():
            raise ValueError(f'model.{field} is required')
    if not isinstance(model.get('local_files_only'), bool):
        raise ValueError('model.local_files_only must be boolean')
    if not training and (not isinstance(model.get('adapter_checkpoint'), str)):
        raise ValueError('model.adapter_checkpoint is required for inference/evaluation')
    port = config['runtime'].get('master_port')
    if isinstance(port, bool) or not isinstance(port, int) or (not 1024 <= port <= 65535):
        raise ValueError('runtime.master_port must be a valid user port')
    section = 'training' if training else 'evaluation'
    if not isinstance(config.get(section, {}).get('output_dir'), str):
        raise ValueError(f'{section}.output_dir is required')
    if training and (not isinstance(config['training'].get('manifest'), str)):
        raise ValueError('training.manifest is required')
    return config
