import ast
import importlib
import subprocess
import sys
from pathlib import Path
import pytest
import yaml
from streamzip.config import load_config
from streamzip.model import compact_top8_shard
from streamzip.utils import interleave_local_qkv
ROOT = Path(__file__).resolve().parents[1]

def test_imports_and_public_only_dependency_graph():
    for p in (ROOT / 'streamzip').glob('*.py'):
        importlib.import_module('streamzip.' + p.stem)
        for node in ast.walk(ast.parse(p.read_text())):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert not any((part.startswith('phase') for part in node.module.split('.')))
    assert 'megatron.bridge' not in sys.modules

@pytest.mark.parametrize('script', ['train.py', 'inference.py', 'evaluate.py', 'prepare_model.py'])
def test_cli_help_without_model(script):
    proc = subprocess.run([sys.executable, str(ROOT / 'scripts' / script), '--help'], capture_output=True, text=True)
    assert proc.returncode == 0 and '--config' in proc.stdout

def test_configs_and_unsupported_scientific_settings(tmp_path):
    config = load_config(ROOT / 'configs/train.yaml', training=True)
    config['training']['epochs'] = 3
    path = tmp_path / 'bad.yaml'
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError):
        load_config(path, training=True)
    assert load_config(ROOT / 'configs/inference.yaml')['streamzip']['budgets'] == [1.0, 0.5, 0.25, 0.125]

def test_native_checkpoint_factory_and_local_qkv_layout():
    from dataclasses import dataclass
    import torch

    @dataclass
    class Shard:
        data: object
        global_shape: tuple = (36, 8, 4)
        global_offset: tuple = (28, 0, 0)
        axis_fragmentations: tuple = (36, 2, 1)
    s = Shard(torch.ones(4, 4))
    out = compact_top8_shard(s, 28)
    assert out.data is s.data and out.global_shape == (8, 8, 4) and (out.global_offset == (0, 0, 0))
    q = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
    k = torch.tensor([[4.0, 5.0]])
    v = torch.tensor([[6.0, 7.0]])
    assert interleave_local_qkv(q, k, v, 1).tolist() == [[0.0, 1.0, 4.0, 6.0, 2.0, 3.0, 5.0, 7.0]]
