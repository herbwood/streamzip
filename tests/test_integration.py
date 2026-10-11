"""Explicit opt-in only; existing local model/checkpoints/videos are required."""
import os
from pathlib import Path
import subprocess
import sys
import json
import pytest
import yaml

@pytest.mark.integration
@pytest.mark.skipif(os.environ.get('STREAMZIP_MODEL_TEST') != '1', reason='opt-in native 8B test disabled')
def test_local_native_evaluation(tmp_path):
    config_path = Path(os.environ['STREAMZIP_TEST_CONFIG'])
    manifest = Path(os.environ['STREAMZIP_TEST_MANIFEST'])
    config = yaml.safe_load(config_path.read_text())
    config['model']['local_files_only'] = True
    config['evaluation']['output_dir'] = str(tmp_path / 'result')
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump(config))
    root = Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable, str(root / 'scripts/evaluate.py'), '--config', str(path), '--manifest', str(manifest)], cwd=root, check=True)
    result = json.loads((tmp_path / 'result/metrics.json').read_text())
    assert set(result) == {'1.0', '0.5', '0.25', '0.125'}
