"""Public argument parsing and automatic two-rank launch; --help never loads a model."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from .config import load_config

def parser_for(task):
    parser = argparse.ArgumentParser(description={'train': 'Train final StreamZip with four budget views and two natural epochs.', 'inference': 'Score four answer choices with causal bounded StreamZip memory.', 'evaluate': 'Evaluate StreamZip at all four budgets on a labeled JSON manifest.', 'prepare': 'Convert the configured Qwen3-VL backbone into a native TP2 checkpoint.'}[task])
    parser.add_argument('--config', required=True, help='YAML config; relative paths resolve from the working directory')
    parser.add_argument('--validate-only', action='store_true', help='validate configuration/schema without loading a model or initializing CUDA')
    if task == 'inference':
        parser.add_argument('--video', required=True, help='local source video')
        parser.add_argument('--question', required=True)
        parser.add_argument('--choices', nargs=4, required=True, metavar=('A', 'B', 'C', 'D'), help='four exact option texts in A/B/C/D order')
        parser.add_argument('--timestamp', type=float, required=True, help='causal query endpoint in seconds')
        parser.add_argument('--budget', type=float, choices=[1.0, 0.5, 0.25, 0.125], required=True, help='nominal visual-token budget ratio')
    if task == 'evaluate':
        parser.add_argument('--manifest', required=True, help='evaluation manifest JSON array')
    return parser

def main(task):
    args = parser_for(task).parse_args()
    cfg = load_config(args.config, training=task in ('train', 'prepare'))
    request = {}
    if task == 'inference':
        from .data import parse_record
        request = parse_record(dict(id='inference', video=args.video, question=args.question, choices=args.choices, timestamp=args.timestamp), Path.cwd(), require_answer=False)
        request['budget'] = args.budget
    if task in ('train', 'evaluate'):
        from .data import load_manifest
        manifest = cfg['training']['manifest'] if task == 'train' else args.manifest
        records = load_manifest(manifest)
        request = {'manifest': str(Path(manifest).resolve())}
    if args.validate_only:
        print(json.dumps({'configuration_valid': True, 'task': task, 'record_count': len(records) if task in ('train', 'evaluate') else 1, 'model_loaded': False, 'cuda_initialized': False}))
        return
    worker = Path(__file__).resolve().parents[1] / 'scripts/_worker.py'
    command = [sys.executable, '-m', 'torch.distributed.run', '--nproc_per_node=2', '--master_port', str(cfg['runtime']['master_port']), str(worker), task, '--config', str(Path(args.config).resolve()), '--request-json', json.dumps(request)]
    env = dict(os.environ)
    env['CUDA_DEVICE_MAX_CONNECTIONS'] = '1'
    subprocess.run(command, env=env, check=True)
