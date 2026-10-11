"""Internal TP2 worker; use the documented public entrypoints."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('task', choices=['train', 'inference', 'evaluate', 'prepare'])
    p.add_argument('--config', required=True)
    p.add_argument('--request-json', required=True)
    args = p.parse_args()
    from streamzip.config import load_config
    from streamzip.runtime import initialize, load_backbone, load_student, predict, evaluate
    from streamzip.data import load_manifest
    from streamzip.utils import write_json
    import torch.distributed as dist
    cfg = load_config(args.config, training=args.task in ('train', 'prepare'))
    request = json.loads(args.request_json)
    initialize()
    try:
        if args.task == 'prepare':
            if Path(cfg['model']['native_checkpoint']).exists():
                raise FileExistsError('native checkpoint output already exists')
            load_backbone(cfg, prepare=True)
        elif args.task == 'train':
            from streamzip.training import train
            train(cfg, load_manifest(request['manifest']))
        elif args.task == 'evaluate':
            evaluate(cfg, load_manifest(request['manifest']))
        else:
            runner, processor = load_student(cfg)
            rows = predict(runner, processor, request, (request['budget'],))
            if dist.get_rank() == 0:
                write_json(Path(cfg['evaluation']['output_dir']) / 'prediction.json', rows[0])
                print(json.dumps(rows[0], indent=2), flush=True)
    finally:
        dist.destroy_process_group()
if __name__ == '__main__':
    main()
