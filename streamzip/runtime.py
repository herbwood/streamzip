"""Native TP2 execution shared by the public entrypoints."""
from datetime import timedelta
from pathlib import Path
import os
import time
import torch
import torch.distributed as dist
from .budget import BASE_MODEL_SEED, VIEWS, RATIOS, budget_id
from .data import build_context
from .model import StreamZipModel
from .scoring import score_answer
from .utils import write_json

def initialize():
    """Match the qualified TP2/DP1, BF16, uncached execution contract."""
    if int(os.environ.get('WORLD_SIZE', 0)) != 2:
        raise ValueError('native execution requires exactly two torchrun ranks')
    local = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    dist.init_process_group('nccl', timeout=timedelta(seconds=1800), device_id=torch.device('cuda', local))

def shared_latency(seconds):
    """Use the slowest tensor-parallel rank for synchronized latency reporting."""
    values = torch.tensor(seconds, dtype=torch.float64, device='cuda')
    dist.all_reduce(values, op=dist.ReduceOp.MAX)
    return float(values)

def load_backbone(config, *, prepare=False):
    """Use the same Bridge provider and native checkpoint loader as qualification."""
    from megatron.bridge import AutoBridge
    from megatron.core.utils import unwrap_model
    from transformers import AutoProcessor
    cfg = config['model']
    bridge = AutoBridge.from_hf_pretrained(cfg['name_or_path'], local_files_only=cfg['local_files_only'])
    provider = bridge.to_megatron_provider(load_weights=prepare)
    provider.tensor_model_parallel_size = 2
    provider.pipeline_model_parallel_size = 1
    provider.context_parallel_size = 1
    provider.sequence_parallel = False
    provider.params_dtype = torch.bfloat16
    provider.pipeline_dtype = torch.bfloat16
    provider.bf16 = True
    provider.finalize()
    provider.initialize_model_parallel(seed=BASE_MODEL_SEED)
    if prepare:
        models = provider.provide_distributed_model(wrap_with_ddp=False)
        bridge.save_megatron_model(models, cfg['native_checkpoint'], hf_tokenizer_path=cfg['name_or_path'], low_memory_save=True)
    else:
        models = bridge.load_megatron_model(cfg['native_checkpoint'], mp_overrides={'tensor_model_parallel_size': 2, 'pipeline_model_parallel_size': 1, 'context_parallel_size': 1, 'sequence_parallel': False, 'pipeline_dtype': torch.bfloat16}, wrap_with_ddp=False)
    inner = unwrap_model(models)[0]
    inner.eval().requires_grad_(False)
    processor = AutoProcessor.from_pretrained(cfg['name_or_path'], local_files_only=cfg['local_files_only'])
    return (models, inner, processor)

def load_student(config, *, training=False):
    models, inner, processor = load_backbone(config)
    runner = StreamZipModel(models, inner, processor.tokenizer, inner.video_token_id)
    if training:
        runner.reset()
    else:
        runner.load(Path(config['model']['adapter_checkpoint']))
        runner.model.eval()
    return (runner, processor)

@torch.no_grad()
def predict(runner, processor, record, ratios):
    """Score answer letters only; labels are accessed after the model forward."""
    runner.model.eval()
    torch.cuda.synchronize()
    start = time.perf_counter()
    context = build_context(runner.inner, processor, record, ratios)
    runner.attach_context(record, context)
    torch.cuda.synchronize()
    preparation = shared_latency(time.perf_counter() - start)
    rows = []
    try:
        for ratio in ratios:
            view = VIEWS[budget_id(ratio)]
            torch.cuda.synchronize()
            tick = time.perf_counter()
            logits, diag = runner.logits(record, view, True)
            torch.cuda.synchronize()
            elapsed = shared_latency(time.perf_counter() - tick)
            option_logits = logits[runner.option_ids].float()
            predicted = int(option_logits.argmax())
            row = dict(**diag, budget=ratio, source_id=record['source_id'], prediction=chr(65 + predicted), predicted_option=predicted, answer_text=record['options'][predicted], option_probabilities=option_logits.softmax(0).tolist(), inference_latency_seconds=elapsed, preparation_latency_seconds=preparation, total_latency_seconds=preparation + elapsed)
            if 'gt' in record:
                row.update(score_answer(logits, runner.option_ids, record['gt']))
            rows.append(row)
    finally:
        runner.clear_context()
    return rows

def summarize(rows):
    """Report QA accuracy and equal-source accuracy separately at each budget."""
    if not rows:
        raise ValueError('no evaluation rows')
    tables = {}
    for ratio in RATIOS:
        selected = [r for r in rows if r['budget'] == ratio]
        if not selected:
            continue
        sources = sorted({r['source_id'] for r in selected})
        correct = sum((r['correct'] for r in selected))
        tables[str(ratio)] = dict(correct=correct, total=len(selected), accuracy=correct / len(selected), source_macro_accuracy=sum((sum((r['correct'] for r in selected if r['source_id'] == s)) / sum((r['source_id'] == s for r in selected)) for s in sources)) / len(sources), mean_gt_nll=sum((r['gt_nll'] for r in selected)) / len(selected), mean_retained_visual_tokens=sum((r['visual_tokens'] for r in selected)) / len(selected), mean_sequence_length=sum((r['sequence_length'] for r in selected)) / len(selected), mean_inference_latency_seconds=sum((r['inference_latency_seconds'] for r in selected)) / len(selected), mean_preparation_latency_seconds=sum((r['preparation_latency_seconds'] for r in selected)) / len(selected))
    return tables

def evaluate(config, records):
    runner, processor = load_student(config)
    rows = []
    for record in records:
        rows.extend(predict(runner, processor, record, RATIOS))
    peers = [None, None]
    dist.all_gather_object(peers, rows)
    if peers[0] != peers[1]:
        raise RuntimeError('TP rank predictions disagree')
    if dist.get_rank() == 0:
        directory = Path(config['evaluation']['output_dir'])
        write_json(directory / 'predictions.json', rows)
        summary = summarize(rows)
        write_json(directory / 'metrics.json', summary)
        print(summary, flush=True)
