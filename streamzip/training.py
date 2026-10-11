"""Two natural epochs with four-view accumulation and a frozen full-context teacher."""
import hashlib
from pathlib import Path
import torch
import torch.distributed as dist
from .budget import SEED, RATIOS, VIEWS
from .data import build_context
from .objective import teacher_target
from .runtime import load_student
from .utils import write_json, tensor_sha

def ordered_epoch(records, epoch):
    """Preserve the final-fit deterministic QA-group order."""
    return sorted(records, key=lambda r: hashlib.sha256(f'{SEED}:3:{epoch}:{r['id']}'.encode()).hexdigest())

def train(config, records):
    """One optimizer update per QA, averaging its four budget-view losses."""
    directory = Path(config['training']['output_dir'])
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / 'adapter').exists():
        raise FileExistsError('output adapter exists; choose a new output_dir')
    runner, processor = load_student(config, training=True)
    teachers = {}
    trace = []
    for epoch in range(2):
        for record in ordered_epoch(records, epoch):
            context = build_context(runner.inner, processor, record, RATIOS)
            runner.attach_context(record, context)
            try:
                if record['id'] not in teachers:
                    runner.model.eval()
                    with torch.no_grad():
                        logits, _ = runner.logits(record, VIEWS[0], student=False)
                        teachers[record['id']] = teacher_target(logits[runner.option_ids], record['gt'])
                        teachers[record['id']]['probability'] = teachers[record['id']]['probability'].cpu()
                step = runner.step(record, teachers[record['id']], audit=len(trace) == 0)
                trace.append(dict(epoch=epoch, qa_id=record['id'], **step))
            finally:
                runner.clear_context()
            if dist.get_rank() == 0:
                print(f'epoch={epoch + 1}/2 update={len(trace)} loss={step['loss']:.6f}', flush=True)
    runner.save(directory / 'adapter')
    audit = runner.base_audit()
    hashes = [None, None]
    dist.all_gather_object(hashes, tensor_sha(runner.budget.weight))
    if hashes[0] != hashes[1]:
        raise RuntimeError('budget parameters differ across TP ranks')
    if dist.get_rank() == 0:
        write_json(directory / 'training_trace.json', trace)
        write_json(directory / 'checkpoint_info.json', dict(epochs=2, qa_group_updates=len(trace), view_forwards=4 * len(trace), trained_budget_sha256=hashes[0], base=audit, budgets=list(RATIOS), final_epoch_only=True))
