"""Native TP2 StreamZip LoRA, budget conditioning and checkpoint logic."""
import gc
import re
import time
from dataclasses import replace
from itertools import chain
import torch
import torch.distributed as dist
from .utils import checksum, interleave_local_qkv, packed_bundle, unpack_bundle, tensor_sha
from .sparse import DenseVideoPrompt, sparse_prefill, language_kwargs, combine
from .budget import SEED, LAYERS, TARGETS, VIEWS, approved_parameter
from .inputs import BudgetEmbedding, insert_budget
from .objective import view_loss

def last_token_output(*, hidden_states, output_layer, output_weight, scale_logits, **kwargs):
    """Public Core output_processor, original frozen vocabulary projection."""
    logits, _ = output_layer(hidden_states[-1:], weight=output_weight, runtime_gather_output=True)
    return scale_logits(logits).transpose(0, 1).contiguous()

def same_tree(a, b):
    if isinstance(a, torch.Tensor):
        return torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all((same_tree(a[k], b[k]) for k in a))
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all((same_tree(x, y) for x, y in zip(a, b)))
    return a == b

def compact_top8_shard(shard, layer_index):
    if hasattr(shard, 'build_fn'):
        original_build = shard.build_fn

        def build(*args, **kwargs):

            def visit(value):
                if isinstance(value, dict):
                    return {k: visit(v) for k, v in value.items()}
                if isinstance(value, (list, tuple)):
                    return type(value)((visit(v) for v in value))
                if hasattr(value, 'global_shape') or hasattr(value, 'build_fn'):
                    return compact_top8_shard(value, layer_index)
                return value
            return visit(original_build(*args, **kwargs))
        return replace(shard, build_fn=build)
    if layer_index not in LAYERS or shard.global_shape[0] != 36 or shard.global_offset[0] != layer_index:
        raise ValueError('expected original native36-layer offset28..35')
    kwargs = dict(global_shape=(8, *shard.global_shape[1:]), global_offset=(layer_index - 28, *shard.global_offset[1:]))
    if hasattr(shard, 'axis_fragmentations'):
        kwargs['axis_fragmentations'] = (8, *shard.axis_fragmentations[1:])
    return replace(shard, **kwargs)

def mixed_optimizer_state(optimizer, model_state, parameter_count, loading=False):
    """Native mixed BF16/FP32 mapping, including budget Adam state."""
    from megatron.core.dist_checkpointing.optimizer import get_param_id_to_sharded_param_map, make_sharded_optimizer_tensor, optim_state_to_sharding_state
    assert len(optimizer.chained_optimizers) == 1
    wrapped = optimizer.chained_optimizers[0]
    if loading:
        wrapped.init_state_fn(wrapped.optimizer, wrapped.config)
    state = wrapped.state_dict()
    originals = []
    for half, full in zip(wrapped.float16_groups, wrapped.fp32_from_fp32_groups):
        assert not half or not full
        originals.append([*half, *full])
    mapping = get_param_id_to_sharded_param_map(model_state, chain.from_iterable(originals))
    ids = {i for g in state['optimizer']['param_groups'] for i in g['params']}
    assert set(mapping) == ids and len(ids) == parameter_count
    state['fp32_from_fp16_params'] = {f'group{g}_param{j}': make_sharded_optimizer_tensor(mapping[i], p, prefix='optimizer.state.fp32_param') for g, (group, masters) in enumerate(zip(state['optimizer']['param_groups'], state['fp32_from_fp16_params'])) for j, (i, p) in enumerate(zip(group['params'], masters))}
    step = wrapped._extract_common_per_param_step(state['optimizer'])
    optim_state_to_sharding_state(state['optimizer'], mapping, exclude_keys='step')
    if step:
        state['optimizer']['state']['common_step'] = step
    return state

class StreamZipModel:

    def __init__(self, models, inner, tokenizer, video_token_id):
        from megatron.bridge.peft.canonical_lora import CanonicalLoRA, LoRALinearSplitQKV
        from megatron.bridge.peft.lora import LoRA
        from megatron.core.utils import unwrap_model
        from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
        attention = CanonicalLoRA(target_modules=[p for p in TARGETS if '.self_attention.' in p], dim=8, alpha=16, dropout=0.0)

        class LocalQKV(LoRALinearSplitQKV):

            def _interleave_qkv(self, query, key, value):
                return interleave_local_qkv(query, key, value, self.to_wrap.config.kv_channels)

        class CompositeLoRA(LoRA):

            def transform(self, module, name=None, prefix=None):
                adapted = attention.transform(module, name, prefix)
                if type(adapted) is LoRALinearSplitQKV:
                    return LocalQKV(adapted.to_wrap, adapted.adapter)
                return super().transform(module, name, prefix) if adapted is module else adapted
        physical_targets = [p for p in TARGETS if '.mlp.' in p or p.endswith('linear_proj')]
        physical_targets += [f'*.language_model.decoder.layers.{i}.self_attention.linear_qkv' for i in LAYERS]
        self.base = list(models[0].named_parameters())
        self.base_versions = [(p, p._version) for _, p in self.base]
        self.base_before = checksum(self.base)
        from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
        model_parallel_cuda_manual_seed(SEED)
        torch.manual_seed(SEED)
        self.peft = CompositeLoRA(target_modules=physical_targets, dim=8, alpha=16, dropout=0.0)
        self.models = self.peft(models, training=True)
        self.inner = unwrap_model(self.models)[0]
        assert len(self.inner.language_model.decoder.layers) == 36
        self.budget = BudgetEmbedding(self.inner.language_model.config.hidden_size, seed=SEED, device='cuda')
        self.inner.language_model.add_module('streamzip_budget', self.budget)
        self.params = {n: p for n, p in self.models[0].named_parameters() if p.requires_grad}
        assert len(self.params) == 97 and all((approved_parameter(n) for n in self.params)), list(self.params)
        assert all((not p.requires_grad for _, p in self.base))
        for config in (self.inner.language_model.config, self.inner.language_model.decoder.config):
            config.recompute_granularity = 'full'
            config.recompute_method = 'uniform'
            config.recompute_num_layers = 1
            config.distribute_saved_activations = False
        self.initial = {n: p.detach().cpu().clone() for n, p in self.params.items()}
        self.model = DistributedDataParallel(self.inner.config, DistributedDataParallelConfig(grad_reduce_in_fp32=True, overlap_grad_reduce=False, use_distributed_optimizer=False), self.inner.language_model)
        self.tokenizer = tokenizer
        self.video_token_id = video_token_id
        self.option_ids = [tokenizer.encode(c, add_special_tokens=False)[0] for c in 'ABCD']
        assert all((len(tokenizer.encode(c, add_special_tokens=False)) == 1 for c in 'ABCD'))
        space = tokenizer.encode(' ', add_special_tokens=False)
        assert len(space) == 1
        self.dummy_id = space[0]
        self.optimizer = self.new_optimizer()
        self.forward_count = 0
        self.bank_id = None
        self.audit = []

    def new_optimizer(self):
        from megatron.core.optimizer import get_megatron_optimizer
        from megatron.core.optimizer.optimizer_config import OptimizerConfig, ParamKey
        cfg = OptimizerConfig(optimizer='adam', lr=0.0001, weight_decay=0.0001, bf16=True, params_dtype=torch.bfloat16, use_distributed_optimizer=False, clip_grad=1.0)
        opt = get_megatron_optimizer(cfg, [self.model], config_overrides={ParamKey(name='*streamzip_budget*'): dict(max_lr=0.0003, wd_mult=0.0)}, use_gloo_process_groups=False, pg_collection=self.inner.pg_collection)
        for group in opt.param_groups:
            group['lr'] = group['max_lr']
            group['weight_decay'] = cfg.weight_decay * group.get('wd_mult', 1.0)
        assert {(g['lr'], g['weight_decay']) for g in opt.param_groups} == {(0.0001, 0.0001), (0.0003, 0.0)}
        return opt

    def reset(self):
        self.optimizer = None
        gc.collect()
        with torch.no_grad():
            for n, p in self.params.items():
                p.copy_(self.initial[n])
        self.optimizer = self.new_optimizer()
        self.model.zero_grad_buffer()
        self.model.eval()
        assert all((torch.equal(p.detach().cpu(), self.initial[n]) for n, p in self.params.items()))

    def attach_context(self, record, context):
        """Attach only the requested bounded views, never an implicit history archive."""
        self.dense_cpu = context['dense']
        self.bank = {'question_start': context['question_start']}
        self.views = context['views']
        self.view_audit = context['audit']
        self.bank_id = record['id']

    def clear_context(self):
        """Drop per-question features after use."""
        self.bank_id = None
        for name in ('dense_cpu', 'bank', 'views', 'view_audit'):
            if hasattr(self, name):
                delattr(self, name)

    def prepare(self, record, view, student):
        if self.bank_id != record['id']:
            raise ValueError('Attach the causal prepared context before scoring this QA.')
        dense = DenseVideoPrompt(*(getattr(self.dense_cpu, k).cuda() for k in ('input_ids', 'attention_mask', 'mm_token_type_ids', 'position_ids')), self.video_token_id)
        selected = tuple((unpack_bundle(packed_bundle(c), 'cuda') for c in self.views[view]))
        with torch.no_grad():
            sparse = sparse_prefill(self.inner.language_model, dense, combine(selected, dense))
        if student:
            sparse, at = insert_budget(sparse, self.budget, VIEWS.index(view), self.bank['question_start'], self.dummy_id)
        else:
            at = None
        n = sum((len(c.final_features) for c in selected))
        assert int(sparse.visual_pos_masks.sum()) == n and int((sparse.input_ids == self.video_token_id).sum()) == n
        assert all((len(x) == n for x in sparse.bundle.deepstack_features))
        assert torch.equal(sparse.position_ids[:, 0, sparse.visual_pos_masks[0]].T, sparse.bundle.mrope_coordinates)
        diag = dict(qa_id=record['id'], video=record['video'], view=view, student=student, visual_tokens=n, sequence_length=sparse.input_ids.shape[1], budget_position=at, gt_in_input=False, retained_indices=sparse.bundle.original_token_index.tolist(), input_ids_sha256=tensor_sha(sparse.input_ids), position_ids_sha256=tensor_sha(sparse.position_ids), violations=0)
        return (sparse, diag)

    def logits(self, record, view, student=True, audit=False):
        if student:
            self.peft.enable_adapter_layers(self.models)
        else:
            self.peft.disable_adapter_layers(self.models)
        sparse, diag = self.prepare(record, view, student)
        logits = self.model(**language_kwargs(sparse, output_processor=last_token_output, runtime_gather_output=True))[0, 0].float()
        self.forward_count += 1
        assert logits.ndim == 1 and torch.isfinite(logits).all()
        if audit:
            self.audit.append(diag)
        return (logits, diag)

    def step(self, record, teacher, audit=False):
        from megatron.core.distributed.finalize_model_grads import finalize_model_grads
        self.model.train()
        self.optimizer.zero_grad()
        self.model.zero_grad_buffer()
        torch.cuda.synchronize()
        start = time.perf_counter()
        rows = []
        for view in VIEWS:
            t = time.perf_counter()
            logits, diag = self.logits(record, view, True, audit)
            loss, parts = view_loss(logits[self.option_ids], record['gt'], teacher)
            self.optimizer.scale_loss(loss / 4).backward()
            torch.cuda.synchronize()
            rows.append(dict(view=view, loss=float(loss.detach()), **{k: float(v.detach()) for k, v in parts.items()}, seconds=time.perf_counter() - t, visual_tokens=diag['visual_tokens'], sequence_length=diag['sequence_length']))
            del logits, loss, parts
        finalize_model_grads([self.model])
        torch.cuda.synchronize()
        assert all((torch.isfinite(p.main_grad).all() for p in self.params.values()))
        assert all((not p.requires_grad and p.grad is None and (getattr(p, 'main_grad', None) is None) for _, p in self.base))
        if audit:
            local = dict(budget=tensor_sha(self.budget.weight), budget_gradient=tensor_sha(self.budget.weight.main_grad), ownership={n: dict(shape=list(p.shape), dtype=str(p.dtype), tp=bool(getattr(p, 'tensor_model_parallel', False)), partition_dim=getattr(p, 'partition_dim', None), gradient_norm=float(p.main_grad.double().norm())) for n, p in self.params.items()})
            peers = [None, None]
            dist.all_gather_object(peers, local)
            assert peers[0]['budget'] == peers[1]['budget'] and peers[0]['budget_gradient'] == peers[1]['budget_gradient']
            assert all((r['ownership'][n]['tp'] for r in peers for n in self.params if '.adapter.' in n))
            assert self.budget.weight.main_grad.reshape(4, -1).norm(dim=1).gt(0).all()
            self.gradient_audit = peers
        success, norm, _ = self.optimizer.step()
        torch.cuda.synchronize()
        assert success and all((torch.isfinite(p).all() for p in self.params.values()))
        assert all((p._version == v for p, v in self.base_versions))
        if audit:
            peers = [None, None]
            dist.all_gather_object(peers, tensor_sha(self.budget.weight))
            assert peers[0] == peers[1]
        return dict(loss=sum((r['loss'] for r in rows)) / 4, views=rows, seconds=time.perf_counter() - start, global_grad_norm=float(norm), finite_gradients=True, base_gradients_absent=True)

    def checkpoint_state(self, loading=False):
        self.peft.set_params_to_save(self.models)
        full = self.models[0].sharded_state_dict()
        for name, value in list(full.items()):
            if '.adapter.' in name:
                layer = int(re.search('layers\\.(\\d+)\\.', name)[1])
                full[name] = compact_top8_shard(value, layer)
        selected = {k: v for k, v in full.items() if self.peft.adapter_key_filter(k) or 'streamzip_budget' in k}
        assert sum((not k.endswith('_extra_state') for k in selected)) == len(self.params)
        return dict(model=selected, optimizer=mixed_optimizer_state(self.optimizer, full, len(self.params), loading))

    def save(self, path):
        from megatron.core import dist_checkpointing
        if dist.get_rank() == 0:
            assert not path.exists()
            path.mkdir()
        dist.barrier()
        dist_checkpointing.save(self.checkpoint_state(), str(path))
        dist.barrier()

    def load(self, path):
        from megatron.core import dist_checkpointing
        loaded = dist_checkpointing.load(self.checkpoint_state(True), str(path))
        self.models[0].load_state_dict(loaded['model'], strict=False)
        state = loaded['optimizer']
        masters = state['fp32_from_fp16_params']
        wrapped = self.optimizer.chained_optimizers[0]
        state['fp32_from_fp16_params'] = [[masters[f'group{g}_param{i}'] for i in range(len(ps))] for g, ps in enumerate(wrapped.fp32_from_float16_groups)]
        self.optimizer.load_state_dict(state)
        dist.barrier()

    def base_audit(self):
        assert all((p._version == v and (not p.requires_grad) for p, v in self.base_versions))
        after = checksum(self.base)
        assert after == self.base_before
        return dict(before=self.base_before, after=after, frozen_base_unchanged=True)
