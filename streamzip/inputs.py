"""One soft text-position budget token; no tokenizer or visual-coordinate edits."""
from dataclasses import replace
import torch
from torch import nn

class BudgetEmbedding(nn.Module):

    def __init__(self, hidden_size, *, seed=20261006, device=None):
        super().__init__()
        if hidden_size < 1:
            raise ValueError('positive language width required')
        generator = torch.Generator(device='cpu').manual_seed(seed)
        self.weight = nn.Parameter(torch.randn(4, hidden_size, generator=generator).mul(0.02).to(device=device))
        self.weight.tensor_model_parallel = False

    def forward(self, budget_id):
        if isinstance(budget_id, bool) or budget_id not in range(4):
            raise ValueError('exactly four budget conditions')
        return self.weight[budget_id]

def find_question_start(ids, query_ids, last_visual_position):
    a = ids[0].tolist()
    q = query_ids.tolist()
    matches = [i for i in range(last_visual_position + 1, len(a) - len(q) + 1) if a[i:i + len(q)] == q]
    if len(matches) != 1:
        raise ValueError('unambiguous original question/options boundary required')
    return matches[0]

def insert_budget(sparse, budget, budget_id, question_original_position, dummy_id):
    match = (sparse.original_prompt_positions == question_original_position).nonzero().flatten()
    if len(match) != 1:
        raise ValueError('question boundary absent from sparse prompt')
    at = int(match[0])
    ids = sparse.input_ids
    if dummy_id < 0:
        raise ValueError('existing nonnegative vocabulary ID required')
    if (sparse.input_ids[sparse.visual_pos_masks] == dummy_id).any():
        raise ValueError('budget dummy must be a text token, not a visual placeholder')
    if sparse.visual_pos_masks[0, at:].any():
        raise ValueError('budget token must follow all visual positions')
    position = sparse.position_ids[:, :, at:at + 1]
    if not torch.equal(position, position[:1].expand_as(position)):
        raise ValueError('soft token requires three equal text M-RoPE axes')

    def insert(x, value, axis=-1):
        return torch.cat((x.narrow(axis, 0, at), value, x.narrow(axis, at, x.shape[axis] - at)), axis)
    embedding = budget(budget_id).to(dtype=sparse.inputs_embeds.dtype).view(1, 1, -1)
    if embedding.shape[-1] != sparse.inputs_embeds.shape[-1]:
        raise ValueError('budget width must match language embedding width')
    p = torch.cat((sparse.position_ids[:, :, :at], position, sparse.position_ids[:, :, at:] + 1), -1)
    out = replace(sparse, input_ids=insert(ids, ids.new_full((1, 1), dummy_id)), attention_mask=insert(sparse.attention_mask, sparse.attention_mask.new_ones((1, 1))), mm_token_type_ids=insert(sparse.mm_token_type_ids, ids.new_zeros((1, 1))), position_ids=p, inputs_embeds=insert(sparse.inputs_embeds, embedding, axis=1), visual_pos_masks=insert(sparse.visual_pos_masks, sparse.visual_pos_masks.new_zeros((1, 1))), original_prompt_positions=insert(sparse.original_prompt_positions, ids.new_full((1,), -1), axis=0), next_mrope_position=int(p.max()) + 1)
    assert torch.equal(out.position_ids[:, 0, out.visual_pos_masks[0]].T, sparse.bundle.mrope_coordinates)
    assert torch.equal(out.inputs_embeds[0, out.visual_pos_masks[0]], sparse.bundle.final_features)
    assert int(out.visual_pos_masks.sum()) == len(sparse.bundle.final_features)
    return (out, at)
