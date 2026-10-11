import torch
from torch.nn import functional as F
from streamzip.objective import teacher_target, view_loss
from streamzip.scoring import score_answer

def test_fp32_loss_arithmetic_and_selective_teacher_gradient():
    z = torch.tensor([1.0, 2.0, 0.0, -0.5], dtype=torch.bfloat16, requires_grad=True)
    teacher = teacher_target(torch.tensor([0.0, 3.0, 0.0, 0.0]), 1)
    value, parts = view_loss(z, 1, teacher)
    f = z.float()
    expected = F.cross_entropy(f[None], torch.tensor([1])) + 0.5 * F.relu(0.2 - f[1] + f[[0, 2, 3]].max()) + 0.3 * teacher['weight'] * F.kl_div(f.log_softmax(0), teacher['probability'], reduction='sum')
    torch.testing.assert_close(value, expected)
    assert value.dtype == torch.float32
    value.backward()
    assert torch.isfinite(z.grad).all()
    incorrect = teacher_target(torch.tensor([9.0, 0.0, 0.0, 0.0]), 1)
    assert incorrect['weight'] == 0
    assert view_loss(z.detach(), 1, incorrect)[1]['distill'] == 0
    assert teacher_target(torch.zeros(4), 0)['weight'] == 0

def test_full_vocabulary_nll_differs_from_option_ce():
    logits = torch.tensor([10.0, 1.0, 2.0, 3.0, 4.0, 9.0])
    result = score_answer(logits, [1, 2, 3, 4], 3)
    assert result['correct'] == 1 and result['prediction'] == 'D'
    assert result['gt_nll'] > float(F.cross_entropy(logits[[1, 2, 3, 4]][None], torch.tensor([3])))
