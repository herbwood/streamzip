"""FP32 supervised MC, answer margin, and correct-teacher-only distillation."""
import math
import torch
from torch.nn import functional as F

def teacher_target(option_logits, gt):
    z = option_logits.detach().float()
    if z.shape != (4,) or not torch.isfinite(z).all() or (not 0 <= gt < 4):
        raise ValueError('four finite option logits and valid GT required')
    p = z.softmax(0)
    logp = z.log_softmax(0)
    entropy = -(p * logp).sum()
    correct = int(z.argmax()) == gt
    confidence = (1 - entropy / math.log(4)).clamp(0, 1)
    return dict(probability=p, predicted=int(z.argmax()), correct=correct, entropy=float(entropy), confidence=float(confidence), weight=float(confidence) if correct else 0.0)

def view_loss(option_logits, gt, teacher):
    z = option_logits.float()
    if z.shape != (4,) or not torch.isfinite(z).all() or (not 0 <= gt < 4):
        raise ValueError('four finite option logits and valid GT required')
    p = teacher['probability'].detach().to(z.device, dtype=torch.float32)
    if p.shape != (4,) or (p < 0).any() or (not torch.isfinite(p).all()) or (not torch.isclose(p.sum(), p.new_tensor(1.0), atol=1e-06)):
        raise ValueError('normalized four-option teacher distribution required')
    weight = float(teacher['weight'])
    if not 0 <= weight <= 1 or (not teacher['correct'] and weight != 0):
        raise ValueError('incorrect teachers must have zero weight')
    mc = F.cross_entropy(z[None], torch.tensor([gt], device=z.device))
    wrong = torch.arange(4, device=z.device) != gt
    margin = F.relu(z.new_tensor(0.2) - z[gt] + z[wrong].max())
    kl = F.kl_div(z.log_softmax(0), p, reduction='sum')
    distill = weight * kl
    return (mc + 0.5 * margin + 0.3 * distill, dict(mc=mc, margin=margin, distill=distill, kl=kl))
