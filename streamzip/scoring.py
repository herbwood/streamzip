"""Four-choice argmax and full-vocabulary ground-truth NLL."""
import torch

def score_answer(last_logits, option_ids, gt):
    if last_logits.ndim != 1 or not last_logits.is_floating_point() or (not torch.isfinite(last_logits).all()):
        raise ValueError('finite full-vocabulary next-token logits [V] required')
    ids = torch.as_tensor(option_ids, device=last_logits.device, dtype=torch.long)
    if ids.shape != (4,) or ids.unique().numel() != 4 or (ids < 0).any() or (ids >= len(last_logits)).any() or (not 0 <= gt < 4):
        raise ValueError('four unique valid single-letter option IDs and GT index required')
    z = last_logits.float()
    option_logits = z[ids]
    logp = z.log_softmax(-1)
    predicted = int(option_logits.argmax())
    return dict(predicted_option=predicted, prediction=chr(65 + predicted), correct=int(predicted == gt), gt_nll=float(-logp[ids[gt]]), option_logits=option_logits.tolist(), option_probabilities=option_logits.softmax(-1).tolist(), option_full_vocabulary_probabilities=logp[ids].exp().tolist(), full_vocabulary_top1_token=int(z.argmax()), full_vocabulary_top1_is_option=bool((ids == z.argmax()).any()), log_partition=float(z.logsumexp(-1)))
