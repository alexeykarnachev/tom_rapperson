import torch
import torch.nn.functional


def get_unlikelihood_loss(logits, target):
    candidates_logp = _get_candidates_logp(logits=logits, target=target)
    loss = _get_loss(candidates_logp)
    return loss


def _get_candidates_logp(logits, target):
    logp = torch.nn.functional.log_softmax(logits, 2)
    bs = logits.size()[0]
    seq_len = logits.size()[1]

    logp_flat = logp.view(bs * seq_len, -1)
    tril = torch.tril(torch.ones((seq_len, seq_len), device=logits.device), diagonal=-1)

    cols = target.repeat(1, seq_len).view(seq_len * bs, -1)
    rows = torch.arange(0, cols.size()[0]).unsqueeze(-1).repeat(1, seq_len)

    not_ignore_mask = ~(cols == target.flatten().unsqueeze(-1).repeat(1, seq_len))
    candidates_mask = tril.repeat(bs, 1).bool()
    candidates_mask *= not_ignore_mask

    cols_flat = cols[candidates_mask]
    rows_flat = rows[candidates_mask]

    logp = [] if len(cols_flat) == 0 else logp_flat[rows_flat, cols_flat]

    return logp


def _get_loss(candidates_logp):
    if len(candidates_logp) == 0:
        return 0
    probs = torch.clamp((1.0 - candidates_logp.exp()), min=1e-5)
    loss = -torch.log(probs)
    loss = loss.mean()
    return loss
