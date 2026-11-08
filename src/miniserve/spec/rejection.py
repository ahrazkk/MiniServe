"""Verify speculative drafts so that the output distribution is *exactly* the target model's.

The target model scores [last token, d1, ..., dk] in one forward pass, giving its own
distribution p_i at every position. For i = 1..k we accept draft d_i with probability
min(1, p_i(d_i) / q_i(d_i)), where q is the draft's distribution. At the first rejection we
sample a replacement from normalize(max(0, p_i - q_i)) and stop; if all k are accepted we sample
one bonus token from p_{k+1}. Leviathan et al. (2023) prove the result is distributed exactly as
sampling from the target alone. With greedy decoding it reduces to "accept while the draft equals
the target's argmax". An n-gram proposer is deterministic (q is one-hot), so the rule becomes
"accept d_i with probability p_i(d_i)" and the residual is p_i with d_i removed.

Everything is batched over sequences and stays on the GPU (no .item() syncs). Sequences may have
fewer drafts than the padded width K; `num_drafts` says how many are real.
"""

from __future__ import annotations

import torch
from torch import Tensor


def _leading_run(flags: Tensor) -> Tensor:
    """Length of the leading run of True in each row: [B, K] bool -> [B]."""
    return flags.int().cumprod(dim=1).sum(dim=1)


def verify_greedy(
    target_logits: Tensor, drafts: Tensor, num_drafts: Tensor
) -> tuple[Tensor, Tensor]:
    """Greedy verification.

    target_logits [B, K+1, V], drafts [B, K], num_drafts [B] -> (accepted [B], next_token [B]).
    """
    k = drafts.shape[1]
    target = target_logits.argmax(dim=-1)  # [B, K+1] the target's own pick at each position
    real = torch.arange(k, device=drafts.device)[None, :] < num_drafts[:, None]
    accepted = _leading_run((target[:, :k] == drafts) & real)
    next_token = target.gather(1, accepted.unsqueeze(1)).squeeze(1)  # correction, or the bonus
    return accepted, next_token


def verify_sampled(
    target_probs: Tensor,
    drafts: Tensor,
    num_drafts: Tensor,
    draft_probs: Tensor | None,
    uniform: Tensor,
    noise: Tensor,
) -> tuple[Tensor, Tensor]:
    """Rejection sampling.

    target_probs [B, K+1, V]; drafts [B, K]; num_drafts [B]; draft_probs [B, K, V] or None for a
    deterministic proposer (n-gram); uniform [B, K] ~ U(0,1); noise [B, V] ~ Exp(1).
    Returns (accepted [B], next_token [B]).
    """
    b, k = drafts.shape
    rows = torch.arange(b, device=drafts.device)
    real = torch.arange(k, device=drafts.device)[None, :] < num_drafts[:, None]
    p_draft = target_probs[:, :k].gather(2, drafts.unsqueeze(2)).squeeze(2)  # p_i(d_i)
    if draft_probs is None:
        ratio = p_draft
    else:
        q_draft = draft_probs.gather(2, drafts.unsqueeze(2)).squeeze(2)
        ratio = p_draft / q_draft.clamp(min=1e-20)
    accepted = _leading_run((uniform < ratio.clamp(max=1.0)) & real)

    p_row = target_probs[rows, accepted]  # [B, V] target at the first rejected (or bonus) position
    at = accepted.clamp(max=k - 1)
    if draft_probs is None:
        rejected_token = drafts[rows, at]
        residual = p_row.scatter(1, rejected_token.unsqueeze(1), 0.0)  # max(0, p - one_hot)
    else:
        residual = (p_row - draft_probs[rows, at]).clamp(min=0.0)
    was_rejected = (accepted < num_drafts).unsqueeze(1)
    dist = torch.where(was_rejected, residual, p_row)  # all accepted: bonus token from p_{k+1}
    total = dist.sum(dim=1, keepdim=True)
    dist = torch.where(total > 0, dist / total.clamp(min=1e-20), p_row)
    next_token = (dist / noise.clamp(min=torch.finfo(noise.dtype).tiny)).argmax(dim=1)
    return accepted, next_token
