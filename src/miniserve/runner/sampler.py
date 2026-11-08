"""Logits to tokens: greedy, temperature, top-k and top-p, batched with per-request settings."""

from __future__ import annotations

import torch
from torch import Tensor

from miniserve.engine.sequence import SamplingParams
from miniserve.utils import to_device


def apply_top_k_top_p(logits: Tensor, top_k: Tensor, top_p: Tensor) -> Tensor:
    """Set every logit outside a row's top-k and top-p nucleus to -inf.

    top_k: [R] (vocab size = disabled), top_p: [R] (1.0 = disabled). One sort serves both.
    """
    vocab = logits.shape[-1]
    sorted_logits, sorted_idx = logits.sort(dim=-1, descending=False)
    # top-k: in ascending order the k-th largest value sits at index vocab - k.
    kth = sorted_logits.gather(1, (vocab - top_k).clamp(0, vocab - 1).unsqueeze(1))
    sorted_logits = sorted_logits.masked_fill(sorted_logits < kth, float("-inf"))
    # top-p: drop the least likely tokens while their total probability stays below 1 - p.
    cum = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
    drop = cum <= (1.0 - top_p).unsqueeze(1)
    drop[:, -1] = False  # always keep the most likely token
    sorted_logits = sorted_logits.masked_fill(drop, float("-inf"))
    return torch.empty_like(logits).scatter_(1, sorted_idx, sorted_logits)


def compute_probs(logits: Tensor, params: list[SamplingParams]) -> Tensor:
    """Probabilities after temperature, top-k and top-p, each row with its request's settings."""
    vocab = logits.shape[-1]
    dev = logits.device
    temps = to_device([max(p.temperature, 1e-5) for p in params], dev, logits.dtype)
    logits = logits / temps.unsqueeze(1)
    if any(0 < p.top_k < vocab or p.top_p < 1.0 for p in params):
        top_k = to_device([p.top_k if 0 < p.top_k < vocab else vocab for p in params], dev)
        top_p = to_device([p.top_p for p in params], dev, logits.dtype)
        logits = apply_top_k_top_p(logits, top_k, top_p)
    return logits.softmax(dim=-1)


def exponential_noise(
    shape: tuple[int, int], generators: list[torch.Generator | None], device: torch.device
) -> Tensor:
    """Exp(1) noise per row; rows with their own generator (a seeded request) use it."""
    noise = torch.empty(shape, device=device).exponential_()
    for i, g in enumerate(generators):
        if g is not None:
            noise[i].exponential_(generator=g)
    # exponential_ can return exactly 0, and 0/0 = NaN would win the argmax.
    return noise.clamp_(min=torch.finfo(noise.dtype).tiny)


def sample(
    logits: Tensor, params: list[SamplingParams], generators: list[torch.Generator | None]
) -> Tensor:
    """logits [R, vocab] -> token ids [R], staying on the GPU (the caller syncs once per step)."""
    greedy_tokens = logits.argmax(dim=-1)
    if all(p.greedy for p in params):
        return greedy_tokens
    probs = compute_probs(logits, params)
    # The "exponential race": argmax(p / E) with E ~ Exp(1) is an exact sample from p (the same
    # trick as Gumbel-max). Unlike torch.multinomial it is one batched op, and a per-row
    # generator makes seeded requests reproducible no matter what else is in the batch.
    noise = exponential_noise((probs.shape[0], probs.shape[1]), generators, logits.device)
    sampled = (probs / noise).argmax(dim=-1)
    is_greedy = to_device([p.greedy for p in params], logits.device, torch.bool)
    return torch.where(is_greedy, greedy_tokens, sampled)
