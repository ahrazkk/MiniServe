"""Sampler, rejection sampler and n-gram proposer."""

from __future__ import annotations

import torch

from miniserve.engine.sequence import SamplingParams
from miniserve.runner.sampler import apply_top_k_top_p, compute_probs, sample
from miniserve.spec.ngram import propose_ngram
from miniserve.spec.rejection import verify_greedy, verify_sampled

CHI2_DF4_P001 = 18.47  # chi-square critical value: 4 degrees of freedom, p = 0.001


def chi_square(counts: torch.Tensor, probs: torch.Tensor) -> float:
    expected = probs * counts.sum()
    return float(((counts - expected) ** 2 / expected).sum())


# ---- sampler ------------------------------------------------------------------------------


def test_greedy_is_argmax() -> None:
    logits = torch.randn(6, 50)
    out = sample(logits, [SamplingParams(temperature=0)] * 6, [None] * 6)
    assert torch.equal(out, logits.argmax(-1))


def test_top_k_1_and_tiny_top_p_reduce_to_greedy() -> None:
    logits = torch.randn(4, 100)
    params = [SamplingParams(top_k=1), SamplingParams(top_p=1e-6)] * 2
    assert torch.equal(sample(logits, params, [None] * 4), logits.argmax(-1))


def test_top_k_top_p_masks() -> None:
    logits = torch.tensor([[1.0, 4.0, 3.0, 2.0, 0.0]])
    kept = apply_top_k_top_p(logits, torch.tensor([2]), torch.tensor([1.0])) > float("-inf")
    assert kept.tolist() == [[False, True, True, False, False]]
    # probabilities ~ [.03, .64, .23, .09, .01]: the smallest set reaching 0.8 is {4.0, 3.0}
    kept = apply_top_k_top_p(logits, torch.tensor([5]), torch.tensor([0.8])) > float("-inf")
    assert kept.tolist() == [[False, True, True, False, False]]


def test_sampling_matches_the_softmax_distribution() -> None:
    torch.manual_seed(0)
    logits = torch.tensor([[0.5, 1.5, -1.0, 0.0, 1.0]]).repeat(100_000, 1)
    tokens = sample(logits, [SamplingParams(temperature=1.0)] * 100_000, [None] * 100_000)
    counts = torch.bincount(tokens, minlength=5).double()
    assert chi_square(counts, logits[0].double().softmax(-1)) < CHI2_DF4_P001


def test_seeded_rows_are_reproducible() -> None:
    logits = torch.randn(3, 1000)
    params = [SamplingParams(temperature=1.0, seed=7)] * 3

    def gens() -> list[torch.Generator | None]:
        return [torch.Generator().manual_seed(7) for _ in range(3)]

    assert torch.equal(sample(logits, params, gens()), sample(logits, params, gens()))


# ---- speculative decoding -----------------------------------------------------------------


def test_verify_greedy_accepts_the_matching_prefix() -> None:
    target = torch.full((1, 4, 10), -5.0)
    for pos, tok in enumerate([4, 2, 9, 1]):  # what the target itself would pick
        target[0, pos, tok] = 5.0
    accepted, nxt = verify_greedy(target, torch.tensor([[4, 2, 7]]), torch.tensor([3]))
    assert accepted.item() == 2 and nxt.item() == 9  # d3=7 rejected, replaced by the target's 9
    accepted, nxt = verify_greedy(target, torch.tensor([[4, 2, 9]]), torch.tensor([3]))
    assert accepted.item() == 3 and nxt.item() == 1  # all accepted, plus the bonus token


def test_rejection_sampling_output_is_exactly_the_target_distribution() -> None:
    """The heart of speculative decoding: drafts change the speed, never the distribution."""
    torch.manual_seed(0)
    b, v = 200_000, 5
    p = torch.tensor([[0.1, 0.4, 0.2, 0.25, 0.05], [0.3, 0.1, 0.1, 0.2, 0.3]])  # positions 1, 2
    target = p.unsqueeze(0).expand(b, 2, v).contiguous()
    uniform = torch.rand(b, 1)
    noise = torch.empty(b, v).exponential_()

    # (a) A deterministic proposer (n-gram) that always guesses token 3.
    drafts = torch.full((b, 1), 3)
    acc, nxt = verify_sampled(target, drafts, torch.ones(b, dtype=torch.long), None, uniform, noise)
    first = torch.where(acc == 1, drafts[:, 0], nxt)
    assert chi_square(torch.bincount(first, minlength=v).double(), p[0].double()) < CHI2_DF4_P001
    second = nxt[acc == 1]  # the bonus token after an accepted draft
    assert chi_square(torch.bincount(second, minlength=v).double(), p[1].double()) < CHI2_DF4_P001

    # (b) A random draft model with its own distribution q.
    q = torch.tensor([0.3, 0.1, 0.3, 0.1, 0.2])
    drafts = torch.multinomial(q, b, replacement=True).unsqueeze(1)
    draft_probs = q.expand(b, 1, v).contiguous()
    acc, nxt = verify_sampled(
        target,
        drafts,
        torch.ones(b, dtype=torch.long),
        draft_probs,
        torch.rand(b, 1),
        torch.empty(b, v).exponential_(),
    )
    first = torch.where(acc == 1, drafts[:, 0], nxt)
    assert chi_square(torch.bincount(first, minlength=v).double(), p[0].double()) < CHI2_DF4_P001


def test_temperature_changes_the_distribution() -> None:
    logits = torch.tensor([[2.0, 1.0, 0.0]])
    sharp = compute_probs(logits, [SamplingParams(temperature=0.5)])
    flat = compute_probs(logits, [SamplingParams(temperature=2.0)])
    assert sharp[0, 0] > logits.softmax(-1)[0, 0] > flat[0, 0]


def test_ngram_proposes_what_followed_the_last_match() -> None:
    text = [1, 2, 3, 4, 5, 9, 9, 1, 2, 3]
    assert propose_ngram(text, k=3) == [4, 5, 9]  # "1 2 3" seen before, followed by 4 5 9
    assert propose_ngram([1, 2, 3], k=3) == []  # nothing repeats
    assert propose_ngram([7, 8, 7], k=2, n_max=3, n_min=1) == [8, 7]  # falls back to a 1-gram
