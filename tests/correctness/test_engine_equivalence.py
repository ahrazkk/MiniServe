"""The engine's optimizations must never change *what* is generated, only how fast.

Greedy decoding on the tiny float64 model, compared token for token:
* the engine vs plain Hugging Face greedy decoding;
* everything batched together vs each request alone;
* paged KV vs full reservation, prefix caching on vs off, chunked prefill, a tiny KV pool that
  forces preemption, and speculative decoding: all identical to the default configuration.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from miniserve.engine.engine import LLMEngine
from miniserve.engine.sequence import SamplingParams

N_OUT = 24
GREEDY = SamplingParams(max_tokens=N_OUT, temperature=0.0, ignore_eos=True)


def test_engine_matches_hugging_face(
    make_engine: Callable[..., LLMEngine],
    hf_greedy: Callable[[list[int], int], list[int]],
    prompts: list[list[int]],
) -> None:
    engine = make_engine()
    for prompt in prompts[:4]:
        assert engine.generate([prompt], GREEDY)[0] == hf_greedy(prompt, N_OUT)


def test_batched_equals_one_at_a_time(
    make_engine: Callable[..., LLMEngine], prompts: list[list[int]]
) -> None:
    engine = make_engine()
    alone = [engine.generate([p], GREEDY)[0] for p in prompts]
    assert engine.generate(prompts, GREEDY) == alone


@pytest.mark.parametrize(
    "overrides",
    [
        {"enable_prefix_caching": False},
        {"kv_reservation": "full", "enable_prefix_caching": False},
        {"max_num_batched_tokens": 12},  # long prompts get split across steps
        {"num_gpu_blocks": 64},  # 256 tokens of KV for ~700 needed: forces preemption
        {"speculative": "ngram"},
        {"speculative": "ngram", "num_gpu_blocks": 64, "max_num_batched_tokens": 12},
        {"max_num_seqs": 3},
    ],
    ids=[
        "no-prefix-cache",
        "full-reservation",
        "chunked-prefill",
        "preemption",
        "spec-ngram",
        "spec+preemption+chunking",
        "few-seqs",
    ],
)
def test_variants_match_default(
    make_engine: Callable[..., LLMEngine], prompts: list[list[int]], overrides: dict[str, Any]
) -> None:
    expected = make_engine().generate(prompts, GREEDY)
    engine = make_engine(**overrides)
    assert engine.generate(prompts, GREEDY) == expected
    # And nothing leaked: every block is free (or cached and evictable) again.
    assert engine.block_manager.num_free_blocks == engine.block_manager.num_blocks
    engine.block_manager.check_invariants([])


def test_tiny_pool_really_preempts(
    make_engine: Callable[..., LLMEngine], prompts: list[list[int]]
) -> None:
    engine = make_engine(num_gpu_blocks=64)
    engine.generate(prompts, GREEDY)
    assert engine.scheduler.num_preemptions > 0


def test_prefix_cache_skips_the_shared_prefix(make_engine: Callable[..., LLMEngine]) -> None:
    engine = make_engine(block_size=4)
    shared = list(range(10, 42))  # 32 tokens = 8 full blocks
    first, second = shared + [100, 101, 102], shared + [200, 201]
    engine.generate([first], GREEDY)
    hits_before = engine.prefix_cache_hits
    engine.generate([second], GREEDY)
    assert engine.prefix_cache_hits - hits_before == 32  # only the 2-token suffix was computed


def test_speculation_accepts_drafts_on_repetitive_text(
    make_engine: Callable[..., LLMEngine],
) -> None:
    engine = make_engine(speculative="ngram")
    # Greedy decoding of a tiny random model loops quickly, which n-gram drafts predict well.
    engine.generate(
        [[5, 6, 7, 8, 5, 6, 7, 8, 5, 6]],
        SamplingParams(max_tokens=60, temperature=0, ignore_eos=True),
    )
    assert engine.runner.spec_accepted > 0


def test_eos_stops_and_max_tokens_caps(make_engine: Callable[..., LLMEngine]) -> None:
    engine = make_engine()
    out = engine.generate([[9, 10, 11]], SamplingParams(max_tokens=7, temperature=0))[0]
    assert 1 <= len(out) <= 7
    assert len(out) == 7 or out[-1] == 3  # ran to the limit, or stopped on EOS (id 3)


def test_seeded_sampling_is_reproducible(
    make_engine: Callable[..., LLMEngine], prompts: list[list[int]]
) -> None:
    params = SamplingParams(max_tokens=16, temperature=1.0, top_p=0.9, seed=1234, ignore_eos=True)
    a = make_engine().generate([prompts[2]], params)[0]
    b = make_engine().generate(prompts, [params if i == 2 else GREEDY for i in range(8)])[2]
    assert a == b  # same seed, same tokens, regardless of what else was in the batch
