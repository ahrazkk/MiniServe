"""Fixed, seeded prompt sets for the benchmarks (same seed -> same prompts for every system).

Prompts are sent as token-id lists, so every system sees exactly `input_len` tokens (decoding
random ids to text and re-tokenizing would change the length). This matches what
`vllm bench serve --dataset-name random` measures: throughput and latency, not answer quality.
"""

from __future__ import annotations

import random

FIRST_SPECIAL_ID = 151643  # Qwen3: ids from here on are special tokens (<|endoftext|>, ...)


def random_prompts(n: int, input_len: int, seed: int = 0) -> list[list[int]]:
    rng = random.Random(seed)
    return [[rng.randrange(FIRST_SPECIAL_ID) for _ in range(input_len)] for _ in range(n)]


def shared_prefix_prompts(
    n: int, prefix_len: int, suffix_len: int, seed: int = 0
) -> list[list[int]]:
    """W3: one long shared prefix (a system prompt, say) + a unique suffix per request."""
    rng = random.Random(seed)
    prefix = [rng.randrange(FIRST_SPECIAL_ID) for _ in range(prefix_len)]
    return [prefix + [rng.randrange(FIRST_SPECIAL_ID) for _ in range(suffix_len)] for _ in range(n)]
