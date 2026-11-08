"""One request inside the engine: its tokens, its sampling settings, where its KV lives."""

from __future__ import annotations

import enum
import time
from collections.abc import Iterable
from dataclasses import dataclass, field

import torch


@dataclass
class SamplingParams:
    """How to pick tokens and when to stop (the subset of OpenAI's parameters we support)."""

    max_tokens: int = 256
    temperature: float = 1.0  # 0 means greedy: always take the highest-scoring token
    top_p: float = 1.0  # 1.0 disables nucleus sampling
    top_k: int = -1  # -1 disables top-k
    seed: int | None = None
    ignore_eos: bool = False  # keep generating past end-of-sequence (benchmarks fix output length)
    stop_token_ids: list[int] = field(default_factory=list)

    @property
    def greedy(self) -> bool:
        return self.temperature < 1e-5


class SeqStatus(enum.Enum):
    WAITING = enum.auto()  # queued, owns no KV blocks
    RUNNING = enum.auto()  # in the batch, owns KV blocks
    FINISHED = enum.auto()


class Sequence:
    """A request inside the engine.

    `token_ids` is the prompt followed by everything generated so far. `num_computed` counts the
    tokens whose keys and values are already in the KV cache, so the work left before the next
    token can be sampled is `num_tokens - num_computed`: the whole prompt for a new request, the
    rest of it during chunked prefill, and exactly 1 (the last sampled token) while decoding.
    """

    def __init__(
        self,
        request_id: str,
        prompt_ids: Iterable[int],
        params: SamplingParams,
        eos_token_ids: frozenset[int] = frozenset(),
    ) -> None:
        self.request_id = request_id
        self.token_ids: list[int] = list(prompt_ids)
        self.num_prompt_tokens = len(self.token_ids)
        self.params = params
        self.eos_token_ids = eos_token_ids
        self.status = SeqStatus.WAITING

        self.block_table: list[int] = []  # physical KV block of each logical block, in order
        self.block_hashes: list[int] = []  # prefix-cache hash of each *full* block
        self.num_computed = 0
        self.num_cached_tokens = 0  # tokens reused from the prefix cache at admission (metrics)
        self.spec_token_ids: list[int] = []  # draft tokens being verified in the current step

        self.finish_reason: str | None = None
        self.num_preemptions = 0
        self.arrival_time = time.perf_counter()
        self.first_token_time: float | None = None
        self.generator: torch.Generator | None = None  # per-request RNG when a seed is given

    @property
    def num_tokens(self) -> int:
        return len(self.token_ids)

    @property
    def num_output_tokens(self) -> int:
        return len(self.token_ids) - self.num_prompt_tokens

    @property
    def output_ids(self) -> list[int]:
        return self.token_ids[self.num_prompt_tokens :]

    @property
    def is_finished(self) -> bool:
        return self.status is SeqStatus.FINISHED

    def append_token(self, token_id: int, max_model_len: int) -> str | None:
        """Add one sampled token. Returns the finish reason if the sequence must stop now."""
        self.token_ids.append(token_id)
        if self.first_token_time is None:
            self.first_token_time = time.perf_counter()
        if not self.params.ignore_eos and token_id in self.eos_token_ids:
            return "stop"
        if token_id in self.params.stop_token_ids:
            return "stop"
        if self.num_output_tokens >= self.params.max_tokens or self.num_tokens >= max_model_len:
            return "length"
        return None

    def __repr__(self) -> str:
        return (
            f"Sequence({self.request_id!r}, {self.status.name}, tokens={self.num_tokens}, "
            f"computed={self.num_computed}, blocks={len(self.block_table)})"
        )
