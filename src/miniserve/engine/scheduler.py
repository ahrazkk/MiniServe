"""Continuous batching: at every step, decide which sequences run and how many tokens each gets.

There is no separate "prefill phase" and "decode phase". Each step has a token budget
(`max_num_batched_tokens`); every running sequence just needs its next N tokens computed:
N = 1 for a decode, N = a chunk of the prompt for a (chunked) prefill, N = 1 + drafts when
speculating. Running sequences are served first (oldest first), then new requests fill whatever
budget and memory are left. A finished sequence leaves immediately and a new one can take its
place at the very next step: that is the whole difference from static batching.

When KV memory runs out, the newest running sequence is *preempted*: its blocks are freed and it
goes back to the front of the waiting queue. Later its prompt plus everything it generated so far
is prefilled again ("recompute" preemption, what vLLM's V1 engine does). Its output so far is kept,
so the client never notices except as a pause.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

from miniserve.config import EngineConfig
from miniserve.engine.block_manager import BlockManager
from miniserve.engine.sequence import SeqStatus, Sequence

# A draft proposer: (all token ids so far, max drafts) -> proposed next tokens.
Proposer = Callable[[list[int], int], list[int]]


@dataclass
class ScheduledSeq:
    seq: Sequence
    num_new_tokens: int  # 1 for a decode, a chunk for a prefill, 1 + drafts when speculating


@dataclass
class SchedulerOutput:
    scheduled: list[ScheduledSeq] = field(default_factory=list)
    num_preempted: int = 0

    @property
    def num_tokens(self) -> int:
        return sum(s.num_new_tokens for s in self.scheduled)


class Scheduler:
    def __init__(
        self, cfg: EngineConfig, block_manager: BlockManager, proposer: Proposer | None = None
    ) -> None:
        self.cfg = cfg
        self.bm = block_manager
        self.proposer = proposer
        self.waiting: deque[Sequence] = deque()
        self.running: list[Sequence] = []  # admission order: oldest first
        self.num_preemptions = 0
        self.num_prefix_hit_tokens = 0  # tokens skipped thanks to the prefix cache

    def add(self, seq: Sequence) -> None:
        self.waiting.append(seq)

    def has_unfinished(self) -> bool:
        return bool(self.waiting or self.running)

    def abort(self, request_id: str) -> Sequence | None:
        for queue in (self.running, self.waiting):
            for seq in queue:
                if seq.request_id == request_id:
                    queue.remove(seq)
                    self._finish(seq, "abort")
                    return seq
        return None

    # ---- the per-step decision ------------------------------------------------------------

    def schedule(self) -> SchedulerOutput:
        cfg, bm = self.cfg, self.bm
        out = SchedulerOutput()
        budget = cfg.max_num_batched_tokens

        # 1) Running sequences, oldest first.
        i = 0
        while i < len(self.running) and budget > 0:
            seq = self.running[i]
            num_new = seq.num_tokens - seq.num_computed
            drafts: list[int] = []
            if num_new == 1 and self.proposer is not None:
                drafts = self._propose(seq, budget)
                num_new += len(drafts)
            num_new = min(num_new, budget)
            # Make room for the new tokens' KV, preempting the newest running sequences.
            while not bm.can_append(seq, num_new):
                victim = self.running.pop()
                self._preempt(victim)
                out.num_preempted += 1
                if victim is seq:
                    break
            if seq.status is not SeqStatus.RUNNING:
                break  # it had to preempt itself: nothing newer was left to free
            bm.append_slots(seq, num_new)
            seq.spec_token_ids = drafts
            out.scheduled.append(ScheduledSeq(seq, num_new))
            budget -= num_new
            i += 1

        # 2) Waiting requests, first come first served. Skipped in a step that had to preempt:
        #    memory is already short, admitting more would only cause more preemption.
        if out.num_preempted:
            return out
        while self.waiting and budget > 0 and len(self.running) < cfg.max_num_seqs:
            seq = self.waiting[0]
            cached = bm.find_cached_prefix(seq)
            num_cached = len(cached) * bm.block_size
            num_new = seq.num_tokens - num_cached
            if num_new > budget:
                if not cfg.enable_chunked_prefill:
                    break  # the whole prompt must fit in one step; wait for a lighter step
                num_new = budget  # chunked prefill: do what fits now, the rest next step
            reserve = cfg.max_model_len if cfg.kv_reservation == "full" else num_cached + num_new
            if not bm.can_allocate(cached, reserve):
                break
            self.waiting.popleft()
            bm.allocate(seq, cached, reserve)
            self.num_prefix_hit_tokens += num_cached
            seq.status = SeqStatus.RUNNING
            self.running.append(seq)
            out.scheduled.append(ScheduledSeq(seq, num_new))
            budget -= num_new
        return out

    def update(
        self, out: SchedulerOutput, sampled: list[list[int]]
    ) -> list[tuple[Sequence, list[int]]]:
        """Apply one step's results: count computed tokens, append new ones, retire finished."""
        results: list[tuple[Sequence, list[int]]] = []
        for s, tokens in zip(out.scheduled, sampled, strict=True):
            seq = s.seq
            if seq.status is not SeqStatus.RUNNING:
                continue
            if seq.spec_token_ids:
                # The step checked [last token, d1..dk]; `tokens` = accepted drafts + one more.
                # Rejected drafts' KV stays in the slots but is not counted, so it gets overwritten.
                seq.num_computed += len(tokens)
            else:
                seq.num_computed += s.num_new_tokens
            seq.spec_token_ids = []
            appended: list[int] = []
            reason = None
            for token in tokens:  # empty while a prefill chunk hasn't reached the prompt's end
                appended.append(token)
                reason = seq.append_token(token, self.cfg.max_model_len)
                if reason:
                    break
            self.bm.cache_full_blocks(seq)
            if reason:
                self.running.remove(seq)
                self._finish(seq, reason)
            results.append((seq, appended))
        return results

    # ---- helpers --------------------------------------------------------------------------

    def _propose(self, seq: Sequence, budget: int) -> list[int]:
        assert self.proposer is not None
        k = min(
            self.cfg.num_speculative_tokens,
            budget - 1,
            seq.params.max_tokens - seq.num_output_tokens - 1,  # room for drafts + bonus token
            self.cfg.max_model_len - seq.num_tokens - 1,
        )
        return self.proposer(seq.token_ids, k)[:k] if k > 0 else []

    def _preempt(self, seq: Sequence) -> None:
        self.bm.free(seq)
        seq.status = SeqStatus.WAITING
        seq.num_computed = 0
        seq.spec_token_ids = []
        seq.num_preemptions += 1
        self.num_preemptions += 1
        self.waiting.appendleft(seq)  # it was admitted before everything still waiting

    def _finish(self, seq: Sequence, reason: str) -> None:
        self.bm.free(seq)
        seq.status = SeqStatus.FINISHED
        seq.finish_reason = reason
