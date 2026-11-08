"""Scheduling rules, checked without a model: budgets, chunking, FCFS, preemption, fairness."""

from __future__ import annotations

from typing import Any

from hypothesis import given, settings
from hypothesis import strategies as st

from miniserve.config import EngineConfig
from miniserve.engine.block_manager import BlockManager
from miniserve.engine.scheduler import Scheduler, SchedulerOutput
from miniserve.engine.sequence import SamplingParams, Sequence


def make(**overrides: Any) -> Scheduler:
    settings_: dict[str, Any] = {
        "model": "x",
        "device": "cpu",
        "block_size": 4,
        "max_model_len": 64,
        "max_num_seqs": 8,
        "max_num_batched_tokens": 32,
        **overrides,
    }
    cfg = EngineConfig(**settings_)
    blocks = overrides.get("num_gpu_blocks", 64)
    return Scheduler(cfg, BlockManager(blocks, cfg.block_size, cfg.enable_prefix_caching))


def fake_tokens(out: SchedulerOutput) -> list[list[int]]:
    """What the model runner would return: a token for each sequence whose chunk ends its input."""
    return [
        [7] if s.seq.num_computed + s.num_new_tokens >= s.seq.num_tokens else []
        for s in out.scheduled
    ]


def run_to_completion(sched: Scheduler, max_steps: int = 10_000) -> list[SchedulerOutput]:
    steps = []
    while sched.has_unfinished() and len(steps) < max_steps:
        out = sched.schedule()
        sched.update(out, fake_tokens(out))
        steps.append(out)
    assert not sched.has_unfinished(), "requests starved"
    return steps


def add(sched: Scheduler, n_prompt: int, n_out: int, rid: str) -> Sequence:
    s = Sequence(
        rid, list(range(1, n_prompt + 1)), SamplingParams(max_tokens=n_out, ignore_eos=True)
    )
    sched.add(s)
    return s


def test_budget_and_batch_size_are_never_exceeded() -> None:
    sched = make(max_num_batched_tokens=32, max_num_seqs=4)
    for i in range(12):
        add(sched, 10 + i, 5, str(i))
    for out in run_to_completion(sched):
        assert out.num_tokens <= 32
        assert len(out.scheduled) <= 4


def test_long_prompt_is_chunked_across_steps() -> None:
    sched = make(max_num_batched_tokens=16)
    s = add(sched, 40, 2, "long")
    first = sched.schedule()
    assert first.scheduled[0].num_new_tokens == 16 and s.num_computed == 0
    sched.update(first, fake_tokens(first))
    assert s.num_computed == 16 and s.num_output_tokens == 0  # no token until the prompt is done


def test_without_chunking_a_prompt_runs_in_one_step() -> None:
    sched = make(enable_chunked_prefill=False)  # the budget is raised to max_model_len
    add(sched, 40, 2, "long")
    assert sched.schedule().scheduled[0].num_new_tokens == 40


def test_decodes_are_scheduled_before_new_prefills() -> None:
    sched = make(max_num_batched_tokens=20)
    a = add(sched, 8, 50, "a")
    sched.update(out := sched.schedule(), fake_tokens(out))  # a is now decoding
    add(sched, 40, 5, "b")
    out = sched.schedule()
    assert out.scheduled[0].seq is a and out.scheduled[0].num_new_tokens == 1
    assert out.scheduled[1].num_new_tokens == 19  # b gets the rest of the budget


def test_queued_requests_all_finish_with_few_slots() -> None:
    sched = make(max_num_seqs=2)
    seqs = [add(sched, 6, 4, str(i)) for i in range(6)]
    run_to_completion(sched)
    assert all(s.is_finished for s in seqs)


def test_preemption_under_memory_pressure_still_finishes_everything() -> None:
    sched = make(num_gpu_blocks=16, max_num_seqs=8, max_num_batched_tokens=64)  # 64 tokens of KV
    seqs = [add(sched, 10, 20, str(i)) for i in range(6)]  # wants 6 x 30 = 180 tokens
    run_to_completion(sched)
    assert sched.num_preemptions > 0
    assert all(s.num_output_tokens == 20 for s in seqs)
    assert sched.bm.num_free_blocks == sched.bm.num_blocks


def test_full_reservation_admits_far_fewer_sequences() -> None:
    paged = make(num_gpu_blocks=64, max_model_len=64, max_num_batched_tokens=64)
    full = make(
        num_gpu_blocks=64, max_model_len=64, max_num_batched_tokens=64, kv_reservation="full"
    )
    for sched in (paged, full):
        for i in range(8):
            add(sched, 8, 4, str(i))
        sched.schedule()
    assert len(paged.running) == 8
    assert len(full.running) == 4  # 64 blocks / (64 tokens / 4 per block) = 4 slots


def test_abort_frees_blocks() -> None:
    sched = make()
    add(sched, 20, 10, "x")
    sched.schedule()
    assert sched.abort("x") is not None
    assert sched.bm.num_free_blocks == sched.bm.num_blocks and not sched.has_unfinished()


@settings(max_examples=60, deadline=None)
@given(
    lens=st.lists(st.tuples(st.integers(1, 40), st.integers(1, 20)), min_size=1, max_size=12),
    blocks=st.integers(16, 48),
    budget=st.integers(4, 64),
)
def test_every_request_finishes_with_its_tokens(
    lens: list[tuple[int, int]], blocks: int, budget: int
) -> None:
    sched = make(num_gpu_blocks=blocks, max_num_batched_tokens=budget)
    seqs = [add(sched, p, o, str(i)) for i, (p, o) in enumerate(lens)]
    for out in run_to_completion(sched):
        assert out.num_tokens <= budget
    assert all(s.num_output_tokens == o for s, (_, o) in zip(seqs, lens, strict=True))
    sched.bm.check_invariants([])
