"""The paged-KV bookkeeping: allocation, sharing, eviction, and invariants under random use."""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from miniserve.engine.block_manager import BlockManager
from miniserve.engine.sequence import SamplingParams, Sequence


def seq(tokens: list[int], rid: str = "r") -> Sequence:
    return Sequence(rid, tokens, SamplingParams())


def admit(bm: BlockManager, s: Sequence) -> None:
    cached = bm.find_cached_prefix(s)
    assert bm.can_allocate(cached, s.num_tokens)
    bm.allocate(s, cached, s.num_tokens)
    s.num_computed = s.num_tokens  # pretend the prefill ran
    bm.cache_full_blocks(s)


def test_blocks_allocated_on_demand_and_returned() -> None:
    bm = BlockManager(num_blocks=8, block_size=4, enable_prefix_caching=False)
    s = seq(list(range(10)))  # 10 tokens -> 3 blocks (the last one partly filled)
    admit(bm, s)
    assert len(s.block_table) == 3 and bm.num_free_blocks == 5
    bm.append_slots(s, 2)  # tokens 11, 12 still fit in block 3
    assert len(s.block_table) == 3
    s.num_computed = 12
    bm.append_slots(s, 1)  # token 13 needs a 4th block
    assert len(s.block_table) == 4
    bm.free(s)
    assert bm.num_free_blocks == 8
    bm.check_invariants([])


def test_shared_prefix_blocks_are_reused_with_ref_counts() -> None:
    bm = BlockManager(num_blocks=16, block_size=4, enable_prefix_caching=True)
    a = seq(list(range(12)) + [99], "a")
    admit(bm, a)
    b = seq(list(range(12)) + [77, 78], "b")
    cached = bm.find_cached_prefix(b)
    assert cached == a.block_table[:3]  # the three full blocks of tokens 0..11
    bm.allocate(b, cached, b.num_tokens)
    assert b.num_computed == 12 and all(bm.ref_counts[x] == 2 for x in cached)
    bm.check_invariants([a, b])


def test_last_token_is_never_served_from_cache() -> None:
    bm = BlockManager(num_blocks=16, block_size=4, enable_prefix_caching=True)
    admit(bm, seq(list(range(8)), "a"))
    # Same 8 tokens: only the first block may be reused, so token 7 still runs and gives logits.
    assert len(bm.find_cached_prefix(seq(list(range(8)), "b"))) == 1


def test_same_block_after_a_different_prefix_is_not_reused() -> None:
    bm = BlockManager(num_blocks=16, block_size=4, enable_prefix_caching=True)
    admit(bm, seq([1, 2, 3, 4, 5, 6, 7, 8, 0], "a"))
    assert bm.find_cached_prefix(seq([9, 9, 9, 9, 5, 6, 7, 8, 0], "b")) == []


def test_freed_cached_blocks_are_evicted_lru_only_when_needed() -> None:
    bm = BlockManager(num_blocks=4, block_size=2, enable_prefix_caching=True)
    a = seq([1, 2, 3, 4, 5], "a")
    admit(bm, a)  # 3 blocks, 2 of them full and cached
    bm.free(a)
    assert bm.num_free_blocks == 4 and len(bm.cached) == 2  # contents kept
    b = seq([1, 2, 3, 4, 9], "b")
    assert len(bm.find_cached_prefix(b)) == 2  # still a hit after a finished
    big = seq(list(range(100, 108)), "big")  # needs all 4 blocks: evicts the cached ones
    admit(bm, big)
    assert bm.find_cached_prefix(b) == []
    bm.check_invariants([big])


ops = st.lists(
    st.tuples(st.sampled_from(["admit", "grow", "free"]), st.integers(0, 50), st.integers(1, 9)),
    max_size=60,
)


@settings(max_examples=200, deadline=None)
@given(ops=ops, prefix_caching=st.booleans())
def test_invariants_hold_under_random_operations(
    ops: list[tuple[str, int, int]], prefix_caching: bool
) -> None:
    """Whatever the sequence of admissions, growth and frees: no leaks, no double ownership."""
    bm = BlockManager(num_blocks=24, block_size=4, enable_prefix_caching=prefix_caching)
    live: list[Sequence] = []
    for i, (op, a, b) in enumerate(ops):
        if op == "admit":
            # Few distinct token values, so prefixes collide often and sharing gets exercised.
            s = seq([(a + j) % 3 for j in range(b * 3)], str(i))
            cached = bm.find_cached_prefix(s)
            if bm.can_allocate(cached, s.num_tokens):
                bm.allocate(s, cached, s.num_tokens)
                s.num_computed = s.num_tokens
                bm.cache_full_blocks(s)
                live.append(s)
        elif op == "grow" and live:
            s = live[a % len(live)]
            if bm.can_append(s, b):
                bm.append_slots(s, b)
                s.token_ids.extend([a % 3] * b)
                s.num_computed += b
                bm.cache_full_blocks(s)
        elif op == "free" and live:
            bm.free(live.pop(a % len(live)))
        bm.check_invariants(live)
        assert all(len(s.block_table) * 4 >= s.num_computed for s in live)
    for s in live:
        bm.free(s)
    assert bm.num_free_blocks == bm.num_blocks
    bm.check_invariants([])
