"""Paged KV-cache memory: a pool of fixed-size blocks handed out on demand, like OS virtual memory.

Every sequence owns a *block table*: logical block i of the sequence lives in physical block
`block_table[i]`, so the token at position p is stored in slot

    block_table[p // block_size] * block_size + p % block_size

Blocks are taken from a free list as a sequence grows and given back when it finishes, so at most
one partly filled block per sequence is wasted (instead of reserving max_model_len for everyone).

Prefix caching: once a block is *full*, it is registered under a hash of (the previous block's
hash, its own tokens). A later request whose prompt starts with the same tokens reuses those
physical blocks instead of recomputing them. The hash chains through every earlier block because
a token's keys and values depend on all the tokens before it: the same 16 tokens after a
different prefix are a different block. Freed blocks that are cached keep their contents in an
LRU list and are only overwritten when memory runs out.

(Simplification: we trust Python's 64-bit hash() and never compare tokens. A collision would
silently reuse the wrong block; vLLM offers SHA-256 hashing for exactly this reason.)
"""

from __future__ import annotations

from collections import OrderedDict, deque

from miniserve.engine.sequence import Sequence


def cdiv(a: int, b: int) -> int:
    return -(-a // b)


class BlockManager:
    def __init__(self, num_blocks: int, block_size: int, enable_prefix_caching: bool) -> None:
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.enable_prefix_caching = enable_prefix_caching

        self.ref_counts = [0] * num_blocks  # how many sequences use each block
        self.free_ids: deque[int] = deque(range(num_blocks))  # free and holding nothing useful
        # Prefix cache.
        self.block_hash: list[int | None] = [None] * num_blocks  # hash of a cached block
        self.cached: dict[int, int] = {}  # hash -> physical block holding those tokens
        self.evictable: OrderedDict[int, None] = OrderedDict()  # cached, unused, oldest first

    # ---- capacity -------------------------------------------------------------------------

    @property
    def num_free_blocks(self) -> int:
        """Blocks we could hand out right now (truly free + cached-but-unused)."""
        return len(self.free_ids) + len(self.evictable)

    @property
    def num_used_blocks(self) -> int:
        return self.num_blocks - self.num_free_blocks

    def blocks_needed(self, seq: Sequence, num_tokens: int) -> int:
        """New blocks `seq` needs to hold `num_tokens` tokens in total."""
        return max(0, cdiv(num_tokens, self.block_size) - len(seq.block_table))

    # ---- admission ------------------------------------------------------------------------

    def find_cached_prefix(self, seq: Sequence) -> list[int]:
        """Physical blocks already holding this sequence's leading full blocks. No side effects."""
        if not self.enable_prefix_caching:
            return []
        hits: list[int] = []
        parent: int | None = None
        bs = self.block_size
        # Never match the last token: it has to go through the model to produce logits.
        for i in range((seq.num_tokens - 1) // bs):
            h = hash((parent, tuple(seq.token_ids[i * bs : (i + 1) * bs])))
            block = self.cached.get(h)
            if block is None:
                break
            hits.append(block)
            parent = h
        return hits

    def can_allocate(self, cached_blocks: list[int], num_tokens: int) -> bool:
        """Can a new sequence that reuses `cached_blocks` get room for `num_tokens` tokens?"""
        num_new = cdiv(num_tokens, self.block_size) - len(cached_blocks)
        # A cached hit that nobody is using counts as "free" until we take it.
        unused_hits = sum(1 for b in cached_blocks if self.ref_counts[b] == 0)
        return num_new <= self.num_free_blocks - unused_hits

    def allocate(self, seq: Sequence, cached_blocks: list[int], num_tokens: int) -> None:
        """Admit `seq`: share its cached prefix blocks, then take fresh blocks for the rest."""
        assert not seq.block_table, "sequence already holds blocks"
        for block in cached_blocks:
            if self.ref_counts[block] == 0:
                del self.evictable[block]
            self.ref_counts[block] += 1
        seq.block_table = list(cached_blocks)
        seq.block_hashes = [self.block_hash[b] for b in cached_blocks]  # type: ignore[misc]
        seq.num_computed = seq.num_cached_tokens = len(cached_blocks) * self.block_size
        for _ in range(self.blocks_needed(seq, num_tokens)):
            seq.block_table.append(self._take_free_block())

    # ---- growth ---------------------------------------------------------------------------

    def can_append(self, seq: Sequence, num_new_tokens: int) -> bool:
        need = self.blocks_needed(seq, seq.num_computed + num_new_tokens)
        return need <= self.num_free_blocks

    def append_slots(self, seq: Sequence, num_new_tokens: int) -> None:
        """Make sure `seq` has blocks for its next `num_new_tokens` tokens."""
        for _ in range(self.blocks_needed(seq, seq.num_computed + num_new_tokens)):
            seq.block_table.append(self._take_free_block())

    def cache_full_blocks(self, seq: Sequence) -> None:
        """Register blocks that just became full (every token computed) in the prefix cache."""
        if not self.enable_prefix_caching:
            return
        bs = self.block_size
        for i in range(len(seq.block_hashes), seq.num_computed // bs):
            parent = seq.block_hashes[-1] if seq.block_hashes else None
            h = hash((parent, tuple(seq.token_ids[i * bs : (i + 1) * bs])))
            seq.block_hashes.append(h)
            block = seq.block_table[i]
            if h not in self.cached:  # first copy wins; an identical duplicate stays private
                self.cached[h] = block
                self.block_hash[block] = h

    # ---- release --------------------------------------------------------------------------

    def free(self, seq: Sequence) -> None:
        """Give back every block of `seq`. Cached blocks keep their contents for reuse."""
        # Reverse order: a sequence's tail is the least likely to be shared, so it should be
        # evicted first (the evictable list pops from its oldest end).
        for block in reversed(seq.block_table):
            self.ref_counts[block] -= 1
            assert self.ref_counts[block] >= 0, f"block {block} freed twice"
            if self.ref_counts[block] == 0:
                if self.block_hash[block] is not None:
                    self.evictable[block] = None
                else:
                    self.free_ids.append(block)
        seq.block_table = []
        seq.block_hashes = []

    def _take_free_block(self) -> int:
        if self.free_ids:
            block = self.free_ids.popleft()
        elif self.evictable:
            block, _ = self.evictable.popitem(last=False)  # least recently freed cached block
            h = self.block_hash[block]
            assert h is not None
            del self.cached[h]
            self.block_hash[block] = None
        else:
            raise RuntimeError("out of KV blocks (the scheduler should have checked)")
        self.ref_counts[block] = 1
        return block

    # ---- checks (used by the property-based tests) ----------------------------------------

    def check_invariants(self, live: list[Sequence]) -> None:
        """Every block is in exactly one state, and ref counts match the block tables."""
        expected = [0] * self.num_blocks
        for seq in live:
            for b in seq.block_table:
                expected[b] += 1
        assert expected == self.ref_counts, "ref counts disagree with block tables"
        free, evictable = set(self.free_ids), set(self.evictable)
        assert len(free) == len(self.free_ids), "duplicate in free list"
        assert not free & evictable, "block both free and evictable"
        for b in range(self.num_blocks):
            in_use = self.ref_counts[b] > 0
            assert in_use + (b in free) + (b in evictable) == 1, f"block {b} in {in_use=} state"
            if b in free:
                assert self.block_hash[b] is None
            if b in evictable:
                assert self.block_hash[b] is not None
        for h, b in self.cached.items():
            assert self.block_hash[b] == h, "hash table out of sync"
