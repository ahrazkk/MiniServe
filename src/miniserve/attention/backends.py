"""Paged attention in plain PyTorch.

Per layer, keys and values live in one pool of slots, `[num_slots, kv_heads, head_dim]`, where
slot = physical block * block_size + offset in the block. Each step:

1. write the new tokens' keys and values into their slots (`slot_mapping`), then
2. for each sequence, read its keys and values back through its block table and attend.

"Reading through the block table" here means *gathering* a sequence's keys into a contiguous
tensor first. That extra copy is the price of not having a custom kernel: a paged-attention
kernel (FlashAttention's paged mode, or our own Triton kernel, a planned level-up) reads the
blocks in place instead.

Two paths:
* sequences with only a few new tokens (decodes, speculative verifies) are padded into one batch
  per group and attended with a single call;
* longer prefill chunks are attended one sequence at a time (each is already a big matmul).

The metadata is built once per step and shared by all 28 layers.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from miniserve.utils import to_device

SHORT_QUERY = 8  # sequences with at most this many new tokens go through the batched path
MAX_GATHER_TOKENS = 16384  # cap on (sequences x context) gathered at once, to bound memory


@dataclass
class SeqChunk:
    """What attention needs to know about one scheduled sequence."""

    flat_start: int  # where its new tokens start in the flattened batch
    num_new: int  # new tokens this step (its queries)
    num_cached: int  # tokens already in the KV cache before this step
    block_table: list[int]

    @property
    def context_len(self) -> int:
        return self.num_cached + self.num_new


@dataclass
class BatchedGroup:
    q_index: Tensor  # [b, Q] flat-batch row of each (sequence, query); padding repeats a real one
    valid_rows: Tensor | None  # [n] rows of the [b * Q] output that are real (None: all are)
    out_index: Tensor  # [n] flat-batch rows those real queries write to
    ctx_slots: Tensor  # [b, L] KV slot of every context position, via the block table
    mask: Tensor  # [b, Q, L] True where query i may attend to key j


@dataclass
class PrefillChunk:
    q_start: int
    q_end: int
    ctx_slots: Tensor | None  # [L] slots of the whole context; None for a fresh prompt
    mask: Tensor | None  # [n, L] causal mask shifted by the cached length; None = plain causal


@dataclass
class AttentionMetadata:
    slot_mapping: Tensor  # [T] slot where each new token's K/V is written
    groups: list[BatchedGroup]
    prefills: list[PrefillChunk]


def context_slots(block_tables: Tensor, length: int, block_size: int) -> Tensor:
    """Slot of positions 0..length-1 through a block table: the page-table lookup, vectorized."""
    pos = torch.arange(length, device=block_tables.device)
    return block_tables[..., pos // block_size] * block_size + pos % block_size


def build_attention_metadata(
    chunks: list[SeqChunk], slot_mapping: Tensor, block_size: int
) -> AttentionMetadata:
    device = slot_mapping.device
    # Longest context first, so each group pads to a similar length.
    short = sorted(
        (c for c in chunks if c.num_new <= SHORT_QUERY), key=lambda c: c.context_len, reverse=True
    )
    groups: list[BatchedGroup] = []
    current: list[SeqChunk] = []
    for c in short:
        if current and (len(current) + 1) * current[0].context_len > MAX_GATHER_TOKENS:
            groups.append(_batched_group(current, block_size, device))
            current = []
        current.append(c)
    if current:
        groups.append(_batched_group(current, block_size, device))
    prefills = [_prefill_chunk(c, block_size, device) for c in chunks if c.num_new > SHORT_QUERY]
    return AttentionMetadata(slot_mapping, groups, prefills)


def _batched_group(chunks: list[SeqChunk], block_size: int, device: torch.device) -> BatchedGroup:
    q_len = max(c.num_new for c in chunks)
    nb = -(-max(c.context_len for c in chunks) // block_size)
    ctx_len = nb * block_size  # whole blocks: keeps the key length a multiple of 16 for kernels
    tables = [c.block_table[:nb] + [0] * (nb - min(nb, len(c.block_table))) for c in chunks]
    q_index = [[c.flat_start + min(i, c.num_new - 1) for i in range(q_len)] for c in chunks]
    q_pos = [[c.num_cached + min(i, c.num_new - 1) for i in range(q_len)] for c in chunks]
    out_index = [c.flat_start + i for c in chunks for i in range(c.num_new)]
    # Index tensors, never boolean masks: indexing with a bool mask makes PyTorch ask the GPU how
    # many entries are True, a hidden CPU-GPU sync that would happen in every layer.
    all_valid = all(c.num_new == q_len for c in chunks)
    valid_rows = (
        None
        if all_valid
        else to_device(
            [b * q_len + i for b, c in enumerate(chunks) for i in range(c.num_new)], device
        )
    )

    ctx_slots = context_slots(to_device(tables, device), ctx_len, block_size)
    key_pos = torch.arange(ctx_len, device=device)
    query_pos = to_device(q_pos, device)
    # Causal: a query at position p sees keys 0..p. Keys past a short sequence's own context
    # (padding, read from block 0) are masked out by the same rule.
    mask = key_pos[None, None, :] <= query_pos[:, :, None]
    return BatchedGroup(
        q_index=to_device(q_index, device),
        valid_rows=valid_rows,
        out_index=to_device(out_index, device),
        ctx_slots=ctx_slots,
        mask=mask,
    )


def _prefill_chunk(c: SeqChunk, block_size: int, device: torch.device) -> PrefillChunk:
    if c.num_cached == 0:  # a fresh prompt attends only to itself: plain causal attention
        return PrefillChunk(c.flat_start, c.flat_start + c.num_new, None, None)
    nb = -(-c.context_len // block_size)
    ctx_slots = context_slots(to_device(c.block_table[:nb], device), c.context_len, block_size)
    # Query i sits at position num_cached + i, so it sees keys 0..num_cached + i.
    mask = torch.ones(c.num_new, c.context_len, dtype=torch.bool, device=device)
    return PrefillChunk(c.flat_start, c.flat_start + c.num_new, ctx_slots, mask.tril(c.num_cached))


def paged_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    k_cache: Tensor,
    v_cache: Tensor,
    meta: AttentionMetadata,
    scale: float,
) -> Tensor:
    """Write this step's keys and values into the cache, then attend through it.

    q: [T, heads, dim]; k, v: [T, kv_heads, dim]; caches: [slots, kv_heads, dim].
    Returns [T, heads, dim].
    """
    k_cache.index_copy_(0, meta.slot_mapping, k)
    v_cache.index_copy_(0, meta.slot_mapping, v)
    out = torch.empty_like(q)
    heads, kv_heads, dim = q.shape[1], k.shape[1], q.shape[2]
    group = heads // kv_heads  # query heads per KV head (2 for Qwen3)

    for g in meta.groups:
        b, q_len = g.q_index.shape
        # GQA without copying K/V: the `group` query heads that share a KV head are folded into
        # the query dimension, [b, kv_heads, group * Q, dim], so one KV head serves them all.
        qg = q[g.q_index].view(b, q_len, kv_heads, group, dim).permute(0, 2, 3, 1, 4)
        qg = qg.reshape(b, kv_heads, group * q_len, dim)
        kg = k_cache[g.ctx_slots].transpose(1, 2)  # [b, kv_heads, L, dim]
        vg = v_cache[g.ctx_slots].transpose(1, 2)
        if q_len == 1:
            mask = g.mask.unsqueeze(1)  # [b, 1, 1, L], the same row for every folded head
        else:
            mask = g.mask.unsqueeze(1).expand(b, group, q_len, -1).reshape(b, 1, group * q_len, -1)
        og = F.scaled_dot_product_attention(qg, kg, vg, attn_mask=mask, scale=scale)
        og = og.view(b, kv_heads, group, q_len, dim).permute(0, 3, 1, 2, 4)
        og = og.reshape(b * q_len, heads, dim)
        out[g.out_index] = og if g.valid_rows is None else og[g.valid_rows]

    for p in meta.prefills:
        qp = q[p.q_start : p.q_end].transpose(0, 1).unsqueeze(0)  # [1, heads, n, dim]
        if p.ctx_slots is None:
            kp, vp = k[p.q_start : p.q_end], v[p.q_start : p.q_end]
        else:
            kp, vp = k_cache[p.ctx_slots], v_cache[p.ctx_slots]
        # Repeat KV heads up to the query heads so PyTorch can pick a memory-efficient kernel
        # (the score matrix of a 2,000-token prompt would otherwise be materialized).
        kp = kp.repeat_interleave(group, dim=1).transpose(0, 1).unsqueeze(0)
        vp = vp.repeat_interleave(group, dim=1).transpose(0, 1).unsqueeze(0)
        if p.mask is None:
            op = F.scaled_dot_product_attention(qp, kp, vp, is_causal=True, scale=scale)
        else:
            op = F.scaled_dot_product_attention(qp, kp, vp, attn_mask=p.mask, scale=scale)
        out[p.q_start : p.q_end] = op[0].transpose(0, 1)
    return out
