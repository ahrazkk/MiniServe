"""Test helpers that need no fixtures (importable because pytest adds tests/ to sys.path)."""

from __future__ import annotations

import torch
from torch import Tensor

from miniserve.attention.backends import SeqChunk, build_attention_metadata
from miniserve.engine.block_manager import cdiv
from miniserve.model.qwen3 import Qwen3ForCausalLM


def new_kv_caches(model: Qwen3ForCausalLM, num_slots: int) -> list[tuple[Tensor, Tensor]]:
    c = model.cfg
    p = next(model.parameters())
    shape = (num_slots, c.num_key_value_heads, c.head_dim)
    return [
        (
            torch.zeros(shape, dtype=p.dtype, device=p.device),
            torch.zeros(shape, dtype=p.dtype, device=p.device),
        )
        for _ in range(c.num_hidden_layers)
    ]


@torch.inference_mode()
def run_chunk(
    model: Qwen3ForCausalLM,
    kv: list[tuple[Tensor, Tensor]],
    ids: list[int],
    start: int,
    block_size: int,
) -> Tensor:
    """Feed tokens `ids` at positions start.. for a single sequence whose blocks are 0, 1, 2, ...

    Returns logits for every fed position. start == 0 is a fresh prefill; start > 0 attends to
    what earlier calls left in `kv` (a chunked prefill, or one decode step when len(ids) == 1).
    """
    device = next(model.parameters()).device
    n = len(ids)
    table = list(range(cdiv(start + n, block_size)))
    slots = torch.tensor(
        [table[p // block_size] * block_size + p % block_size for p in range(start, start + n)],
        device=device,
    )
    meta = build_attention_metadata([SeqChunk(0, n, start, table)], slots, block_size)
    positions = torch.arange(start, start + n, device=device)
    hidden = model(torch.tensor(ids, device=device), positions, kv, meta)
    return model.compute_logits(hidden)


def full_logits(model: Qwen3ForCausalLM, ids: list[int], block_size: int = 16) -> Tensor:
    kv = new_kv_caches(model, cdiv(len(ids), block_size) * block_size)
    return run_chunk(model, kv, ids, 0, block_size)
