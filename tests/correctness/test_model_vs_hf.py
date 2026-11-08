"""M3: our Qwen3 must produce the same logits as Hugging Face's, through every attention path."""

from __future__ import annotations

from typing import Any

import torch
from helpers import full_logits, new_kv_caches, run_chunk

from miniserve.model.loader import from_hf_model


def _ours(tiny_hf: Any) -> Any:
    return from_hf_model(tiny_hf, torch.device("cpu"), torch.float64, 512)


def test_prefill_logits_match_hf(tiny_hf: Any) -> None:
    ids = torch.randint(0, 512, (40,), generator=torch.Generator().manual_seed(0)).tolist()
    ref = tiny_hf(torch.tensor([ids])).logits[0].double()
    got = full_logits(_ours(tiny_hf), ids, block_size=4)
    torch.testing.assert_close(got.double(), ref, atol=1e-6, rtol=1e-5)


def test_chunked_prefill_and_decode_match_hf(tiny_hf: Any) -> None:
    """Fresh prefill -> prefill continuing from the cache -> one-token decodes: same logits."""
    ids = torch.randint(0, 512, (50,), generator=torch.Generator().manual_seed(1)).tolist()
    ref = tiny_hf(torch.tensor([ids])).logits[0].double()
    model = _ours(tiny_hf)
    kv = new_kv_caches(model, 64)
    parts = [
        run_chunk(model, kv, ids[:20], 0, 4),  # fresh prompt
        run_chunk(model, kv, ids[20:43], 20, 4),  # chunk that attends to cached keys
        *[run_chunk(model, kv, [ids[p]], p, 4) for p in range(43, 50)],  # decodes
    ]
    torch.testing.assert_close(torch.cat(parts).double(), ref, atol=1e-6, rtol=1e-5)


def test_untied_lm_head_is_loaded() -> None:
    from conftest import build_hf_model

    hf = build_hf_model(tie_word_embeddings=False, seed=3)
    ids = list(range(5, 25))
    ref = hf(torch.tensor([ids])).logits[0].double()
    torch.testing.assert_close(full_logits(_ours(hf), ids, 4).double(), ref, atol=1e-6, rtol=1e-5)
