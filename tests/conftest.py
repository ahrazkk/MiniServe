"""Shared fixtures.

The trick that makes CI possible without a GPU: a *tiny random* Qwen3 (2 layers, hidden size 64)
built from transformers' own Qwen3Config. It runs on the CPU in milliseconds, needs no download,
and exercises every code path (model, paging, scheduling, sampling, API). We run it in float64
so that "batched output == single output" can be checked for *exact* equality: in BF16 on a GPU,
different batch shapes change the order of floating-point additions and can flip near-ties.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch

from miniserve.config import EngineConfig
from miniserve.engine.engine import LLMEngine
from miniserve.model.loader import from_hf_model

TINY_CONFIG: dict[str, Any] = {
    "vocab_size": 512,
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "max_position_embeddings": 1024,
    "rms_norm_eps": 1e-6,
    "tie_word_embeddings": True,
}
EOS_ID = 3


def build_hf_model(dtype: torch.dtype = torch.float64, seed: int = 0, **overrides: Any) -> Any:
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(seed)
    model = Qwen3ForCausalLM(Qwen3Config(**{**TINY_CONFIG, **overrides}))
    return model.to(dtype).eval()


@pytest.fixture(scope="session")
def tiny_hf() -> Any:
    return build_hf_model()


@pytest.fixture(scope="session")
def make_engine(tiny_hf: Any) -> Callable[..., LLMEngine]:
    def make(**overrides: Any) -> LLMEngine:
        settings: dict[str, Any] = {
            "model": "tiny-qwen3",
            "device": "cpu",
            "dtype": "float64",
            "block_size": 4,
            "max_model_len": 256,
            "num_gpu_blocks": 512,
            "max_num_seqs": 16,
            "max_num_batched_tokens": 256,
            **overrides,
        }
        cfg = EngineConfig(**settings)
        max_positions = cfg.max_model_len + cfg.num_speculative_tokens + 1
        model = from_hf_model(tiny_hf, torch.device("cpu"), torch.float64, max_positions)
        return LLMEngine(cfg, model=model, eos_token_ids=[EOS_ID])

    return make


@pytest.fixture(scope="session")
def hf_greedy(tiny_hf: Any) -> Callable[[list[int], int], list[int]]:
    """The reference: plain greedy decoding with Hugging Face's model, no cache, no batching."""

    @torch.no_grad()
    def run(prompt: list[int], n: int) -> list[int]:
        ids = list(prompt)
        for _ in range(n):
            ids.append(int(tiny_hf(torch.tensor([ids])).logits[0, -1].argmax()))
        return ids[len(prompt) :]

    return run


@pytest.fixture(scope="session")
def prompts() -> list[list[int]]:
    g = torch.Generator().manual_seed(1)
    return [
        torch.randint(4, 512, (n,), generator=g).tolist() for n in (5, 9, 16, 17, 23, 31, 40, 57)
    ]


@pytest.fixture(scope="session")
def qwen_tokenizer() -> Any:
    from transformers import AutoTokenizer

    local = os.environ.get("MINISERVE_MODEL", "/root/models/Qwen3-0.6B")
    try:
        return AutoTokenizer.from_pretrained(local if Path(local).exists() else "Qwen/Qwen3-0.6B")
    except Exception as exc:  # no local copy and no network
        pytest.skip(f"Qwen3 tokenizer unavailable: {exc}")
