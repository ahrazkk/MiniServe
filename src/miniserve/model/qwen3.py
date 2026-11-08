"""Qwen3 (dense) in plain PyTorch, written for a serving engine.

Unlike a training-style model, `forward` takes one *flat* batch of tokens from many sequences at
once ([T] token ids plus each token's own position), writes their keys and values into the paged
KV cache, and returns hidden states. Logits are computed separately, and only for the rows that
will be sampled.

One layer:  x -> RMSNorm -> attention (GQA + QK-norm + RoPE) -> +x -> RMSNorm -> SwiGLU -> +
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch.nn.functional as F
from torch import Tensor, nn

from miniserve.attention.backends import AttentionMetadata, paged_attention
from miniserve.model.layers import RMSNorm, RotaryEmbedding, SwiGLU, apply_rope


@dataclass(frozen=True)
class Qwen3Config:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    tie_word_embeddings: bool

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Qwen3Config:
        rope = d.get("rope_parameters") or {}  # newer transformers nest RoPE settings here
        return cls(
            vocab_size=d["vocab_size"],
            hidden_size=d["hidden_size"],
            intermediate_size=d["intermediate_size"],
            num_hidden_layers=d["num_hidden_layers"],
            num_attention_heads=d["num_attention_heads"],
            num_key_value_heads=d["num_key_value_heads"],
            head_dim=d.get("head_dim") or d["hidden_size"] // d["num_attention_heads"],
            rms_norm_eps=d["rms_norm_eps"],
            rope_theta=float(d.get("rope_theta") or rope.get("rope_theta") or 10000.0),
            max_position_embeddings=d["max_position_embeddings"],
            tie_word_embeddings=bool(d.get("tie_word_embeddings", False)),
        )

    @classmethod
    def from_pretrained(cls, model_dir: str | Path) -> Qwen3Config:
        return cls.from_dict(json.loads((Path(model_dir) / "config.json").read_text()))

    def kv_bytes_per_token(self, dtype_bytes: int) -> int:
        """K and V, for every layer: 2 x 28 x 8 x 128 x 2 bytes = 112 KiB for Qwen3-0.6B in BF16."""
        return 2 * self.num_hidden_layers * self.num_key_value_heads * self.head_dim * dtype_bytes


class Qwen3Attention(nn.Module):
    def __init__(self, cfg: Qwen3Config) -> None:
        super().__init__()
        self.num_heads = cfg.num_attention_heads  # 16 query heads ...
        self.num_kv_heads = cfg.num_key_value_heads  # ... share 8 key/value heads (GQA)
        self.head_dim = cfg.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scale = self.head_dim**-0.5
        # q, k and v projections fused into one matmul (one kernel launch instead of three).
        self.qkv_proj: nn.Module = nn.Linear(
            cfg.hidden_size, self.q_size + 2 * self.kv_size, bias=False
        )
        self.o_proj: nn.Module = nn.Linear(self.q_size, cfg.hidden_size, bias=False)
        # QK-norm: Qwen3 normalizes each query and key head before RoPE. Forget it and the
        # model still produces fluent-looking nonsense.
        self.q_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps)

    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        k_cache: Tensor,
        v_cache: Tensor,
        meta: AttentionMetadata,
    ) -> Tensor:
        t = x.shape[0]  # x: [T, hidden], every scheduled token of every sequence
        q, k, v = self.qkv_proj(x).split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = self.q_norm(q.view(t, self.num_heads, self.head_dim))
        k = self.k_norm(k.view(t, self.num_kv_heads, self.head_dim))
        v = v.view(t, self.num_kv_heads, self.head_dim)
        q, k = apply_rope(q, k, cos, sin)
        out = paged_attention(q, k, v, k_cache, v_cache, meta, self.scale)
        result: Tensor = self.o_proj(out.reshape(t, self.q_size))
        return result


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, cfg: Qwen3Config) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.self_attn = Qwen3Attention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = SwiGLU(cfg.hidden_size, cfg.intermediate_size)

    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        k_cache: Tensor,
        v_cache: Tensor,
        meta: AttentionMetadata,
    ) -> Tensor:
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, k_cache, v_cache, meta)
        return x + self.mlp(self.post_attention_layernorm(x))


class Qwen3ForCausalLM(nn.Module):
    def __init__(self, cfg: Qwen3Config, max_positions: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(Qwen3DecoderLayer(cfg) for _ in range(cfg.num_hidden_layers))
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.rotary = RotaryEmbedding(cfg.head_dim, max_positions, cfg.rope_theta)
        # Tied embeddings: the LM head reuses the embedding matrix (saves 311 MB for 0.6B).
        self.lm_head: nn.Linear | None = (
            None
            if cfg.tie_word_embeddings
            else nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        )

    def forward(
        self,
        input_ids: Tensor,
        positions: Tensor,
        kv_caches: list[tuple[Tensor, Tensor]],
        meta: AttentionMetadata,
    ) -> Tensor:
        """input_ids, positions: [T]. Returns final hidden states [T, hidden]."""
        x = self.embed_tokens(input_ids)
        cos, sin = self.rotary(positions)
        for layer, (k_cache, v_cache) in zip(self.layers, kv_caches, strict=True):
            x = layer(x, cos, sin, k_cache, v_cache, meta)
        out: Tensor = self.norm(x)
        return out

    def compute_logits(self, hidden: Tensor) -> Tensor:
        """[n, hidden] -> [n, vocab] float32. Call it only on the rows that will be sampled."""
        weight = self.embed_tokens.weight if self.lm_head is None else self.lm_head.weight
        return F.linear(hidden, weight).float()
