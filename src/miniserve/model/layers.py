"""The building blocks of a Qwen3 layer: RMSNorm, rotary position embeddings and the SwiGLU MLP."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class RMSNorm(nn.Module):
    """x / rms(x) * weight. Computed in float32 for stability, in the same order as Hugging Face."""

    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        x32 = x.to(torch.float32)
        x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x32.to(dtype)


class RotaryEmbedding(nn.Module):
    """Precomputed cos/sin tables for rotary position embeddings (RoPE).

    RoPE encodes a token's position by rotating pairs of dimensions of its query and key by an
    angle proportional to the position. Qwen pairs dimension i with dimension i + head_dim/2 (the
    "rotate half" layout), not neighbours. The table is indexed by each token's *own* position,
    which matters in a flattened batch where positions are not simply 0..T-1.
    """

    cos: Tensor
    sin: Tensor

    def __init__(self, head_dim: int, max_positions: int, theta: float) -> None:
        super().__init__()
        self.head_dim, self.max_positions, self.theta = head_dim, max_positions, theta
        self.register_buffer("cos", torch.empty(0), persistent=False)
        self.register_buffer("sin", torch.empty(0), persistent=False)

    def build(self, device: torch.device) -> None:
        exponents = torch.arange(0, self.head_dim, 2, dtype=torch.int64, device=device).float()
        inv_freq = 1.0 / (self.theta ** (exponents / self.head_dim))  # [head_dim / 2]
        t = torch.arange(self.max_positions, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)  # angle for each (position, pair)
        emb = torch.cat((freqs, freqs), dim=-1)  # [max_positions, head_dim]
        self.cos = emb.cos()
        self.sin = emb.sin()

    def forward(self, positions: Tensor) -> tuple[Tensor, Tensor]:
        return self.cos[positions], self.sin[positions]


def rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
    """q: [T, heads, head_dim], k: [T, kv_heads, head_dim], cos/sin: [T, head_dim]."""
    cos = cos.unsqueeze(1).to(q.dtype)
    sin = sin.unsqueeze(1).to(q.dtype)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


class SwiGLU(nn.Module):
    """down(silu(gate(x)) * up(x)), with gate and up fused into one matmul."""

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_up_proj: nn.Module = nn.Linear(hidden_size, 2 * intermediate_size, bias=False)
        self.down_proj: nn.Module = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        out: Tensor = self.down_proj(F.silu(gate) * up)
        return out
