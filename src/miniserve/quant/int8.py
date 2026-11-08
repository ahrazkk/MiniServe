"""Weight-only INT8 quantization: each linear layer's weights stored as int8, one scale per row.

For output row r:  scale_r = max|w_r| / 127,  q_r = round(w_r / scale_r)  in [-127, 127].
The embedding (which is also the LM head, since Qwen3-0.6B ties them) stays in BF16: it is the
most sensitive matrix and quantizing it costs the most quality.

This is the *naive* version: forward rebuilds the full BF16 matrix on every call and then does a
normal matmul. It saves memory (more room for KV cache) but not time; decode is memory-bound, so
the real speed-up needs a fused kernel that reads int8 and dequantizes inside the matmul.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from miniserve.model.qwen3 import Qwen3ForCausalLM


class Int8Linear(nn.Module):
    qweight: Tensor
    scale: Tensor

    def __init__(self, weight: Tensor) -> None:
        super().__init__()
        w = weight.detach().float()
        scale = w.abs().amax(dim=1).clamp(min=1e-8) / 127.0
        self.register_buffer(
            "qweight", torch.round(w / scale[:, None]).clamp(-127, 127).to(torch.int8)
        )
        self.register_buffer("scale", scale.to(weight.dtype))
        self.out_features, self.in_features = weight.shape

    def forward(self, x: Tensor) -> Tensor:
        w = self.qweight.to(x.dtype) * self.scale.unsqueeze(1)  # dequantize the whole matrix
        return F.linear(x, w)


def quantize_int8_(model: Qwen3ForCausalLM) -> int:
    """Swap every linear layer inside the decoder layers for Int8Linear, in place.

    Returns the number of weight bytes saved.
    """
    saved = 0
    for layer in model.layers:
        for parent, name in (
            (layer.self_attn, "qkv_proj"),
            (layer.self_attn, "o_proj"),
            (layer.mlp, "gate_up_proj"),
            (layer.mlp, "down_proj"),
        ):
            linear = getattr(parent, name)
            q = Int8Linear(linear.weight)
            saved += linear.weight.numel() * linear.weight.element_size()
            saved -= q.qweight.numel() + q.scale.numel() * q.scale.element_size()
            setattr(parent, name, q)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return saved
