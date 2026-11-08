"""Small shared helpers."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


def to_device(data: Any, device: torch.device, dtype: torch.dtype = torch.long) -> Tensor:
    """Python data -> GPU tensor without stalling the CPU.

    `torch.tensor(data, device="cuda")` copies from pageable memory and waits for the GPU to
    finish *everything* queued so far (a hidden sync). Staging through pinned memory with
    non_blocking=True lets the CPU keep queueing work while the copy happens.
    """
    if device.type != "cuda":
        return torch.tensor(data, dtype=dtype, device=device)
    return torch.tensor(data, dtype=dtype, pin_memory=True).to(device, non_blocking=True)
