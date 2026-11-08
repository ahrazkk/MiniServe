"""M0: measure the GPU's real memory bandwidth and record the environment.

Decode speed is limited by how fast the GPU can read the weights, so this number sets the
speed limit for everything else. The RTX 3060 Laptop spec says 336 GB/s; a copy kernel shows
what we actually get.

    python bench/bandwidth.py            # prints GB/s and writes results/m0_env.json
"""

from __future__ import annotations

import json
import platform
import subprocess
import time
from pathlib import Path

import torch


def measure_copy_bandwidth(num_bytes: int = 512 * 1024**2, iters: int = 20) -> float:
    """Copy a 512 MiB buffer `iters` times. Each copy reads and writes it, so traffic = 2x size."""
    n = num_bytes // 2  # BF16 = 2 bytes
    x = torch.empty(n, dtype=torch.bfloat16, device="cuda")
    y = torch.empty_like(x)
    for _ in range(3):  # warm-up: first launches include one-time setup costs
        y.copy_(x)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        y.copy_(x)
    torch.cuda.synchronize()  # GPU work is asynchronous: wait for it before stopping the clock
    dt = time.perf_counter() - t0
    return 2 * x.numel() * x.element_size() * iters / dt / 1e9


def nvidia_smi(query: str) -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def main() -> None:
    assert torch.cuda.is_available(), "PyTorch can't see a CUDA GPU"
    idle_used = nvidia_smi("memory.used")  # before we allocate anything big
    torch.cuda.init()
    free, total = torch.cuda.mem_get_info()
    runs = [measure_copy_bandwidth() for _ in range(5)]
    runs.sort()
    median = runs[len(runs) // 2]

    weights_gb = 1.19  # Qwen3-0.6B in BF16: 596M parameters x 2 bytes (computed from config.json)
    env = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "driver": nvidia_smi("driver_version"),
        "vram_total_mib": total // 2**20,
        "vram_free_mib_after_cuda_init": free // 2**20,
        "vram_used_by_others_mib_before_init": idle_used,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "python": platform.python_version(),
        "kernel": platform.release(),
        "copy_bandwidth_gbps_runs": [round(r, 1) for r in runs],
        "copy_bandwidth_gbps_median": round(median, 1),
        "decode_speed_limit_ms_per_step": round(weights_gb / median * 1000, 2),
        "decode_speed_limit_tokens_per_s_single_request": round(median / weights_gb),
    }
    out = Path(__file__).resolve().parent.parent / "results" / "m0_env.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(env, indent=2))
    for k, v in env.items():
        print(f"{k:50s} {v}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
