"""Decode step time vs batch size, measured inside the engine (no HTTP).

This is the curve behind the whole project: a decode step must read all the weights no matter
how many sequences are in the batch, so adding sequences is nearly free until the GPU's compute
(or the KV reads) catch up. For each batch size B we fill the batch with B sequences of
`--context` tokens, let their prefills finish, then time pure decode steps.

    python bench/step_time.py --batch-sizes 1 2 4 8 16 32 64 --context 256
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch

from miniserve.config import EngineConfig
from miniserve.engine.engine import LLMEngine
from miniserve.engine.sequence import SamplingParams

BANDWIDTH_GBPS = 318.8  # measured in M0 (results/m0_env.json)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/root/models/Qwen3-0.6B")
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    p.add_argument("--context", type=int, default=256)
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--out", default="results/step_time.json")
    args = p.parse_args()

    engine = LLMEngine(
        EngineConfig(
            model=args.model, max_num_seqs=max(args.batch_sizes), enable_prefix_caching=False
        )
    )
    engine.warmup()
    weights = sum(t.numel() * t.element_size() for t in engine.model.parameters())
    kv_per_token = engine.model.cfg.kv_bytes_per_token(2)
    rng = random.Random(0)
    rows = []
    for b in args.batch_sizes:
        for i in range(b):
            prompt = [rng.randrange(1000, 100_000) for _ in range(args.context)]
            engine.add_request(
                f"b{b}-{i}",
                prompt,
                SamplingParams(max_tokens=args.steps + 8, temperature=0, ignore_eos=True),
            )
        # Run until every sequence has finished its prefill and is decoding.
        while (
            any(s.num_computed < s.num_prompt_tokens for s in engine.scheduler.running)
            or engine.scheduler.waiting
        ):
            engine.step()
        for _ in range(3):  # warm-up decodes at this batch size
            engine.step()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.steps):
            engine.step()  # each step ends with a GPU -> CPU sync, so this is wall time
        dt = (time.perf_counter() - t0) / args.steps
        for s in list(engine.scheduler.running):
            engine.abort_request(s.request_id)
        ctx = args.context + 8
        bound = (weights + b * ctx * kv_per_token) / (BANDWIDTH_GBPS * 1e9)
        rows.append(
            {
                "batch": b,
                "ms_per_step": dt * 1e3,
                "tokens_per_s": b / dt,
                "memory_bound_ms": bound * 1e3,
            }
        )
        print(
            f"B={b:3d}  {dt * 1e3:7.2f} ms/step  {b / dt:8.1f} tok/s   "
            f"(memory-traffic floor {bound * 1e3:.2f} ms)"
        )
    out = Path(args.out)
    out.parent.mkdir(exist_ok=True)
    meta = {
        "model": args.model,
        "context": args.context,
        "steps": args.steps,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "date": time.strftime("%Y-%m-%d %H:%M"),
    }
    out.write_text(json.dumps({"meta": meta, "rows": rows}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
