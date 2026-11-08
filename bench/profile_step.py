"""Profile decode steps: where does the time go, CPU or GPU? (M9: profile before optimizing.)

    python bench/profile_step.py --batch 8
    # then open results/profile_b8.trace.json in https://ui.perfetto.dev and look for GPU gaps

Prints wall time per step split into: building inputs (prepare), queueing the forward pass
(CPU launch time), and waiting for the GPU (sync), plus the top operators by CPU time.
"""

from __future__ import annotations

import argparse
import random
import time

import torch
from torch.profiler import ProfilerActivity, profile

from miniserve.config import EngineConfig
from miniserve.engine.engine import LLMEngine
from miniserve.engine.sequence import SamplingParams


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/root/models/Qwen3-0.6B")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--context", type=int, default=256)
    args = p.parse_args()

    engine = LLMEngine(EngineConfig(model=args.model, enable_prefix_caching=False))
    engine.warmup()
    rng = random.Random(0)
    for i in range(args.batch):
        prompt = [rng.randrange(1000, 100_000) for _ in range(args.context)]
        engine.add_request(
            str(i), prompt, SamplingParams(max_tokens=200, temperature=0, ignore_eos=True)
        )
    for _ in range(5):  # prefill, then a few decodes
        engine.step()

    runner = engine.runner
    sched = engine.scheduler
    n = 20
    t_prep = t_launch = t_sync = 0.0
    for _ in range(n):
        plan = sched.schedule()
        t0 = time.perf_counter()
        inp = runner.prepare(plan.scheduled)
        t1 = time.perf_counter()
        with torch.inference_mode():
            hidden = runner.model(inp.input_ids, inp.positions, runner.kv_caches, inp.attn)
            logits = runner.model.compute_logits(hidden[inp.logits_rows])
            tokens = logits.argmax(-1)
        t2 = time.perf_counter()
        out = tokens.tolist()  # waits for the GPU
        t3 = time.perf_counter()
        sched.update(plan, [[t] for t in out])
        t_prep, t_launch, t_sync = t_prep + t1 - t0, t_launch + t2 - t1, t_sync + t3 - t2
    print(
        f"batch {args.batch}: prepare {t_prep / n * 1e3:.2f} ms | queue forward (CPU) "
        f"{t_launch / n * 1e3:.2f} ms | wait for GPU {t_sync / n * 1e3:.2f} ms  (per step)"
    )

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(5):
            engine.step()
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=18))
    path = f"results/profile_b{args.batch}.trace.json"
    prof.export_chrome_trace(path)
    print(f"chrome trace: {path}")


if __name__ == "__main__":
    main()
