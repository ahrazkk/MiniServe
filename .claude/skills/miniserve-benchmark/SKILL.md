---
name: miniserve-benchmark
description: Use when measuring MiniServe performance, comparing it with the Hugging Face baseline or vLLM, or writing any number into docs/RESULTS.md, the README, the textbook or a resume bullet.
---

# Measuring MiniServe honestly

## Rules

- Every number comes from a JSON file in `results/` produced by `bench/loadgen.py` (via
  `bench/run_matrix.py`) or another script in `bench/`. Cite the file next to the number.
- State the conditions with the number: GPU (RTX 3060 Laptop, 6 GB), model (Qwen3-0.6B BF16),
  workload (input/output lengths), concurrency, number of requests, date, commit.
- P99 claims need at least 500 requests. Below that, report P50/P90 only or say how many.
- Report negative results too (e.g. "speculation slowed things down at concurrency 16").
- Compare like with like: same client, same prompts (seeded), `ignore_eos` for fixed output
  length, one server on the GPU at a time, warm-up excluded. For vLLM: same `max_model_len` and
  memory fraction, and name the configuration (`--enforce-eager` or default).

## Running

```bash
# from PowerShell: .\scripts\wsl.ps1 <command>
python bench/run_matrix.py --system miniserve --tag NAME --concurrency 1 4 16 32 64 [--runs 3]
python bench/run_matrix.py --system hf-baseline --tag hf-baseline --concurrency 1 4 16 --num-prompts 64
python bench/run_matrix.py --system miniserve --tag NAME --server-args "--no-prefix-caching"
python bench/step_time.py      # engine-only decode step time vs batch size
python bench/profile_step.py   # where a step's time goes (CPU launch vs GPU)
```

Long sweeps must run detached (see CLAUDE.md, `scripts/run_detached.sh`), and nothing else
CPU-heavy may run in WSL meanwhile: the engine is launch-bound on WSL2, so CPU load skews results.

## Checking a run is trustworthy

- `failed` is 0 in the summary.
- `results/logs/<tag>_gpu.csv`: SM clock stays steady and temperature below ~85 C (a falling
  clock means thermal throttling: let the laptop cool and rerun).
- Laptop plugged in, Windows power mode on "Best performance".
- Little's law sanity check: concurrency ~= request throughput x mean E2E latency.

## Recording

Add a row to the right table in `docs/RESULTS.md` (main comparison, ablation, or experiment
tables), with the result file names. Then update the README table and, only from RESULTS.md,
the resume bullet in docs/PLAN.md section 14.
