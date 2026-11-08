# Results

Every benchmark number here comes from a file in `results/`. The exceptions say where they come
from: the correctness checks come from the test runs listed with them, and the start-up memory
profile and the profiler breakdown come from a server log and from `bench/profile_step.py` (its
console output and a 25 MB trace), which are kept out of the repo; rerun them to reproduce. The W1
and W3 tables are pasted from `python bench/summarize.py`, not typed in by hand. Rules: same client
and prompts for every system, warm-up excluded, fixed output lengths, one server on the GPU at a
time, and negative results reported too. P99 is only meaningful for the 500-request MiniServe W1
runs (16, 32 and 64 clients); in the smaller runs it is set by the slowest one or two requests.

## Environment (M0)

From `results/m0_env.json` (`bench/bandwidth.py`), 2026-11-07:

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 3060 Laptop GPU, compute capability 8.6, 6,143 MiB |
| Driver / CUDA (PyTorch) | 617.42 / 13.2 |
| PyTorch / Python | 2.15.0+cu132 / 3.12.3 |
| OS | Windows 11 + WSL2 Ubuntu 24.04 (kernel 6.18.54.1-microsoft-standard-WSL2) |
| VRAM used by other apps before start | 701 MiB |
| VRAM free after CUDA init | 5,118 MiB |
| Copy bandwidth (5 runs) | 312.5, 318.8, 318.8, 319.5, 319.7 GB/s: **median 318.8 GB/s** (95% of the 336 GB/s spec) |
| Decode floor for Qwen3-0.6B (1.19 GB of weights) | **3.73 ms per step**, at most ~268 tokens/s for one sequence |

Start-up memory profile (server log, `gpu_memory_utilization` 0.80, budget 2,048 tokens, 64 seqs):
weights 1,141 MiB; in use at profiling 2,554 MiB; peak activations 488 MiB; **KV pool 1,872 MiB =
1,070 blocks of 16 = 17,120 tokens**; 4 requests of max_model_len 4,096 fit.

Software versions in the venv: transformers 5.22.2, fastapi 0.162.2, uvicorn 0.54.0, openai 3.46.0.

## Correctness

| Check | Result | Source |
|---|---|---|
| CPU test suite (tiny float64 Qwen3, unit, property-based, API) | 60 passed in ~10 s | `pytest -q` |
| Next-token agreement with Hugging Face, real model, BF16, 20 prompts | **20/20**, max abs logit diff 0.453 | `pytest -m gpu -s` |
| Exact greedy match with HF, 48 tokens, 20 prompts (batched) | **11/20**; mean matching prefix 35.3 tokens | `pytest -m gpu -s` |
| KV leak test: 300 requests, random aborts | all blocks free, invariants hold | `pytest -m gpu -s` |

BF16 match rates below 100% are expected: different batch shapes and fused matmuls change
floating-point reduction order and can flip near-ties. Exact equality is tested in float64 on the CPU.

## Decode step time vs batch size (engine only, no HTTP)

`results/step_time_v1.json` (`bench/step_time.py`), 256-token prompts, 32 timed decode steps per
batch size, after the hidden-sync fix (`results/step_time_v0.json` is before it):

| Batch | ms / step | tokens / s | memory-traffic floor (computed) |
|---|---|---|---|
| 1 | 55.29 | 18.1 | 3.83 ms |
| 2 | 43.94 | 45.5 | 3.93 ms |
| 4 | 34.01 | 117.6 | 4.12 ms |
| 8 | 34.25 | 233.6 | 4.50 ms |
| 16 | 35.88 | 446.0 | 5.26 ms |
| 32 | 30.66 | 1,043.5 | 6.78 ms |
| 64 | 37.29 | 1,716.5 (overstated, see below) | 9.82 ms |

Throughput grows 58x from batch 1 to 32 while the step time stays nearly flat: steps are dominated
by fixed per-step overhead, not batch-dependent GPU work. Two caveats. The batch-64 tokens/s is too
high: it divides 64 by the step time, but fewer than 64 sequences were decoding for at least part
of the timed window (each stops after 40 tokens, so the first ones finish before the timed steps
end, and 64 sequences of 257+ tokens need 1,088 KV blocks (computed), more than the server's 1,070).
And batches 1 and 2 are slower per step (55 and 44 ms) than 4 to 64 (31-37 ms); the cause is not
yet known.

## Profile of one decode step (batch 8)

`bench/profile_step.py` with `torch.profiler`: building inputs 1.8 ms; CPU time to queue the forward
pass ~100 ms under the profiler (about 34 ms without it); waiting for the GPU afterwards 1.3 ms; GPU
busy ~20 ms per step; **~1,700 `cudaLaunchKernel` calls per step at ~28 µs each**. The engine is
CPU-launch-bound under WSL2. RMSNorm accounts for about half of the launches.

<!-- from: python bench/summarize.py -->

## W1: 256-token prompts, 128 output tokens, closed loop

### MiniServe

| Clients | Requests | Output tok/s (all runs) | TTFT P50 / P99 ms | TPOT P50 / P99 ms | ITL P99 ms | Peak running | Preemptions | Little's law check | File |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 20 | **15.7** (15.7) | 65 / 176 | 56.9 / 97.2 | 196 | 1 | 0 | 1.0 | `ms_in256_out128_c1_r0.json` |
| 4 | 64 | **63.5** (63.5) | 164 / 8136 | 60.1 / 75.8 | 179 | 4 | 0 | 4.0 | `ms_in256_out128_c4_r0.json` |
| 16 | 500 | **220.6** (220.6) | 320 / 8924 | 65.5 / 101.7 | 211 | 16 | 0 | 15.7 | `ms_in256_out128_c16_r0.json` |
| 32 | 500 | **420.0** (410.4, 420.0, 475.0) | 440 / 8160 | 72.9 / 90.0 | 230 | 32 | 0 | 31.3 | `ms_in256_out128_c32_r2.json` |
| 64 | 500 | **565.9** (565.9) | 1053 / 15186 | 85.1 / 167.0 | 289 | 63 | 154 | 61.9 | `ms_in256_out128_c64_r0.json` |

### MiniServe, fixed KV slots

| Clients | Requests | Output tok/s (all runs) | TTFT P50 / P99 ms | TPOT P50 / P99 ms | ITL P99 ms | Peak running | Preemptions | Little's law check | File |
|---|---|---|---|---|---|---|---|---|---|
| 16 | 100 | **70.9** (70.9) | 20609 / 31420 | 50.1 / 104.6 | 194 | 4 | 0 | 15.2 | `fixed-slots_in256_out128_c16_r0.json` |
| 64 | 100 | **84.7** (84.7) | 69950 / 92910 | 46.5 / 52.9 | 104 | 4 | 0 | 44.1 | `fixed-slots_in256_out128_c64_r0.json` |

### HF baseline

| Clients | Requests | Output tok/s (all runs) | TTFT P50 / P99 ms | TPOT P50 / P99 ms | ITL P99 ms | Peak running | Preemptions | Little's law check | File |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 20 | **24.5** (24.5) | 51 / 77 | 41.3 / 46.2 | 85 | - | - | 1.0 | `hf-baseline_in256_out128_c1_r0.json` |
| 4 | 48 | **24.4** (24.4) | 15805 / 16846 | 41.0 / 45.2 | 83 | - | - | 3.9 | `hf-baseline_in256_out128_c4_r0.json` |
| 16 | 48 | **27.8** (27.8) | 66603 / 73554 | 35.0 / 43.4 | 82 | - | - | 13.5 | `hf-baseline_in256_out128_c16_r0.json` |

## W3: 1,024-token shared prefix + 128 unique tokens, 128 output tokens

### Prefix caching on

| Clients | Requests | Output tok/s (all runs) | TTFT P50 / P99 ms | TPOT P50 / P99 ms | ITL P99 ms | Peak running | Preemptions | Little's law check | File |
|---|---|---|---|---|---|---|---|---|---|
| 16 | 100 | **246.4** (246.4) | 318 / 6539 | 49.3 / 80.6 | 155 | 16 | 0 | 13.6 | `prefix-on_in128_out128_c16_r0.json` |

### Prefix caching off

| Clients | Requests | Output tok/s (all runs) | TTFT P50 / P99 ms | TPOT P50 / P99 ms | ITL P99 ms | Peak running | Preemptions | Little's law check | File |
|---|---|---|---|---|---|---|---|---|---|
| 16 | 100 | **219.1** (219.1) | 627 / 8386 | 56.5 / 79.4 | 208 | 15 | 10 | 15.1 | `prefix-off_in128_out128_c16_r0.json` |

## Conditions and caveats

- **Thermal throttling.** During the runs the laptop GPU sat at 75-84 °C with power capped near
  54 W, and the SM clock varied between 795 and 1,987 MHz (nvidia-smi logs recorded during the runs,
  not committed). Run-to-run
  variance at the same setting was up to about ±15%. Where 3 runs exist, the table shows the median
  and lists all three. For cleaner numbers: plugged in, Windows power mode "Best performance",
  laptop on a hard surface or stand, a cool-down between runs.
- **Discarded runs.** A first sweep used the same prompt seed for every run while the server stayed
  up, so later runs got prefix-cache hits on prompts repeated from earlier runs (e.g. 4,800 cached
  prompt tokens at concurrency 4). Those results were thrown out. The `ms_*` runs use a distinct seed
  per run
  (`seed = 1000 * concurrency + run`).
- **Fewer requests** at low concurrency (20 at 1 client, 64 at 4) and for the fixed-slot, baseline
  and W3 comparisons. Read only their medians: with 100 or fewer requests, P99 is set by the slowest
  one or two.
- **TTFT P99 is inflated by a streaming artifact, not by slow requests.** All values in this item
  are computed from the per-request data in the same files.
  - *What happens.* The server sends an SSE chunk only when the detokenizer releases new text,
    plus a final chunk when the request ends (`generate_text` in `src/miniserve/api/server.py`).
    65 of the 2,984 MiniServe requests (2.2%) produced no text until the end. They got a single
    chunk for all 128 tokens, so their measured TTFT is their whole latency (7.3-13.5 s at 4-32
    clients).
  - *Why.* It follows the prompt, not the load: with the same prompt seed, the same requests went
    silent with paged or fixed KV slots and with prefix caching on or off. Most likely the model
    started with an end-of-sequence token (`<|endoftext|>` or `<|im_end|>`) and kept producing
    tokens that decode to nothing, which `ignore_eos` lets continue (inferred: token ids are not
    recorded). The HF baseline uses the same chunk rule but masks both end-of-sequence ids
    (`min_new_tokens`); none of its 116 requests got a single chunk, including on the same prompts.
  - *Size of the effect.* Without these requests, TTFT P99 is 551 ms instead of 8,924 at
    16 clients, 1,029 ms instead of 8,160 at 32 (median run; 1,029-2,008 ms across the three runs)
    and 11,232 ms instead of 15,186 at 64, where most of the tail is real (inferred from the 154
    preemptions: the KV pool holds about 44 finished requests). These requests also count as 0 ms
    in TPOT, so TPOT P50 reads about 2 ms low (65.5 instead of 67.5 ms at 16 clients, 85.1 instead
    of 87.1 at 64). In W3, the prefix-on P99 of 6,539 ms is two such requests (529 ms without
    them).
  - *Fix.* Send a chunk for every engine step that produced tokens, even with empty text, in both
    servers, then re-measure TTFT, TPOT and ITL.
