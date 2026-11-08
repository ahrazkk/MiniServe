# Design notes (v0.5)

One short note per milestone: what, why, interfaces, how it's tested, what number it should move.
Later milestones append their own section here before any code is written.

## M0. Environment and GPU facts

Repo on the Windows drive (visible to the app and editors), venv and weights on WSL's disk
(`/mnt/c` is slow for thousands of small files and a 1.5 GB weights file). Measured in
`bench/bandwidth.py`: copy bandwidth 318.8 GB/s (95% of the 336 GB/s spec), so reading Qwen3-0.6B's
1.19 GB of weights takes 3.73 ms: the floor for one decode step (`results/m0_env.json`).

## M1. Naive baseline (`baseline/hf_server.py`)

Hugging Face `generate()` behind `/v1/completions`, one request at a time (an `asyncio.Lock` to
queue requests, a `threading.Lock` so the GPU is never shared even when a client disconnects).
Same protocol as MiniServe, so the same client measures both. It still has a KV cache; what it
lacks is batching.

## M2. Load generator (`bench/loadgen.py`, `bench/run_matrix.py`)

asyncio + httpx streaming client, closed loop (N clients) or open loop (Poisson). TTFT, TPOT,
ITL and E2EL defined exactly as `vllm bench serve` defines them, so the two can be cross-checked
later. Prompts are seeded random token ids sent as id lists (exact input length). Every run saves
per-request JSON plus the environment; `/stats` is polled for peak concurrency and server counters.

## M3. Our Qwen3 (`model/`)

Plain PyTorch, flat-batch interface: `forward(input_ids [T], positions [T], kv_caches, meta)`.
q/k/v and gate/up are fused into one matmul each at load time (fewer kernel launches). The tied
LM head reuses the embedding, and the checkpoint's redundant `lm_head.weight` is never read
(311 MB saved). Tested against Hugging Face logits on a tiny float64 model, through all three
attention paths (fresh prefill, prefill on top of cached keys, decode).

## M4 to M6. KV cache, continuous batching, paging (`engine/`, `runner/`, `attention/`)

- **KV pool**: one tensor `[layers, 2, blocks * 16, kv_heads, head_dim]`, sized at start-up by
  profiling a worst-case prefill step and a worst-case decode step
  (`total * gpu_memory_utilization - everything in use - peak activations`).
- **Block manager**: free list, per-sequence block tables, ref counts. Invariants checked after
  every operation in a Hypothesis test (no leaks, no double ownership).
- **Scheduler**: one token budget per step; running sequences first (oldest first), then waiting
  ones (FCFS). Out of memory -> preempt the newest running sequence, free its blocks, recompute
  it later (vLLM V1's policy). `kv_reservation="full"` reserves max_model_len per request at
  admission instead: the pre-PagedAttention baseline for the ablation, no separate code path.
- **Attention**: no paged kernel yet, so each sequence's keys are gathered through its block
  table. Short queries (decode, speculative verify) are padded into one batched SDPA call per
  group; GQA is done by folding the 2 query heads of each KV head into the query dimension, so
  K/V are never duplicated. Prefill chunks are attended one sequence at a time.
- Tested: batched == one at a time, paged == full reservation, tiny pool with preemption ==
  roomy pool, all token for token (float64, CPU).

## M7. OpenAI API (`api/`)

`/v1/completions` and `/v1/chat/completions`, streaming (SSE written by hand: exactly OpenAI's
bytes, no keep-alive comments that some clients choke on) and not. The engine runs in a
background thread; requests go in through a `queue.Queue`, tokens come back through one
`asyncio.Queue` per request via `loop.call_soon_threadsafe`. Incremental detokenizer (emoji-safe),
stop strings with hold-back, abort on disconnect, 503 when the waiting queue is full.

## M8. Prefix caching and chunked prefill

Chained block hashes (`hash((previous block's hash, tokens))`), only full blocks, LRU of freed
cached blocks, never matching the last prompt token. Chunked prefill falls out of the token
budget: a prompt longer than what's left of the budget gets the rest next step.

## M10. Speculative decoding (`spec/`)

n-gram (prompt lookup) drafts, verified in one forward pass of k+1 tokens. Greedy: accept while
the draft equals the target's argmax. Sampling: rejection sampling (accept with p(x), resample
from the residual), tested with a chi-square test (200,000 samples). Rejected drafts' KV is simply
not counted (`num_computed`), so it is overwritten later; no rollback copy needed.

## M11. INT8 weight-only (`quant/int8.py`)

Per-output-row absmax scales, embedding/LM head kept in BF16. Naive dequantize-then-matmul:
expected to save memory but not time.

## Deferred, with the evidence for doing them next

- **M9 CUDA graphs**: `bench/profile_step.py` shows ~1,700 kernel launches per decode step at
  ~28 us each on WSL2: the engine is CPU-launch-bound. Graph capture needs fixed shapes, which
  the gather-based attention doesn't have; it pairs naturally with a paged-attention kernel.
- **Triton paged-attention kernel**: removes the per-step gather (extra memory traffic).
- **Separate engine process**: removes GIL contention between the HTTP thread and the engine.
- **vLLM head-to-head** with `vllm bench serve` as the common client.
