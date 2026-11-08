# Learning log

Two or three lines per surprise, with the number. This is interview material.

**2026-11-07, M0.** The RTX 3060 Laptop copies at 318.8 GB/s, 95% of its 336 GB/s spec. Reading
Qwen3-0.6B's 1.19 GB of weights therefore takes 3.73 ms, so one sequence can never decode faster
than ~268 tokens/s on this GPU, however good the code (`results/m0_env.json`).

**2026-11-07, M3.** The checkpoint ships a separate `lm_head.weight` even though the config says
the embeddings are tied. Loading it would waste 311 MB of a 6 GB GPU; the loader skips it.
Qwen3 also has *two* end-of-sequence ids (`<|im_end|>` 151645 and `<|endoftext|>` 151643).

**2026-11-07, M3.** In BF16, our model and Hugging Face's agree on the next token for 20/20
prompts (max logit difference 0.45), yet only 11/20 greedy 48-token generations match exactly;
on average they agree for 35 tokens, then a near-tie flips. That's floating point, not a bug: the
float64 CPU tests match exactly. (tests/gpu)

**2026-11-07, M5/M6.** Batching works: in the engine, 64 sequences decode at 1,717 tok/s versus
18 tok/s for one (95x), and the step time barely depends on the batch (55 ms at 1 sequence,
37 ms at 64). But even one sequence takes ~50 ms per step against a 3.8 ms memory floor
(`results/step_time_v1.json`): the step cost is a fixed overhead, not batch-dependent work.

**2026-11-07, M9 preview (profiling).** Why 50 ms: ~1,700 kernel launches per decode step, and on
WSL2 each `cudaLaunchKernel` costs ~28 us (3-5 us on native Linux). The GPU finishes its work long
before the CPU finishes asking for it: CPU-launch-bound, not memory-bound. RMSNorm alone is
about half the launches (8 tiny kernels, 4 norms per layer). Batching amortizes *launch overhead*
as well as weight reads: the same 1,700 launches serve 1 sequence or 64. CUDA graphs are the fix.

**2026-11-07, M6.** Indexing a tensor with a boolean mask (`x[mask]`) makes PyTorch ask the GPU
how many entries are True: a hidden CPU-GPU sync. It was happening in every layer. Precomputed
index tensors avoid it; so do pinned, non-blocking host-to-device copies.

**2026-11-07, M6.** Start-up profiling found 488 MiB of peak activation memory, mostly the
sampler's top-k/top-p sort over 64 x 151,936 logits (float32 values plus int64 indices). At 112 KiB
per token that is ~4,400 tokens of KV cache we could get back by sampling in smaller row chunks.

**2026-11-07, M7.** Through HTTP, 4 concurrent requests got 68 tok/s versus 117 tok/s inside the
engine alone: the engine thread and the HTTP thread compete for Python's GIL (the engine holds
it while launching kernels). vLLM runs its engine in a separate process for exactly this reason.
