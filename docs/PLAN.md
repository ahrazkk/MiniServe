# MiniServe: an LLM inference server built from scratch

**Research brief, learning guide and build plan. Hand this whole file to a new chat.**

| | |
|---|---|
| Owner | Ahraz Kibria (hireahraz.dev, github.com/ahrazkk) |
| Written | 2026-11-07 |
| Status | Not started |
| Hardware | RTX 3060 Laptop GPU (6 GB), Windows 11 + WSL2 Ubuntu 24.04 |
| Main model | Qwen3-0.6B (BF16), with Qwen3-1.7B for some experiments |
| Yardstick | vLLM (v0.33.0 was the latest stable release on 2026-11-01) |
| Time budget | About 14 to 15 focused weekends for v1.0 (roughly 7 weeks full-time), plus optional level-ups. M0 to M7 alone is already a complete, resume-worthy project. |

---

## Contents

0. [How to use this file](#0-how-to-use-this-file)
1. [Instructions for the assistant](#1-instructions-for-the-assistant)
2. [About me and my machine](#2-about-me-and-my-machine)
3. [The project in one page](#3-the-project-in-one-page)
4. [Goals, non-goals and definition of done](#4-goals-non-goals-and-definition-of-done)
5. [Everything to know: the concepts from first principles](#5-everything-to-know-the-concepts-from-first-principles)
6. [Prior art and how to use it](#6-prior-art-and-how-to-use-it)
7. [MiniServe architecture](#7-miniserve-architecture)
8. [Repository layout and tooling](#8-repository-layout-and-tooling)
9. [Milestones M0 to M12](#9-milestones-m0-to-m12)
10. [Benchmark protocol](#10-benchmark-protocol)
11. [Testing strategy](#11-testing-strategy)
12. [Risks and pitfalls](#12-risks-and-pitfalls)
13. [Honesty rules for results and the resume](#13-honesty-rules-for-results-and-the-resume)
14. [Resume bullets, write-up and the 60-second story](#14-resume-bullets-write-up-and-the-60-second-story)
15. [Interview questions with answers](#15-interview-questions-with-answers)
16. [Level-ups after v1.0](#16-level-ups-after-v10)
17. [Reading list, in order](#17-reading-list-in-order)
18. [Glossary](#18-glossary)
19. [Timeline and checklist](#19-timeline-and-checklist)
20. [Starter prompt for the new chat](#20-starter-prompt-for-the-new-chat)
21. [Facts checked for this brief](#21-facts-checked-for-this-brief)

---

## 0. How to use this file

**For me (Ahraz):** start a new chat, attach this file, and send the starter prompt in [section 20](#20-starter-prompt-for-the-new-chat). Keep this file in the repo as `docs/PLAN.md` and update the status line as milestones finish.

**For the assistant:** this file is the project's source of truth. Read sections 1 to 4 before anything else, then work one milestone at a time from section 9. When something here turns out to be wrong or outdated (library APIs change fast in this field), say so, check the current docs, and propose an edit to this file.

---

## 1. Instructions for the assistant

### Your role

You are my mentor and pair programmer for this project. The goal is twofold: build a working, measured inference server, **and** make sure I understand every part of it well enough to explain and modify it live in a FAANG-level interview.

### How to teach me

- I'm strong in Python, TypeScript, REST backends (FastAPI, Django), Docker, CI and PyTorch at the "train a model" level. I'm **new to GPU internals, CUDA, Triton, attention kernels, inference-engine design and GPU setup under WSL2**. Explain those from first principles, with a small example, the exact command, and the output I should expect.
- Work in this loop for every milestone:
  1. **Design note first** (half a page in `docs/design/M#.md`): what we're building, why, the interfaces, how we'll test it, and what number we'll measure.
  2. **Code**, in small pieces I can read. Explain every non-obvious line.
  3. **Tests** before calling it done.
  4. **Measure** and write the result in `docs/RESULTS.md`, with the settings.
  5. **Learning log**: two or three lines in `docs/LEARNING_LOG.md` about what surprised me. This becomes interview material.
- Give me a recommendation, not a menu of options. If there's a real trade-off, state it in one line and pick one.
- When I ask "why", answer with the mechanism **and a number**: bytes, milliseconds, FLOPs, tokens.
- Tell me what you verified (ran it, read the docs today) and what you're assuming.
- Quiz me briefly at the end of each milestone with 3 interview-style questions from [section 15](#15-interview-questions-with-answers) that match what we just built.

### Rules (non-negotiable)

1. **Never invent a benchmark number.** Every number in the README, the resume or a post must come from a run saved in `results/`, with hardware, model, dtype, settings, commit hash and date. Estimates are labelled "estimate".
2. **No resume claim before it's measured.** The bullet templates in section 14 have placeholders; fill them only from `docs/RESULTS.md`.
3. **Ask me first before:** spending money; starting any paid cloud GPU (and always set a budget alert and an automatic shutdown first); buying hardware; pushing to a public GitHub repo; publishing a blog post or LinkedIn post.
4. **Licences:** check a dependency's licence before adding it. Reading vLLM (Apache-2.0), SGLang (Apache-2.0) and nano-vllm (MIT) to learn is fine. If we copy or closely adapt any code, keep its licence notice and say so in the README's credits.
5. **Write it ourselves.** Use reference projects to understand ideas, then write our own code. The point of the project is that I can explain every line.
6. **Finish before extending.** Don't start a level-up until the current milestone's exit criteria pass.
7. **Don't touch my other projects** (ARIS lives in the same WSL install).

---

## 2. About me and my machine

### Me

- B.Eng. Computer Engineering (Software Engineering specialization), Toronto Metropolitan University, May 2026. Toronto. Job hunting for software roles, including ML-systems and inference roles.
- Experience: Geotab (Python/BigQuery pipelines, C#/.NET, GCP Cloud Run, a GenAI data-agent prototype); CAPREIT (networking, Cisco Meraki).
- Relevant projects: Project Aegis (PyTorch GNNs, Django REST, Docker, Cloud Run), TransiNet (FastAPI, RabbitMQ, gRPC, WebSockets), ARIS (ROS 2, PX4, CI with ruff, strict mypy and pytest), Myelin (FastAPI, React, GitHub Actions releases).
- Languages: Python, TypeScript, C#, Java, SQL, C (embedded/coursework level). **No production C++.**

### The machine

| Item | Value | Why it matters here |
|---|---|---|
| GPU | NVIDIA RTX 3060 Laptop GPU, **6 GB GDDR6**, 192-bit bus, 336 GB/s spec bandwidth, 3,840 CUDA cores, Ampere (compute capability 8.6), 60 to 115 W depending on the laptop | 6 GB is the main constraint. 336 GB/s sets the speed limit for decoding. Ampere supports BF16 and FlashAttention-2, but **not FP8** tensor cores. |
| CPU | Intel i7-11370H (4 cores, 8 threads) | Python overhead per step matters for small models. |
| RAM | 16 GB | WSL2 gets part of it; compiling CUDA extensions can run out. |
| Disk | One 477 GB NVMe, about 98 GB free (checked 2026-10-26; recheck) | Models: Qwen3-0.6B about 1.5 GB, Qwen3-1.7B about 4 GB. |
| OS | Windows 11 + **WSL2 Ubuntu 24.04** (already set up for ARIS; Linux commands run as `root`) | vLLM needs Linux, and WSL counts. |
| Python | `uv` with Python 3.12 | vLLM 0.32+ dropped Python 3.10; 3.12 is safe. |
| Other | Git, Node 22, Claude Code | |

Open WSL from PowerShell with:

```powershell
wsl -d Ubuntu-24.04 -u root
```

Keep the repo and model files **inside the WSL file system** (for example `~/miniserve`, `~/models`), not under `/mnt/c/...`. Access to the Windows drive from WSL is much slower, and loading model weights from it hurts start-up time.

---

## 3. The project in one page

**MiniServe** is a small LLM inference server written in Python and PyTorch. It serves an open model (Qwen3-0.6B) on a 6 GB laptop GPU and implements the core ideas of production engines like vLLM and SGLang:

- **continuous batching** (requests join and leave the running batch at every step);
- a **paged KV cache** (memory in fixed-size blocks, like an operating system's virtual memory);
- **prefix caching** (reuse work for prompts that start the same way);
- **chunked prefill** (long prompts don't freeze everyone else's output);
- **streaming** over an **OpenAI-compatible API**;
- **CUDA graphs** to cut CPU overhead;
- **speculative decoding** (guess several tokens, verify them in one pass);
- **quantization** (smaller weights, measured quality cost).

Every feature is measured twice: against a **naive Hugging Face baseline** (to show what each idea buys), and against **vLLM on the same GPU with the same benchmark client** (to show how close a from-scratch version gets to the industry standard).

### Why this project

- It fills the biggest gap in my resume for ML-systems roles: I can't currently claim LLM inference work.
- It's what inference teams at Google, Meta, Cerebras, AMD, Qualcomm and others work on every day.
- Few new grads have built one, and the result is a number that anyone can check.
- It goes deep enough for system-design and ML-systems interviews: memory math, scheduling, latency percentiles, GPU performance.
- Later it becomes the model server for my TriageBench agent project (it keeps the OpenAI-compatible API for that reason).

### What the finished repo shows a reviewer in two minutes

1. One architecture diagram.
2. A results table: MiniServe vs Hugging Face baseline vs vLLM, same GPU, same client.
3. An ablation table: how much each technique added.
4. One command to reproduce the numbers.
5. Tests and CI, including correctness checks against Hugging Face.
6. An honest "limitations and what I'd do next" section.

---

## 4. Goals, non-goals and definition of done

### Goals (v1.0)

| # | Goal |
|---|---|
| G1 | Serve Qwen3-0.6B in BF16 on the RTX 3060 through `/v1/completions` and `/v1/chat/completions`, streaming and non-streaming, compatible with the official `openai` Python SDK. |
| G2 | My own model code for Qwen3 in plain PyTorch (Hugging Face is used only for the tokenizer, the baseline and reference outputs). |
| G3 | Continuous batching with a token-budget scheduler and recompute-based preemption. |
| G4 | Paged KV cache with a block manager, plus prefix caching and chunked prefill. |
| G5 | CUDA graphs for decode steps. |
| G6 | Speculative decoding (n-gram first; draft model if memory allows), with a correctness proof by test. |
| G7 | INT8 weight-only quantization, with memory, speed and quality measured. |
| G8 | A reproducible benchmark matrix against the baseline and vLLM, using `vllm bench serve` as the common client. |
| G9 | Tests and CI: unit, property-based, correctness against Hugging Face on a tiny model (runs on CPU in CI), and API contract tests. |
| G10 | A public README and one technical write-up (published only after I approve). |

### Non-goals

- **Multi-GPU** (tensor or pipeline parallelism). I have one GPU. I'll learn the ideas for interviews ([section 5.17](#517-what-big-systems-add-interview-awareness)) but not build them.
- **Training or fine-tuning.**
- **Many model families.** Qwen3 dense models only. Llama 3.2 1B is an optional second model.
- **Beating vLLM.** The aim is to understand and quantify the gap, and explain where it comes from.
- **Production hardening** beyond the basics (no auth system, no multi-tenant quotas).
- **Native Windows.** WSL2 only.

### Definition of done (v1.0)

- [ ] All of G1 to G9 work, and their tests pass in CI (CPU) and locally (GPU).
- [ ] `docs/RESULTS.md` holds every number, each with its settings.
- [ ] The README has the diagram, results table, ablation table, reproduce commands, limitations and credits.
- [ ] I can whiteboard the architecture and answer every question in section 15 without notes.
- [ ] The resume bullet is filled from real numbers.

---
## 5. Everything to know: the concepts from first principles

Read this section once before writing any code, then come back to the relevant part at each milestone. Numbers marked **(computed)** were calculated from the real Qwen3 configs; numbers marked **(estimate)** are back-of-envelope and must be measured on my machine.

### 5.1 How an LLM generates text

1. A **tokenizer** turns text into integer **token IDs**. Qwen3's vocabulary has **151,936** tokens.
2. The model takes the token IDs and outputs **logits**: one score per vocabulary entry, for the next position.
3. A **sampler** turns logits into one chosen token:
   - **greedy**: take the highest score (temperature 0);
   - **temperature**: divide logits by T before softmax (T < 1 sharper, T > 1 flatter);
   - **top-k**: keep only the k highest-scoring tokens;
   - **top-p (nucleus)**: keep the smallest set of tokens whose probabilities add up to p.
4. Append the chosen token and repeat, until an end-of-sequence token, a stop string, or `max_tokens`.

This loop is **autoregressive**: each new token depends on all previous ones, so tokens for one request are produced one at a time. That single fact drives everything else in this project.

```python
# The whole idea in 8 lines (no cache, no batching: the naive version)
ids = tokenizer.encode(prompt)
for _ in range(max_tokens):
    logits = model(torch.tensor([ids]))[0, -1]     # scores for the next token only
    next_id = sample(logits, temperature, top_p)   # greedy / temperature / top-p
    if next_id == eos_id:
        break
    ids.append(next_id)
text = tokenizer.decode(ids[len_prompt:])
```

### 5.2 Inside Qwen3 (the model I'll implement)

Qwen3 is a standard **decoder-only transformer**. Verified config values:

| | Qwen3-0.6B | Qwen3-1.7B |
|---|---|---|
| Layers | 28 | 28 |
| Hidden size | 1,024 | 2,048 |
| MLP (intermediate) size | 3,072 | 6,144 |
| Query heads / KV heads | 16 / 8 | 16 / 8 |
| Head dimension | 128 | 128 |
| Vocabulary | 151,936 | 151,936 |
| Tied embeddings (LM head reuses the embedding matrix) | Yes | Yes |
| RoPE base (theta) | 1,000,000 | (check config) |
| Parameters **(computed)** | 596M total, 440M without embeddings | 1.72B total, 1.41B without embeddings |
| Weights in BF16 **(computed)** | **1.19 GB** | **3.44 GB** |

One layer, top to bottom:

```
x ─► RMSNorm ─► Attention (GQA + QK-Norm + RoPE) ─► + x  ─► RMSNorm ─► SwiGLU MLP ─► + ─► next layer
     └──────────────── residual connection ────────┘       └──── residual connection ──┘
```

- **RMSNorm**: `x * rsqrt(mean(x²) + eps) * weight`, computed in FP32 for stability.
- **Attention with GQA (grouped-query attention):** 16 query heads share 8 key/value heads (2 query heads per KV head). This halves the KV cache compared with 16 KV heads.
- **QK-Norm:** Qwen3 applies an RMSNorm to each query head and each key head **before** RoPE. (Qwen3 removed the QKV bias that Qwen2 had.) Forgetting QK-Norm gives fluent-looking garbage.
- **RoPE (rotary position embeddings):** rotates each query and key vector by an angle that depends on the token's position. Qwen uses the "rotate half" layout (the first half and second half of the head dimension are paired), not interleaved pairs. In a batch, every token needs **its own position**, which is why the engine passes a `positions` tensor.
- **SwiGLU MLP:** `down(silu(gate(x)) * up(x))`.
- **LM head:** a final RMSNorm, then a matmul with the (tied) embedding matrix gives logits.

Attention for a flat batch of tokens, as I'll write it:

```python
class Qwen3Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.nq, self.nkv, self.hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        self.q_proj = nn.Linear(cfg.hidden_size, self.nq * self.hd, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, self.nkv * self.hd, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, self.nkv * self.hd, bias=False)
        self.o_proj = nn.Linear(self.nq * self.hd, cfg.hidden_size, bias=False)
        self.q_norm = RMSNorm(self.hd, cfg.rms_norm_eps)   # per-head norm (QK-Norm)
        self.k_norm = RMSNorm(self.hd, cfg.rms_norm_eps)

    def forward(self, x, cos, sin, attn_meta):            # x: [T, hidden], all tokens of all sequences
        T = x.shape[0]
        q = self.q_norm(self.q_proj(x).view(T, self.nq, self.hd))
        k = self.k_norm(self.k_proj(x).view(T, self.nkv, self.hd))
        v = self.v_proj(x).view(T, self.nkv, self.hd)
        q, k = apply_rope(q, k, cos, sin)                  # cos/sin already gathered per token position
        out = attention_backend(q, k, v, attn_meta)        # writes k, v into the KV cache, then attends
        return self.o_proj(out.reshape(T, self.nq * self.hd))
```

Later optimization: fuse `q_proj`, `k_proj`, `v_proj` into one `qkv_proj` matmul and `gate`/`up` into one `gate_up` matmul. Fewer, bigger matmuls means fewer kernel launches (vLLM does this).

### 5.3 Prefill vs decode, and why the GPU's memory speed is the limit

Serving one request has two phases:

| | Prefill | Decode |
|---|---|---|
| What | Process the whole prompt in one forward pass | Produce one new token per step |
| Tokens per forward | Hundreds or thousands | 1 per sequence |
| Bottleneck | **Compute** (big matmuls) | **Memory bandwidth** (reading all weights for very little math) |
| Latency metric | **TTFT** (time to first token) | **TPOT / ITL** (time per output token) |

**Why decode is memory-bound.** For every decode step the GPU must read **every weight** from memory. With Qwen3-0.6B that's 1.19 GB per step. At 336 GB/s:

> 1.19 GB / 336 GB/s ≈ **3.5 ms per step**, so at most about **280 tokens/s for a single request**, even with infinitely fast math. **(computed from the spec; real bandwidth is lower)**

**Arithmetic intensity** is FLOPs done per byte read. A decode matmul with batch size B reads each 2-byte weight once and does 2·B FLOPs with it, so its intensity is about **B FLOPs per byte**. A consumer Ampere GPU needs very roughly 60 to 150 FLOPs per byte before compute becomes the limit **(estimate)**. So until the batch reaches dozens of sequences, adding sequences to a decode step is **almost free**. That's the whole reason batching works.

**Attention is different.** Each sequence reads **its own** KV cache, so there's no sharing across the batch. Attention's intensity is about the number of query heads per KV head (2 for Qwen3), which is memory-bound at every batch size. Long contexts make attention the dominant cost.

**A speed-limit model for one decode step (memory traffic only):**

> step time ≈ (weight bytes + total KV bytes read) / bandwidth

| Batch × context | Bytes read per step | Step time | Upper bound on throughput **(computed, optimistic)** |
|---|---|---|---|
| 1 × 512 | 1.25 GB | 3.7 ms | 269 tokens/s |
| 8 × 512 | 1.66 GB | 4.9 ms | 1,620 tokens/s |
| 32 × 512 | 3.07 GB | 9.1 ms | 3,503 tokens/s |

Real numbers will be well below these (kernel efficiency, Python overhead, compute in prefill), but the **shape** is what to explain in interviews: throughput rises with batch size while each step gets slower, until KV memory or compute runs out. I should measure my real bandwidth in M0 and redo this table with it.

### 5.4 The KV cache

**Problem:** without a cache, generating token n means recomputing keys and values for all n-1 previous tokens, so the work grows quadratically.

**Fix:** store each layer's keys and values for every token once computed, and only compute K and V for the new token each step.

**Size per token** (Qwen3-0.6B and 1.7B have the same attention shape, so the same size):

> 2 (K and V) × 28 layers × 8 KV heads × 128 dims × 2 bytes (BF16) = **114,688 bytes = 112 KiB per token (computed)**

| KV memory | Tokens it holds **(computed)** | 16-token blocks |
|---|---|---|
| 1 GiB | 9,362 | 585 |
| 2 GiB | 18,724 | 1,170 |
| 3 GiB | 28,086 | 1,755 |

- One request at 4,096 tokens uses **448 MiB** of KV cache.
- Reserving space naively for 32 requests at a 2,048-token maximum would need **7 GiB**, more than the whole GPU.

That's the key insight: **on a small GPU, KV-cache memory, not compute, decides how many requests you can serve at once.**

### 5.5 The 6 GB memory budget

| Item | Size | Notes |
|---|---|---|
| CUDA context + PyTorch runtime | ~0.3 to 0.6 GB **(estimate)** | Measure with `nvidia-smi` after `torch.cuda.init()`. |
| Windows desktop on the same GPU | 0 to ~0.5 GB **(estimate)** | Depends on whether the laptop drives the display from the NVIDIA GPU. Measure. |
| Qwen3-0.6B weights (BF16) | 1.19 GB **(computed)** | |
| Activations for one forward | Depends on tokens per step | Measured by profiling at start-up. |
| Logits | See below | Easy to waste. |
| **KV cache pool** | **Everything left**, about 2.5 to 3.5 GB **(estimate)** | ~22,000 to 31,000 tokens. |

**A classic mistake:** during prefill, computing logits for **every** prompt token. A 2,048-token prefill would produce 2,048 × 151,936 logits = **622 MB in BF16 (computed)**, for nothing. Only the **last** token of each sequence needs logits. Gather those hidden states first, then apply the LM head. (64 decoding sequences × 151,936 × 4 bytes in FP32 = 39 MB, fine.)

**How real engines size the KV pool (do the same):** at start-up, run one dummy forward pass at the largest batch size, record peak memory, then give the KV pool `total_gpu_memory × gpu_memory_utilization − peak_usage`.

### 5.6 Batching: static, dynamic, continuous

- **Static batching** (what `model.generate()` on a padded batch does): wait for a batch, pad everyone to the longest prompt, run until **every** sequence finishes. Short answers sit idle waiting for the longest one, and new requests wait for the whole batch.
- **Dynamic batching:** collect requests for a few milliseconds, then run them as a static batch. Helps arrival, not the finishing problem.
- **Continuous batching** (also called iteration-level scheduling, from the **Orca** paper, OSDI 2022): the scheduler decides batch membership **at every decoding step**. A sequence that finishes leaves immediately; a new request joins at the next step. GPU slots never sit idle waiting for the slowest request.

How it's implemented (the Orca idea, also used by vLLM's V1 engine): **flatten** all tokens scheduled this step, from all sequences, into one long "super-sequence". Every non-attention operation (the linear layers, norms, MLP) runs on the whole flat batch at once. Only attention needs to know where each sequence starts and ends, which it gets from metadata (sequence lengths and offsets).

```
Step N: seq A decoding (1 token), seq B decoding (1 token), seq C prefilling (5 tokens)
input_ids  = [a7, b3, c0, c1, c2, c3, c4]          # 7 tokens, one flat tensor
positions  = [ 7,  3,  0,  1,  2,  3,  4]          # each token's position in its own sequence
seq bounds = A:[0,1)  B:[1,2)  C:[2,7)             # attention uses these
logits for = token index 0 (A), 1 (B), 6 (C)       # only the last token of each sequence
```

### 5.7 Paged KV cache (PagedAttention)

The **vLLM paper** ("Efficient Memory Management for Large Language Model Serving with PagedAttention", SOSP 2023) showed that earlier systems wasted most of their KV memory:

- **Over-reservation:** reserving space for the maximum length up front, though most outputs are short.
- **Fragmentation:** gaps between variable-sized allocations.

**The fix borrows from operating-system virtual memory:**

- Split KV memory into fixed-size **blocks** (vLLM's default is **16 tokens** per block).
- Each sequence has a **block table**: logical block i maps to physical block j, like a page table.
- Blocks are allocated **on demand** as the sequence grows, from a **free list**, and returned when it finishes.
- At most one partly-filled block per sequence is wasted.

**Slot mapping:** where does token position `p` of a sequence go?

```python
block_size = 16
physical_block = block_table[p // block_size]
slot = physical_block * block_size + p % block_size   # index into the flattened KV pool
```

The attention kernel reads K and V through the block table instead of from one contiguous array. Blocks can also be **shared** between sequences with reference counts (prefix caching, parallel sampling), using copy-on-write if a shared block would be modified.

The paper reports **2 to 4× higher throughput** than prior systems at the same latency. On my GPU the effect shows up directly as "how many concurrent requests fit before running out of memory".

**Block size trade-off:**

| | Smaller blocks (e.g., 16) | Larger blocks (e.g., 256) |
|---|---|---|
| Wasted memory per sequence | Less | More |
| Prefix-cache granularity | Finer (more reuse) | Coarser |
| Block-table size, kernel efficiency | Larger tables, more indirection | Smaller tables, more efficient reads |

FlashAttention-2's paged mode requires a page size that's a multiple of 256, so if I use its kernel, I use 256-token blocks.

### 5.8 Prefix caching

Many requests start with the same tokens: a system prompt, few-shot examples, or earlier chat turns. Their KV values are identical, so compute them once and share.

How vLLM does it (prefix caching is on by default there):

1. Split the prompt into full blocks.
2. Give each block a hash that includes **the previous block's hash** and the block's tokens. The same token block at a different position gets a different hash, because the KV values depend on everything before them.
3. Look up each hash in a table. Reuse every matching physical block (increase its reference count) and skip computing those tokens in prefill.
4. When a block's reference count drops to 0, keep it in an **LRU** list instead of freeing its contents. Evict it only when memory is needed.

Only full blocks are cached. The last partial block is always computed. **SGLang's RadixAttention** does the same thing with a radix tree over token sequences instead of a hash table.

### 5.9 Chunked prefill

**Problem:** a 2,000-token prompt arrives while 30 sequences are decoding. If its whole prefill runs in one step, that step takes much longer, so all 30 users see a sudden stall in their token stream (a spike in inter-token latency).

**Fix (Sarathi-Serve, OSDI 2024):** give every step a **token budget** (for example 512 tokens):

1. Schedule all decodes first (1 token each).
2. Fill the remaining budget with a **chunk** of a pending prefill.
3. Continue the rest of that prefill in later steps.

**Trade-off:** smaller chunks give smoother ITL for everyone, but a slightly slower TTFT for the long prompt and a little less prefill efficiency. I'll measure both sides of it.

### 5.10 Preemption

When the KV pool is full and a running sequence needs a new block:

- **Recompute** (what vLLM's V1 engine does): pick a victim (the most recently started sequence), free all its blocks, put it back at the front of the waiting queue. Later its prompt plus generated-so-far tokens are prefilled again. That costs compute, but it's simple and safe.
- **Swap** (vLLM's older V0 engine): copy the victim's blocks to CPU memory and copy them back later. That needs PCIe bandwidth and more bookkeeping.

I'll implement recompute.

### 5.11 Speculative decoding

**Insight:** decode is memory-bound, so a forward pass over **k+1 tokens costs about the same** as over 1 token. If something can **guess** the next k tokens cheaply, the big model can **check all of them in one pass**.

**Algorithm (Leviathan et al. 2023; Chen et al. 2023):**

1. A **draft** proposes k tokens x₁…x_k, with draft probabilities q(xᵢ).
2. The **target** model runs one forward over the k proposed tokens and gets its own probabilities p(·) at every position.
3. For i = 1…k: accept xᵢ with probability `min(1, p(xᵢ) / q(xᵢ))`.
   - On the first rejection, sample a replacement token from `normalize(max(0, p − q))` and stop.
4. If all k are accepted, sample one **bonus** token from the target's distribution at position k+1.
5. With greedy decoding this reduces to "accept while the draft token equals the target's argmax".

**Why it's exact:** this rejection-sampling rule makes the output distribution **identical** to sampling from the target alone. It's provable, and I'll test it statistically.

**Expected tokens per target forward**, with per-token acceptance rate α and k drafted tokens: `(1 − α^(k+1)) / (1 − α)` **(computed)**.

| α \ k | 2 | 4 | 6 |
|---|---|---|---|
| 0.6 | 1.96 | 2.31 | 2.43 |
| 0.7 | 2.19 | 2.77 | 3.06 |
| 0.8 | 2.44 | 3.36 | 3.95 |

The real speed-up is lower, because drafting costs time and the verify step is a bit slower than one decode.

**Where drafts come from:**

| Method | How | Cost | Good for |
|---|---|---|---|
| **n-gram / prompt lookup** | Find the last few tokens earlier in the context and propose what followed | No model, almost free | Summarizing, editing code, RAG answers (output copies the input) |
| **Draft model** | A small model of the same family with the same tokenizer (Qwen3-0.6B drafting for Qwen3-1.7B) | A second model in memory | General text |
| **EAGLE / Medusa / MTP** | Small trained heads on top of the target model | Training or downloading heads | Best acceptance rates in production |

**Caveat to measure and explain:** the speed-up **shrinks as the batch grows**, because a busy GPU no longer has idle capacity to spend on checking guesses. Report results at concurrency 1, 4 and 16.

### 5.12 Quantization

| Format | Bytes per weight | Qwen3-0.6B weights **(computed, embeddings/LM head kept in BF16)** |
|---|---|---|
| BF16 | 2 | 1.19 GB |
| INT8 weight-only | 1 | ~0.75 GB |
| INT4 weight-only (group-wise) | 0.5 | ~0.53 GB |

- **Weight-only** quantization stores weights in INT8 or INT4 and converts back to BF16 inside the matmul. It saves memory, and, because decode is memory-bound, it can **speed up decode, but only with a fused kernel**. Naive "dequantize the whole matrix, then matmul" in PyTorch is often **slower** than BF16. Measure it.
- **W8A8** (weights and activations both INT8) uses INT8 tensor cores (Ampere has them) but needs activation calibration. Out of scope for v1.0.
- **FP8 needs Ada or Hopper GPUs** (RTX 40-series, H100). My Ampere card doesn't have FP8 tensor cores, so skip FP8 compute.
- **KV-cache quantization** (INT8 storage with a scale per token and head) doubles the tokens that fit in the pool. A good stretch goal.
- Known methods to read about: per-channel absmax INT8, **LLM.int8()** (outlier handling), **GPTQ** and **AWQ** (4-bit, group-wise, calibration-based).
- **Measure quality**, not just speed: perplexity on WikiText-2, agreement of greedy outputs with BF16, and optionally a small task benchmark. Keep the embedding and LM head in BF16 at first; Qwen3-0.6B ties them, and quantizing them costs the most quality.

### 5.13 Overheads: why a small model can be CPU-bound

With a 0.6B model, the GPU work per decode step is only a few milliseconds. In eager PyTorch, each of the 28 layers launches roughly 10 to 20 small kernels, and every launch costs the CPU a few microseconds **(estimate)**. On top of that come Python logic, building input tensors and copying them to the GPU. **The CPU can easily take as long as the GPU.** Tools and fixes:

- **Profile first:** `torch.profiler` (exports a Chrome trace) and NVIDIA **Nsight Systems** (`nsys`). Look for gaps on the GPU timeline.
- **CUDA graphs:** record the whole decode step once for a fixed batch size, then replay it with a single launch. Needs fixed shapes, so capture batch sizes 1, 2, 4, 8, 16, 32, 64 and pad the real batch up to the next captured size. vLLM does this by default unless `--enforce-eager`.
- **Avoid CPU–GPU syncs** inside the loop: `.item()`, `.tolist()`, `print(tensor)`, and Python `if` on GPU values all force the CPU to wait. Do one sync per step to read the sampled token IDs.
- **Pinned memory and `non_blocking=True`** for host-to-device copies.
- **`torch.compile`** can fuse the small element-wise ops (norms, activations).
- **Overlap scheduling** (SGLang, Mini-SGLang): prepare step N+1 on the CPU while the GPU runs step N. An advanced level-up.

### 5.14 FlashAttention and attention kernels

- **Standard attention** writes the full n × n score matrix to GPU memory and reads it back. FlashAttention computes attention **in tiles that stay in on-chip SRAM**, so that matrix never touches GPU memory. It's exact (not an approximation) and much faster for long sequences, because it moves far fewer bytes.
- **FlashAttention-2** supports Ampere, Ada and Hopper (my RTX 3060 included) in FP16 and BF16. It needs CUDA 12.0+ and PyTorch 2.2+. It supports a **paged KV cache** through `flash_attn_with_kvcache(..., block_table=...)`, with page size a multiple of 256. **FlashAttention-3 needs an H100**, so it's not for me.
- PyTorch's built-in `F.scaled_dot_product_attention` already uses a FlashAttention-style kernel for ordinary (non-paged) attention. That's handy for the reference implementation and for prefill.
- **FlashDecoding** splits a long KV sequence across thread blocks during decode, so even a batch of 1 with a long context keeps the GPU busy.
- Building a paged decode kernel in **Triton** is the best stretch goal for the resume (section 16).

### 5.15 The serving layer

- **OpenAI-compatible API:** `POST /v1/completions` (prompt in) and `POST /v1/chat/completions` (messages in, formatted with the model's chat template). Clients, SDKs and benchmark tools already speak this format.
- **Streaming with Server-Sent Events (SSE):** with `stream: true`, the server keeps the HTTP response open and sends one chunk per new piece of text:

  ```
  data: {"id":"cmpl-1","object":"text_completion","choices":[{"index":0,"text":" Hello","finish_reason":null}]}

  data: {"id":"cmpl-1","object":"text_completion","choices":[{"index":0,"text":"!","finish_reason":"stop"}]}

  data: [DONE]
  ```

- **Incremental detokenization:** one token isn't always valid text on its own. A multi-byte character such as an emoji or a Chinese character can span several tokens. Decode a sliding window of recent tokens and only emit text once it's complete (vLLM keeps a "prefix offset" and a "read offset" per request). Test this with emoji and non-English prompts.
- **Concurrency:** the HTTP server runs on an `asyncio` event loop. The engine loop is a blocking `while True` that runs GPU work, so it must **not** run on the event loop. Put it in a background thread (v1.0) or a separate process (vLLM's design; a level-up).
- **Backpressure:** cap the waiting queue, and return HTTP 503 (or 429) when it's full instead of letting latency grow without limit.
- **Cancellation:** when a client disconnects, abort the request and free its KV blocks. Leaking blocks here is a classic bug.
- **Qwen3 thinking mode:** Qwen3's chat template can emit `<think>…</think>` reasoning. Benchmark with thinking turned off (`enable_thinking=False` in `apply_chat_template`), and say so in the results.

### 5.16 Metrics and how to measure them properly

| Metric | Definition (as `vllm bench serve` uses it) | Who cares |
|---|---|---|
| **TTFT** | Time from sending the request to receiving the first streamed output | Chat users ("is it responding?") |
| **TPOT** | Per request: (end-to-end latency − TTFT) / (output tokens − 1) | Reading speed |
| **ITL** | Gaps between consecutive streamed outputs, across all requests | Smoothness, stalls |
| **E2EL** | Total time per request | Batch jobs |
| **Throughput** | Output tokens/s, total tokens/s, requests/s | Cost per token |
| **Goodput** | Requests/s that **meet a latency SLO** (e.g., TTFT < 500 ms and TPOT < 50 ms) | The honest single number (from the DistServe paper) |

Rules for trustworthy numbers:

- Report **P50, P90 and P99**. A P99 needs enough samples: with 200 requests, P99 rests on 2 data points. Use **≥ 500 requests** for P99 claims.
- **Closed-loop** load (fixed concurrency: each client sends its next request when the last one finishes) shows capacity. **Open-loop** load (requests arrive at a set rate, Poisson by default) shows how latency behaves under real traffic, including queueing. Report both.
- **Warm up** before measuring (first requests include CUDA initialization and graph capture).
- **Little's law:** concurrency = throughput × latency. Use it to sanity-check results.
- The headline chart is a **latency–throughput curve**: x = output tokens/s, y = P99 TPOT (or TTFT), one line per system.

### 5.17 What big systems add (interview awareness)

I won't build these, but I should be able to explain them:

- **Tensor parallelism:** split each weight matrix across GPUs; one all-reduce per layer. Used within one machine because it needs fast links (NVLink).
- **Pipeline parallelism:** different layers on different GPUs or machines; micro-batches keep every stage busy.
- **Expert parallelism:** for mixture-of-experts models, place different experts on different GPUs.
- **Disaggregated prefill and decode** (DistServe, Splitwise; vLLM supports it with KV connectors): run prefill and decode on separate GPU pools so they stop interfering, and send the KV cache between them.
- **KV-cache offloading** to CPU memory or SSD for long contexts and multi-turn reuse.
- **Prefix-aware routing:** send requests that share a prefix to the replica that already has it cached.
- **Multi-LoRA serving:** many fine-tuned adapters on one base model, batched together.
- **Structured outputs:** constrain sampling with a grammar so outputs are guaranteed valid JSON.

---
## 6. Prior art and how to use it

| Project | What it is | Licence | How I use it |
|---|---|---|---|
| **vLLM** | The most widely used open-source inference engine. Its V1 engine has a unified token-budget scheduler, paged KV cache with 16-token blocks, prefix caching on by default, chunked prefill, CUDA graphs, and several speculative-decoding methods. v0.33.0 was released 2026-11-01. | Apache-2.0 | **The yardstick.** Run it on the same GPU and benchmark it with the same client. Read its design, not its code line by line (it's huge). |
| **SGLang** | A fast engine known for RadixAttention (prefix caching with a radix tree) and an overlap scheduler. | Apache-2.0 | Read for prefix caching and CPU/GPU overlap ideas. |
| **nano-vllm** | A from-scratch vLLM in about **1,200 lines of Python**: prefix caching, tensor parallelism, `torch.compile`, CUDA graphs. Its README benchmark: an **RTX 4070 Laptop (8 GB)** with **Qwen3-0.6B** and 256 sequences (random 100 to 1,024 input and output tokens) produced 1,434 tokens/s vs vLLM's 1,362 tokens/s in an offline batch. | MIT | **The closest reference to my setup.** Read all of it in week 1, then close it and write my own. |
| **Mini-SGLang** | LMSYS's compact SGLang (about 5,000 lines of Python): radix cache, chunked prefill, overlap scheduling, tensor parallelism, an OpenAI-compatible API. Tuned for Hopper GPUs. | Check the repo | Read the scheduler and overlap-scheduling code later (M9 and level-ups). |
| **llama.cpp** | C/C++ inference for CPUs and GPUs with GGUF quantized models. | MIT | A different design point (single user, quantized, CPU-friendly). Good for interview comparisons. |
| **TensorRT-LLM** | NVIDIA's compiled inference engine. | Apache-2.0 | Know what it is. Not needed. |

### Positioning: be upfront about nano-vllm

nano-vllm already shows a small Python engine can match vLLM's **offline** throughput. MiniServe is still worth building, and stands out from a nano-vllm clone, because it:

1. is an **online server** (HTTP, streaming, an OpenAI-compatible API, cancellation, backpressure), benchmarked with **vLLM's own client** under real arrival patterns;
2. has an **ablation table**: the measured gain of each technique, one at a time;
3. adds **speculative decoding and quantization experiments** with correctness tests;
4. has **tests and CI**, including correctness against Hugging Face;
5. comes with a **write-up** explaining the results on a 6 GB GPU.

The README credits nano-vllm and vLLM as the references I studied. In interviews: "I studied vLLM and nano-vllm, then built my own and measured each technique." That's honest and strong.

---

## 7. MiniServe architecture

### 7.1 The big picture

```
  Clients: curl, the openai Python SDK, `vllm bench serve`, later TriageBench
        │   POST /v1/completions, /v1/chat/completions  (stream=true → SSE)
        ▼
┌──────────────────────── API server (FastAPI on asyncio) ────────────────────────┐
│ validate request → apply chat template → tokenize → engine.add_request()         │
│ await per-request token queue → detokenize incrementally → SSE chunks → [DONE]   │
│ client disconnect → engine.abort(request_id)                                     │
└───────────────┬───────────────────────────────────────────────▲─────────────────┘
      thread-safe inbox (queue.Queue)            per-request asyncio.Queue
                │                                (filled via call_soon_threadsafe)
                ▼                                               │
┌─────────────────────── Engine loop (one background thread) ───┴─────────────────┐
│ while running:                                                                   │
│     drain inbox (new requests, aborts)                                           │
│     sched = scheduler.schedule()          # who runs this step, how many tokens  │
│     out   = runner.execute(sched)         # one GPU forward pass + sampling      │
│     finished = scheduler.update(sched, out)  # append tokens, stop checks, free  │
│     publish(out)                          # push new tokens to the API side      │
│                                                                                  │
│  Scheduler ──► BlockManager (free list, block tables, ref counts, prefix hashes) │
│  ModelRunner ─► builds flat inputs ─► Qwen3 (my PyTorch code) ─► AttentionBackend│
│              └► Sampler (greedy, temperature, top-k, top-p)                      │
│              └► KV pool tensor [layers, 2, num_blocks, block_size, kv_heads, hd] │
│  (later) SpecDecoder: proposer (n-gram or draft model) + rejection sampler       │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### 7.2 Core data structures

```python
@dataclass
class SamplingParams:
    max_tokens: int = 256
    temperature: float = 1.0          # 0 means greedy
    top_p: float = 1.0
    top_k: int = -1                   # -1 means disabled
    stop: list[str] = field(default_factory=list)
    ignore_eos: bool = False          # benchmarks use fixed output lengths
    seed: int | None = None

class SeqStatus(Enum):
    WAITING = auto(); RUNNING = auto(); FINISHED = auto()

@dataclass
class Sequence:
    request_id: str
    prompt_ids: list[int]
    params: SamplingParams
    output_ids: list[int] = field(default_factory=list)
    status: SeqStatus = SeqStatus.WAITING
    block_table: list[int] = field(default_factory=list)   # physical block ids
    num_computed: int = 0             # tokens whose K/V are already in the cache
    num_cached_prefix: int = 0        # tokens reused from the prefix cache
    t_arrival: float = 0.0            # timestamps for server-side metrics
    t_first_token: float | None = None

    @property
    def all_ids(self) -> list[int]: return self.prompt_ids + self.output_ids
    @property
    def num_tokens(self) -> int: return len(self.prompt_ids) + len(self.output_ids)

@dataclass
class ScheduledSeq:
    seq: Sequence
    num_new_tokens: int               # 1 for decode; a chunk size for (chunked) prefill

@dataclass
class ModelInput:                     # everything one forward pass needs, already on the GPU
    input_ids: Tensor                 # [T]   all scheduled tokens, flattened
    positions: Tensor                 # [T]   each token's position in its own sequence
    slot_mapping: Tensor              # [T]   where each token's K/V goes in the pool
    query_start_loc: Tensor           # [B+1] offsets of each sequence's tokens in the flat batch
    seq_lens: Tensor                  # [B]   total length (cached + new) per sequence
    block_tables: Tensor              # [B, max_blocks] padded block tables
    logits_indices: Tensor            # [B]   index of each sequence's last token
```

### 7.3 The scheduler (V1-style unified token budget)

The idea: there's no separate "prefill phase" and "decode phase". Every step has a budget of tokens (`max_num_batched_tokens`). Each sequence just needs "its next N tokens computed". A decode is N = 1; a prefill chunk is N = chunk size.

```python
def schedule(self) -> list[ScheduledSeq]:
    budget = self.cfg.max_num_batched_tokens        # e.g. 512 to 2048
    out: list[ScheduledSeq] = []

    # 1) Running sequences first (oldest first): decodes, or the next chunk of an unfinished prefill.
    for seq in list(self.running):
        if budget == 0:
            break
        if seq.status is not SeqStatus.RUNNING:
            continue                                # preempted earlier in this loop
        n = min(seq.num_tokens - seq.num_computed, budget)
        while not self.blocks.can_append(seq, n):
            victim = self.running[-1]               # newest running sequence (not yet scheduled)
            self.preempt(victim)                    # free its blocks, status WAITING, front of waiting queue
            if victim is seq:
                break
        if seq.status is SeqStatus.RUNNING:
            self.blocks.append_slots(seq, n)
            out.append(ScheduledSeq(seq, n)); budget -= n

    # 2) New requests, while there is budget and room.
    while self.waiting and budget > 0 and len(self.running) < self.cfg.max_num_seqs:
        seq = self.waiting[0]
        cached = self.blocks.match_prefix(seq)      # prefix-cache hits (M8)
        n = min(seq.num_tokens - cached, budget)    # chunked prefill if the prompt is long
        if not self.blocks.can_allocate(seq, cached + n):
            break
        self.blocks.allocate(seq, cached + n)
        seq.num_computed = cached
        self.waiting.popleft(); self.running.append(seq); seq.status = SeqStatus.RUNNING
        out.append(ScheduledSeq(seq, n)); budget -= n
    return out
```

This is a sketch to reason with, not final code. The design note for M5 must define: the preemption victim policy, what happens to a preempted sequence's partial output, fairness (first come, first served), and the invariants the tests check.

### 7.4 The block manager

Responsibilities: a free list of physical block IDs; `can_allocate`, `allocate`, `can_append`, `append_slots`, `free`; block tables per sequence; reference counts; and (M8) the prefix-cache hash table plus an LRU of cached free blocks.

**Invariants to test** (property-based tests with Hypothesis):

- A physical block is never owned by two sequences unless it's a shared prefix block with reference count ≥ 2.
- free + in use (+ cached and evictable) = total, always. No leaks.
- `len(block_table) == ceil(num_computed_tokens / block_size)` after every step.
- Freeing everything returns the pool to its starting state.

### 7.5 The model runner and the KV pool

- **KV pool:** one big tensor allocated at start-up, `[num_layers, 2, num_blocks, block_size, num_kv_heads, head_dim]` in BF16. Each layer gets a view of its slice.
- **Storing K/V:** after computing k and v for the T new tokens, write them into the pool with the slot mapping: reshape the layer's pool to `[num_blocks * block_size, kv_heads, head_dim]` and use `index_copy_` (or a tiny Triton kernel later).
- **Building inputs:** one pass over the scheduled sequences builds the Python lists, then **one** copy to the GPU with pinned memory, rather than many small copies.
- **Logits:** gather the hidden states at `logits_indices` first, then apply the LM head (section 5.5).
- **Sampler:** temperature → top-k → top-p → softmax → `torch.multinomial`, batched across sequences with per-sequence parameters. Greedy uses `argmax`.

### 7.6 Attention backends (one interface, three implementations)

| Backend | How | Use |
|---|---|---|
| `TorchRefBackend` | For each sequence, gather its K/V from the pool through its block table into a contiguous tensor, then call `F.scaled_dot_product_attention` with a causal mask. Simple loop, slow. | Correctness reference; runs on CPU in CI. |
| `FlashAttnBackend` | FlashAttention-2: a variable-length kernel for prefill and `flash_attn_with_kvcache` with `block_table` for decode (block size 256). | The fast path for v1.0. Check the exact function signatures for the installed flash-attn version; nano-vllm's attention layer is a compact example of the split. |
| `TritonPagedBackend` (stretch) | My own Triton kernel for paged decode attention with 16-token blocks. | Resume and interview material (section 16). |

Every backend must pass the **same** correctness tests against `TorchRefBackend`.

### 7.7 Threading model

- The API server runs on the main thread's asyncio loop.
- The engine runs in **one background thread**. GPU kernels run asynchronously, but Python code in both threads shares the GIL, so heavy Python in the API thread can slow the engine. Measure it.
- Inbox: `queue.Queue` (thread-safe). Outbox: each request owns an `asyncio.Queue`; the engine thread pushes with `loop.call_soon_threadsafe(q.put_nowait, item)`.
- **Level-up:** move the engine to its own process (vLLM's EngineCore runs in a separate process and talks over ZeroMQ). Measure the difference in ITL under load.

---

## 8. Repository layout and tooling

```
miniserve/
├── pyproject.toml            # uv project; ruff, mypy (strict), pytest settings
├── uv.lock
├── README.md                 # public README: diagram, results, ablation, how to reproduce, credits
├── docs/
│   ├── PLAN.md               # this file
│   ├── design/M1.md …        # one short design note per milestone
│   ├── RESULTS.md            # every measured number with its settings
│   └── LEARNING_LOG.md       # surprises and lessons (interview material)
├── src/miniserve/
│   ├── config.py             # EngineConfig (pydantic): model path, dtype, block size, budgets, memory fraction
│   ├── cli.py                # `miniserve serve …` entry point
│   ├── api/
│   │   ├── server.py         # FastAPI app, routes, SSE streaming, abort on disconnect
│   │   ├── protocol.py       # pydantic models for the OpenAI request/response subset
│   │   └── detokenizer.py    # incremental detokenization
│   ├── engine/
│   │   ├── engine.py         # engine thread, add_request, abort, step loop
│   │   ├── sequence.py       # Sequence, SamplingParams, status
│   │   ├── scheduler.py
│   │   └── block_manager.py  # paging, ref counts, prefix cache
│   ├── model/
│   │   ├── qwen3.py          # my Qwen3 implementation
│   │   ├── layers.py         # RMSNorm, RoPE, SwiGLU, fused linear layers
│   │   └── loader.py         # safetensors → my modules (name mapping, fusing)
│   ├── attention/backends.py # TorchRef, FlashAttn, (Triton)
│   ├── runner/
│   │   ├── model_runner.py   # inputs, KV pool, memory profiling, CUDA graphs
│   │   └── sampler.py
│   ├── spec/                 # ngram.py, draft.py, rejection.py
│   ├── quant/int8.py
│   └── baseline/hf_server.py # M1 naive baseline
├── bench/
│   ├── loadgen.py            # my own async load generator (M2)
│   ├── datasets.py           # builds fixed prompt files with a seed
│   ├── run_matrix.py         # runs a sweep, saves JSON to results/
│   ├── plot.py               # latency–throughput curves
│   └── gpu_monitor.sh        # nvidia-smi logging during runs
├── results/                  # raw JSON per run (small; committed)
├── tests/
│   ├── unit/                 # block manager, scheduler, sampler, rejection sampler, detokenizer
│   ├── correctness/          # my model vs Hugging Face on a tiny random Qwen3 (CPU)
│   ├── gpu/                  # @pytest.mark.gpu: real model checks (local only)
│   └── api/                  # openai SDK contract tests against a test server
└── .github/workflows/ci.yml  # ruff, mypy, pytest -m "not gpu" on ubuntu-latest (CPU)
```

**Conventions** (same as my other projects): Python 3.12 with `uv`; `ruff` for lint and format; `mypy --strict` on `src/miniserve`; `pytest`; type hints everywhere; pydantic for config and API schemas; small commits with imperative messages.

**Two virtual environments, on purpose:**

- `~/miniserve/.venv` (managed by `uv`): my project, with its own PyTorch.
- `~/venvs/vllm`: vLLM only. vLLM pins specific PyTorch and transformers versions (for example, vLLM 0.32 narrowed transformers to 5.16.1–5.20.0), and they shouldn't fight with mine. Only one server runs at a time anyway; the GPU can't hold both.

---
## 9. Milestones M0 to M12

Each milestone has: **goal**, **learn first**, **build**, **exit criteria** (tests that must pass) and **measure** (numbers that go in `docs/RESULTS.md`). Time is in focused weekends (about 2 days each).

**Minimum resume-worthy cut:** if time runs short, **M0 to M7 plus M12** is already a complete project: continuous batching, a paged KV cache, an OpenAI-compatible streaming API, and a measured comparison with vLLM. M8 to M11 are what make it stand out.

| # | Milestone | Time | Depends on |
|---|---|---|---|
| M0 | Environment and GPU facts | 0.5 | — |
| M1 | Naive Hugging Face baseline (OpenAI-style endpoint) | 1 | M0 |
| M2 | Load generator and metrics | 1 | M1 |
| M3 | My own Qwen3 model, proven correct | 1 | M0 |
| M4 | KV cache for one sequence | 0.5 | M3 |
| M5 | Continuous batching (fixed KV slots) | 2 | M4 |
| M6 | Paged KV cache and preemption | 2 | M5 |
| M7 | Full OpenAI API, streaming, cancellation | 1 | M6 |
| M8 | Prefix caching and chunked prefill | 1.5 | M7 |
| M9 | CUDA graphs and overhead reduction | 1 | M7 |
| M10 | Speculative decoding | 1.5 | M7 |
| M11 | Quantization | 1 | M7 |
| M12 | Head-to-head benchmark, README, write-up | 1 | all |

---

### M0. Environment and GPU facts (0.5 weekend)

**Goal:** a working GPU stack in WSL2, vLLM running as a reference, and the real numbers for my GPU.

**Learn first:** how WSL2 shares the Windows GPU driver; what a CUDA "compute capability" is (mine is 8.6, Ampere); the difference between the CUDA driver, the CUDA runtime that ships inside PyTorch wheels, and the CUDA toolkit (`nvcc`, only needed to compile extensions).

**Build:**

1. *(Optional, Windows side)* Give WSL more memory. By default WSL2 gets half of the RAM. Create or edit `%USERPROFILE%\.wslconfig`:

   ```ini
   [wsl2]
   memory=10GB
   swap=8GB
   ```

   Then run `wsl --shutdown` in PowerShell and reopen Ubuntu. This affects ARIS too, which is fine.

2. Check the GPU from inside WSL:

   ```bash
   nvidia-smi    # should list "NVIDIA GeForce RTX 3060 Laptop GPU" with 6144MiB
   ```

   **Do not install a Linux NVIDIA driver inside WSL.** The Windows driver provides the GPU. Install the CUDA toolkit only if a build needs `nvcc`, and then only the WSL-Ubuntu toolkit package (it contains no driver).

3. Create the project inside the WSL file system:

   ```bash
   cd ~ && uv init --package --python 3.12 miniserve && cd miniserve
   uv add torch transformers safetensors fastapi "uvicorn[standard]" pydantic numpy httpx
   uv add --dev pytest pytest-asyncio hypothesis ruff mypy openai
   uv run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
   # expect: <version> <cuda version> True NVIDIA GeForce RTX 3060 Laptop GPU (8, 6)
   ```

   If `is_available()` prints `False`, follow uv's "Using uv with PyTorch" guide to point at a CUDA build of PyTorch.

4. Download the models:

   ```bash
   uv run hf download Qwen/Qwen3-0.6B --local-dir ~/models/Qwen3-0.6B
   uv run hf download Qwen/Qwen3-1.7B --local-dir ~/models/Qwen3-1.7B   # for M10/M11 experiments
   # older huggingface_hub versions: use `huggingface-cli download` instead of `hf download`
   ```

5. Measure real memory bandwidth (the spec says 336 GB/s; expect less):

   ```python
   # bench/bandwidth.py
   import time, torch
   n = 256 * 1024 * 1024                     # 256M BF16 values = 512 MiB
   x = torch.empty(n, dtype=torch.bfloat16, device="cuda"); y = torch.empty_like(x)
   for _ in range(3): y.copy_(x)             # warm-up
   torch.cuda.synchronize(); t = time.perf_counter()
   for _ in range(20): y.copy_(x)
   torch.cuda.synchronize(); dt = time.perf_counter() - t
   print(f"{2 * x.numel() * x.element_size() * 20 / dt / 1e9:.0f} GB/s")   # each copy reads and writes
   ```

6. Record free VRAM with nothing running: `nvidia-smi --query-gpu=memory.used,memory.total --format=csv`. Note how much Windows itself uses.

7. Install vLLM in its own environment and check it serves:

   ```bash
   uv venv ~/venvs/vllm --python 3.12
   source ~/venvs/vllm/bin/activate
   uv pip install vllm --torch-backend=auto
   vllm serve ~/models/Qwen3-0.6B --served-model-name Qwen/Qwen3-0.6B \
       --max-model-len 4096 --gpu-memory-utilization 0.80 --port 8001
   # second terminal:
   curl http://localhost:8001/v1/completions -H "Content-Type: application/json" \
       -d '{"model": "Qwen/Qwen3-0.6B", "prompt": "The capital of Canada is", "max_tokens": 16}'
   ```

   If it runs out of memory at start-up, lower `--gpu-memory-utilization` (Windows may be using VRAM). Stop it with Ctrl+C and `deactivate` when done.

**Exit criteria:** PyTorch sees compute capability (8, 6); vLLM answers the curl request; bandwidth, idle VRAM and driver/CUDA/PyTorch/vLLM versions are written in `docs/RESULTS.md` under "Environment".

**Measure:** real bandwidth (GB/s); VRAM used at idle; redo the speed-limit table in section 5.3 with the real bandwidth.

---

### M1. Naive Hugging Face baseline (1 weekend)

**Goal:** the "before" picture. The simplest server that works, serving one request at a time.

**Learn first:** `AutoModelForCausalLM.generate`; why `asyncio.to_thread` keeps the event loop responsive; the OpenAI completions request and response format, including SSE streaming.

**Build:** `src/miniserve/baseline/hf_server.py`, exposing a minimal **`POST /v1/completions`** (non-streaming and streaming), so the **same benchmark client** can measure the baseline and everything later.

```python
tok = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.bfloat16).cuda().eval()
lock = asyncio.Lock()      # one request at a time: this IS the naive baseline

@app.post("/v1/completions")
async def completions(req: CompletionRequest):
    async with lock:
        if req.stream:
            return StreamingResponse(stream_with_text_iterator_streamer(req), media_type="text/event-stream")
        out = await asyncio.to_thread(run_generate, req)   # model.generate(..., max_new_tokens, min_new_tokens if ignore_eos)
        return completion_response(req, out)
```

(Older transformers versions use `torch_dtype=` instead of `dtype=`.) For streaming, use `TextIteratorStreamer` with `generate` running in a thread, and send each text piece as an SSE chunk. Respect `ignore_eos` by setting `min_new_tokens = max_tokens`, so benchmark outputs have a fixed length.

*Optional second baseline:* a **static batching** server that collects up to 8 requests for 50 ms, pads them, and calls `generate` once. It shows why static batching isn't enough (short requests wait for long ones).

**Exit criteria:** the `openai` SDK can call it (streaming and non-streaming); a stream ends with `data: [DONE]`.

**Measure:** done in M2.

---

### M2. Load generator and metrics (1 weekend)

**Goal:** a trustworthy way to measure, built **before** any optimization, so every later change has a before-and-after number.

**Learn first:** section 5.16 (metrics, percentiles, open vs closed loop, warm-up).

**Build:**

1. `bench/datasets.py`: writes fixed prompt files with a seed: random-token prompts of a given length (decoded from random token IDs and re-tokenized to check the length), a shared-prefix set, and later a ShareGPT sample.
2. `bench/loadgen.py`: `asyncio` + `httpx` streaming client.
   - Modes: `--concurrency N` (closed loop) or `--rate R` (open loop, Poisson arrivals).
   - Per request it records send time, every chunk's arrival time and token count, and end time.
   - It computes TTFT, TPOT, ITL and E2EL **exactly as `vllm bench serve` defines them** (section 5.16) and prints P50/P90/P99 plus throughput.
   - It saves raw per-request JSON to `results/`.
3. `bench/gpu_monitor.sh`: `nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,clocks.sm,temperature.gpu,power.draw --format=csv -l 1 > results/<run>_gpu.csv` in the background during each run.
4. **Cross-check:** run my load generator and `vllm bench serve` against the baseline with the same settings. Their numbers should agree within a few percent. If they don't, find out why; it's usually a TTFT or token-counting definition.

**Exit criteria:** one command produces a results JSON and a summary table; the cross-check agrees.

**Measure:** baseline at concurrency 1, 4 and 16 with 256 input / 128 output tokens: output tokens/s, TTFT and TPOT P50/P99. This is row 1 of the results table.

---

### M3. My own Qwen3 model, proven correct (1 weekend)

**Goal:** a plain-PyTorch Qwen3 that I fully understand and can change (paging needs control over attention), loaded from the official weights, matching Hugging Face's outputs.

**Learn first:** section 5.2; how `safetensors` files store named tensors; RoPE's "rotate half" layout.

**Build:**

- `model/layers.py`: `RMSNorm`, `RotaryEmbedding` (precompute cos/sin for all positions up to `max_model_len` with theta = 1,000,000; index them by the `positions` tensor), SwiGLU MLP.
- `model/qwen3.py`: attention (section 5.2), decoder layer, model, tied LM head. Forward signature: `forward(input_ids [T], positions [T], attn_meta) -> hidden [T, H]`, plus `compute_logits(hidden_at_last_tokens)`.
- `model/loader.py`: read the `.safetensors` files and map names. Print the keys first to check; they look like `model.layers.0.self_attn.q_proj.weight`, `...q_norm.weight`, `...mlp.gate_proj.weight`, `model.norm.weight`. With tied embeddings there may be no `lm_head.weight`.
- For now the attention backend is the simple full-sequence `F.scaled_dot_product_attention(is_causal=True)` with no cache.

**Exit criteria:**

1. **CPU test, runs in CI.** Build a tiny random Qwen3 with transformers' `Qwen3Config` (for example 2 layers, hidden 64, 4 query heads, 2 KV heads, head_dim 16, vocab 1,000, `tie_word_embeddings=True`). Load its `state_dict` into my model and assert logits match in FP32 (`torch.testing.assert_close(..., atol=1e-5, rtol=1e-4)`). No download needed.
2. **GPU test, local.** Real Qwen3-0.6B in BF16 on 20 prompts: the top-1 next token matches Hugging Face for every prompt; the max absolute logit difference is small (record it); greedy 64-token generations match for most prompts. BF16 rounding can flip near-ties, so record the match rate rather than demanding 100%.

**Measure:** nothing yet (no cache, so it's slow by design).

---

### M4. KV cache for one sequence (0.5 weekend)

**Goal:** see the O(n²) → O(n) change with my own eyes.

**Build:** a contiguous per-sequence cache `[layers, 2, max_len, kv_heads, head_dim]`. Prefill writes positions 0…P-1; each decode step writes one position and attends over the cache.

**Exit criteria:** greedy outputs with and without the cache are identical (FP32 tiny model in CI; BF16 match rate on the GPU).

**Measure:** milliseconds per token at positions 100, 500, 1,000 and 2,000, with and without the cache. Plot it; it goes in the write-up.

---

### M5. Continuous batching with fixed KV slots (2 weekends)

**Goal:** the core of an inference engine: many requests in flight, joining and leaving at every step.

**Learn first:** sections 5.6 and 7.1 to 7.3 and 7.7; read nano-vllm's scheduler and model runner.

**Build:**

- `engine/sequence.py`, `engine/scheduler.py` (waiting and running queues, `max_num_seqs`, `max_num_batched_tokens`, first come first served), `engine/engine.py` (background thread, inbox and outbox as in 7.7).
- Flat batching as in section 5.6: one `input_ids` tensor for all sequences, per-token `positions`, `logits_indices`.
- **KV storage for this milestone:** each running sequence gets a **fixed slot** of `max_model_len` tokens in a preallocated cache, the way older servers reserved memory. Attention loops over sequences, each using its own slot. This is deliberately wasteful: M6 replaces it, and the before-and-after is a great result.
- Plug the engine into the M1-style `/v1/completions` route.

**Exit criteria:**

- 64 concurrent requests all complete with correct text. With greedy decoding, each output matches the same prompt run alone (exact match on the CPU tiny model; record the match rate on the GPU).
- No request starves: every request finishes in a soak test of 1,000 requests.

**Measure:**

- Throughput and latency vs the baseline at concurrency 1, 4, 16 and 32.
- **Maximum concurrent sequences:** with fixed slots, that's KV memory ÷ (max_model_len × 112 KiB). For a 3 GiB pool **(computed)**: about 6 slots at max_model_len 4,096, or 13 at 2,048. Put the measured number in RESULTS.md.

---

### M6. Paged KV cache and preemption (2 weekends)

**Goal:** stop reserving memory for the worst case. Memory is used block by block.

**Learn first:** section 5.7 and 5.10; the PagedAttention paper (sections 1 to 4); vLLM's "Inside vLLM" post on the KV cache manager.

**Build:**

- `engine/block_manager.py` (section 7.4) with block size 16 for the reference backend.
- **Start-up memory profiling** (section 5.5) to size `num_blocks`, with a `gpu_memory_utilization` setting (default 0.80).
- KV pool tensor and slot-mapping writes (section 7.5).
- `TorchRefBackend` first (gather through block tables). Then `FlashAttnBackend` with block size 256. Installing `flash-attn` can mean a long compile. Look for a prebuilt wheel matching the exact PyTorch, CUDA and Python versions first. If compiling, limit parallel jobs (`MAX_JOBS=2`) so 16 GB of RAM isn't exhausted.
- Preemption by recompute (section 5.10).

**Exit criteria:**

- Hypothesis property tests for the block manager invariants (section 7.4) pass.
- Paged outputs match M5 outputs (same match-rate method).
- **Leak test:** after 10,000 requests, including aborted ones, the free-block count returns to the starting value.
- Forcing a tiny pool (for example 64 blocks) still completes every request through preemption.

**Measure:** maximum concurrent sequences before preemption (vs M5's fixed slots); throughput and latency at concurrency 16, 32 and 64; the number of preemptions under overload. **This is the headline paging result.**

---

### M7. Full OpenAI API, streaming and cancellation (1 weekend)

**Goal:** a server any OpenAI client and `vllm bench serve` can use, robust to real clients.

**Learn first:** section 5.15.

**Build:**

- `/v1/completions` and `/v1/chat/completions` (subset: `model`, `prompt` or `messages`, `max_tokens`, `temperature`, `top_p`, `top_k`, `stop`, `stream`, `seed`, `ignore_eos` as an extension; `n = 1` only, and say so), `/v1/models`, `/health`.
- Chat uses `tokenizer.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False)`.
- The `usage` field (prompt, completion and total tokens).
- `api/detokenizer.py`: incremental detokenization, tested with emoji and Chinese text.
- **Abort on disconnect:** detect the closed connection, call `engine.abort(id)`, free blocks.
- **Backpressure:** cap the waiting queue (`max_waiting`), returning 503 when it's full.
- *Optional:* a `/metrics` endpoint in Prometheus text format (running and waiting counts, free blocks, tokens/s).

**Exit criteria:**

- API contract tests: the official `openai` Python SDK with `base_url="http://localhost:8000/v1"` works for both endpoints, streaming and non-streaming.
- Closing 100 streaming clients mid-generation frees every block (leak test).
- `vllm bench serve` runs against MiniServe without errors.

**Measure:** from now on, use `vllm bench serve` as the main client (section 10). Re-measure M6 with it to start the official results table.

---

### M8. Prefix caching and chunked prefill (1.5 weekends)

**Goal:** reuse shared prompt work, and keep token streams smooth when long prompts arrive.

**Learn first:** sections 5.8 and 5.9; the Sarathi-Serve paper (sections 1 to 4); SGLang's RadixAttention section.

**Build:**

- **Prefix cache** in the block manager: chained block hashes, hash → block table, reference counts, an LRU of evictable cached blocks, and `match_prefix(seq)` used by the scheduler. Only full blocks are cached.
- **Chunked prefill** in the scheduler: decodes first, then fill the remaining token budget with prefill chunks; config `max_num_batched_tokens` (try 256, 512, 1,024, 2,048).
- A config flag to turn each feature off, for the ablation.

**Exit criteria:**

- Outputs are identical with prefix caching on and off (greedy, CPU tiny model).
- A test where two requests share a prefix proves the second one computed only its suffix (count computed tokens).
- Eviction under memory pressure doesn't break correctness.

**Measure:**

- **Shared-prefix workload** (1,024-token shared prefix + 128 unique tokens, 128 output tokens): TTFT P50/P99 and throughput with the cache off vs on; cache hit rate.
- **Mixed workload** (steady short chats plus a 2,048-token prompt every few seconds): ITL P99 and TTFT with chunked prefill off vs on, for two or three budget values.

---

### M9. CUDA graphs and overhead reduction (1 weekend)

**Goal:** stop the CPU from being the bottleneck for a small model.

**Learn first:** section 5.13; the PyTorch CUDA Graphs docs (`torch.cuda.CUDAGraph`, `torch.cuda.graph`).

**Build:**

1. **Profile first** with `torch.profiler` (export a Chrome trace and open it in Perfetto) under a decode-heavy load. Note the GPU idle gaps and the top CPU costs. Nsight Systems (`nsys profile`) gives a fuller picture if it works under WSL on this setup; check NVIDIA's docs.
2. Remove CPU syncs from the step loop; use pinned memory and `non_blocking` copies; build input tensors in one copy.
3. **CUDA graphs** for decode-only steps: static input and output buffers, capture batch sizes [1, 2, 4, 8, 16, 32, 64], pad to the next captured size, replay. The attention call must have static shapes (fixed maximum blocks per sequence in the block-table tensor). Mixed prefill steps stay eager.
4. *Optional:* `torch.compile` on the MLP and norm modules.

**Exit criteria:** outputs identical with graphs on and off; a `--enforce-eager` flag turns graphs off (same name as vLLM's, so comparisons are clear).

**Measure:** TPOT P50 at concurrency 1 and 16, graphs off vs on; a profiler screenshot before and after for the write-up.

---

### M10. Speculative decoding (1.5 weekends)

**Goal:** more tokens per forward pass when the GPU has spare capacity, with exact output distributions.

**Learn first:** section 5.11; the Leviathan et al. paper (sections 1 to 3); vLLM's speculative decoding docs (n-gram and draft-model methods).

**Build:**

1. `spec/ngram.py`: prompt-lookup proposer. Search the sequence's tokens for the most recent earlier match of its last n tokens (try n = 3, then 2, then 1) and propose the k tokens that followed. If there's no match, propose nothing (plain decode).
2. **Verify step:** the target processes, for each sequence, its last accepted token plus the k proposals as a small multi-token step (a mini-prefill). Scheduling and the KV cache must handle k+1 new tokens per sequence, then **roll back** the rejected ones (reset `num_computed` and release now-unused slots).
3. `spec/rejection.py`: the rejection sampler (section 5.11), batched. Greedy is the special case "accept while draft equals argmax". An n-gram proposer is deterministic (q is one-hot), so the rule becomes "accept with probability p(x)".
4. *Optional, memory permitting:* a **draft model** (Qwen3-0.6B drafting for Qwen3-1.7B). Weights alone are 1.19 + 3.44 = **4.63 GB (computed)**, leaving very little KV space on 6 GB. Try it with short contexts or with the INT8 target from M11. If it doesn't fit usefully, write that up honestly; that's a valid engineering finding.

**Exit criteria:**

- **Greedy equivalence:** with speculation on, greedy outputs are token-for-token identical to speculation off (FP32 tiny model in CI).
- **Distribution test:** for a fixed toy target p and draft q over 5 tokens, 200,000 samples from the rejection sampler pass a chi-square test against p.

**Measure** at concurrency 1, 4 and 16, on two workloads: **copy-heavy** (summarize or edit a provided passage; n-gram should shine) and **open-ended chat** (n-gram should be weak). Report acceptance rate, mean accepted tokens per step, TPOT and throughput vs no speculation, and vLLM's n-gram speculation on the same workloads (check its current `--speculative-config` syntax in the docs). Negative results stay in the table.

---

### M11. Quantization (1 weekend)

**Goal:** know the real memory/speed/quality trade-off on my GPU.

**Learn first:** section 5.12; the LLM.int8() and AWQ papers' introductions.

**Build:**

1. `quant/int8.py`: per-output-channel symmetric INT8 weight-only for every linear layer except the embedding and LM head. Scale `s = max(|w_row|) / 127`; store `int8` weights plus a BF16 scale per row; forward = `x @ (w_int8.to(bf16) * s).T`. Simple and educational, and probably **slower** than BF16 because it dequantizes the whole matrix every call.
2. Then try a fused implementation (for example, torchao's INT8 weight-only configuration; check its current API name) and compare.
3. *Stretch:* INT8 KV-cache storage with a scale per token and head, which roughly doubles the tokens in the pool.

**Exit criteria:** quantized model loads and serves through the same API; the quality evaluation script runs end to end.

**Measure:**

| Variant | Weight memory | KV tokens that fit | Output tokens/s at concurrency 16 | WikiText-2 perplexity | Greedy agreement with BF16 |
|---|---|---|---|---|---|
| BF16 | | | | | 100% |
| INT8 naive | | | | | |
| INT8 fused | | | | | |

Perplexity on the WikiText-2 test set uses a sliding window (for example 2,048-token context, stride 512) and must use the same script for every variant. Greedy agreement is the share of 200 prompts whose first 64 greedy tokens match BF16 exactly.

---

### M12. Head-to-head benchmark, README and write-up (1 weekend)

**Goal:** turn the work into proof that a reviewer understands in two minutes.

**Build:**

1. Run the full matrix in section 10 for: HF baseline, MiniServe at each milestone's configuration (the ablation), MiniServe final, and vLLM (default and `--enforce-eager`).
2. `bench/plot.py`: the latency–throughput curve (all systems on one chart) and an ablation bar chart.
3. Public **README**: one-paragraph pitch, the architecture diagram, a feature list, the results table, the ablation table, the chart, "reproduce in three commands", limitations, what I'd do next, and credits (vLLM, nano-vllm, papers).
4. Write-up draft for hireahraz.dev (section 14). **Ask me before publishing anything or making the repo public.**
5. Fill the resume bullet from `docs/RESULTS.md` (section 14).

**Exit criteria:** someone else can clone the repo, follow the README on a CUDA GPU, and reproduce the main table within reason.

---
## 10. Benchmark protocol

### 10.1 Fairness rules (every run)

- **One server at a time** on the GPU. Restart the server between systems.
- **Same model files, dtype (BF16), `max_model_len` (4,096) and memory fraction** (`--gpu-memory-utilization 0.80`, or whatever M0 shows is safe) for MiniServe and vLLM.
- **Same client:** `vllm bench serve` for everything from M7 on (and for the baseline, which already speaks `/v1/completions`).
- **Fixed output lengths** with `--ignore-eos`, so every system generates the same number of tokens.
- **Same prompts:** fixed seed (`--seed 0`) or saved dataset files.
- **Warm-up run first** (discard it), then **3 measured runs**; report the median and note the spread.
- **Laptop conditions:** plugged in, Windows power mode "Best performance", other GPU apps closed, `bench/gpu_monitor.sh` logging clocks, temperature and power. If clocks drop during a run (thermal throttling), note it and rerun after cooling.
- **Record versions:** NVIDIA driver, CUDA, PyTorch, vLLM, flash-attn, MiniServe commit hash.
- **vLLM in two configurations:** default (CUDA graphs on) and `--enforce-eager`. Before M9, MiniServe compares fairly only against the eager one; after M9, against both.
- **Match features per experiment:** vLLM has prefix caching on by default. Turn it off (check the exact flag with `vllm serve --help`) for experiments where MiniServe's is off.

### 10.2 Workloads

| ID | Name | Shape | Tests |
|---|---|---|---|
| W1 | Short chat | Random prompts, 256 in / 128 out | Core throughput and latency |
| W2 | Long prompt | Random prompts, 2,048 in / 128 out | Prefill cost, TTFT |
| W3 | Shared prefix | 1,024-token shared prefix + 128 unique tokens, 128 out (vLLM's `prefix_repetition` dataset, see `--prefix-repetition-prefix-len`; check `--help` for the rest) | Prefix caching |
| W4 | Real conversations | ShareGPT sample (benchmark use only; check the dataset card's terms) | Realistic length mix |
| W5 | Copy-heavy | "Summarize or rewrite this passage" prompts (a custom dataset I generate) | Speculative decoding (n-gram) |
| W6 | Mixed | W1 at steady concurrency plus a W2 request every few seconds (two clients at once) | Chunked prefill (ITL stalls) |

**Sweeps:**

- **Closed loop:** `--max-concurrency` 1, 2, 4, 8, 16, 32, 64, with 500 prompts each (fewer at concurrency 1 and 2 to save time, but note it).
- **Open loop:** `--request-rate` 1, 2, 4, 8, 16, then `inf`, with 500 prompts each.

### 10.3 The commands

```bash
# Terminal 1: the server under test (one at a time)
uv run miniserve serve --model ~/models/Qwen3-0.6B --served-model-name Qwen/Qwen3-0.6B \
    --port 8000 --max-model-len 4096 --gpu-memory-utilization 0.80
#   or vLLM:
# source ~/venvs/vllm/bin/activate && vllm serve ~/models/Qwen3-0.6B --served-model-name Qwen/Qwen3-0.6B \
#     --port 8000 --max-model-len 4096 --gpu-memory-utilization 0.80 [--enforce-eager]

# Terminal 2: the client (always vLLM's)
source ~/venvs/vllm/bin/activate
vllm bench serve --backend vllm --endpoint /v1/completions --host 127.0.0.1 --port 8000 \
    --model Qwen/Qwen3-0.6B --tokenizer ~/models/Qwen3-0.6B \
    --dataset-name random --random-input-len 256 --random-output-len 128 \
    --num-prompts 500 --max-concurrency 16 --ignore-eos --seed 0 \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
    --goodput ttft:500 tpot:50 \
    --save-result --result-dir results/ --result-filename miniserve_W1_c16_run1.json
```

All of these flags exist in vLLM's current `bench serve` docs. Confirm the exact `--goodput` syntax with `--help`. `bench/run_matrix.py` wraps this command, loops over the sweep, and starts and stops `gpu_monitor.sh` around each run.

### 10.4 Results tables (templates for `docs/RESULTS.md` and the README)

**Main comparison** (fill only from saved runs):

| System | Config | Workload | Concurrency | Output tok/s | Req/s | TTFT P50 / P99 (ms) | TPOT P50 / P99 (ms) | ITL P99 (ms) | Goodput (req/s) | Peak VRAM (GB) |
|---|---|---|---|---|---|---|---|---|---|---|
| HF baseline | one at a time | W1 | 16 | | | | | | | |
| MiniServe | final | W1 | 16 | | | | | | | |
| vLLM | `--enforce-eager` | W1 | 16 | | | | | | | |
| vLLM | default | W1 | 16 | | | | | | | |

**Ablation** (W1, concurrency 16 unless noted):

| Step | Change | Output tok/s | TPOT P99 (ms) | Max concurrent sequences | Change vs previous step |
|---|---|---|---|---|---|
| 0 | HF baseline | | | 1 | — |
| 1 | + continuous batching (fixed slots) | | | | |
| 2 | + paged KV cache | | | | |
| 3 | + FlashAttention backend | | | | |
| 4 | + CUDA graphs | | | | |
| 5 | + prefix caching (W3) | | | | |
| 6 | + chunked prefill (W6, ITL P99) | | | | |
| 7 | + n-gram speculation (W5, concurrency 1) | | | | |
| 8 | + INT8 weights | | | | |

**Every table states:** GPU (RTX 3060 Laptop, 6 GB), model (Qwen3-0.6B, BF16 unless noted), versions, date, commit.

---

## 11. Testing strategy

| Layer | What | Where it runs |
|---|---|---|
| Unit | Block manager, scheduler decisions, sampler (top-k/top-p masks), rejection sampler, detokenizer, SSE formatting | CI (CPU) |
| Property-based (Hypothesis) | Block-manager invariants under random allocate/append/free/abort sequences; scheduler never exceeds its token budget or `max_num_seqs`; every request eventually finishes | CI (CPU) |
| Model correctness | My Qwen3 vs Hugging Face on a tiny random config, FP32, `assert_close` on logits | CI (CPU) |
| Engine correctness | Batched outputs = single-request outputs; paged = contiguous; prefix cache on = off; speculation on = off (greedy, tiny model, exact match) | CI (CPU) |
| Statistical | Rejection sampler output distribution vs target (chi-square, 200k samples) | CI (CPU, fast) |
| API contract | `openai` SDK against a test server running the tiny model on CPU: completions and chat, streaming and not, `[DONE]`, `usage`, stop strings, abort | CI (CPU) |
| GPU | Real Qwen3-0.6B: top-1 agreement with HF, greedy match rate, leak tests (10,000 requests), CUDA-graph equivalence | Local, `pytest -m gpu` |
| Benchmarks | Section 10 | Local, by hand or with `run_matrix.py` |

**Why the tiny-model trick matters:** CI has no GPU. A randomly initialized 2-layer Qwen3 built from `Qwen3Config` runs on CPU in milliseconds, needs no download, and exercises every code path (model, paging, scheduling, API). It catches most bugs.

**About exact-match tests:** on the GPU, results can differ slightly between a sequence run alone and in a batch, because different batch shapes use different kernel reduction orders (floating-point addition isn't associative). Exact-match tests therefore run in FP32 on CPU; GPU tests report match **rates**. Thinking Machines' 2025 post "Defeating Nondeterminism in LLM Inference" explains this well; it's a good interview topic.

**CI skeleton** (`.github/workflows/ci.yml`):

```yaml
name: ci
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4          # use the current major versions
      - uses: astral-sh/setup-uv@v6
      - run: uv sync                       # configure CI to install the CPU-only PyTorch build (see uv's PyTorch guide);
                                           # the default Linux wheel pulls several GB of CUDA libraries
      - run: uv run ruff check . && uv run ruff format --check .
      - run: uv run mypy src/miniserve
      - run: uv run pytest -m "not gpu" -q
```

---

## 12. Risks and pitfalls

| Risk | Sign | What to do |
|---|---|---|
| 6 GB runs out | `CUDA out of memory` at start-up or under load | Lower `gpu_memory_utilization` and `max_num_batched_tokens`; compute logits only for last tokens (5.5); check Windows' VRAM use. |
| Windows uses the NVIDIA GPU for the display | Idle VRAM well above zero in `nvidia-smi` | Measure it in M0 and budget for it. If the laptop has an Intel iGPU, plug the monitor into the port it drives where possible. |
| flash-attn install fails or eats RAM | Compile runs for an hour, or WSL freezes | Look for a prebuilt wheel matching the exact PyTorch, CUDA and Python versions; otherwise `MAX_JOBS=2`, more WSL memory and swap (M0). Keep `TorchRefBackend` as a fallback. |
| FP16 overflow | NaNs or garbage | Use BF16 (Qwen3 is distributed in BF16; Ampere supports it). |
| Wrong RoPE layout or missing QK-Norm | Fluent-looking nonsense | The tiny-model logit test (M3) catches it. |
| Batch-dependent numerical differences | "Batched output differs from single" on GPU | Expected in BF16; test exact match in FP32 on CPU, report rates on GPU (section 11). |
| Broken multi-byte text in streams | `�` characters with emoji or Chinese prompts | Incremental detokenizer with offsets; test it (M7). |
| Leaked KV blocks | Free blocks shrink over time; eventual stalls | Leak tests after aborts and preemption (M6, M7). |
| Engine thread fighting the API thread for the GIL | ITL spikes when many clients connect | Keep API handlers light; measure; level-up to a separate engine process. |
| Laptop thermal throttling | Throughput drops across repeated runs; SM clock falls | Log clocks; cool down between runs; report the median of 3. |
| Unfair comparison with vLLM | vLLM looks impossibly far ahead (or behind) | Compare eager to eager; match prefix caching and memory settings; same client and workloads (10.1). |
| Qwen3 "thinking" output | `<think>` text in chat responses | `enable_thinking=False` in the chat template for benchmarks. |
| Slow file access | Model loading takes far longer than expected | Keep repo and models in WSL's own file system (`~/`), not `/mnt/c`. |
| Scope creep | Three half-built features | Milestone exit criteria are gates; the M0–M7 + M12 cut is a complete project. |
| Disk space | WSL virtual disk grows | Models need ~5.5 GB total; clear old `results/` traces and pip caches; check free space monthly. |

---

## 13. Honesty rules for results and the resume

- **Every claim traces to a saved run.** If a number isn't in `docs/RESULTS.md`, it doesn't go in the README, resume or posts.
- **State the conditions with the number:** "on Qwen3-0.6B (BF16), an RTX 3060 Laptop GPU (6 GB), 256-token prompts and 128-token outputs, 16 concurrent requests".
- **"From scratch" means:** my own scheduler, block manager, model code, API server and benchmark harness, using PyTorch. If I use the flash-attn library's kernels, say "using FlashAttention-2 kernels". Only say "my own CUDA/Triton kernel" if I wrote one.
- **Don't say "production"**; say "a server benchmarked under realistic load".
- **Comparisons with vLLM name the configuration** (default or `--enforce-eager`) and the vLLM version.
- **Negative results are reported**, for example "speculative decoding slowed things down at concurrency 16". They show judgment and make the positive results believable.
- **Credit references:** the README credits the vLLM and Sarathi-Serve papers, nano-vllm, and any code adapted under its licence.

---

## 14. Resume bullets, write-up and the 60-second story

### 14.1 Resume entries (fill the brackets only from `docs/RESULTS.md`)

**Project header:**

> **MiniServe — LLM Inference Server from Scratch** | GitHub | *Python, PyTorch, CUDA, FastAPI*

**ML-systems version (for inference, ML infrastructure and AI-platform roles):**

- Built an LLM inference server in Python and PyTorch with continuous batching, a paged KV cache, prefix caching, chunked prefill and CUDA graphs, serving Qwen3 through an OpenAI-compatible streaming API; reached **[X] output tokens/s ([N]× a Hugging Face baseline, [P]% of vLLM)** on a 6 GB RTX 3060.
- Raised maximum concurrent requests from **[a] to [b]** by replacing fixed KV reservations with 16-token paged blocks, and cut P99 inter-token latency **[c]%** under long-prompt load with chunked prefill.
- Added n-gram speculative decoding with a rejection sampler proven exact by tests (**[A]% acceptance, [S]× lower time per token** on copy-heavy prompts) and INT8 weight-only quantization (**[M] GB saved, perplexity +[d]**).

**Backend / general SWE version:**

- Built a streaming LLM API server (FastAPI, SSE, OpenAI-compatible) with a custom batch scheduler and memory manager, serving **[K] concurrent streams at P99 time-per-token under [T] ms** on a single 6 GB GPU; benchmarked against vLLM with its own load-testing client.
- Verified correctness with property-based tests (Hypothesis), logit-level comparisons against Hugging Face and a statistical test of the sampler, all running in GitHub Actions CI.

**Keywords this project legitimately adds (once built):** LLM inference, LLM serving, continuous batching, KV cache, PagedAttention, prefix caching, chunked prefill, speculative decoding, quantization (INT8), CUDA graphs, FlashAttention, Triton (only if built), PyTorch, GPU profiling, TTFT, TPOT, P99 latency, throughput, OpenAI-compatible API, SSE streaming, vLLM.

### 14.2 The write-up (hireahraz.dev; publish only after I approve)

Working title: **"Building an LLM inference server on a 6 GB laptop GPU: what each technique actually buys"**

1. The problem in one picture: a GPU sitting idle while serving one request.
2. The two numbers that explain everything: 112 KiB of KV cache per token, and 3.5 ms just to read the weights.
3. Continuous batching: before/after chart.
4. Paged KV: max concurrency before/after.
5. Prefix caching and chunked prefill: TTFT and ITL charts.
6. CUDA graphs: profiler timelines before/after.
7. Speculative decoding: where it helped and where it hurt.
8. Quantization: memory vs speed vs perplexity.
9. MiniServe vs vLLM: where the remaining gap comes from.
10. What I'd do next.

### 14.3 The 60-second story (practise out loud)

> "I wanted to understand LLM inference properly, so I built an inference server from scratch for a 6 GB laptop GPU and benchmarked it against vLLM with vLLM's own client. The key insight is that decode is memory-bound: reading the weights takes about 3.5 ms per step whether you serve one request or thirty, so batching is nearly free until memory runs out, and on a small GPU what runs out is the KV cache, at 112 KiB per token. So I built continuous batching and a paged KV cache, which took me from **[a]** to **[b]** concurrent requests and **[N]×** the throughput of a Hugging Face baseline, ending at **[P]%** of vLLM. The surprise was **[from LEARNING_LOG.md, e.g. how much time went to CPU overhead before CUDA graphs]**."

---

## 15. Interview questions with answers

Short answers to learn from; each should end with a number or an example from my own results once I have them.

1. **Why is decoding memory-bound?** Each step reads every weight (1.19 GB for Qwen3-0.6B) but does only about 2 FLOPs per weight per sequence. At 336 GB/s that's ~3.5 ms of pure reading per step, while the math takes far less. Batching shares one read of the weights across many sequences.
2. **What's in the KV cache and how big is it?** The keys and values of every past token at every layer. For Qwen3-0.6B: 2 × 28 layers × 8 KV heads × 128 dims × 2 bytes = 112 KiB per token, so 3 GiB holds ~28,000 tokens.
3. **Continuous vs static batching?** Static batching runs a fixed group until all finish, so short requests wait for long ones and new ones wait for the group. Continuous batching re-decides the batch every step: finished sequences leave, new ones join, and the GPU stays full.
4. **What problem does PagedAttention solve?** Over-reservation and fragmentation of KV memory. Fixed-size blocks allocated on demand through a block table waste at most one partial block per sequence, so many more sequences fit.
5. **How do you choose a block size?** Small blocks waste less memory and allow finer prefix sharing; large blocks mean smaller block tables and more efficient kernel reads. vLLM uses 16; FlashAttention-2's paged mode needs multiples of 256.
6. **What happens when KV memory runs out?** Preempt: evict the newest running sequence. Either free its blocks and recompute it later (vLLM V1), or swap its blocks to CPU memory (V0). Recompute is simpler and usually cheaper for short sequences.
7. **How does prefix caching work, and when can't you reuse a block?** Hash each full block together with the previous block's hash, so a hit means the entire prefix matches token for token. Partial blocks, or the same text at a different position, can't be reused. Freed cached blocks are kept in an LRU and evicted only under pressure.
8. **What's the trade-off in chunked prefill?** A big prefill in one step stalls every decoding user. Chunking under a token budget keeps ITL smooth, but the long prompt's TTFT grows a little and prefill efficiency drops slightly.
9. **TTFT vs TPOT: which matters?** Chat cares about TTFT (responsiveness) and steady TPOT (reading speed); offline batch jobs care about throughput. Goodput (requests meeting both SLOs per second) combines them.
10. **Why does latency rise with batch size, and where do you stop?** Each step does more work, so it takes longer, even though total throughput rises. Stop where the P99 TPOT or TTFT SLO is about to be broken: the knee of the latency–throughput curve.
11. **Why is speculative decoding exact?** Each draft token x is accepted with probability min(1, p(x)/q(x)), and on rejection we sample from the normalized max(0, p − q). The combination provably gives exactly the target distribution.
12. **When doesn't speculative decoding help?** At large batch sizes (no idle GPU capacity left), when acceptance is low (the draft disagrees with the target), or when the draft itself is too slow. It helps most for a single user, and for outputs that copy the input.
13. **What is GQA and why does it matter for serving?** Grouped-query attention lets several query heads share one KV head. Qwen3 has 16 query heads and 8 KV heads, which halves the KV cache and the memory read per decode step.
14. **Why does each token carry its own position in a flattened batch?** RoPE rotates queries and keys by their position within their own sequence. In a flat batch of mixed sequences, positions aren't simply 0…T−1.
15. **What does FlashAttention optimize?** Memory traffic. It tiles the computation so the n × n score matrix never leaves on-chip SRAM. It's exact, and the speed-up comes from moving fewer bytes.
16. **What are CUDA graphs and why do they need fixed shapes?** A recording of a sequence of kernel launches, replayed with a single launch to remove CPU overhead. The recording bakes in tensor shapes and memory addresses, so you capture a few batch sizes and pad to the nearest one.
17. **Weight-only quantization: why can INT4 be faster for decode?** Decode is bound by reading weights; INT4 reads a quarter of the bytes of BF16. That only pays off with a fused dequantize-and-matmul kernel. Always measure quality: perplexity plus task agreement.
18. **How do you measure P99 properly?** Enough samples (≥ 500 requests), warm-up excluded, repeated runs, open-loop arrivals for realistic queueing, and the same client for every system. Report the conditions.
19. **How would you serve a 70B model?** Tensor parallelism across GPUs in one node (fast interconnect), pipeline parallelism across nodes if needed, quantization to cut memory, possibly disaggregated prefill and decode, and prefix-aware routing across replicas.
20. **How do you keep the server fair and stable under overload?** First come, first served with preemption of the newest; caps on `max_tokens` and the waiting queue; 503s instead of unbounded queues; abort on disconnect so dead clients don't hold memory.
21. **SSE or WebSockets for token streaming?** SSE: one-way server-to-client over plain HTTP, works through proxies, and it's what the OpenAI API uses. WebSockets only if the client needs to send mid-stream.
22. **Why compute logits only for the last token during prefill?** Logits for every prompt token are useless for generation and huge: 2,048 tokens × 151,936 vocabulary × 2 bytes = 622 MB.
23. **How did you know your engine was correct?** Logit comparisons with Hugging Face on a tiny model in CI; exact-match tests for batched vs single, paged vs contiguous, cache on vs off, speculation on vs off; property-based tests for the block manager; a chi-square test for the sampler.
24. **Why is your server slower than vLLM?** Fill in from measurements. Likely candidates: kernel efficiency, Python overhead in the scheduler, no CPU/GPU overlap, the engine running in a thread instead of a separate process, and vLLM's tuned attention backends.
25. **What would you do next?** Section 16: a Triton paged-attention kernel, a separate engine process with overlap scheduling, and an upstream contribution to vLLM.

---

## 16. Level-ups after v1.0

Pick **one** at a time, finish it, measure it, update the README and resume bullet.

| Level-up | What | Why it impresses | Effort |
|---|---|---|---|
| **Triton paged-attention decode kernel** | My own kernel with 16-token blocks, benchmarked against FlashAttention-2 and `TorchRefBackend` | Real GPU programming; lets me honestly write "Triton" | +2 weekends |
| **Separate engine process + overlap scheduling** | Engine in its own process (ZeroMQ, as vLLM does); prepare step N+1 on the CPU while the GPU runs step N (as SGLang does) | Closes much of the CPU-overhead gap; a deep systems topic | +2 weekends |
| **Upstream pull request** | A merged fix or feature in vLLM or SGLang (start from "good first issue") | The strongest single signal on a new-grad resume | Varies |
| **INT8 KV cache** | Store K/V as INT8 with scales | Doubles concurrency on 6 GB; measurable | +1 weekend |
| **Prefix-aware router** | Two MiniServe processes behind a router that sends shared prefixes to the same replica | Distributed-systems angle | +1 weekend |
| **Structured outputs** | JSON-schema-constrained decoding with token masks | Practical for agents (TriageBench) | +1 weekend |
| **Observability** | Prometheus metrics + a Grafana dashboard (queue depth, KV usage, TTFT/TPOT histograms) | Production thinking | +1 weekend |
| **Bigger GPU run** | The same benchmark on a rented cloud GPU (L4 or A10G class) | Shows scaling. **Ask me first; set a budget alert and auto-shutdown.** | Half a day |

**Connecting projects:** keep the OpenAI-compatible API, so TriageBench (project 3 in my plan) can use MiniServe as its model server: "I built the inference server, the retrieval engine and the agent on top, and measured each layer."

---

## 17. Reading list, in order

**Week 1: foundations**

1. Aleksa Gordić, "Inside vLLM: Anatomy of a High-Throughput LLM Inference System" (vLLM blog, Sep 2025): https://vllm.ai/blog/2025-09-05-anatomy-of-vllm — the single best overview; reread it at M5 and M8.
2. kipply, "Transformer Inference Arithmetic": https://kipp.ly/blog/transformer-inference-arithmetic/ — the memory and FLOPs math.
3. Horace He, "Making Deep Learning Go Brrrr From First Principles": https://horace.io/brrr_intro.html — compute vs memory vs overhead.
4. nano-vllm source code: https://github.com/GeeeekExplorer/nano-vllm — read all of it (about 1,200 lines).
5. Qwen3 Technical Report: https://arxiv.org/abs/2505.09388 — section 2 (architecture) only.

**Weeks 2 to 4: batching and memory (M5, M6)**

6. Anyscale, "How continuous batching enables 23x throughput in LLM inference": https://www.anyscale.com/blog/continuous-batching-llm-inference
7. Orca (OSDI 2022), iteration-level scheduling: https://www.usenix.org/conference/osdi22/presentation/yu
8. Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention" (SOSP 2023): https://arxiv.org/abs/2309.06180
9. vLLM V1 announcement: https://vllm.ai/blog/2025-01-27-v1-alpha-release.html

**Weeks 5 to 7: scheduling, kernels, overhead (M8, M9)**

10. Sarathi-Serve (OSDI 2024), chunked prefill: https://arxiv.org/abs/2403.02310
11. SGLang (RadixAttention): https://arxiv.org/abs/2312.07104
12. FlashAttention: https://arxiv.org/abs/2205.14135 and FlashAttention-2: https://arxiv.org/abs/2307.08691 ; repository: https://github.com/Dao-AILab/flash-attention
13. Mini-SGLang blog (overlap scheduling): https://lmsys.org/blog/2025-12-17-minisgl
14. PyTorch docs: CUDA Graphs (`torch.cuda.graph`) and `torch.profiler`.

**Weeks 8 to 10: speculation and quantization (M10, M11)**

15. Leviathan et al., "Fast Inference from Transformers via Speculative Decoding": https://arxiv.org/abs/2211.17192
16. Chen et al., "Accelerating Large Language Model Decoding with Speculative Sampling": https://arxiv.org/abs/2302.01318
17. Prompt lookup decoding: https://github.com/apoorvumang/prompt-lookup-decoding
18. EAGLE: https://arxiv.org/abs/2401.15077 and Medusa: https://arxiv.org/abs/2401.10774 (know the ideas)
19. LLM.int8(): https://arxiv.org/abs/2208.07339 , GPTQ: https://arxiv.org/abs/2210.17323 , AWQ: https://arxiv.org/abs/2306.00978

**Interview depth (any time)**

20. DistServe (goodput, disaggregated prefill/decode): https://arxiv.org/abs/2401.09670
21. Pope et al., "Efficiently Scaling Transformer Inference": https://arxiv.org/abs/2211.05102
22. GQA: https://arxiv.org/abs/2305.13245 ; RoPE (RoFormer): https://arxiv.org/abs/2104.09864
23. Thinking Machines, "Defeating Nondeterminism in LLM Inference" (Sep 2025), summarized with a link at https://simonwillison.net/2025/Sep/11/defeating-nondeterminism
24. Triton tutorial, "Fused Attention": https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html (for the level-up)
25. GPU MODE lectures (CUDA and Triton from scratch): https://github.com/gpu-mode/lectures

**Docs to keep open:** vLLM `bench serve` CLI (https://docs.vllm.ai/en/latest/cli/bench/serve.html), vLLM speculative decoding (https://docs.vllm.ai/en/latest/features/speculative_decoding/), NVIDIA CUDA on WSL user guide, uv's PyTorch guide.

---

## 18. Glossary

| Term | Meaning |
|---|---|
| Token | A piece of text (word, part of a word, symbol) mapped to an integer ID. |
| Logits | Raw scores the model outputs for every vocabulary entry; softmax turns them into probabilities. |
| Autoregressive | Generating one token at a time, each conditioned on all previous ones. |
| Prefill | Processing the prompt in one forward pass; produces the first token. Compute-bound. |
| Decode | Generating each following token, one step per token. Memory-bound. |
| KV cache | Stored keys and values of past tokens so they aren't recomputed. |
| GQA | Grouped-query attention: several query heads share one key/value head. |
| RoPE | Rotary position embeddings: encode position by rotating query and key vectors. |
| RMSNorm | A normalization that rescales by the root-mean-square of the vector. |
| SwiGLU | The gated MLP used in Qwen and Llama: `down(silu(gate(x)) * up(x))`. |
| BF16 | 16-bit floating point with FP32's exponent range; safer than FP16 for LLMs. |
| VRAM / HBM | The GPU's main memory (GDDR6 on my laptop, HBM on data-center GPUs). |
| Memory bandwidth | Bytes per second the GPU can read from VRAM (336 GB/s spec for mine). |
| Arithmetic intensity | FLOPs performed per byte read; low means memory-bound. |
| Roofline | The model saying performance is capped by min(peak compute, bandwidth × intensity). |
| Continuous batching | Changing batch membership at every decoding step. |
| Block / page | A fixed-size chunk of KV memory (e.g., 16 tokens). |
| Block table | Per-sequence list mapping logical blocks to physical blocks. |
| Slot mapping | For each new token, the index in the KV pool where its K/V is written. |
| Prefix caching | Reusing KV blocks for identical prompt prefixes. |
| Chunked prefill | Splitting a long prefill across steps under a token budget. |
| Preemption | Evicting a running sequence when memory runs out (recompute or swap). |
| Speculative decoding | Drafting several tokens cheaply and verifying them in one target pass. |
| Acceptance rate (α) | Fraction of drafted tokens the target accepts. |
| Quantization | Storing numbers in fewer bits (INT8, INT4) to save memory and bandwidth. |
| Kernel | A function that runs on the GPU. |
| Kernel launch overhead | CPU time to start each kernel; matters when kernels are tiny. |
| CUDA graph | A recorded sequence of kernel launches replayed as one launch. |
| Triton | A Python-like language for writing GPU kernels. |
| FlashAttention | An exact attention algorithm that avoids writing the score matrix to VRAM. |
| SSE | Server-Sent Events: a one-way HTTP stream of `data:` lines. |
| TTFT | Time to first token. |
| TPOT | Time per output token after the first. |
| ITL | Inter-token latency: gaps between streamed outputs. |
| E2EL | End-to-end latency of a request. |
| Goodput | Requests per second that meet latency targets (SLOs). |
| P50 / P90 / P99 | Percentiles: the value below which 50% / 90% / 99% of measurements fall. |
| Open / closed loop | Load by arrival rate vs by fixed number of concurrent clients. |
| Little's law | Concurrency = throughput × latency. |
| Tensor parallelism | Splitting each weight matrix across GPUs. |
| Disaggregated serving | Running prefill and decode on separate GPU pools. |

---

## 19. Timeline and checklist

About **14 to 15 focused weekends** for v1.0 (roughly 7 weeks full-time), plus level-ups.

- [ ] M0 Environment and GPU facts (0.5)
- [ ] M1 Naive HF baseline with `/v1/completions` (1)
- [ ] M2 Load generator, cross-checked with `vllm bench serve` (1)
- [ ] M3 My own Qwen3, logit-exact on a tiny model (1)
- [ ] M4 Single-sequence KV cache (0.5)
- [ ] M5 Continuous batching, fixed KV slots (2)
- [ ] M6 Paged KV cache, preemption, leak tests (2)
- [ ] M7 Full OpenAI API, streaming, abort, backpressure (1)
- [ ] **Checkpoint:** README v0.5 with results vs baseline and vLLM; update the resume bullet (this is already a complete project)
- [ ] M8 Prefix caching and chunked prefill (1.5)
- [ ] M9 CUDA graphs and overhead cuts (1)
- [ ] M10 Speculative decoding with exactness tests (1.5)
- [ ] M11 INT8 quantization with quality numbers (1)
- [ ] M12 Full benchmark matrix, README v1.0, write-up draft (1)
- [ ] Ask Ahraz before: making the repo public, publishing the write-up, any paid cloud run

---

## 20. Starter prompt for the new chat

Copy this into the new chat together with this file:

> I'm attaching `MiniServe_README.md`, the full plan for my next project: an LLM inference server built from scratch on my RTX 3060 laptop GPU. Please read sections 0 to 4 first and follow the instructions in section 1 for the whole project (teach me as we go, one milestone at a time, design note → code → tests → measure, never invent numbers, ask before anything public or paid). Then start **M0**: walk me through each step with the exact commands for my WSL2 Ubuntu 24.04 setup, tell me what output to expect, and wait for me to paste my results before moving on. If anything in the plan looks outdated (library versions, flags, APIs), check the current docs and tell me.

---

## 21. Facts checked for this brief

Checked on 2026-11-07 against the sources below. Anything not listed here is general knowledge or marked as an estimate in the text.

| Fact | Source |
|---|---|
| Qwen3-0.6B config: 28 layers, 16 Q / 8 KV heads, head_dim 128, hidden 1,024, MLP 3,072, vocab 151,936, tied embeddings, RoPE theta 1,000,000, QK-Norm | Model config, as reproduced in a Qwen3-0.6B study note; Qwen3 Technical Report, Table 1 (layers, heads, tied embeddings) and its note that QKV-bias was removed and QK-Norm added |
| Qwen3-1.7B config: hidden 2,048, MLP 6,144, 28 layers, 16 / 8 heads, head_dim 128, tied | A config.json copy of Qwen3-1.7B on Hugging Face; Qwen3 Technical Report, Table 1 |
| RTX 3060 Laptop GPU: 6 GB GDDR6, 192-bit, 336 GB/s, 3,840 FP32 ALUs, 60–115 W | Notebookcheck spec page |
| vLLM v0.33.0 released 2026-11-01; v0.32.0 in October 2026 dropped Python 3.10 and narrowed transformers to 5.16.1–5.20.0 | Release tracker for vllm-project/vllm; v0.32.0 release summary |
| vLLM runs on Linux including WSL (not native Windows), needs compute capability 7.0+, installs with `uv pip install vllm --torch-backend=auto` | vLLM installation docs (mirror) |
| vLLM V1: 16-token blocks by default, prefix caching on by default, chunked prefill, recompute preemption (swap was V0 only), CUDA graphs unless `--enforce-eager` | "Inside vLLM: Anatomy of a High-Throughput LLM Inference System" |
| vLLM speculative decoding docs list draft models, n-gram, EAGLE, MTP and suffix decoding | vLLM v0.31.0 docs navigation |
| `vllm bench serve` flags used in section 10 (incl. `--ignore-eos`, `--tokenizer`, `--goodput`, `--metric-percentiles`, `prefix_repetition`) and metric definitions (TTFT, TPOT, ITL, E2EL) | vLLM `bench serve` CLI docs and benchmarking docs (v0.32.0) |
| nano-vllm: ~1,200 lines, MIT, features, RTX 4070 Laptop benchmark numbers | nano-vllm GitHub README |
| Mini-SGLang: ~5,000 lines, features, Hopper-tuned kernels | LMSYS blog, 2025-12-17 |
| FlashAttention-2 supports Ampere/Ada/Hopper in FP16/BF16, CUDA ≥ 12.0, PyTorch ≥ 2.2, paged KV with page size a multiple of 256; FA3 needs H100 | Dao-AILab flash-attention README |
| Sarathi-Serve is OSDI 2024 (arXiv 2403.02310); Orca is OSDI 2022 | arXiv and USENIX pages |

**Computed for this brief** (Python, from the configs above): parameter counts and BF16 weight sizes; 112 KiB of KV cache per token; tokens per GiB; fixed-slot counts; speed-limit table; logits sizes; INT8/INT4 weight sizes; speculative-decoding expected tokens per step.
