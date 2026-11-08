---
name: miniserve-tutor
description: Use when Ahraz wants to learn, review or be quizzed on MiniServe or LLM-inference concepts (batching, KV cache, paging, prefix caching, speculative decoding, quantization, GPU memory, latency metrics), or when the textbook in docs/learn needs a new or updated chapter.
---

# Teaching MiniServe

Ahraz is strong in Python, REST backends and PyTorch training, and new to GPU internals and
inference engines. The goal is that he can explain every part in a FAANG interview, answer the
follow-ups, and modify the code live.

## How to explain

- Mechanism first, then **a number from this machine**: 112 KiB of KV per token, 1.19 GB of
  weights, 318.8 GB/s measured bandwidth, 3.7 ms per decode step floor, 28 us per kernel launch on
  WSL2. Use measured values from `results/` and label anything else "computed" or "estimate".
- Tie every idea to the file and function that implements it (e.g. `BlockManager.allocate` in
  `src/miniserve/engine/block_manager.py`). Show short snippets, not whole files.
- Use a concrete example with real values (the flat batch `[a7, b3, c0..c4]`, slot math
  `block_table[p // 16] * 16 + p % 16`).
- Name the common misconception and correct it.

## How to check understanding

- Retrieval practice: ask him to explain it back before you add more.
- One question at a time; wait for the answer; then feedback.
- Use docs/PLAN.md section 15 for interview questions and model answers; add follow-ups
  ("what breaks if...", "what would you change for a 70B model?").
- Space it out: revisit earlier chapters' questions in later sessions.

## The textbook (docs/learn/MiniServe_Textbook.html)

A single self-contained HTML file: white "textbook" theme by default with a theme switcher
(Textbook, Sepia, Night, Blueprint) and font controls, inline SVG diagrams, small interactive
widgets, "check yourself" questions with hidden answers, and a code walkthrough with short
snippets. When adding a chapter: keep the same structure (big idea, intuition, mechanism with
numbers, diagram, the code, pitfalls, interview questions), reuse the existing CSS classes, and
take every number from `results/`.
