---
name: miniserve-milestone
description: Use when starting, building or finishing a MiniServe milestone or level-up (M8 to M12, CUDA graphs, Triton kernel, separate engine process, vLLM comparison). Runs the design-note, code, test, measure, learning-log, quiz loop from docs/PLAN.md.
---

# MiniServe milestone loop

MiniServe is a learning project as much as a product: Ahraz must be able to explain and modify
every line in an interview. Work one milestone at a time, in this loop, and don't start the next
until the current one's exit criteria (docs/PLAN.md section 9) pass.

## 1. Design note first

Write half a page in `docs/design/DESIGN_NOTES.md` (append a section): what we build, why, the
interfaces it touches, how it will be tested, and the number it should move. Show it to Ahraz
and get a yes before coding anything large.

## 2. Code in small, readable pieces

- Match the existing style: module docstrings that teach the idea, comments that explain *why*,
  type hints everywhere, no clever one-liners.
- Every feature gets an `EngineConfig` flag so it can be switched off for the ablation table.
- Prefer one clear implementation over a fast obscure one; optimize only after a profile shows it.

## 3. Tests before "done"

- CPU, in CI: the tiny float64 Qwen3 (`tests/conftest.py`). New engine features must keep
  `tests/correctness/test_engine_equivalence.py` exact: add the new flag to its parameter list.
- Invariants with Hypothesis when there is state (blocks, queues).
- GPU, local: `python -m pytest -m gpu -s` against the real model.
- Commands: `.\scripts\wsl.ps1 python -m pytest -q` and `... -m gpu -s`; `ruff check src tests bench`.

## 4. Measure, then record

Use the `miniserve-benchmark` skill. Every number goes into `docs/RESULTS.md` with its settings
and the result file it came from. No number, no claim.

## 5. Learning log and the textbook

- Two or three lines in `docs/LEARNING_LOG.md`: what surprised us, with the number.
- Add or update the milestone's chapter in `docs/learn/MiniServe_Textbook.html` (use the
  `miniserve-tutor` skill for the teaching style).

## 6. Quiz

End with three interview-style questions on what was just built (docs/PLAN.md section 15 has
model answers). Ask them one at a time and wait for Ahraz's answer before giving feedback.
