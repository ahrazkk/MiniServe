# MiniServe: notes for Claude

MiniServe is Ahraz's from-scratch LLM inference server (Python + PyTorch, Qwen3-0.6B on an RTX 3060
Laptop GPU, 6 GB). `docs/PLAN.md` is the source of truth: read its sections 1 to 4 before changing
anything, and follow its rules. The learning guide lives in `docs/learn/MiniServe_Textbook.html`.

## Non-negotiable rules (from docs/PLAN.md section 1)

1. **Never invent a benchmark number.** Every number in README.md, docs/RESULTS.md, the textbook or
   a resume line comes from a JSON file in `results/`, with GPU, model, dtype, settings and date.
   Estimates are labelled "estimate"; computed values are labelled "computed".
2. **Ask Ahraz first** before: spending money, any paid cloud GPU, pushing to a public repo,
   publishing a post. Commit only when asked.
3. **Write it ourselves.** Reference projects (vLLM, nano-vllm, SGLang) are for ideas; credit them.
4. **Teach while building.** Every milestone: design note, small code, tests, measure, learning log,
   then 3 interview questions. Explain mechanisms with a number (bytes, ms, FLOPs, tokens).
5. **Don't touch ARIS**: `/root/aris_ws`, `/root/aris_px4_ws`, `/root/PX4-Autopilot` live in the
   same WSL install.

## Environment

- Windows 11 + WSL2 `Ubuntu-24.04` (commands run as root). The repo is on the Windows drive so the
  app and editors see it; the venv (`/root/.venvs/miniserve`) and the weights
  (`/root/models/Qwen3-0.6B`) live on WSL's own disk because `/mnt/c` is slow for them.
- Run anything inside the environment from PowerShell: `.\scripts\wsl.ps1 <command>`.
  PowerShell 5.1 mangles nested quotes in `wsl ... bash -c '...'`; put multi-step shell logic in a
  script file under `scripts/` instead.
- Long jobs (benchmarks) must outlive the terminal: launch with
  `Start-Process wsl.exe -ArgumentList "-d Ubuntu-24.04 -u root -- bash /mnt/c/Users/1kibr/Desktop/ResumeProjects/MiniServe/scripts/run_detached.sh <log> <command>" -WindowStyle Hidden`
  and check progress with `scripts/bench_status.sh`.
- `pkill -f "miniserve serve"` inside `bash -c` matches its own shell; use `pgrep -af "[m]iniserve"`.
- **Do not run CPU-heavy work in WSL while a benchmark runs.** On WSL2 each kernel launch costs
  ~28 us and the engine is CPU-launch-bound, so extra CPU load changes the numbers.

## Commands (run through `.\scripts\wsl.ps1` from PowerShell, or `bash scripts/dev.sh` in WSL)

| What | Command |
|---|---|
| CPU tests (tiny float64 model, ~15 s) | `python -m pytest -q` |
| GPU tests vs Hugging Face | `python -m pytest -q -m gpu -s` |
| Lint / types | `ruff check src tests bench` / `mypy` |
| Serve | `miniserve serve --port 8000` (`--help` lists every engine flag) |
| Naive baseline | `miniserve baseline --port 8000` |
| Offline check | `miniserve generate --prompt "Hi"` |
| Benchmark one system | `python bench/run_matrix.py --system miniserve --tag NAME --concurrency 1 16 32` |
| Full v0.5 matrix | `bash scripts/bench_v05.sh` (about an hour; run it detached) |
| Step time vs batch | `python bench/step_time.py` |
| Profile a step | `python bench/profile_step.py --batch 8` |

## Code map

`src/miniserve/`: `config.py` (every knob) -> `engine/` (sequence, block_manager, scheduler,
engine + AsyncEngine thread) -> `runner/` (model_runner: KV pool, flat batch, sampling) ->
`model/` (Qwen3, loader) + `attention/backends.py` (paged attention in PyTorch) -> `spec/`
(n-gram drafts, rejection sampler) -> `quant/int8.py` -> `api/` (OpenAI server, detokenizer) ->
`baseline/hf_server.py` (M1). Benchmarks in `bench/`, results in `results/`, docs in `docs/`.

## Skills in this repo

`.claude/skills/`: `miniserve-milestone` (the per-milestone loop), `miniserve-benchmark`
(measuring and recording numbers), `miniserve-tutor` (teaching and quizzing), plus installed
community skills: `fastapi` (official), `python-testing-patterns`, `pytorch-patterns`,
`code-tour`, `teach` and `grill-me`/`grilling` (Matt Pocock). Versions are pinned in
`skills-lock.json`.
