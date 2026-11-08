#!/usr/bin/env bash
# One-time WSL setup for MiniServe: uv, a Python 3.12 venv with CUDA PyTorch, and the Qwen3-0.6B weights.
# Safe to re-run: steps that are already done are skipped or are no-ops.
#
# The repo can live on the Windows drive (/mnt/c/...) so Windows tools can see it, but the venv and
# the model weights live on WSL's own disk, because reading thousands of small files (torch) or a
# 1.5 GB weights file through /mnt/c is slow.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${MINISERVE_VENV:-/root/.venvs/miniserve}"
MODELS="${MINISERVE_MODELS:-/root/models}"
export PATH="$HOME/.local/bin:$PATH"

step() { printf '\n=== %s ===\n' "$*"; }

step "uv"
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
uv --version

step "virtual environment at $VENV"
if [ ! -x "$VENV/bin/python" ]; then
    uv venv "$VENV" --python 3.12
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

step "PyTorch (the default Linux wheel on PyPI is a CUDA build)"
uv pip install torch

step "MiniServe itself (editable) and the dev tools"
uv pip install -e "$REPO[dev]"

step "Does PyTorch see the GPU?"
python - <<'PY'
import torch
print("torch", torch.__version__, "| cuda", torch.version.cuda, "| available", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0), "compute capability", torch.cuda.get_device_capability(0))
PY

step "Qwen3-0.6B weights into $MODELS/Qwen3-0.6B"
mkdir -p "$MODELS"
python - "$MODELS/Qwen3-0.6B" <<'PY'
import sys
from huggingface_hub import snapshot_download
path = snapshot_download("Qwen/Qwen3-0.6B", local_dir=sys.argv[1])
print("weights at", path)
PY
du -sh "$MODELS/Qwen3-0.6B"

step "done"
