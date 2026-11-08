#!/usr/bin/env bash
# Run any command inside the MiniServe WSL environment, from the repo root.
#
#   From PowerShell:  .\scripts\wsl.ps1 pytest -q
#   Inside WSL:       bash scripts/dev.sh pytest -q
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${MINISERVE_VENV:-/root/.venvs/miniserve}"

export VIRTUAL_ENV="$VENV"
export PATH="$VENV/bin:$HOME/.local/bin:$PATH"
export MINISERVE_MODEL="${MINISERVE_MODEL:-/root/models/Qwen3-0.6B}"
export PYTHONUNBUFFERED=1

cd "$REPO"
exec "$@"
