#!/usr/bin/env bash
# Clean rerun of the MiniServe W1 sweep with a distinct prompt set per run (seed = 1000*c + run).
# Supersedes results/miniserve_in256_out128_* from bench_v05.sh, where every run used seed 0 and
# later runs on the same server got prefix-cache hits on prompts repeated from earlier runs.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python bench/run_matrix.py --system miniserve --tag ms --concurrency 1 4 16 64
python bench/run_matrix.py --system miniserve --tag ms --concurrency 32 --runs 3
echo "CLEAN SWEEP DONE"
