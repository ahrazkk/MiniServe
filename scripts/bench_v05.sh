#!/usr/bin/env bash
# The v0.5 benchmark matrix (W1: 256-token prompts, 128 output tokens, closed loop).
# One server at a time on the GPU; each run_matrix call starts and stops its own server.
#   bash scripts/dev.sh bash scripts/bench_v05.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
rm -f results/smoke*.json results/logs/smoke*

# MiniServe, default config: paged KV, prefix caching, chunked prefill.
python bench/run_matrix.py --system miniserve --tag miniserve --concurrency 1 4 16 64
python bench/run_matrix.py --system miniserve --tag miniserve --concurrency 32 --runs 3

# Ablation: fixed KV slots (reserve max_model_len per request, the pre-PagedAttention way).
python bench/run_matrix.py --system miniserve --tag fixed-slots --concurrency 16 32 64 \
    --num-prompts 300 --server-args "--kv-reservation full --no-prefix-caching"

# The naive baseline: Hugging Face generate(), one request at a time.
python bench/run_matrix.py --system hf-baseline --tag hf-baseline --concurrency 1 4 16 \
    --num-prompts 64

# W3, shared prefix: 1,024-token shared prefix + 128 unique tokens, 128 out; cache on vs off.
python bench/run_matrix.py --system miniserve --tag prefix-on --concurrency 16 --num-prompts 200 \
    --input-len 128 --loadgen-args "--dataset shared-prefix --prefix-len 1024"
python bench/run_matrix.py --system miniserve --tag prefix-off --concurrency 16 --num-prompts 200 \
    --input-len 128 --loadgen-args "--dataset shared-prefix --prefix-len 1024" \
    --server-args "--no-prefix-caching"
echo "ALL BENCHMARKS DONE"
