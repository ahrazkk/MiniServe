#!/usr/bin/env bash
# The rest of the v0.5 matrix, trimmed: these comparisons only need medians (no P99 claims),
# so they use fewer prompts. W1 = 256-token prompts, 128 output tokens.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Paging ablation: the same engine reserving max_model_len (4,096 tokens) per request up front.
python bench/run_matrix.py --system miniserve --tag fixed-slots --concurrency 16 64 \
    --num-prompts 100 --server-args "--kv-reservation full --no-prefix-caching"

# The naive baseline: Hugging Face generate(), one request at a time.
python bench/run_matrix.py --system hf-baseline --tag hf-baseline --concurrency 1 4 16 \
    --num-prompts 48

# W3, shared prefix: 1,024-token shared prefix + 128 unique tokens, 128 out; cache on vs off.
python bench/run_matrix.py --system miniserve --tag prefix-on --concurrency 16 --num-prompts 100 \
    --input-len 128 --loadgen-args "--dataset shared-prefix --prefix-len 1024"
python bench/run_matrix.py --system miniserve --tag prefix-off --concurrency 16 --num-prompts 100 \
    --input-len 128 --loadgen-args "--dataset shared-prefix --prefix-len 1024" \
    --server-args="--no-prefix-caching"   # '=': a lone flag-like value would be parsed as a flag
echo "REST DONE"
