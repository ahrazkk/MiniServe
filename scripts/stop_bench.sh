#!/usr/bin/env bash
# Stop a running benchmark sweep: its sweep runner, load generator, server and GPU monitor,
# then the sweep script itself. The bracket in each pattern stops pkill from matching itself.
# run_detached.sh is left alone so it can log "EXIT CODE", which releases any job queued
# behind it with after.sh.
for pat in "[r]un_matrix.py" "[b]ench/loadgen.py" "[m]iniserve (serve|baseline)" \
           "[n]vidia-smi --query-gpu=timestamp" "^bash scripts/[b]ench_v05(_clean|_rest)?.sh"; do
    pkill -f "$pat" && echo "stopped: $pat"
done
true
