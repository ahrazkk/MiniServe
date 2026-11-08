#!/usr/bin/env bash
# Queue a job behind another: wait until <log> contains "EXIT CODE", then run the command.
#   bash scripts/after.sh results/logs/first.log bash scripts/second.sh
# Only one benchmark may use the GPU at a time, so sweeps are chained instead of overlapped.
log="$1"
shift
until grep -q "EXIT CODE" "$log" 2>/dev/null; do sleep 20; done
exec "$@"
