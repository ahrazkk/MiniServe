#!/usr/bin/env bash
# Run a long command in the MiniServe environment with all output in a log file.
#   bash scripts/run_detached.sh results/logs/bench.log bash scripts/bench_v05.sh
# Launch it from Windows with Start-Process (see CLAUDE.md) so it outlives the terminal.
log="$1"
shift
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
mkdir -p "$(dirname "$log")"
bash scripts/dev.sh "$@" > "$log" 2>&1
echo "EXIT CODE $?" >> "$log"
