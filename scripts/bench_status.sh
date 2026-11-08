#!/usr/bin/env bash
# What benchmark processes are running inside WSL, plus the latest results lines.
#   wsl -d Ubuntu-24.04 -u root -- bash /mnt/c/.../MiniServe/scripts/bench_status.sh
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
echo "--- processes ---"
pgrep -af "[m]iniserve (serve|baseline)" || echo "no server running"
pgrep -af "[r]un_matrix" || echo "no sweep running"
pgrep -af "[n]vidia-smi --query-gpu=timestamp" || echo "no GPU monitor running"
pgrep -af "[a]fter.sh" || echo "no queued job"
echo "--- latest results ---"
grep -E "^=== |output throughput|TPOT|errors|DONE|EXIT CODE|Traceback|Error" \
    results/logs/bench_v05_driver.log 2>/dev/null | tail -n 12
