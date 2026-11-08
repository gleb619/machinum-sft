#!/bin/bash
# Hourly progress report for the remote worker. Prints one compact block; the caller decides
# where it goes (cron job → chat, or a logged file).
set -uo pipefail
REPO=${MSFT_REPO_DIR:-$HOME/WORKSPACE/machinum-sft}
cd "$REPO" 2>/dev/null || { echo "no repo at $REPO"; exit 0; }

echo "## machinum-sft $(date '+%H:%M %Z')"

if command -v docker >/dev/null && docker ps --format '{{.Names}}' | grep -q '^msft-worker$'; then
    echo "worker: up ($(docker ps --format '{{.Status}}' --filter name=msft-worker))"
    docker exec msft-worker python -m msft worker status --dir /workspace/.worker 2>/dev/null \
        | sed 's/^/  /'
    for job in $(docker exec msft-worker ls /workspace/.worker/running 2>/dev/null); do
        echo "  running: $job"
        docker exec msft-worker tail -3 "/workspace/.worker/logs/${job%.json}.log" 2>/dev/null \
            | sed 's/^/    /'
    done
    last=$(docker exec msft-worker ls -t /workspace/.worker/logs 2>/dev/null | head -1)
    if [ -n "${last:-}" ]; then
        echo "  last log: $last"
        docker exec msft-worker tail -5 "/workspace/.worker/logs/$last" 2>/dev/null | sed 's/^/    /'
    fi
else
    echo "worker: NOT running"
fi

echo "gpu:  $(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | tr '\n' ' ')"
echo "out:  $(du -sh out 2>/dev/null | cut -f1) adapter=$(du -sh out/adapter 2>/dev/null | cut -f1) gguf=$(ls -la out/*.gguf 2>/dev/null | awk '{print $5" B"}' | head -1)"
echo "git:  $(git log --oneline -1)"
