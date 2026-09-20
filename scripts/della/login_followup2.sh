#!/bin/bash
# Runs on the Della login node AFTER login_evals2.sh releases the GPU. The waiting loop lives
# here, in a detached script, so that no SSH session has to stay open while it waits.
#   cd $ROOT; ( nohup scripts/della/login_followup2.sh > $LOGS/login_followup2.log 2>&1 < /dev/null & )
set -uo pipefail
ROOT=/scratch/gpfs/ARORA/hh9077/ttt-big-fast; RES=/scratch/gpfs/ARORA/hh9077/results; LOGS=/scratch/gpfs/ARORA/hh9077/logs
export HF_HOME=/scratch/gpfs/ARORA/hh9077/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH="$ROOT"
cd "$ROOT"
while pgrep -u hh9077 -f login_evals2.sh > /dev/null; do sleep 15; done

echo "[start] context_value $(date +%H:%M:%S)"
timeout -s KILL 780 .venv/bin/python -u scripts/context_value.py --data /scratch/gpfs/ARORA/hh9077/data/pg19_32k \
    --out "$RES/context_value_32k_s8192.json" > "$LOGS/login_context_value.log" 2>&1
echo "[done]  context_value exit=$?"; tail -6 "$LOGS/login_context_value.log"

echo "[start] armD_probe_32k $(date +%H:%M:%S)"
timeout -s KILL 720 .venv/bin/python scripts/memory_probe.py --seq-len 32768 --chunk 1024 --fast-blocks 4 --remat-group 1 \
    --prefix-segment 8192 --truncate-bptt 2 --remat-blocks --full-slow > "$LOGS/login_armD_probe_32k.log" 2>&1
echo "[done]  armD_probe_32k exit=$?"
grep -E 'config:|slow params|full sequence|after backward|OOM:|peak at' "$LOGS/login_armD_probe_32k.log"
echo "=== followup2 finished ==="
