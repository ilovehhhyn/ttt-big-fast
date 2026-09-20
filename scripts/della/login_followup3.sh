#!/bin/bash
# Headroom versus window size: context_value.py at smaller restart lengths. Runs on the
# login node after login_followup2.sh; the wait happens here, not in an SSH session.
#   cd $ROOT; ( nohup scripts/della/login_followup3.sh > $LOGS/login_followup3.log 2>&1 < /dev/null & )
set -uo pipefail
ROOT=/scratch/gpfs/ARORA/hh9077/ttt-big-fast; RES=/scratch/gpfs/ARORA/hh9077/results; LOGS=/scratch/gpfs/ARORA/hh9077/logs
export HF_HOME=/scratch/gpfs/ARORA/hh9077/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH="$ROOT"
cd "$ROOT"
while pgrep -u hh9077 -f 'login_evals2.sh|login_followup2.sh' > /dev/null; do sleep 15; done
for S in 2048 1024; do
  out="$RES/context_value_32k_s$S.json"
  [ -e "$out" ] && { echo "[skip] S=$S"; continue; }
  echo "[start] context_value S=$S $(date +%H:%M:%S)"
  timeout -s KILL 780 .venv/bin/python -u scripts/context_value.py --data /scratch/gpfs/ARORA/hh9077/data/pg19_32k \
      --segment "$S" --out "$out" > "$LOGS/login_context_value_s$S.log" 2>&1
  echo "[done]  context_value S=$S exit=$?"; tail -5 "$LOGS/login_context_value_s$S.log"
done
echo "=== followup3 finished ==="
