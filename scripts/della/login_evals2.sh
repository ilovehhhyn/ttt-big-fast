#!/bin/bash
# Second batch of evaluation-only runs for the Della LOGIN node (see login_evals.sh for the
# rules: short separate processes, hard timeout, failures reported, existing results skipped).
#
#   cd $ROOT; ( nohup scripts/della/login_evals2.sh > $LOGS/login_evals2.log 2>&1 < /dev/null & )
set -uo pipefail
ROOT=/scratch/gpfs/ARORA/hh9077/ttt-big-fast
RES=/scratch/gpfs/ARORA/hh9077/results
LOGS=/scratch/gpfs/ARORA/hh9077/logs
export HF_HOME=/scratch/gpfs/ARORA/hh9077/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH="$ROOT"
MAX_SECONDS=720
cd "$ROOT"
BASE="--mode eval --data /scratch/gpfs/ARORA/hh9077/data/pg19_32k --chunk 1024 --window 8192
      --fast-blocks 4 --remat-group 1 --prefix-segment 8192 --remat-blocks --dtype bf16"

failed=(); ran=0; skipped=0
run_eval() {   # run_eval <name> <ttt.run args...>
  local name=$1; shift
  local out="$RES/$name.json"
  if [ -e "$out" ]; then echo "[skip] $name (result exists)"; skipped=$((skipped + 1)); return; fi
  echo "[start] $name  $(date +%H:%M:%S)"
  local t0=$SECONDS
  timeout -s KILL "$MAX_SECONDS" .venv/bin/python -u -m ttt.run $BASE --out "$out" "$@" > "$LOGS/login_$name.log" 2>&1
  local rc=$?
  if [ $rc -eq 0 ] && [ -e "$out" ]; then
    echo "[done]  $name  $((SECONDS - t0))s  $(grep -E '^\[eval\]' "$LOGS/login_$name.log" | tail -1)"
    ran=$((ran + 1))
  else
    echo "[FAIL]  $name  exit=$rc after $((SECONDS - t0))s  -> $LOGS/login_$name.log"
    failed+=("$name")
  fi
}

# 1. The matched T/k = 1 row the context-scaling table never had: PG-19 at T = 8192 with the
#    SAME inner rule as the 16K and 32K rows (normalized SGD, 4e-6). 128 sequences of 8192
#    tokens = the same token count as 32 sequences of 32768.
run_eval A_8k_pg19 --arm A --seq-len 8192 --eval-sequences 128
run_eval B_8k_pg19 --arm B --seq-len 8192 --eval-sequences 128 --inner normalized_sgd --inner-lr 4e-6

# 2. AdamW was still improving at the top of the first grid (2e-5 -> 3.6175), so its optimum
#    had not been bracketed. Extend upward until it turns.
for lr in 5e-5 1e-4 2e-4; do
  run_eval "B_32k_adamw_lr$lr" --arm B --seq-len 32768 --eval-sequences 32 --inner adamw --inner-lr "$lr" --adam-eps 1e-8
done

echo "=== login_evals2: ran=$ran skipped=$skipped failed=${#failed[@]} ${failed[*]:-} ==="
[ ${#failed[@]} -eq 0 ]
