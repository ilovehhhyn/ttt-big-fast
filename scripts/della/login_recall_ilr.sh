#!/bin/bash
# Recall of the 40-step weights that were TRAINED at a larger inner learning rate, each scored at
# the rate it was trained with (PG-19, window 1024, the same 32 pairs as every other recall run).
# Reference: weights trained at 4e-6 recall +0.1054 at 4e-6 and +0.2652 / +0.3880 when the rate is
# raised to 1e-5 / 2e-5 only at test time, where their loss rises from 2.6777 to 2.7088 / 2.8318.
# Waits, detached, for each training result; login GPU, one short process per run.
#   cd $ROOT; ( nohup scripts/della/login_recall_ilr.sh > $LOGS/login_recall_ilr.log 2>&1 < /dev/null & )
set -uo pipefail
ROOT=/scratch/gpfs/ARORA/hh9077/ttt-big-fast; RES=/scratch/gpfs/ARORA/hh9077/results
LOGS=/scratch/gpfs/ARORA/hh9077/logs; DATA=/scratch/gpfs/ARORA/hh9077/data/pg19_32k
export HF_HOME=/scratch/gpfs/ARORA/hh9077/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH="$ROOT"
cd "$ROOT"
failed=0
for lr in 1e-5 2e-5; do
  trained="C_32k_k1024_t4_s40_ilr$lr"; name="recall_pg19_k1024_meta40_trained_ilr$lr"
  [ -e "$RES/$name.json" ] && { echo "[skip] $name"; continue; }
  # The result file appears only after the job's own evaluation, so training is complete.
  for i in $(seq 1 240); do [ -e "$RES/$trained.json" ] && break; sleep 30; done   # up to 2 h
  [ -e "$RES/$trained.json" ] || { echo "[FAIL]  $trained.json did not appear within 2 hours"; failed=$((failed + 1)); continue; }
  echo "[start] $name $(date +%H:%M:%S)"
  timeout -s KILL 780 .venv/bin/python -u scripts/recall_probe.py --arm C --lora-rank 64 --mode eval --data "$DATA" \
      --seq-len 32768 --chunk 1024 --window 1024 --prefix-segment 1024 --fast-blocks 4 --remat-group 1 --remat-blocks \
      --dtype bf16 --inner normalized_sgd --inner-lr "$lr" --eval-ttt-off --source-start 2048 --length 1024 --cue 32 \
      --gap 17408 --pairs 32 --load-slow "$RES/$trained.ckpt" --out "$RES/$name.json" > "$LOGS/login_$name.log" 2>&1
  rc=$?; [ $rc -eq 0 ] || { echo "[FAIL]  $name exit=$rc -> $LOGS/login_$name.log"; failed=$((failed + 1)); }
  grep -E '^(ttt_on|ttt_off|what TTT adds|floor check|FLOOR CHECK)' "$LOGS/login_$name.log" | cut -c1-200
done
echo "=== login_recall_ilr: failed=$failed ==="
