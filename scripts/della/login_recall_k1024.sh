#!/bin/bash
# Recall from beyond the window (scripts/recall_probe.py), PG-19 at 32K, k = 1024, login GPU.
# A 1024-token passage from another book is planted in chunk 2 and repeated 17 chunks later:
# gap 17408 > 16 x 1023 = 16368, attention's whole reach, so WITHOUT test-time training recall
# is exactly 0 and the script verifies that. Same 32 (carrier, donor) pairs in every run.
#   1. un-tuned window, TTT on and off        (the floor check on real hardware comes first)
#   2. 40-step arm C weights, TTT on and off  (slow weights trained through the inner loop)
#   3. 40-step plain fine-tune, TTT on / off  (the control's weights)
#   4. un-tuned FULL attention                (the ceiling)
#   5. how recall scales with the inner step size, un-tuned (write strength)
#   6. a gap inside attention's reach but beyond one window: do stacked windows carry anything?
# Separate short processes, hard timeouts, finished work skipped, so it can simply be rerun:
#   cd $ROOT; ( nohup scripts/della/login_recall_k1024.sh > $LOGS/login_recall_k1024.log 2>&1 < /dev/null & )
set -uo pipefail
ROOT=/scratch/gpfs/ARORA/hh9077/ttt-big-fast; RES=/scratch/gpfs/ARORA/hh9077/results
LOGS=/scratch/gpfs/ARORA/hh9077/logs; DATA=/scratch/gpfs/ARORA/hh9077/data/pg19_32k
export HF_HOME=/scratch/gpfs/ARORA/hh9077/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH="$ROOT"
cd "$ROOT"
failed=()
# One GPU job of ours at a time on this node.
while pgrep -u hh9077 -f 'login_slimpajama_k1024|login_cell_|login_cv_' > /dev/null; do sleep 20; done

BASE="--mode eval --data $DATA --seq-len 32768 --chunk 1024 --fast-blocks 4 --remat-group 1 --remat-blocks
      --dtype bf16 --source-start 2048 --length 1024 --cue 32 --pairs 32"
K1024="--window 1024 --prefix-segment 1024"
TTT="--inner normalized_sgd --inner-lr 4e-6"
run_probe() {   # run_probe <name> <recall_probe.py args...>
  local name=$1; shift
  [ -e "$RES/$name.json" ] && { echo "[skip] $name"; return; }
  echo "[start] $name $(date +%H:%M:%S)"; local t0=$SECONDS
  timeout -s KILL 780 .venv/bin/python -u scripts/recall_probe.py $BASE --out "$RES/$name.json" "$@" > "$LOGS/login_$name.log" 2>&1
  local rc=$?
  if [ $rc -eq 0 ]; then echo "[done]  $name $((SECONDS - t0))s"
  else echo "[FAIL]  $name exit=$rc -> $LOGS/login_$name.log"; failed+=("$name"); fi
  grep -E '^(ttt_on|ttt_off|no_ttt|what TTT adds|floor check|FLOOR CHECK)' "$LOGS/login_$name.log" | cut -c1-200
}
run_probe recall_pg19_k1024_untuned --arm B $K1024 $TTT --eval-ttt-off --gap 17408
# The floor must hold before anything else is worth running.
grep -q 'floor check passed' "$LOGS/login_recall_pg19_k1024_untuned.log" || { echo "[FAIL]  no exact floor on real hardware. Stopping."; exit 1; }
run_probe recall_pg19_k1024_meta40  --arm C --lora-rank 64 $K1024 $TTT --eval-ttt-off --gap 17408 --load-slow "$RES/C_32k_k1024_t4_s40.ckpt"
run_probe recall_pg19_k1024_plain40 --arm C --lora-rank 64 $K1024 $TTT --eval-ttt-off --gap 17408 --load-slow "$RES/C_32k_k1024_ctl_s40.ckpt"
run_probe recall_pg19_fullattn --arm A --window 32768 --prefix-segment 8192 --gap 17408
for lr in 2e-5 5e-5; do   # LM loss at these rates, un-tuned: 4.7039 and 6.9575 (4.5232 at 4e-6)
  run_probe "recall_pg19_k1024_untuned_lr$lr" --arm B $K1024 --inner normalized_sgd --inner-lr "$lr" --gap 17408
done
run_probe recall_pg19_k1024_untuned_gap4096 --arm B $K1024 $TTT --eval-ttt-off --gap 4096
echo "=== login_recall_k1024: failed=${#failed[@]} ${failed[*]:-} ==="
