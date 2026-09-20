#!/bin/bash
# The 2x2 behind H1: slow weights trained {through the inner loop, without it} x evaluated
# {with TTT, without}. Both sets of 10-step weights already exist as checkpoints, so this is
# evaluation only. Each run evaluates with TTT on and off; ONE of the two must reproduce a
# number already on record, which checks that the right weights were loaded.
#   cd $ROOT; ( nohup scripts/della/login_followup4.sh > $LOGS/login_followup4.log 2>&1 < /dev/null & )
set -uo pipefail
ROOT=/scratch/gpfs/ARORA/hh9077/ttt-big-fast; RES=/scratch/gpfs/ARORA/hh9077/results; LOGS=/scratch/gpfs/ARORA/hh9077/logs
export HF_HOME=/scratch/gpfs/ARORA/hh9077/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH="$ROOT"
cd "$ROOT"
while pgrep -u hh9077 -f 'login_evals2.sh|login_followup2.sh|login_followup3.sh' > /dev/null; do sleep 15; done
COMMON="--arm C --mode eval --data /scratch/gpfs/ARORA/hh9077/data/pg19_32k --seq-len 32768 --chunk 1024 --window 8192
        --fast-blocks 4 --remat-group 1 --prefix-segment 8192 --remat-blocks --dtype bf16 --lora-rank 64
        --inner normalized_sgd --inner-lr 4e-6 --eval-sequences 32 --eval-ttt-off"
#   name                 checkpoint                   the evaluation that must reproduce a known number
for spec in "cell_plainft_10|C_32k_ctl_none10|TTT-OFF must equal 2.71244" "cell_meta_10|C_32k_resumecheck|TTT-ON must equal 2.66892"; do
  IFS='|' read -r name ckpt expect <<< "$spec"
  [ -e "$RES/$name.json" ] && { echo "[skip] $name"; continue; }
  [ -e "$RES/$ckpt.ckpt" ] || { echo "[FAIL]  $name: missing checkpoint $RES/$ckpt.ckpt"; continue; }
  echo "[start] $name $(date +%H:%M:%S)   ($expect)"; t0=$SECONDS
  timeout -s KILL 780 .venv/bin/python -u -m ttt.run $COMMON --load-slow "$RES/$ckpt.ckpt" --out "$RES/$name.json" > "$LOGS/login_$name.log" 2>&1; rc=$?
  if [ $rc -eq 0 ]; then echo "[done]  $name $((SECONDS - t0))s"; grep -E '^\[eval\]' "$LOGS/login_$name.log" | sed 's/ peak=.*//'
  else echo "[FAIL]  $name exit=$rc -> $LOGS/login_$name.log"; fi
done
echo "=== followup4 finished ==="
