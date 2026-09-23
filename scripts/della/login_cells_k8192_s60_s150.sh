#!/bin/bash
# The 2x2 at window 8192 after 60 and 150 steps. The meta-trained rows already hold both cells
# (C_32k_s60.json, C_32k_s150.json were evaluated with --eval-ttt-off); this evaluates the plain
# fine-tuned weights (C_32k_ctl60, C_32k_ctl150) with TTT on and off. The TTT-off evaluation must
# reproduce the control's own number. Waits, detached, for the 150-step control's result.
#   cd $ROOT; ( nohup scripts/della/login_cells_k8192_s60_s150.sh > $LOGS/login_cells_k8192_s60_s150.log 2>&1 < /dev/null & )
set -uo pipefail
ROOT=/scratch/gpfs/ARORA/hh9077/ttt-big-fast; RES=/scratch/gpfs/ARORA/hh9077/results; LOGS=/scratch/gpfs/ARORA/hh9077/logs
export HF_HOME=/scratch/gpfs/ARORA/hh9077/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=$ROOT
cd $ROOT
COMMON="--arm C --mode eval --data /scratch/gpfs/ARORA/hh9077/data/pg19_32k --seq-len 32768 --chunk 1024 --window 8192
        --fast-blocks 4 --remat-group 1 --prefix-segment 8192 --remat-blocks --dtype bf16 --lora-rank 64
        --inner normalized_sgd --inner-lr 4e-6 --eval-sequences 32 --eval-ttt-off"
failed=0
for spec in 'cell_plainft_60|C_32k_ctl60|C_32k_s60' 'cell_plainft_150|C_32k_ctl150|C_32k_s150'; do
  IFS='|' read -r name ckpt meta <<< "$spec"
  [ -e $RES/$name.json ] && { echo "[skip] $name"; continue; }
  for i in $(seq 1 480); do [ -e $RES/$ckpt.json ] && break; sleep 30; done   # up to 4 h for the control to finish
  [ -e $RES/$ckpt.ckpt ] && [ -e $RES/$ckpt.json ] || { echo "[FAIL]  $name: $ckpt did not finish within 4 hours"; failed=1; continue; }
  while pgrep -u hh9077 -f 'login_recall_|login_sp_|login_cell_|login_cv_' > /dev/null; do sleep 20; done
  echo "[start] $name $(date +%H:%M:%S)"; t0=$SECONDS
  timeout -s KILL 780 .venv/bin/python -u -m ttt.run $COMMON --load-slow $RES/$ckpt.ckpt --out $RES/$name.json > $LOGS/login_$name.log 2>&1; rc=$?
  if [ $rc -eq 0 ]; then echo "[done]  $name $((SECONDS - t0))s"; grep -E '^\[eval\]' $LOGS/login_$name.log | sed 's/ peak=.*//'
    .venv/bin/python scripts/two_by_two.py --meta $RES/$meta.json --plain $RES/$name.json
  else echo "[FAIL]  $name exit=$rc -> $LOGS/login_$name.log"; failed=1; fi
done
echo "=== login_cells_k8192_s60_s150: failed=$failed ==="
