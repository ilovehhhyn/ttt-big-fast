#!/bin/bash
# The 2x2 at window 8192, 20 steps: both sets of 20-step slow weights (trained through the inner loop:
# C_32k_t2; plain fine-tune: C_32k_ctl20) evaluated with TTT on and off on the standard 32 sequences.
# One evaluation of each pair must reproduce the training job's own number, which checks the load.
#   cd $ROOT; ( nohup scripts/della/login_cells_k8192_s20.sh > $LOGS/login_cells_k8192_s20.log 2>&1 < /dev/null & )
set -uo pipefail
ROOT=/scratch/gpfs/ARORA/hh9077/ttt-big-fast; RES=/scratch/gpfs/ARORA/hh9077/results; LOGS=/scratch/gpfs/ARORA/hh9077/logs
export HF_HOME=/scratch/gpfs/ARORA/hh9077/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=$ROOT
cd $ROOT
while pgrep -u hh9077 -f 'login_recall_|login_sp_|login_cell_|login_cv_' > /dev/null; do sleep 20; done
COMMON="--arm C --mode eval --data /scratch/gpfs/ARORA/hh9077/data/pg19_32k --seq-len 32768 --chunk 1024 --window 8192
        --fast-blocks 4 --remat-group 1 --prefix-segment 8192 --remat-blocks --dtype bf16 --lora-rank 64
        --inner normalized_sgd --inner-lr 4e-6 --eval-sequences 32 --eval-ttt-off"
failed=0
for spec in 'cell_meta_20|C_32k_t2|TTT-ON must equal 2.5958' 'cell_plainft_20|C_32k_ctl20|TTT-OFF must equal 2.6139'; do
  IFS='|' read -r name ckpt expect <<< "$spec"
  [ -e $RES/$name.json ] && { echo "[skip] $name"; continue; }
  [ -e $RES/$ckpt.ckpt ] || { echo "[FAIL]  $name: missing $RES/$ckpt.ckpt"; failed=1; continue; }
  echo "[start] $name $(date +%H:%M:%S)   ($expect)"; t0=$SECONDS
  timeout -s KILL 780 .venv/bin/python -u -m ttt.run $COMMON --load-slow $RES/$ckpt.ckpt --out $RES/$name.json > $LOGS/login_$name.log 2>&1; rc=$?
  if [ $rc -eq 0 ]; then echo "[done]  $name $((SECONDS - t0))s"; grep -E '^\[eval\]' $LOGS/login_$name.log | sed 's/ peak=.*//'
  else echo "[FAIL]  $name exit=$rc -> $LOGS/login_$name.log"; failed=1; fi
done
[ $failed -eq 0 ] && .venv/bin/python scripts/two_by_two.py --meta $RES/cell_meta_20.json --plain $RES/cell_plainft_20.json
echo "=== login_cells_k8192_s20: failed=$failed ==="
