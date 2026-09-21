#!/bin/bash
# The missing cell of the 40-step 2x2 at k = 1024: plain fine-tuned slow weights, evaluated with
# TTT on and off. The TTT-OFF evaluation must reproduce the control's own number (2.7196),
# which checks that the right weights were loaded. Detached; one short process.
#   cd $ROOT; ( nohup scripts/della/login_cell_k1024_s40.sh > $LOGS/login_cell_k1024_s40.log 2>&1 < /dev/null & )
set -uo pipefail
ROOT=/scratch/gpfs/ARORA/hh9077/ttt-big-fast; RES=/scratch/gpfs/ARORA/hh9077/results; LOGS=/scratch/gpfs/ARORA/hh9077/logs
export HF_HOME=/scratch/gpfs/ARORA/hh9077/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=$ROOT
cd $ROOT
name=cell_k1024_plainft_40; ckpt=C_32k_k1024_ctl_s40
[ -e $RES/$name.json ] && { echo "[skip] $name"; exit 0; }
[ -e $RES/$ckpt.ckpt ] || { echo "[FAIL] missing $RES/$ckpt.ckpt"; exit 1; }
echo "[start] $name $(date +%H:%M:%S)  (TTT-OFF must equal 2.7196)"; t0=$SECONDS
timeout -s KILL 780 .venv/bin/python -u -m ttt.run --arm C --mode eval --data /scratch/gpfs/ARORA/hh9077/data/pg19_32k \
  --seq-len 32768 --chunk 1024 --window 1024 --fast-blocks 4 --remat-group 1 --prefix-segment 1024 --remat-blocks \
  --dtype bf16 --lora-rank 64 --inner normalized_sgd --inner-lr 4e-6 --eval-sequences 32 --eval-ttt-off \
  --load-slow $RES/$ckpt.ckpt --out $RES/$name.json > $LOGS/login_$name.log 2>&1; rc=$?
if [ $rc -eq 0 ]; then echo "[done]  $name $((SECONDS - t0))s"; grep -E '^\[eval\]' $LOGS/login_$name.log | sed 's/ peak=.*//'
  .venv/bin/python scripts/two_by_two.py --meta $RES/C_32k_k1024_t4_s40.json --plain $RES/$name.json
else echo "[FAIL]  $name exit=$rc -> $LOGS/login_$name.log"; tail -5 $LOGS/login_$name.log | cut -c1-200; fi
echo '=== login_cell_k1024_s40 finished ==='
