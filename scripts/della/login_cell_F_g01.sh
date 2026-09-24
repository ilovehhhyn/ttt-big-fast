#!/bin/bash
# The arm F 2x2 at gate init 0.1: scores the plain control (F_32k_k1024_g01_ctl_s40) with the
# write on and off, then runs scripts/two_by_two.py against F_32k_k1024_g01_s40. Waits for the
# login GPU to be free of other evaluation processes (not for other waiting scripts).
#   cd $ROOT; ( nohup scripts/della/login_cell_F_g01.sh > $LOGS/login_cell_F_g01.log 2>&1 < /dev/null & )
set -uo pipefail
ROOT=/scratch/gpfs/ARORA/hh9077/ttt-big-fast; RES=/scratch/gpfs/ARORA/hh9077/results; LOGS=/scratch/gpfs/ARORA/hh9077/logs
export HF_HOME=/scratch/gpfs/ARORA/hh9077/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=$ROOT
cd $ROOT
while pgrep -u hh9077 -f 'ttt.run|recall_probe|memory_probe|context_value' > /dev/null; do sleep 20; done
NAME=cell_F_g01_plainft_40
if [ -e $RES/$NAME.json ]; then echo "[skip] $NAME"; else
  echo "[start] $NAME $(date +%H:%M:%S)"; t0=$SECONDS
  timeout -s KILL 780 .venv/bin/python -u -m ttt.run --arm F --prime-intermediate 2048 --prime-gate-init 0.1 --mode eval \
    --data /scratch/gpfs/ARORA/hh9077/data/pg19_32k --seq-len 32768 --chunk 1024 --window 1024 --prefix-segment 1024 \
    --fast-blocks 4 --remat-group 1 --remat-blocks --dtype bf16 --lora-rank 64 --inner normalized_sgd --inner-lr 4e-6 \
    --eval-sequences 32 --eval-ttt-off --load-slow $RES/F_32k_k1024_g01_ctl_s40.ckpt --out $RES/$NAME.json \
    > $LOGS/login_$NAME.log 2>&1; rc=$?
  if [ $rc -eq 0 ]; then echo "[done]  $NAME $((SECONDS - t0))s"; grep -E '^\[eval\]' $LOGS/login_$NAME.log | sed 's/ peak=.*//'
  else echo "[FAIL]  $NAME exit=$rc -> $LOGS/login_$NAME.log"; exit 1; fi
fi
.venv/bin/python scripts/two_by_two.py --meta $RES/F_32k_k1024_g01_s40.json --plain $RES/$NAME.json
echo "=== login_cell_F_g01: done ==="
