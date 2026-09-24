#!/bin/bash
# The 2x2 for the Muon meta-training run: scores the plain 40-step control (C_32k_k1024_ctl_s40)
# with Muon at 1.2e-4 on and off, runs scripts/two_by_two.py against C_32k_k1024_muon_s40, probes
# GPU memory at chunk 2048 (window 2048, truncation 2), then hands the GPU back to the 150-step
# 2x2 script, which waits for the C32k_ctl150 job. One process at a time, each under the login
# node's 13-minute limit.
#   cd $ROOT; ( nohup scripts/della/login_muon_cell.sh > $LOGS/login_muon_cell.log 2>&1 < /dev/null & )
set -uo pipefail
ROOT=/scratch/gpfs/ARORA/hh9077/ttt-big-fast; RES=/scratch/gpfs/ARORA/hh9077/results; LOGS=/scratch/gpfs/ARORA/hh9077/logs
export HF_HOME=/scratch/gpfs/ARORA/hh9077/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=$ROOT
cd $ROOT
while pgrep -u hh9077 -f 'login_recall_|login_sp_|login_cell|login_cv_' | grep -v $$ > /dev/null; do sleep 20; done
failed=0
NAME=cell_k1024_plainft_40_muon
if [ -e $RES/$NAME.json ]; then echo "[skip] $NAME"; else
  echo "[start] $NAME $(date +%H:%M:%S)"; t0=$SECONDS
  timeout -s KILL 780 .venv/bin/python -u -m ttt.run --arm C --mode eval --data /scratch/gpfs/ARORA/hh9077/data/pg19_32k \
    --seq-len 32768 --chunk 1024 --window 1024 --prefix-segment 1024 --fast-blocks 4 --remat-group 1 --remat-blocks \
    --dtype bf16 --lora-rank 64 --inner muon --inner-lr 1.2e-4 --eval-sequences 32 --eval-ttt-off \
    --load-slow $RES/C_32k_k1024_ctl_s40.ckpt --out $RES/$NAME.json > $LOGS/login_$NAME.log 2>&1; rc=$?
  if [ $rc -eq 0 ]; then echo "[done]  $NAME $((SECONDS - t0))s"; grep -E '^\[eval\]' $LOGS/login_$NAME.log | sed 's/ peak=.*//'
  else echo "[FAIL]  $NAME exit=$rc -> $LOGS/login_$NAME.log"; failed=1; fi
fi
[ -e $RES/$NAME.json ] && .venv/bin/python scripts/two_by_two.py --meta $RES/C_32k_k1024_muon_s40.json --plain $RES/$NAME.json
echo "[start] memprobe_chunk2048 $(date +%H:%M:%S)"
timeout -s KILL 600 .venv/bin/python scripts/memory_probe.py --seq-len 32768 --chunk 2048 --window 2048 --prefix-segment 2048 \
  --fast-blocks 4 --remat-group 1 --truncate-bptt 2 --remat-blocks > $LOGS/login_memprobe_chunk2048.log 2>&1; echo "[done]  memprobe_chunk2048 exit=$?"
grep -E 'config:|slow params|full sequence|after backward|OOM:|peak at' $LOGS/login_memprobe_chunk2048.log
echo "=== login_muon_cell: failed=$failed ==="
exec scripts/della/login_cells_k8192_s60_s150.sh
