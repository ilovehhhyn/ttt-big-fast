#!/bin/bash
# Recall and loss with --inner preconditioned_sgd (the shared key directions are scaled down in every
# fast-weight update), PG-19, 32K tokens, window 1024, the same 32 pairs and 32 sequences as the
# normalized-SGD runs. Reference, normalized SGD, recall / loss:
#   un-tuned:        4e-6 +0.0717 / 4.5232   1e-5 +0.1438 / 4.5493   2e-5 +0.1348 / 4.7039
#   40-step weights: 4e-6 +0.1054 / 2.6777   1e-5 +0.2652 / 2.7088   2e-5 +0.3880 / 2.8318
# The idea is wrong if, at a loss no worse than normalized SGD's, recall is not at least doubled.
# Login GPU, one short process per run, finished work skipped:
#   cd $ROOT; ( nohup scripts/della/login_recall_preconditioned.sh > $LOGS/login_recall_preconditioned.log 2>&1 < /dev/null & )
set -uo pipefail
ROOT=/scratch/gpfs/ARORA/hh9077/ttt-big-fast; RES=/scratch/gpfs/ARORA/hh9077/results
LOGS=/scratch/gpfs/ARORA/hh9077/logs; DATA=/scratch/gpfs/ARORA/hh9077/data/pg19_32k
export HF_HOME=/scratch/gpfs/ARORA/hh9077/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH="$ROOT"
cd "$ROOT"
failed=0
RANK=64
META="$RES/C_32k_k1024_t4_s40.ckpt"
MODEL="--mode eval --data $DATA --seq-len 32768 --chunk 1024 --window 1024 --prefix-segment 1024 --fast-blocks 4
       --remat-group 1 --dtype bf16"
PROBE="--source-start 2048 --length 1024 --cue 32 --gap 17408 --pairs 32"
run() {   # run <result file name> <script and its arguments...>
  local file=$1; shift
  [ -e "$RES/$file" ] && { echo "[skip] $file"; return; }
  echo "[start] $file $(date +%H:%M:%S)"
  timeout -s KILL 780 .venv/bin/python -u "$@" --out "$RES/$file" > "$LOGS/login_$file.log" 2>&1; local rc=$?
  [ $rc -eq 0 ] || { echo "[FAIL]  $file exit=$rc -> $LOGS/login_$file.log"; tail -3 "$LOGS/login_$file.log" | cut -c1-200; failed=$((failed + 1)); }
  grep -E '^(block [0-9]+ |ttt_on|\[eval\])' "$LOGS/login_$file.log" | sed 's/ peak=.*//' | cut -c1-200
}
# The shared key directions, from 65,536 TRAINING tokens, for the un-tuned model and for the 40-step weights.
run key_basis_k1024_r$RANK.pt scripts/key_basis.py --arm A $MODEL --rank $RANK --tokens 65536
run key_basis_k1024_meta40_r$RANK.pt scripts/key_basis.py --arm C --lora-rank 64 --inner none --load-slow "$META" $MODEL --rank $RANK --tokens 65536
[ -e "$RES/key_basis_k1024_r$RANK.pt" ] && [ -e "$RES/key_basis_k1024_meta40_r$RANK.pt" ] || { echo "[FAIL]  no key basis. Stopping."; exit 1; }

PRE="--inner preconditioned_sgd --remat-blocks"
for lr in 4e-6 1e-5 2e-5 4e-5 1e-4; do
  tag="pre_r${RANK}_c0_lr$lr"; basis="--key-basis $RES/key_basis_k1024_r$RANK.pt"
  run "recall_pg19_k1024_untuned_$tag.json" scripts/recall_probe.py --arm B $MODEL $PRE $basis --inner-lr "$lr" $PROBE
  run "B_32k_k1024_$tag.json" -m ttt.run --arm B $MODEL $PRE $basis --inner-lr "$lr" --eval-sequences 32
done
for lr in 1e-5 2e-5 4e-5 1e-4; do
  tag="pre_r${RANK}_c0_lr$lr"; basis="--key-basis $RES/key_basis_k1024_meta40_r$RANK.pt"
  run "recall_pg19_k1024_meta40_$tag.json" scripts/recall_probe.py --arm C --lora-rank 64 --load-slow "$META" $MODEL $PRE $basis --inner-lr "$lr" $PROBE
  run "C_32k_k1024_t4_s40_$tag.json" -m ttt.run --arm C --lora-rank 64 --load-slow "$META" $MODEL $PRE $basis --inner-lr "$lr" --eval-sequences 32
done
# Keep 30% of the shared part: on the un-tuned model those directions may be what repairs the window.
tag="pre_r${RANK}_c0.3_lr2e-5"; basis="--key-basis $RES/key_basis_k1024_r$RANK.pt --shared-keep 0.3"
run "recall_pg19_k1024_untuned_$tag.json" scripts/recall_probe.py --arm B $MODEL $PRE $basis --inner-lr 2e-5 $PROBE
run "B_32k_k1024_$tag.json" -m ttt.run --arm B $MODEL $PRE $basis --inner-lr 2e-5 --eval-sequences 32
echo "=== login_recall_preconditioned: failed=$failed ==="
