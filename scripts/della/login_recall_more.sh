#!/bin/bash
# Recall probe, second batch (login GPU; run after login_recall_k1024.sh):
#   1. the reference window k = 8192 on PG-19: un-tuned, the 10-step arm C weights and the 10-step
#      plain fine-tune. Attention's reach at k = 8192 is 16 x 8191 tokens, more than T, so the
#      no-TTT floor is MEASURED here (TTT off), not zero by construction; the gap, 17408 tokens,
#      is still more than two windows.
#   2. SlimPajama at k = 1024 (gap beyond attention's reach, exact floor): un-tuned, then the
#      40-step arm C weights and plain fine-tune trained on SlimPajama.
#   3. capacity: un-tuned PG-19 at k = 1024 with 2 and 8 fast blocks instead of 4, same per-element
#      inner step (the thesis says bigger fast weights hold more).
# Separate short processes, hard timeouts, finished work skipped, so it can simply be rerun:
#   cd $ROOT; ( nohup scripts/della/login_recall_more.sh > $LOGS/login_recall_more.log 2>&1 < /dev/null & )
set -uo pipefail
ROOT=/scratch/gpfs/ARORA/hh9077/ttt-big-fast; RES=/scratch/gpfs/ARORA/hh9077/results
LOGS=/scratch/gpfs/ARORA/hh9077/logs; DATA=/scratch/gpfs/ARORA/hh9077/data
export HF_HOME=/scratch/gpfs/ARORA/hh9077/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH="$ROOT"
cd "$ROOT"
failed=()
# One GPU job of ours at a time on this node.
while pgrep -u hh9077 -f 'login_slimpajama_k1024|login_recall_k1024|login_sp_cells|login_cell_|login_cv_' > /dev/null; do sleep 20; done

BASE="--mode eval --seq-len 32768 --chunk 1024 --remat-group 1 --remat-blocks --dtype bf16
      --source-start 2048 --length 1024 --cue 32 --gap 17408"
TTT="--inner normalized_sgd --inner-lr 4e-6"
run_probe() {   # run_probe <name> <recall_probe.py args...>
  local name=$1; shift
  [ -e "$RES/$name.json" ] && { echo "[skip] $name"; return; }
  echo "[start] $name $(date +%H:%M:%S)"; local t0=$SECONDS
  timeout -s KILL 780 .venv/bin/python -u scripts/recall_probe.py $BASE --out "$RES/$name.json" "$@" > "$LOGS/login_$name.log" 2>&1
  local rc=$?
  if [ $rc -eq 0 ]; then echo "[done]  $name $((SECONDS - t0))s"
  else echo "[FAIL]  $name exit=$rc -> $LOGS/login_$name.log"; failed+=("$name"); tail -2 "$LOGS/login_$name.log" | cut -c1-200; fi
  grep -E '^(ttt_on|ttt_off|no_ttt|what TTT adds|floor check|FLOOR CHECK)' "$LOGS/login_$name.log" | cut -c1-200
}

# ---- 1. k = 8192, PG-19. 16 pairs x (present, absent) x (TTT on, off) = 64 sequence passes, the size
#         of one 2x2 cell evaluation at this window, which is known to fit the time limit here.
K8192="--data $DATA/pg19_32k --window 8192 --prefix-segment 8192 --fast-blocks 4 --pairs 16"
run_probe recall_pg19_k8192_untuned --arm B $K8192 $TTT --eval-ttt-off
run_probe recall_pg19_k8192_meta10  --arm C --lora-rank 64 $K8192 $TTT --eval-ttt-off --load-slow "$RES/C_32k_resumecheck.ckpt"
run_probe recall_pg19_k8192_plain10 --arm C --lora-rank 64 $K8192 $TTT --eval-ttt-off --load-slow "$RES/C_32k_ctl_none10.ckpt"

# ---- 2. SlimPajama, k = 1024
SP="--data $DATA/slimpajama_32k --window 1024 --prefix-segment 1024 --fast-blocks 4 --pairs 32"
run_probe recall_sp_k1024_untuned --arm B $SP $TTT --eval-ttt-off
run_probe recall_sp_k1024_meta40  --arm C --lora-rank 64 $SP $TTT --eval-ttt-off --load-slow "$RES/SP_32k_k1024_t4_s40.ckpt"
run_probe recall_sp_k1024_plain40 --arm C --lora-rank 64 $SP $TTT --eval-ttt-off --load-slow "$RES/SP_32k_k1024_ctl_s40.ckpt"
run_probe recall_sp_fullattn --arm A --data $DATA/slimpajama_32k --window 32768 --prefix-segment 8192 --fast-blocks 4 --pairs 32

# ---- 3. capacity: number of fast blocks, un-tuned, PG-19, k = 1024 (4 blocks is recall_pg19_k1024_untuned)
for fb in 2 8; do
  run_probe "recall_pg19_k1024_untuned_fb$fb" --arm B --data $DATA/pg19_32k --window 1024 --prefix-segment 1024 \
      --fast-blocks "$fb" --pairs 32 $TTT
done
echo "=== login_recall_more: failed=${#failed[@]} ${failed[*]:-} ==="
