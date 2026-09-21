#!/bin/bash
# Round 2, login-node part (2026-09-21): everything that needs the internet or is a short
# evaluation-only GPU run. Separate short processes, hard timeouts, failures reported,
# finished work skipped -- so it can simply be rerun. Launch detached:
#   cd $ROOT; ( nohup scripts/della/round2_login.sh > $LOGS/round2_login.log 2>&1 < /dev/null & )
set -uo pipefail
ROOT=/scratch/gpfs/ARORA/hh9077/ttt-big-fast; RES=/scratch/gpfs/ARORA/hh9077/results
LOGS=/scratch/gpfs/ARORA/hh9077/logs; DATA=/scratch/gpfs/ARORA/hh9077/data
export HF_HOME=/scratch/gpfs/ARORA/hh9077/hf TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH="$ROOT"
cd "$ROOT"
failed=()

# ---- 1. k = 1024 on PG-19 at 32K, nothing trained: how broken is the un-tuned model at this
#         window, what does TTT alone recover, and where is the inner-LR optimum?
export HF_HUB_OFFLINE=1
K1024="--mode eval --data $DATA/pg19_32k --seq-len 32768 --chunk 1024 --window 1024 --prefix-segment 1024
       --fast-blocks 4 --remat-group 1 --remat-blocks --dtype bf16 --eval-sequences 32"
run_eval() {
  local name=$1; shift
  [ -e "$RES/$name.json" ] && { echo "[skip] $name"; return; }
  echo "[start] $name $(date +%H:%M:%S)"; local t0=$SECONDS
  timeout -s KILL 720 .venv/bin/python -u -m ttt.run $K1024 --out "$RES/$name.json" "$@" > "$LOGS/login_$name.log" 2>&1
  local rc=$?
  if [ $rc -eq 0 ] && [ -e "$RES/$name.json" ]; then echo "[done]  $name $((SECONDS - t0))s $(grep -E '^\[eval\]' "$LOGS/login_$name.log" | tail -1 | sed 's/ peak=.*//')"
  else echo "[FAIL]  $name exit=$rc -> $LOGS/login_$name.log"; failed+=("$name"); fi
}
run_eval A_32k_k1024 --arm A
for lr in 2e-6 4e-6 7e-6 2e-5 5e-5; do
  run_eval "B_32k_k1024_lr$lr" --arm B --inner normalized_sgd --inner-lr "$lr"
done

# ---- 2. Memory of arm C training at k = 1024. A smaller window shrinks the per-window
#         backward spike, so a LONGER truncation window (less biased meta-gradient) may fit.
for t in 2 4 8; do
  log="$LOGS/login_probe_k1024_t$t.log"
  grep -q 'after backward\|OOM' "$log" 2>/dev/null && { echo "[skip] probe k1024 t$t"; continue; }
  echo "[start] probe k=1024 truncate_bptt=$t $(date +%H:%M:%S)"
  timeout -s KILL 720 .venv/bin/python scripts/memory_probe.py --seq-len 32768 --chunk 1024 --window 1024 \
      --fast-blocks 4 --remat-group 1 --prefix-segment 1024 --truncate-bptt "$t" --remat-blocks > "$log" 2>&1
  echo "[done]  probe t=$t exit=$?  $(grep -E 'full sequence|after backward|OOM:' "$log" | tr '\n' ' ' | cut -c1-200)"
done

# ---- 3. What is out-of-window context worth on SlimPajama, per source domain? The corpus is
#         prepared by a CPU compute job (download_and_prep.sh); wait for it here, inside this
#         detached script, never in an SSH session.
for i in $(seq 1 1440); do [ -e "$DATA/slimpajama_32k/val_docs.json" ] && break; sleep 30; done   # up to 12 h
if [ -e "$DATA/slimpajama_32k/val_docs.json" ]; then
  # One run per domain that has at least 4 validation documents, 24 sequences each.
  labels=$(.venv/bin/python -c "
import json, collections
c = collections.Counter(json.load(open('$DATA/slimpajama_32k/val_docs.json'))['labels'])
print(' '.join(l for l, n in sorted(c.items()) if n >= 4))")
  echo "[info] SlimPajama val labels with >= 4 documents: $labels"
  for lab in $labels; do
    out="$RES/context_value_slimpajama_${lab}_s8192.json"
    [ -e "$out" ] && { echo "[skip] context_value $lab"; continue; }
    echo "[start] context_value slimpajama $lab $(date +%H:%M:%S)"
    timeout -s KILL 780 .venv/bin/python -u scripts/context_value.py --data "$DATA/slimpajama_32k" --only-label "$lab" \
        --eval-sequences 24 --out "$out" > "$LOGS/login_cv_slimpajama_$lab.log" 2>&1
    rc=$?; echo "[done]  context_value $lab exit=$rc"; [ $rc -eq 0 ] || failed+=("cv_$lab")
    grep -E 'recent context|sanity|documents\.' "$LOGS/login_cv_slimpajama_$lab.log" | cut -c1-200
  done
else
  echo "[FAIL]  slimpajama_32k was not prepared within 12 hours"; failed+=("slimpajama_missing")
fi
echo "=== round2_login: failed=${#failed[@]} ${failed[*]:-} ==="
