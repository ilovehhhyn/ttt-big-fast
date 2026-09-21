#!/bin/bash
# SlimPajama long documents at k = 1024, login-GPU part (2026-09-21). Nothing is trained here.
# For the FIRST 96 validation sequences of the evaluation order (the ones the trained runs are
# scored on), per sequence:
#   1. how broken the un-tuned model is under the window (arm A at k = 1024; the healthy
#      full-attention loss of the same sequences comes from step 3's report, "per_sequence");
#   2. what TTT alone recovers, and whether the inner-LR optimum found on PG-19 (4e-6) holds here;
#   3. what out-of-window context is worth to a healthy model (context_value.py, S = 1024),
#      in four disjoint pieces of 24 sequences, because one process may not run past ~12 minutes.
# Separate short processes, hard timeouts, finished work skipped, so it can simply be rerun:
#   cd $ROOT; ( nohup scripts/della/login_slimpajama_k1024.sh > $LOGS/login_slimpajama_k1024.log 2>&1 < /dev/null & )
set -uo pipefail
ROOT=/scratch/gpfs/ARORA/hh9077/ttt-big-fast; RES=/scratch/gpfs/ARORA/hh9077/results
LOGS=/scratch/gpfs/ARORA/hh9077/logs; DATA=/scratch/gpfs/ARORA/hh9077/data/slimpajama_32k
export HF_HOME=/scratch/gpfs/ARORA/hh9077/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH="$ROOT"
cd "$ROOT"
N=96
failed=()
# One GPU job at a time on this node: wait for any other detached login script of ours.
while pgrep -u hh9077 -f 'login_cell_|login_cv_|round2_login' > /dev/null; do sleep 20; done

EVAL="--mode eval --data $DATA --seq-len 32768 --chunk 1024 --fast-blocks 4 --remat-group 1 --remat-blocks
      --dtype bf16 --eval-sequences $N"
run_eval() {   # run_eval <name> <ttt.run args...>
  local name=$1; shift
  [ -e "$RES/$name.json" ] && { echo "[skip] $name"; return; }
  echo "[start] $name $(date +%H:%M:%S)"; local t0=$SECONDS
  timeout -s KILL 720 .venv/bin/python -u -m ttt.run $EVAL --out "$RES/$name.json" "$@" > "$LOGS/login_$name.log" 2>&1
  local rc=$?
  if [ $rc -eq 0 ] && [ -e "$RES/$name.json" ]; then echo "[done]  $name $((SECONDS - t0))s $(grep -E '^\[eval\]' "$LOGS/login_$name.log" | tail -1 | sed 's/ peak=.*//')"
  else echo "[FAIL]  $name exit=$rc -> $LOGS/login_$name.log"; failed+=("$name"); fi
}
K1024="--window 1024 --prefix-segment 1024"
run_eval SP_A_32k_k1024 --arm A $K1024
run_eval SP_B_32k_k1024_lr4e-6 --arm B --inner normalized_sgd --inner-lr 4e-6 $K1024

# context_value.py --skip-sequences is new. Prove the slicing on a number already on record:
# arXiv sequences [12, 24) must reproduce differences[12:24] of the 24-sequence arXiv run.
ref="$RES/context_value_slimpajama_RedPajamaArXiv_s1024.json"; chk="$RES/context_value_slicecheck_arxiv_s1024_skip12.json"
if [ ! -e "$chk" ]; then
  echo "[start] slicing self-check $(date +%H:%M:%S)"
  timeout -s KILL 600 .venv/bin/python -u scripts/context_value.py --data "$DATA" --only-label RedPajamaArXiv --segment 1024 \
      --eval-sequences 12 --skip-sequences 12 --out "$chk" > "$LOGS/login_cv_slicecheck.log" 2>&1 || echo "[FAIL]  slicing self-check run"
fi
.venv/bin/python - "$ref" "$chk" <<'PY' || { echo "[FAIL]  slicing self-check: pieces would not be trusted. Stopping."; exit 1; }
import json, sys
ref, chk = (json.load(open(p)) for p in sys.argv[1:3])
for band in ref["bands"]:
    want, got = ref["bands"][band]["differences"][12:24], chk["bands"][band]["differences"]
    assert len(want) == len(got) == 12, (len(want), len(got))
    worst = max(abs(w - g) for w, g in zip(want, got))
    assert worst < 1e-4, f"band {band}: piece differs from the full run by {worst:.2e}"
assert chk["documents"] == ref["documents"][12:24], "piece scored different documents"
print(f"[done]  slicing self-check: sequences [12, 24) reproduce the 24-sequence run (< 1e-4 nats)")
PY

for skip in 0 24 48 72; do
  out="$RES/context_value_slimpajama_first96_s1024_skip$skip.json"
  [ -e "$out" ] && { echo "[skip] context_value skip=$skip"; continue; }
  echo "[start] context_value S=1024 sequences [$skip, $((skip + 24))) $(date +%H:%M:%S)"
  timeout -s KILL 780 .venv/bin/python -u scripts/context_value.py --data "$DATA" --segment 1024 --eval-sequences 24 \
      --skip-sequences "$skip" --out "$out" > "$LOGS/login_cv_sp_first96_skip$skip.log" 2>&1
  rc=$?; echo "[done]  context_value skip=$skip exit=$rc"; [ $rc -eq 0 ] || failed+=("cv_skip$skip")
  grep -E 'recent context >= +896' "$LOGS/login_cv_sp_first96_skip$skip.log" | cut -c1-200
done

# The inner-LR scan comes last: it informs, but nothing else waits for it.
for lr in 2e-6 7e-6 2e-5; do
  run_eval "SP_B_32k_k1024_lr$lr" --arm B --inner normalized_sgd --inner-lr "$lr" $K1024
done
echo "=== login_slimpajama_k1024: failed=${#failed[@]} ${failed[*]:-} ==="
