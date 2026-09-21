#!/bin/bash
# SlimPajama at k = 1024, 40 steps: the four cells of the 2x2 on the first 96 validation
# sequences (the training jobs score only 32), then the per-domain / per-document analysis
# against what out-of-window context is worth (scripts/ttt_vs_context_value.py).
# Waits, detached, for the two training jobs' checkpoints and results; login GPU, short processes.
#   cd $ROOT; ( nohup scripts/della/login_sp_cells.sh > $LOGS/login_sp_cells.log 2>&1 < /dev/null & )
set -uo pipefail
ROOT=/scratch/gpfs/ARORA/hh9077/ttt-big-fast; RES=/scratch/gpfs/ARORA/hh9077/results
LOGS=/scratch/gpfs/ARORA/hh9077/logs; DATA=/scratch/gpfs/ARORA/hh9077/data/slimpajama_32k
export HF_HOME=/scratch/gpfs/ARORA/hh9077/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH="$ROOT"
cd "$ROOT"
META=SP_32k_k1024_t4_s40; PLAIN=SP_32k_k1024_ctl_s40
# The result file appears only after the job's own evaluation, i.e. once training is complete.
for i in $(seq 1 360); do [ -e "$RES/$META.json" ] && [ -e "$RES/$PLAIN.json" ] && break; sleep 30; done   # up to 3 h
[ -e "$RES/$META.json" ] && [ -e "$RES/$PLAIN.json" ] || { echo "[FAIL]  training results did not appear within 3 hours"; exit 1; }
while pgrep -u hh9077 -f 'login_slimpajama_k1024|login_recall_k1024|login_cell_|login_cv_' > /dev/null; do sleep 20; done

CELL="--arm C --mode eval --data $DATA --seq-len 32768 --chunk 1024 --window 1024 --prefix-segment 1024 --fast-blocks 4
      --remat-group 1 --remat-blocks --dtype bf16 --lora-rank 64 --inner normalized_sgd --inner-lr 4e-6
      --eval-sequences 96 --eval-ttt-off"
for spec in "cell_sp_k1024_meta_40|$META" "cell_sp_k1024_plainft_40|$PLAIN"; do
  IFS='|' read -r name ckpt <<< "$spec"
  [ -e "$RES/$name.json" ] && { echo "[skip] $name"; continue; }
  echo "[start] $name $(date +%H:%M:%S)"; t0=$SECONDS
  timeout -s KILL 780 .venv/bin/python -u -m ttt.run $CELL --load-slow "$RES/$ckpt.ckpt" --out "$RES/$name.json" > "$LOGS/login_$name.log" 2>&1; rc=$?
  if [ $rc -eq 0 ]; then echo "[done]  $name $((SECONDS - t0))s"; grep -E '^\[eval\]' "$LOGS/login_$name.log" | sed 's/ peak=.*//'
  else echo "[FAIL]  $name exit=$rc -> $LOGS/login_$name.log"; tail -3 "$LOGS/login_$name.log" | cut -c1-200; fi
done

# The first 32 of the 96 sequences are the ones each training job evaluated itself: the
# loaded weights must reproduce those numbers (A100 there, H100 here: allow 2e-3).
.venv/bin/python - "$RES" "$META" "$PLAIN" <<'PY' || { echo "[FAIL]  loaded weights do not reproduce the training jobs' own evaluation. Stopping."; exit 1; }
import json, sys
res, meta, plain = sys.argv[1:4]
load = lambda n: json.load(open(f"{res}/{n}.json"))
first32 = lambda r, block: sum(r[block]["per_sequence_loss"][:32]) / 32
checks = [("meta, TTT on", first32(load("cell_sp_k1024_meta_40"), "eval"), load(meta)["eval"]["loss"]),
          ("meta, TTT off", first32(load("cell_sp_k1024_meta_40"), "eval_ttt_off"), load(meta)["eval_ttt_off"]["loss"]),
          ("plain, TTT off", first32(load("cell_sp_k1024_plainft_40"), "eval_ttt_off"), load(plain)["eval"]["loss"])]
for name, here, there in checks:
    print(f"[check] {name:<15} first 32 of 96 here {here:.4f}   training job {there:.4f}   |diff| {abs(here - there):.1e}")
    assert abs(here - there) < 2e-3, name
PY

.venv/bin/python scripts/two_by_two.py --meta "$RES/cell_sp_k1024_meta_40.json" --plain "$RES/cell_sp_k1024_plainft_40.json"
echo
.venv/bin/python scripts/ttt_vs_context_value.py --data "$DATA" \
    --context-value "$RES"/context_value_slimpajama_first96_s1024_skip{0,24,48,72}.json \
    --untuned-a "$RES/SP_A_32k_k1024.json" --untuned-b "$RES/SP_B_32k_k1024_lr4e-6.json" \
    --meta "$RES/cell_sp_k1024_meta_40.json" --plain "$RES/cell_sp_k1024_plainft_40.json" \
    --out "$RES/ttt_vs_context_value_sp_k1024_s40.json"
echo "=== login_sp_cells finished ==="
