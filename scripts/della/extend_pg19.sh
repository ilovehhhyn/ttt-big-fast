#!/bin/bash
# Extend the PG-19 32K TRAINING split, keeping the VALIDATION split byte-identical.
#
# pg19_32k was prepared with a 300M-token target; the matched extension budget consumes 760M,
# and the loader would silently cycle. Re-preparing with a larger target also grows the val
# split, and evaluation selects sequences from a permutation of the split's LENGTH, so every
# existing result would stop being comparable. Therefore:
#
#   1. prepare pg19_32k_full with the SAME dataset / split / min_doc_tokens / val_every / tokenizer;
#   2. PROVE compatibility: the new val stream must START WITH the old val stream's bytes. That
#      holds iff the source order and the keep/assign rule are unchanged, which is also what
#      keeps the new train split disjoint from the old val split (old val = kept documents with
#      k % val_every == 0 among the first N; new train = kept documents with k % val_every != 0);
#   3. replace the new val split by the old one, and record that this was done.
#
# Runs on the LOGIN node (needs the internet), CPU only, threads capped, niced, detached:
#   cd $ROOT; ( nohup scripts/della/extend_pg19.sh > $LOGS/extend_pg19.log 2>&1 < /dev/null & )
set -euo pipefail
ROOT=/scratch/gpfs/ARORA/hh9077/ttt-big-fast; LOGS=/scratch/gpfs/ARORA/hh9077/logs; DATA=/scratch/gpfs/ARORA/hh9077/data
OLD="$DATA/pg19_32k"; NEW="$DATA/pg19_32k_full"
export HF_HOME=/scratch/gpfs/ARORA/hh9077/hf PYTHONPATH="$ROOT" TOKENIZERS_PARALLELISM=true RAYON_NUM_THREADS=4 OMP_NUM_THREADS=4
cd "$ROOT"
say() { echo "[$(date +%m-%d\ %H:%M:%S)] $*"; }

if [ ! -e "$NEW/READY" ]; then
  say "preparing $NEW (target 9e8 train tokens; the matched budget needs 760,217,600)"
  nice -n 10 .venv/bin/python -m ttt.data.prepare --dataset emozilla/pg19 --split train --out-dir "$NEW" \
      --min-doc-tokens 32769 --target-tokens 9e8 --val-every 40 --tokenizer-id unsloth/Llama-3.2-1B \
      > "$LOGS/prep_pg19_32k_full.log" 2>&1
  say "prepare finished: $(tail -1 "$LOGS/prep_pg19_32k_full.log" | cut -c1-200)"

  # The proof. cmp -n compares exactly the first <size of old val> bytes.
  old_bytes=$(stat -c %s "$OLD/val.bin")
  new_bytes=$(stat -c %s "$NEW/val.bin")
  [ "$new_bytes" -ge "$old_bytes" ] || { say "ABORT: new val ($new_bytes B) is smaller than old val ($old_bytes B)"; exit 1; }
  if ! cmp -s -n "$old_bytes" "$OLD/val.bin" "$NEW/val.bin"; then
    say "ABORT: the new validation stream does not start with the old one. The source order or the"
    say "       keep/assign rule changed, so disjointness from the old val split is NOT guaranteed."
    exit 1
  fi
  say "verified: new val stream starts with the old val stream ($old_bytes bytes identical)"
  # The same argument for train, as a second witness: old train must be a prefix of new train.
  old_tb=$(stat -c %s "$OLD/train.bin")
  cmp -s -n "$old_tb" "$OLD/train.bin" "$NEW/train.bin" \
    && say "verified: new train stream starts with the old train stream ($old_tb bytes identical)" \
    || { say "ABORT: new train does not start with the old train"; exit 1; }

  mv "$NEW/val.bin" "$NEW/val_from_full_prepare.bin"; mv "$NEW/val.json" "$NEW/val_from_full_prepare.json"
  cp "$OLD/val.bin" "$NEW/val.bin"; cp "$OLD/val.json" "$NEW/val.json"
  cat > "$NEW/README" <<EOT
train.*: emozilla/pg19:train, min_doc_tokens 32769, val_every 40, target 9e8 tokens ($(date +%F)).
val.*:   COPIED from $OLD so that evaluation uses exactly the sequences of every earlier result.
         Verified before copying: the val and train streams of this prepare START WITH those of
         $OLD byte for byte, so this train split is disjoint from that val split.
val_from_full_prepare.*: the (larger) val split this prepare produced itself; unused.
EOT
  touch "$NEW/READY"
  say "READY: $(.venv/bin/python -c "import json;print(f\"train {json.load(open('$NEW/train.json'))['num_tokens']:,} tokens, val {json.load(open('$NEW/val.json'))['num_tokens']:,} tokens\")")"
else
  say "$NEW already READY"
fi

# The orchestrator refused to submit the matched-budget jobs for lack of data. Once any running
# instance has finished (it may still be waiting on the 2-GPU validation), run it again: it
# skips what is done and submits.
while pgrep -u hh9077 -f round2_start.sh > /dev/null; do sleep 60; done
say "re-running the orchestrator"
scripts/della/round2_start.sh >> "$LOGS/round2_start.log" 2>&1 || say "orchestrator exited non-zero; see round2_start.log"
say "=== extend_pg19 finished ==="
