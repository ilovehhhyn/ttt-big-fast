#!/bin/bash
# Round 2 data preparation (needs the internet, so it runs on the LOGIN node; CPU only).
# Tokenizer threads are capped and the job is niced: this is a shared node.
#   cd $ROOT; ( nohup scripts/della/round2_data.sh > $LOGS/round2_data.log 2>&1 < /dev/null & )
set -uo pipefail
ROOT=/scratch/gpfs/ARORA/hh9077/ttt-big-fast; LOGS=/scratch/gpfs/ARORA/hh9077/logs; DATA=/scratch/gpfs/ARORA/hh9077/data
export HF_HOME=/scratch/gpfs/ARORA/hh9077/hf PYTHONPATH="$ROOT" TOKENIZERS_PARALLELISM=true RAYON_NUM_THREADS=4
cd "$ROOT"
prep() {   # prep <name> <done-marker> <prepare args...>
  local name=$1 marker=$2; shift 2
  [ -e "$DATA/$name/$marker" ] && { echo "[skip] $name"; return; }
  echo "[start] prepare $name $(date +%H:%M:%S)"
  nice -n 10 .venv/bin/python -m ttt.data.prepare --out-dir "$DATA/$name" --tokenizer-id unsloth/Llama-3.2-1B \
      --target-tokens 2e9 "$@" > "$LOGS/prep_$name.log" 2>&1
  echo "[done]  prepare $name exit=$? $(date +%H:%M:%S)"; tail -2 "$LOGS/prep_$name.log" | cut -c1-220
}
# SlimPajama (free, ungated): documents of at least 32,769 tokens, with their source domain.
# The exact byte-length prefilter skips the tokenizer for nearly every web document.
prep slimpajama_32k val_docs.json --dataset DKYoon/SlimPajama-6B --split train --min-doc-tokens 32769 \
     --val-every 10 --label-field meta.redpajama_set_name
# PG-19 at 128K: only books of at least 131,073 tokens, so a sequence never spans two books.
prep pg19_128k val.json --dataset emozilla/pg19 --split train --min-doc-tokens 131073 --val-every 25
echo "=== round2_data finished ==="
