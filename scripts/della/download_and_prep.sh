#!/bin/bash
# Download the raw parquet shards on the LOGIN node (network I/O: little CPU), then hand the
# tokenisation to CPU compute jobs. Detached:
#   cd $ROOT; ( nohup scripts/della/download_and_prep.sh > $LOGS/download_and_prep.log 2>&1 < /dev/null & )
set -uo pipefail
ROOT=/scratch/gpfs/ARORA/hh9077/ttt-big-fast; LOGS=/scratch/gpfs/ARORA/hh9077/logs; DATA=/scratch/gpfs/ARORA/hh9077/data
export HF_HOME=/scratch/gpfs/ARORA/hh9077/hf OMP_NUM_THREADS=2
cd "$ROOT"
say() { echo "[$(date +%m-%d\ %H:%M:%S)] $*"; }
fetch() {   # fetch <repo> <local dir>; one process per dataset keeps each one's CPU time small
  [ -e "$2/DOWNLOADED" ] && { say "$1 already downloaded"; return; }
  say "downloading $1"
  .venv/bin/python - "$1" "$2" <<'PY' || { say "download of $1 FAILED"; return 1; }
import sys
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1], repo_type="dataset", allow_patterns=["data/train-*.parquet"], local_dir=sys.argv[2], max_workers=4)
PY
  touch "$2/DOWNLOADED"; say "$1: $(ls "$2"/data/train-*.parquet | wc -l) shards, $(du -sh "$2" | cut -f1)"
}
fetch emozilla/pg19        "$DATA/raw/pg19"        || exit 1
fetch DKYoon/SlimPajama-6B "$DATA/raw/slimpajama6b" || exit 1

submit() {  # submit <job name> <done marker> <sbatch script and args...>
  local name=$1 marker=$2; shift 2
  [ -e "$marker" ] && { say "$name already done"; return; }
  squeue -u hh9077 -h -o %j | grep -qx "$name" && { say "$name already queued"; return; }
  say "submitted $name: job $(sbatch --parsable --job-name="$name" "$@")"
}
submit extend-pg19 "$DATA/pg19_32k_full/READY" scripts/della/extend_pg19.sbatch
submit prep-slimpj "$DATA/slimpajama_32k/val_docs.json" scripts/della/prep_cpu.sbatch \
    --dataset DKYoon/SlimPajama-6B --split train --data-files "$DATA/raw/slimpajama6b/data/train-*.parquet" \
    --out-dir "$DATA/slimpajama_32k" --min-doc-tokens 32769 --target-tokens 2e9 --val-every 10 \
    --label-field meta.redpajama_set_name
submit prep-pg128k "$DATA/pg19_128k/val.json" scripts/della/prep_cpu.sbatch \
    --dataset emozilla/pg19 --split train --data-files "$DATA/raw/pg19/data/train-*.parquet" \
    --out-dir "$DATA/pg19_128k" --min-doc-tokens 131073 --target-tokens 2e9 --val-every 25
say "=== download_and_prep finished ==="
