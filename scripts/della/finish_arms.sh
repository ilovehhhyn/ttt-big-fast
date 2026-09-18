#!/bin/bash
# Finish and report arms A, C, E. Safe to run repeatedly; it only reads.
# Run ON della:   bash /scratch/gpfs/ARORA/hh9077/ttt-big-fast/scripts/della/finish_arms.sh
set -uo pipefail
ROOT=/scratch/gpfs/ARORA/hh9077/ttt-big-fast
LOGS=/scratch/gpfs/ARORA/hh9077/logs
cd "$ROOT"

echo "=== queued/running ==="
squeue -u "$USER" -o "%.10i %.16j %.9T %.8M %.20R" | grep -E "armC|JOBID" || echo "  none"

echo
echo "=== per-job training/eval lines ==="
for f in "$LOGS"/armCquick-*.out "$LOGS"/armC_olr*-*.out; do
  [ -e "$f" ] || continue
  echo "--- $(basename "$f") ---"
  grep -E "\[train\]|\[eval\]|TIME LIMIT|CANCELLED|OutOfMemory|Traceback" "$f" | tail -6
done

echo
echo "=== summary table ==="
.venv/bin/python scripts/collect_results.py results/ --markdown 2>/dev/null || \
  echo "  (no results yet)"
