#!/bin/bash
# Submit one long training run as a CHAIN of resumable jobs.
#
# A run longer than the wall limit cannot be one job. Every link runs the IDENTICAL command:
# the first starts the run, each later one resumes from the checkpoint (exact resume, see
# ttt/train/checkpoint.py), and a link that finds the run finished exits at once. Links depend
# on the previous one with "afterany", because a link that hits its wall limit ends in state
# TIMEOUT, which "afterok" would never satisfy.
#
#   scripts/della/submit_chain.sh <name> <links> <hours-per-link> <ttt.run args...>
#
# <links>: ceil(total training hours / (hours-per-link - 0.5)) plus one or two spare.
set -euo pipefail
NAME=$1; LINKS=$2; HOURS=$3; shift 3
ROOT=/scratch/gpfs/ARORA/hh9077/ttt-big-fast
RES=/scratch/gpfs/ARORA/hh9077/results
cd "$ROOT"
[[ "$LINKS" =~ ^[0-9]+$ && "$LINKS" -ge 1 ]] || { echo "links must be a positive integer, got '$LINKS'"; exit 2; }
for a in "$@"; do [ "$a" = "--out" ] && { echo "do not pass --out: the chain sets it from <name>"; exit 2; }; done

prev=""
for i in $(seq 1 "$LINKS"); do
  dep=(); [ -n "$prev" ] && dep=(--dependency="afterany:$prev")
  prev=$(sbatch --parsable --job-name="${NAME}_$i" --time="${HOURS}:00:00" "${dep[@]}" \
         scripts/della/run_arm.sbatch "$@" --out "$RES/$NAME.json")
  echo "link $i/$LINKS: job $prev${dep:+  (${dep[*]})}"
done
echo "chain '$NAME': $LINKS links x ${HOURS}h -> $RES/$NAME.json (checkpoint $RES/$NAME.ckpt)"
