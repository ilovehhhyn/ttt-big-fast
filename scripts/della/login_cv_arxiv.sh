#!/bin/bash
# arXiv (the domain with the largest ceiling at k=8192) at smaller windows. Detached; waits for
# round2_login.sh to release the login GPU. Each run is one short process.
set -uo pipefail
ROOT=/scratch/gpfs/ARORA/hh9077/ttt-big-fast; RES=/scratch/gpfs/ARORA/hh9077/results; LOGS=/scratch/gpfs/ARORA/hh9077/logs
export HF_HOME=/scratch/gpfs/ARORA/hh9077/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=$ROOT
cd $ROOT
while pgrep -u hh9077 -f round2_login.sh > /dev/null; do sleep 20; done
for spec in 'RedPajamaArXiv 2048' 'RedPajamaArXiv 1024' 'RedPajamaGithub 1024'; do
  set -- $spec; lab=$1; S=$2; out=$RES/context_value_slimpajama_${lab}_s$S.json
  [ -e $out ] && { echo "[skip] $lab S=$S"; continue; }
  echo "[start] $lab S=$S $(date +%H:%M:%S)"
  timeout -s KILL 780 .venv/bin/python -u scripts/context_value.py --data /scratch/gpfs/ARORA/hh9077/data/slimpajama_32k --only-label $lab --segment $S --eval-sequences 24 --out $out > $LOGS/login_cv_${lab}_s$S.log 2>&1
  echo "[done]  $lab S=$S exit=$?"; grep -E 'recent context' $LOGS/login_cv_${lab}_s$S.log | tail -1 | cut -c1-200
done
echo '=== login_cv_arxiv finished ==='
