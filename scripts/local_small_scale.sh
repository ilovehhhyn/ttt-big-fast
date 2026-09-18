#!/bin/bash
# Scale-reduced replication of arms A / B / C on one machine.
#
# Mirrors the structure of the 1B/8K experiment exactly so the comparison is
# like-for-like in shape, only smaller:
#   * 8 chunks per sequence            (2048 / 256, as 8192 / 1024)
#   * window == context                (SWA degenerates to full attention, as k=8192 at 8K)
#   * fast weights = last 1/4 of blocks (7 of 30, as 4 of 16)
# It does NOT replace the Della runs; it is a smaller, internally consistent check.
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL=HuggingFaceTB/SmolLM2-135M
DATA=localdata/dclm2k
DEV=${DEV:-cpu}
COMMON="--repo $MODEL --data $DATA --seq-len 2048 --chunk 256 --window 2048 \
        --fast-blocks 7 --remat-group 1 --remat-blocks --device $DEV --eval-sequences ${NEVAL:-16}"
mkdir -p results_local

echo "### arm A: no TTT"
python3 -m ttt.run --arm A --mode eval $COMMON --out results_local/A.json

# e2e-equivalent per-element step for 7 blocks x 3 matrices of 576x1536 = 18.6M params
# is 1/sqrt(18.6e6) = 2.32e-4. Scan around it, as on the cluster.
for lr in 7e-5 2.3e-4 7e-4; do
  echo "### arm B: TTT-naive, inner lr $lr"
  python3 -m ttt.run --arm B --mode eval $COMMON --inner normalized_sgd --inner-lr "$lr" \
      --out "results_local/B_lr${lr}.json"
done

echo "### arm C: meta-learned LoRA"
python3 -m ttt.run --arm C --mode train $COMMON --inner normalized_sgd \
    --inner-lr "${CLR:-7e-5}" --outer-lr "${COLR:-1e-3}" --lora-rank 64 \
    --steps "${CSTEPS:-24}" --tokens-per-step $((2048*4)) --out results_local/C.json

python3 scripts/collect_results.py results_local/ --markdown
