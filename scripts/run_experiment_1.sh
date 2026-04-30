#!/usr/bin/env bash
# Reproduce Experiment 1 (scalability frontier).
#
# Trains MA-DBP once at n_train=4 and evaluates at n_deploy ∈ {2,4,6,8,10,12,16}
# across seeds {0..4}. One CSV per (method, scenario, n, seed). Aggregate
# with scripts/aggregate.py.
set -euo pipefail

CKPT="${CKPT:-checkpoints/madbp_navigation_v2_n4_epoch_01999.pth}"
SCENARIO="${SCENARIO:-navigation_v2}"
RESULTS="${RESULTS:-results/exp1}"

mkdir -p "$RESULTS"

for N in 2 4 6 8 10 12 16; do
  for SEED in 0 1 2 3 4; do
    OUT="$RESULTS/madp_${SCENARIO}_n${N}_s${SEED}.csv"
    if [[ -f "$OUT" ]]; then
      echo "skip $OUT (already exists)"
      continue
    fi
    echo "=== madp  n=$N  seed=$SEED ==="
    madbp-eval \
      --method madp \
      --scenario "$SCENARIO" \
      --n-agents "$N" \
      --seed "$SEED" \
      --n-episodes 50 \
      --checkpoint "$CKPT" \
      --out "$OUT"
  done
done

echo "done. CSVs in $RESULTS"
