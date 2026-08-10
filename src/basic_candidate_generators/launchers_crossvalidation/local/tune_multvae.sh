#!/usr/bin/env bash
set -e

# GPU: RTX 4060ti 16GB — fits MultVAE fine at batch_size=512
REPO=/mnt/Locale/Progetti/RecSys-challenge-2026
PKG=$REPO/src/basic_candidate_generators

cd $PKG

for mode in session user; do
    echo "[$(date)] multvae urm_mode=$mode"
    uv run python -m launchers_crossvalidation.tune_crossvalidation \
        --monitor \
        --model multvae \
        --urm_mode $mode
    echo "[$(date)] done $mode"
done
