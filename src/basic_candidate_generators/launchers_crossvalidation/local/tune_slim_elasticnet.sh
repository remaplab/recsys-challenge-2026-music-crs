#!/usr/bin/env bash
set -e

# workers=8 (fixed) × n_jobs=2 (optuna) = 16 threads total — saturates 8c/16t machine
REPO=/mnt/Locale/Progetti/RecSys-challenge-2026
PKG=$REPO/src/basic_candidate_generators

cd $PKG

for mode in session user; do
    echo "[$(date)] slim_elasticnet urm_mode=$mode"
    uv run python -m launchers_crossvalidation.tune_crossvalidation \
        --monitor \
        --model slim_elasticnet \
        --urm_mode $mode
    echo "[$(date)] done $mode"
done
