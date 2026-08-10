#!/usr/bin/env bash
set -e

REPO=/mnt/Locale/Progetti/RecSys-challenge-2026
PKG=$REPO/src/basic_candidate_generators

cd $PKG

for mode in session user; do
    echo "[$(date)] gf_cf urm_mode=$mode"
    uv run python -m launchers_crossvalidation.tune_crossvalidation \
        --monitor \
        --model gf_cf \
        --urm_mode $mode
    echo "[$(date)] done $mode"
done
