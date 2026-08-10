#!/usr/bin/env bash
set -e

REPO=/mnt/Locale/Progetti/RecSys-challenge-2026
PKG=$REPO/src/basic_candidate_generators

cd $PKG

echo "[$(date)] session_knn urm_mode=session"
uv run python -m launchers_crossvalidation.tune_crossvalidation \
    --monitor \
    --model session_knn \
    --urm_mode session
echo "[$(date)] done"
