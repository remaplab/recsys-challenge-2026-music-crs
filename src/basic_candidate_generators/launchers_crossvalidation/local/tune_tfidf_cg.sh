#!/usr/bin/env bash
set -e

REPO=/mnt/Locale/Progetti/RecSys-challenge-2026
PKG=$REPO/src/basic_candidate_generators

cd $PKG

echo "[$(date)] tfidf_cg urm_mode=session"
uv run python -m launchers_crossvalidation.tune_crossvalidation \
    --monitor \
    --model tfidf_cg \
    --urm_mode session
echo "[$(date)] done"
