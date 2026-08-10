#!/bin/bash
#SBATCH --job-name=cv_item_knn_session
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH --time=12:00:00
#SBATCH --output=logs/cv_item_knn_session-%j.out
#SBATCH --error=logs/cv_item_knn_session-%j.err
#SBATCH --account=studenti
#SBATCH --partition=all-nodes
#SBATCH --requeue

REPO=/SLURM_shared/st_challenge26/nicolo/recsys-challenge-2026
PKG=$REPO/src/basic_candidate_generators

echo "[$(date)] node=$(hostname) job=$SLURM_JOB_ID"
mkdir -p $PKG/logs
cd $PKG

uv lock && uv sync

srun --unbuffered uv run python -u -m launchers_dro.tune_crossvalidation_dro \
    --model item_knn \
    --urm_mode session

echo "[$(date)] done"
