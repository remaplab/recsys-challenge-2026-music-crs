#!/bin/bash
#SBATCH --job-name=cv_session_knn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=6GB
#SBATCH --time=4:00:00
#SBATCH --output=logs/cv_session_knn-%j.out
#SBATCH --error=logs/cv_session_knn-%j.err
#SBATCH --account=studenti
#SBATCH --partition=all-nodes
#SBATCH --requeue

REPO=/SLURM_shared/st_challenge26/nicolo/recsys-challenge-2026
PKG=$REPO/src/basic_candidate_generators

echo "[$(date)] node=$(hostname) job=$SLURM_JOB_ID"
mkdir -p $PKG/logs
cd $PKG

uv lock && uv sync

srun --unbuffered uv run python -u -m launchers_crossvalidation.tune_crossvalidation \
    --model session_knn \
    --urm_mode session

echo "[$(date)] done"
