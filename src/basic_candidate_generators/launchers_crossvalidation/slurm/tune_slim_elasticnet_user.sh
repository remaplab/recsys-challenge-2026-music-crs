#!/bin/bash
#SBATCH --job-name=cv_slim_en_user
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=30GB
#SBATCH --time=20:00:00
#SBATCH --output=logs/cv_slim_elasticnet_user-%j.out
#SBATCH --error=logs/cv_slim_elasticnet_user-%j.err
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
    --model slim_elasticnet \
    --urm_mode user

echo "[$(date)] done"
