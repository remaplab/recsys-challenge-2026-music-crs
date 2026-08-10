#!/bin/bash
#SBATCH --job-name=cv_slim_bpr_session
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20GB
#SBATCH --time=12:00:00
#SBATCH --output=logs/cv_slim_bpr_session-%j.out
#SBATCH --error=logs/cv_slim_bpr_session-%j.err
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
    --model slim_bpr \
    --urm_mode session

echo "[$(date)] done"
