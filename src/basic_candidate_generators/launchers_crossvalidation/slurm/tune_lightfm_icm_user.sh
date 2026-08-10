#!/bin/bash
#SBATCH --job-name=cv_lightfm_icm_user
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=GPU_MEM_16GB
#SBATCH --output=logs/cv_lightfm_icm_user-%j.out
#SBATCH --error=logs/cv_lightfm_icm_user-%j.err
#SBATCH --account=studenti
#SBATCH --partition=all-nodes
#SBATCH --requeue

REPO=/SLURM_shared/st_challenge26/andrea/recsys-challenge-2026
PKG=$REPO/src/basic_candidate_generators

echo "[$(date)] node=$(hostname) job=$SLURM_JOB_ID"
mkdir -p $PKG/logs
cd $PKG

uv lock && uv sync

srun --unbuffered uv run python -u -m launchers_crossvalidation.tune_crossvalidation \
    --model lightfm_icm \
    --urm_mode user \
    --run_id $SLURM_JOB_ID

echo "[$(date)] done"
