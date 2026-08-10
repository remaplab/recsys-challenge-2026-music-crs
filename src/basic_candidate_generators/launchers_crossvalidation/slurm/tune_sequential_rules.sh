#!/bin/bash
#SBATCH --job-name=cv_seq_rules
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=2:00:00
#SBATCH --output=logs/cv_sequential_rules-%j.out
#SBATCH --error=logs/cv_sequential_rules-%j.err
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
    --model sequential_rules \
    --urm_mode session

echo "[$(date)] done"
