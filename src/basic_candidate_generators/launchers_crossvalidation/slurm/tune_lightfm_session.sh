#!/bin/bash
#SBATCH --job-name=cv_lightfm_session
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=24:00:00
#SBATCH --output=logs/cv_lightfm_session-%j.out
#SBATCH --error=logs/cv_lightfm_session-%j.err
#SBATCH --account=studenti
#SBATCH --partition=all-nodes
#SBATCH --requeue

REPO=/SLURM_shared/st_challenge26/andrea/recsys-challenge-2026
PKG=$REPO/src/basic_candidate_generators

echo "[$(date)] node=$(hostname) job=$SLURM_JOB_ID"
mkdir -p $PKG/logs
cd $PKG

uv lock && uv sync

# lightfm 1.17 su PyPI ha setup.py rotto con setuptools moderni.
# Il repo git ha il fix — installa da lì.
uv pip install git+https://github.com/lyst/lightfm.git

srun --unbuffered uv run python -u -m launchers_crossvalidation.tune_crossvalidation \
    --model lightfm \
    --urm_mode session

echo "[$(date)] done"
