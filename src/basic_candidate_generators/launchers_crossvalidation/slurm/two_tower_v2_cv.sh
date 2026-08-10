#!/bin/bash
#SBATCH --job-name=cv_two_tower_v2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --time=24:00:00
#SBATCH --output=logs/cv_two_tower_v2-%j.out
#SBATCH --error=logs/cv_two_tower_v2-%j.err
#SBATCH --account=studenti
#SBATCH --partition=all-nodes
#SBATCH --gres=gpu:1
#SBATCH --constraint=GPU_MEM_16GB
#SBATCH --requeue

REPO=/SLURM_shared/st_challenge26/andrea/recsys-challenge-2026
PKG=$REPO/src/basic_candidate_generators

echo "[$(date)] node=$(hostname) job=$SLURM_JOB_ID"
mkdir -p $PKG/logs
cd $PKG

uv lock && uv sync

# sentence-transformers e faiss-cpu non sono in pyproject.toml
uv pip install sentence-transformers faiss-cpu

echo "--- GPU check ---"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
uv run python -c "import torch; print('cuda:', torch.cuda.is_available(), '| device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
echo "-----------------"

RUN_ID=$SLURM_JOB_ID   # stabile anche su --requeue; cambia ad ogni sbatch

srun --unbuffered uv run python -u -m launchers_crossvalidation.tune_crossvalidation \
    --model two_tower_v2 \
    --urm_mode session \
    --n_trials 100 

echo "[$(date)] done"
