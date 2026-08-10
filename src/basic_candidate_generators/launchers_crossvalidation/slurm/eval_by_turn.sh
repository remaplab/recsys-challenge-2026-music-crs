#!/bin/bash
#SBATCH --job-name=eval_by_turn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=6:00:00
#SBATCH --output=logs/eval_by_turn-%j.out
#SBATCH --error=logs/eval_by_turn-%j.err
#SBATCH --account=studenti
#SBATCH --partition=all-nodes
#SBATCH --gres=gpu:1
#SBATCH --constraint=GPU_MEM_16GB
#SBATCH --requeue

set -e

REPO=/SLURM_shared/st_challenge26/andrea/recsys-challenge-2026
PKG=$REPO/src/basic_candidate_generators

echo "[$(date)] node=$(hostname) job=$SLURM_JOB_ID"
mkdir -p $PKG/logs
cd $PKG

uv lock && uv sync
uv pip install sentence-transformers faiss-cpu

echo "--- GPU check ---"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
uv run python -c "import torch; print('cuda:', torch.cuda.is_available(), '| device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
echo "-----------------"

echo "[$(date)] === EVAL two_tower_v2 session ==="
srun --unbuffered uv run python -u -m launchers_crossvalidation.eval_by_turn \
    --model two_tower_v2 \
    --urm_mode session

echo "[$(date)] === EVAL lightfm_icm session ==="
srun --unbuffered uv run python -u -m launchers_crossvalidation.eval_by_turn \
    --model lightfm_icm \
    --urm_mode session

echo "[$(date)] === EVAL lightfm_icm user ==="
srun --unbuffered uv run python -u -m launchers_crossvalidation.eval_by_turn \
    --model lightfm_icm \
    --urm_mode user

echo "[$(date)] === EVAL BY TURN DONE ==="
