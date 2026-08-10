#!/bin/bash
#SBATCH --job-name=two_tower_v2_pipeline
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=48:00:00
#SBATCH --output=logs/two_tower_v2_pipeline-%j.out
#SBATCH --error=logs/two_tower_v2_pipeline-%j.err
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

# ── 1. Tuning ────────────────────────────────────────────────────────────────

echo "[$(date)] === TUNE session ==="
srun --unbuffered uv run python -u -m launchers_crossvalidation.tune_crossvalidation \
    --model two_tower_v2 \
    --urm_mode session \
    --n_trials 100

# ── 2. Extract best params ────────────────────────────────────────────────────

echo "[$(date)] === EXTRACT session ==="
uv run python -u -m launchers_crossvalidation.extract_best_params \
    --model two_tower_v2 \
    --urm_mode session \
    --source cv --objective ndcg --objective_k 20

# ── 3. Retrain + export ───────────────────────────────────────────────────────

echo "[$(date)] === RETRAIN session ==="
srun --unbuffered uv run python -u -m launchers_crossvalidation.retrain_and_export \
    --model two_tower_v2 \
    --urm_mode session \
    --objective ndcg --objective_k 20

echo "[$(date)] === PIPELINE DONE ==="
