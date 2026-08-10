#!/bin/bash
#SBATCH --job-name=ttv2_improved_pipeline
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=12:00:00
#SBATCH --output=logs/two_tower_v2_improved_pipeline-%j.out
#SBATCH --error=logs/two_tower_v2_improved_pipeline-%j.err
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --requeue

set -euo pipefail

REPO=${REPO:-$HOME/RecSys-challenge-2026}
PKG=$REPO/src/basic_candidate_generators
MODEL=two_tower_v2_improved
URM_MODE=session
OBJECTIVE=ndcg
OBJECTIVE_K=20
N_TRIALS=${N_TRIALS:-50}

echo "[$(date)] node=$(hostname) job=$SLURM_JOB_ID"
mkdir -p "$PKG/logs"
cd "$PKG"

export PATH="$HOME/.local/bin:$PATH"
unset VIRTUAL_ENV

export HF_HOME="${HF_HOME:-$REPO/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found. Install it on the login node with:"
    echo "curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Leonardo compute nodes may not have internet access. Prepare/update the
# environment on the login node, then keep the job offline and reproducible:
#   cd ~/RecSys-challenge-2026/src/basic_candidate_generators
#   uv add sentence-transformers
#   uv sync
#   HF_HOME=~/RecSys-challenge-2026/.cache/huggingface uv run python -c \
#     'from sentence_transformers import SentenceTransformer; SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")'
uv sync --frozen --offline

uv run python - <<'PY'
missing = []
for module in ("faiss", "sentence_transformers"):
    try:
        __import__(module)
    except ImportError:
        missing.append(module)
if missing:
    raise SystemExit(
        "Missing Python modules in .venv: "
        + ", ".join(missing)
        + "\nPrepare the environment on the login node before submitting the job."
    )

from sentence_transformers import SentenceTransformer
try:
    SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
except Exception as exc:
    raise SystemExit(
        "SentenceTransformer model is not available in the offline Hugging Face cache.\n"
        "Run this on the login node before submitting:\n"
        "  cd ~/RecSys-challenge-2026/src/basic_candidate_generators\n"
        "  export HF_HOME=~/RecSys-challenge-2026/.cache/huggingface\n"
        "  uv run python -c 'from sentence_transformers import SentenceTransformer; "
        "SentenceTransformer(\"sentence-transformers/all-MiniLM-L6-v2\")'\n"
        f"Original error: {exc}"
    )
PY

echo "--- GPU check ---"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
uv run python -c "import torch; print('cuda:', torch.cuda.is_available(), '| device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
echo "-----------------"

echo "[$(date)] === TUNE splitK: $MODEL/$URM_MODE ==="
srun --unbuffered uv run python -u -m launchers_crossvalidation.tune_crossvalidation \
    --model "$MODEL" \
    --urm_mode "$URM_MODE" \
    --n_trials "$N_TRIALS"

echo "[$(date)] === EXTRACT best params: $OBJECTIVE@$OBJECTIVE_K ==="
uv run python -u -m launchers_crossvalidation.extract_best_params \
    --model "$MODEL" \
    --urm_mode "$URM_MODE" \
    --source cv \
    --objective "$OBJECTIVE" \
    --objective_k "$OBJECTIVE_K"

echo "[$(date)] === RETRAIN + EXPORT with extracted best params ==="
srun --unbuffered uv run python -u -m launchers_crossvalidation.retrain_and_export \
    --model "$MODEL" \
    --urm_mode "$URM_MODE" \
    --objective "$OBJECTIVE" \
    --objective_k "$OBJECTIVE_K"

echo "[$(date)] === PIPELINE DONE ==="
echo "Outputs:"
echo "  models/CG_crossvalidation/${MODEL}_${URM_MODE}/best_params_${MODEL}_${URM_MODE}_${OBJECTIVE}${OBJECTIVE_K}.yaml"
echo "  models/CG_crossvalidation/${MODEL}_${URM_MODE}/datasets/"
echo "  models/CG_crossvalidation/${MODEL}_${URM_MODE}/submission/blind_A_${MODEL}_${URM_MODE}.json"
