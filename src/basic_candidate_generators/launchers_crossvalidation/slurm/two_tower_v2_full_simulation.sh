#!/bin/bash
#SBATCH --job-name=ttv2_full_sim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=04:00:00
#SBATCH --output=logs/two_tower_v2_full_simulation-%j.out
#SBATCH --error=logs/two_tower_v2_full_simulation-%j.err
#SBATCH --account=studenti
#SBATCH --partition=all-nodes
#SBATCH --gres=gpu:1
#SBATCH --requeue

set -euo pipefail

REPO=${REPO:-$HOME/andrea/recsys-challenge-2026}
PKG="$REPO/src/basic_candidate_generators"

MODEL=${MODEL:-two_tower_v2_improved}
URM_MODE=${URM_MODE:-session}
TOP_K=${TOP_K:-20}
TEXT_MODEL=${TEXT_MODEL:-sentence-transformers/all-MiniLM-L6-v2}

INPUT_PARQUET=${INPUT_PARQUET:-"$REPO/data/talkpl-ai/TalkPlayData-Challenge-Blind-A/data/test-00000-of-00001.parquet"}
CHECKPOINT=${CHECKPOINT:-"$REPO/models/CG_crossvalidation/${MODEL}_${URM_MODE}/checkpoints/full.pkl"}
OUTPUT_DIR=${OUTPUT_DIR:-"$REPO/models/${MODEL}_full_sim_blind"}

echo "[$(date)] node=$(hostname) job=${SLURM_JOB_ID:-local}"
echo "REPO=$REPO"
echo "MODEL=$MODEL"
echo "URM_MODE=$URM_MODE"
echo "INPUT_PARQUET=$INPUT_PARQUET"
echo "CHECKPOINT=$CHECKPOINT"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "TOP_K=$TOP_K"
echo "TEXT_MODEL=$TEXT_MODEL"

mkdir -p "$PKG/logs"
cd "$PKG"

export PATH="$HOME/.local/bin:$PATH"
unset VIRTUAL_ENV

export HF_HOME="${HF_HOME:-$REPO/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
unset TRANSFORMERS_CACHE

if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found. Install it on the login node with:"
    echo "curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

if [ ! -f "$INPUT_PARQUET" ]; then
    echo "Input parquet not found: $INPUT_PARQUET"
    exit 1
fi

if [ ! -f "$CHECKPOINT" ]; then
    echo "Checkpoint not found: $CHECKPOINT"
    echo "Run retrain_and_export first, or pass CHECKPOINT=/path/to/full.pkl."
    exit 1
fi

uv sync --frozen --offline

TEXT_MODEL="$TEXT_MODEL" uv run python - <<'PY'
import os
missing = []
for module in ("faiss", "sentence_transformers", "torch", "polars"):
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
text_model = os.environ["TEXT_MODEL"]
print("HF_HOME=", os.environ.get("HF_HOME"))
print("HF_HUB_CACHE=", os.environ.get("HF_HUB_CACHE"))
print("TRANSFORMERS_OFFLINE=", os.environ.get("TRANSFORMERS_OFFLINE"))
print("Checking SentenceTransformer cache for:", text_model)
try:
    SentenceTransformer(text_model)
except Exception as exc:
    raise SystemExit(
        "SentenceTransformer model is not available in the offline cache.\n"
        f"Requested TEXT_MODEL={text_model}\n"
        "On the login node, prepare the same cache with:\n"
        f"  cd {os.getcwd()}\n"
        f"  export HF_HOME={os.environ.get('HF_HOME')}\n"
        "  unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE\n"
        f"  uv run python -c 'from sentence_transformers import SentenceTransformer; SentenceTransformer(\"{text_model}\")'\n"
        f"Original error: {exc}"
    )
PY

echo "--- GPU check ---"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
uv run python -c "import torch; print('cuda:', torch.cuda.is_available(), '| device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
echo "-----------------"

echo "[$(date)] === TWO-TOWER V2 FULL SIMULATION ==="
srun --unbuffered uv run python -u -m launchers_crossvalidation.export_two_tower_v2_text_embeddings \
    --input "$INPUT_PARQUET" \
    --output-dir "$OUTPUT_DIR" \
    --checkpoint "$CHECKPOINT" \
    --top-k "$TOP_K" \
    --text-model "$TEXT_MODEL" \
    --skip-text-encode \
    --overwrite

echo "[$(date)] === SIMULATION DONE ==="
echo "Outputs:"
echo "  $OUTPUT_DIR/predictions.parquet"
echo "  $OUTPUT_DIR/submission_like.json"
echo "  $OUTPUT_DIR/user_vectors.npy"
echo "  $OUTPUT_DIR/user_vectors_meta.parquet"
echo "  $OUTPUT_DIR/item_vectors.npy"
echo "  $OUTPUT_DIR/item_vectors_meta.parquet"
