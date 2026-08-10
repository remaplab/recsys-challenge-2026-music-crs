#!/bin/bash
#SBATCH --job-name=ttv2_emb_splitk
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=24:00:00
#SBATCH --output=logs/export_two_tower_v2_splitk_embeddings-%j.out
#SBATCH --error=logs/export_two_tower_v2_splitk_embeddings-%j.err
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --requeue

set -euo pipefail

REPO=${REPO:-$HOME/RecSys-challenge-2026}
PKG=$REPO/src/basic_candidate_generators

BEST_PARAMS=${BEST_PARAMS:-models/CG_crossvalidation/two_tower_v2_session/best_params_two_tower_v2_session_ndcg20.yaml}
SPLITK_DIR=${SPLITK_DIR:-data/splitK}
TRACK_METADATA_PATH=${TRACK_METADATA_PATH:-data/talkpl-ai/TalkPlayData-Challenge-Track-Metadata/data/all_tracks-00000-of-00001.parquet}
OUTPUT_DIR=${OUTPUT_DIR:-models/CG_crossvalidation/two_tower_v2_session/embeddings_splitK}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-models/CG_crossvalidation/two_tower_v2_session/checkpoints}

N_FOLDS=${N_FOLDS:-5}
FIT_MISSING=${FIT_MISSING:-1}
OVERWRITE=${OVERWRITE:-0}
INCLUDE_INPUTS=${INCLUDE_INPUTS:-0}
INCLUDE_BLIND=${INCLUDE_BLIND:-0}
SKIP_HOLDOUT=${SKIP_HOLDOUT:-0}
DEVICE=${DEVICE:-auto}

echo "[$(date)] node=$(hostname) job=${SLURM_JOB_ID:-local}"
echo "REPO=$REPO"
echo "PKG=$PKG"
echo "BEST_PARAMS=$BEST_PARAMS"
echo "SPLITK_DIR=$SPLITK_DIR"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "CHECKPOINT_DIR=$CHECKPOINT_DIR"

mkdir -p "$PKG/logs"
cd "$PKG"

export PATH="$HOME/.local/bin:$PATH"
unset VIRTUAL_ENV

export HF_HOME="${HF_HOME:-$REPO/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}

if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found. Install it on the login node with:"
    echo "curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo "[$(date)] === BASIC CANDIDATE GENERATORS ENV ==="
uv sync --frozen --offline

uv run python - <<'PY'
missing = []
for module in ("torch", "faiss", "polars", "yaml", "sentence_transformers"):
    try:
        __import__(module)
    except ImportError:
        missing.append(module)
if missing:
    raise SystemExit(
        "Missing Python modules in .venv: "
        + ", ".join(missing)
        + "\nPrepare the environment on the login node before submitting."
    )

from sentence_transformers import SentenceTransformer
try:
    SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
except Exception as exc:
    raise SystemExit(
        "SentenceTransformer all-MiniLM-L6-v2 is not available in the offline HF cache.\n"
        "Prepare it once on a login node:\n"
        "  cd ~/RecSys-challenge-2026/src/basic_candidate_generators\n"
        "  export HF_HOME=~/RecSys-challenge-2026/.cache/huggingface\n"
        "  export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0\n"
        "  uv run python -c 'from sentence_transformers import SentenceTransformer; "
        "SentenceTransformer(\"sentence-transformers/all-MiniLM-L6-v2\")'\n"
        f"Original error: {exc}"
    )
PY

echo "--- GPU check ---"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
uv run python -c "import torch; print('cuda:', torch.cuda.is_available(), '| device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
echo "-----------------"

ARGS=(
    --best-params "$BEST_PARAMS"
    --splitk-dir "$SPLITK_DIR"
    --track-metadata-path "$TRACK_METADATA_PATH"
    --output-dir "$OUTPUT_DIR"
    --checkpoint-dir "$CHECKPOINT_DIR"
    --n-folds "$N_FOLDS"
)

if [[ "$FIT_MISSING" == "1" ]]; then
    ARGS+=(--fit-missing)
fi

if [[ "$OVERWRITE" == "1" ]]; then
    ARGS+=(--overwrite)
fi

if [[ "$INCLUDE_INPUTS" == "1" ]]; then
    ARGS+=(--include-inputs)
fi

if [[ "$INCLUDE_BLIND" == "1" ]]; then
    ARGS+=(--include-blind)
fi

if [[ "$SKIP_HOLDOUT" == "1" ]]; then
    ARGS+=(--skip-holdout)
fi

if [[ "$DEVICE" != "auto" ]]; then
    ARGS+=(--device "$DEVICE")
fi

echo "[$(date)] === EXPORT TWO_TOWER_V2 SPLITK EMBEDDINGS ==="
printf 'python args:'
printf ' %q' "${ARGS[@]}"
printf '\n'

srun --unbuffered uv run python -u -m launchers_crossvalidation.export_two_tower_v2_splitk_embeddings "${ARGS[@]}"

echo "[$(date)] === DONE ==="
echo "Embeddings written under:"
echo "  $REPO/$OUTPUT_DIR"
