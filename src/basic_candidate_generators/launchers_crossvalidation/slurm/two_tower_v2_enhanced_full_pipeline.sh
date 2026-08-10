#!/bin/bash
#SBATCH --job-name=ttv2_enhanced_pipeline
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=24:00:00
#SBATCH --output=logs/two_tower_v2_enhanced_pipeline-%j.out
#SBATCH --error=logs/two_tower_v2_enhanced_pipeline-%j.err
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --requeue

set -euo pipefail

REPO=${REPO:-$HOME/RecSys-challenge-2026}
EMB_PKG=$REPO/src/embeddings-package
CG_PKG=$REPO/src/basic_candidate_generators
MODEL=two_tower_v2_enhanced
URM_MODE=session
OBJECTIVE=${OBJECTIVE:-ndcg}
OBJECTIVE_K=${OBJECTIVE_K:-20}
N_TRIALS=${N_TRIALS:-100}
EMB_MODELS=${EMB_MODELS:-qwen3_0p6b}
EMB_STAGES=${EMB_STAGES:-tracks}
QWEN_MODEL=${QWEN_MODEL:-Qwen/Qwen3-Embedding-0.6B}
SKIP_IMPUTE=${SKIP_IMPUTE:-0}
SKIP_QWEN_CACHE=${SKIP_QWEN_CACHE:-0}
ALLOW_DOWNLOADS=${ALLOW_DOWNLOADS:-0}

echo "[$(date)] node=$(hostname) job=${SLURM_JOB_ID:-local}"
mkdir -p "$CG_PKG/logs"

export PATH="$HOME/.local/bin:$PATH"
unset VIRTUAL_ENV

export HF_HOME="${HF_HOME:-$REPO/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
unset TRANSFORMERS_CACHE
if [[ "$ALLOW_DOWNLOADS" == "1" ]]; then
    export HF_HUB_OFFLINE=0
    export TRANSFORMERS_OFFLINE=0
else
    export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
    export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found. Install it on the login node with:"
    echo "curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo "--- GPU check ---"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "-----------------"

echo "[$(date)] === EMBEDDINGS PACKAGE ENV ==="
cd "$EMB_PKG"
if [[ -x "$EMB_PKG/.venv/bin/python" ]]; then
    echo "using existing embeddings-package .venv"
else
    if ! uv sync --offline; then
        cat <<'EOF'
Failed to create src/embeddings-package/.venv offline.

Prepare this environment once on a Leonardo login node with network/proxy enabled:

  cd ~/RecSys-challenge-2026/src/embeddings-package
  export HF_HOME=~/RecSys-challenge-2026/.cache/huggingface
  uv lock
  uv sync
  uv run python -c 'import torch, transformers; print(torch.__version__)'

Then resubmit the SLURM job. The compute-node job intentionally runs offline.
EOF
        exit 1
    fi
fi

echo "[$(date)] === CHECK QWEN CACHE ==="
if [[ "$ALLOW_DOWNLOADS" != "1" && ( "$SKIP_IMPUTE" != "1" || "$SKIP_QWEN_CACHE" != "1" ) ]]; then
    uv run python - <<PY
from transformers import AutoModel, AutoTokenizer
model = "$QWEN_MODEL"
try:
    AutoTokenizer.from_pretrained(model, local_files_only=True)
    AutoModel.from_pretrained(model, local_files_only=True)
except Exception as exc:
    raise SystemExit(
        "Missing Hugging Face cache for " + model + "\\n"
        "Prepare it once on a login node, then resubmit:\\n\\n"
        "  cd ~/RecSys-challenge-2026/src/embeddings-package\\n"
        "  export HF_HOME=~/RecSys-challenge-2026/.cache/huggingface\\n"
        "  export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0\\n"
        "  uv run python - <<'PY2'\\n"
        "from transformers import AutoModel, AutoTokenizer\\n"
        f"m = {model!r}\\n"
        "AutoTokenizer.from_pretrained(m)\\n"
        "AutoModel.from_pretrained(m)\\n"
        "PY2\\n\\n"
        "Original error: " + repr(exc)
    )
print("Qwen cache OK:", model)
PY
else
    echo "Qwen cache check skipped"
fi

echo "[$(date)] === IMPUTE MISSING METADATA-QWEN EMBEDDINGS ==="
IMPUTED_PATH="$REPO/data/imputed_embeddings/missing_track_metadata_qwen3_0.6b.parquet"
if [[ "$SKIP_IMPUTE" == "1" ]]; then
    echo "SKIP_IMPUTE=1, skipping metadata imputation"
elif [[ -f "$IMPUTED_PATH" && "${OVERWRITE_IMPUTED:-0}" != "1" ]]; then
    echo "imputed metadata already exists: $IMPUTED_PATH"
else
    srun --unbuffered uv run python -u "$REPO/scripts/launchers/impute_missing_track_metadata_embeddings.py" \
        --output-path "$IMPUTED_PATH" \
        --missing-mode metadata-empty \
        --model-name "$QWEN_MODEL" \
        --device auto
fi

echo "[$(date)] === GENERATE CLEAN QWEN TRACK/QUERY CACHES ==="
if [[ "$SKIP_QWEN_CACHE" == "1" ]]; then
    echo "SKIP_QWEN_CACHE=1, skipping clean Qwen cache generation"
else
    echo "models=$EMB_MODELS stages=$EMB_STAGES"
    QWEN_DOWNLOAD_ARGS=()
    if [[ "$ALLOW_DOWNLOADS" == "1" ]]; then
        QWEN_DOWNLOAD_ARGS+=(--allow-downloads)
    fi
    srun --unbuffered uv run python -u scripts/12_encode_gambling_caches.py \
        --models $EMB_MODELS \
        --stages $EMB_STAGES \
        --device auto \
        "${QWEN_DOWNLOAD_ARGS[@]}"
fi

echo "[$(date)] === BASIC CANDIDATE GENERATORS ENV ==="
cd "$CG_PKG"
if [[ -x "$CG_PKG/.venv/bin/python" ]]; then
    echo "using existing basic_candidate_generators .venv"
else
    if ! uv sync --frozen --offline; then
        cat <<'EOF'
Failed to create src/basic_candidate_generators/.venv offline.

Prepare this environment once on a Leonardo login node with network/proxy enabled:

  cd ~/RecSys-challenge-2026/src/basic_candidate_generators
  uv sync
  uv run python -c 'import torch, faiss, sentence_transformers; print(torch.__version__)'

Then resubmit the SLURM job.
EOF
        exit 1
    fi
fi

uv run python - <<'PY'
missing = []
for module in ("faiss", "sentence_transformers", "torch", "polars", "numba"):
    try:
        __import__(module)
    except ImportError:
        missing.append(module)
if missing:
    raise SystemExit(
        "Missing Python modules in the basic_candidate_generators environment: "
        + ", ".join(missing)
        + "\nPrepare the environment on the login node, then resubmit."
    )
PY

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
