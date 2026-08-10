#!/bin/bash
#SBATCH --job-name=export_3cgs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=48:00:00
#SBATCH --output=logs/export_3cgs-%j.out
#SBATCH --error=logs/export_3cgs-%j.err
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --requeue

set -euo pipefail

REPO=${REPO:-$HOME/RecSys-challenge-2026}
PKG=$REPO/src/basic_candidate_generators

URM_MODE=${URM_MODE:-session}
OBJECTIVE=${OBJECTIVE:-ndcg}
OBJECTIVE_K=${OBJECTIVE_K:-20}
TOP_K=${TOP_K:-200}
N_FOLDS=${N_FOLDS:-5}

HEURISTIC_MODEL=${HEURISTIC_MODEL:-heuristic}
BERT_MODEL=${BERT_MODEL:-split_hidim_xattn_hardneg_query_full}
TWO_TOWER_MODEL=${TWO_TOWER_MODEL:-two_tower_v2_improved}

echo "[$(date)] node=$(hostname) job=${SLURM_JOB_ID:-local}"
echo "REPO=$REPO"
echo "PKG=$PKG"
echo "OBJECTIVE=${OBJECTIVE}@${OBJECTIVE_K}"
echo "MODELS: $HEURISTIC_MODEL  $BERT_MODEL  $TWO_TOWER_MODEL"

mkdir -p "$PKG/logs"
cd "$PKG"

export PATH="$HOME/.local/bin:$PATH"
unset VIRTUAL_ENV

# Compute nodes are offline: use only packages/models already present in the
# local uv environment and Hugging Face cache.
export UV_OFFLINE=1
export HF_HOME="${HF_HOME:-$REPO/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
unset TRANSFORMERS_CACHE
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found. Install/sync the environment on the login node before sbatch."
    exit 1
fi

uv sync --frozen --offline

uv run python - <<'PY'
import importlib
missing = []
for module in ("optuna", "yaml", "polars", "torch"):
    try:
        importlib.import_module(module)
    except Exception:
        missing.append(module)
if missing:
    raise SystemExit(
        "Missing Python modules in .venv: "
        + ", ".join(missing)
        + "\nPrepare the uv environment on the login node before submitting."
    )
PY

echo "--- GPU check ---"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
uv run python -c "import torch; print('cuda:', torch.cuda.is_available(), '| device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
echo "-----------------"

extract_best() {
    local model="$1"
    echo "[$(date)] === EXTRACT best params: ${model}/${URM_MODE} (${OBJECTIVE}@${OBJECTIVE_K}) ==="
    uv run python -u -m launchers_crossvalidation.extract_best_params \
        --model "$model" \
        --urm_mode "$URM_MODE" \
        --source cv \
        --objective "$OBJECTIVE" \
        --objective_k "$OBJECTIVE_K"
}

retrain_export() {
    local model="$1"
    echo "[$(date)] === RETRAIN + EXPORT: ${model}/${URM_MODE} (${OBJECTIVE}@${OBJECTIVE_K}) ==="
    srun --unbuffered uv run python -u -m launchers_crossvalidation.retrain_and_export \
        --model "$model" \
        --urm_mode "$URM_MODE" \
        --objective "$OBJECTIVE" \
        --objective_k "$OBJECTIVE_K" \
        --top_k "$TOP_K" \
        --n_folds "$N_FOLDS"
}

# Export best params for models that are tuned via Optuna DBs. The two-tower
# improved config is expected to already exist because it comes from the patched
# two_tower_v2 setup and may share/rename the two_tower_v2_session directory.
extract_best "$HEURISTIC_MODEL"
extract_best "$BERT_MODEL"

for model in "$HEURISTIC_MODEL" "$BERT_MODEL" "$TWO_TOWER_MODEL"; do
    cfg="configs/cv_best_${model}_${URM_MODE}_${OBJECTIVE}${OBJECTIVE_K}.yaml"
    if [[ ! -f "$cfg" ]]; then
        echo "Missing required config: $PKG/$cfg"
        echo "Run extract_best_params first, or copy the existing best-params YAML to this name."
        exit 1
    fi
done

retrain_export "$HEURISTIC_MODEL"
retrain_export "$BERT_MODEL"
retrain_export "$TWO_TOWER_MODEL"

echo "[$(date)] === DONE ==="
echo "Submissions:"
echo "  $REPO/models/CG_crossvalidation/${HEURISTIC_MODEL}_${URM_MODE}/submission/blind_A_${HEURISTIC_MODEL}_${URM_MODE}.json"
echo "  $REPO/models/CG_crossvalidation/${BERT_MODEL}_${URM_MODE}/submission/blind_A_${BERT_MODEL}_${URM_MODE}.json"
echo "  $REPO/models/CG_crossvalidation/${TWO_TOWER_MODEL}_${URM_MODE}/submission/blind_A_${TWO_TOWER_MODEL}_${URM_MODE}.json"
