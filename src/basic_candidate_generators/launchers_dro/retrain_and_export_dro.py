"""Retrain best DRO model and export OOF candidate parquets for the reranker.

Wraps `launchers_crossvalidation.retrain_and_export._export_one` so the OOF /
holdout / blind candidate generation, checkpoints, and submission JSON are
identical to the non-DRO pipeline. Difference: output dir is
`{storage_dir}/{model}_{urm}_dro/` (so reranker dataset builders can pick up
robust-tuned candidates side-by-side with the recall@200-tuned ones).

Reads `configs/cv_best_{model}_{urm}_dro{suffix}.yaml` (written by
`extract_best_params_dro.py`). `suffix = _cvar70`, `_mean`, `_group_dro`.

Run:
    cd src/basic_candidate_generators
    uv run python -m launchers_dro.retrain_and_export_dro \\
        --model item_knn --urm_mode session [--top_k 200]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parent.parent
_SRC_ROOT = _PKG_ROOT / "src"
_CV_ROOT  = _PKG_ROOT / "launchers_crossvalidation"
_DRO_ROOT = _PKG_ROOT / "launchers_dro"

for p in (_SRC_ROOT, _CV_ROOT, _DRO_ROOT):
    sys.path.insert(0, str(p))

# Reuse the existing exporter end-to-end. We just point it at the DRO YAML
# and a `_dro`-suffixed folder so the non-DRO pipeline stays untouched.
from launchers_crossvalidation.retrain_and_export import (  # noqa: E402
    _export_one,
)
from _cv_utils import pkg_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Retrain best DRO model + export.")
    p.add_argument("--model",    required=True)
    p.add_argument("--urm_mode", required=True, choices=["session", "user"])
    p.add_argument("--robust_mode", default="cvar",
                   choices=["mean", "cvar", "group_dro"])
    p.add_argument("--robust_alpha", type=float, default=0.7)
    p.add_argument("--top_k", type=int, default=200)
    p.add_argument("--storage_dir", default="models/CG_crossvalidation")
    p.add_argument("--splitk_dir",  default="data/splitK")
    p.add_argument("--n_folds",     type=int, default=5)
    p.add_argument("--skip_datasets",    action="store_true")
    p.add_argument("--skip_checkpoints", action="store_true")
    p.add_argument("--skip_submission",  action="store_true")
    p.add_argument("--skip_holdout_candidates", action="store_true")
    p.add_argument("--skip_blind_candidates",   action="store_true")
    return p.parse_args()


def _suffix(robust_mode: str, robust_alpha: float) -> str:
    if robust_mode == "cvar":
        return f"_cvar{int(round(robust_alpha * 100))}"
    return f"_{robust_mode}"


def main() -> None:
    args = parse_args()
    suffix = _suffix(args.robust_mode, args.robust_alpha)

    # Prefer the YAML living next to the optuna DB (mirrors non-DRO pipeline),
    # fall back to the legacy package configs/ copy.
    _REPO_ROOT = Path(__file__).resolve().parents[3]
    storage_yaml = (
        _REPO_ROOT / args.storage_dir / f"{args.model}_{args.urm_mode}_dro"
        / f"best_params_{args.model}_{args.urm_mode}_dro{suffix}.yaml"
    )
    legacy_yaml = pkg_path(
        f"configs/cv_best_{args.model}_{args.urm_mode}_dro{suffix}.yaml"
    )
    cfg_path = storage_yaml if storage_yaml.exists() else legacy_yaml
    if not cfg_path.exists():
        sys.exit(
            f"[export-dro] missing config: tried {storage_yaml} and {legacy_yaml}. "
            "Run extract_best_params_dro.py first."
        )

    class _ShimArgs:
        pass

    shim = _ShimArgs()
    shim.model = args.model
    shim.urm_mode = args.urm_mode
    shim.top_k = args.top_k
    shim.storage_dir = args.storage_dir
    shim.splitk_dir = args.splitk_dir
    shim.n_folds = args.n_folds
    shim.skip_datasets = args.skip_datasets
    shim.skip_checkpoints = args.skip_checkpoints
    shim.skip_submission = args.skip_submission
    shim.skip_holdout_candidates = args.skip_holdout_candidates
    shim.skip_blind_candidates = args.skip_blind_candidates
    shim.objective = "ndcg"
    shim.objective_k = 20

    # Folder = `{model}_{urm}_dro` so DRO + non-DRO sit side-by-side.
    folder_key = f"{args.model}_{args.urm_mode}_dro"
    _export_one(
        args.model, args.urm_mode, cfg_path, shim, folder_key=folder_key,
    )


if __name__ == "__main__":
    main()
