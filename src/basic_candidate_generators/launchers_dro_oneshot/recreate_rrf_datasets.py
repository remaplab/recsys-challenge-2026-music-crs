"""Recreate the ``rrf_oneshot`` datasets from a SAVED best-params yaml.

``tune_rrf_oneshot.py --export`` re-touches the Optuna study before fusing. This
launcher skips tuning entirely: it loads a frozen
``best_params_rrf_oneshot_*.yaml`` (written by the tuner) and re-runs the SAME
fusion export — fusing every split with those weights → the canonical CG layout
under ``rrf_oneshot/datasets/`` (fold_*_oof_cg_val, fold_*_oof_reranker_val,
holdout_candidates, blind_candidates) + the blind submission JSON.

The fusion logic is reused verbatim from ``tune_rrf_oneshot._export`` so there is
one source of truth; only the parameter source differs (yaml vs live study).

Run:
    cd src/basic_candidate_generators
    uv run python -m launchers_dro_oneshot.recreate_rrf_datasets \\
        --best models/CG_crossvalidation/rrf_oneshot/best_params_rrf_oneshot_cvar70.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parent.parent
_SRC_ROOT = _PKG_ROOT / "src"
_CV_ROOT  = _PKG_ROOT / "launchers_crossvalidation"
_DRO_ROOT = _PKG_ROOT / "launchers_dro"
_OS_ROOT  = _PKG_ROOT / "launchers_dro_oneshot"
_REPO_ROOT = _PKG_ROOT.parent.parent
_LBO_SRC = _REPO_ROOT / "src" / "lower_bound_optimization" / "src"
for p in (_SRC_ROOT, _CV_ROOT, _DRO_ROOT, _OS_ROOT, str(_LBO_SRC)):
    sys.path.insert(0, str(p))

import yaml  # noqa: E402

from _cv_utils import pkg_path, repo_path  # noqa: E402
from tune_rrf_oneshot import _export        # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--best", required=True,
                   help="Path to a best_params_rrf_oneshot_*.yaml (repo-relative "
                        "or absolute). No default — pick the metric explicitly.")
    p.add_argument("--config", default="launchers_dro_oneshot/configs/tune_oneshot.yaml",
                   help="Tune config for the data block (n_folds, splitk_dir).")
    p.add_argument("--storage_dir", default="models/CG_crossvalidation",
                   help="CG dataset store (component inputs + rrf_oneshot output).")
    p.add_argument("--submit_k", type=int, default=20,
                   help="Tracks per blind submission record.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    best_path = Path(args.best)
    if not best_path.is_absolute():
        best_path = repo_path(args.best)
    if not best_path.exists():
        sys.exit(f"[recreate] best-params yaml not found: {best_path}")
    with open(best_path) as f:
        bp = yaml.safe_load(f)

    components = list(bp["components"])
    anchor     = bp["anchor_component"]
    best       = bp["weights"]          # dict: w_<m> + k_rrf (study.best_params)
    fuse_top_k = int(bp["fuse_top_k"])

    with open(pkg_path(args.config)) as f:
        cfg = yaml.safe_load(f)

    storage_dir = repo_path(args.storage_dir)
    out_dir = storage_dir / "rrf_oneshot"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[recreate] best={best_path.name} components={len(components)} "
          f"anchor={anchor} k_rrf={best.get('k_rrf')} fuse_top_k={fuse_top_k}")
    _export(cfg, components, best, anchor, fuse_top_k,
            storage_dir, out_dir, args.submit_k)


if __name__ == "__main__":
    main()
