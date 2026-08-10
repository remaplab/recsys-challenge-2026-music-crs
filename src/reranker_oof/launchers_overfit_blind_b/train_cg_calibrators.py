"""Fit + persist the per-CG isotonic/Platt calibrators + conformal quantiles.

Split out of `s03_assemble_dataset.py`'s inline fit (previously re-run on
every assembly call, even a blind-B-only one, forcing all 14 CGs' fold OOF
parquets to be present just to build one split). Run this once per dataset
config; downstream single-split launchers (e.g. a blind-B-only re-featurize)
load the saved artifacts instead of re-fitting.

Only touches `fold_{0..4}_oof_cg_val.parquet` for every CG in `cgs:` — no
holdout/blind/RRF/embedding dependency.

Output: `models/reranker_oof/calibrators/<dataset_name>/calibrators.pkl`
(a pickled `CalibrationArtifacts`, see `src/features/cg_calibration.py`).

Run:
    cd src/reranker_oof
    uv run python -m launchers_overfit_blind_b.train_cg_calibrators \\
        --config configs/blind_no_filter/dataset.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PKG_ROOT))
sys.path.insert(0, str(_PKG_ROOT / "src"))

from src.features.cg_calibration import fit_artifacts, save_artifacts   # noqa: E402
from src.features.resources import cg_candidate_path                   # noqa: E402
from src.paths import CALIBRATORS_DIR, ensure_output_dirs               # noqa: E402

from launchers_overfit_blind_b._common import (                        # noqa: E402
    assert_cgs_have, load_config, register_cg_paths,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--k_per_cg", type=int, default=None)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    ensure_output_dirs()
    cfg = load_config(args.config)
    register_cg_paths(cfg)
    cg_keep = list(cfg["cgs"])
    folds = [int(x) for x in args.folds.split(",") if x.strip()]
    assert_cgs_have(cfg, ["train"])

    cc = cfg.get("cg_calibration") or {}
    if not cc.get("enabled", True):
        sys.exit("[train-calib] cg_calibration.enabled is false in this config — nothing to fit")

    k_per_cg = int(args.k_per_cg if args.k_per_cg is not None
                   else cfg.get("max_candidates_per_cg", 200))

    cg_oof = {k: {cg: cg_candidate_path(cg, "train", k) for cg in cg_keep}
              for k in folds}
    print(f"[train-calib] {cfg['name']}: fitting CG calibrators "
          f"({cc.get('method', 'isotonic')}, {cc.get('feature', 'reciprocal_rank')}) "
          f"over {len(cg_keep)} CGs × {len(folds)} folds …")
    art = fit_artifacts(
        cg_keep, cg_oof, method=cc.get("method", "isotonic"),
        feature_col=cc.get("feature", "reciprocal_rank"),
        alpha=float(cc.get("alpha", 0.1)), k_per_cg=k_per_cg, verbose=True,
    )

    out_path = CALIBRATORS_DIR / cfg["name"] / "calibrators.pkl"
    save_artifacts(art, out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
