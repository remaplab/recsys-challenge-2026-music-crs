"""Assemble ONLY the Blind-B dataset chunk from SAVED calibrators and resubmit
from saved boosters — no retrain, no Blind-A, no folds/holdout rebuild, and
(unlike the original version of this script) no re-fit of the per-CG
calibrators either — those must already be on disk (see
`train_cg_calibrators.py`), so this needs none of the 14 CGs' fold OOF
parquets, only their `blind_b_candidates.parquet`.

For when CG Blind-B candidates were (re)exported and you just want a fresh
blind_b scoring off an already-tuned/trained model, without paying for the
full `s03_assemble_dataset` (all splits, incl. the calibrator fit) +
`s06_retrain_submit` (retrain) pipeline.

Three steps, same two config files as the full pipeline (RERANKER_README.md):
  1. Load `models/reranker_oof/calibrators/<name>/calibrators.pkl`
     (`train_cg_calibrators.py` — run once per dataset config, whenever the
     14 CGs' fold OOF candidates change).
  2. `_build_split("blind_b", ...)` (reused from `s03_assemble_dataset.py`) —
     builds `models/reranker_oof/datasets/<name>/blind_b/chunk_*.parquet`
     using the loaded calibrators.
  3. `s06b_submit_from_boosters` scoped to `kinds=("blind_b",)` — loads
     `<out_dir>/boosters/booster_*.json`, scores blind_b, rewrites
     `submissions/`, `scored_blind_b*.parquet`, `metrics_blind_b*.csv`,
     `candidates/cand_..._blind_b*.parquet` in place. Blind-A outputs are
     left untouched.

Usage:
    cd src/reranker_oof
    uv run python -m launchers_overfit_blind_b.s06c_blind_b_only \\
        --dataset_config configs/blind_no_filter/dataset.yaml \\
        --config configs/blind_no_filter/xgb_v5.yaml \\
        --variants v2_blind_last
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PKG_ROOT))
sys.path.insert(0, str(_PKG_ROOT / "src"))

import yaml  # noqa: E402

from src.features.cg_calibration import load_artifacts  # noqa: E402
from src.features.pipeline import _split_save_dir       # noqa: E402
from src.paths import BLIND_B_RAW, CALIBRATORS_DIR, ensure_output_dirs  # noqa: E402

from launchers_overfit_blind_b._common import (          # noqa: E402
    BLIND_B_GT, assert_cgs_have, load_config, register_cg_paths,
)
from launchers_overfit_blind_b.s03_assemble_dataset import (  # noqa: E402
    _build_split, load_assembly_context,
)
from launchers_overfit_blind_b.s06b_submit_from_boosters import run  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset_config", type=Path,
                    default=Path("configs/blind_no_filter/dataset.yaml"),
                    help="same config s03_assemble_dataset takes")
    ap.add_argument("--config", type=Path, required=True,
                    help="xgb config (same one s06/s06b take)")
    ap.add_argument("--variants", nargs="+", choices=["v1_blind_all", "v2_blind_last"],
                    default=["v2_blind_last"])
    ap.add_argument("--force", action="store_true",
                    help="reassemble the blind_b chunk even if already present")
    args = ap.parse_args()

    with open(args.config) as f:
        xgb_cfg = yaml.safe_load(f)

    ensure_output_dirs()
    ds_cfg = load_config(args.dataset_config)
    register_cg_paths(ds_cfg)
    assert_cgs_have(ds_cfg, ["blind_b"])

    calib_path = CALIBRATORS_DIR / ds_cfg["name"] / "calibrators.pkl"
    if not calib_path.exists():
        sys.exit(f"[s06c] no saved calibrators at {calib_path} — run "
                  f"train_cg_calibrators.py --config {args.dataset_config} first")
    calib = load_artifacts(calib_path)

    ctx = load_assembly_context(ds_cfg)

    # ── step 1: assemble ONLY the blind_b chunk, calibrators loaded not fit ──
    _build_split("blind_b", 0, ctx.cg_keep, ctx.k_per_cg, ctx.groups_per_chunk,
                 _split_save_dir(ctx.out_root, "blind_b", 0), args.force,
                 ctx.rrf, ctx.embeddings_cfg, ctx.tt_spec, ctx.emb_cache_dir, calib,
                 blind_raw=BLIND_B_RAW, blind_gt=BLIND_B_GT)

    # ── step 2: score blind_b only from the saved boosters ──────────────────
    return run(xgb_cfg, variants_sel=args.variants, kinds=("blind_b",))


if __name__ == "__main__":
    raise SystemExit(main())
