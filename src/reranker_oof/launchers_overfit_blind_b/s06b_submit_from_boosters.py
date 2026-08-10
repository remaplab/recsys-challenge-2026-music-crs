"""Regenerate a Blind-B/-A submission from boosters already saved by s06,
without refitting. Reuses s06_retrain_submit's own scoring/ensemble/emit
helpers.

Usage:
    cd src/reranker_oof
    uv run python -m launchers_overfit_blind_b.s06b_submit_from_boosters \\
        --config configs/blind_no_filter/xgb_v5.yaml --variants v2_blind_last
"""
from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PKG_ROOT))
sys.path.insert(0, str(_PKG_ROOT / "src"))

import polars as pl                                                    # noqa: E402
import xgboost as xgb                                                  # noqa: E402
import yaml                                                            # noqa: E402

from src.paths import BLIND_A_RAW, REPO_ROOT, ensure_output_dirs, set_active_dataset  # noqa: E402
from src.rerankers.xgb_ranker import XGBReranker                       # noqa: E402

from launchers_overfit_blind_b._rerank import (                        # noqa: E402
    blind_a_chunks, blind_chunks, resolve_feats,
)
from launchers_overfit_blind_b.s06_retrain_submit import (             # noqa: E402
    _emit_scored, _export_candidates, _rank_average, _score_chunks,
)


def run(cfg: dict, variants_sel: list[str],
        kinds: tuple[str, ...] = ("blind_b", "blind_a")) -> int:
    ensure_output_dirs()
    set_active_dataset(cfg["dataset_name"])
    blind_raw = Path(cfg["blind_raw"])
    if not blind_raw.is_absolute():
        blind_raw = REPO_ROOT / blind_raw
    tag = cfg.get("run_tag") or cfg["dataset_name"]
    cand_top = int(cfg.get("cand_top", 200))

    chunk_fns = {"blind_b": blind_chunks, "blind_a": blind_a_chunks}
    raw_paths = {"blind_b": blind_raw, "blind_a": BLIND_A_RAW}

    for name in variants_sel:
        out_dir = REPO_ROOT / "models" / "reranker_oof" / "blind_b_retrain" / tag / name
        booster_paths = sorted((out_dir / "boosters").glob("booster_*.json"),
                               key=lambda p: int(p.stem.split("_")[1]))
        if not booster_paths:
            raise SystemExit(f"[s06b] no boosters found in {out_dir / 'boosters'}")
        print(f"\n========== resubmit {name}: {len(booster_paths)} booster(s) from "
              f"{out_dir} ({', '.join(kinds)}) ==========")

        # feat_cols must match what the boosters were trained with — recomputed
        # the same deterministic way s06 does (cfg.feat_cols_keep + schema probe).
        # Always probed against blind_b's schema (no goal_* cols), even when only
        # scoring blind_a, since that's the schema the boosters were fit against.
        feat_cols = resolve_feats(pl.read_parquet(blind_chunks()[0], n_rows=10),
                                  cfg.get("feat_cols_keep"),
                                  pl.read_parquet(blind_chunks()[0], n_rows=1))

        scored_members: dict[str, list[pl.DataFrame]] = {k: [] for k in kinds}
        for j, bp in enumerate(booster_paths):
            model = XGBReranker()
            model._booster = xgb.Booster()
            model._booster.load_model(str(bp))
            print(f"  loaded {bp}")
            for k in kinds:
                scored_members[k].append(_score_chunks(model, feat_cols, chunk_fns[k]()))
            model.release(); gc.collect()

        n = len(booster_paths)
        c_alpha = float(cfg.get("conformal_alpha", 0.1))
        for k in kinds:
            ens = _rank_average(scored_members[k])
            _emit_scored(k, ens, raw_paths[k], out_dir, tag, name,
                         n_members=n, conformal_alpha=c_alpha, bag_label="")
            _export_candidates(k, scored_members[k], ens, out_dir,
                               tag, name, cand_top, n, bag_label="")
        print(f"[s06b] {name}: resubmitted from {n} saved booster(s) → {out_dir}")

    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--variants", nargs="+", choices=["v1_blind_all", "v2_blind_last"],
                    default=["v2_blind_last"])
    ap.add_argument("--kinds", nargs="+", choices=["blind_b", "blind_a"],
                    default=["blind_b", "blind_a"])
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    return run(cfg, variants_sel=args.variants, kinds=tuple(args.kinds))


if __name__ == "__main__":
    raise SystemExit(main())
