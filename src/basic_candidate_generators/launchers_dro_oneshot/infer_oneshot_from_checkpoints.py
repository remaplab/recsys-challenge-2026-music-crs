"""Load already-saved one-shot checkpoints and re-run inference only (no fit).

Mirrors `launchers_dro/infer_from_checkpoints.py`, adapted to the one-shot
checkpoint layout produced by `retrain_and_export_oneshot_ckpt.py`.

Checkpoint -> split -> output mapping:
    fold_{k}_cg_train.pkl.gz      -> cg_val split        -> fold_{k}_oof_cg_val.parquet
    fold_{k}_cg_train_val.pkl.gz  -> reranker_val split  -> fold_{k}_oof_reranker_val.parquet
    non_holdout.pkl.gz            -> holdout_test        -> holdout_candidates.parquet
    full.pkl.gz                   -> blind-A (real turn) -> blind_candidates.parquet
    full.pkl.gz                   -> blind-B (all turns) -> blind_b_candidates.parquet  (--splits blind_b)

Writes to `<out_dir>/datasets/` (NOT `datasets_reference/` or
`datasets_after_training/`), so it can be diffed against both without
clobbering either.

Run:
    cd src/basic_candidate_generators
    uv run python -m launchers_dro_oneshot.infer_oneshot_from_checkpoints --model tower_a
    uv run python -m launchers_dro_oneshot.infer_oneshot_from_checkpoints --model tower_a --splits blind_b
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parent.parent
_SRC_ROOT = _PKG_ROOT / "src"
_CV_ROOT  = _PKG_ROOT / "launchers_crossvalidation"
for p in (_SRC_ROOT, _CV_ROOT):
    sys.path.insert(0, str(p))

import polars as pl    # noqa: E402
import yaml            # noqa: E402

from _cv_utils import load_blind_b_eval, repo_path, run_inference_dispatch  # noqa: E402

from launchers_dro_oneshot._ckpt_io import WARM_REFIT_MODELS, load_ckpt     # noqa: E402
from launchers_dro_oneshot.export_oneshot_candidates import (               # noqa: E402
    _blind_context, _save,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inference-only replay from saved one-shot checkpoints.")
    p.add_argument("--model", required=True)
    p.add_argument("--config", default="launchers_dro_oneshot/configs/tune_oneshot.yaml")
    p.add_argument("--storage_dir", default="models/CG_crossvalidation")
    p.add_argument("--top_n", type=int, default=300)
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--splits", nargs="+", default=["folds", "holdout", "full"],
                   choices=["folds", "holdout", "full", "blind_b"],
                   help="Which checkpoint groups to replay. 'full' → Blind-A. "
                        "'blind_b' also reuses full.pkl.gz (not a separate checkpoint).")
    p.add_argument("--blind_b_path",
                   default="data/talkpl-ai/TalkPlayData-Challenge-Blind-B/data/test-00000-of-00001.parquet")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with open(_PKG_ROOT / args.config) as f:
        cfg = yaml.safe_load(f)
    if args.model not in cfg["models"]:
        sys.exit(f"[infer-ckpt] unknown model {args.model!r}")
    mcfg = cfg["models"][args.model]
    data_cfg = cfg["data"]
    splitk_dir = repo_path(data_cfg["splitk_dir"])
    track_meta = pl.read_parquet(repo_path(data_cfg["track_metadata_path"]))
    inf_mode = mcfg.get("inference_mode", "standard")
    class_name, module_name = mcfg["class"], mcfg["module"]

    storage_dir = repo_path(args.storage_dir)
    model_dir = storage_dir / f"{args.model}_oneshot"
    ckpt_dir = model_dir / "checkpoints"
    out_dir = model_dir / "datasets"
    warm_refit = args.model in WARM_REFIT_MODELS

    print(f"[infer-ckpt] {args.model}_oneshot ({class_name})  mode={inf_mode}")
    print(f"[infer-ckpt] ckpt_dir={ckpt_dir}")
    print(f"[infer-ckpt] out={out_dir}")
    if warm_refit:
        print(f"[infer-ckpt] warm-refit path active for {args.model} "
              "(rebuilds cached frozen arrays via fit(None, ...), no retrain)")

    if "folds" in args.splits:
        for fold in range(args.n_folds):
            print(f"\n--- fold {fold} ---")
            ckpt_cg = ckpt_dir / f"fold_{fold}_cg_train.pkl.gz"
            if ckpt_cg.exists():
                rec = load_ckpt(class_name, module_name, ckpt_cg,
                                warm_refit=warm_refit, track_meta=track_meta)
                ev = pl.read_parquet(splitk_dir / f"fold_{fold}_cg_val.parquet")
                recs = run_inference_dispatch(rec, ev, args.top_n, inf_mode, track_meta)
                _save(recs, out_dir / f"fold_{fold}_oof_cg_val.parquet",
                      gt_parquet=splitk_dir / f"fold_{fold}_cg_val.parquet")
            else:
                print(f"  [skip] missing {ckpt_cg}")

            ckpt_cr = ckpt_dir / f"fold_{fold}_cg_train_val.pkl.gz"
            if ckpt_cr.exists():
                rec = load_ckpt(class_name, module_name, ckpt_cr,
                                warm_refit=warm_refit, track_meta=track_meta)
                rv = pl.read_parquet(splitk_dir / f"fold_{fold}_reranker_val.parquet")
                recs = run_inference_dispatch(rec, rv, args.top_n, inf_mode, track_meta)
                _save(recs, out_dir / f"fold_{fold}_oof_reranker_val.parquet",
                      gt_parquet=splitk_dir / f"fold_{fold}_reranker_val.parquet")
            else:
                print(f"  [skip] missing {ckpt_cr}")

    if "holdout" in args.splits:
        print("\n--- holdout ---")
        ckpt_nh = ckpt_dir / "non_holdout.pkl.gz"
        if ckpt_nh.exists():
            rec = load_ckpt(class_name, module_name, ckpt_nh,
                            warm_refit=warm_refit, track_meta=track_meta)
            holdout = pl.read_parquet(splitk_dir / "holdout_test.parquet")
            recs = run_inference_dispatch(rec, holdout, args.top_n, inf_mode, track_meta)
            _save(recs, out_dir / "holdout_candidates.parquet",
                  gt_parquet=splitk_dir / "holdout_test.parquet")
        else:
            print(f"  [skip] missing {ckpt_nh}")

    if "full" in args.splits or "blind_b" in args.splits:
        print("\n--- full ---")
        ckpt_full = ckpt_dir / "full.pkl.gz"
        if not ckpt_full.exists():
            print(f"  [skip] missing {ckpt_full}")
        else:
            rec = load_ckpt(class_name, module_name, ckpt_full,
                            warm_refit=warm_refit, track_meta=track_meta)

            if "full" in args.splits:
                ctx = _blind_context(pl.read_parquet(repo_path(data_cfg["blind_parquet"])), "session")
                recs = rec.recommend(ctx, top_k=args.top_n, remove_seen=True)
                _save(recs, out_dir / "blind_candidates.parquet")

            if "blind_b" in args.splits:
                eval_df, gt_map = load_blind_b_eval(repo_path(args.blind_b_path))
                recs = run_inference_dispatch(rec, eval_df, args.top_n, inf_mode, track_meta)
                recs = recs.with_columns(pl.col("turn").cast(pl.Int64))
                recs = recs.drop([c for c in ("gt_track_id", "gt_track_id_right") if c in recs.columns])
                recs = recs.join(gt_map, on=["session_id", "turn"], how="left")
                _save(recs, out_dir / "blind_b_candidates.parquet")

    print(f"\n[infer-ckpt] done — {out_dir}")


if __name__ == "__main__":
    main()
