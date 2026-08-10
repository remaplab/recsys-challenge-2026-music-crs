"""Retrain a tuned one-shot CG component and save gzip-compressed checkpoints
alongside the normal dataset export. Mirrors `launchers_dro/retrain_and_export.py`'s
checkpoint pattern, adapted to the one-shot pipeline (see `export_oneshot_candidates.py`
for the un-checkpointed original this is built from — same fit/infer recipe).

Two full-data fits, saved as SEPARATE checkpoints (not one — they are trained on
different data):
  * non_holdout.pkl.gz — fit on all folds' cg_train (excludes holdout), same
    recipe `export_oneshot_candidates.py` already uses → infers
    holdout_candidates.parquet.
  * full.pkl.gz — fit on all folds' cg_train+cg_val+reranker_val PLUS
    holdout_test.parquet (a strict superset of non_holdout) → infers BOTH
    blind_candidates.parquet (Blind-A) and blind_b_candidates.parquet
    (Blind-B). One fit, reused for both — Blind-B does NOT get its own refit.
    (Deliberately built from splitK long-format frames, not the raw nested
    challenge dataset: lexical CGs' `_compute_query_stopwords`/colisten doc
    building read `turn_number`/`track_id` at the top level, which the raw
    nested `conversations` schema doesn't have until exploded.)

Checkpoints are gzip pickles (`.pkl.gz`, via `_ckpt_io.py`) — NOT plain
BaseRecommender `.save()`/`.load()`, so this can't touch any DRO checkpoint.

Run (after tuning <model> with tune_crossvalidation_oneshot.py):
    cd src/basic_candidate_generators
    uv run python -m launchers_dro_oneshot.retrain_and_export_oneshot_ckpt --model tower_a
    uv run python -m launchers_dro_oneshot.retrain_and_export_oneshot_ckpt --model tower_a --blind_b_only
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parent.parent
_SRC_ROOT = _PKG_ROOT / "src"
_CV_ROOT  = _PKG_ROOT / "launchers_crossvalidation"
_REPO_ROOT = _PKG_ROOT.parent.parent
_LBO_SRC = _REPO_ROOT / "src" / "lower_bound_optimization" / "src"
for p in (_SRC_ROOT, _CV_ROOT, str(_LBO_SRC)):
    sys.path.insert(0, str(p))

import optuna          # noqa: E402
import polars as pl    # noqa: E402
import yaml            # noqa: E402

from _cv_utils import (  # noqa: E402
    instantiate_rec, load_blind_b_eval, load_fold, repo_path,
    resolve_param_paths, run_inference_dispatch,
)

from launchers_dro_oneshot._ckpt_io import save_ckpt          # noqa: E402
from launchers_dro_oneshot.export_oneshot_candidates import (  # noqa: E402
    _best_params, _blind_context, _save,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Retrain a one-shot CG and save checkpoints + datasets.")
    p.add_argument("--model", required=True)
    p.add_argument("--config", default="launchers_dro_oneshot/configs/tune_oneshot.yaml")
    p.add_argument("--storage_dir", default="models/CG_crossvalidation")
    p.add_argument("--top_n", type=int, default=300, help="candidates stored per session")
    p.add_argument("--robust_alpha", type=float, default=0.7)
    p.add_argument("--blind_b_only", action="store_true",
                   help="Skip folds/holdout/blind-A; only (re)build full.pkl.gz if "
                        "missing and export blind_b_candidates.parquet.")
    p.add_argument("--blind_b_path",
                   default="data/talkpl-ai/TalkPlayData-Challenge-Blind-B/data/test-00000-of-00001.parquet")
    p.add_argument("--skip_checkpoints", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with open(_PKG_ROOT / args.config) as f:
        cfg = yaml.safe_load(f)
    if args.model not in cfg["models"]:
        sys.exit(f"[retrain-ckpt] unknown model {args.model!r}")
    mcfg = cfg["models"][args.model]
    data_cfg = cfg["data"]
    n_folds = int(data_cfg.get("n_folds", 5))
    splitk_dir = repo_path(data_cfg["splitk_dir"])
    track_meta = pl.read_parquet(repo_path(data_cfg["track_metadata_path"]))
    inf_mode = mcfg.get("inference_mode", "standard")
    class_name, module_name = mcfg["class"], mcfg["module"]
    fixed = resolve_param_paths(mcfg.get("fixed_params") or {})

    storage_dir = repo_path(args.storage_dir)
    best = _best_params(args.model, storage_dir, args.robust_alpha)
    params = {**best, **fixed}
    out_dir = storage_dir / f"{args.model}_oneshot" / "datasets"
    ckpt_dir = storage_dir / f"{args.model}_oneshot" / "checkpoints"
    uses_colisten = mcfg.get("uses_colisten", False)

    def _new_rec():
        return instantiate_rec(class_name, module_name, params, "session")

    def fit_infer(train_df, eval_df, colisten_df=None):
        rec = _new_rec()
        fit_kwargs = {"track_metadata": track_meta}
        if colisten_df is not None:
            fit_kwargs["colisten_df"] = colisten_df
        rec.fit(train_df, **fit_kwargs)
        return rec, run_inference_dispatch(rec, eval_df, args.top_n, inf_mode, track_meta)

    def _fit_non_holdout():
        """Excludes holdout — same recipe the un-checkpointed exporter uses."""
        train_df = pl.concat(
            [load_fold(splitk_dir, f) for f in range(n_folds)]
        ).unique(subset=["session_id", "turn_number"])
        rec = _new_rec()
        fit_kwargs = {"track_metadata": track_meta}
        if uses_colisten:
            fit_kwargs["colisten_df"] = train_df
        rec.fit(train_df, **fit_kwargs)
        return rec

    def _fit_full_everything():
        """Strict superset of non_holdout: every fold's cg_train+cg_val+
        reranker_val, plus holdout_test itself. Used for blind (A and B) only —
        holdout inference must never see holdout in training."""
        frames = [load_fold(splitk_dir, f, parts=["cg_train", "cg_val", "reranker_val"])
                  for f in range(n_folds)]
        frames.append(pl.read_parquet(splitk_dir / "holdout_test.parquet"))
        train_df = pl.concat(frames, how="diagonal_relaxed").unique(
            subset=["session_id", "turn_number"])
        rec = _new_rec()
        fit_kwargs = {"track_metadata": track_meta}
        if uses_colisten:
            fit_kwargs["colisten_df"] = train_df
        rec.fit(train_df, **fit_kwargs)
        return rec

    # ── Blind-B only: reuse full.pkl.gz if present, else fit it fresh ──
    if args.blind_b_only:
        full_ckpt = ckpt_dir / "full.pkl.gz"
        if full_ckpt.exists():
            from launchers_dro_oneshot._ckpt_io import WARM_REFIT_MODELS, load_ckpt
            print(f"[retrain-ckpt] reusing existing {full_ckpt}")
            rec = load_ckpt(class_name, module_name, full_ckpt,
                            warm_refit=args.model in WARM_REFIT_MODELS, track_meta=track_meta)
        else:
            rec = _fit_full_everything()
            if not args.skip_checkpoints:
                save_ckpt(rec, full_ckpt)
        eval_df, gt_map = load_blind_b_eval(repo_path(args.blind_b_path))
        print(f"[retrain-ckpt] blind-B long: {eval_df.height} (session, turn) rows, "
              f"{eval_df['session_id'].n_unique()} sessions, {gt_map.height} with known GT")
        recs = run_inference_dispatch(rec, eval_df, args.top_n, inf_mode, track_meta)
        recs = recs.with_columns(pl.col("turn").cast(pl.Int64))
        recs = recs.drop([c for c in ("gt_track_id", "gt_track_id_right") if c in recs.columns])
        recs = recs.join(gt_map, on=["session_id", "turn"], how="left")
        _save(recs, out_dir / "blind_b_candidates.parquet")
        return

    # ── OOF per fold (ALL turns), one checkpoint pair per fold ──
    for fold in range(n_folds):
        train = load_fold(splitk_dir, fold)
        ev = pl.read_parquet(splitk_dir / f"fold_{fold}_cg_val.parquet")
        colisten = load_fold(splitk_dir, fold) if uses_colisten else None
        rec_cg, recs_cg = fit_infer(train, ev, colisten)
        if not args.skip_checkpoints:
            save_ckpt(rec_cg, ckpt_dir / f"fold_{fold}_cg_train.pkl.gz")
        _save(recs_cg, out_dir / f"fold_{fold}_oof_cg_val.parquet",
              gt_parquet=splitk_dir / f"fold_{fold}_cg_val.parquet")

        cr = load_fold(splitk_dir, fold, parts=["cg_train", "cg_val"])
        rv = pl.read_parquet(splitk_dir / f"fold_{fold}_reranker_val.parquet")
        cr_colisten = cr if uses_colisten else None
        rec_cr, recs_cr = fit_infer(cr, rv, cr_colisten)
        if not args.skip_checkpoints:
            save_ckpt(rec_cr, ckpt_dir / f"fold_{fold}_cg_train_val.pkl.gz")
        _save(recs_cr, out_dir / f"fold_{fold}_oof_reranker_val.parquet",
              gt_parquet=splitk_dir / f"fold_{fold}_reranker_val.parquet")

    # ── non_holdout → holdout ──
    rec_nh = _fit_non_holdout()
    if not args.skip_checkpoints:
        save_ckpt(rec_nh, ckpt_dir / "non_holdout.pkl.gz")
    holdout = pl.read_parquet(splitk_dir / "holdout_test.parquet")
    _save(run_inference_dispatch(rec_nh, holdout, args.top_n, inf_mode, track_meta),
          out_dir / "holdout_candidates.parquet",
          gt_parquet=splitk_dir / "holdout_test.parquet")

    # ── full → blind-A (Blind-B stays opt-in via --blind_b_only) ──
    rec_full = _fit_full_everything()
    if not args.skip_checkpoints:
        save_ckpt(rec_full, ckpt_dir / "full.pkl.gz")
    ctx = _blind_context(pl.read_parquet(repo_path(data_cfg["blind_parquet"])), "session")
    _save(rec_full.recommend(ctx, top_k=args.top_n, remove_seen=True),
          out_dir / "blind_candidates.parquet")


if __name__ == "__main__":
    main()
