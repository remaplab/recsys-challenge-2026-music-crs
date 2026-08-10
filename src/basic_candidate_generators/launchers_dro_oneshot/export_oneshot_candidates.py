"""Export a tuned one-shot component's candidate lists for ALL turns (the
component HPs are tuned on turn-1, but every oneshot CG keys query/query-text
by (session, turn), so inference runs on any turn → it exports like a normal CG).

Reads the component's best params from its Optuna study, then:
  * per fold: refit on fold cg_train, infer fold cg_val (all turns) →
    datasets/fold_{f}_oof_cg_val.parquet   (OOF lists → stage-2 RRF tuning)
  * per fold: refit on fold cg_train+cg_val, infer reranker_val (all turns) →
    datasets/fold_{f}_oof_reranker_val.parquet  (reranker val / PoSI split)
  * full train (all folds' cg_train, unique sessions): refit, infer holdout
    (all turns) + blind (at each session's real target turn) →
    datasets/holdout_candidates.parquet, blind_candidates.parquet

Output dir: models/CG_crossvalidation/<model>_oneshot/datasets/. Each parquet:
(session_id, user_id, turn, track_ids[list[str]], scores[list[f64]]).

Run (after tuning <model>):
    cd src/basic_candidate_generators
    uv run python -m launchers_dro_oneshot.export_oneshot_candidates --model dense_text_8b
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
    instantiate_rec, load_blind_b_eval, load_fold, make_storage, pkg_path,
    repo_path, resolve_param_paths, run_inference_dispatch,
)
from recommenders.user_base import build_context_df  # noqa: E402

optuna.logging.set_verbosity(optuna.logging.WARNING)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export turn-1 candidate lists for a tuned component.")
    p.add_argument("--model", required=True)
    p.add_argument("--config", default="launchers_dro_oneshot/configs/tune_oneshot.yaml")
    p.add_argument("--storage_dir", default="models/CG_crossvalidation")
    p.add_argument("--top_n", type=int, default=300, help="candidates stored per session")
    p.add_argument("--robust_alpha", type=float, default=0.7)
    p.add_argument("--blind_b_only", action="store_true",
                   help="Skip folds/holdout/blind-A; only refit on full train and export "
                        "blind_b_candidates.parquet (the existing artifacts are left intact).")
    p.add_argument("--blind_b_path",
                   default="data/talkpl-ai/TalkPlayData-Challenge-Blind-B/data/test-00000-of-00001.parquet")
    return p.parse_args()


def _best_params(model: str, storage_dir: Path, alpha: float) -> dict:
    folder_key = f"{model}_oneshot"
    db = storage_dir / folder_key / f"optuna_{folder_key}.db"
    if not db.exists():
        sys.exit(f"[export] no study db at {db} — tune {model} first.")
    study = optuna.load_study(
        study_name=f"{folder_key}_cvar{int(round(alpha*100))}", storage=make_storage(db))
    if study.best_trial is None:
        sys.exit(f"[export] study for {model} has no complete trial.")
    print(f"[export] best trial #{study.best_trial.number}  value={study.best_value:.4f}")
    return dict(study.best_params)


def _blind_context(blind: pl.DataFrame, urm_mode: str) -> pl.DataFrame:
    """Build the Blind-A inference context predicted at each session's REAL
    target turn (the last conversation turn), not turn 1.

    Explodes the nested conversations into the long splitK shape, then
    ``build_context_df`` picks ``target_turn = max(turn_number)`` per session and
    uses the prior music turns as context — exactly the normal-CG blind setup.
    Carries session-level goal/profile so goal/profile-aware CGs populate.
    """
    convs = blind.explode("conversations").unnest("conversations")
    music = (convs.filter(pl.col("role") == "music")
             .rename({"content": "track_id"})
             .select("session_id", "user_id", "turn_number", "track_id"))
    # one target row per session at the last conversation turn (track_id null)
    # maintain_order: unordered group_by row order otherwise leaks into GPU
    # batch composition downstream (float32 reduction noise → top-k reshuffle).
    target = (convs.group_by("session_id", maintain_order=True)
              .agg(pl.col("turn_number").max().alias("turn_number"))
              .join(blind.select("session_id", "user_id").unique(subset=["session_id"]),
                    on="session_id", how="left")
              .with_columns(pl.lit(None, dtype=pl.Utf8).alias("track_id"))
              .select("session_id", "user_id", "turn_number", "track_id"))
    long = pl.concat([music, target])
    if "session_date" in blind.columns:
        sd = blind.select("session_id", "session_date").unique(subset=["session_id"])
    else:
        sd = (blind.select("session_id").unique()
              .with_columns(pl.lit(None, dtype=pl.Utf8).alias("session_date")))
    long = long.join(sd, on="session_id", how="left")
    for col in ("conversation_goal", "user_profile"):
        if col in blind.columns:
            long = long.join(
                blind.select("session_id", col).unique(subset=["session_id"]),
                on="session_id", how="left",
            )
    ctx, _ = build_context_df(long, inject_multi_session=(urm_mode == "user"))
    return ctx


def _gt_map(gt_parquet: Path) -> pl.DataFrame:
    """(session_id, turn, gt_track_id) from a splitK long parquet (all turns).
    GT for a (session, turn) is the track played at that turn_number."""
    return (
        pl.read_parquet(gt_parquet, columns=["session_id", "turn_number", "track_id"])
        .select(
            "session_id",
            pl.col("turn_number").cast(pl.Int64).alias("turn"),
            pl.col("track_id").alias("gt_track_id"),
        )
        .unique(subset=["session_id", "turn"], keep="first")
    )


def _save(recs: pl.DataFrame, path: Path, gt_parquet: Path | None = None) -> None:
    """Write a candidate parquet. ``gt_parquet`` joins the splitK GT per
    (session, turn); when None the GT column is null (blind submission turn).
    Downstream CG calibration requires the ``gt_track_id`` column to exist."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if gt_parquet is not None:
        recs = recs.with_columns(pl.col("turn").cast(pl.Int64)).join(
            _gt_map(gt_parquet), on=["session_id", "turn"], how="left",
        )
    elif "gt_track_id" not in recs.columns:
        recs = recs.with_columns(pl.lit(None, dtype=pl.Utf8).alias("gt_track_id"))
    cols = [c for c in ("session_id", "user_id", "turn", "track_ids", "scores", "gt_track_id")
            if c in recs.columns]
    recs.select(cols).write_parquet(path)
    n_gt = recs.filter(pl.col("gt_track_id").is_not_null()).height if "gt_track_id" in recs.columns else 0
    print(f"[export] wrote {path}  ({recs.height} sessions, gt={n_gt})")


def main() -> None:
    args = parse_args()
    with open(pkg_path(args.config)) as f:
        cfg = yaml.safe_load(f)
    if args.model not in cfg["models"]:
        sys.exit(f"[export] unknown model {args.model!r}")
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
    uses_colisten = mcfg.get("uses_colisten", False)   # E3.6: needs unfiltered fold

    def fit_infer(train_df, eval_df, colisten_df=None):
        rec = instantiate_rec(class_name, module_name, params, "session")
        fit_kwargs = {"track_metadata": track_meta}
        if colisten_df is not None:
            fit_kwargs["colisten_df"] = colisten_df
        rec.fit(train_df, **fit_kwargs)
        return run_inference_dispatch(rec, eval_df, args.top_n, inf_mode, track_meta)

    def _fit_full():
        full_train = pl.concat(
            [load_fold(splitk_dir, f) for f in range(n_folds)]
        ).unique(subset=["session_id", "turn_number"])
        rec = instantiate_rec(class_name, module_name, params, "session")
        fit_kwargs = {"track_metadata": track_meta}
        if uses_colisten:
            fit_kwargs["colisten_df"] = full_train
        rec.fit(full_train, **fit_kwargs)
        return rec

    # ── Blind-B only: refit on full train, infer ALL turns of Blind-B (visible
    # turns carry the known GT for internal validation; the withheld submission
    # turn has gt null), write blind_b_candidates.parquet. ──
    if args.blind_b_only:
        rec = _fit_full()
        eval_df, gt_map = load_blind_b_eval(repo_path(args.blind_b_path))
        print(f"[export] blind-B long: {eval_df.height} (session, turn) rows, "
              f"{eval_df['session_id'].n_unique()} sessions, {gt_map.height} with known GT")
        recs = run_inference_dispatch(rec, eval_df, args.top_n, inf_mode, track_meta)
        recs = recs.with_columns(pl.col("turn").cast(pl.Int64))
        recs = recs.drop([c for c in ("gt_track_id", "gt_track_id_right") if c in recs.columns])
        recs = recs.join(gt_map, on=["session_id", "turn"], how="left")
        _save(recs, out_dir / "blind_b_candidates.parquet")
        return

    # ── OOF per fold (ALL turns — like a normal CG) ──
    for fold in range(n_folds):
        # A: fit on cg_train → infer cg_val (every turn)
        train = load_fold(splitk_dir, fold)
        ev = pl.read_parquet(splitk_dir / f"fold_{fold}_cg_val.parquet")
        colisten = load_fold(splitk_dir, fold) if uses_colisten else None
        _save(fit_infer(train, ev, colisten), out_dir / f"fold_{fold}_oof_cg_val.parquet",
              gt_parquet=splitk_dir / f"fold_{fold}_cg_val.parquet")

        # B: fit on cg_train+cg_val → infer reranker_val (every turn). Mirrors the
        # non-oneshot _export_one so the reranker has a val split (HP tuning / PoSI).
        cr = load_fold(splitk_dir, fold, parts=["cg_train", "cg_val"])
        rv = pl.read_parquet(splitk_dir / f"fold_{fold}_reranker_val.parquet")
        cr_colisten = cr if uses_colisten else None
        _save(fit_infer(cr, rv, cr_colisten),
              out_dir / f"fold_{fold}_oof_reranker_val.parquet",
              gt_parquet=splitk_dir / f"fold_{fold}_reranker_val.parquet")

    # ── full-train model → holdout (all turns) + blind (real target turn) ──
    rec = _fit_full()

    holdout = pl.read_parquet(splitk_dir / "holdout_test.parquet")
    _save(run_inference_dispatch(rec, holdout, args.top_n, inf_mode, track_meta),
          out_dir / "holdout_candidates.parquet",
          gt_parquet=splitk_dir / "holdout_test.parquet")

    # Blind-A: one prediction per session at its real target turn (last conv
    # turn). Standard recommend over the rebuilt context.
    ctx = _blind_context(pl.read_parquet(repo_path(data_cfg["blind_parquet"])), "session")
    _save(rec.recommend(ctx, top_k=args.top_n, remove_seen=True),
          out_dir / "blind_candidates.parquet")


if __name__ == "__main__":
    main()
