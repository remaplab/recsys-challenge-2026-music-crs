"""Retrain best CV model and export checkpoints, reranker datasets, blind submission.

Reads best hyperparameters from configs/cv_best_{model}_{urm_mode}_{metric}{K}.yaml
(produced by extract_best_params.py); falls back to the legacy bare name
configs/cv_best_{model}_{urm_mode}.yaml if the tagged file is missing. The
objective tag is controlled by --objective / --objective_k (defaults:
ndcg@20). Then:

  1. Per fold (0..n_folds-1):
       a. Fit on cg_train (80%) → OOF predictions on cg_val (16%) → reranker training data
       b. Fit on cg_train+cg_val (96%) → predictions on reranker_val (4%) → reranker val data

  2. Fit on full dataset (train + test raw) → save checkpoint + blind submission JSON

All per-fold inference uses `run_inference_dispatch` and emits one row per
(session, target_turn).

Output layout under {storage_dir}/{model}_{urm_mode}/:
  checkpoints/
    fold_{k}_cg_train.pkl
    fold_{k}_cg_train_val.pkl
    full.pkl
  datasets/
    fold_{k}_oof_cg_val.parquet          (OOF reranker training data, all turns)
    fold_{k}_oof_reranker_val.parquet    (reranker HP tuning / validation, all turns)
  submission/
    blind_A_{model}_{urm_mode}.json

Usage:
    cd src/basic_candidate_generators

    # Single model (default: ndcg@20 config)
    uv run python -m launchers_crossvalidation.retrain_and_export \\
        --model session_knn --urm_mode session

    # Single model (recall@200 config)
    uv run python -m launchers_crossvalidation.retrain_and_export \\
        --model session_knn --urm_mode session \\
        --objective recall --objective_k 200

    # All configs matching the chosen objective in configs/cv_best_*{metric}{K}.yaml
    uv run python -m launchers_crossvalidation.retrain_and_export --all
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_PKG_ROOT  = Path(__file__).resolve().parent.parent
_SRC_ROOT  = _PKG_ROOT / "src"
_CV_ROOT   = _PKG_ROOT / "launchers_crossvalidation"
sys.path.insert(0, str(_SRC_ROOT))
sys.path.insert(0, str(_CV_ROOT))

import polars as pl   # noqa: E402
import yaml           # noqa: E402

from _cv_utils import (    # noqa: E402
    instantiate_rec,
    load_eval,
    load_fold,
    load_reranker_val,
    pkg_path,
    repo_path,
    resolve_param_paths,
    run_inference_dispatch,
)
from recommenders.interactions import explode_music_turns   # noqa: E402
from recommenders.user_base import build_context_df         # noqa: E402
from embedding_based.query_tower import (                   # noqa: E402
    QUERY_TOWER_BASE,
    load_query_bundle,
)

_FULL_TRAIN_PATHS = [
    "data/talkpl-ai/TalkPlayData-Challenge-Dataset/data/train-00000-of-00001.parquet",
    "data/talkpl-ai/TalkPlayData-Challenge-Dataset/data/test-00000-of-00001.parquet",
]
_BLIND_PATH = (
    "data/talkpl-ai/TalkPlayData-Challenge-Blind-A/data/test-00000-of-00001.parquet"
)
_HOLDOUT_PATH = "data/splitK/holdout_test.parquet"
_TRACK_META_PATH = (
    "data/talkpl-ai/TalkPlayData-Challenge-Track-Metadata/"
    "data/all_tracks-00000-of-00001.parquet"
)
_MODELS_NO_META = {"gf_cf"}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Retrain best CV model and export artefacts.")
    p.add_argument("--model",       default=None,  help="Model key (must match config filename)")
    p.add_argument("--urm_mode",    default=None,  choices=["session", "user", None])
    p.add_argument("--all",         action="store_true",
                   help="Process every configs/cv_best_*.yaml found in the package configs dir.")
    p.add_argument("--config",      default=None,
                   help="Explicit path to a cv_best YAML (single-model mode only).")
    p.add_argument("--top_k",       type=int, default=200, help="Candidates per session for datasets")
    p.add_argument("--storage_dir", default="models/CG_crossvalidation",
                   help="Repo-root relative root dir (same as tune_crossvalidation.py)")
    p.add_argument("--splitk_dir",  default="data/splitK")
    p.add_argument("--n_folds",     type=int, default=5)
    p.add_argument("--skip_datasets",    action="store_true")
    p.add_argument("--skip_submission",  action="store_true")
    p.add_argument("--skip_checkpoints", action="store_true")
    p.add_argument("--skip_holdout_candidates", action="store_true",
                   help="Skip fitting on non-holdout and inferring top_k candidates on holdout_test.")
    p.add_argument("--skip_blind_candidates", action="store_true",
                   help="Skip saving full top_k candidate parquet for blind-A.")
    p.add_argument("--objective", choices=("ndcg", "recall"), default="ndcg",
                   help="Objective tag of the cv_best YAML to load. "
                        "Must match what was passed to extract_best_params.py.")
    p.add_argument("--objective_k", type=int, default=None,
                   help="K for the objective. Defaults: ndcg=20, recall=200.")
    return p.parse_args()


_OBJECTIVE_DEFAULT_K = {"ndcg": 20, "recall": 200}


def _cfg_suffix(objective: str, objective_k: int) -> str:
    return f"_{objective}{objective_k}"


def _discover_configs(suffix: str) -> list[tuple[str, str, Path]]:
    """Return (model_name, urm_mode, cfg_path) for cv_best_*{suffix}.yaml.

    Bare files (no objective suffix) are also picked up for backwards
    compatibility with pre-objective-tagged YAMLs.
    """
    results = []
    pattern = f"cv_best_*{suffix}.yaml"
    paths = sorted((_PKG_ROOT / "configs").glob(pattern))
    if not suffix:
        # Also include legacy bare files when no suffix requested.
        paths = sorted(set(paths) | set((_PKG_ROOT / "configs").glob("cv_best_*.yaml")))
    for p in paths:
        stem = p.stem[len("cv_best_"):]
        if suffix and stem.endswith(suffix):
            stem = stem[: -len(suffix)]
        if stem.endswith("_session"):
            urm_mode   = "session"
            model_name = stem[: -len("_session")]
        elif stem.endswith("_user"):
            urm_mode   = "user"
            model_name = stem[: -len("_user")]
        else:
            continue
        results.append((model_name, urm_mode, p))
    return results


# ---------------------------------------------------------------------------
# Blind prediction (adapted from launchers/predict_blind.py)
# ---------------------------------------------------------------------------

def _recs_to_submission(recs: pl.DataFrame, top_k: int) -> list[dict]:
    return [
        {
            "session_id":          row["session_id"],
            "user_id":             row["user_id"],
            "turn_number":         row["turn"],
            "predicted_track_ids": (row["track_ids"] or [])[:top_k],
            "predicted_response":  "",
        }
        for row in recs.iter_rows(named=True)
    ]


def _predict_sessions(
    rec, blind_df: pl.DataFrame, top_k: int, remove_seen: bool
) -> pl.DataFrame:
    """Last-turn recs via rec.recommend() (standard recommenders).

    Returns the raw recs DataFrame (session_id, user_id, turn, track_ids, scores)
    so callers can both export a candidate parquet and build a top-k submission.
    """
    music_df = explode_music_turns(blind_df)

    last_conv_turns = (
        blind_df.explode("conversations").unnest("conversations")
        .group_by("session_id").agg(pl.col("turn_number").max().alias("turn_number"))
    )

    all_meta = blind_df.select(["session_id", "user_id", "session_date"]).unique(
        subset=["session_id"]
    ).with_columns(
        pl.col("session_date").cast(pl.Utf8).str.strptime(pl.Date, format="%Y-%m-%d", strict=False)
    )
    gt_rows = (
        all_meta.join(last_conv_turns, on="session_id")
        .with_columns(pl.lit(None, dtype=pl.Utf8).alias("track_id"))
        .select(["session_id", "user_id", "session_date", "turn_number", "track_id"])
    )

    inject = getattr(rec, "urm_mode", "user") == "user"
    full_df = pl.concat([music_df, gt_rows])
    # Attach session-level conversation_goal so build_context_df can pass it
    # through to recommenders that consume it (e.g. HeuristicRecommender).
    if "conversation_goal" in blind_df.columns:
        goal_meta = blind_df.select(["session_id", "conversation_goal"]).unique(
            subset=["session_id"]
        )
        full_df = full_df.join(goal_meta, on="session_id", how="left")
    context_df, _ = build_context_df(full_df, inject_multi_session=inject)
    recs = rec.recommend(context_df, top_k=top_k, remove_seen=remove_seen)

    # feature_bert4rec.recommend() returns a `turn` column hardcoded to the
    # `turn=8` default kwarg for every session (it doesn't read target_turn
    # back from context_df). That's invisible on cg_val (where every session's
    # max turn_number is 8) but on blind-A the position of the last user turn
    # varies per session, so the submission JSON would carry an incorrect
    # constant 8. Overwrite with the real per-session last conversation turn
    # — equivalent to last_turn=tn in legacy predict_blind_a launchers.
    recs = recs.drop("turn").join(
        last_conv_turns.rename({"turn_number": "turn"}),
        on="session_id", how="left",
    )
    if "user_id" not in recs.columns:
        recs = recs.join(all_meta.select(["session_id", "user_id"]), on="session_id", how="left")
    return recs


def _predict_sessions_text(
    rec, blind_df: pl.DataFrame, top_k: int, remove_seen: bool, query_bundle=None,
) -> pl.DataFrame:
    """Last-turn recs via rec.recommend_text() (TextCGRecommender subclasses).

    Builds sess_info with ctx_tracks (all PRIOR music turns) and the last
    conversation turn as the target turn. When `query_bundle` is given, the
    precomputed query_text + query embedding for that target turn are attached
    (the blind parquet stores the query inside `conversations`, not as the
    named fields recommend_text reads). Returns the raw recs DataFrame.
    """
    music_df = explode_music_turns(blind_df)

    last_conv_turns = (
        blind_df.explode("conversations").unnest("conversations")
        .group_by("session_id").agg(pl.col("turn_number").max().alias("turn_number"))
    )

    session_meta = (
        blind_df.select(["session_id", "user_id", "session_date"]).unique(subset=["session_id"])
        .with_columns(
            pl.col("session_date").cast(pl.Utf8).str.strptime(pl.Date, format="%Y-%m-%d", strict=False)
        )
    )

    # ctx_tracks = music tracks from turns strictly before the target turn
    # (the last conversation turn). Matches the OOF builder, which uses prior
    # turns only — so a cold turn-1 session has empty ctx and must rely on the
    # query signals attached below.
    ctx_df = (
        music_df.join(last_conv_turns.rename({"turn_number": "target_turn"}), on="session_id")
        .filter(pl.col("turn_number") < pl.col("target_turn"))
        .group_by("session_id")
        .agg(pl.col("track_id").drop_nulls().alias("ctx_tracks"))
    )

    sess_info = (
        last_conv_turns
        .join(session_meta, on="session_id")
        .join(ctx_df, on="session_id", how="left")
        .with_columns([
            pl.col("ctx_tracks").fill_null([]),
            pl.lit(None, dtype=pl.Utf8).alias("track_id"),
        ])
    )
    if query_bundle is not None:
        from embedding_based.query_tower import attach_query
        sess_info = attach_query(sess_info, query_bundle)

    return rec.recommend_text(sess_info, top_k=top_k, remove_seen=remove_seen)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_full_params(best_params: dict, fixed_params: dict) -> dict:
    """Merge tuned + fixed params and resolve repo-root-relative path-like keys.

    Tuned params win over fixed on key collision. Path resolution is delegated
    to `resolve_param_paths` so the tuner and the exporter stay in sync.
    """
    return resolve_param_paths({**(fixed_params or {}), **(best_params or {})})


def _fit(class_name: str, module_name: str, params: dict, urm_mode: str,
         train_df: pl.DataFrame, track_meta, label: str):
    t0 = time.time()
    rec = instantiate_rec(class_name, module_name, params, urm_mode)
    rec.fit(train_df, track_metadata=track_meta)
    print(f"  [fit] {label}: {time.time() - t0:.1f}s")
    return rec


def _infer_multiturn(rec, df: pl.DataFrame, top_k: int, inference_mode: str, track_meta,
                     out_path: Path, label: str, n_turns: int = 8, query_bundle=None):
    """Run inference at every (session, turn) in df, concatenate into single parquet.

    Delegates the per-turn slicing to `run_inference_dispatch`, which now
    natively iterates over every turn T present in the input and emits one
    row per (session, target_turn). The `n_turns` argument is retained for
    the printed summary only.
    """
    t0 = time.time()
    recs = run_inference_dispatch(rec, df, top_k, inference_mode, track_meta, query_bundle)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    recs.write_parquet(out_path)
    n_sessions = recs["session_id"].n_unique() if "session_id" in recs.columns else "?"
    print(f"  [infer_mt] {label}: {recs.shape[0]} rows ({n_sessions} sess × {n_turns} turns)"
          f"  {time.time()-t0:.1f}s  → {out_path.name}")


# ---------------------------------------------------------------------------
# Core export logic
# ---------------------------------------------------------------------------

def _export_one(
    model_name: str, urm_mode: str, cfg_path: Path, args: argparse.Namespace,
    *, folder_key: str | None = None,
) -> None:
    """Export checkpoints + datasets + submission for a single (model, urm_mode).

    `folder_key` overrides the default `f"{model_name}_{urm_mode}"` output
    sub-directory under `args.storage_dir`. Used by DRO launcher to write to
    `{model}_{urm}_dro/` while keeping the actual model_name / urm_mode used
    by the recommender constructor unchanged.
    """
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    class_name     = cfg["class"]
    module_name    = cfg["module"]
    # cv_best YAML stores tuned (Optuna) params and fixed params separately,
    # mirroring tune_crossvalidation.py. The recommender constructor needs
    # both merged, with paths resolved. Older YAMLs that lack fixed_params
    # still work (the merge with {} is a no-op).
    best_params    = _resolve_full_params(
        cfg.get("best_params") or {}, cfg.get("fixed_params") or {},
    )
    inference_mode = cfg.get("inference_mode", "standard")
    folder_key     = folder_key if folder_key is not None else f"{model_name}_{urm_mode}"

    out_dir    = repo_path(args.storage_dir) / folder_key
    splitk_dir = repo_path(args.splitk_dir)
    n_folds    = args.n_folds
    top_k      = args.top_k

    print(f"\n{'='*60}")
    print(f"[export] model:     {model_name} ({class_name})")
    print(f"[export] urm_mode:  {urm_mode}")
    print(f"[export] inference: {inference_mode}")
    print(f"[export] top_k:     {top_k}  n_folds={n_folds}")
    print(f"[export] out_dir:   {out_dir}")
    print(f"[export] tasks:     datasets={'NO' if args.skip_datasets else 'YES'}"
          f"  checkpoints={'NO' if args.skip_checkpoints else 'YES'}"
          f"  submission={'NO' if args.skip_submission else 'YES'}")

    if model_name in _MODELS_NO_META:
        track_meta = None
    else:
        print("\nLoading track metadata...")
        track_meta = pl.read_parquet(repo_path(_TRACK_META_PATH))

    # Precomputed query-tower bundles (query_text + query embeddings) are only
    # consumed by text-mode recommenders; standard mode ignores them.
    def _qbundle(set_key: str, fold: int | None = None):
        if inference_mode != "text":
            return None
        return load_query_bundle(repo_path(QUERY_TOWER_BASE), set_key, fold)

    # --- Per-fold: datasets + checkpoints ---
    if not (args.skip_datasets and args.skip_checkpoints):
        ds_dir   = out_dir / "datasets"
        ckpt_dir = out_dir / "checkpoints"
        ds_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        for fold in range(n_folds):
            print(f"\n--- fold {fold} ---")

            cg_val_df      = load_eval(splitk_dir, fold)           # fold_{k}_cg_val (16%)
            reranker_val_df = load_reranker_val(splitk_dir, fold)  # fold_{k}_reranker_val (4%)

            # A: fit on cg_train (80%) → OOF predictions on cg_val
            cg_df  = load_fold(splitk_dir, fold, parts=["cg_train"])
            rec_cg = _fit(class_name, module_name, best_params, urm_mode,
                          cg_df, track_meta, f"fold_{fold}_cg_train")

            if not args.skip_checkpoints:
                p = ckpt_dir / f"fold_{fold}_cg_train.pkl"
                rec_cg.save(p)
                print(f"  [save] {p.name}")

            if not args.skip_datasets:
                _infer_multiturn(rec_cg, cg_val_df, top_k, inference_mode, track_meta,
                                 ds_dir / f"fold_{fold}_oof_cg_val.parquet",
                                 f"fold_{fold}_cg_train→cg_val",
                                 query_bundle=_qbundle("cg_val", fold))

            # B: fit on cg_train+cg_val (96%) → predictions on reranker_val
            cr_df  = load_fold(splitk_dir, fold, parts=["cg_train", "cg_val"])
            rec_cr = _fit(class_name, module_name, best_params, urm_mode,
                          cr_df, track_meta, f"fold_{fold}_cg_train_val")

            if not args.skip_checkpoints:
                p = ckpt_dir / f"fold_{fold}_cg_train_val.pkl"
                rec_cr.save(p)
                print(f"  [save] {p.name}")

            if not args.skip_datasets:
                _infer_multiturn(rec_cr, reranker_val_df, top_k, inference_mode, track_meta,
                                 ds_dir / f"fold_{fold}_oof_reranker_val.parquet",
                                 f"fold_{fold}_cg_train_val→reranker_val",
                                 query_bundle=_qbundle("reranker_val", fold))

    # --- Holdout candidates: fit on NON-holdout, infer on holdout_test (leakage-free) ---
    if not args.skip_holdout_candidates:
        print("\n--- holdout candidates ---")
        non_holdout = load_fold(splitk_dir, 0,
                                parts=["cg_train", "cg_val", "reranker_val"])
        print(f"  non-holdout: {non_holdout.shape[0]} rows")
        rec_nh = _fit(class_name, module_name, best_params, urm_mode,
                      non_holdout, track_meta, "non_holdout")
        if not args.skip_checkpoints:
            ckpt_dir = out_dir / "checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            rec_nh.save(ckpt_dir / "non_holdout.pkl")
            print("  [save] non_holdout.pkl")
        holdout_df = pl.read_parquet(repo_path(_HOLDOUT_PATH))
        _infer_multiturn(rec_nh, holdout_df, top_k, inference_mode, track_meta,
                         out_dir / "datasets" / "holdout_candidates.parquet",
                         "non_holdout→holdout",
                         query_bundle=_qbundle("holdout"))

    # --- Full dataset: checkpoint + blind submission + blind candidates ---
    if not args.skip_submission or not args.skip_checkpoints or not args.skip_blind_candidates:
        print("\n--- full dataset ---")
        print("Loading full training data...")
        full_df = pl.concat([pl.read_parquet(repo_path(p)) for p in _FULL_TRAIN_PATHS])
        print(f"  {full_df.shape[0]} rows")

        rec_full = _fit(class_name, module_name, best_params, urm_mode,
                        full_df, track_meta, "full")

        if not args.skip_checkpoints:
            ckpt_dir = out_dir / "checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            p = ckpt_dir / "full.pkl"
            rec_full.save(p)
            print(f"  [save] {p.name}")

        if not args.skip_submission or not args.skip_blind_candidates:
            print("Loading blind set...")
            blind_df = pl.read_parquet(repo_path(_BLIND_PATH))
            print(f"  {blind_df.shape[0]} sessions")
            if hasattr(rec_full, "encode_additional"):
                rec_full.encode_additional(blind_df)
            # One inference at the larger top_k serves both parquet and submission.
            blind_k = top_k if not args.skip_blind_candidates else 20
            print(f"Predicting blind-A (top_k={blind_k})...")
            if inference_mode == "text":
                recs = _predict_sessions_text(
                    rec_full, blind_df, top_k=blind_k, remove_seen=True,
                    query_bundle=_qbundle("blind"),
                )
            else:
                recs = _predict_sessions(
                    rec_full, blind_df, top_k=blind_k, remove_seen=True,
                )

            if not args.skip_blind_candidates:
                bc_path = out_dir / "datasets" / "blind_candidates.parquet"
                bc_path.parent.mkdir(parents=True, exist_ok=True)
                recs.with_columns(
                    pl.lit(None, dtype=pl.Utf8).alias("gt_track_id")
                ).write_parquet(bc_path)
                print(f"  [blind candidates] {recs.shape[0]} sessions → {bc_path.name}")

            if not args.skip_submission:
                results = _recs_to_submission(recs, 20)
                sub_path = out_dir / "submission" / f"blind_A_{folder_key}.json"
                sub_path.parent.mkdir(parents=True, exist_ok=True)
                with open(sub_path, "w") as f:
                    json.dump(results, f, indent=2)
                print(f"  [submission] {len(results)} sessions → {sub_path}")

    print(f"\n[export] done — {out_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    objective_k = args.objective_k or _OBJECTIVE_DEFAULT_K[args.objective]
    suffix = _cfg_suffix(args.objective, objective_k)

    if args.all:
        entries = _discover_configs(suffix)
        if not entries:
            sys.exit(f"[export] No cv_best_*{suffix}.yaml found in {_PKG_ROOT / 'configs'}")
        print(f"[export] Found {len(entries)} configs to process (suffix={suffix})")
        for model_name, urm_mode, cfg_path in entries:
            _export_one(model_name, urm_mode, cfg_path, args)
        return

    if args.model is None or args.urm_mode is None:
        sys.exit("[export] Provide --model and --urm_mode, or use --all")

    model_name = args.model
    urm_mode   = args.urm_mode
    folder_key = f"{model_name}_{urm_mode}"

    if args.config:
        cfg_path = Path(args.config)
    else:
        # Prefer objective-tagged YAML; fall back to legacy bare name so
        # older configs (pre-objective-tag) still load.
        tagged = pkg_path(f"configs/cv_best_{folder_key}{suffix}.yaml")
        bare   = pkg_path(f"configs/cv_best_{folder_key}.yaml")
        cfg_path = tagged if tagged.exists() else bare
    if not cfg_path.exists():
        sys.exit(f"[export] Config not found: {cfg_path}\nRun extract_best_params.py first.")

    _export_one(model_name, urm_mode, cfg_path, args)


if __name__ == "__main__":
    main()