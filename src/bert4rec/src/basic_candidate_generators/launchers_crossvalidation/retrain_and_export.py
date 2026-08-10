from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

_PKG_ROOT  = Path(__file__).resolve().parent.parent
_SRC_ROOT  = _PKG_ROOT / "src"
_CV_ROOT   = _PKG_ROOT / "launchers_crossvalidation"
sys.path.insert(0, str(_SRC_ROOT))
sys.path.insert(0, str(_CV_ROOT))

import polars as pl
import yaml

from _cv_utils import (
    format_by_turn,
    instantiate_rec,
    load_blind_b_eval,
    load_eval,
    load_fold,
    load_reranker_val,
    pkg_path,
    repo_path,
    resolve_param_paths,
    run_inference_dispatch,
    score_by_turn,
    score_fold,
)
from recommenders.interactions import explode_music_turns
from recommenders.user_base import build_context_df

_FULL_TRAIN_PATHS = [
    "data/talkpl-ai/TalkPlayData-Challenge-Dataset/data/train-00000-of-00001.parquet",
    "data/talkpl-ai/TalkPlayData-Challenge-Dataset/data/test-00000-of-00001.parquet",
]
_BLIND_PATH = (
    "data/talkpl-ai/TalkPlayData-Challenge-Blind-A/data/test-00000-of-00001.parquet"
)
_BLIND_B_PATH = "data/talkpl-ai/TalkPlayData-Challenge-Blind-B/data/test-00000-of-00001.parquet"
_HOLDOUT_PATH = "data/splitK/holdout_test.parquet"
_TRACK_META_PATH = (
    "data/talkpl-ai/TalkPlayData-Challenge-Track-Metadata/"
    "data/all_tracks-00000-of-00001.parquet"
)
_MODELS_NO_META = {"gf_cf"}

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

    p.add_argument("--predict_blindB", action="store_true",
                   help="INFERENCE-ONLY: skip all training, load the existing "
                        "full.pkl (the SAME checkpoint that produced the blind-A "
                        "submission), and replicate the blind-A export (candidate "
                        "parquet + submission JSON) on the blind-B set. Writes "
                        "datasets/blind_candidates_B.parquet and "
                        "submission/blind_B_{folder_key}.json under the same out_dir.")
    p.add_argument("--blindB_path", default=_BLIND_B_PATH,
                   help="Repo-root-relative path to the blind-B test parquet "
                        f"(default: {_BLIND_B_PATH}). Only used with --predict_blindB.")
    p.add_argument("--checkpoint", default=None,
                   help="Override the full.pkl path for --predict_blindB. Default: "
                        "{storage_dir}/{folder_key}/checkpoints/full.pkl (the blind-A "
                        "checkpoint). Only used with --predict_blindB.")
    p.add_argument("--blindB_query_split", default="blind_b",
                   help="Query-cache split name to APPEND to the checkpoint's "
                        "query_cache_splits for blind-B (query-injection models). "
                        "Its <split>.npy must exist under the model's query_emb_dir, "
                        "else blind-B turns silently lose query injection. "
                        "Only used with --predict_blindB.")
    p.add_argument("--blindB_min_query_cov", type=float, default=0.5,
                   help="Abort blind-B export if query coverage of the target turns "
                        "falls below this (guards the silent no-query degradation). "
                        "Only used with --predict_blindB; ignored for non-query models.")

    p.add_argument("--refresh_fold_datasets", action="store_true",
                   help="INFERENCE-ONLY: skip all training, reload the per-fold "
                        "checkpoints (fold_{k}_cg_train.pkl and "
                        "fold_{k}_cg_train_val.pkl) and regenerate the OOF parquets "
                        "(fold_{k}_oof_cg_val.parquet and "
                        "fold_{k}_oof_reranker_val.parquet) from them. Use to "
                        "re-export the fold candidates (e.g. at a different --top_k) "
                        "without refitting.")
    p.add_argument("--refresh_holdout_candidates", action="store_true",
                   help="INFERENCE-ONLY: skip all training, reload non_holdout.pkl "
                        "and regenerate datasets/holdout_candidates.parquet from it "
                        "(with the same holdout score report as a full run). "
                        "Combinable with --refresh_fold_datasets.")
    p.add_argument("--refresh_blind_candidates", action="store_true",
                   help="INFERENCE-ONLY: skip all training, reload checkpoints/full.pkl "
                        "and regenerate datasets/blind_candidates.parquet (Blind-A, "
                        "submission-turn) from it, plus submission/blind_A_<folder>.json "
                        "unless --skip_blind_candidates/--skip_submission. "
                        "Combinable with --refresh_fold_datasets and "
                        "--refresh_holdout_candidates.")
    p.add_argument("--folder_suffix", default=None,
                   help="Override the output folder suffix appended after "
                        "'<model>_<urm_mode>'. Default: '_ndcg' when "
                        "--objective ndcg, empty otherwise (matches the "
                        "reranker's <model>_session / <model>_session_ndcg "
                        "convention). Pass '' to force no suffix.")
    return p.parse_args()

_OBJECTIVE_DEFAULT_K = {"ndcg": 20, "recall": 200}

def _cfg_suffix(objective: str, objective_k: int) -> str:
    return f"_{objective}{objective_k}"

def _discover_configs(suffix: str) -> list[tuple[str, str, Path]]:
    results = []
    pattern = f"cv_best_*{suffix}.yaml"
    paths = sorted((_PKG_ROOT / "configs").glob(pattern))
    if not suffix:

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

    if "conversation_goal" in blind_df.columns:
        goal_meta = blind_df.select(["session_id", "conversation_goal"]).unique(
            subset=["session_id"]
        )
        full_df = full_df.join(goal_meta, on="session_id", how="left")
    context_df, _ = build_context_df(full_df, inject_multi_session=inject)
    recs = rec.recommend(context_df, top_k=top_k, remove_seen=remove_seen)

    recs = recs.drop("turn").join(
        last_conv_turns.rename({"turn_number": "turn"}),
        on="session_id", how="left",
    )
    if "user_id" not in recs.columns:
        recs = recs.join(all_meta.select(["session_id", "user_id"]), on="session_id", how="left")
    return recs

def _predict_sessions_text(
    rec, blind_df: pl.DataFrame, top_k: int, remove_seen: bool
) -> pl.DataFrame:
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

    ctx_df = (
        music_df.group_by("session_id")
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

    return rec.recommend_text(sess_info, top_k=top_k, remove_seen=remove_seen)

def _resolve_full_params(best_params: dict, fixed_params: dict) -> dict:
    return resolve_param_paths({**(fixed_params or {}), **(best_params or {})})

def _fit(class_name: str, module_name: str, params: dict, urm_mode: str,
         train_df: pl.DataFrame, track_meta, label: str):
    t0 = time.time()
    rec = instantiate_rec(class_name, module_name, params, urm_mode)
    rec.fit(train_df, track_metadata=track_meta)
    print(f"  [fit] {label}: {time.time() - t0:.1f}s")
    return rec

def _infer_multiturn(rec, df: pl.DataFrame, top_k: int, inference_mode: str, track_meta,
                     out_path: Path, label: str, n_turns: int = 8):
    t0 = time.time()
    recs = run_inference_dispatch(rec, df, top_k, inference_mode, track_meta)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    recs.write_parquet(out_path)
    n_sessions = recs["session_id"].n_unique() if "session_id" in recs.columns else "?"
    print(f"  [infer_mt] {label}: {recs.shape[0]} rows ({n_sessions} sess × {n_turns} turns)"
          f"  {time.time()-t0:.1f}s  → {out_path.name}")

def _score_holdout(recs_path: Path, train_df: pl.DataFrame, min_turn: int) -> None:
    recs = pl.read_parquet(recs_path)
    warm = set(
        explode_music_turns(train_df)["track_id"].drop_nulls().unique().to_list()
    )

    def _report(tag: str, sub: pl.DataFrame) -> None:
        if sub.is_empty():
            print(f"  [holdout score] {tag}: n=0")
            return
        m = score_fold(sub, recall_ks=[20, 200], ndcg_ks=[20])
        print(f"  [holdout score] {tag}: n={sub.height}  "
              f"ndcg@20={m['ndcg@20']:.4f}  recall@20={m['recall@20']:.4f}  "
              f"recall@200={m['recall@200']:.4f}")

    print(f"\n  [holdout score] warm track set = {len(warm)} tracks; min_turn={min_turn}")
    recs_t = recs.filter(pl.col("gt_turn_number") >= min_turn) if min_turn > 1 else recs
    is_warm = pl.col("gt_track_id").is_in(list(warm))
    _report("overall", recs_t)
    _report("warm-GT", recs_t.filter(is_warm))
    _report("cold-GT", recs_t.filter(~is_warm))
    print(format_by_turn(score_by_turn(recs_t, recall_ks=[20, 200], ndcg_ks=[20])))

def _load_ckpt(ckpt_path: Path, class_name: str, module_name: str,
               params: dict, urm_mode: str):
    if not ckpt_path.is_absolute():
        ckpt_path = repo_path(str(ckpt_path))
    if not ckpt_path.exists():
        sys.exit(f"[refresh] checkpoint not found: {ckpt_path}\n"
                 f"        Run the retrain first (it writes the fold checkpoints).")
    print(f"  [load] {ckpt_path.name}")
    with open(ckpt_path, "rb") as f:
        state = pickle.load(f)
    rec = instantiate_rec(class_name, module_name, params,
                          state.get("urm_mode", urm_mode))
    rec._set_model_state(state)
    return rec


def _refresh_fold_datasets(
    class_name: str, module_name: str, best_params: dict, urm_mode: str,
    inference_mode: str, out_dir: Path, args: argparse.Namespace,
) -> None:
    splitk_dir = repo_path(args.splitk_dir)
    ds_dir   = out_dir / "datasets"
    ckpt_dir = out_dir / "checkpoints"
    ds_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"[refresh] out_dir:   {out_dir}")
    print(f"[refresh] inference: {inference_mode}")
    print(f"[refresh] top_k:     {args.top_k}  n_folds={args.n_folds}")

    for fold in range(args.n_folds):
        print(f"\n--- fold {fold} (from checkpoints) ---")
        for ckpt_name, eval_loader, out_name, label in (
            (f"fold_{fold}_cg_train.pkl", load_eval,
             f"fold_{fold}_oof_cg_val.parquet",
             f"fold_{fold}_cg_train→cg_val"),
            (f"fold_{fold}_cg_train_val.pkl", load_reranker_val,
             f"fold_{fold}_oof_reranker_val.parquet",
             f"fold_{fold}_cg_train_val→reranker_val"),
        ):
            rec = _load_ckpt(ckpt_dir / ckpt_name, class_name, module_name,
                             best_params, urm_mode)
            eval_df = eval_loader(splitk_dir, fold)
            _infer_multiturn(rec, eval_df, args.top_k, inference_mode, None,
                             ds_dir / out_name, label)

    print(f"\n[refresh] done — {ds_dir}")


def _refresh_holdout_candidates(
    class_name: str, module_name: str, best_params: dict, urm_mode: str,
    inference_mode: str, out_dir: Path, args: argparse.Namespace,
) -> None:
    splitk_dir = repo_path(args.splitk_dir)
    ds_dir   = out_dir / "datasets"
    ckpt_dir = out_dir / "checkpoints"
    ds_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"[refresh] holdout candidates from non_holdout.pkl")
    print(f"[refresh] out_dir:   {out_dir}")
    print(f"[refresh] inference: {inference_mode}")
    print(f"[refresh] top_k:     {args.top_k}")

    rec = _load_ckpt(ckpt_dir / "non_holdout.pkl", class_name, module_name,
                     best_params, urm_mode)

    holdout_path = splitk_dir / "holdout_test.parquet"
    if not holdout_path.exists():
        holdout_path = repo_path(_HOLDOUT_PATH)
    holdout_df = pl.read_parquet(holdout_path)
    print(f"[refresh] holdout:   {holdout_path}  ({holdout_df.shape[0]} rows)")

    hc_path = ds_dir / "holdout_candidates.parquet"
    _infer_multiturn(rec, holdout_df, args.top_k, inference_mode, None,
                     hc_path, "non_holdout->holdout")

    non_holdout = load_fold(splitk_dir, 0,
                            parts=["cg_train", "cg_val", "reranker_val"])
    _score_holdout(hc_path, non_holdout, min_turn=best_params.get("eval_min_turn", 2))

    print(f"\n[refresh] done: {hc_path}")


def _refresh_blind_candidates(
    model_name: str, class_name: str, module_name: str, best_params: dict, urm_mode: str,
    inference_mode: str, out_dir: Path, folder_key: str, args: argparse.Namespace,
) -> None:
    ds_dir   = out_dir / "datasets"
    ckpt_dir = out_dir / "checkpoints"
    ds_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"[refresh] blind-A candidates from full.pkl")
    print(f"[refresh] model:     {model_name}")
    print(f"[refresh] out_dir:   {out_dir}")
    print(f"[refresh] inference: {inference_mode}")
    print(f"[refresh] top_k:     {args.top_k}")

    rec_full = _load_ckpt(ckpt_dir / "full.pkl", class_name, module_name,
                          best_params, urm_mode)

    print("Loading blind set...")
    blind_df = pl.read_parquet(repo_path(_BLIND_PATH))
    print(f"  {blind_df.shape[0]} sessions")
    if hasattr(rec_full, "encode_additional"):
        rec_full.encode_additional(blind_df)
    _pred_fn = _predict_sessions_text if inference_mode == "text" else _predict_sessions

    blind_k = args.top_k if not args.skip_blind_candidates else 20
    print(f"Predicting blind-A (top_k={blind_k})...")
    recs = _pred_fn(rec_full, blind_df, top_k=blind_k, remove_seen=True)

    if not args.skip_blind_candidates:
        bc_path = ds_dir / "blind_candidates.parquet"
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

    print(f"\n[refresh] done — {ds_dir}")


def _export_blindB(
    model_name: str, class_name: str, module_name: str, best_params: dict, urm_mode: str,
    inference_mode: str, out_dir: Path, folder_key: str, args: argparse.Namespace,
) -> None:
    blind_path = repo_path(args.blindB_path)
    if not blind_path.exists():
        sys.exit(f"[blindB] blind-B data not found: {blind_path}\n"
                 f"        Pass --blindB_path with the real blind-B test parquet location.")

    print(f"\n{'='*60}")
    print(f"[blindB] model:      {model_name} ({class_name})  (urm_mode={urm_mode})")
    print(f"[blindB] inference:  {inference_mode}")
    print(f"[blindB] blind data: {blind_path}")
    print(f"[blindB] out_dir:    {out_dir}")
    print(f"[blindB] top_k:      {args.top_k}")

    ckpt_path = Path(args.checkpoint) if args.checkpoint else out_dir / "checkpoints" / "full.pkl"
    if not ckpt_path.is_absolute():
        ckpt_path = repo_path(str(ckpt_path))
    if not ckpt_path.exists():
        sys.exit(f"[blindB] checkpoint not found: {ckpt_path}\n"
                 f"        Run the blind-A retrain first (writes checkpoints/full.pkl), "
                 f"or pass --checkpoint.")
    print(f"[blindB] checkpoint: {ckpt_path}")
    with open(ckpt_path, "rb") as f:
        state = pickle.load(f)

    if "query_cache_splits" in state:
        splits = list(state["query_cache_splits"])
        if args.blindB_query_split not in splits:
            splits.append(args.blindB_query_split)
        state["query_cache_splits"] = splits
        print(f"[blindB] query model: query_emb_dir={state.get('query_emb_dir')}  "
              f"query_cache_splits -> {splits}")

    ckpt_urm = state.get("urm_mode", urm_mode)
    rec_full = instantiate_rec(class_name, module_name, best_params, ckpt_urm)
    rec_full._set_model_state(state)

    blind_df = pl.read_parquet(blind_path)
    print(f"  {blind_df.shape[0]} sessions")
    if hasattr(rec_full, "encode_additional"):
        rec_full.encode_additional(blind_df)

    if getattr(rec_full, "_query_lookup", None):
        target_turns = (
            blind_df.explode("conversations").unnest("conversations")
            .group_by("session_id").agg(pl.col("turn_number").max().alias("turn_number"))
        )
        tgt = [(r["session_id"], int(r["turn_number"]))
               for r in target_turns.iter_rows(named=True)]
        n_cov = sum(1 for k in tgt if k in rec_full._query_lookup)
        cov = n_cov / max(1, len(tgt))
        print(f"[blindB] query coverage of target turns: {n_cov}/{len(tgt)} = {cov:.1%}")
        if cov < args.blindB_min_query_cov:
            sys.exit(
                f"[blindB] ABORT: query coverage {cov:.1%} < {args.blindB_min_query_cov:.0%}. "
                f"The blind-B cache for split '{args.blindB_query_split}' is missing or "
                f"keyed differently under {state.get('query_emb_dir')}. Encode it first.")

    _pred_fn = _predict_sessions_text if inference_mode == "text" else _predict_sessions
    blind_k = args.top_k if not args.skip_blind_candidates else 20
    print(f"Predicting blind-B (top_k={blind_k})...")
    recs = _pred_fn(rec_full, blind_df, top_k=blind_k, remove_seen=True)

    if "fallback_used" in recs.columns:
        fb = recs["fallback_used"]

        try:
            n_fb = sum(1 for v in fb.to_list() if v and (v[0] if isinstance(v, list) else v))
        except Exception:
            n_fb = 0
        if n_fb:
            print(f"[blindB] WARNING: {n_fb}/{recs.height} sessions fell back to "
                  f"popularity (empty history).")

    if not args.skip_blind_candidates:
        bc_path = out_dir / "datasets" / "blind_candidates_B.parquet"
        bc_path.parent.mkdir(parents=True, exist_ok=True)
        recs.with_columns(
            pl.lit(None, dtype=pl.Utf8).alias("gt_track_id")
        ).write_parquet(bc_path)
        print(f"  [blind-B candidates] {recs.shape[0]} sessions → {bc_path.name}")

    if not args.skip_submission:
        results = _recs_to_submission(recs, 20)
        sub_path = out_dir / "submission" / f"blind_B_{folder_key}.json"
        sub_path.parent.mkdir(parents=True, exist_ok=True)
        with open(sub_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  [submission] {len(results)} sessions → {sub_path}")

    if not args.skip_datasets:
        eval_df, gt_map = load_blind_b_eval(blind_path)
        print(f"[blindB] per-turn: {eval_df.shape[0]} (session,turn) rows, "
              f"{eval_df['session_id'].n_unique()} sessions, {gt_map.height} with known GT")
        pt_recs = run_inference_dispatch(rec_full, eval_df, args.top_k, inference_mode, None)
        turn_col = "turn" if "turn" in pt_recs.columns else "gt_turn_number"
        pt_recs = pt_recs.with_columns(pl.col(turn_col).cast(pl.Int64).alias("turn"))
        pt_recs = pt_recs.drop(
            [c for c in ("gt_track_id", "gt_track_id_right") if c in pt_recs.columns]
        )
        pt_recs = pt_recs.join(gt_map, on=["session_id", "turn"], how="left")
        pt_path = out_dir / "datasets" / "blind_b_candidates.parquet"
        pt_path.parent.mkdir(parents=True, exist_ok=True)
        pt_recs.write_parquet(pt_path)
        n_gt = pt_recs.filter(pl.col("gt_track_id").is_not_null()).height
        print(f"  [blind-B per-turn] {pt_recs.shape[0]} rows "
              f"({pt_recs['session_id'].n_unique()} sess, gt={n_gt}) → {pt_path.name}")

    print(f"\n[blindB] done — {out_dir}")

def _export_one(
    model_name: str, urm_mode: str, cfg_path: Path, args: argparse.Namespace,
    *, folder_key: str | None = None,
) -> None:
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    class_name     = cfg["class"]
    module_name    = cfg["module"]

    best_params    = _resolve_full_params(
        cfg.get("best_params") or {}, cfg.get("fixed_params") or {},
    )

    _obj_k = args.objective_k or (20 if args.objective == "ndcg" else 200)
    best_params.setdefault("early_stop_metric", args.objective)
    best_params.setdefault("eval_min_turn", 2)
    if args.objective == "recall":
        best_params.setdefault("eval_recall_k", _obj_k)
        best_params["early_stop_patience"] = 25
    else:
        best_params.setdefault("eval_ndcg_k", _obj_k)
    print(f"[export] early-stop: {best_params['early_stop_metric']}@"
          f"{best_params.get('eval_recall_k') if args.objective=='recall' else best_params.get('eval_ndcg_k')}"
          f"  turns>={best_params['eval_min_turn']}  patience={best_params.get('early_stop_patience','default')}")
    inference_mode = cfg.get("inference_mode", "standard")
    if folder_key is None:
        folder_key = f"{model_name}_{urm_mode}"
        suffix = args.folder_suffix
        if suffix is None:
            suffix = "_ndcg" if args.objective == "ndcg" else ""
        folder_key += suffix

    out_dir    = repo_path(args.storage_dir) / folder_key

    if args.predict_blindB:
        _export_blindB(model_name, class_name, module_name, best_params, urm_mode,
                       inference_mode, out_dir, folder_key, args)
        return

    if (args.refresh_fold_datasets or args.refresh_holdout_candidates
            or args.refresh_blind_candidates):
        if args.refresh_fold_datasets:
            _refresh_fold_datasets(class_name, module_name, best_params, urm_mode,
                                   inference_mode, out_dir, args)
        if args.refresh_holdout_candidates:
            _refresh_holdout_candidates(class_name, module_name, best_params, urm_mode,
                                        inference_mode, out_dir, args)
        if args.refresh_blind_candidates:
            _refresh_blind_candidates(model_name, class_name, module_name, best_params, urm_mode,
                                      inference_mode, out_dir, folder_key, args)
        return

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

    if not (args.skip_datasets and args.skip_checkpoints):
        ds_dir   = out_dir / "datasets"
        ckpt_dir = out_dir / "checkpoints"
        ds_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        for fold in range(n_folds):
            print(f"\n--- fold {fold} ---")

            cg_val_df      = load_eval(splitk_dir, fold)
            reranker_val_df = load_reranker_val(splitk_dir, fold)

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
                                 f"fold_{fold}_cg_train→cg_val")

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
                                 f"fold_{fold}_cg_train_val→reranker_val")

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

        holdout_path = splitk_dir / "holdout_test.parquet"
        if not holdout_path.exists():
            holdout_path = repo_path(_HOLDOUT_PATH)
        holdout_df = pl.read_parquet(holdout_path)
        hc_path = out_dir / "datasets" / "holdout_candidates.parquet"
        _infer_multiturn(rec_nh, holdout_df, top_k, inference_mode, track_meta,
                         hc_path, "non_holdout→holdout")
        _score_holdout(hc_path, non_holdout, min_turn=best_params.get("eval_min_turn", 2))

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
            _pred_fn = _predict_sessions_text if inference_mode == "text" else _predict_sessions

            blind_k = top_k if not args.skip_blind_candidates else 20
            print(f"Predicting blind-A (top_k={blind_k})...")
            recs = _pred_fn(rec_full, blind_df, top_k=blind_k, remove_seen=True)

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

        tagged = pkg_path(f"configs/cv_best_{folder_key}{suffix}.yaml")
        bare   = pkg_path(f"configs/cv_best_{folder_key}.yaml")
        cfg_path = tagged if tagged.exists() else bare
    if not cfg_path.exists():
        sys.exit(f"[export] Config not found: {cfg_path}\nRun extract_best_params.py first.")

    _export_one(model_name, urm_mode, cfg_path, args)

if __name__ == "__main__":
    main()
