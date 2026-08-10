"""Retrain best DRO CG on the full dataset and export Blind-B candidates.

Mirrors the "full dataset" block of
`launchers_crossvalidation.retrain_and_export._export_one`, but:
  (a) RETRAINS from scratch (does not load full.pkl),
  (b) infers on Blind-B instead of Blind-A,
  (c) writes only `datasets/blind_b_candidates.parquet` — no submission JSON,
      no checkpoint (the existing full.pkl is left untouched).

Why retrain instead of loading full.pkl: text / ensemble recommenders bake
their (session_id, turn_number) → query-row lookup at fit() time by globbing
`dense_*query*` under each Qwen folder. full.pkl was fit BEFORE the Blind-B
caches existed, so it cannot resolve Blind-B queries. A fresh fit re-globs and
picks up `dense_blindb_all_query_len512_poollast`. Standard CGs (no query) just
retrain identically and infer.

Reads `best_params_{model}_{urm}_dro{suffix}.yaml` next to the optuna DB
(suffix = _cvar70 for robust_mode=cvar alpha=0.7), falling back to the legacy
package `configs/cv_best_*` copy — same resolution as retrain_and_export_dro.

Run:
    cd src/basic_candidate_generators
    uv run python -m launchers_dro.export_blind_b_dro --model hybrid_8b --urm_mode session
    uv run python -m launchers_dro.export_blind_b_dro --all
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parent.parent
_SRC_ROOT = _PKG_ROOT / "src"
_CV_ROOT  = _PKG_ROOT / "launchers_crossvalidation"
for p in (_SRC_ROOT, _CV_ROOT):
    sys.path.insert(0, str(p))

import polars as pl   # noqa: E402
import yaml           # noqa: E402

from _cv_utils import (  # noqa: E402
    load_blind_b_eval,
    repo_path,
    run_inference_dispatch,
)
from launchers_crossvalidation.retrain_and_export import (  # noqa: E402
    _FULL_TRAIN_PATHS,
    _MODELS_NO_META,
    _TRACK_META_PATH,
    _fit,
    _resolve_full_params,
)
from embedding_based.query_tower import (  # noqa: E402
    QUERY_TOWER_BASE,
    load_query_bundle,
)

# (model, urm_mode) pairs this launcher targets by default. All session-DRO,
# robust_mode=cvar alpha=0.7 (suffix _cvar70). Ordered so the two FUSION
# heuristics run LAST: heuristic_v2_hybrid consumes hybrid_all_qwen's blind-B
# candidates, heuristic_v3 consumes tower_cf_ensemble's (+ rrf_oneshot's, which
# comes from the separate oneshot pipeline). Run those upstreams first.
_DEFAULT_MODELS = [
    ("emb_item_knn_8b",     "session"),
    ("hybrid_8b",           "session"),
    ("hybrid_all_qwen",     "session"),
    ("tower_ensemble",      "session"),
    ("tower_cf_ensemble",   "session"),
    ("heuristic_v2_hybrid", "session"),
    ("heuristic_v3",        "session"),
]

_BLIND_B_PATH = (
    "data/talkpl-ai/TalkPlayData-Challenge-Blind-B/data/test-00000-of-00001.parquet"
)
# Blind-B query-tower set key (all user turns; matches dense_blindb_all_*).
_BLIND_B_SET = "blindb_all"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Retrain best DRO CG and export Blind-B candidates.")
    p.add_argument("--model",    default=None)
    p.add_argument("--urm_mode", default="session", choices=["session", "user"])
    p.add_argument("--all", action="store_true",
                   help=f"Process all default targets: {[m for m, _ in _DEFAULT_MODELS]}")
    p.add_argument("--robust_mode", default="cvar", choices=["mean", "cvar", "group_dro"])
    p.add_argument("--robust_alpha", type=float, default=0.7)
    p.add_argument("--top_k", type=int, default=200)
    p.add_argument("--storage_dir", default="models/CG_crossvalidation")
    p.add_argument("--blind_b_path", default=_BLIND_B_PATH)
    return p.parse_args()


def _suffix(robust_mode: str, robust_alpha: float) -> str:
    if robust_mode == "cvar":
        return f"_cvar{int(round(robust_alpha * 100))}"
    return f"_{robust_mode}"


def _resolve_cfg(model: str, urm_mode: str, storage_dir: str, suffix: str) -> Path:
    folder = f"{model}_{urm_mode}_dro"
    storage_yaml = (
        repo_path(storage_dir) / folder / f"best_params_{folder}{suffix}.yaml"
    )
    legacy_yaml = _PKG_ROOT / "configs" / f"cv_best_{folder}{suffix}.yaml"
    if storage_yaml.exists():
        return storage_yaml
    if legacy_yaml.exists():
        return legacy_yaml
    sys.exit(
        f"[blindb] missing config for {folder}: tried {storage_yaml} and {legacy_yaml}."
    )


def _export_one_blind_b(
    model: str, urm_mode: str, cfg_path: Path, args: argparse.Namespace,
) -> None:
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    class_name     = cfg["class"]
    module_name    = cfg["module"]
    best_params    = _resolve_full_params(
        cfg.get("best_params") or {}, cfg.get("fixed_params") or {},
    )
    inference_mode = cfg.get("inference_mode", "standard")
    folder         = f"{model}_{urm_mode}_dro"
    out_dir        = repo_path(args.storage_dir) / folder

    print(f"\n{'='*60}")
    print(f"[blindb] {model} ({class_name})  urm={urm_mode}  inference={inference_mode}")
    print(f"[blindb] cfg={cfg_path}")
    print(f"[blindb] out={out_dir}")

    track_meta = None if model in _MODELS_NO_META else pl.read_parquet(repo_path(_TRACK_META_PATH))

    full_df = pl.concat([pl.read_parquet(repo_path(p)) for p in _FULL_TRAIN_PATHS])
    print(f"[blindb] full train rows: {full_df.shape[0]}")
    rec = _fit(class_name, module_name, best_params, urm_mode, full_df, track_meta, "full")

    # All-turn inference: every visible turn (known GT → internal validation) +
    # the withheld submission turn (gt null). Mirrors the holdout export path.
    eval_df, gt_map = load_blind_b_eval(repo_path(args.blind_b_path))
    print(f"[blindb] blind-B long: {eval_df.shape[0]} (session, turn) rows, "
          f"{eval_df['session_id'].n_unique()} sessions, {gt_map.height} with known GT")
    if hasattr(rec, "encode_additional"):
        rec.encode_additional(pl.read_parquet(repo_path(args.blind_b_path)))

    t0 = time.time()
    bundle = (load_query_bundle(repo_path(QUERY_TOWER_BASE), _BLIND_B_SET)
              if inference_mode == "text" else None)
    recs = run_inference_dispatch(rec, eval_df, args.top_k, inference_mode,
                                  track_meta, bundle)
    print(f"[blindb] inference {time.time()-t0:.1f}s — {recs.shape[0]} (session, turn) rows")

    turn_col = "turn" if "turn" in recs.columns else "gt_turn_number"
    recs = recs.with_columns(pl.col(turn_col).cast(pl.Int64).alias("turn"))
    recs = recs.drop([c for c in ("gt_track_id", "gt_track_id_right") if c in recs.columns])
    recs = recs.join(gt_map, on=["session_id", "turn"], how="left")
    out_path = out_dir / "datasets" / "blind_b_candidates.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    recs.write_parquet(out_path)
    n_gt = recs.filter(pl.col("gt_track_id").is_not_null()).height
    print(f"[blindb] wrote {recs.shape[0]} rows ({recs['session_id'].n_unique()} sess, "
          f"gt={n_gt}) → {out_path}")


def main() -> None:
    args = parse_args()
    suffix = _suffix(args.robust_mode, args.robust_alpha)

    if args.all:
        targets = list(_DEFAULT_MODELS)
    elif args.model is not None:
        targets = [(args.model, args.urm_mode)]
    else:
        sys.exit("[blindb] provide --model, or use --all")

    for model, urm_mode in targets:
        cfg_path = _resolve_cfg(model, urm_mode, args.storage_dir, suffix)
        _export_one_blind_b(model, urm_mode, cfg_path, args)


if __name__ == "__main__":
    main()
