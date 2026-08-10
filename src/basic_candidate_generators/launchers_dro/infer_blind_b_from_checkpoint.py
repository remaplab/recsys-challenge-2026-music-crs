"""Predict Blind-B from an already-saved DRO `full.pkl` checkpoint — no retrain.

`export_blind_b_dro.py` always retrains, reasoning (in its docstring) that
text/ensemble recommenders bake their (session, turn) -> query-row lookup at
fit() time by globbing `dense_*query*`, so a `full.pkl` fit before the Blind-B
query caches existed can't resolve Blind-B queries.

That reasoning doesn't hold for the actual SCORING path though: `recommend_text()`
(`tower_ensemble.py`/`tower_cf_ensemble.py`/`hybrid_all_qwen.py`) reads
`query_emb` straight off the `sess_info` rows it's given
(`r.get("query_emb")`) — supplied externally via `attach_query(sess_info,
query_bundle)`, where `query_bundle` is loaded standalone via
`load_query_bundle(QUERY_TOWER_BASE, "blindb_all")`, independent of the model
instance. `export_blind_b_dro.py` already does exactly this attach for its
own (freshly retrained) `rec`. The model's own fit-time `load_query_store()`
glob only ever builds TRAINING pairs — never consulted again at inference.
So whatever query coverage the checkpoint saw at fit time is irrelevant to
Blind-B scoring; loading `full.pkl` instead of refitting should work.

Same checkpoint-load path as `infer_from_checkpoints.py` (`_load_checkpoint`,
`_WARM_REFIT_MODELS` for the two ensembles + hybrid — catalogue-array rebuild
only, unrelated to the query bundle), just wired to the Blind-B eval frame +
external query bundle instead of folds/holdout/blind-A.

Writes `datasets/blind_b_candidates.parquet` — no checkpoint written, no
submission JSON (same output contract as `export_blind_b_dro.py`).

Run:
    cd src/basic_candidate_generators
    uv run python -m launchers_dro.infer_blind_b_from_checkpoint --model tower_cf_ensemble --urm_mode session
    uv run python -m launchers_dro.infer_blind_b_from_checkpoint --all
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
    _PATH_PARAM_KEYS,
    load_blind_b_eval,
    repo_path,
    resolve_param_paths,
    run_inference_dispatch,
)
from launchers_crossvalidation.retrain_and_export import (  # noqa: E402
    _MODELS_NO_META,
    _TRACK_META_PATH,
)
from embedding_based.query_tower import (  # noqa: E402
    QUERY_TOWER_BASE,
    load_query_bundle,
)

from launchers_dro.infer_from_checkpoints import (  # noqa: E402
    _WARM_REFIT_MODELS,
    _load_checkpoint,
)

# The 6 session-DRO CGs the reranker actually consumes. Ordered so the two
# fusion heuristics run LAST: heuristic_v2_hybrid reads hybrid_all_qwen's
# blind-B candidates, heuristic_v3 reads tower_cf_ensemble's (+ rrf_oneshot's,
# from the separate oneshot pipeline) — their upstreams must already be on disk.
_DEFAULT_MODELS = [
    ("emb_item_knn_8b",     "session"),
    ("hybrid_all_qwen",     "session"),
    ("tower_ensemble",      "session"),
    ("tower_cf_ensemble",   "session"),
    ("heuristic_v2_hybrid", "session"),
    ("heuristic_v3",        "session"),
]

_BLIND_B_PATH = (
    "data/talkpl-ai/TalkPlayData-Challenge-Blind-B/data/test-00000-of-00001.parquet"
)
_BLIND_B_SET = "blindb_all"  # matches dense_blindb_all_query_len512_poollast


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predict Blind-B from full.pkl — no retrain.")
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
    storage_yaml = repo_path(storage_dir) / folder / f"best_params_{folder}{suffix}.yaml"
    legacy_yaml = _PKG_ROOT / "configs" / f"cv_best_{folder}{suffix}.yaml"
    if storage_yaml.exists():
        return storage_yaml
    if legacy_yaml.exists():
        return legacy_yaml
    sys.exit(f"[blindb-ckpt] missing config for {folder}: tried {storage_yaml} and {legacy_yaml}.")


def _infer_one_blind_b(
    model: str, urm_mode: str, cfg_path: Path, args: argparse.Namespace,
) -> None:
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    class_name     = cfg["class"]
    module_name    = cfg["module"]
    inference_mode = cfg.get("inference_mode", "standard")
    resolved_fixed = resolve_param_paths(cfg.get("fixed_params", {}))
    path_overrides = {k: v for k, v in resolved_fixed.items() if k in _PATH_PARAM_KEYS}
    folder         = f"{model}_{urm_mode}_dro"
    out_dir        = repo_path(args.storage_dir) / folder
    ckpt           = out_dir / "checkpoints" / "full.pkl"

    print(f"\n{'='*60}")
    print(f"[blindb-ckpt] {model} ({class_name})  urm={urm_mode}  inference={inference_mode}")
    print(f"[blindb-ckpt] cfg={cfg_path}")
    print(f"[blindb-ckpt] ckpt={ckpt}")
    print(f"[blindb-ckpt] out={out_dir}")

    if not ckpt.exists():
        print(f"  [skip] missing {ckpt}")
        return

    track_meta = None if model in _MODELS_NO_META else pl.read_parquet(repo_path(_TRACK_META_PATH))
    warm_refit = model in _WARM_REFIT_MODELS
    rec = _load_checkpoint(class_name, module_name, ckpt,
                           warm_refit=warm_refit, track_meta=track_meta,
                           path_overrides=path_overrides)

    eval_df, gt_map = load_blind_b_eval(repo_path(args.blind_b_path))
    print(f"[blindb-ckpt] blind-B long: {eval_df.shape[0]} (session, turn) rows, "
          f"{eval_df['session_id'].n_unique()} sessions, {gt_map.height} with known GT")
    if hasattr(rec, "encode_additional"):
        rec.encode_additional(pl.read_parquet(repo_path(args.blind_b_path)))

    t0 = time.time()
    bundle = (load_query_bundle(repo_path(QUERY_TOWER_BASE), _BLIND_B_SET)
              if inference_mode == "text" else None)
    recs = run_inference_dispatch(rec, eval_df, args.top_k, inference_mode,
                                  track_meta, bundle)
    print(f"[blindb-ckpt] inference {time.time()-t0:.1f}s — {recs.shape[0]} (session, turn) rows")

    turn_col = "turn" if "turn" in recs.columns else "gt_turn_number"
    recs = recs.with_columns(pl.col(turn_col).cast(pl.Int64).alias("turn"))
    recs = recs.drop([c for c in ("gt_track_id", "gt_track_id_right") if c in recs.columns])
    recs = recs.join(gt_map, on=["session_id", "turn"], how="left")
    out_path = out_dir / "datasets" / "blind_b_candidates.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    recs.write_parquet(out_path)
    n_gt = recs.filter(pl.col("gt_track_id").is_not_null()).height
    print(f"[blindb-ckpt] wrote {recs.shape[0]} rows ({recs['session_id'].n_unique()} sess, "
          f"gt={n_gt}) → {out_path}")


def main() -> None:
    args = parse_args()
    suffix = _suffix(args.robust_mode, args.robust_alpha)

    if args.all:
        targets = list(_DEFAULT_MODELS)
    elif args.model is not None:
        targets = [(args.model, args.urm_mode)]
    else:
        sys.exit("[blindb-ckpt] provide --model, or use --all")

    for model, urm_mode in targets:
        cfg_path = _resolve_cfg(model, urm_mode, args.storage_dir, suffix)
        _infer_one_blind_b(model, urm_mode, cfg_path, args)


if __name__ == "__main__":
    main()
