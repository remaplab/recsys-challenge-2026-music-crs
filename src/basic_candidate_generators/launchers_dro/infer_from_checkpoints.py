"""Load already-saved DRO checkpoints and re-run inference only (no fit).

Mirrors `launchers_crossvalidation.retrain_and_export._export_one`'s per-split
inference calls exactly, but loads `<class>.load(ckpt)` instead of `_fit(...)`.
Meant to reproduce `datasets/` from `checkpoints/` bit-for-bit so it can be
diffed against a previously-computed `datasets_reference/`.

Checkpoint -> split -> output mapping (same as retrain_and_export_dro):
    fold_{k}_cg_train.pkl      -> cg_val split       -> fold_{k}_oof_cg_val.parquet
    fold_{k}_cg_train_val.pkl  -> reranker_val split  -> fold_{k}_oof_reranker_val.parquet
    non_holdout.pkl            -> holdout_test        -> holdout_candidates.parquet
    full.pkl                   -> blind-A (last turn) -> blind_candidates.parquet

Writes to `<out_dir>/datasets/` (NOT `datasets_reference/`), so the two can be
compared without clobbering the reference copy.

Run:
    cd src/basic_candidate_generators
    uv run python -m launchers_dro.infer_from_checkpoints \\
        --model emb_item_knn_8b --urm_mode session
    uv run python -m launchers_dro.infer_from_checkpoints \\
        --model tower_ensemble --urm_mode session --splits full holdout
"""
from __future__ import annotations

import argparse
import importlib
import pickle
import sys
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parent.parent
_SRC_ROOT = _PKG_ROOT / "src"
_CV_ROOT  = _PKG_ROOT / "launchers_crossvalidation"
_DRO_ROOT = _PKG_ROOT / "launchers_dro"
for p in (_SRC_ROOT, _CV_ROOT, _DRO_ROOT):
    sys.path.insert(0, str(p))

import polars as pl   # noqa: E402
import yaml            # noqa: E402

from _cv_utils import (   # noqa: E402
    _PATH_PARAM_KEYS,
    load_eval,
    load_reranker_val,
    repo_path,
    resolve_param_paths,
    run_inference_dispatch,
)
from launchers_crossvalidation.retrain_and_export import (   # noqa: E402
    _HOLDOUT_PATH,
    _MODELS_NO_META,
    _TRACK_META_PATH,
    _infer_multiturn,
    _predict_sessions,
    _predict_sessions_text,
)
from embedding_based.query_tower import (   # noqa: E402
    QUERY_TOWER_BASE,
    load_query_bundle,
)


# HybridCG-family DRO CGs: `_get_model_state`/`_set_model_state` only persist
# hyperparams (+ trained tower members for the two ensembles) — the base
# catalogue arrays (track_emb, tfidf/bm25 matrices, W_cbf) are NOT pickled,
# they live in a signature-keyed on-disk cache under `cache_dir` instead.
# `load()` alone leaves them unset -> AttributeError in recommend_text().
# fit(None, track_metadata=...) is an existing warm-cache-only call path
# (already used by the DRO tuner) that rebuilds exactly those arrays from
# `cache_dir` hits without retraining/touching the checkpointed tower
# members (fit() early-returns on train_df=None before member training).
_WARM_REFIT_MODELS = {"hybrid_all_qwen", "tower_ensemble", "tower_cf_ensemble"}


def _load_checkpoint(class_name: str, module_name: str, ckpt: Path,
                     warm_refit: bool = False, track_meta=None,
                     path_overrides: dict | None = None):
    cls = getattr(importlib.import_module(module_name), class_name)
    # Checkpoints pickle whatever absolute path-like hyperparams were
    # resolved at original fit time (track_emb_dir, cache_dir,
    # fallback_datasets_dir, ...) — stale if the repo has since moved/been
    # renamed. Re-resolve from the current config's `fixed_params` (relative,
    # checked into the yaml) instead of trusting the pickle, same as
    # retrain_and_export_dro does on every run.
    #
    # Can't apply this via `cls.load(ckpt)` + post-hoc setattr: some
    # `_set_model_state()` overrides (e.g. HeuristicV2Hybrid) eagerly read
    # from disk using a path attribute *during* load() itself, before a
    # post-load setattr would ever run. So patch the unpickled state dict
    # in place first, replicating BaseRecommender.load()'s own two lines.
    with open(ckpt, "rb") as f:
        state = pickle.load(f)
    if path_overrides:
        for k, v in path_overrides.items():
            if k in state:
                state[k] = v
    rec = cls.__new__(cls)
    rec._set_model_state(state)
    print(f"    ✅ {state['recommender_name']} loaded from {ckpt}")
    if warm_refit:
        rec.fit(None, track_metadata=track_meta)
    return rec


def _suffix(robust_mode: str, robust_alpha: float) -> str:
    if robust_mode == "cvar":
        return f"_cvar{int(round(robust_alpha * 100))}"
    return f"_{robust_mode}"


def _resolve_cfg(folder: str, storage_dir: str, suffix: str) -> Path:
    storage_yaml = repo_path(storage_dir) / folder / f"best_params_{folder}{suffix}.yaml"
    legacy_yaml = _PKG_ROOT / "configs" / f"cv_best_{folder}{suffix}.yaml"
    if storage_yaml.exists():
        return storage_yaml
    if legacy_yaml.exists():
        return legacy_yaml
    sys.exit(f"[infer-ckpt] missing config for {folder}: tried {storage_yaml} and {legacy_yaml}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inference-only replay from saved DRO checkpoints.")
    p.add_argument("--model",    required=True)
    p.add_argument("--urm_mode", default="session", choices=["session", "user"])
    p.add_argument("--robust_mode", default="cvar", choices=["mean", "cvar", "group_dro"])
    p.add_argument("--robust_alpha", type=float, default=0.7)
    p.add_argument("--top_k", type=int, default=200)
    p.add_argument("--storage_dir", default="models/CG_crossvalidation")
    p.add_argument("--splitk_dir",  default="data/splitK")
    p.add_argument("--n_folds",     type=int, default=5)
    p.add_argument("--splits", nargs="+", default=["folds", "holdout", "full"],
                   choices=["folds", "holdout", "full"],
                   help="Which checkpoint groups to replay.")
    p.add_argument("--blind_path", default=(
        "data/talkpl-ai/TalkPlayData-Challenge-Blind-A/data/test-00000-of-00001.parquet"
    ))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    suffix = _suffix(args.robust_mode, args.robust_alpha)
    folder = f"{args.model}_{args.urm_mode}_dro"
    cfg_path = _resolve_cfg(folder, args.storage_dir, suffix)

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    class_name     = cfg["class"]
    module_name    = cfg["module"]
    inference_mode = cfg.get("inference_mode", "standard")
    resolved_fixed = resolve_param_paths(cfg.get("fixed_params", {}))
    path_overrides = {k: v for k, v in resolved_fixed.items() if k in _PATH_PARAM_KEYS}

    out_dir    = repo_path(args.storage_dir) / folder
    ckpt_dir   = out_dir / "checkpoints"
    ds_dir     = out_dir / "datasets"
    splitk_dir = repo_path(args.splitk_dir)
    top_k      = args.top_k

    print(f"[infer-ckpt] {folder} ({class_name})  mode={inference_mode}")
    print(f"[infer-ckpt] ckpt_dir={ckpt_dir}")
    print(f"[infer-ckpt] out={ds_dir}")

    track_meta = None if args.model in _MODELS_NO_META else pl.read_parquet(repo_path(_TRACK_META_PATH))
    warm_refit = args.model in _WARM_REFIT_MODELS
    if warm_refit:
        print(f"[infer-ckpt] warm-refit path active for {args.model} "
              "(rebuilds cached catalogue arrays via fit(None, ...), keeps checkpointed tower members)")

    def _qbundle(set_key: str, fold: int | None = None):
        if inference_mode != "text":
            return None
        return load_query_bundle(repo_path(QUERY_TOWER_BASE), set_key, fold)

    if "folds" in args.splits:
        for fold in range(args.n_folds):
            print(f"\n--- fold {fold} ---")

            ckpt_cg = ckpt_dir / f"fold_{fold}_cg_train.pkl"
            if ckpt_cg.exists():
                rec_cg = _load_checkpoint(class_name, module_name, ckpt_cg,
                                          warm_refit=warm_refit, track_meta=track_meta,
                                          path_overrides=path_overrides)
                cg_val_df = load_eval(splitk_dir, fold)
                _infer_multiturn(rec_cg, cg_val_df, top_k, inference_mode, track_meta,
                                 ds_dir / f"fold_{fold}_oof_cg_val.parquet",
                                 f"fold_{fold}_cg_train→cg_val",
                                 query_bundle=_qbundle("cg_val", fold))
            else:
                print(f"  [skip] missing {ckpt_cg}")

            ckpt_cr = ckpt_dir / f"fold_{fold}_cg_train_val.pkl"
            if ckpt_cr.exists():
                rec_cr = _load_checkpoint(class_name, module_name, ckpt_cr,
                                          warm_refit=warm_refit, track_meta=track_meta,
                                          path_overrides=path_overrides)
                reranker_val_df = load_reranker_val(splitk_dir, fold)
                _infer_multiturn(rec_cr, reranker_val_df, top_k, inference_mode, track_meta,
                                 ds_dir / f"fold_{fold}_oof_reranker_val.parquet",
                                 f"fold_{fold}_cg_train_val→reranker_val",
                                 query_bundle=_qbundle("reranker_val", fold))
            else:
                print(f"  [skip] missing {ckpt_cr}")

    if "holdout" in args.splits:
        print("\n--- holdout ---")
        ckpt_nh = ckpt_dir / "non_holdout.pkl"
        if ckpt_nh.exists():
            rec_nh = _load_checkpoint(class_name, module_name, ckpt_nh,
                                      warm_refit=warm_refit, track_meta=track_meta,
                                      path_overrides=path_overrides)
            holdout_df = pl.read_parquet(repo_path(_HOLDOUT_PATH))
            _infer_multiturn(rec_nh, holdout_df, top_k, inference_mode, track_meta,
                             ds_dir / "holdout_candidates.parquet",
                             "non_holdout→holdout",
                             query_bundle=_qbundle("holdout"))
        else:
            print(f"  [skip] missing {ckpt_nh}")

    if "full" in args.splits:
        print("\n--- full (blind-A) ---")
        ckpt_full = ckpt_dir / "full.pkl"
        if ckpt_full.exists():
            rec_full = _load_checkpoint(class_name, module_name, ckpt_full,
                                        warm_refit=warm_refit, track_meta=track_meta,
                                        path_overrides=path_overrides)
            blind_df = pl.read_parquet(repo_path(args.blind_path))
            if hasattr(rec_full, "encode_additional"):
                rec_full.encode_additional(blind_df)
            if inference_mode == "text":
                recs = _predict_sessions_text(
                    rec_full, blind_df, top_k=top_k, remove_seen=True,
                    query_bundle=_qbundle("blind"),
                )
            else:
                recs = _predict_sessions(rec_full, blind_df, top_k=top_k, remove_seen=True)
            bc_path = ds_dir / "blind_candidates.parquet"
            bc_path.parent.mkdir(parents=True, exist_ok=True)
            recs.with_columns(
                pl.lit(None, dtype=pl.Utf8).alias("gt_track_id")
            ).write_parquet(bc_path)
            print(f"  [blind candidates] {recs.shape[0]} sessions → {bc_path}")
        else:
            print(f"  [skip] missing {ckpt_full}")

    print(f"\n[infer-ckpt] done — {ds_dir}")


if __name__ == "__main__":
    main()
