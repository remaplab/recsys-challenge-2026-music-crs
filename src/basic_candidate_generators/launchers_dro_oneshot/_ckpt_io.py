"""Gzip-compressed checkpoint I/O for one-shot CGs.

Deliberately separate from `BaseRecommender.save()`/`.load()` (plain pickle,
shared by the DRO checkpoint pipeline) so compressing one-shot checkpoints
can't affect any already-produced DRO `.pkl` file or its loader. Same state
dict contract (`_get_model_state()` / `_set_model_state()`), just gzip'd.
"""
from __future__ import annotations

import gzip
import importlib
import pickle
from pathlib import Path

import polars as pl

# DenseQueryCG (dense_text_8b/4b/0p6b) deliberately only pickles params/paths,
# not track_emb/query_emb (frozen, cheap to reload from track_emb_dir/
# query_cache_root — see its _get_model_state docstring). fit(None, ...)
# rebuilds those arrays without touching train_df (unused by its fit()).
# Same warm-refit idea as DRO's hybrid_all_qwen/tower_ensemble/tower_cf_ensemble.
WARM_REFIT_MODELS = {"dense_text_8b", "dense_text_4b", "dense_text_0p6b"}


def save_ckpt(rec, path: Path, compresslevel: int = 6) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {"recommender_name": rec.RECOMMENDER_NAME}
    state.update(rec._get_model_state())
    with gzip.open(path, "wb", compresslevel=compresslevel) as f:
        pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"    [ckpt] saved {path}  ({path.stat().st_size / 1e6:.1f} MB)")


def load_ckpt(class_name: str, module_name: str, path: Path,
              *, warm_refit: bool = False, track_meta: pl.DataFrame | None = None):
    cls = getattr(importlib.import_module(module_name), class_name)
    with gzip.open(path, "rb") as f:
        state = pickle.load(f)
    rec = cls.__new__(cls)
    rec._set_model_state(state)
    print(f"    [ckpt] loaded {state['recommender_name']} from {path}")
    if warm_refit:
        rec.fit(None, track_metadata=track_meta)
        print("    [ckpt] warm-refit done (rebuilt cached arrays, no retrain)")
    return rec
