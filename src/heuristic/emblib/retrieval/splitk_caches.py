"""emblib/retrieval/splitk_caches.py

Loaders for the splitK query-embedding caches written by
scripts/12_encode_gambling_caches.py --stages splitk.

DESIGN
======
Script 12 encodes ONE cache PER BUCKET (the 11 atomic units of the splitK
assignment: `holdout`, `fold_{k}_cg_val`, `fold_{k}_reranker_val` for k=0..4),
NOT per fold-part. Every session is therefore encoded exactly once. A fold's
parts are reconstructed here, at load time, by concatenating bucket caches:

    cg_val(k)       = [fold_k_cg_val]
    reranker_val(k) = [fold_k_reranker_val]
    cg_train(k)     = union of fold_j_cg_val + fold_j_reranker_val for all j != k
    holdout         = [holdout]

Row alignment: each bucket cache holds (query_embeddings.npy, query_meta.parquet)
written by the same pass, row-aligned. Concatenation preserves that alignment,
so emb[i] always corresponds to meta row i (session_id, user_id, turn_number,
gt_track_id, prior_track_ids, query_text).

USAGE
=====
    from pathlib import Path
    from emblib.retrieval.splitk_caches import load_fold_cache, load_bucket_cache

    OUT_ROOT = Path("models/retrieval_text_towers")          # package-relative
    emb, meta = load_fold_cache(OUT_ROOT, "qwen3_0p6b", fold=0, part="cg_train")
    emb, meta = load_fold_cache(OUT_ROOT, "qwen3_0p6b", fold=0, part="cg_val")
    emb, meta = load_bucket_cache(OUT_ROOT, "qwen3_0p6b", "holdout")

`model` accepts either the script-12 key ("qwen3_0p6b") or the HF folder name
("Qwen__Qwen3-Embedding-0.6B") or the gambling_updated flag spelling ("0.6").
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

N_FOLDS = 5
SPLITK_SUBDIR_FMT = "dense_splitk_{bucket}_query_len512_poollast"

_MODEL_FOLDER = {
    "qwen3_0p6b": "Qwen__Qwen3-Embedding-0.6B",
    "qwen3_4b":   "Qwen__Qwen3-Embedding-4B",
    "qwen3_8b":   "Qwen__Qwen3-Embedding-8B",
    "0.6":        "Qwen__Qwen3-Embedding-0.6B",
    "4":          "Qwen__Qwen3-Embedding-4B",
    "8":          "Qwen__Qwen3-Embedding-8B",
}

PARTS = ("cg_train", "cg_val", "reranker_val", "holdout")


def _model_folder(model: str) -> str:
    if model in _MODEL_FOLDER:
        return _MODEL_FOLDER[model]
    if model.startswith("Qwen__"):
        return model
    raise ValueError(
        f"unknown model {model!r}; expected one of {sorted(_MODEL_FOLDER)} "
        f"or an HF folder name like 'Qwen__Qwen3-Embedding-0.6B'"
    )


def all_buckets(n_folds: int = N_FOLDS) -> list[str]:
    out = ["holdout"]
    for k in range(n_folds):
        out += [f"fold_{k}_cg_val", f"fold_{k}_reranker_val"]
    return out


def fold_part_buckets(fold: int, part: str, n_folds: int = N_FOLDS) -> list[str]:
    """Atomic bucket names whose union forms (fold, part)."""
    if part not in PARTS:
        raise ValueError(f"unknown part {part!r}; choose from {PARTS}")
    if part == "holdout":
        return ["holdout"]
    if not (0 <= fold < n_folds):
        raise ValueError(f"fold must be in [0, {n_folds}), got {fold}")
    if part == "cg_val":
        return [f"fold_{fold}_cg_val"]
    if part == "reranker_val":
        return [f"fold_{fold}_reranker_val"]
    # cg_train = every other fold's cg_val + reranker_val (mirrors
    # _write_per_fold_parquets in scripts/splitK_crossvalidation.py)
    return [
        f"fold_{j}_{s}"
        for j in range(n_folds) if j != fold
        for s in ("cg_val", "reranker_val")
    ]


def bucket_cache_dir(out_root: Path, model: str, bucket: str) -> Path:
    return Path(out_root) / _model_folder(model) / SPLITK_SUBDIR_FMT.format(bucket=bucket)


def load_bucket_cache(
    out_root: Path, model: str, bucket: str, *, mmap: bool = False,
) -> tuple[np.ndarray, pl.DataFrame]:
    """(query_embeddings, query_meta) for one atomic bucket, row-aligned."""
    d = bucket_cache_dir(out_root, model, bucket)
    emb_path = d / "query_embeddings.npy"
    meta_path = d / "query_meta.parquet"
    if not emb_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"missing splitK cache for bucket {bucket!r} in {d}. Run:\n"
            f"  uv run python scripts/12_encode_gambling_caches.py "
            f"--stages splitk --models {model} --splitk-buckets {bucket}"
        )
    emb = np.load(emb_path, mmap_mode="r" if mmap else None)
    if not mmap:
        emb = np.asarray(emb, dtype=np.float32)
    meta = pl.read_parquet(meta_path)
    if emb.shape[0] != meta.height:
        raise ValueError(
            f"row mismatch in {d}: embeddings={emb.shape[0]} vs meta={meta.height}. "
            f"Cache is corrupt/partial — re-encode with --splitk-overwrite."
        )
    return emb, meta


def load_fold_cache(
    out_root: Path, model: str, fold: int, part: str,
    *, n_folds: int = N_FOLDS,
) -> tuple[np.ndarray, pl.DataFrame]:
    """Concatenated (embeddings, meta) for one fold part. Embeddings are fp32,
    meta gains a `bucket` column recording each row's source bucket."""
    embs: list[np.ndarray] = []
    metas: list[pl.DataFrame] = []
    for bucket in fold_part_buckets(fold, part, n_folds=n_folds):
        e, m = load_bucket_cache(out_root, model, bucket)
        embs.append(e)
        metas.append(m.with_columns(pl.lit(bucket).alias("bucket")))
    emb = np.concatenate(embs, axis=0) if len(embs) > 1 else embs[0]
    meta = pl.concat(metas, how="vertical") if len(metas) > 1 else metas[0]
    # Buckets are session-disjoint by construction; assert it anyway.
    sid_turn = list(zip(meta["session_id"].to_list(), meta["turn_number"].to_list()))
    if len(set(sid_turn)) != len(sid_turn):
        raise ValueError(
            f"duplicate (session_id, turn_number) rows assembling fold={fold} part={part!r} "
            f"— the bucket caches overlap; rebuild the splitK assignment / caches."
        )
    return emb, meta