"""Shared streaming utilities for the reranker backends.

Two main pieces:

1. :func:`materialize_subsamples` — pre-write a deterministic on-disk
   subsample of the input parquets so that any iterator (XGBoost DataIter,
   LightGBM file reader, NN IterableDataset) sees IDENTICAL rows across
   multiple passes. Without this, repeated passes would re-shuffle and
   produce inconsistent quantile bins (XGBoost) or epoch-to-epoch label
   mismatches (NN).

2. :func:`load_parquet_grouped` — helper for backends that need the per-group
   ``len`` vector (XGBoost / LightGBM / NN).

3. :func:`build_session_hash_split` — session-coherent 75/25 split for
   retrain (and 4× bagging variants).

These utilities are pure-Python; they don't import torch / xgboost / lgbm
so they can be reused even in CPU-only smoke tests.
"""
from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl


# ---------------------------------------------------------------------------
# Deterministic subsampled parquets
# ---------------------------------------------------------------------------

def materialize_subsamples(
    paths: list[Path],
    out_dir: Path,
    max_groups: Optional[int],
    seed: int = 42,
) -> list[Path]:
    """Write subsampled copies of each input parquet under ``out_dir``.

    Parameters
    ----------
    paths
        Source parquets (each holds one fold).
    out_dir
        Destination directory. Files reuse the source filename; existing
        outputs are skipped (idempotent).
    max_groups
        If non-None, cap each output to at most ``max_groups`` distinct
        ``(session_id, turn_number)`` groups (sampled by hash of session_id).
        Use ``None`` for "full data".
    seed
        Sampling seed (used only when ``max_groups`` is enforced).

    Returns
    -------
    list[Path]
        The output paths in the same order as ``paths``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-output manifest: ``(max_groups, seed, source_path, source_mtime)``.
    # If any of these change, the existing subsample is INVALIDATED and
    # rewritten. Avoids the silent-staleness footgun where dropping
    # ``max_groups`` in the YAML had no effect because chunks were already
    # on disk from a previous run.
    def _manifest_for(src: Path) -> dict:
        return {
            "max_groups": max_groups,
            "seed": seed,
            "source": str(src),
            "source_mtime": src.stat().st_mtime,
        }

    out_paths: list[Path] = []
    for p in paths:
        out_p = out_dir / p.name
        manifest_p = out_dir / (p.name + ".manifest.json")
        current = _manifest_for(p)
        if out_p.exists() and manifest_p.exists():
            try:
                saved = json.loads(manifest_p.read_text())
            except Exception:                                          # noqa: BLE001
                saved = None
            if saved == current:
                print(f"  [pre] skip {p.name} (exists, manifest matches)")
                out_paths.append(out_p)
                continue
            print(f"  [pre] {p.name}: manifest mismatch → rewriting")
            out_p.unlink(missing_ok=True)
        elif out_p.exists():
            print(f"  [pre] {p.name}: missing manifest → rewriting")
            out_p.unlink(missing_ok=True)
        df = pl.read_parquet(p)
        if max_groups is not None:
            uniq = df.select("session_id", "turn_number").unique()
            if uniq.height > max_groups:
                sampled = (
                    uniq.with_columns(pl.col("session_id").hash(seed=seed).alias("_h"))
                        .sort("_h")
                        .head(max_groups)
                        .drop("_h")
                )
                df = df.join(sampled, on=["session_id", "turn_number"], how="inner")
        df = df.sort("session_id", "turn_number", "track_id")
        df.write_parquet(out_p)
        # Persist the manifest AFTER the parquet write so an interrupted
        # run leaves a stale-manifest-less parquet that next call will detect.
        manifest_p.write_text(json.dumps(current, indent=2))
        n_g = df.select("session_id", "turn_number").unique().height
        print(f"  [pre] {p.name}: {df.height:,} rows × {n_g:,} groups → {out_p.name}")
        del df
        gc.collect()
        out_paths.append(out_p)
    return out_paths


# ---------------------------------------------------------------------------
# Generic per-group helpers used by every backend
# ---------------------------------------------------------------------------

def load_parquet_grouped(
    path: Path,
    feat_cols: list[str],
    *,
    label_col: str = "label",
) -> tuple[np.ndarray, Optional[np.ndarray], list[int], pl.DataFrame]:
    """Read one parquet and return (X, y, group_sizes, meta).

    Parameters
    ----------
    path
        Parquet path.
    feat_cols
        Feature columns in the order the backend expects.
    label_col
        Label column name. ``None`` returned for ``y`` if missing.

    Returns
    -------
    X : ``np.ndarray`` (float32, shape ``[n_rows, n_features]``)
    y : ``np.ndarray`` (int8, shape ``[n_rows]``) or ``None``
    group_sizes : ``list[int]`` — per-group row counts in the same order as
        rows. Sums to ``n_rows``.
    meta : ``polars.DataFrame`` with ``(session_id, turn_number, track_id)``
        kept side-by-side with the rows (same order). Useful for predict-time
        joins back to the candidate identity.
    """
    df = pl.read_parquet(path)
    df = df.sort("session_id", "turn_number", "track_id")
    groups = (
        df.group_by(["session_id", "turn_number"], maintain_order=True)
        .len()["len"]
        .to_list()
    )
    X = df.select(feat_cols).cast(pl.Float32).fill_null(float("nan")).to_numpy()
    y = df[label_col].to_numpy() if label_col in df.columns else None
    meta = df.select("session_id", "turn_number", "track_id")
    return X, y, groups, meta


# ---------------------------------------------------------------------------
# Session-hash 75/25 split for retrain (and bagging)
# ---------------------------------------------------------------------------

def _stable_hash(s: str, seed: int) -> int:
    """Deterministic, language-independent hash so re-runs are reproducible
    across Python sessions (Python's built-in ``hash`` is salt-randomised).
    """
    h = hashlib.sha1(f"{seed}:{s}".encode("utf-8")).digest()
    # Use the first 8 bytes as a big-endian unsigned integer.
    return int.from_bytes(h[:8], byteorder="big", signed=False)


def spill_per_model_splits(
    chunk_paths: list[Path],
    n_models: int,
    *,
    base_seed: int,
    train_frac: float,
    out_dir: Path,
    tag_prefix: str,
) -> list[tuple[list[Path], list[Path], int]]:
    """Stream session-hash split N chunk parquets into per-(model, chunk) outputs.

    Why not pre-concat?
    -------------------
    Concatenating every chunk in RAM (e.g. 15 reranker_val chunks ≈ 22 GB at
    our scale) is the OOM trigger. The session-hash split is deterministic
    per ``session_id``, so the split decision for a session is the SAME
    regardless of which chunk it currently lives in. We can therefore split
    each chunk independently and end up with per-(model, chunk) parquets
    that, taken together, reproduce the exact 75/25 partition we would have
    gotten from the global concat.

    Peak RAM
    --------
    Bounded by ``size_of_one_chunk × ~3`` (input DF + train slice + ES
    slice). For a 600 MB chunk that's ~1.8 GB — fits 32 GB host easily.

    Returns
    -------
    ``list[(train_paths, es_paths, seed)]`` of length ``n_models``. Each
    entry's path lists are ready to plug into :class:`DatasetSpec`.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    per_model_train: list[list[Path]] = [[] for _ in range(n_models)]
    per_model_es:    list[list[Path]] = [[] for _ in range(n_models)]
    seeds = [base_seed + k for k in range(n_models)]

    for ci, src in enumerate(chunk_paths):
        df = pl.read_parquet(src)
        for k, seed_k in enumerate(seeds):
            tr, es = build_session_hash_split(df, train_frac=train_frac, seed=seed_k)
            if tr.height > 0:
                tr_p = out_dir / f"{tag_prefix}_m{k}_train_c{ci:03d}.parquet"
                tr.write_parquet(tr_p)
                per_model_train[k].append(tr_p)
            if es.height > 0:
                es_p = out_dir / f"{tag_prefix}_m{k}_es_c{ci:03d}.parquet"
                es.write_parquet(es_p)
                per_model_es[k].append(es_p)
            del tr, es
        del df
        gc.collect()
        print(f"  [spill] {src.name} → {n_models} (train, es) splits")

    return [(per_model_train[k], per_model_es[k], seeds[k]) for k in range(n_models)]


def build_session_hash_split(
    df: pl.DataFrame,
    train_frac: float = 0.75,
    seed: int = 42,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Partition rows of ``df`` into (train, early-stop) by session hash.

    All rows of a given ``session_id`` land in the same partition.

    Parameters
    ----------
    df
        Source DataFrame (must contain ``session_id``).
    train_frac
        Fraction of sessions assigned to train. The rest go to early-stop.
    seed
        Hash seed; change to obtain a different bagging fold.

    Returns
    -------
    (train_df, es_df)
    """
    sessions = df["session_id"].unique().to_list()
    bucket = np.array(
        [_stable_hash(s, seed) % 10_000 for s in sessions],
        dtype=np.int64,
    )
    threshold = int(train_frac * 10_000)
    train_sessions = {s for s, b in zip(sessions, bucket) if b < threshold}

    train_df = df.filter(pl.col("session_id").is_in(list(train_sessions)))
    es_df = df.filter(~pl.col("session_id").is_in(list(train_sessions)))
    return train_df, es_df
