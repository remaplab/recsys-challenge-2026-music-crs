"""Track-embedding I/O helpers.

Added to fix an import (`from ._embedding_io import ...` in `two_tower.py`)
that referenced this module but the file was missing in `origin/main`. The
functions below cover exactly the two names the existing two_tower needs:

  - `load_track_embeddings(source, embedding_cols)`
    Load per-track embedding vectors from one or more parquet files
    (or a pre-loaded DataFrame) and horizontally concatenate the requested
    list-typed columns. Returns `(track_ids, emb)` where `emb` is
    `np.ndarray` of shape `(n_tracks, sum(col_dims))`.

  - `l2_normalize_rows(arr)`
    Row-wise L2 normalisation with safe handling of zero rows.

These mirror the semantics implied by the call-sites in two_tower.py at fit
time and at ICM-projection time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl


def load_track_embeddings(
    source: Any,
    embedding_cols: list[str] | None = None,
) -> tuple[list[str], np.ndarray]:
    """Load track embeddings from parquet path(s) or a Polars DataFrame.

    Parameters
    ----------
    source
        One of: str/Path (single parquet), list of str/Path (multiple
        parquets concatenated), or a pl.DataFrame already in memory.
    embedding_cols
        Names of list-typed columns to concatenate horizontally. Defaults to
        ``["metadata-qwen3_embedding_0.6b"]`` if not provided.

    Returns
    -------
    track_ids : list[str]
        Track IDs in the same order as the rows of `emb`.
    emb : np.ndarray
        Float32 array of shape (n_tracks, sum(col_dims)).
    """
    if isinstance(source, pl.DataFrame):
        df = source
    elif isinstance(source, (str, Path)):
        df = pl.read_parquet(str(source))
    elif isinstance(source, (list, tuple)):
        if not source:
            raise ValueError("load_track_embeddings: empty list of paths")
        df = pl.concat([pl.read_parquet(str(p)) for p in source]).unique(subset=["track_id"])
    else:
        raise TypeError(
            f"track_embeddings must be a path, list of paths, or pl.DataFrame; "
            f"got {type(source).__name__}"
        )

    cols = list(embedding_cols) if embedding_cols else ["metadata-qwen3_embedding_0.6b"]
    chunks: list[np.ndarray] = []
    for c in cols:
        if c not in df.columns:
            raise KeyError(f"load_track_embeddings: column {c!r} not in DataFrame")
        # list-typed column → (n_tracks, dim) array. None rows become zeros.
        raw = df[c].to_list()
        first_non_null = next((v for v in raw if v is not None), None)
        if first_non_null is None:
            raise ValueError(f"load_track_embeddings: column {c!r} has no non-null values")
        dim = len(first_non_null)
        arr = np.zeros((len(raw), dim), dtype=np.float32)
        for i, v in enumerate(raw):
            if v is not None:
                arr[i] = np.asarray(v, dtype=np.float32)
        chunks.append(arr)

    emb = np.concatenate(chunks, axis=1) if len(chunks) > 1 else chunks[0]
    track_ids = df["track_id"].to_list()
    return track_ids, emb


def l2_normalize_rows(arr: np.ndarray) -> np.ndarray:
    """Return a row-L2-normalised copy of `arr`.

    Zero-norm rows are returned as zero (no division by zero). Operates on
    float arrays; integer input is cast to float32.
    """
    if not np.issubdtype(arr.dtype, np.floating):
        arr = arr.astype(np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    safe = np.where(norms > 0.0, norms, 1.0)
    return (arr / safe).astype(arr.dtype, copy=False)
