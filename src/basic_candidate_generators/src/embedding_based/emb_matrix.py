"""W_emb builder + IO + adapters for embedding-based CGs.

Frozen Qwen3 track-tower embeddings are L2-normed, so dot product = cosine.
Item-item top-k similarity is computed by tiled GEMM (GPU if available, else
numpy BLAS); the N x N matrix is never materialised (only N x block at a time).

The similarity matrix is cached in STABLE tower-id space (catalogue-global,
fold-independent) and remapped to a recommender's per-fold `id_map` at fit time
via `remap_sim_to_idmap` — so one `.npz` serves every fold.
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix


# ---------------------------------------------------------------------------
# Track-tower loading
# ---------------------------------------------------------------------------

def load_track_tower(cache_dir: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load (track_ids, embeddings) from a `dense_tracks_*` tower cache dir.

    Returns
    -------
    track_ids : np.ndarray[object]  shape (n,)
    emb       : np.ndarray[float32] shape (n, d), L2-normed (asserted)
    """
    cache_dir = Path(cache_dir)
    track_ids = np.load(cache_dir / "track_ids.npy", allow_pickle=True)
    emb = np.asarray(np.load(cache_dir / "embeddings.npy"), dtype=np.float32)
    if emb.shape[0] != track_ids.shape[0]:
        raise ValueError(
            f"track tower row mismatch in {cache_dir}: "
            f"ids={track_ids.shape[0]} vs emb={emb.shape[0]}"
        )
    # caches are L2-normed; assert on a sample so dot == cosine downstream
    norms = np.linalg.norm(emb[: min(256, emb.shape[0])], axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        raise ValueError(
            f"track tower {cache_dir} is not L2-normed (sample norm "
            f"range [{norms.min():.4f}, {norms.max():.4f}])"
        )
    return track_ids, emb


def load_modality_tower(parquet_glob: str | Path,
                        column: str) -> tuple[np.ndarray, np.ndarray]:
    """Load (track_ids, embeddings) from any list-of-float `column` across the
    track-embedding parquet shards matching `parquet_glob`.

    Drops rows with a missing/empty vector and L2-normalises so dot == cosine
    downstream (raw vectors are NOT unit-norm). Same (object ids, float32
    normed) contract as `load_track_tower`. `column` is one of the parquet's
    embedding columns, e.g. `image-siglip2`, `cf-bpr`, `audio-laion_clap`.
    """
    import pyarrow.parquet as pq

    paths = sorted(glob.glob(str(parquet_glob)))
    if not paths:
        raise FileNotFoundError(f"no parquet shards match {parquet_glob!r}")

    ids_l: list[np.ndarray] = []
    emb_l: list[np.ndarray] = []
    for p in paths:
        df = pq.read_table(p, columns=["track_id", column]).to_pandas()
        keep = df[column].map(lambda v: v is not None and len(v) > 0)
        df = df[keep]
        ids_l.append(df["track_id"].to_numpy(dtype=object))
        emb_l.append(np.stack([np.asarray(v, dtype=np.float32)
                               for v in df[column].to_numpy()]))

    track_ids = np.concatenate(ids_l)
    emb = np.concatenate(emb_l, axis=0)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    emb /= norms
    return track_ids, emb


def load_image_tower(parquet_glob: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Back-compat wrapper: SigLIP2 image tower = `image-siglip2` column."""
    return load_modality_tower(parquet_glob, "image-siglip2")


# ---------------------------------------------------------------------------
# Item-item similarity (tiled GEMM, top-k per row)
# ---------------------------------------------------------------------------

def build_emb_item_sim(
    emb: np.ndarray, k: int = 150, block: int = 4096, use_gpu: bool = True,
) -> csr_matrix:
    """Top-k cosine item-item similarity over L2-normed `emb` (n, d).

    Row i holds the top-k most similar items to i (self excluded), positive
    sims only. Shape (n, n) CSR — drops directly into `profile @ W` scoring.
    """
    n = emb.shape[0]
    k = min(k, n - 1)
    if k <= 0:
        return csr_matrix((n, n), dtype=np.float32)

    rows_l: list[np.ndarray] = []
    cols_l: list[np.ndarray] = []
    vals_l: list[np.ndarray] = []

    torch = _maybe_torch(use_gpu)
    if torch is not None:
        dev = "cuda"
        E = torch.from_numpy(np.ascontiguousarray(emb)).to(dev)
        for a in range(0, n, block):
            b = min(a + block, n)
            S = E[a:b] @ E.T                                   # (bb, n)
            local = torch.arange(b - a, device=dev)
            S[local, torch.arange(a, b, device=dev)] = -1.0    # mask self
            topv, topi = torch.topk(S, k, dim=1)
            topv = topv.cpu().numpy()
            topi = topi.cpu().numpy().astype(np.int32)
            _collect_topk(a, b, topv, topi, rows_l, cols_l, vals_l)
        del E
        torch.cuda.empty_cache()
    else:
        ET = np.ascontiguousarray(emb.T)                        # (d, n)
        for a in range(0, n, block):
            b = min(a + block, n)
            S = emb[a:b] @ ET                                   # (bb, n)
            S[np.arange(b - a), np.arange(a, b)] = -1.0
            topi = np.argpartition(-S, k, axis=1)[:, :k].astype(np.int32)
            topv = np.take_along_axis(S, topi, axis=1)
            _collect_topk(a, b, topv, topi, rows_l, cols_l, vals_l)

    if not rows_l:
        return csr_matrix((n, n), dtype=np.float32)
    rows = np.concatenate(rows_l)
    cols = np.concatenate(cols_l)
    vals = np.concatenate(vals_l)
    return csr_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.float32)


def sparsify_topk(W: csr_matrix, k: int) -> csr_matrix:
    """Keep only the top-`k` entries by value in each row of `W`.

    Exact downward trim of an item-item similarity matrix: a row's top-`k`
    neighbours are a subset of its top-`K` for any K >= k, so a matrix cached at
    high K can be sparsified to any smaller k at load with no approximation. Rows
    with <= k entries pass through untouched; a no-op fast path returns `W` as-is
    when no row exceeds `k`.
    """
    W = W.tocsr()
    if k <= 0:
        return W
    counts = np.diff(W.indptr)
    if counts.max(initial=0) <= k:
        return W

    indptr, data, indices = W.indptr, W.data, W.indices
    new_data, new_ind, new_ptr = [], [], np.empty(W.shape[0] + 1, dtype=np.int64)
    new_ptr[0] = 0
    for r in range(W.shape[0]):
        s, e = indptr[r], indptr[r + 1]
        if e - s <= k:
            new_data.append(data[s:e]); new_ind.append(indices[s:e])
            new_ptr[r + 1] = new_ptr[r] + (e - s)
        else:
            d = data[s:e]
            top = np.argpartition(-d, k)[:k]
            new_data.append(d[top]); new_ind.append(indices[s:e][top])
            new_ptr[r + 1] = new_ptr[r] + k

    return csr_matrix(
        (np.concatenate(new_data), np.concatenate(new_ind), new_ptr),
        shape=W.shape, dtype=W.dtype,
    )


def _collect_topk(a, b, topv, topi, rows_l, cols_l, vals_l) -> None:
    """Append positive (row, col, val) triples from a row-block's top-k."""
    pos = topv > 0.0
    if not pos.any():
        return
    bb = b - a
    row_ids = (np.arange(a, b)[:, None] + np.zeros((1, topv.shape[1]), dtype=np.int64))
    rows_l.append(row_ids[pos].astype(np.int64))
    cols_l.append(topi[pos].astype(np.int64))
    vals_l.append(topv[pos].astype(np.float32))


def _maybe_torch(use_gpu: bool):
    if not use_gpu:
        return None
    try:
        import torch
    except ImportError:
        return None
    return torch if torch.cuda.is_available() else None


# ---------------------------------------------------------------------------
# Disk cache
# ---------------------------------------------------------------------------

def save_emb_sim(W: csr_matrix, track_ids: np.ndarray, path: str | Path) -> None:
    """Persist (similarity matrix, tower track-id order) to a single `.npz`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    W = W.tocsr()
    np.savez(
        path,
        data=W.data, indices=W.indices, indptr=W.indptr, shape=np.array(W.shape),
        track_ids=np.asarray(track_ids, dtype=object),
    )


def load_emb_sim(path: str | Path) -> tuple[csr_matrix, np.ndarray]:
    """Load (similarity matrix in tower-id space, tower track-id order)."""
    path = Path(path)
    if not path.exists() and path.suffix != ".npz":
        path = path.with_suffix(".npz")
    z = np.load(path, allow_pickle=True)
    W = csr_matrix(
        (z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]),
    )
    return W, z["track_ids"]


# ---------------------------------------------------------------------------
# Adapters: tower-space -> recommender id_map space
# ---------------------------------------------------------------------------

def remap_sim_to_idmap(W_tower: csr_matrix, tower_ids: np.ndarray, id_map) -> csr_matrix:
    """Reindex a tower-space similarity matrix into a recommender's id_map.

    Tower rows/cols whose track_id is absent from id_map are dropped; id_map
    tracks absent from the tower get no neighbours (empty rows). Fold-safe:
    the same cached `W_tower` maps to any fold's id_map.
    """
    t2i = id_map.track_to_idx
    n_idmap = id_map.n_tracks
    new = np.fromiter((t2i.get(t, -1) for t in tower_ids), dtype=np.int64,
                      count=len(tower_ids))
    coo = W_tower.tocoo()
    r = new[coo.row]
    c = new[coo.col]
    m = (r >= 0) & (c >= 0)
    return csr_matrix((coo.data[m], (r[m], c[m])), shape=(n_idmap, n_idmap),
                      dtype=np.float32)


def align_emb_to_idmap(tower_ids: np.ndarray, emb: np.ndarray, id_map) -> np.ndarray:
    """Reorder tower embedding rows into id_map track-index order.

    Returns (id_map.n_tracks, d); tracks missing from the tower are zero rows.
    Used when an aligned dense ICM matrix is needed (CBF blend).
    """
    d = emb.shape[1]
    out = np.zeros((id_map.n_tracks, d), dtype=np.float32)
    t2i = id_map.track_to_idx
    for row, tid in enumerate(tower_ids):
        j = t2i.get(tid, -1)
        if j >= 0:
            out[j] = emb[row]
    return out


def build_emb_cbf_similarity(emb_aligned: np.ndarray, k: int, use_gpu: bool = True) -> csr_matrix:
    """CBF item-item similarity from an id_map-aligned embedding ICM.

    Signature-compatible with `interactions.build_item_cbf_similarity_fast` so it
    can substitute the tag-based ICM in the `icm_weight` blend. Zero rows (tracks
    missing from the tower) yield empty neighbour rows.
    """
    return build_emb_item_sim(emb_aligned, k=k, use_gpu=use_gpu)
