"""Extract interactions from talkpl-ai train DataFrame and build URMs.

Conversations are stored as List[Struct{content, role, turn_number}]. The
'music' role entries are 36-char track UUIDs and represent the actual track
played at that turn. We treat (session_id, track_id) pairs as interactions.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, datetime
from typing import Iterable

import numpy as np
import polars as pl
from scipy.sparse import csr_matrix, hstack as sp_hstack
from sklearn.preprocessing import normalize as sk_normalize


def explode_music_turns(df: pl.DataFrame) -> pl.DataFrame:
    """Return long-format DataFrame with one row per music turn.

    Accepts two input schemas:

    Raw TalkPlay format — has a ``conversations`` column of type
    ``List[Struct{content, role, turn_number}]``.  Only rows where
    ``role == "music"`` are kept; ``content`` is the 36-char track UUID.

    Pre-exploded format — already has individual columns
    ``[session_id, user_id, session_date, turn_number, track_id]``.
    Any subset of those columns is accepted; missing ones are silently dropped.

    Output columns: session_id, user_id, session_date (pl.Date), turn_number, track_id.
    """
    # Check if the dataframe is already in the 5-column long format
    if "conversations" not in df.columns:
        required = ["session_id", "user_id", "session_date", "turn_number", "track_id"]
        # Select available columns
        out = df.select([c for c in required if c in df.columns])

        # Ensure session_date is a Polars Date type for filtering logic
        if "session_date" in out.columns and out["session_date"].dtype != pl.Date:
            out = out.with_columns(
                pl.col("session_date").cast(pl.Utf8).str.strptime(pl.Date, format="%Y-%m-%d", strict=False)
            )
        return out

    # Original logic for raw TalkPlay format
    cols = ["session_id", "user_id", "session_date", "conversations"]
    df = df.select([c for c in cols if c in df.columns])

    df = df.with_columns(
        pl.col("session_date").cast(pl.Utf8).str.strptime(pl.Date, format="%Y-%m-%d", strict=False)
    )

    long = (
        df.explode("conversations")
        .unnest("conversations")
        .filter(pl.col("role") == "music")
        .rename({"content": "track_id"})
        .select(["session_id", "user_id", "session_date", "turn_number", "track_id"])
    )
    return long


@dataclass
class IdMap:
    """Bidirectional mapping between string IDs and contiguous integer indices.

    mode="user"    — URM rows are users (one row aggregates all their sessions)
    mode="session" — URM rows are sessions (each session is an independent row)
    """

    track_to_idx: dict[str, int]
    idx_to_track: list[str]
    user_to_idx: dict[str, int]   # maps user_id (mode="user") or session_id (mode="session")
    idx_to_user: list[str]
    mode: str = "user"            # "user" | "session"

    @property
    def n_tracks(self) -> int:
        return len(self.idx_to_track)

    @property
    def n_users(self) -> int:
        return len(self.idx_to_user)


def build_id_map(
    interactions: pl.DataFrame,
    extra_track_ids: Iterable[str] | None = None,
    mode: str = "user",
) -> IdMap:
    """Map track_id and row-entity to contiguous integer indices.

    mode="user"    — row entity is user_id (one row per user, sessions merged)
    mode="session" — row entity is session_id (one row per session)

    extra_track_ids: include these even if absent from interactions
        (e.g., the full track catalogue for cold tracks).
    """
    if mode not in ("user", "session"):
        raise ValueError(f"mode must be 'user' or 'session', got {mode!r}")

    track_ids = list(interactions["track_id"].unique().to_list())
    if extra_track_ids:
        track_ids = list({*track_ids, *extra_track_ids})
    track_ids.sort()

    row_col = "user_id" if mode == "user" else "session_id"
    row_ids = sorted(interactions[row_col].unique().to_list())

    return IdMap(
        track_to_idx={t: i for i, t in enumerate(track_ids)},
        idx_to_track=track_ids,
        user_to_idx={u: i for i, u in enumerate(row_ids)},
        idx_to_user=row_ids,
        mode=mode,
    )


def build_urm(interactions: pl.DataFrame, id_map: IdMap) -> csr_matrix:
    """row-entity × tracks binary CSR matrix.

    Row entity = user_id (mode='user') or session_id (mode='session').
    Multiple plays of the same track by the same entity collapse to 1.
    """
    row_col = "user_id" if id_map.mode == "user" else "session_id"
    rows = np.array([id_map.user_to_idx[r] for r in interactions[row_col].to_list()], dtype=np.int32)
    cols = np.array([id_map.track_to_idx[t] for t in interactions["track_id"].to_list()], dtype=np.int32)
    data = np.ones(len(rows), dtype=np.float32)
    urm = csr_matrix((data, (rows, cols)), shape=(id_map.n_users, id_map.n_tracks))
    urm.sum_duplicates()
    urm.data = np.minimum(urm.data, 1.0)
    return urm


def build_user_seen_items(interactions: pl.DataFrame) -> dict[str, frozenset[str]]:
    """Per-user set of track IDs seen in the provided interactions."""
    result: dict[str, frozenset[str]] = {}
    for uid, group in interactions.group_by("user_id"):
        uid_str = uid[0] if isinstance(uid, tuple) else uid
        result[uid_str] = frozenset(group["track_id"].to_list())
    return result


def build_track_release_dates(track_metadata: pl.DataFrame, id_map: IdMap) -> np.ndarray:
    """Per-track-idx release_date as np.datetime64[D]; NaT for tracks without metadata."""
    df = track_metadata.select(["track_id", "release_date"]).unique(subset=["track_id"])
    df = df.with_columns(pl.col("release_date").str.strptime(pl.Date, format="%Y-%m-%d", strict=False))
    # Arrow supports year 0 but Python's datetime.date does not; treat as missing
    df = df.with_columns(
        pl.when(pl.col("release_date").dt.year() > 0)
        .then(pl.col("release_date"))
        .otherwise(pl.lit(None, dtype=pl.Date))
        .alias("release_date")
    )
    lookup = dict(zip(df["track_id"].to_list(), df["release_date"].to_list()))
    out = np.full(id_map.n_tracks, np.datetime64("NaT", "D"))
    for tid, idx in id_map.track_to_idx.items():
        d = lookup.get(tid)
        if d is not None:
            out[idx] = np.datetime64(d, "D")
    return out


def parse_date(d) -> date | None:
    """Coerce string/datetime/date to datetime.date, returning None on failure."""
    if d is None:
        return None
    if isinstance(d, date) and not isinstance(d, datetime):
        return d
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, str):
        try:
            return datetime.strptime(d, "%Y-%m-%d").date()
        except ValueError:
            return None
    return None


def build_icm_blocks(
    track_metadata: pl.DataFrame,
    id_map: IdMap,
    interactions: pl.DataFrame | None = None,
) -> dict[str, csr_matrix]:
    """Same content as `build_icm` but returns each feature GROUP separately.

    Returns a dict {group_name: (n_tracks, group_dim) sparse CSR}. Caller can
    pick / re-combine / per-group-compress as it wishes (e.g. one TruncatedSVD
    per group → one modality per group in the mmh stack).
    Groups: artist, album, tag, decade, popularity, interaction_popularity, duration.
    """
    n = id_map.n_tracks
    tt = id_map.track_to_idx
    out: dict[str, csr_matrix] = {}

    def _list_col_block(col: str) -> csr_matrix:
        fmap: dict[str, int] = {}
        rows_, cols_ = [], []
        for r in track_metadata.iter_rows(named=True):
            tid = r["track_id"]
            if tid not in tt:
                continue
            ti = tt[tid]
            vals = r[col] or []
            if isinstance(vals, str):
                vals = [vals]
            for v in vals:
                if v is None:
                    continue
                if v not in fmap:
                    fmap[v] = len(fmap)
                rows_.append(ti)
                cols_.append(fmap[v])
        if not rows_:
            return csr_matrix((n, 1), dtype=np.float32)
        return csr_matrix(
            (np.ones(len(rows_), np.float32), (rows_, cols_)),
            shape=(n, len(fmap)), dtype=np.float32,
        )

    def _scalar_bin_block(tids: list[str], values: np.ndarray, edges: np.ndarray) -> csr_matrix:
        bins = np.digitize(values, edges)
        n_bins = len(edges) + 1
        rows_, cols_ = [], []
        for i, tid in enumerate(tids):
            if tid not in tt:
                continue
            rows_.append(tt[tid])
            cols_.append(int(bins[i]))
        if not rows_:
            return csr_matrix((n, n_bins), dtype=np.float32)
        return csr_matrix(
            (np.ones(len(rows_), np.float32), (rows_, cols_)),
            shape=(n, n_bins), dtype=np.float32,
        )

    out["artist"] = _list_col_block("artist_id")
    out["album"]  = _list_col_block("album_id")
    out["tag"]    = _list_col_block("tag_list")

    # decade
    dec_rows, dec_cols = [], []
    decade_map: dict[int, int] = {}
    for r in track_metadata.iter_rows(named=True):
        tid = r["track_id"]
        if tid not in tt:
            continue
        rd = r.get("release_date")
        if not rd or not isinstance(rd, str) or len(rd) < 4:
            continue
        try:
            year = int(rd[:4])
        except ValueError:
            continue
        if year <= 0:
            continue
        decade = (year // 10) * 10
        if decade not in decade_map:
            decade_map[decade] = len(decade_map)
        dec_rows.append(tt[tid])
        dec_cols.append(decade_map[decade])
    out["decade"] = csr_matrix(
        (np.ones(len(dec_rows), np.float32), (dec_rows, dec_cols)),
        shape=(n, max(1, len(decade_map))), dtype=np.float32,
    ) if dec_rows else csr_matrix((n, 1), dtype=np.float32)

    # dataset popularity
    pop_df = track_metadata.select(["track_id", "popularity"]).filter(pl.col("popularity").is_not_null())
    if pop_df.height > 0:
        tids = pop_df["track_id"].to_list()
        pops = pop_df["popularity"].to_numpy(allow_copy=True).astype(np.float32)
        nz = pops[pops > 0]
        edges = np.percentile(nz, [20, 40, 60, 80]).astype(np.float32) if len(nz) >= 5 else np.array([0.25, 0.5, 0.75], np.float32)
        out["popularity"] = _scalar_bin_block(tids, pops, edges)
    else:
        out["popularity"] = csr_matrix((n, 1), dtype=np.float32)

    # interaction popularity
    if interactions is not None and interactions.height > 0:
        play_counts = (
            interactions
            .group_by("track_id")
            .agg(pl.col("user_id").n_unique().alias("play_count"))
        )
        tids = play_counts["track_id"].to_list()
        counts = play_counts["play_count"].to_numpy(allow_copy=True).astype(np.float32)
        nz = counts[counts > 0]
        edges = np.percentile(nz, [20, 40, 60, 80]).astype(np.float32) if len(nz) >= 5 else np.array([1, 2, 5, 10], np.float32)
        out["interaction_popularity"] = _scalar_bin_block(tids, counts, edges)
    else:
        out["interaction_popularity"] = csr_matrix((n, 1), dtype=np.float32)

    # duration
    dur_df = track_metadata.select(["track_id", "duration"]).filter(pl.col("duration").is_not_null())
    if dur_df.height > 0:
        tids = dur_df["track_id"].to_list()
        durs = dur_df["duration"].to_numpy(allow_copy=True).astype(np.float64)
        out["duration"] = _scalar_bin_block(tids, durs, np.array([120_000, 180_000, 240_000, 360_000], np.float64))
    else:
        out["duration"] = csr_matrix((n, 1), dtype=np.float32)

    return out


def build_icm(
    track_metadata: pl.DataFrame,
    id_map: IdMap,
    interactions: pl.DataFrame | None = None,
) -> csr_matrix:
    """Build (n_tracks, n_features) binary ICM from track metadata.

    Feature groups:
        artist_id, album_id, tag_list  — identity one-hot (binary, List[String] columns)
        release_decade                 — bucketed one-hot (10-year bins)
        dataset_popularity_bin         — 5 quantile bins on the dataset-provided popularity field
        interaction_popularity_bin     — 5 quantile bins on unique-user play count from interactions
                                         (only added when `interactions` is not None)
        duration_bin                   — 5 fixed bins (<2 min, 2-3, 3-4, 4-6, >6 min)
    """
    n  = id_map.n_tracks
    tt = id_map.track_to_idx
    tid_df = pl.DataFrame({"track_id": list(tt.keys()), "track_idx": list(tt.values())})
    blocks: list[csr_matrix] = []

    def _list_block(col: str) -> csr_matrix:
        df = (
            track_metadata.select(["track_id", col]).explode(col).drop_nulls()
            .join(tid_df, on="track_id")
        )
        if df.is_empty():
            return csr_matrix((n, 1), dtype=np.float32)
        # Deterministic feature columns: sorted-unique index (NOT Categorical
        # physical codes, whose order is discovery-dependent → nondeterministic
        # column layout → unstable downstream W_cbf cache hash).
        uniq = df.select(col).unique().sort(col).with_row_index("fi")
        df = df.join(uniq, on=col)
        r, c = df["track_idx"].to_numpy(), df["fi"].to_numpy().astype(np.int32)
        return csr_matrix((np.ones(len(r), np.float32), (r, c)), shape=(n, uniq.height), dtype=np.float32)

    def _bin_block(col: str, edges: np.ndarray) -> csr_matrix:
        df = track_metadata.select(["track_id", col]).drop_nulls().join(tid_df, on="track_id")
        if df.is_empty():
            return csr_matrix((n, len(edges) + 1), dtype=np.float32)
        vals = df[col].to_numpy(allow_copy=True).astype(np.float64)
        bins = np.digitize(vals, edges)
        r = df["track_idx"].to_numpy()
        return csr_matrix((np.ones(len(r), np.float32), (r, bins)), shape=(n, len(edges) + 1), dtype=np.float32)

    blocks.append(_list_block("artist_id"))
    blocks.append(_list_block("album_id"))
    blocks.append(_list_block("tag_list"))

    # release_decade
    rd = (
        track_metadata.select(["track_id", "release_date"]).drop_nulls()
        .join(tid_df, on="track_id")
        .with_columns(pl.col("release_date").str.slice(0, 4).cast(pl.Int32, strict=False).alias("year"))
        .filter(pl.col("year").is_not_null() & (pl.col("year") > 0))
        .with_columns(((pl.col("year") // 10) * 10).alias("decade"))
    )
    if rd.height > 0:
        uniq = sorted(rd["decade"].unique().to_list())
        d2i  = {d: i for i, d in enumerate(uniq)}
        r    = rd["track_idx"].to_numpy()
        c    = np.array([d2i[d] for d in rd["decade"].to_list()], dtype=np.int32)
        blocks.append(csr_matrix((np.ones(len(r), np.float32), (r, c)), shape=(n, len(uniq)), dtype=np.float32))

    # dataset_popularity_bin
    pop_df = track_metadata.filter(pl.col("popularity").is_not_null())
    if pop_df.height > 0:
        pops = pop_df["popularity"].to_numpy(allow_copy=True).astype(np.float32)
        nz   = pops[pops > 0]
        edges = np.percentile(nz, [20, 40, 60, 80]).astype(np.float32) if len(nz) >= 5 else np.array([0.25, 0.5, 0.75], np.float32)
        blocks.append(_bin_block("popularity", edges))

    # interaction_popularity_bin
    if interactions is not None and interactions.height > 0:
        pc = (
            interactions.group_by("track_id")
            .agg(pl.col("user_id").n_unique().alias("pc"))
            .join(tid_df, on="track_id").drop_nulls()
        )
        if pc.height > 0:
            counts = pc["pc"].to_numpy(allow_copy=True).astype(np.float32)
            nz     = counts[counts > 0]
            edges  = np.percentile(nz, [20, 40, 60, 80]).astype(np.float32) if len(nz) >= 5 else np.array([1, 2, 5, 10], np.float32)
            bins   = np.digitize(counts, edges)
            r      = pc["track_idx"].to_numpy()
            blocks.append(csr_matrix((np.ones(len(r), np.float32), (r, bins)), shape=(n, len(edges) + 1), dtype=np.float32))

    # duration_bin: fixed thresholds (ms): <2min, 2-3, 3-4, 4-6, >6min
    if "duration" in track_metadata.columns:
        blocks.append(_bin_block("duration", np.array([120_000, 180_000, 240_000, 360_000], np.float64)))

    return sp_hstack(blocks, format="csr").astype(np.float32)


def build_item_cbf_similarity(icm: csr_matrix, k: int = 150, batch: int = 500) -> csr_matrix:
    """Cosine item-item similarity from ICM. Delegates to build_item_cbf_similarity_fast (CPU)."""
    return build_item_cbf_similarity_fast(icm, k=k, use_gpu=False)


def build_item_cbf_similarity_fast(
    icm: csr_matrix,
    k: int = 150,
    batch: int | None = None,
    shrink: float = 0.0,
    use_gpu: bool | None = None,
    verbose: bool = False,
) -> csr_matrix:
    """Fast cosine item-item similarity from sparse ICM.

    Speed-ups over :func:`build_item_cbf_similarity`:
      * vectorised per-batch ``np.argpartition`` (no per-row Python loop)
      * larger default batch size
      * optional GPU path via torch.sparse when ``use_gpu`` and CUDA available

    Parameters
    ----------
    icm:
        (n_items, n_features) sparse CSR.
    k:
        Top-k neighbours per item to retain.
    batch:
        Rows materialised as dense per chunk. Tune for RAM/VRAM.
    shrink:
        Cosine shrinkage term added to the denominator (0 = pure cosine on
        L2-normed rows).
    use_gpu:
        When True (and torch+CUDA available) compute the dense ``b × n`` chunk
        on GPU via ``torch.sparse.mm``. None = auto.
    """
    n = icm.shape[0]
    if k >= n:
        k = n - 1

    icm_norm = sk_normalize(icm, norm="l2", axis=1).astype(np.float32).tocsr()

    use_gpu = _resolve_use_gpu(use_gpu)
    # GPU batches can be large (cheap matmul); CPU batches stay small to avoid
    # the dense .toarray() blowing up RAM (the original 500 ≈ optimum on most
    # ICMs).
    if batch is None:
        batch = 4096 if use_gpu else 500
    if use_gpu:
        try:
            return _cbf_fast_gpu(icm_norm, k, batch, shrink, verbose)
        except Exception as exc:  # pragma: no cover
            print(f"[build_item_cbf_similarity_fast] GPU failed ({exc!r}); CPU fallback")
    return _cbf_fast_cpu(icm_norm, k, batch, shrink, verbose)


def _resolve_use_gpu(flag: bool | None) -> bool:
    if flag is False:
        return False
    try:
        import torch  # noqa: F401
        if not torch.cuda.is_available():
            return False
    except ImportError:
        return False
    return True if flag else (flag is None and torch.cuda.is_available())


def _topk_chunk(sims: np.ndarray, start: int, end: int, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (rows, cols, vals) of length b*k for a (b, n) sim chunk."""
    b, n = sims.shape
    diag_cols = np.arange(start, end)
    sims[np.arange(b), diag_cols] = 0.0
    if k < n:
        part = np.argpartition(-sims, k - 1, axis=1)[:, :k]
    else:
        part = np.broadcast_to(np.arange(n), (b, n)).copy()
    vals = np.take_along_axis(sims, part, axis=1)
    rows = np.repeat(diag_cols, part.shape[1])
    cols = part.ravel()
    vals_flat = vals.ravel()
    mask = vals_flat > 0
    return rows[mask], cols[mask], vals_flat[mask]


try:
    import numba as _numba

    @_numba.njit(parallel=True, cache=True)
    def _cbf_topk_kernel(
        data: np.ndarray, indices: np.ndarray, indptr: np.ndarray,
        data_T: np.ndarray, indices_T: np.ndarray, indptr_T: np.ndarray,
        n: int, k: int, shrink: float,
        out_cols: np.ndarray, out_vals: np.ndarray, out_nnz: np.ndarray,
    ) -> None:
        """Inverted-index top-k per row, parallel. Tracks touched items to avoid O(n) scan."""
        shrink_f = np.float32(shrink)
        for i in _numba.prange(n):
            acc = np.zeros(n, dtype=np.float32)
            touched = np.empty(n, dtype=np.int32)
            n_touched = np.int32(0)

            for fi in range(indptr[i], indptr[i + 1]):
                f = indices[fi]
                v = data[fi]
                for ji in range(indptr_T[f], indptr_T[f + 1]):
                    j = indices_T[ji]
                    if acc[j] == np.float32(0.0):
                        touched[n_touched] = j
                        n_touched += np.int32(1)
                    acc[j] += v * data_T[ji]

            acc[i] = np.float32(0.0)

            # collect only touched items — O(n_touched) not O(n)
            nz_idx = np.empty(n_touched, dtype=np.int32)
            nz_val = np.empty(n_touched, dtype=np.float32)
            nz = np.int32(0)
            for t in range(n_touched):
                j = touched[t]
                v = acc[j]
                acc[j] = np.float32(0.0)  # reset for next prange iteration
                if v > np.float32(0.0):
                    nz_idx[nz] = j
                    nz_val[nz] = v / (np.float32(1.0) + shrink_f) if shrink_f > np.float32(0.0) else v
                    nz += np.int32(1)

            actual_k = min(k, nz)
            if actual_k == 0:
                out_nnz[i] = 0
                continue

            order = np.argsort(nz_val[:nz])  # ascending
            for r in range(actual_k):
                idx = order[nz - 1 - r]
                out_cols[i, r] = nz_idx[idx]
                out_vals[i, r] = nz_val[idx]
            out_nnz[i] = actual_k

    @_numba.njit(cache=True)
    def _flatten_coo_icm(out_rows, out_vals, out_nnz, n):
        """Prefix-sum scatter: (n, k) dense → flat COO (row-major)."""
        prefix = np.empty(n + 1, dtype=np.int64)
        prefix[0] = np.int64(0)
        for i in range(n):
            prefix[i + 1] = prefix[i] + np.int64(out_nnz[i])
        total = prefix[n]
        rows_o = np.empty(total, dtype=np.int32)
        cols_o = np.empty(total, dtype=np.int32)
        vals_o = np.empty(total, dtype=np.float32)
        for i in range(n):
            p = prefix[i]
            cnt = out_nnz[i]
            for r in range(cnt):
                rows_o[p + r] = i
                cols_o[p + r] = out_rows[i, r]
                vals_o[p + r] = out_vals[i, r]
        return rows_o, cols_o, vals_o

    _NUMBA_OK = True
except ImportError:
    _NUMBA_OK = False


def _slurm_numba_threads() -> None:
    """Cap numba threads to SLURM allocation when running on a cluster."""
    if not _NUMBA_OK:
        return
    n = int(os.environ.get("SLURM_CPUS_PER_TASK", "0"))
    if n > 0:
        _numba.set_num_threads(n)


def _cbf_fast_cpu(icm_norm: csr_matrix, k: int, batch: int, shrink: float, verbose: bool) -> csr_matrix:
    n = icm_norm.shape[0]
    _slurm_numba_threads()

    if _NUMBA_OK:
        if verbose:
            print("[cbf_fast_cpu] using numba parallel kernel")
        icm_T = icm_norm.T.tocsr()
        data      = icm_norm.data.astype(np.float32)
        indices   = icm_norm.indices.astype(np.int32)
        indptr    = icm_norm.indptr.astype(np.int32)
        data_T    = icm_T.data.astype(np.float32)
        indices_T = icm_T.indices.astype(np.int32)
        indptr_T  = icm_T.indptr.astype(np.int32)

        out_cols = np.full((n, k), -1, dtype=np.int32)
        out_vals = np.zeros((n, k), dtype=np.float32)
        out_nnz  = np.zeros(n, dtype=np.int32)

        _cbf_topk_kernel(data, indices, indptr, data_T, indices_T, indptr_T,
                         n, k, float(shrink), out_cols, out_vals, out_nnz)

        rows_out, cols_out, vals_out = _flatten_coo_icm(out_cols, out_vals, out_nnz, n)
        return csr_matrix((vals_out, (rows_out, cols_out)), shape=(n, n), dtype=np.float32)

    # fallback: original batched scipy path
    icm_T = icm_norm.T.tocsr()
    out_rows, out_cols, out_vals = [], [], []
    iters = range(0, n, batch)
    if verbose:
        from tqdm.auto import tqdm
        iters = tqdm(list(iters), desc="cbf_fast_cpu")
    for start in iters:
        end = min(start + batch, n)
        sims = (icm_norm[start:end] @ icm_T).toarray().astype(np.float32, copy=False)
        if shrink:
            sims = sims / (1.0 + shrink)
        r, c, v = _topk_chunk(sims, start, end, k)
        out_rows.append(r); out_cols.append(c); out_vals.append(v)
    rows = np.concatenate(out_rows) if out_rows else np.empty(0, np.int32)
    cols = np.concatenate(out_cols) if out_cols else np.empty(0, np.int32)
    vals = np.concatenate(out_vals) if out_vals else np.empty(0, np.float32)
    return csr_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.float32)


def _cbf_fast_gpu(icm_norm: csr_matrix, k: int, batch: int, shrink: float, verbose: bool) -> csr_matrix:
    import torch
    torch.sparse.check_sparse_tensor_invariants.disable()
    n = icm_norm.shape[0]
    device = torch.device("cuda")

    coo = icm_norm.tocoo()
    indices = torch.from_numpy(np.vstack([coo.row, coo.col])).long().to(device)
    values = torch.from_numpy(coo.data.astype(np.float32)).to(device)
    sp = torch.sparse_coo_tensor(indices, values, size=coo.shape).coalesce()
    sp_T = sp.transpose(0, 1).coalesce()  # (n_feat, n)

    out_rows, out_cols, out_vals = [], [], []
    iters = range(0, n, batch)
    if verbose:
        from tqdm.auto import tqdm
        iters = tqdm(list(iters), desc="cbf_fast_gpu")
    for start in iters:
        end = min(start + batch, n)
        idx = torch.arange(start, end, device=device)
        # Slice rows of sp by gather on coalesced COO: build a small CSR for the slice.
        # Simpler: use torch.index_select on rows via a sparse re-index trick.
        # We use torch.sparse.mm(sub_sparse, sp_T) where sub_sparse holds only rows [start:end].
        sub_mask = (indices[0] >= start) & (indices[0] < end)
        sub_ind = indices[:, sub_mask].clone()
        sub_ind[0] -= start
        sub_val = values[sub_mask]
        sub = torch.sparse_coo_tensor(sub_ind, sub_val, size=(end - start, coo.shape[1])).coalesce()
        sims = torch.sparse.mm(sub, sp_T).to_dense()  # (b, n)
        if shrink:
            sims = sims / (1.0 + shrink)
        diag = torch.arange(start, end, device=device)
        sims[torch.arange(end - start, device=device), diag] = 0.0
        topv, topi = torch.topk(sims, k=min(k, n), dim=1)
        topv_np = topv.cpu().numpy()
        topi_np = topi.cpu().numpy()
        rows = np.repeat(np.arange(start, end), topi_np.shape[1])
        cols = topi_np.ravel()
        vals_flat = topv_np.ravel()
        mask = vals_flat > 0
        out_rows.append(rows[mask]); out_cols.append(cols[mask]); out_vals.append(vals_flat[mask])
    rows = np.concatenate(out_rows) if out_rows else np.empty(0, np.int32)
    cols = np.concatenate(out_cols) if out_cols else np.empty(0, np.int32)
    vals = np.concatenate(out_vals) if out_vals else np.empty(0, np.float32)
    return csr_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.float32)


def build_emb_similarity(
    emb: np.ndarray,
    k: int = 150,
    batch: int = 2048,
    shrink: float = 0.0,
    use_gpu: bool | None = None,
    verbose: bool = False,
) -> csr_matrix:
    """Top-k cosine item-item similarity from a dense embedding matrix.

    Assumes ``emb`` rows are not yet L2-normed (will be normed in-function).
    GPU path uses ``torch.matmul`` on the dense matrix; CPU uses BLAS.
    """
    n, d = emb.shape
    if k >= n:
        k = n - 1
    emb_n = emb.astype(np.float32, copy=False)
    norms = np.linalg.norm(emb_n, axis=1, keepdims=True)
    emb_n = emb_n / np.maximum(norms, 1e-8)

    use_gpu = _resolve_use_gpu(use_gpu)
    if use_gpu:
        try:
            return _emb_sim_gpu(emb_n, k, batch, shrink, verbose)
        except Exception as exc:  # pragma: no cover
            print(f"[build_emb_similarity] GPU failed ({exc!r}); CPU fallback")
    return _emb_sim_cpu(emb_n, k, batch, shrink, verbose)


def _emb_sim_cpu(emb_n: np.ndarray, k: int, batch: int, shrink: float, verbose: bool) -> csr_matrix:
    n = emb_n.shape[0]
    out_rows, out_cols, out_vals = [], [], []
    iters = range(0, n, batch)
    if verbose:
        from tqdm.auto import tqdm
        iters = tqdm(list(iters), desc="emb_sim_cpu")
    for start in iters:
        end = min(start + batch, n)
        sims = emb_n[start:end] @ emb_n.T
        if shrink:
            sims = sims / (1.0 + shrink)
        r, c, v = _topk_chunk(sims.astype(np.float32, copy=False), start, end, k)
        out_rows.append(r); out_cols.append(c); out_vals.append(v)
    rows = np.concatenate(out_rows) if out_rows else np.empty(0, np.int32)
    cols = np.concatenate(out_cols) if out_cols else np.empty(0, np.int32)
    vals = np.concatenate(out_vals) if out_vals else np.empty(0, np.float32)
    return csr_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.float32)


def _emb_sim_gpu(emb_n: np.ndarray, k: int, batch: int, shrink: float, verbose: bool) -> csr_matrix:
    import torch
    device = torch.device("cuda")
    n = emb_n.shape[0]
    et = torch.from_numpy(emb_n).to(device)
    out_rows, out_cols, out_vals = [], [], []
    iters = range(0, n, batch)
    if verbose:
        from tqdm.auto import tqdm
        iters = tqdm(list(iters), desc="emb_sim_gpu")
    for start in iters:
        end = min(start + batch, n)
        sims = et[start:end] @ et.T  # (b, n)
        if shrink:
            sims = sims / (1.0 + shrink)
        diag = torch.arange(start, end, device=device)
        sims[torch.arange(end - start, device=device), diag] = 0.0
        topv, topi = torch.topk(sims, k=k, dim=1)
        topv_np = topv.cpu().numpy()
        topi_np = topi.cpu().numpy()
        rows = np.repeat(np.arange(start, end), topi_np.shape[1])
        cols = topi_np.ravel()
        vals_flat = topv_np.ravel()
        mask = vals_flat > 0
        out_rows.append(rows[mask]); out_cols.append(cols[mask]); out_vals.append(vals_flat[mask])
    rows = np.concatenate(out_rows) if out_rows else np.empty(0, np.int32)
    cols = np.concatenate(out_cols) if out_cols else np.empty(0, np.int32)
    vals = np.concatenate(out_vals) if out_vals else np.empty(0, np.float32)
    return csr_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.float32)
