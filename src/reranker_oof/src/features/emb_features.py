"""Embedding-similarity features (family M) for the reranker.

Within-space cosine of each candidate track against:
  - the conversation query   (Qwen3-8B space ONLY — only space with a query side)
  - the previous turn's track (t-1)
  - the session-history centroid (mean of the past-track vectors, re-normed)
  - the closest session track (max cosine to any past track)

Every tower is L2-normed at load, so a dot product equals cosine. You can only
compare embeddings of the SAME origin — query↔track is therefore Qwen-only, and
each modality (image-SigLIP2 / audio-CLAP) only does track↔track sims.

Missing values (candidate absent from a tower, no t-1 track, empty session,
un-encoded query) are left as NaN so XGBoost treats them as native-missing.

The catalogue is small (~47k tracks); all towers fit comfortably in RAM, so the
loaders are cached module-level and the cosines are computed with plain numpy in
row batches (no GPU needed).
"""
from __future__ import annotations

import glob
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

# repo root: .../src/reranker_oof/src/features/emb_features.py → parents[4]
_REPO_ROOT = Path(__file__).resolve().parents[4]

# Defaults (overridable via the YAML ``embeddings`` block). Paths are relative
# to the repo root.
_DEFAULTS = {
    "qwen_track_dir": "models/retrieval_text_towers/Qwen__Qwen3-Embedding-8B/dense_tracks_len256_poollast",
    "qwen_query_root": "models/retrieval_text_towers/Qwen__Qwen3-Embedding-8B",
    "modality_glob": "data/talkpl-ai/TalkPlayData-Challenge-Track-Embeddings/data/all_tracks-*.parquet",
}
# (feature prefix, parquet column) for the track-only modality towers.
# Only image-SigLIP2 and audio-CLAP are loaded.
_MODALITY: tuple[tuple[str, str], ...] = (
    ("siglip", "image-siglip2"),
    ("audio", "audio-laion_clap"),
)

_ROW_BATCH = 200_000


def _resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (_REPO_ROOT / p)


# ---------------------------------------------------------------------------
# Tower loaders (small reimplementations to keep this package self-contained)
# ---------------------------------------------------------------------------

def _load_track_tower(cache_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """(track_ids, emb) from a ``dense_tracks_*`` cache. Asserted L2-normed."""
    track_ids = np.load(cache_dir / "track_ids.npy", allow_pickle=True)
    emb = np.asarray(np.load(cache_dir / "embeddings.npy"), dtype=np.float32)
    norms = np.linalg.norm(emb[: min(256, emb.shape[0])], axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        raise ValueError(f"track tower {cache_dir} is not L2-normed")
    return track_ids, emb


def _load_modality_tower(parquet_glob: Path, column: str) -> tuple[np.ndarray, np.ndarray]:
    """(track_ids, emb) from a list-of-float ``column`` across parquet shards.

    Drops empty vectors and L2-normalises (raw modality vectors are not unit).
    """
    import pyarrow.parquet as pq

    paths = sorted(glob.glob(str(parquet_glob)))
    if not paths:
        raise FileNotFoundError(f"no parquet shards match {parquet_glob!r}")
    ids_l, emb_l = [], []
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


def _load_query_lookup(root: Path) -> tuple[np.ndarray, dict[tuple[str, int], int]]:
    
    dirs = sorted(
        d for d in root.glob("dense_*query*")
        if (d / "query_meta.parquet").exists() and (d / "query_embeddings.npy").exists()
    )
    if not dirs:
        raise FileNotFoundError(f"no dense_*query* caches under {root}")
    embs: list[np.ndarray] = []
    key_to_row: dict[tuple[str, int], int] = {}
    offset = 0
    for d in dirs:
        e = np.asarray(np.load(d / "query_embeddings.npy"), dtype=np.float32)
        meta = pl.read_parquet(d / "query_meta.parquet")
        for i, (sid, tn) in enumerate(zip(meta["session_id"].to_list(),
                                          meta["turn_number"].to_list())):
            key_to_row[(sid, int(tn))] = offset + i
        embs.append(e)
        offset += e.shape[0]
    query_emb = np.concatenate(embs, axis=0) if len(embs) > 1 else embs[0]
    return query_emb, key_to_row


# ---------------------------------------------------------------------------
# Resource bundle
# ---------------------------------------------------------------------------

@dataclass
class _Space:
    name: str                                  # feature prefix
    emb: np.ndarray                            # (n, d) L2-normed track tower
    tid_to_row: dict                           # track_id -> row in ``emb``
    has_query: bool = False
    query_emb: np.ndarray | None = None        # (m, d) L2-normed
    query_key_to_row: dict | None = None       # (sid, turn) -> row


@dataclass
class EmbeddingResources:
    spaces: list[_Space]
    row_batch: int = _ROW_BATCH


_CACHE: dict[str, "EmbeddingResources"] = {}


def load_embedding_resources(cfg: dict | None) -> EmbeddingResources | None:
    """Build (and cache) the embedding towers from a YAML ``embeddings`` block.

    ``cfg`` is the parsed ``embeddings:`` mapping (or ``None`` to disable).
    Recognised keys (all optional — sensible defaults baked in):
        qwen_track_dir, qwen_query_root, modality_glob, row_batch.
    Returns ``None`` when ``cfg`` is falsy.
    """
    if not cfg:
        return None
    key = repr(sorted(cfg.items()))
    if key in _CACHE:
        return _CACHE[key]

    t0 = time.time()
    qwen_dir = _resolve(cfg.get("qwen_track_dir", _DEFAULTS["qwen_track_dir"]))
    qwen_root = _resolve(cfg.get("qwen_query_root", _DEFAULTS["qwen_query_root"]))
    glob_pat = _resolve(cfg.get("modality_glob", _DEFAULTS["modality_glob"]))

    spaces: list[_Space] = []

    # Qwen3-8B text tower + query side.
    q_ids, q_emb = _load_track_tower(qwen_dir)
    query_emb, query_lut = _load_query_lookup(qwen_root)
    spaces.append(_Space(
        name="qwen", emb=q_emb,
        tid_to_row={t: i for i, t in enumerate(q_ids)},
        has_query=True, query_emb=query_emb, query_key_to_row=query_lut,
    ))

    # Track-only modality towers (image-SigLIP2 + audio-CLAP).
    for prefix, column in _MODALITY:
        ids, emb = _load_modality_tower(glob_pat, column)
        spaces.append(_Space(
            name=prefix, emb=emb,
            tid_to_row={t: i for i, t in enumerate(ids)},
        ))

    res = EmbeddingResources(spaces=spaces, row_batch=int(cfg.get("row_batch", _ROW_BATCH)))
    print(f"[emb_features] loaded {len(spaces)} spaces "
          f"({', '.join(s.name for s in spaces)}) in {time.time() - t0:.1f}s")
    _CACHE[key] = res
    return res


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------

def _pair_cos(
    E_a: np.ndarray, rows_a: np.ndarray,
    E_b: np.ndarray, rows_b: np.ndarray,
    batch: int,
) -> np.ndarray:
    """Row-wise cosine ``dot(E_a[rows_a[i]], E_b[rows_b[i]])``.

    Both matrices are L2-normed, so the dot IS the cosine. Entries where either
    row index is < 0 (missing) — or where a gathered vector is NaN (e.g. an
    empty-session centroid) — come back as NaN.
    """
    n = rows_a.shape[0]
    out = np.full(n, np.nan, dtype=np.float32)
    valid = np.nonzero((rows_a >= 0) & (rows_b >= 0))[0]
    for s in range(0, valid.size, batch):
        c = valid[s:s + batch]
        out[c] = np.einsum("ij,ij->i", E_a[rows_a[c]], E_b[rows_b[c]]).astype(np.float32)
    return out


def add_embedding_features(
    df: pl.DataFrame,
    res: EmbeddingResources,
    prev_df: pl.DataFrame,
    ctx_df: pl.DataFrame,
) -> pl.DataFrame:
    """Append the family-M cosine columns to ``df``.

    Parameters
    ----------
    df       : candidate pool (needs ``session_id``, ``turn_number``, ``track_id``).
    res      : loaded :class:`EmbeddingResources`.
    prev_df  : ``(session_id, turn_number, last_track_id)`` — the t-1 track.
    ctx_df   : ``(session_id, turn_number, ctx_track_ids)`` — list of past tracks.
    """
    if res is None or df.height == 0:
        return df
    n = df.height

    # Dense group id per row (stable within this chunk). Aligned to df row order.
    df = df.with_columns(
        (pl.struct("session_id", "turn_number").rank("dense") - 1)
        .cast(pl.Int64).alias("_gid")
    )
    gid = df["_gid"].to_numpy()
    n_groups = int(gid.max()) + 1
    cand_tid = df["track_id"].to_list()

    # gid → (session_id, turn_number) and per-group references.
    gmap = df.select("session_id", "turn_number", "_gid").unique(subset=["_gid"])
    prev = prev_df.join(gmap, on=["session_id", "turn_number"], how="inner")
    prev_tid_by_gid: list[str | None] = [None] * n_groups
    for g, t in zip(prev["_gid"].to_list(), prev["last_track_id"].to_list()):
        prev_tid_by_gid[g] = t

    cx = ctx_df.join(gmap, on=["session_id", "turn_number"], how="inner")
    members_by_gid: list[list[str]] = [[] for _ in range(n_groups)]
    for g, lst in zip(cx["_gid"].to_list(), cx["ctx_track_ids"].to_list()):
        members_by_gid[g] = lst or []
    max_m = max((len(m) for m in members_by_gid), default=0)

    # (sid, turn) per gid for the query lookup.
    gmap_g = gmap["_gid"].to_list()
    gmap_sid = gmap["session_id"].to_list()
    gmap_turn = gmap["turn_number"].to_list()

    new_cols: list[pl.Series] = []
    for sp in res.spaces:
        t2r = sp.tid_to_row
        E = sp.emb
        cand_row = np.fromiter((t2r.get(t, -1) for t in cand_tid),
                               dtype=np.int64, count=n)

        # --- candidate ↔ previous track ---
        prev_row_by_gid = np.fromiter(
            (t2r.get(prev_tid_by_gid[g], -1) if prev_tid_by_gid[g] is not None else -1
             for g in range(n_groups)),
            dtype=np.int64, count=n_groups,
        )
        prev_row = prev_row_by_gid[gid]
        new_cols.append(pl.Series(
            f"emb_{sp.name}_cand_prev_cos",
            _pair_cos(E, cand_row, E, prev_row, res.row_batch),
        ))

        # --- per-group member rows + centroid (for sessmean / sessmax) ---
        member_rows = np.full((n_groups, max_m), -1, dtype=np.int64)
        for g, lst in enumerate(members_by_gid):
            for j, t in enumerate(lst):
                member_rows[g, j] = t2r.get(t, -1)

        centroids = np.full((n_groups, E.shape[1]), np.nan, dtype=np.float32)
        for g in range(n_groups):
            rws = member_rows[g][member_rows[g] >= 0]
            if rws.size:
                c = E[rws].mean(axis=0)
                nrm = float(np.linalg.norm(c))
                if nrm > 0:
                    centroids[g] = c / nrm

        new_cols.append(pl.Series(
            f"emb_{sp.name}_cand_sessmean_cos",
            _pair_cos(E, cand_row, centroids, gid, res.row_batch),
        ))

        smax = np.full(n, np.nan, dtype=np.float32)
        for j in range(max_m):
            sims = _pair_cos(E, cand_row, E, member_rows[gid, j], res.row_batch)
            smax = np.fmax(smax, sims)         # NaN-aware: ignores missing slots
        new_cols.append(pl.Series(f"emb_{sp.name}_cand_sessmax_cos", smax))

        # --- candidate ↔ query (Qwen only) ---
        if sp.has_query:
            qrow_by_gid = np.full(n_groups, -1, dtype=np.int64)
            for g, sid, tn in zip(gmap_g, gmap_sid, gmap_turn):
                qrow_by_gid[g] = sp.query_key_to_row.get((sid, int(tn)), -1)
            qrow = qrow_by_gid[gid]
            new_cols.append(pl.Series(
                f"emb_{sp.name}_cand_query_cos",
                _pair_cos(E, cand_row, sp.query_emb, qrow, res.row_batch),
            ))

    df = df.drop("_gid").with_columns(new_cols)
    # emb_query_minus_sess: how much MORE the candidate matches the user's QUERY
    # text than the session-history centroid (Qwen).
    if {"emb_qwen_cand_query_cos", "emb_qwen_cand_sessmean_cos"} <= set(df.columns):
        df = df.with_columns(
            (pl.col("emb_qwen_cand_query_cos") - pl.col("emb_qwen_cand_sessmean_cos"))
            .alias("emb_query_minus_sess"))
    return df
