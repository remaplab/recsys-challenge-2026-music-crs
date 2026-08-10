"""Aggregate qwen3 query + track embeddings per augmented session.

Query embedding cache layout (3 shards):
    train.npy + train_meta.parquet  — raw TRAIN sessions (15199 × 8 turns)
    dev.npy   + dev_meta.parquet    — raw TEST sessions (1000 × 8 turns)
    blind_a.npy + blind_a_meta.parquet — BLIND sessions (80, variable turns)

Per augmented session we compute:
    q_mean ∈ R^1024 — mean of query embeddings at turns 1..max_turn+1
    t_mean ∈ R^1024 — mean of track embeddings for prior_track_ids (size max_turn)
                      (zeros when max_turn == 0)
"""
from __future__ import annotations

import numpy as np
import polars as pl

from lbo.paths import QUERY_EMB_DIR, TRACK_EMB_DIR

EMB_DIM = 1024


def _load_query_emb_table() -> tuple[pl.DataFrame, np.ndarray]:
    """Concatenate the 3 query-emb shards into one (meta, emb_matrix)."""
    metas = []
    embs = []
    for shard in ("train", "dev", "blind_a"):
        m = pl.read_parquet(QUERY_EMB_DIR / f"{shard}_meta.parquet")
        e = np.load(QUERY_EMB_DIR / f"{shard}.npy", mmap_mode="r")
        assert m.shape[0] == e.shape[0], shard
        metas.append(m.select("session_id", "turn_number"))
        embs.append(np.asarray(e))
    meta = pl.concat(metas, how="vertical").with_row_index(name="row_idx")
    emb_matrix = np.concatenate(embs, axis=0)
    assert meta.shape[0] == emb_matrix.shape[0]
    return meta, emb_matrix


def _load_track_emb_table() -> tuple[dict[str, int], np.ndarray]:
    emb = np.load(TRACK_EMB_DIR / "embeddings.npy", mmap_mode="r")
    ids = np.load(TRACK_EMB_DIR / "track_ids.npy", allow_pickle=True)
    id_to_idx = {tid: i for i, tid in enumerate(ids)}
    return id_to_idx, np.asarray(emb)


def aggregate_embeddings(
    aug_df: pl.DataFrame,
    per_turn_df: pl.DataFrame,
) -> tuple[np.ndarray, np.ndarray, list[str], list[int]]:
    """Return (q_mean, t_mean, session_ids, aug_ids).

    `aug_df` columns: session_id, user_id, aug_id, max_turn, source.
    `per_turn_df` is the assemble output (used only to know which turns each
    session has — but we re-derive from the emb meta so this argument is unused;
    kept for signature parity).
    """
    del per_turn_df

    print("[emb] loading query embedding shards…")
    qmeta, qmat = _load_query_emb_table()
    print(f"[emb] query rows: {qmat.shape}")

    print("[emb] loading track embeddings…")
    tid_to_idx, tmat = _load_track_emb_table()
    print(f"[emb] track rows: {tmat.shape}")

    # Build session -> sorted (turn_number, row_idx) list, once.
    qmeta_sorted = qmeta.sort("session_id", "turn_number")
    sess_groups: dict[str, list[tuple[int, int]]] = {}
    for sid, tn, ridx in zip(
        qmeta_sorted["session_id"].to_list(),
        qmeta_sorted["turn_number"].to_list(),
        qmeta_sorted["row_idx"].to_list(),
    ):
        sess_groups.setdefault(sid, []).append((int(tn), int(ridx)))

    # Pre-build track-list lookup from one of the meta shards (prior_track_ids
    # column lives there at every turn). Concatenate all shards' full meta.
    full_metas = []
    for shard in ("train", "dev", "blind_a"):
        full_metas.append(pl.read_parquet(QUERY_EMB_DIR / f"{shard}_meta.parquet"))
    full_meta = pl.concat(full_metas, how="vertical")
    # (session_id, turn_number) -> prior_track_ids
    prior_lookup: dict[tuple[str, int], list[str]] = {
        (sid, int(tn)): list(p) if p is not None else []
        for sid, tn, p in zip(
            full_meta["session_id"].to_list(),
            full_meta["turn_number"].to_list(),
            full_meta["prior_track_ids"].to_list(),
        )
    }

    n = aug_df.shape[0]
    q_out = np.zeros((n, EMB_DIM), dtype=np.float32)
    t_out = np.zeros((n, EMB_DIM), dtype=np.float32)
    session_ids: list[str] = []
    aug_ids: list[int] = []

    sid_arr = aug_df["session_id"].to_list()
    aug_arr = aug_df["aug_id"].to_list()
    mt_arr = aug_df["max_turn"].to_list()

    missing_sess = 0
    for i in range(n):
        sid = sid_arr[i]
        aug_id = int(aug_arr[i])
        max_turn = int(mt_arr[i])
        session_ids.append(sid)
        aug_ids.append(aug_id)

        # Query mean over turns 1..max_turn+1
        turns = sess_groups.get(sid)
        if turns is None:
            missing_sess += 1
            continue
        wanted_rows = [r for (t, r) in turns if 1 <= t <= max_turn + 1]
        if wanted_rows:
            q_out[i] = qmat[wanted_rows].mean(axis=0)

        # Track mean over prior_track_ids at the prediction turn (max_turn+1)
        prior = prior_lookup.get((sid, max_turn + 1), [])
        idxs = [tid_to_idx[t] for t in prior if t in tid_to_idx]
        if idxs:
            t_out[i] = tmat[idxs].mean(axis=0)

    if missing_sess:
        print(f"[emb] WARNING: {missing_sess} sessions missing query embeddings")
    return q_out, t_out, session_ids, aug_ids


if __name__ == "__main__":
    from lbo.shift.assemble import assemble
    from lbo.shift.augment import build_augmented

    df = assemble()
    aug = build_augmented(df, n_augs=3, seed=42)
    q, t, sids, augs = aggregate_embeddings(aug, df)
    print("q shape:", q.shape, "t shape:", t.shape)
    print("q norms:", np.linalg.norm(q, axis=1)[:5])
    print("t norms:", np.linalg.norm(t, axis=1)[:5])
    print("t zero rows (max_turn=0 expected):", (np.linalg.norm(t, axis=1) == 0).sum())
