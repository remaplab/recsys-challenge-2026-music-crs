"""Data plumbing for tower_ensemble: query-embedding store from the splitK
query caches, frozen track-tower alignment, and training-pair assembly.
"""
from __future__ import annotations

import glob
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import torch


@dataclass
class QueryStore:
    emb_by_key: dict[tuple[str, int], np.ndarray]
    gt_by_key: dict[tuple[str, int], str]
    dim: int


def load_query_store(tower_base: str) -> QueryStore:
    """Build a global (session_id, turn_number) → (query_emb, gt_track_id) store
    from every `dense_splitk_*_query_len512_poollast` bucket under `tower_base`.
    """
    emb_by_key: dict[tuple[str, int], np.ndarray] = {}
    gt_by_key: dict[tuple[str, int], str] = {}
    dim = 0
    pattern = str(Path(tower_base) / "dense_splitk_*_query_len512_poollast")
    dirs = sorted(glob.glob(pattern))
    if not dirs:
        raise FileNotFoundError(f"no splitK query caches under {pattern}")
    for d in dirs:
        emb = np.asarray(np.load(Path(d) / "query_embeddings.npy"), dtype=np.float32)
        meta = pl.read_parquet(Path(d) / "query_meta.parquet",
                               columns=["session_id", "turn_number", "gt_track_id"])
        dim = emb.shape[1]
        for i, (sid, tn, gt) in enumerate(zip(
            meta["session_id"].to_list(), meta["turn_number"].to_list(),
            meta["gt_track_id"].to_list(),
        )):
            emb_by_key[(sid, int(tn))] = emb[i]
            gt_by_key[(sid, int(tn))] = gt
    return QueryStore(emb_by_key, gt_by_key, dim)


def align_tower_to_idx(tower_ids: np.ndarray, emb: np.ndarray,
                       track_to_idx: dict[str, int], n_tracks: int) -> np.ndarray:
    """Reorder a tower's rows into `track_to_idx` order; missing tracks → zeros."""
    out = np.zeros((n_tracks, emb.shape[1]), dtype=np.float32)
    for row, tid in enumerate(tower_ids):
        j = track_to_idx.get(tid)
        if j is not None:
            out[j] = emb[row]
    return out


def build_training_pairs(store: QueryStore, session_set: set[str],
                         track_to_idx: dict[str, int]) -> tuple[torch.Tensor, torch.Tensor]:
    """(query_emb, positive catalogue idx) pairs for training sessions.

    Drops pairs whose gt track is absent from `track_to_idx` (e.g. tower B's
    image-less tracks when called with the image catalogue's idx map).
    """
    qs: list[np.ndarray] = []
    pos: list[int] = []
    for key, emb in store.emb_by_key.items():
        if key[0] not in session_set:
            continue
        gt = store.gt_by_key.get(key)
        j = track_to_idx.get(gt) if gt is not None else None
        if j is None:
            continue
        qs.append(emb); pos.append(j)
    if not qs:
        return torch.empty(0, store.dim), torch.empty(0, dtype=torch.long)
    return (torch.from_numpy(np.stack(qs)),
            torch.tensor(pos, dtype=torch.long))
