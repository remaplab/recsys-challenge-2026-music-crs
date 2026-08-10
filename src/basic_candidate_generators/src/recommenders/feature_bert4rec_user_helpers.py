"""Shared utilities for user-aware variants of the CG family.

Provides:
  - `load_user_cf_embeddings`: load the `cf-bpr` 128d user embeddings from
    `TalkPlayData-Challenge-User-Embeddings/data/{train,test_warm,test_cold}.parquet`.
    Returns dict `user_id -> np.ndarray (128,)`. Cold users get zeros
    (the test_cold parquet stores empty embeddings by design — same source
    as `src/embeddings-package/src/data/user_features.py`).

  - `compute_user_taste_embeddings`: aggregate the trained item_encoder output
    over each user's training tracks. Used by variants A and C
    (history-based). Cold users (no training history) get a zero vector.

  - `_UserAwareDataset`: extends `_FeatureBert4RecDataset` to also yield an
    `(int)` user_idx per sample. The model can then look up the user's
    vector in a buffered table for scoring.

  - `build_session_user_map`: dict session_id -> user_id, for inference
    routing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import polars as pl
import torch

from .feature_bert4rec import _FeatureBert4RecDataset


USER_CF_DIM = 128  # matches src/embeddings-package/src/data/user_features.py


def load_user_cf_embeddings(
    parquet_paths: Iterable[str | Path],
    user_ids_filter: set[str] | None = None,
) -> tuple[dict[str, np.ndarray], int]:
    """Load `cf-bpr` (128d) user embeddings from one or more parquets.

    Returns
    -------
    user_to_emb : dict[user_id, np.ndarray (128,)]
        L2-normalised vectors. Empty/null vectors are skipped (cold users
        get nothing here; caller substitutes zeros).
    dim : int
        Embedding dimension (always USER_CF_DIM=128 for the cf-bpr column).
    """
    out: dict[str, np.ndarray] = {}
    for path in parquet_paths:
        df = pl.read_parquet(Path(path))
        if "user_id" not in df.columns or "cf-bpr" not in df.columns:
            continue
        for uid, vec in zip(df["user_id"].to_list(), df["cf-bpr"].to_list()):
            if user_ids_filter is not None and uid not in user_ids_filter:
                continue
            if vec is None:
                continue
            arr = np.asarray(vec, dtype=np.float32)
            if arr.size != USER_CF_DIM or not arr.any():
                continue
            n = float(np.linalg.norm(arr))
            if n <= 0:
                continue
            out[uid] = arr / n
    return out, USER_CF_DIM


def build_session_user_map(df: pl.DataFrame) -> dict[str, str]:
    """session_id -> user_id (single value per session — TalkPlay is 1-to-1)."""
    if "session_id" not in df.columns or "user_id" not in df.columns:
        raise KeyError("DataFrame must have session_id + user_id columns")
    sub = df.select(["session_id", "user_id"]).unique(subset=["session_id"])
    return dict(zip(sub["session_id"].to_list(), sub["user_id"].to_list()))


@torch.no_grad()
def compute_user_taste_embeddings(
    item_encoder,
    feature_matrix: torch.Tensor,
    warm_global_to_local: dict[int, int],
    user_to_played_global: dict[str, list[int]],
    hidden_size: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """For each user, mean of item_encoder(feature_matrix[user's warm tracks]).

    Computes the model-space user vector as the average of the item-embedding
    of the warm tracks the user actually played in training. Cold users
    (no warm tracks) are returned as zeros.

    Parameters
    ----------
    item_encoder : the trained encoder (e.g. `model.item_encoder`)
    feature_matrix : (n_warm, total_dim) tensor of WARM-track features
    warm_global_to_local : maps global track idx -> warm-local idx in feature_matrix
    user_to_played_global : maps user_id -> list of global track idxs they played in train
    hidden_size : dim of encoder output (= dim of returned vectors)
    device : torch device

    Returns
    -------
    dict[user_id, torch.Tensor(hidden_size,)]  on `device`, L2-normalised.
    """
    item_emb_all = item_encoder(feature_matrix)              # (n_warm, hidden)
    item_emb_all = torch.nn.functional.normalize(item_emb_all, dim=-1)

    out: dict[str, torch.Tensor] = {}
    zero = torch.zeros(hidden_size, dtype=item_emb_all.dtype, device=device)
    for uid, global_idxs in user_to_played_global.items():
        local = [warm_global_to_local[g] for g in global_idxs if g in warm_global_to_local]
        if not local:
            out[uid] = zero
            continue
        idxs = torch.tensor(local, dtype=torch.long, device=device)
        mean = item_emb_all[idxs].mean(dim=0)
        n = torch.linalg.vector_norm(mean) + 1e-12
        out[uid] = mean / n
    return out


# ---------------------------------------------------------------------------
# Dataset: yields (masked_seq, labels, user_idx_int) per sample
# ---------------------------------------------------------------------------

class _UserAwareDataset(_FeatureBert4RecDataset):
    """`_FeatureBert4RecDataset` + parallel `user_idxs` list, one int per sample.

    user_idx is an integer index into the model's user-vector lookup table.
    Cold/unknown users use index 0 (reserved zero-vector slot).
    """

    def __init__(
        self,
        sequences: list[list[int]],
        user_idxs: list[int],
        n_warm: int,
        max_seq_len: int,
        mask_prob: float,
    ) -> None:
        assert len(sequences) == len(user_idxs)
        super().__init__(sequences, n_warm, max_seq_len, mask_prob)
        self.user_idxs = user_idxs

    def __getitem__(self, idx: int):
        masked, labels = super().__getitem__(idx)
        return masked, labels, int(self.user_idxs[idx])
