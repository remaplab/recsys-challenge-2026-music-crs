from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import polars as pl
import torch

from .feature_bert4rec import _FeatureBert4RecDataset

USER_CF_DIM = 128

def load_user_cf_embeddings(
    parquet_paths: Iterable[str | Path],
    user_ids_filter: set[str] | None = None,
) -> tuple[dict[str, np.ndarray], int]:
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
    item_emb_all = item_encoder(feature_matrix)
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

class _UserAwareDataset(_FeatureBert4RecDataset):

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
