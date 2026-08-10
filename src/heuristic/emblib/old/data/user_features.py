"""Load user metadata + user CF embeddings into a unified per-user feature object."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

import numpy as np
import polars as pl


USER_CF_DIM = 128


@dataclass
class UserFeatures:
    """Per-user features keyed by user_id.

    Attributes:
        user_ids: list of user ids (length N), in row order.
        id_to_idx: dict mapping user_id -> row index.
        cf: (N, 128) float32, L2-normalized CF embedding. Cold users get zeros.
        is_cold: (N,) bool, True if no CF embedding was available.
        metadata: dict mapping user_id -> dict with age, gender, country_name, etc.
                  (kept as Python dicts since these are categorical / heterogeneous).
    """
    user_ids: list[str]
    id_to_idx: Dict[str, int]
    cf: np.ndarray
    is_cold: np.ndarray
    metadata: Dict[str, dict] = field(default_factory=dict)

    @property
    def n_users(self) -> int:
        return len(self.user_ids)

    def coverage(self) -> dict:
        return {
            "warm": float((~self.is_cold).mean()),
            "cold": float(self.is_cold.mean()),
        }

    def __repr__(self) -> str:
        cov = self.coverage()
        return (f"UserFeatures(n_users={self.n_users}, "
                f"warm={cov['warm']:.3f}, cold={cov['cold']:.3f})")


def load_user_features(
    user_meta_path: Path,
    user_emb_train_path: Path,
    user_emb_warm_path: Path | None = None,
    user_emb_cold_path: Path | None = None,
    cache_dir: Path | None = None,
) -> UserFeatures:
    """Load user metadata + user CF embeddings, joined by user_id.

    The user CF embedding files (train / test_warm / test_cold) collectively
    cover all users in user_metadata. test_cold has empty embeddings by design.
    Any user appearing in metadata but not in the embedding files is treated as cold.
    """
    user_meta_path = Path(user_meta_path)
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        if all((cache_dir / f).exists() for f in ["user_ids.npy", "cf.npy", "is_cold.npy"]):
            print(f"Loading cached UserFeatures from {cache_dir}")
            user_ids = np.load(cache_dir / "user_ids.npy", allow_pickle=True).tolist()
            cf = np.load(cache_dir / "cf.npy")
            is_cold = np.load(cache_dir / "is_cold.npy")
            # Metadata kept fresh from parquet rather than caching dicts
            meta_df = pl.read_parquet(user_meta_path)
            metadata = {row["user_id"]: row for row in meta_df.to_dicts()}
            uf = UserFeatures(
                user_ids=user_ids,
                id_to_idx={uid: i for i, uid in enumerate(user_ids)},
                cf=cf,
                is_cold=is_cold,
                metadata=metadata,
            )
            print(uf)
            return uf

    print(f"Loading user metadata from {user_meta_path}")
    meta_df = pl.read_parquet(user_meta_path)
    metadata = {row["user_id"]: row for row in meta_df.to_dicts()}
    user_ids = meta_df["user_id"].to_list()
    n = len(user_ids)
    id_to_idx = {uid: i for i, uid in enumerate(user_ids)}
    print(f"  {n} users")

    cf = np.zeros((n, USER_CF_DIM), dtype=np.float32)
    is_cold = np.ones(n, dtype=bool)  # default: cold; turned off when CF found

    def absorb(path: Path, label: str) -> None:
        if path is None or not path.exists():
            return
        print(f"Loading user CF from {path} ({label})")
        df = pl.read_parquet(path)
        absorbed = 0
        for row in df.to_dicts():
            uid = row["user_id"]
            v = row["cf-bpr"]
            if uid not in id_to_idx:
                continue
            idx = id_to_idx[uid]
            if v is not None and len(v) == USER_CF_DIM:
                vec = np.asarray(v, dtype=np.float32)
                norm = np.linalg.norm(vec) + 1e-12
                cf[idx] = vec / norm
                is_cold[idx] = False
                absorbed += 1
        print(f"  absorbed {absorbed} non-empty CF vectors")

    absorb(Path(user_emb_train_path), "train")
    if user_emb_warm_path is not None:
        absorb(Path(user_emb_warm_path), "warm")
    if user_emb_cold_path is not None:
        absorb(Path(user_emb_cold_path), "cold")

    uf = UserFeatures(
        user_ids=user_ids, id_to_idx=id_to_idx, cf=cf, is_cold=is_cold, metadata=metadata,
    )

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(cache_dir / "user_ids.npy", np.asarray(user_ids, dtype=object))
        np.save(cache_dir / "cf.npy", cf)
        np.save(cache_dir / "is_cold.npy", is_cold)
        print(f"Cached to {cache_dir}")

    print(uf)
    return uf