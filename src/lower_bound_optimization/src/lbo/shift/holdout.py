"""User-coherent holdout split for honest tuning evaluation.

A deterministic hash on `user_id` (salted with seed + a fixed namespace string
so the holdout split is independent from the CV fold assignment) selects a
fraction of users for the frozen holdout. The holdout is applied INDEPENDENTLY
within each source (train, blind) so the holdout has the same blind/train ratio
as the full dataset.
"""
from __future__ import annotations

import hashlib

import numpy as np
import polars as pl

HOLDOUT_SALT = "lbo_holdout_v1"


def _user_hash_unit(user_id: str, seed: int) -> float:
    """Deterministic uniform[0, 1) draw per user_id, salted by seed + namespace."""
    h = hashlib.sha256(f"{HOLDOUT_SALT}:{seed}:{user_id}".encode()).digest()
    return int.from_bytes(h[:8], "big") / 2**64


def compute_holdout_mask(tab: pl.DataFrame, holdout_frac: float, seed: int) -> np.ndarray:
    """Return boolean array of length tab.height: True = row in holdout.

    Holdout fraction is enforced per source (`tab['source']` values), so blind
    and train are sampled at the same fraction. User-coherent: every (session,
    aug_id) row of the same user lands on the same side of the split.
    """
    if holdout_frac <= 0.0:
        return np.zeros(tab.height, dtype=bool)
    if holdout_frac >= 1.0:
        return np.ones(tab.height, dtype=bool)

    holdout_users: set[str] = set()
    for source in tab["source"].unique().to_list():
        users = tab.filter(pl.col("source") == source)["user_id"].unique().to_list()
        for u in users:
            if _user_hash_unit(u, seed) < holdout_frac:
                holdout_users.add(u)

    mask = np.array(
        [u in holdout_users for u in tab["user_id"].to_list()], dtype=bool
    )
    return mask


def summary(tab: pl.DataFrame, mask: np.ndarray) -> dict:
    """Per-source counts in cv vs holdout."""
    out: dict = {"cv": {}, "holdout": {}}
    src = tab["source"].to_numpy()
    user = tab["user_id"].to_numpy()
    for source in np.unique(src):
        for split, m in (("cv", ~mask), ("holdout", mask)):
            sel = (src == source) & m
            out[split][str(source)] = {
                "rows": int(sel.sum()),
                "users": int(len(np.unique(user[sel]))),
            }
    return out
