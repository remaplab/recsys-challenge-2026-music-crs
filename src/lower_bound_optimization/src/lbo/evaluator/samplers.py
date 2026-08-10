"""Subset samplers: density-weighted + stratified-marginal.

All return shape (n_subsets, subset_size) of *row indices* into the eval pool.
"""
from __future__ import annotations

import numpy as np


def density_weighted_subsets(
    weights: np.ndarray, n_subsets: int, subset_size: int, seed: int,
    with_replacement_inside: bool = False,
) -> np.ndarray:
    """Sample n_subsets × subset_size row indices with prob ∝ weights.

    By default, each subset samples WITHOUT replacement so the same session
    can't appear twice in the same draw.
    """
    rng = np.random.default_rng(seed)
    n = len(weights)
    p = np.asarray(weights, dtype=np.float64)
    p = np.maximum(p, 1e-12)
    p = p / p.sum()

    out = np.empty((n_subsets, subset_size), dtype=np.int64)
    for i in range(n_subsets):
        # WithoutReplacement when subset_size <= n
        if not with_replacement_inside and subset_size <= n:
            out[i] = rng.choice(n, size=subset_size, replace=False, p=p)
        else:
            out[i] = rng.choice(n, size=subset_size, replace=True, p=p)
    return out


def stratified_subsets(
    weights: np.ndarray, n_subsets: int, subset_size: int, seed: int,
) -> np.ndarray:
    """Same machinery as density_weighted_subsets but uses precomputed
    stratification weights (Dirichlet-smoothed marginal product).
    Kept as a separate function for clarity / future divergence.
    """
    return density_weighted_subsets(
        weights, n_subsets=n_subsets, subset_size=subset_size, seed=seed
    )
