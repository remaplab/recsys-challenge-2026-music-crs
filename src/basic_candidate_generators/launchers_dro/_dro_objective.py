"""Robust scoring functions used as Optuna objectives for DRO CG tuning.

Operates on the per-subset NDCG@K (or any bounded metric) distribution
produced by `lbo.evaluator.BlindLikeEvaluator.score(...).per_subset`.

Three modes:
    mean       : arithmetic mean of per-subset scores (baseline / sanity)
    cvar       : CVaR_α — mean over the worst (1 - α) fraction of subsets.
                 Picks HPs robust to bad-subset behaviour; aligns with the
                 winner's-curse-safe selection rule in LBO doc B+C.
    group_dro  : minimum over per-group means (groups defined by per-subset
                 labels — see docstring of `group_dro_score`). Picks HPs that
                 maximise worst-group NDCG. Per LBO doc B group-DRO recipe.
"""
from __future__ import annotations

import numpy as np


def mean_score(per_subset: np.ndarray) -> float:
    clean = per_subset[~np.isnan(per_subset)]
    return float(clean.mean()) if clean.size else float("nan")


def cvar_score(per_subset: np.ndarray, *, alpha: float = 0.7) -> float:
    """CVaR_α = mean over the worst (1 - α) fraction.

    alpha=0.7 → mean of the bottom 30 % of subsets. Higher alpha → tighter
    tail focus. alpha=0 reduces to overall mean.
    """
    clean = per_subset[~np.isnan(per_subset)]
    if clean.size == 0:
        return float("nan")
    n_tail = max(1, int(round((1.0 - alpha) * clean.size)))
    return float(np.sort(clean)[:n_tail].mean())


def group_dro_score(
    per_subset: np.ndarray, *, subset_groups: np.ndarray,
) -> float:
    """Minimum over per-group means.

    `subset_groups` is an int array of shape (n_subsets,) labelling each
    subset with a group id (e.g. dominant `group_id_mode` from v2 density
    ratios). Empty groups skipped.
    """
    clean_mask = ~np.isnan(per_subset)
    if not clean_mask.any():
        return float("nan")
    groups = np.unique(subset_groups[clean_mask])
    if groups.size == 0:
        return float("nan")
    means = []
    for g in groups:
        m = clean_mask & (subset_groups == g)
        if m.any():
            means.append(float(per_subset[m].mean()))
    return float(np.min(means)) if means else float("nan")


def robust_score(
    per_subset: np.ndarray,
    *,
    mode: str = "cvar",
    alpha: float = 0.7,
    subset_groups: np.ndarray | None = None,
) -> float:
    """Dispatch entry point used by the Optuna objective.

    Args:
        per_subset: shape (n_subsets,), metric value per Blind-A-like subset.
        mode:       one of {"mean", "cvar", "group_dro"}.
        alpha:      CVaR level when mode="cvar".
        subset_groups: required when mode="group_dro".

    Returns the robust scalar to maximise.
    """
    if mode == "mean":
        return mean_score(per_subset)
    if mode == "cvar":
        return cvar_score(per_subset, alpha=alpha)
    if mode == "group_dro":
        if subset_groups is None:
            raise ValueError("group_dro mode requires subset_groups")
        return group_dro_score(per_subset, subset_groups=subset_groups)
    raise ValueError(f"unknown robust mode: {mode!r}")


def assign_subset_groups_by_session_mode(
    subsets: np.ndarray,
    session_ids: np.ndarray,
    sid_to_group: dict[str, int],
    *,
    fallback_group: int = -1,
) -> np.ndarray:
    """Assign each subset a single group via majority vote of its sessions'
    `group_id_mode` (from v2 `density_ratio.parquet`).

    Args:
        subsets:        shape (n_subsets, subset_size) of session-row indices
                        (matches `StratifiedEvaluator._prep.subsets[<strat>]`).
        session_ids:    shape (n_rows,) — session_id per eval row, same order
                        as the indices in `subsets`.
        sid_to_group:   {session_id: group_id_mode}, e.g. built from
                        v2 density_ratio.parquet:
                            {row["session_id"]: row["group_id_mode"] for row in ...}
        fallback_group: group label for sessions absent from sid_to_group.

    Returns shape (n_subsets,) int array.
    """
    n_subsets, _ = subsets.shape
    out = np.empty(n_subsets, dtype=np.int64)
    for i in range(n_subsets):
        sids = session_ids[subsets[i]]
        gids = [sid_to_group.get(s, fallback_group) for s in sids]
        # majority vote
        vals, counts = np.unique(gids, return_counts=True)
        out[i] = int(vals[counts.argmax()])
    return out
