"""Marginal-distribution extraction + Dirichlet-smoothed stratified sampling.

Stratifies a candidate pool of sessions to match per-feature blind-A PMFs.
Each session contributes a per-feature `bucket` label; per subset we draw
N sessions whose joint distribution approximates the blind marginal product.
"""
from __future__ import annotations

import hashlib

import numpy as np
import polars as pl


def bucketize_numeric(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Right-inclusive digitize → returns bin index in [0, len(edges)-2]."""
    return np.clip(np.digitize(x, edges[1:-1]), 0, len(edges) - 2).astype(np.int32)


def extract_marginals(
    eval_df: pl.DataFrame,
    blind_df: pl.DataFrame,
    cols: list[str],
    n_bins: int = 4,
    alpha: float = 0.5,
) -> dict:
    """Per-feature PMF (Dirichlet-smoothed blind), shared bucket edges fitted on eval.

    Returns a dict::
        {
            col: {
                "type": "numeric"|"categorical",
                "edges": np.ndarray,                  # numeric only
                "buckets": list[str],                 # bucket labels for both kinds
                "blind_pmf": dict[str, float],        # smoothed
                "eval_pmf":  dict[str, float],
                "weight_per_bucket": dict[str, float] # blind_pmf / eval_pmf, used downstream
            }
        }
    """
    out: dict = {}
    for c in cols:
        if eval_df[c].dtype.is_numeric():
            full = eval_df[c].drop_nulls().to_numpy().astype(np.float64)
            lo = float(np.percentile(full, 1))
            hi = float(np.percentile(full, 99))
            edges = np.linspace(lo, hi, n_bins + 1)
            eval_b = bucketize_numeric(eval_df[c].fill_null(lo).to_numpy().astype(np.float64), edges)
            blind_b = bucketize_numeric(blind_df[c].fill_null(lo).to_numpy().astype(np.float64), edges)
            buckets = [f"b{i}" for i in range(n_bins)]
            eval_labels = np.array([f"b{int(b)}" for b in eval_b])
            blind_labels = np.array([f"b{int(b)}" for b in blind_b])
            t = "numeric"
        else:
            eval_labels = eval_df[c].fill_null("__nul__").to_numpy().astype(str)
            blind_labels = blind_df[c].fill_null("__nul__").to_numpy().astype(str)
            buckets = sorted(set(eval_labels.tolist()) | set(blind_labels.tolist()))
            edges = np.array([])
            t = "categorical"

        K = len(buckets)
        blind_counts = {b: 0 for b in buckets}
        for b in blind_labels:
            if b in blind_counts:
                blind_counts[b] += 1
        eval_counts = {b: 0 for b in buckets}
        for b in eval_labels:
            if b in eval_counts:
                eval_counts[b] += 1
        n_blind = sum(blind_counts.values())
        n_eval = sum(eval_counts.values())
        blind_pmf = {b: (blind_counts[b] + alpha) / (n_blind + alpha * K) for b in buckets}
        eval_pmf = {b: max(eval_counts[b] / max(n_eval, 1), 1e-9) for b in buckets}
        weight = {b: blind_pmf[b] / eval_pmf[b] for b in buckets}

        out[c] = {
            "type": t,
            "edges": edges,
            "buckets": buckets,
            "blind_pmf": blind_pmf,
            "eval_pmf": eval_pmf,
            "weight_per_bucket": weight,
        }
    return out


def compute_strat_weights(eval_df: pl.DataFrame, marginals: dict) -> np.ndarray:
    """Per-row weight = product of per-feature blind_pmf / eval_pmf."""
    w = np.ones(eval_df.height, dtype=np.float64)
    for c, m in marginals.items():
        if m["type"] == "numeric":
            lo = float(m["edges"][0])
            arr = eval_df[c].fill_null(lo).to_numpy().astype(np.float64)
            b = bucketize_numeric(arr, m["edges"])
            labels = np.array([f"b{int(x)}" for x in b])
        else:
            labels = eval_df[c].fill_null("__nul__").to_numpy().astype(str)
        wpb = m["weight_per_bucket"]
        w *= np.array([wpb.get(l, 1e-9) for l in labels], dtype=np.float64)
    return w


def _bucket_labels(eval_df: pl.DataFrame, marginals: dict) -> dict[str, np.ndarray]:
    """Per-feature bucket-label array (length = eval_df.height)."""
    out: dict[str, np.ndarray] = {}
    for c, m in marginals.items():
        if m["type"] == "numeric":
            lo = float(m["edges"][0])
            arr = eval_df[c].fill_null(lo).to_numpy().astype(np.float64)
            b = bucketize_numeric(arr, m["edges"])
            out[c] = np.array([f"b{int(x)}" for x in b])
        else:
            out[c] = eval_df[c].fill_null("__nul__").to_numpy().astype(str)
    return out


def calibrate_weights_to_marginals(
    init_weights: np.ndarray,
    eval_df: pl.DataFrame,
    marginals: dict,
    n_iter: int = 50,
    tol: float = 1e-6,
) -> np.ndarray:
    """Iterative Proportional Fitting (raking).

    Starts from `init_weights` (e.g. density-ratio weights from V9 XGB) and
    rescales them so that, **per stratification feature**, the weighted
    marginal PMF over eval_df buckets matches the blind PMF from `marginals`.

    Output is rescaled so its mean ≈ 1 (preserves relative magnitudes).

    Combines:
        - density-ratio's high-dim joint information (init weights)
        - hard marginal calibration on chosen features
    No alpha to tune. Convergence in ~5-20 iters in practice.
    """
    bucket_labels = _bucket_labels(eval_df, marginals)
    target_pmf = {c: m["blind_pmf"] for c, m in marginals.items()}

    w = np.maximum(np.asarray(init_weights, dtype=np.float64), 1e-12)
    w = w / w.sum()
    for it in range(n_iter):
        max_log_change = 0.0
        for c in marginals:
            labels = bucket_labels[c]
            target = target_pmf[c]
            for b, p_target in target.items():
                mask = labels == b
                cur_share = float(w[mask].sum())
                if cur_share > 0 and p_target > 0:
                    scale = p_target / cur_share
                    w[mask] *= scale
                    max_log_change = max(max_log_change, abs(np.log(scale)))
            w = w / w.sum()
        if max_log_change < tol:
            break
    return (w * len(w)).astype(np.float32)


def geometric_blend(
    w_density: np.ndarray, w_stratified: np.ndarray, alpha: float = 0.5,
) -> np.ndarray:
    """w_unified = w_density^alpha * w_stratified^(1-alpha). alpha in [0,1]."""
    a = np.maximum(np.asarray(w_density, dtype=np.float64), 1e-12)
    b = np.maximum(np.asarray(w_stratified, dtype=np.float64), 1e-12)
    w = np.power(a, alpha) * np.power(b, 1.0 - alpha)
    return w.astype(np.float32)


def kl_divergence_marginal(
    weights: np.ndarray, eval_df: pl.DataFrame, marginals: dict,
) -> dict[str, float]:
    """Per-feature KL(blind_pmf || weighted_eval_pmf). Sums to a single scalar
    via .values()."""
    bucket_labels = _bucket_labels(eval_df, marginals)
    w = np.maximum(np.asarray(weights, dtype=np.float64), 1e-12)
    w = w / w.sum()
    out: dict[str, float] = {}
    for c, m in marginals.items():
        labels = bucket_labels[c]
        target = m["blind_pmf"]
        kl = 0.0
        for b, p in target.items():
            q = float(w[labels == b].sum())
            if p > 1e-12:
                kl += p * (np.log(p + 1e-12) - np.log(q + 1e-12))
        out[c] = float(kl)
    return out


def _seed_hash(seed: int, key: str) -> float:
    h = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    return int.from_bytes(h[:8], "big") / 2**64


def sample_max_turn_per_session(
    session_ids: list[str], seed: int, pmf: dict[int, int]
) -> np.ndarray:
    """Deterministic per-session max_turn draw from the blind PMF."""
    items = sorted(pmf.items())
    values = np.array([k for k, _ in items], dtype=np.int64)
    probs = np.array([v for _, v in items], dtype=np.float64)
    probs = probs / probs.sum()
    cdf = np.cumsum(probs)
    out = np.empty(len(session_ids), dtype=np.int64)
    for i, sid in enumerate(session_ids):
        u = _seed_hash(seed, f"max_turn:{sid}")
        out[i] = int(values[int(np.searchsorted(cdf, u))])
    return out
