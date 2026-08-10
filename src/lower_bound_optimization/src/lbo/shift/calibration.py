"""Probabilistic calibration for the v2 C2ST ensemble.

Fits isotonic regression on (prob, label); falls back to Platt scaling when
ECE > threshold or isotonic slope < threshold (per LBO doc E).
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def expected_calibration_error(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """ECE with equal-width bins on [0,1]."""
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    ece = 0.0
    n = len(p)
    for b in range(n_bins):
        mask = bin_idx == b
        if not mask.any():
            continue
        ece += abs(p[mask].mean() - y[mask].mean()) * (mask.sum() / n)
    return float(ece)


def fit_isotonic(p_raw: np.ndarray, y: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """Returns callable raw_prob -> isotonic-calibrated prob."""
    ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    ir.fit(np.asarray(p_raw), np.asarray(y, dtype=np.float64))
    return lambda x: np.clip(ir.transform(np.asarray(x)), 0.0, 1.0)


def fit_platt(p_raw: np.ndarray, y: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """Logistic regression on logits — single-parameter Platt scaling."""
    p = np.clip(np.asarray(p_raw, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    logits = np.log(p / (1.0 - p)).reshape(-1, 1)
    lr = LogisticRegression(C=1e6).fit(logits, np.asarray(y, dtype=np.int8))

    def calibrate(x: np.ndarray) -> np.ndarray:
        x = np.clip(np.asarray(x, dtype=np.float64), 1e-7, 1.0 - 1e-7)
        z = np.log(x / (1.0 - x)).reshape(-1, 1)
        return lr.predict_proba(z)[:, 1]

    return calibrate


def _isotonic_slope(cal: Callable, n_grid: int = 50) -> float:
    """LS slope of the isotonic map over a grid in [0,1]."""
    x = np.linspace(0.0, 1.0, n_grid)
    y = cal(x)
    slope = np.polyfit(x, y, 1)[0]
    return float(slope)


def fit_calibrator(
    p_raw: np.ndarray,
    y: np.ndarray,
    *,
    ece_threshold: float = 0.05,
    slope_threshold: float = 0.7,
) -> tuple[Callable[[np.ndarray], np.ndarray], str, float]:
    """Try isotonic. If ECE > threshold OR slope < threshold, fall back to Platt.

    Returns (calibrator, kind, ece_after).
    """
    iso = fit_isotonic(p_raw, y)
    iso_p = iso(p_raw)
    iso_ece = expected_calibration_error(iso_p, y)
    iso_slope = _isotonic_slope(iso)

    if iso_ece > ece_threshold or iso_slope < slope_threshold:
        platt = fit_platt(p_raw, y)
        platt_ece = expected_calibration_error(platt(p_raw), y)
        return platt, "platt", platt_ece
    return iso, "isotonic", iso_ece


def ensemble_probabilities(probs_per_model: list[np.ndarray]) -> np.ndarray:
    """Uniform average of calibrated probabilities across models (doc D shortcut)."""
    if not probs_per_model:
        raise ValueError("ensemble_probabilities requires >=1 array")
    stacked = np.stack(
        [np.asarray(p, dtype=np.float64) for p in probs_per_model], axis=0,
    )
    return stacked.mean(axis=0)
