"""Multi-comparison correction + paired Empirical-Bernstein dedup + PoSI helper.

Used by s06_distribution_shift_v2 to pick a small Holm-significant ensemble of
Optuna trials without winner's curse (per LBO docs C and D).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests


def z_pvalues_against_null(
    aucs: np.ndarray, *, mu_null: float, sigma_null: float,
) -> np.ndarray:
    """One-sided z-test p-values vs a shared permutation null."""
    z = (np.asarray(aucs, dtype=np.float64) - mu_null) / max(sigma_null, 1e-9)
    return norm.sf(z)


def holm_bonferroni(pvals: np.ndarray, *, alpha: float = 0.05) -> np.ndarray:
    """Boolean mask of survivors under Holm-Bonferroni at level `alpha`."""
    pvals = np.asarray(pvals)
    if len(pvals) == 0:
        return np.array([], dtype=bool)
    reject, _, _, _ = multipletests(pvals, alpha=alpha, method="holm")
    return reject


def paired_empirical_bernstein_ci(
    deltas: np.ndarray, *, delta: float = 0.05,
) -> tuple[float, float]:
    """Maurer-Pontil empirical-Bernstein two-sided CI on the mean of paired Δ.

        half = sqrt(2 σ̂² ln(3/δ) / n) + 3 ln(3/δ) / n

    Assumes Δ_i ∈ [-1, 1] (satisfied for AUC / NDCG differences).
    """
    x = np.asarray(deltas, dtype=np.float64)
    n = len(x)
    if n < 2:
        return float("-inf"), float("inf")
    mean = float(x.mean())
    var = float(x.var(ddof=1))
    ln_term = np.log(3.0 / max(delta, 1e-12))
    half = np.sqrt(2.0 * var * ln_term / n) + 3.0 * ln_term / n
    return mean - half, mean + half


def deduplicate_by_paired_eb(
    survivors: list[int],
    fold_aucs: dict[int, np.ndarray],
    *,
    delta: float = 0.05,
) -> list[int]:
    """Drop trials that are CI-significantly dominated by a stronger survivor.

    For each ordered pair (t1, t2) where t1 has strictly higher mean AUC, if the
    paired EB lower bound of (AUC_t1 - AUC_t2) > 0, drop t2.
    """
    keep = set(survivors)
    ordered = sorted(survivors, key=lambda t: -float(fold_aucs[t].mean()))
    for i, t1 in enumerate(ordered):
        if t1 not in keep:
            continue
        for t2 in ordered[i + 1:]:
            if t2 not in keep:
                continue
            d = fold_aucs[t1] - fold_aucs[t2]
            if float(d.mean()) <= 0:
                continue
            lo, _ = paired_empirical_bernstein_ci(d, delta=delta)
            if lo > 0:
                keep.discard(t2)
    return sorted(keep)


def compute_permutation_null(
    *,
    tab,
    q_proj: np.ndarray,
    t_proj: np.ndarray,
    q_dim_indices: list[int],
    t_dim_indices: list[int],
    selected_numeric: list[str],
    selected_categorical: list[str],
    n_perms: int,
    n_folds: int,
    seed: int,
    use_cuda: bool = True,
) -> dict:
    """Shuffle labels n_perms times, fit a baseline XGB, return null AUC stats.

    Reuses the same PCA projection + BH-selected dim indices as `tune_v2`.
    Result feeds (mu_null, sigma_null) used to z-score every Optuna trial.
    """
    from sklearn.metrics import roc_auc_score
    import xgboost as xgb

    from lbo.shift.tune_v2 import _build_X, _user_coherent_folds

    y = tab["label"].to_numpy().astype(np.int8)
    session_ids = tab["session_id"].to_numpy()
    folds = _user_coherent_folds(session_ids, n_folds=n_folds, seed=seed)

    q_block = q_proj[:, q_dim_indices].astype(np.float32) if len(q_dim_indices) else None
    t_block = t_proj[:, t_dim_indices].astype(np.float32) if len(t_dim_indices) else None
    toggles = {
        "use_q_pca": q_block is not None,
        "use_t_pca": t_block is not None,
        "use_categorical": bool(selected_categorical),
    }
    X = _build_X(
        tab, q_block, t_block, selected_numeric, selected_categorical, toggles,
    )

    params = {
        "objective": "binary:logistic", "eval_metric": "auc",
        "tree_method": "hist", "device": "cuda" if use_cuda else "cpu",
        "verbosity": 0, "nthread": -1, "seed": seed,
        "learning_rate": 0.1, "max_depth": 2,
        "min_child_weight": 5, "subsample": 0.8,
        "colsample_bytree": 0.8, "reg_alpha": 0.01,
        "reg_lambda": 10.0, "gamma": 0.01,
    }

    rng = np.random.default_rng(seed)
    null_aucs = np.full(n_perms, np.nan, dtype=np.float64)
    for k in range(n_perms):
        y_perm = rng.permutation(y).astype(np.int8)
        fold_aucs: list[float] = []
        for f in range(n_folds):
            val_mask = folds == f
            train_mask = ~val_mask
            y_tr, y_va = y_perm[train_mask], y_perm[val_mask]
            if y_va.sum() == 0 or y_va.sum() == len(y_va):
                continue
            n_pos = int(y_tr.sum())
            n_neg = len(y_tr) - n_pos
            params["scale_pos_weight"] = n_neg / max(n_pos, 1)
            dtr = xgb.DMatrix(X[train_mask], label=y_tr)
            dva = xgb.DMatrix(X[val_mask], label=y_va)
            bst = xgb.train(
                params, dtr, num_boost_round=200,
                evals=[(dva, "val")], early_stopping_rounds=20,
                verbose_eval=False,
            )
            best_iter = int(bst.best_iteration)
            p = bst.predict(dva, iteration_range=(0, best_iter + 1))
            fold_aucs.append(float(roc_auc_score(y_va, p)))
        if fold_aucs:
            null_aucs[k] = float(np.mean(fold_aucs))

    clean = null_aucs[~np.isnan(null_aucs)]
    return {
        "aucs": null_aucs,
        "mu_null": float(np.mean(clean)) if clean.size else 0.5,
        "sigma_null": float(np.std(clean, ddof=1)) if clean.size > 1 else 1e-3,
    }
