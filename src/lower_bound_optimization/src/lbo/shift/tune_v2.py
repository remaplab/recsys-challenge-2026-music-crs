"""Optuna study v2 — unified feature set with PCA-selected emb dims.

Replaces V1..V10 grid. Embedding PCA projection + dim selection happen
upstream (in the launcher / feature_select); this module operates only on
already-projected and BH-selected `q_proj`/`t_proj` columns.

Persists per-trial OOF logits for §3 (multi-comp) and §4 (calibration).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import optuna
import polars as pl
import xgboost as xgb
from sklearn.metrics import roc_auc_score


@dataclass
class TuneV2Artifacts:
    trials: list[dict] = field(default_factory=list)
    study: optuna.Study | None = None
    permutation_null: dict | None = None


def _user_coherent_folds(
    session_ids: np.ndarray, n_folds: int, seed: int,
) -> np.ndarray:
    """Hash-based fold assignment so identical session_ids share a fold."""
    rng = np.random.default_rng(seed)
    unique = np.unique(session_ids)
    permuted = rng.permutation(unique)
    bucket = {s: i % n_folds for i, s in enumerate(permuted)}
    return np.array([bucket[s] for s in session_ids], dtype=np.int32)


def _categorical_to_codes(s: pl.Series) -> np.ndarray:
    """Stable integer codes for a string categorical column."""
    return s.cast(pl.Categorical).to_physical().to_numpy().astype(np.int32)


def _build_X(
    tab: pl.DataFrame,
    q_block: np.ndarray | None,
    t_block: np.ndarray | None,
    selected_numeric: list[str],
    selected_categorical: list[str],
    toggles: dict,
) -> np.ndarray:
    """Assemble feature matrix from precomputed blocks.

    `q_block` / `t_block` are the BH-selected PCA-projected columns
    (shape (N, n_selected_dims)) — fully prepared by the launcher; this
    function only respects the boolean group toggles.
    """
    blocks: list[np.ndarray] = []
    if selected_numeric:
        blocks.append(
            tab.select(selected_numeric).fill_null(0).to_numpy().astype(np.float32)
        )
    if toggles.get("use_categorical", True) and selected_categorical:
        cat_block = np.column_stack([
            _categorical_to_codes(tab[c]) for c in selected_categorical
        ]).astype(np.float32)
        blocks.append(cat_block)
    if toggles.get("use_q_pca", True) and q_block is not None and q_block.shape[1] > 0:
        blocks.append(q_block.astype(np.float32))
    if toggles.get("use_t_pca", True) and t_block is not None and t_block.shape[1] > 0:
        blocks.append(t_block.astype(np.float32))
    if not blocks:
        return np.zeros((tab.height, 1), dtype=np.float32)
    return np.column_stack(blocks).astype(np.float32)


def tune_v2(
    *,
    tab: pl.DataFrame,
    q_proj: np.ndarray,
    t_proj: np.ndarray,
    q_dim_indices: list[int],
    t_dim_indices: list[int],
    selected_numeric: list[str],
    selected_categorical: list[str],
    n_trials: int,
    n_folds: int,
    seed: int,
    oof_dir: Path,
    use_cuda: bool = True,
    study_name: str = "c2st_v2",
    storage: str | None = None,
    n_jobs: int = 1,
) -> TuneV2Artifacts:
    """Single Optuna study over XGB HPs + 3 group toggles.

    PCA dim selection is fixed upstream (q_dim_indices, t_dim_indices) — Optuna
    only toggles whether each block is used wholesale.
    """
    oof_dir = Path(oof_dir)
    oof_dir.mkdir(parents=True, exist_ok=True)

    y = tab["label"].to_numpy().astype(np.int8)
    session_ids = tab["session_id"].to_numpy()
    folds = _user_coherent_folds(session_ids, n_folds=n_folds, seed=seed)

    q_block = q_proj[:, q_dim_indices].astype(np.float32) if len(q_dim_indices) else None
    t_block = t_proj[:, t_dim_indices].astype(np.float32) if len(t_dim_indices) else None

    trials_log: list[dict] = []

    def objective(trial: optuna.Trial) -> float:
        toggles = {
            "use_q_pca": (
                trial.suggest_categorical("use_q_pca", [True, False])
                if q_block is not None else False
            ),
            "use_t_pca": (
                trial.suggest_categorical("use_t_pca", [True, False])
                if t_block is not None else False
            ),
            "use_categorical": (
                trial.suggest_categorical("use_categorical", [True, False])
                if selected_categorical else False
            ),
        }
        params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "tree_method": "hist",
            "device": "cuda" if use_cuda else "cpu",
            "verbosity": 0,
            "nthread": -1,
            "seed": seed,
            "learning_rate": trial.suggest_float("learning_rate", 1e-1, 3e-1, log=True),
            "max_depth": trial.suggest_int("max_depth", 1, 8),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 100),
            "subsample": trial.suggest_float("subsample", 0.1, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 1e-4, 5.0, log=True),
        }
        n_rounds = trial.suggest_int("n_estimators", 100, 1000)

        X = _build_X(
            tab, q_block, t_block, selected_numeric, selected_categorical, toggles,
        )

        oof_logits = np.full(len(y), np.nan, dtype=np.float32)
        aucs: list[float] = []
        for f in range(n_folds):
            val_mask = folds == f
            train_mask = ~val_mask
            y_tr, y_va = y[train_mask], y[val_mask]
            if y_va.sum() == 0 or y_va.sum() == len(y_va):
                continue

            X_tr, X_va = X[train_mask], X[val_mask]

            n_pos = int(y_tr.sum())
            n_neg = len(y_tr) - n_pos
            params["scale_pos_weight"] = n_neg / max(n_pos, 1)

            dtr = xgb.DMatrix(X_tr, label=y_tr)
            dva = xgb.DMatrix(X_va, label=y_va)
            bst = xgb.train(
                params, dtr, num_boost_round=n_rounds,
                evals=[(dva, "val")],
                early_stopping_rounds=30,
                verbose_eval=False,
            )
            best_iter = int(bst.best_iteration)
            p = bst.predict(dva, iteration_range=(0, best_iter + 1))
            p_clip = np.clip(p, 1e-7, 1.0 - 1e-7)
            oof_logits[val_mask] = np.log(p_clip / (1.0 - p_clip)).astype(np.float32)
            aucs.append(float(roc_auc_score(y_va, p)))

        if not aucs:
            return 0.5

        mean_auc = float(np.mean(aucs))
        oof_path = oof_dir / f"trial_{trial.number:04d}.npy"
        np.save(oof_path, oof_logits)

        trials_log.append({
            "trial_idx": int(trial.number),
            "params": dict(trial.params),
            "mean_auc": mean_auc,
            "fold_aucs": np.array(aucs, dtype=np.float64),
            "oof_logits_path": str(oof_path),
        })
        return mean_auc

    sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True)
    pruner = optuna.pruners.HyperbandPruner(
        min_resource=2, max_resource=n_folds, reduction_factor=3,
    )
    study = optuna.create_study(
        direction="maximize", sampler=sampler, pruner=pruner,
        study_name=study_name, storage=storage,
        load_if_exists=storage is not None,
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False, n_jobs=n_jobs)

    return TuneV2Artifacts(trials=trials_log, study=study)
