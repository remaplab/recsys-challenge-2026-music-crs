"""5-fold user-coherent C2ST evaluation across 3 feature variants × 3 classifiers.

Variants
--------
    V1_pca:  PCA query emb + PCA track emb + tabular (primary)
    V2_umap: UMAP query emb + UMAP track emb + tabular (alt)
    V4_tab:  tabular-only (baseline)

Classifiers per variant: XGBoost (GPU), SGDClassifier, LinearSVC.

User-coherent 5-fold: every (session, aug_id) of the same user stays together.
Class weights balanced via `scale_pos_weight = n_neg / n_pos` per train fold.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

CATEGORICAL_COLS = [
    "age_group", "country_code", "gender", "preferred_language",
    "preferred_musical_culture", "category", "specificity", "top_tag",
]

# Columns that derive from user_profile alone (constant per user). They risk
# inflating C2ST AUC via user-identity memorization rather than capturing
# conversation-distribution shift. Used by V5_pca_nouser variant.
USER_DEMOGRAPHIC_COLS = [
    "age_group", "country_code", "gender", "preferred_language",
    "preferred_musical_culture",
]

# Columns derived from prior_track_ids aggregates. Risk: XGB could memorize
# distinctive tag/artist patterns of blind tracks. Dropping these isolates the
# pure conversational shift in V9/V10 variants.
TRACK_STATS_COLS = [
    "n_prior", "n_unique_artists", "n_unique_albums",
    "pop_mean", "pop_std", "year_mean", "year_std", "year_missing",
    "dur_mean", "dur_std", "tag_entropy", "tag_diversity",
    "prior_empty", "top_tag",
]


VARIANTS = [
    "V1_pca", "V4_tab", "V5_pca_nouser",
    "V6_emb_only", "V7_emb_q_only", "V8_emb_t_only",
    "V9_conv_only", "V10_tab_nouser_notrack",
]


def variant_drops(variant: str) -> tuple[bool, bool]:
    """Return (drop_user, drop_track) for tabular features per variant."""
    if variant in {"V1_pca", "V4_tab"}:
        return False, False
    if variant in {"V5_pca_nouser"}:
        return True, False
    if variant in {"V9_conv_only", "V10_tab_nouser_notrack"}:
        return True, True
    # Pure embedding variants don't use tabular cols — return any flag values
    return True, True


def variant_uses(variant: str) -> tuple[bool, bool, bool]:
    """Return (uses_tabular, uses_q_pca, uses_t_pca) per variant."""
    table = {
        "V1_pca": (True, True, True),
        "V4_tab": (True, False, False),
        "V5_pca_nouser": (True, True, True),
        "V6_emb_only": (False, True, True),
        "V7_emb_q_only": (False, True, False),
        "V8_emb_t_only": (False, False, True),
        "V9_conv_only": (True, True, True),
        "V10_tab_nouser_notrack": (True, False, False),
    }
    return table[variant]


def build_variant_X(
    variant: str,
    tab: pl.DataFrame,
    train_mask: np.ndarray,
    q_red: np.ndarray,
    t_red: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Assemble (X_train, X_val) feature matrices for the given variant."""
    val_mask = ~train_mask
    uses_tab, uses_q, uses_t = variant_uses(variant)
    drop_user, drop_track = variant_drops(variant)

    blocks_tr, blocks_va = [], []
    if uses_tab:
        X_tab_tr, X_tab_va, _ = _tabular_matrix(
            tab, train_mask,
            drop_user_demographics=drop_user, drop_track_stats=drop_track,
        )
        blocks_tr.append(X_tab_tr); blocks_va.append(X_tab_va)
    if uses_q:
        blocks_tr.append(q_red[train_mask]); blocks_va.append(q_red[val_mask])
    if uses_t:
        blocks_tr.append(t_red[train_mask]); blocks_va.append(t_red[val_mask])
    return np.concatenate(blocks_tr, axis=1), np.concatenate(blocks_va, axis=1)


def variant_feature_names(variant: str, pca_dim: int) -> list[str]:
    """Return the feature names (in the same column order as feat_blocks) per variant."""
    def _filt(cols: list[str], drop_user: bool, drop_track: bool) -> list[str]:
        out = []
        for c in cols:
            if drop_user and c in USER_DEMOGRAPHIC_COLS:
                continue
            if drop_track and c in TRACK_STATS_COLS:
                continue
            out.append(c)
        return out

    q_names = [f"q_pca_{i}" for i in range(pca_dim)]
    t_names = [f"t_pca_{i}" for i in range(pca_dim)]

    if variant == "V1_pca":
        return _filt(NUMERIC_COLS, False, False) + _filt(CATEGORICAL_COLS, False, False) + q_names + t_names
    if variant == "V4_tab":
        return _filt(NUMERIC_COLS, False, False) + _filt(CATEGORICAL_COLS, False, False)
    if variant == "V5_pca_nouser":
        return _filt(NUMERIC_COLS, True, False) + _filt(CATEGORICAL_COLS, True, False) + q_names + t_names
    if variant == "V6_emb_only":
        return q_names + t_names
    if variant == "V7_emb_q_only":
        return q_names
    if variant == "V8_emb_t_only":
        return t_names
    if variant == "V9_conv_only":
        return _filt(NUMERIC_COLS, True, True) + _filt(CATEGORICAL_COLS, True, True) + q_names + t_names
    if variant == "V10_tab_nouser_notrack":
        return _filt(NUMERIC_COLS, True, True) + _filt(CATEGORICAL_COLS, True, True)
    raise ValueError(variant)

NUMERIC_COLS = [
    "max_turn", "query_chars_mean", "query_chars_std", "query_words_mean",
    "query_words_std", "qmark_rate", "n_queries_kept", "n_prior",
    "n_unique_artists", "n_unique_albums", "pop_mean", "pop_std",
    "year_mean", "year_std", "year_missing", "dur_mean", "dur_std",
    "tag_entropy", "tag_diversity", "prior_empty",
]


@dataclass
class FoldResult:
    fold: int
    variant: str
    clf: str
    auc: float
    brier: float
    logloss: float
    importances: dict[str, float] = field(default_factory=dict)


# ---------- helpers ----------

def _hash_user_to_fold(user_id: str, n_folds: int, seed: int) -> int:
    h = hashlib.sha256(f"{seed}:{user_id}".encode()).digest()
    return int.from_bytes(h[:8], "big") % n_folds


def _make_folds(tab: pl.DataFrame, n_folds: int, seed: int) -> np.ndarray:
    return np.array(
        [_hash_user_to_fold(u, n_folds, seed) for u in tab["user_id"].to_list()],
        dtype=np.int8,
    )


def _ordinal_encode(train_vals: list[str], val_vals: list[str]) -> tuple[np.ndarray, np.ndarray]:
    mp: dict[str, int] = {"__unk__": 0}
    for v in train_vals:
        if v not in mp:
            mp[v] = len(mp)
    tr = np.array([mp.get(v, 0) for v in train_vals], dtype=np.int32)
    va = np.array([mp.get(v, 0) for v in val_vals], dtype=np.int32)
    return tr, va


def _tabular_matrix(
    tab: pl.DataFrame, train_mask: np.ndarray, *,
    drop_user_demographics: bool = False,
    drop_track_stats: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (X_train_tab, X_val_tab, feature_names) using train fold to fit encoder.

    Optional drops:
      - `drop_user_demographics` removes USER_DEMOGRAPHIC_COLS.
      - `drop_track_stats` removes TRACK_STATS_COLS.
    """
    val_mask = ~train_mask

    def _exclude(c: str) -> bool:
        if drop_user_demographics and c in USER_DEMOGRAPHIC_COLS:
            return True
        if drop_track_stats and c in TRACK_STATS_COLS:
            return True
        return False

    cols_num = [c for c in NUMERIC_COLS if not _exclude(c)]
    cat_cols = [c for c in CATEGORICAL_COLS if not _exclude(c)]
    X_num = tab.select(cols_num).to_numpy().astype(np.float32)
    # NaN guard
    np.nan_to_num(X_num, copy=False, nan=0.0)
    X_train_num = X_num[train_mask]
    X_val_num = X_num[val_mask]

    cat_blocks_train = []
    cat_blocks_val = []
    cat_names: list[str] = []
    for c in cat_cols:
        vals = tab[c].fill_null("__unk__").to_list()
        train_vals = [vals[i] for i in np.where(train_mask)[0]]
        val_vals = [vals[i] for i in np.where(val_mask)[0]]
        tr_enc, va_enc = _ordinal_encode(train_vals, val_vals)
        cat_blocks_train.append(tr_enc.reshape(-1, 1))
        cat_blocks_val.append(va_enc.reshape(-1, 1))
        cat_names.append(c)

    X_train = np.concatenate([X_train_num] + cat_blocks_train, axis=1)
    X_val = np.concatenate([X_val_num] + cat_blocks_val, axis=1)
    return X_train, X_val, cols_num + cat_names


def _reduce_full(emb: np.ndarray, method: str, dim: int, seed: int) -> np.ndarray:
    """Fit dim-reduction on the full matrix once (unsupervised, no label leakage).

    UMAP: pre-PCA to 128 dims, then UMAP without random_state so n_jobs=-1
    actually parallelizes (umap-learn forces single-threaded when random_state set).
    """
    if method == "none":
        return emb
    if method == "pca":
        red = PCA(n_components=dim, random_state=seed)
        return red.fit_transform(emb).astype(np.float32)
    if method == "umap":
        import umap

        pre_dim = min(128, emb.shape[1])
        pre = PCA(n_components=pre_dim, random_state=seed).fit_transform(emb)
        # `init="random"` skips the spectral layout (which fails on multi-component
        # graphs caused by the zero-vector cluster from max_turn=0 sessions).
        red = umap.UMAP(
            n_components=dim, n_neighbors=15, n_jobs=-1,
            low_memory=False, init="random",
        )
        return red.fit_transform(pre).astype(np.float32)
    raise ValueError(method)


# ---------- single-fold trainer ----------

def _fit_predict(
    name: str, X_tr: np.ndarray, y_tr: np.ndarray, X_va: np.ndarray, *, seed: int
) -> tuple[np.ndarray, dict[str, float]]:
    n_pos = int(y_tr.sum())
    n_neg = len(y_tr) - n_pos
    spw = n_neg / max(n_pos, 1)

    if name == "xgb":
        dtr = xgb.DMatrix(X_tr, label=y_tr)
        dva = xgb.DMatrix(X_va)
        params = {
            "objective": "binary:logistic", "eval_metric": "auc",
            "learning_rate": 0.1, "max_depth": 6, "min_child_weight": 1,
            "scale_pos_weight": spw, "reg_alpha": 0.1, "reg_lambda": 0.1,
            "seed": seed, "verbosity": 0,
            "tree_method": "hist", "device": "cuda", "nthread": -1,
        }
        bst = xgb.train(params, dtr, num_boost_round=150)
        p = bst.predict(dva)
        score = bst.get_score(importance_type="gain")
        imp = {k.replace("f", ""): float(v) for k, v in score.items()}
        return p, imp

    if name == "sgd":
        sc = StandardScaler(with_mean=True, with_std=True)
        Xt = sc.fit_transform(X_tr)
        Xv = sc.transform(X_va)
        m = SGDClassifier(
            loss="log_loss", penalty="elasticnet", l1_ratio=0.15,
            alpha=1e-4, class_weight="balanced", max_iter=250, tol=1e-4,
            random_state=seed, n_jobs=-1,
        )
        m.fit(Xt, y_tr)
        p = m.predict_proba(Xv)[:, 1]
        imp = {str(i): float(abs(v)) for i, v in enumerate(m.coef_.ravel())}
        return p, imp

    if name == "svc":
        sc = StandardScaler(with_mean=True, with_std=True)
        Xt = sc.fit_transform(X_tr)
        Xv = sc.transform(X_va)
        m = LinearSVC(
            C=0.5, class_weight="balanced", dual="auto",
            max_iter=2000, random_state=seed,
        )
        m.fit(Xt, y_tr)
        # decision_function → sigmoid for proba-like score; AUC is rank-invariant
        s = m.decision_function(Xv)
        p = 1.0 / (1.0 + np.exp(-s))
        imp = {str(i): float(abs(v)) for i, v in enumerate(m.coef_.ravel())}
        return p, imp

    raise ValueError(name)


# ---------- main loop ----------

def run_c2st(
    tab: pl.DataFrame,
    q_emb: np.ndarray,
    t_emb: np.ndarray,
    *,
    n_folds: int,
    seed: int,
    pca_dim: int,
    umap_dim: int,
    out_dir: Path,
) -> tuple[pl.DataFrame, dict[str, np.ndarray]]:
    """Return (results_df with per-fold AUC/brier/logloss/importances per (variant,clf),
    oof_preds dict keyed `f"{variant}__{clf}"` → array of length len(tab))."""
    folds = _make_folds(tab, n_folds=n_folds, seed=seed)
    y = tab["label"].to_numpy().astype(np.int8)

    # Variants — XGB only (linear / kNN proved unreliable on prior runs).
    # Suffix _nouser: drop user demographics. _emb: pure PCA embeddings, no tabular.
    variants = [
        "V1_pca",            # PCA q + PCA t + ALL tabular (baseline, prone to user-id memorization)
        "V4_tab",            # tabular only (no embeddings)
        "V5_pca_nouser",     # PCA q + PCA t + tabular minus user demographics (recommended primary)
        "V6_emb_only",       # PCA q + PCA t only (no tabular at all)
        "V7_emb_q_only",     # PCA q only
        "V8_emb_t_only",     # PCA t only
        "V9_conv_only",      # PCA q + PCA t + (tab minus demographics AND track stats)
        "V10_tab_nouser_notrack",  # tabular minus demographics minus track stats
    ]
    classifiers_per_variant = {v: ["xgb"] for v in variants}

    results: list[FoldResult] = []
    oof: dict[str, np.ndarray] = {
        f"{v}__{c}": np.full(len(y), np.nan, dtype=np.float32)
        for v in variants
        for c in classifiers_per_variant[v]
    }

    print("[reduce] fitting PCA on q_emb…")
    q_pca = _reduce_full(q_emb, "pca", pca_dim, seed)
    print("[reduce] fitting PCA on t_emb…")
    t_pca = _reduce_full(t_emb, "pca", pca_dim, seed)
    del umap_dim  # kept in signature for backwards compatibility; UMAP only used for the 2-D viz plot now.

    for fold in range(n_folds):
        val_mask = folds == fold
        train_mask = ~val_mask
        y_tr, y_va = y[train_mask], y[val_mask]
        print(f"[fold {fold}] train n={train_mask.sum()} (pos={int(y_tr.sum())}) "
              f"val n={val_mask.sum()} (pos={int(y_va.sum())})")

        X_tab_tr, X_tab_va, _ = _tabular_matrix(tab, train_mask)
        X_tab_tr_nu, X_tab_va_nu, _ = _tabular_matrix(
            tab, train_mask, drop_user_demographics=True
        )
        X_tab_tr_nutk, X_tab_va_nutk, _ = _tabular_matrix(
            tab, train_mask, drop_user_demographics=True, drop_track_stats=True
        )

        q_tr, q_va = q_pca[train_mask], q_pca[val_mask]
        t_tr, t_va = t_pca[train_mask], t_pca[val_mask]

        feat_blocks = {
            "V1_pca": (
                np.concatenate([X_tab_tr, q_tr, t_tr], axis=1),
                np.concatenate([X_tab_va, q_va, t_va], axis=1),
            ),
            "V4_tab": (X_tab_tr, X_tab_va),
            "V5_pca_nouser": (
                np.concatenate([X_tab_tr_nu, q_tr, t_tr], axis=1),
                np.concatenate([X_tab_va_nu, q_va, t_va], axis=1),
            ),
            "V6_emb_only": (
                np.concatenate([q_tr, t_tr], axis=1),
                np.concatenate([q_va, t_va], axis=1),
            ),
            "V7_emb_q_only": (q_tr, q_va),
            "V8_emb_t_only": (t_tr, t_va),
            "V9_conv_only": (
                np.concatenate([X_tab_tr_nutk, q_tr, t_tr], axis=1),
                np.concatenate([X_tab_va_nutk, q_va, t_va], axis=1),
            ),
            "V10_tab_nouser_notrack": (X_tab_tr_nutk, X_tab_va_nutk),
        }

        for v in variants:
            X_tr, X_va = feat_blocks[v]
            for c in classifiers_per_variant[v]:
                p, imp = _fit_predict(c, X_tr, y_tr, X_va, seed=seed + fold)
                if y_va.sum() == 0 or y_va.sum() == len(y_va):
                    auc = float("nan")
                else:
                    auc = float(roc_auc_score(y_va, p))
                br = float(brier_score_loss(y_va, p))
                ll = float(log_loss(y_va, np.clip(p, 1e-6, 1 - 1e-6)))
                results.append(FoldResult(fold=fold, variant=v, clf=c,
                                          auc=auc, brier=br, logloss=ll, importances=imp))
                oof[f"{v}__{c}"][val_mask] = p
                print(f"  [{v} / {c}] AUC={auc:.4f}  brier={br:.4f}  ll={ll:.4f}")

    res_df = pl.DataFrame([{
        "fold": r.fold, "variant": r.variant, "clf": r.clf,
        "auc": r.auc, "brier": r.brier, "logloss": r.logloss,
    } for r in results])

    # Save OOF and per-fold importances
    out_dir.mkdir(parents=True, exist_ok=True)
    res_df.write_parquet(out_dir / "fold_metrics.parquet")
    np.savez(out_dir / "oof_predictions.npz", **oof, label=y)

    importances_serializable = [
        {"fold": r.fold, "variant": r.variant, "clf": r.clf, "importances": r.importances}
        for r in results
    ]
    (out_dir / "fold_importances.json").write_text(json.dumps(importances_serializable))

    return res_df, oof


def aggregate_results(res_df: pl.DataFrame, oof: dict[str, np.ndarray], y: np.ndarray,
                     *, n_boot: int, seed: int) -> dict:
    """Mean ± bootstrap CI per (variant, clf), plus bootstrap CI on pooled OOF AUC."""
    rng = np.random.default_rng(seed)
    summary: dict = {"per_variant_clf": {}, "note": (
        "Primary headline is fold_auc_mean (not pooled_oof_auc). Pooled AUC across "
        "folds can be unstable when per-fold score scales differ (different best_iter); "
        "pooled_oof_auc_rank uses within-fold rank normalisation before pooling."
    )}
    # Need fold ids to compute rank-normalised pooled per (variant, clf).
    fold_arr = res_df.sort(["variant", "clf", "fold"])["fold"].to_numpy()  # unused below; we use oof directly
    folds_per_row = None  # computed below per row of OOF using the same _make_folds order
    for (variant, clf), grp in res_df.group_by(["variant", "clf"], maintain_order=True):
        aucs = grp["auc"].to_numpy()
        # Oriented AUC: classifier label assignment can flip under heavy imbalance
        # + balanced class weights. For a C2ST only |AUC - 0.5| carries shift signal.
        oriented = np.maximum(aucs, 1.0 - aucs)
        key = f"{variant}__{clf}"
        preds = oof[key]
        mask = ~np.isnan(preds)
        y_eval = y[mask]
        p_eval = preds[mask]
        if y_eval.sum() == 0 or y_eval.sum() == len(y_eval):
            pooled_auc = float("nan")
            pooled_oriented = float("nan")
            ci_lo = ci_hi = float("nan")
            ci_lo_o = ci_hi_o = float("nan")
        else:
            pooled_auc = float(roc_auc_score(y_eval, p_eval))
            pooled_oriented = max(pooled_auc, 1.0 - pooled_auc)
            boots = np.empty(n_boot, dtype=np.float64)
            n = len(y_eval)
            for b in range(n_boot):
                idx = rng.integers(0, n, n)
                try:
                    boots[b] = roc_auc_score(y_eval[idx], p_eval[idx])
                except ValueError:
                    boots[b] = np.nan
            boots_oriented = np.maximum(boots, 1.0 - boots)
            ci_lo = float(np.nanpercentile(boots, 2.5))
            ci_hi = float(np.nanpercentile(boots, 97.5))
            ci_lo_o = float(np.nanpercentile(boots_oriented, 2.5))
            ci_hi_o = float(np.nanpercentile(boots_oriented, 97.5))
        summary["per_variant_clf"][key] = {
            "fold_auc_mean": float(np.nanmean(aucs)),
            "fold_auc_std": float(np.nanstd(aucs)),
            "pooled_oof_auc": pooled_auc,
            "pooled_auc_ci95": [ci_lo, ci_hi],
            "fold_auc_min": float(np.nanmin(aucs)),
            "fold_auc_max": float(np.nanmax(aucs)),
            "fold_oriented_auc_mean": float(np.nanmean(oriented)),
            "fold_oriented_auc_std": float(np.nanstd(oriented)),
            "pooled_oriented_auc": pooled_oriented,
            "pooled_oriented_ci95": [ci_lo_o, ci_hi_o],
        }
    return summary


def permutation_null(
    tab: pl.DataFrame, q_emb: np.ndarray, t_emb: np.ndarray,
    *, variant: str, clf: str, n_perms: int, n_folds: int, seed: int, pca_dim: int,
) -> tuple[float, np.ndarray]:  # noqa: ARG001 — variant unused (always V1_pca-like)
    """Compute the null AUC distribution by shuffling labels n_perms times.
    Returns (observed_auc_mean, null_aucs)."""
    folds = _make_folds(tab, n_folds=n_folds, seed=seed)
    y_orig = tab["label"].to_numpy().astype(np.int8)

    # Pre-reduce once (unsupervised), reused across perms
    qp = _reduce_full(q_emb, "pca", pca_dim, seed)
    tp = _reduce_full(t_emb, "pca", pca_dim, seed)

    def _one(y: np.ndarray) -> float:
        aucs = []
        for f in range(n_folds):
            val = folds == f
            tr = ~val
            X_tab_tr, X_tab_va, _ = _tabular_matrix(tab, tr)
            X_tr = np.concatenate([X_tab_tr, qp[tr], tp[tr]], axis=1)
            X_va = np.concatenate([X_tab_va, qp[val], tp[val]], axis=1)
            p, _ = _fit_predict(clf, X_tr, y[tr], X_va, seed=seed + f)
            y_va = y[val]
            if y_va.sum() and y_va.sum() < len(y_va):
                aucs.append(roc_auc_score(y_va, p))
        return float(np.nanmean(aucs)) if aucs else float("nan")

    observed = _one(y_orig)
    nulls = np.empty(n_perms, dtype=np.float64)
    rng = np.random.default_rng(seed + 999)
    for k in range(n_perms):
        y_perm = rng.permutation(y_orig)
        nulls[k] = _one(y_perm)
        print(f"  [perm {k}] null AUC = {nulls[k]:.4f}")
    return observed, nulls
