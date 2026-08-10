"""s06 — v2 C2ST density-ratio pipeline.

Steps
-----
§1  Build augmented feature matrix; canonicalize row order; user-coherent
    two-way split into CV vs PoSI holdout; PCA-project q/t embeddings on full
    data (unsupervised).
§2  Optuna study over XGB HPs + group toggles (use_q_pca, use_t_pca,
    use_categorical) on CV rows; per-trial OOF logits persisted.
§3  Permutation null on CV → shared (mu_null, sigma_null); Holm-Bonferroni
    over trial AUCs; paired empirical-Bernstein dedup; PoSI data-split AUC
    on the held-out 20%.
§4  Per-trial calibration (isotonic + Platt fallback) of OOF logits; uniform
    average across survivors.
§5  Density-ratio weights (clipped + Group-DRO bucket) for the reranker.

No pre-tuning feature-selection step:  all tabular + all PCA dims pass to
Optuna; the group toggles + PoSI gating prune dishonest configurations
post-hoc.  User-derived columns (USER_DEMOGRAPHIC_COLS) and track-aggregate
columns (TRACK_STATS_COLS) are dropped from the pool by default because in
the cold-user split they leak user identity.

Run:
    PYTHONPATH=src uv run python -m launchers.s06_distribution_shift_v2 --seed 42
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _PKG_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import numpy as np
import polars as pl
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

from lbo.paths import SHIFT_V2_OUT
from lbo.shift.calibration import (
    ensemble_probabilities,
    expected_calibration_error,
    fit_calibrator,
)
from lbo.shift.classifiers import (
    CATEGORICAL_COLS,
    NUMERIC_COLS,
    TRACK_STATS_COLS,
    USER_DEMOGRAPHIC_COLS,
)
from lbo.shift.features import build_feature_matrix
from lbo.shift.holdout import _user_hash_unit, summary as holdout_summary
from lbo.shift.multi_comp import (
    compute_permutation_null,
    deduplicate_by_paired_eb,
    holm_bonferroni,
    z_pvalues_against_null,
)
from lbo.shift.tune_v2 import _build_X, tune_v2
from lbo.shift.weights import aggregate_per_session, build_weight_table


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------

def compute_holdout_mask(
    tab: pl.DataFrame, *, posi_frac: float, seed: int,
) -> np.ndarray:
    """User-coherent, per-source stratified holdout mask (length = tab.height).

    Every row of the same user lands on the same side. Hash is salted with
    `seed` via `_user_hash_unit`, so two runs with the same seed produce the
    same partition.
    """
    assert 0.0 < posi_frac < 1.0
    holdout_users: set[str] = set()
    for source in tab["source"].unique().to_list():
        users = tab.filter(pl.col("source") == source)["user_id"].unique().to_list()
        for u in users:
            if _user_hash_unit(u, seed) < posi_frac:
                holdout_users.add(u)
    user_arr = tab["user_id"].to_list()
    return np.array([u in holdout_users for u in user_arr], dtype=bool)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_augs", type=int, default=5)
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--n_trials", type=int, default=200)
    p.add_argument("--n_perms", type=int, default=50)
    p.add_argument("--holm_alpha", type=float, default=0.05)
    p.add_argument("--clip_q", type=float, default=0.95)
    p.add_argument("--posi_frac", type=float, default=0.2,
                   help="User-coherent holdout used only for PoSI data-split AUC.")
    p.add_argument("--pca_max", type=int, default=256,
                   help="PCA components fit on each of q_emb / t_emb. "
                        "All survive into Optuna's feature pool.")
    p.add_argument("--drop_user_features", action="store_true", default=True,
                   help="Exclude USER_DEMOGRAPHIC_COLS — in cold-user split they "
                        "near-perfectly identify blind rows.")
    p.add_argument("--keep_user_features", dest="drop_user_features",
                   action="store_false")
    p.add_argument("--drop_track_stats", action="store_true", default=True,
                   help="Exclude TRACK_STATS_COLS — prior_track_ids aggregates "
                        "proxy for user identity under cold-user split.")
    p.add_argument("--keep_track_stats", dest="drop_track_stats",
                   action="store_false")
    p.add_argument("--use_cuda", action="store_true", default=True)
    p.add_argument("--no_cuda", dest="use_cuda", action="store_false")
    p.add_argument("--out_dir", type=Path, default=SHIFT_V2_OUT)
    p.add_argument("--reset_study", action="store_true",
                   help="Delete prior study.db and oof_logits/*.npy before tuning.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    out = args.out_dir
    (out / "plots").mkdir(parents=True, exist_ok=True)
    (out / "oof_logits").mkdir(parents=True, exist_ok=True)
    (out / "calibration").mkdir(parents=True, exist_ok=True)

    if args.reset_study:
        study_db = out / "study.db"
        if study_db.exists():
            study_db.unlink()
            print(f"[s06] deleted prior study {study_db}")
        for f in (out / "oof_logits").glob("trial_*.npy"):
            f.unlink()

    t0 = time.time()

    # ── §1 build features + split + PCA ─────────────────────────────────────
    print(f"[s06] §1 — building feature matrix "
          f"(n_augs={args.n_augs}, seed={args.seed})…")
    tab, q_emb, t_emb = build_feature_matrix(n_augs=args.n_augs, seed=args.seed)

    # Canonicalize row order. build_feature_matrix is content-deterministic per
    # (session_id, aug_id) but row order is not stable, and PCA's randomized
    # SVD is row-order-sensitive → without this sort the basis rotates between
    # runs.
    canon_order = np.lexsort((
        tab["aug_id"].to_numpy(),
        tab["session_id"].to_numpy(),
    ))
    tab = tab[canon_order.tolist()]
    q_emb = q_emb[canon_order]
    t_emb = t_emb[canon_order]

    # Two-way user-coherent split: posi holdout vs CV.
    ho_mask = compute_holdout_mask(tab, posi_frac=args.posi_frac, seed=args.seed)
    cv_mask = ~ho_mask
    tab = tab.with_columns(
        pl.Series("is_holdout", ho_mask),
        pl.Series("is_cv", cv_mask),
    )
    tab.write_parquet(out / "features_v2.parquet")
    np.save(out / "is_holdout_mask.npy", ho_mask)
    np.save(out / "is_cv_mask.npy", cv_mask)
    (out / "splits_summary.json").write_text(json.dumps({
        "posi_holdout": holdout_summary(tab, ho_mask),
        "cv": holdout_summary(tab, cv_mask),
    }, indent=2))
    print(f"[s06] §1 — splits: cv={int(cv_mask.sum())}, "
          f"posi={int(ho_mask.sum())} (of {tab.height})")

    # PCA fit on FULL embeddings (unsupervised → no label leak).
    print(f"[s06] §1 — PCA (pca_max={args.pca_max})…")
    pca_max_q = min(args.pca_max, q_emb.shape[1])
    pca_max_t = min(args.pca_max, t_emb.shape[1])
    q_proj = PCA(n_components=pca_max_q, random_state=args.seed) \
        .fit_transform(q_emb).astype(np.float32)
    t_proj = PCA(n_components=pca_max_t, random_state=args.seed) \
        .fit_transform(t_emb).astype(np.float32)

    # Feature pool: all tabular minus identity-proxy columns, all PCA dims.
    excluded: set[str] = set()
    if args.drop_user_features:
        excluded |= set(USER_DEMOGRAPHIC_COLS)
    if args.drop_track_stats:
        excluded |= set(TRACK_STATS_COLS)
    selected_numeric = [c for c in NUMERIC_COLS if c not in excluded]
    selected_categorical = [c for c in CATEGORICAL_COLS if c not in excluded]
    q_dim_indices = list(range(q_proj.shape[1]))
    t_dim_indices = list(range(t_proj.shape[1]))
    print(
        f"[s06] §1 — feature pool: "
        f"numeric={len(selected_numeric)}, categorical={len(selected_categorical)}, "
        f"q_pca={len(q_dim_indices)}, t_pca={len(t_dim_indices)} "
        f"(excluded {len(excluded)} cols: {sorted(excluded)})"
    )
    (out / "features_used.json").write_text(json.dumps({
        "selected_numeric": selected_numeric,
        "selected_categorical": selected_categorical,
        "q_pca_dim_indices": q_dim_indices,
        "t_pca_dim_indices": t_dim_indices,
        "excluded": sorted(excluded),
        "pca_max_q": pca_max_q,
        "pca_max_t": pca_max_t,
    }, indent=2))

    # CV-only views.
    tab_cv = tab.filter(pl.col("is_cv"))
    q_proj_cv = q_proj[cv_mask]
    t_proj_cv = t_proj[cv_mask]
    y_full = tab["label"].to_numpy().astype(np.int8)
    y_cv = y_full[cv_mask]

    # ── §2 Optuna study (CV rows only) ──────────────────────────────────────
    print(f"[s06] §2 — Optuna study, n_trials={args.n_trials}…")
    art = tune_v2(
        tab=tab_cv,
        q_proj=q_proj_cv,
        t_proj=t_proj_cv,
        q_dim_indices=q_dim_indices,
        t_dim_indices=t_dim_indices,
        selected_numeric=selected_numeric,
        selected_categorical=selected_categorical,
        n_trials=args.n_trials,
        n_folds=args.n_folds,
        seed=args.seed,
        oof_dir=out / "oof_logits",
        use_cuda=args.use_cuda,
        study_name="c2st_v2",
        storage=f"sqlite:///{out / 'study.db'}",
        n_jobs=4,
    )

    # ── §3 permutation null + Holm + EB-dedup ───────────────────────────────
    print(f"[s06] §3 — permutation null, n_perms={args.n_perms}…")
    null = compute_permutation_null(
        tab=tab_cv,
        q_proj=q_proj_cv,
        t_proj=t_proj_cv,
        q_dim_indices=q_dim_indices,
        t_dim_indices=t_dim_indices,
        selected_numeric=selected_numeric,
        selected_categorical=selected_categorical,
        n_perms=args.n_perms,
        n_folds=args.n_folds,
        seed=args.seed,
        use_cuda=args.use_cuda,
    )
    np.save(out / "permutation_null.npy", null["aucs"])

    aucs = np.array([t["mean_auc"] for t in art.trials])
    trial_indices = [t["trial_idx"] for t in art.trials]
    pvals = z_pvalues_against_null(
        aucs, mu_null=null["mu_null"], sigma_null=null["sigma_null"],
    )
    holm_mask = holm_bonferroni(pvals, alpha=args.holm_alpha)
    holm_survivors = [trial_indices[i] for i in range(len(trial_indices))
                      if holm_mask[i]]

    if not holm_survivors:
        print("[s06] §3 — no Holm survivors. Writing uniform weights and exiting.")
        per_session = (
            tab.filter(pl.col("label") == 0)
            .group_by("session_id")
            .agg(pl.lit(1.0).cast(pl.Float64).alias("weight_mean"))
        )
        per_session.write_parquet(out / "density_ratio.parquet")
        (out / "holm_survivors.json").write_text(json.dumps({
            "mu_null": null["mu_null"],
            "sigma_null": null["sigma_null"],
            "holm_alpha": args.holm_alpha,
            "n_trials": len(trial_indices),
            "holm_survivors": [],
            "eb_deduped": [],
        }, indent=2))
        return

    fold_aucs_by_trial = {t["trial_idx"]: t["fold_aucs"] for t in art.trials}
    s_final = deduplicate_by_paired_eb(
        holm_survivors, fold_aucs_by_trial, delta=args.holm_alpha,
    )
    print(f"[s06] §3 — Holm survivors: {len(holm_survivors)}, "
          f"EB-deduped: {len(s_final)}")
    (out / "holm_survivors.json").write_text(json.dumps({
        "mu_null": null["mu_null"],
        "sigma_null": null["sigma_null"],
        "holm_alpha": args.holm_alpha,
        "n_trials": len(trial_indices),
        "holm_survivors": holm_survivors,
        "eb_deduped": s_final,
    }, indent=2))

    # ── §3b PoSI data-split AUC + full-tab prediction ───────────────────────
    # For each survivor: refit on CV rows with frozen HPs, AUC on PoSI holdout
    # (honest, never seen in tuning), AND predict on full tab so §4 calibration
    # can score every row.
    print("[s06] §3 — PoSI AUC on holdout + full-data prediction…")
    posi: dict[str, float] = {}
    full_logits_per_trial: dict[int, np.ndarray] = {}
    q_block_full = q_proj[:, q_dim_indices].astype(np.float32) if q_dim_indices else None
    t_block_full = t_proj[:, t_dim_indices].astype(np.float32) if t_dim_indices else None
    survivor_trials = [t for t in art.trials if t["trial_idx"] in s_final]
    for t in survivor_trials:
        params = dict(t["params"])
        n_rounds = params.pop("n_estimators", 500)
        toggles = {
            "use_q_pca": params.pop("use_q_pca", True),
            "use_t_pca": params.pop("use_t_pca", True),
            "use_categorical": params.pop("use_categorical", True),
        }
        params.update({
            "objective": "binary:logistic", "eval_metric": "auc",
            "tree_method": "hist",
            "device": "cuda" if args.use_cuda else "cpu",
            "verbosity": 0, "nthread": -1, "seed": args.seed,
        })
        X_full = _build_X(
            tab, q_block_full, t_block_full,
            selected_numeric, selected_categorical, toggles,
        )
        X_cv_arr = X_full[cv_mask]
        X_ho_arr = X_full[ho_mask]
        y_ho_arr = y_full[ho_mask]
        n_pos = int(y_cv.sum())
        n_neg = len(y_cv) - n_pos
        params["scale_pos_weight"] = n_neg / max(n_pos, 1)
        bst = xgb.train(
            params,
            xgb.DMatrix(X_cv_arr, label=y_cv),
            num_boost_round=n_rounds,
            evals=[(xgb.DMatrix(X_ho_arr, label=y_ho_arr), "val")],
            early_stopping_rounds=30,
            verbose_eval=False,
        )
        best_iter = int(bst.best_iteration)
        p_ho = bst.predict(
            xgb.DMatrix(X_ho_arr), iteration_range=(0, best_iter + 1),
        )
        posi[str(t["trial_idx"])] = float(roc_auc_score(y_ho_arr, p_ho))

        p_full = bst.predict(
            xgb.DMatrix(X_full), iteration_range=(0, best_iter + 1),
        )
        p_clip = np.clip(p_full, 1e-7, 1.0 - 1e-7)
        full_logits_per_trial[t["trial_idx"]] = np.log(
            p_clip / (1.0 - p_clip)
        ).astype(np.float32)
    (out / "posi_auc.json").write_text(json.dumps(posi, indent=2))

    # ── §4 calibration ──────────────────────────────────────────────────────
    # Fit calibrator on (OOF prob, y_cv) per trial; apply to full-tab logits
    # from §3b; ensemble via uniform average.
    print("[s06] §4 — calibration…")
    calibrators: dict[int, dict] = {}
    calibrated_per_trial: list[np.ndarray] = []
    for t in survivor_trials:
        oof_logits_cv = np.load(t["oof_logits_path"])
        finite = np.isfinite(oof_logits_cv)
        p_oof = 1.0 / (1.0 + np.exp(-oof_logits_cv[finite]))
        y_oof = y_cv[finite]
        cal, kind, ece = fit_calibrator(p_oof, y_oof)

        full_logits = full_logits_per_trial[t["trial_idx"]]
        p_full_uncal = 1.0 / (1.0 + np.exp(-full_logits))
        p_cal_full = cal(p_full_uncal)
        calibrated_per_trial.append(p_cal_full)
        calibrators[int(t["trial_idx"])] = {"kind": kind, "ece": float(ece)}

    p_ens = ensemble_probabilities(calibrated_per_trial)
    ens_ece = expected_calibration_error(p_ens[cv_mask], y_cv)
    (out / "calibration" / "report.json").write_text(json.dumps({
        "per_trial": calibrators,
        "ensemble_ece": float(ens_ece),
    }, indent=2))

    # ── §5 density-ratio weights + per-aug table ────────────────────────────
    print("[s06] §5 — density-ratio weights…")
    n_train = int((tab["label"] == 0).sum())
    n_blind = int((tab["label"] == 1).sum())
    df_for_weights = (
        tab.with_columns(pl.Series("ensemble_prob", p_ens))
        .filter(pl.col("label") == 0)
        .with_columns(
            pl.int_range(0, pl.len()).over("session_id").alias("aug_id"),
        )
    )
    per_aug = build_weight_table(
        df_for_weights,
        n_train=n_train,
        n_blind=n_blind,
        clip_q=args.clip_q,
    )
    per_aug.write_parquet(out / "augmented_sessions.parquet")
    aggregate_per_session(per_aug).write_parquet(out / "density_ratio.parquet")

    # ── §6 analysis digest + plots ───────────────────────────────────────────
    print("[s06] §6 — analysis + plots…")
    _emit_analysis(out, study=art.study, survivor_trials=survivor_trials)

    elapsed = time.time() - t0
    print(f"[s06] done in {elapsed:.1f}s. Outputs in {out}")


# ---------------------------------------------------------------------------
# §6 — analysis digest + plots (called from main at the end)
# ---------------------------------------------------------------------------

def _emit_analysis(out: Path, *, study, survivor_trials: list[dict]) -> None:
    """Print summary digest, save analysis.json, render plots/analysis_*.png.

    All inputs come from artifacts already on disk (holm_survivors.json,
    posi_auc.json, calibration/report.json, density_ratio.parquet) so this
    routine can be re-run standalone for any completed v2 output.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    holm = json.loads((out / "holm_survivors.json").read_text())
    posi = json.loads((out / "posi_auc.json").read_text())
    cal = json.loads((out / "calibration" / "report.json").read_text())
    dr = pl.read_parquet(out / "density_ratio.parquet")

    # ── pull numbers ─────────────────────────────────────────────────────
    n_trials = int(holm.get("n_trials", 0))
    n_holm = len(holm.get("holm_survivors", []))
    n_eb = len(holm.get("eb_deduped", []))
    mu_null = float(holm.get("mu_null", float("nan")))
    sigma_null = float(holm.get("sigma_null", float("nan")))

    # CV AUC per trial from study object (already-finished trials only)
    cv_values = {int(t.number): float(t.value) for t in study.trials
                 if t.value is not None}
    posi_values = {int(k): float(v) for k, v in posi.items()}

    cv_best_trial = max(cv_values, key=cv_values.get) if cv_values else None
    posi_best_trial = max(posi_values, key=posi_values.get) if posi_values else None
    cv_best_auc = cv_values.get(cv_best_trial)
    posi_top = posi_values.get(posi_best_trial)
    cv_best_posi = posi_values.get(cv_best_trial)
    posi_arr = np.array(list(posi_values.values()))

    ens_ece = float(cal.get("ensemble_ece", float("nan")))
    kinds: dict[str, int] = {}
    for d in cal.get("per_trial", {}).values():
        kinds[d["kind"]] = kinds.get(d["kind"], 0) + 1

    w_mean = dr["weight_mean"].to_numpy()
    w_clip = dr["weight_p99_clipped"].to_numpy()
    p_ens = dr["ensemble_prob"].to_numpy()
    group_counts = dict(
        dr.group_by("group_id_mode").len().sort("group_id_mode").iter_rows()
    )

    # ── printed digest ───────────────────────────────────────────────────
    print()
    print(f"{'─' * 60}")
    print(f"[s06] ANALYSIS DIGEST")
    print(f"{'─' * 60}")
    print(f"  trials:           total={n_trials}  "
          f"holm_survivors={n_holm}  eb_deduped={n_eb}")
    print(f"  null (perm):      mu={mu_null:.4f}  sigma={sigma_null:.4f}")
    if cv_best_trial is not None:
        print(f"  CV best:          trial #{cv_best_trial}  "
              f"AUC={cv_best_auc:.4f}")
        if cv_best_posi is not None:
            gap = cv_best_auc - cv_best_posi
            tag = "  ← winner's curse" if gap > 0.05 else ""
            print(f"  CV-best PoSI:     trial #{cv_best_trial}  "
                  f"PoSI={cv_best_posi:.4f}  gap={gap:+.4f}{tag}")
    if posi_best_trial is not None:
        print(f"  PoSI best:        trial #{posi_best_trial}  "
              f"PoSI={posi_top:.4f}")
    if posi_arr.size:
        print(f"  PoSI distribution: min={posi_arr.min():.4f}  "
              f"median={np.median(posi_arr):.4f}  "
              f"mean={posi_arr.mean():.4f}  max={posi_arr.max():.4f}")
    print(f"  calibration:      ensemble_ECE={ens_ece:.4e}  kinds={kinds}")
    print(f"  density ratio:    min={w_mean.min():.3f}  "
          f"median={np.median(w_mean):.3f}  mean={w_mean.mean():.3f}  "
          f"p95={np.percentile(w_mean, 95):.3f}  max={w_mean.max():.3f}")
    print(f"  weight_p99_clipped max={w_clip.max():.3f}  "
          f"(CVaR-DRO α_eq ≈ {1 - 1.0/max(w_clip.max(), 1e-9):.2f})")
    print(f"  group_id_mode:    {group_counts}")
    print(f"{'─' * 60}")

    # ── analysis.json ────────────────────────────────────────────────────
    analysis = {
        "n_trials": n_trials,
        "holm_survivors": n_holm,
        "eb_deduped": n_eb,
        "mu_null": mu_null,
        "sigma_null": sigma_null,
        "cv_best_trial": cv_best_trial,
        "cv_best_auc": cv_best_auc,
        "cv_best_posi": cv_best_posi,
        "cv_posi_gap": (
            (cv_best_auc - cv_best_posi)
            if cv_best_auc is not None and cv_best_posi is not None else None
        ),
        "posi_best_trial": posi_best_trial,
        "posi_top": posi_top,
        "posi_stats": (
            {
                "min": float(posi_arr.min()),
                "median": float(np.median(posi_arr)),
                "mean": float(posi_arr.mean()),
                "max": float(posi_arr.max()),
            } if posi_arr.size else None
        ),
        "ensemble_ece": ens_ece,
        "calibrator_kinds": kinds,
        "weight_mean_stats": {
            "min": float(w_mean.min()),
            "median": float(np.median(w_mean)),
            "mean": float(w_mean.mean()),
            "p95": float(np.percentile(w_mean, 95)),
            "max": float(w_mean.max()),
        },
        "weight_clipped_max": float(w_clip.max()),
        "cvar_alpha_equivalent": float(1 - 1.0 / max(w_clip.max(), 1e-9)),
        "group_id_mode_counts": {int(k): int(v) for k, v in group_counts.items()},
    }
    (out / "analysis.json").write_text(json.dumps(analysis, indent=2))

    # ── plots ────────────────────────────────────────────────────────────
    plots = out / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    # (1) CV vs PoSI scatter for survivors
    ax = axes[0, 0]
    common = sorted(set(cv_values) & set(posi_values))
    if common:
        x = np.array([cv_values[t] for t in common])
        y = np.array([posi_values[t] for t in common])
        ax.scatter(x, y, s=15, alpha=0.5, color="#4c72b0")
        lo = min(x.min(), y.min())
        hi = max(x.max(), y.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.7, label="y=x")
        if cv_best_trial in common:
            ax.scatter([cv_values[cv_best_trial]], [posi_values[cv_best_trial]],
                       s=80, color="crimson", marker="*", zorder=5,
                       label=f"CV-best #{cv_best_trial}")
        if posi_best_trial in common:
            ax.scatter([cv_values[posi_best_trial]], [posi_values[posi_best_trial]],
                       s=80, color="seagreen", marker="*", zorder=5,
                       label=f"PoSI-best #{posi_best_trial}")
        ax.legend(loc="lower right", fontsize=8)
    ax.set_xlabel("CV AUC")
    ax.set_ylabel("PoSI AUC (holdout)")
    ax.set_title("CV vs PoSI on Holm survivors\n(below y=x = winner's curse)")

    # (2) PoSI histogram
    ax = axes[0, 1]
    if posi_arr.size:
        ax.hist(posi_arr, bins=30, color="#55a868", alpha=0.75)
        ax.axvline(np.median(posi_arr), color="black", lw=1, ls="--",
                   label=f"median={np.median(posi_arr):.3f}")
        if posi_top is not None:
            ax.axvline(posi_top, color="seagreen", lw=1.2,
                       label=f"PoSI-best={posi_top:.3f}")
        if cv_best_posi is not None:
            ax.axvline(cv_best_posi, color="crimson", lw=1.2,
                       label=f"CV-best PoSI={cv_best_posi:.3f}")
        ax.legend(fontsize=8)
    ax.set_xlabel("PoSI AUC")
    ax.set_ylabel("trials")
    ax.set_title("PoSI AUC distribution (survivors)")

    # (3) weight histograms (raw vs clipped)
    ax = axes[0, 2]
    bins = np.linspace(min(w_mean.min(), w_clip.min()),
                       max(w_mean.max(), w_clip.max()), 50)
    ax.hist(w_mean, bins=bins, alpha=0.55, label="weight_mean (raw)", color="#c44e52")
    ax.hist(w_clip, bins=bins, alpha=0.55, label="weight_p99_clipped", color="#4c72b0")
    ax.axvline(1.0, color="black", lw=0.7, ls="--")
    ax.set_xlabel("density-ratio weight")
    ax.set_ylabel("sessions")
    ax.set_title("Density-ratio distribution")
    ax.legend(fontsize=8)

    # (4) ensemble_prob histogram (log scale)
    ax = axes[1, 0]
    ax.hist(p_ens, bins=40, color="#8172b2", alpha=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("ensemble P(blind | x)")
    ax.set_ylabel("sessions (log)")
    ax.set_title("Calibrated ensemble probability")

    # (5) group_id_mode bar chart
    ax = axes[1, 1]
    if group_counts:
        keys = list(range(9))
        vals = [group_counts.get(k, 0) for k in keys]
        bars = ax.bar(keys, vals, color="#937860")
        for k, v in zip(keys, vals):
            if v == 0:
                continue
            ax.text(k, v, str(v), ha="center", va="bottom", fontsize=7)
    ax.set_xlabel("group_id_mode")
    ax.set_ylabel("sessions")
    ax.set_title("Group-DRO bucket population")

    # (6) Per-trial ECE bar chart (calibrators)
    ax = axes[1, 2]
    per_trial = cal.get("per_trial", {})
    if per_trial:
        tns = sorted(per_trial.keys(), key=lambda k: int(k))
        eces = [per_trial[t]["ece"] for t in tns]
        color = ["#c44e52" if per_trial[t]["kind"] == "platt"
                 else "#55a868" for t in tns]
        ax.bar(range(len(tns)), eces, color=color)
        ax.set_yscale("log")
        ax.axhline(ens_ece, color="black", lw=1, ls="--",
                   label=f"ensemble ECE = {ens_ece:.2e}")
        ax.legend(fontsize=8)
    ax.set_xlabel("survivor trial (sorted by idx)")
    ax.set_ylabel("ECE (log)")
    ax.set_title("Per-trial calibration ECE\n(red=Platt, green=isotonic)")

    fig.tight_layout()
    fig.savefig(plots / "analysis_overview.png", dpi=120)
    plt.close(fig)
    print(f"[s06] §6 — analysis saved to {out / 'analysis.json'} "
          f"and {plots / 'analysis_overview.png'}")


if __name__ == "__main__":
    main()
