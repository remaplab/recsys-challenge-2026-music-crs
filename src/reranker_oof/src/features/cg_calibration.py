"""Per-CG probability calibration + conformal set-size features.

Recipe (LBO doc C §1 + E):
    1. For each CG, fit an isotonic / Platt calibrator on (feature, is_gt)
       pairs from the OOF predictions. Feature = `reciprocal_rank` by default
       (bounded, rank-monotonic, comparable across CGs with wildly different
       raw-score scales).
    2. Conformal split: nonconformity_i = 1 - calibrator(feature_at_gt_i).
       q_hat = (1 - alpha) conformal-adjusted quantile of {nonconformity_i}.
       At inference: a candidate is "in the prediction set" iff
       1 - calibrator(its_feature) ≤ q_hat. `set_size_<cg>` per
       (session_id, turn) = number of candidates in the set. High → CG
       uncertain; low → CG confident.

Honesty: per LBO winner's-curse discussion, the calibrator/conformal for
each session must NOT have seen that session's GT. We use the splitK fold
structure: for sessions in fold k, apply the calibrator that was fit on
folds ≠ k. For holdout / blind splits (sessions disjoint from all folds),
apply the "global" calibrator fit on all five OOF folds.

Streaming-safe: the calibrator fits happen ONCE up front (one CG at a time,
peak ~50 MB per CG), independently from the per-chunk feature build.
"""
from __future__ import annotations

import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


# Sentinel key in the per-fold-excluded dicts for the "fit on ALL folds"
# calibrator used by holdout + blind splits.
GLOBAL_KEY: int = -1


# ---------------------------------------------------------------------------
# Calibrator + conformal artifact bundle
# ---------------------------------------------------------------------------

@dataclass
class CalibrationArtifacts:
    """Bundle of per-CG calibrators + conformal quantiles indexed by the
    "excluded fold" (k = which fold's sessions this calibrator may be applied to).

    `score_calibrators[cg][k]` = callable mapping `feature_array → P(is_gt)`.
    `conformal_quantiles[cg][k]` = float `q_hat` so a candidate is "in set"
        when `1 - calibrator(feat) ≤ q_hat`.

    For k = GLOBAL_KEY (-1): fit on the union of all 5 folds → used for
    holdout + blind splits.
    """
    score_calibrators: dict[str, dict[int, Callable[[np.ndarray], np.ndarray]]] = field(default_factory=dict)
    conformal_quantiles: dict[str, dict[int, float]] = field(default_factory=dict)
    method: str = "isotonic"
    feature_col: str = "reciprocal_rank"
    alpha: float = 0.1

    def calibrator_for(self, cg: str, excluded: int) -> Callable[[np.ndarray], np.ndarray]:
        return self.score_calibrators[cg].get(excluded, self.score_calibrators[cg][GLOBAL_KEY])

    def q_hat_for(self, cg: str, excluded: int) -> float:
        return self.conformal_quantiles[cg].get(excluded, self.conformal_quantiles[cg][GLOBAL_KEY])


# ---------------------------------------------------------------------------
# Calibrator + conformal: fit
# ---------------------------------------------------------------------------

class _Identity:
    """Fallback calibrator when there's not enough signal to fit."""
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0)


class _IsotonicCal:
    """Picklable wrapper around a fitted IsotonicRegression (plain lambdas
    aren't stdlib-picklable — needed so CalibrationArtifacts can be persisted)."""
    def __init__(self, ir: IsotonicRegression) -> None:
        self.ir = ir

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.clip(self.ir.transform(np.asarray(x, dtype=np.float64)), 0.0, 1.0)


class _PlattCal:
    """Picklable wrapper around a fitted LogisticRegression."""
    def __init__(self, lr: LogisticRegression) -> None:
        self.lr = lr

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.lr.predict_proba(np.asarray(x, dtype=np.float64).reshape(-1, 1))[:, 1]


def _fit_one_calibrator(
    feat: np.ndarray, y: np.ndarray, method: str,
) -> Callable[[np.ndarray], np.ndarray]:
    feat = np.asarray(feat, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)        # accept NaN, will be masked
    finite = np.isfinite(feat) & np.isfinite(y)
    feat = feat[finite]
    y = y[finite].astype(np.int8)
    if feat.size < 20 or y.sum() == 0 or y.sum() == y.size:
        return _Identity()
    if method == "isotonic":
        ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        ir.fit(feat, y.astype(np.float64))
        return _IsotonicCal(ir)
    if method == "platt":
        lr = LogisticRegression(C=1e6, max_iter=200)
        lr.fit(feat.reshape(-1, 1), y)
        return _PlattCal(lr)
    raise ValueError(f"unknown method: {method!r}")


def _conformal_quantile(
    nonconformity: np.ndarray, alpha: float,
) -> float:
    """(1 - alpha) conformal-adjusted upper quantile.

    Convention (LBO doc C §1): q_idx = ⌈(n + 1)(1 - alpha)⌉ - 1 (1-based →
    0-based). Inflates the quantile slightly to recover finite-sample marginal
    coverage ≥ 1 - alpha.
    """
    finite = np.isfinite(nonconformity)
    nc = nonconformity[finite]
    if nc.size == 0:
        return float("inf")
    n = nc.size
    q_idx = min(n - 1, int(np.ceil((n + 1) * (1 - alpha))) - 1)
    return float(np.sort(nc)[q_idx])


def normalize_cg_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Adapt any CG candidate DataFrame to the canonical schema
    (`models/CG_crossvalidation/item_knn_session/datasets/fold_0_oof_reranker_val.parquet`):

        session_id (str), user_id (str), turn (i64),
        track_ids (list[str]), scores (list[f64]), gt_track_id (str)

    Fixes observed on `split_hidim_xattn_hardneg_query_session`:
        * `gt_turn_number` present + `turn` constant → overwrite `turn` with
          the correct per-row `gt_turn_number`, drop the helper col.
        * `fallback_used` present → drop (not part of the canonical schema).
        * Columns reordered so canonical ones come first; any extras kept
          at the end so this is non-destructive.

    Safe no-op on already-canonical parquets (heuristic_session, item_knn,
    blind_a candidates, etc.).
    """
    out = df
    if "gt_turn_number" in out.columns:
        out = out.with_columns(
            pl.col("gt_turn_number").alias("turn")
        ).drop("gt_turn_number")
        if "turn_number" in out.columns:
            out = out.drop("turn_number")
    if "fallback_used" in out.columns:
        out = out.drop("fallback_used")
    canonical = (
        "session_id", "user_id", "turn",
        "track_ids", "scores", "gt_track_id",
    )
    ordered = [c for c in canonical if c in out.columns]
    extras = [c for c in out.columns if c not in ordered]
    return out.select(ordered + extras)


def _wide_to_long_with_gt(
    df: pl.DataFrame, k_per_cg: int | None = None,
) -> pl.DataFrame:
    """Convert a CG candidate parquet (one row per (session, turn) with
    parallel `track_ids` / `scores` lists) to long format with:
        session_id, turn_number, track_id, rank, reciprocal_rank, is_gt
    GT-less rows (no `gt_track_id`) are dropped — they don't help fit a
    P(is_gt | feature) calibrator.

    If `k_per_cg` is provided, the per-recommendation `track_ids` / `scores`
    lists are truncated to their first `k_per_cg` entries BEFORE exploding.
    This keeps the calibrator's fit budget bounded and matches the cap the
    pool-assembly path applies (same single source of truth).
    """
    # Adapt to canonical schema (overwrite constant `turn` w/ `gt_turn_number`
    # when present, drop `fallback_used`, reorder).
    df = normalize_cg_columns(df)
    # Normalise the turn column name (some recs files use `turn`).
    if "turn_number" not in df.columns and "turn" in df.columns:
        df = df.rename({"turn": "turn_number"})
    df = df.filter(pl.col("gt_track_id").is_not_null())

    has_scores = "scores" in df.columns
    if k_per_cg is not None and k_per_cg > 0:
        cap_exprs = [pl.col("track_ids").list.head(k_per_cg).alias("track_ids")]
        if has_scores:
            cap_exprs.append(
                pl.col("scores").list.head(k_per_cg).alias("scores")
            )
        df = df.with_columns(cap_exprs)
    explode_cols = ["track_ids", "scores"] if has_scores else ["track_ids"]
    long = df.explode(explode_cols).rename(
        {"track_ids": "track_id", **({"scores": "score"} if has_scores else {})}
    )
    long = long.filter(pl.col("track_id").is_not_null())
    long = long.with_columns(
        (pl.int_range(pl.len()).over("session_id", "turn_number") + 1)
        .cast(pl.Int32).alias("rank"),
    ).with_columns(
        (1.0 / pl.col("rank")).alias("reciprocal_rank"),
        (pl.col("track_id") == pl.col("gt_track_id")).cast(pl.Int8).alias("is_gt"),
    )
    return long.select(
        "session_id", "turn_number", "track_id", "rank",
        "reciprocal_rank", "is_gt",
    )


def fit_artifacts(
    cg_keep: list[str],
    cg_oof_paths_per_fold: dict[int, dict[str, Path]],
    *,
    method: str = "isotonic",
    feature_col: str = "reciprocal_rank",
    alpha: float = 0.1,
    k_per_cg: int | None = None,
    verbose: bool = True,
) -> CalibrationArtifacts:
    """One-pass fit of calibrators + conformal quantiles.

    Args
    ----
    cg_keep:
        Logical CG names.
    cg_oof_paths_per_fold:
        Mapping `{fold_idx: {cg_name: path_to_oof_parquet}}`. The parquets
        are the per-fold-cg_val candidates (`fold_{k}_oof_cg_val.parquet`).
    method:
        "isotonic" (recommended) or "platt".
    feature_col:
        Column passed to the calibrator. Default `reciprocal_rank` — bounded,
        rank-monotonic, comparable across CGs with wildly different raw-score
        scales.
    alpha:
        Conformal miscoverage level. `alpha=0.1` → 90 % marginal coverage.

    For each CG, fits one calibrator per held-out fold + one global one.
    Streaming: one CG at a time → peak memory ~50 MB per CG.
    """
    fold_keys = sorted(cg_oof_paths_per_fold.keys())
    excluded_keys: list[int] = [*fold_keys, GLOBAL_KEY]

    art = CalibrationArtifacts(method=method, feature_col=feature_col, alpha=alpha)
    n_cg = len(cg_keep)
    for ci, cg in enumerate(cg_keep, 1):
        t_cg = time.time()
        if verbose:
            print(f"[cg_calibration] [{ci}/{n_cg}] {cg!r}: reading "
                  f"{len(fold_keys)} OOF folds…", flush=True)
        long_per_fold: dict[int, pl.DataFrame] = {}
        for k in fold_keys:
            p = cg_oof_paths_per_fold[k][cg]
            long_per_fold[k] = _wide_to_long_with_gt(
                pl.read_parquet(p), k_per_cg=k_per_cg,
            )
        if verbose:
            n_rows = sum(d.height for d in long_per_fold.values())
            n_gt = sum(int(d["is_gt"].sum()) for d in long_per_fold.values())
            print(f"           {n_rows:,} candidate rows ({n_gt:,} GT); fitting "
                  f"{len(excluded_keys)} calibrators…", flush=True)

        art.score_calibrators[cg] = {}
        art.conformal_quantiles[cg] = {}
        for excluded in excluded_keys:
            train_frames = [
                long_per_fold[k] for k in fold_keys if k != excluded
            ]
            train = pl.concat(train_frames)
            feat = train[feature_col].to_numpy()
            y = train["is_gt"].to_numpy()
            cal = _fit_one_calibrator(feat, y, method)
            art.score_calibrators[cg][excluded] = cal

            gt_rows = train.filter(pl.col("is_gt") == 1)
            if gt_rows.height:
                p_cal_gt = cal(gt_rows[feature_col].to_numpy())
                nonconf = 1.0 - np.asarray(p_cal_gt, dtype=np.float64)
                art.conformal_quantiles[cg][excluded] = _conformal_quantile(
                    nonconf, alpha=alpha,
                )
            else:
                art.conformal_quantiles[cg][excluded] = float("inf")

        if verbose:
            stats = {k: round(art.conformal_quantiles[cg][k], 4) for k in excluded_keys}
            print(f"           done in {time.time() - t_cg:.1f}s · q_hat by "
                  f"excluded fold: {stats}", flush=True)
        del long_per_fold
    return art


# ---------------------------------------------------------------------------
# Persistence — decouples fitting (needs all 14 CGs' fold OOF parquets) from
# applying (needs only the artifacts), so a scoped single-split run doesn't
# have to touch the fold parquets at all.
# ---------------------------------------------------------------------------

def save_artifacts(art: CalibrationArtifacts, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(art, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[cg_calibration] saved {path}")


def load_artifacts(path: Path) -> CalibrationArtifacts:
    with open(path, "rb") as f:
        art = pickle.load(f)
    print(f"[cg_calibration] loaded {path} "
          f"({len(art.score_calibrators)} CGs, method={art.method!r})")
    return art


# ---------------------------------------------------------------------------
# Apply: enrich a long-format pool with calibrated_score_<cg> + set_size_<cg>
# ---------------------------------------------------------------------------

def apply_to_pool(
    pool: pl.DataFrame,
    cg_keep: list[str],
    artifacts: CalibrationArtifacts,
    *,
    fold_excluded: int,
    feature_col: str | None = None,
    calibrated_only: bool = False,
) -> pl.DataFrame:
    """Add per-CG `calibrated_score_<cg>` and `set_size_<cg>` columns.

    `fold_excluded` selects which fitted calibrator to use:
        - `k` in {0..4}     → the calibrator fit on folds ≠ k. Use for the
                              pool of sessions belonging to fold k
                              (so their GT was not in the fit).
        - `GLOBAL_KEY` (-1) → the calibrator fit on all 5 folds. Use for
                              holdout / blind / submission pools.

    `feature_col`: pool column used as calibrator input. Defaults to the
    column the calibrators were fit on (e.g. `reciprocal_rank_<cg>` if
    `artifacts.feature_col == "reciprocal_rank"`).

    `calibrated_only`: fast path for the fusion-cache pool, which keeps only
    `calibrated_score_<cg>`. Skips the conformal flags, the per-CG `set_size`
    groupby-windows, and all cross-CG aggregates (the dominant cost) — they'd be
    discarded by `_compact_pool` anyway. The feature-builder path leaves it False.
    """
    feat_base = feature_col or artifacts.feature_col
    g_keys = (
        ["session_id", "turn_number"] if "turn_number" in pool.columns
        else ["session_id", "turn"]
    )

    extra: list[pl.Series] = []
    flag_cols: list[str] = []
    for cg in cg_keep:
        col_in = f"{feat_base}_{cg}"
        if col_in not in pool.columns:
            continue
        cal = artifacts.calibrator_for(cg, fold_excluded)
        feat = pool[col_in].to_numpy()
        # Replace null with 0 (CG didn't retrieve → reciprocal_rank=0). null
        # values are NaN in numpy, so the calibrator (which clips) handles it.
        feat = np.where(np.isnan(feat), 0.0, feat)
        p_cal = np.asarray(cal(feat), dtype=np.float32)
        extra.append(pl.Series(f"calibrated_score_{cg}", p_cal))

        if calibrated_only:
            continue
        q_hat = artifacts.q_hat_for(cg, fold_excluded)
        in_set = (1.0 - p_cal) <= q_hat
        flag_name = f"_in_conformal_set_{cg}"
        extra.append(pl.Series(flag_name, in_set.astype(np.int8)))
        flag_cols.append(flag_name)

    if not extra:
        return pool
    pool = pool.with_columns(*extra)
    if calibrated_only:
        return pool

    # Per-(session, turn) set_size per CG via groupby-window sum.
    for cg in cg_keep:
        flag = f"_in_conformal_set_{cg}"
        if flag not in pool.columns:
            continue
        pool = pool.with_columns(
            pl.col(flag).sum().over(g_keys).cast(pl.Int32).alias(f"set_size_{cg}")
        )

    # Aggregate set_size across CGs (useful single-feature uncertainty signal).
    setsize_cols = [
        f"set_size_{cg}" for cg in cg_keep if f"set_size_{cg}" in pool.columns
    ]
    if setsize_cols:
        pool = pool.with_columns(
            pl.mean_horizontal(setsize_cols).cast(pl.Float32).alias("mean_set_size_across_cgs"),
            pl.min_horizontal(setsize_cols).cast(pl.Int32).alias("min_set_size_across_cgs"),
        )

    # Aggregate calibrated P(is_gt) across CGs — a candidate that several CGs
    # confidently calibrate as GT is a strong rerank signal. ``calibrated_margin``
    # (best − 2nd-best) captures how decisively the top CG stands out.
    cal_cols = [
        f"calibrated_score_{cg}" for cg in cg_keep
        if f"calibrated_score_{cg}" in pool.columns
    ]
    if cal_cols:
        sorted_desc = pl.concat_list(cal_cols).list.sort(descending=True)
        pool = pool.with_columns(
            pl.max_horizontal(cal_cols).cast(pl.Float32).alias("max_calibrated_across_cgs"),
            pl.mean_horizontal(cal_cols).cast(pl.Float32).alias("mean_calibrated_across_cgs"),
            pl.concat_list(cal_cols).list.std().cast(pl.Float32).alias("std_calibrated_across_cgs"),
            (sorted_desc.list.get(0)
             - sorted_desc.list.get(1, null_on_oob=True)).cast(pl.Float32)
                .alias("calibrated_margin"),
            sum((pl.col(c) > 0.5).cast(pl.Int32) for c in cal_cols)
                .alias("n_cgs_high_conf"),
        )

    # ── Engineered cross-CG reduction features ───────────────────────────────
    # Row-wise reductions over the per-CG columns that a gradient-boosted tree
    # CANNOT synthesise from the individual columns (conditional argmax, row
    # median, per-turn group ops).
    rank_cols = [f"rank_{cg}" for cg in cg_keep if f"rank_{cg}" in pool.columns]
    if cal_cols and rank_cols:
        DEEP = 50                       # "deep" = past the typical rerank cutoff
        row_min_rank = pl.min_horizontal(rank_cols)
        rc_pairs = [cg for cg in cg_keep
                    if f"rank_{cg}" in pool.columns and f"calibrated_score_{cg}" in pool.columns]
        pool = pool.with_columns(
            # minority_deep_calib: strongest calibrated belief among CGs that
            pl.max_horizontal([
                pl.when(pl.col(f"rank_{cg}") > DEEP)
                  .then(pl.col(f"calibrated_score_{cg}")).otherwise(0.0)
                for cg in rc_pairs
            ]).cast(pl.Float32).alias("minority_deep_calib"),
            # calib_at_minrank: calibrated belief of the CG that ranks the
            # candidate BEST. Isolates "one CG is sure" from weak consensus.
            pl.max_horizontal([
                pl.when(pl.col(f"rank_{cg}") == row_min_rank)
                  .then(pl.col(f"calibrated_score_{cg}")).otherwise(0.0)
                for cg in rc_pairs
            ]).cast(pl.Float32).alias("calib_at_minrank"),
            # rank_gap_median_min: spread between the median and best per-CG rank.
            # Large = one CG loves it while the rest bury it (disagreement).
            (pl.concat_list(rank_cols).list.median() - row_min_rank)
                .cast(pl.Float32).alias("rank_gap_median_min"),
            # calib_x_logminrank: top calibrated belief scaled by how deep
            # the best CG rank is (log tames the tail).
            (pl.col("max_calibrated_across_cgs") * (row_min_rank + 1).log())
                .cast(pl.Float32).alias("calib_x_logminrank"),
        )
        rr_pairs = [cg for cg in cg_keep
                    if f"rank_{cg}" in pool.columns and f"reciprocal_rank_{cg}" in pool.columns]
        if rr_pairs:
            # minority_deep_rr: rank-based twin of minority_deep_calib.
            pool = pool.with_columns(
                pl.max_horizontal([
                    pl.when(pl.col(f"rank_{cg}") > DEEP)
                      .then(pl.col(f"reciprocal_rank_{cg}")).otherwise(0.0)
                    for cg in rr_pairs
                ]).cast(pl.Float32).alias("minority_deep_rr"),
            )
    if cal_cols:
        pool = pool.with_columns(
            # calib_max_minus_mean: how far the top CG's calibrated belief stands
            # out from the average CG — distinctiveness of the best signal.
            (pl.col("max_calibrated_across_cgs") - pl.col("mean_calibrated_across_cgs"))
                .cast(pl.Float32).alias("calib_max_minus_mean"),
            # calib_margin_vs_turn: top calibrated belief minus the turn's median
            # — separates a turn's strong candidates from its filler (per-turn
            # group op a tree cannot compute).
            (pl.col("max_calibrated_across_cgs")
             - pl.col("max_calibrated_across_cgs").median().over(g_keys))
                .cast(pl.Float32).alias("calib_margin_vs_turn"),
        )
        # best_sem_calib: strongest calibrated belief among the SEMANTIC CGs
        # (text / embedding retrievers).
        _SEM_KW = ("query", "split_hidim", "emb_item", "qwen")
        sem_cal = [f"calibrated_score_{cg}" for cg in cg_keep
                   if f"calibrated_score_{cg}" in pool.columns and any(k in cg for k in _SEM_KW)]
        if sem_cal:
            pool = pool.with_columns(
                pl.max_horizontal(sem_cal).cast(pl.Float32).alias("best_sem_calib"))

    # Drop helper flag columns.
    if flag_cols:
        pool = pool.drop(flag_cols)
    return pool
