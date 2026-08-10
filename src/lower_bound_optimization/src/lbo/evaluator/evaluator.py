"""StratifiedEvaluator: blind-A-shaped subset evaluation of recommender models.

Workflow
--------
1. Build:    ev = StratifiedEvaluator(...)
2. Prepare:  ev.prepare_subsets(eval_df, blind_features_df, density_ratio_df=...)
3. Evaluate: ev.evaluate(recs_df, metric="ndcg@20", label="heuristic")

prepare_subsets creates K=`n_subsets` subsets of `subset_size` sessions each,
sampled either by density_ratio weights (V9 XGB classifier OOF) and/or by
Dirichlet-smoothed stratified marginal weights against blind-A.

evaluate joins recs to GT at each session's sampled `max_turn + 1` turn,
computes the per-session metric, then per subset computes a macro-by-turn
mean (averaging within each max_turn bucket present in the subset, then
averaging across buckets) → distribution of K scalars. Reports mean, std,
CI at 90 / 95 / 99, plus a histogram with the overlays.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl

from lbo.evaluator.metrics import METRICS
from lbo.evaluator.plots import plot_distribution, plot_distribution_overlay
from lbo.evaluator.samplers import density_weighted_subsets, stratified_subsets
from lbo.evaluator.stratification import (
    calibrate_weights_to_marginals,
    compute_strat_weights,
    extract_marginals,
    geometric_blend,
    kl_divergence_marginal,
    sample_max_turn_per_session,
)
from lbo.paths import BLIND_MUSIC_TURNS_PMF


@dataclass
class _Prepared:
    eval_df: pl.DataFrame
    session_ids: np.ndarray            # ordered as eval_df rows
    max_turn_per_row: np.ndarray       # sampled max_turn per session
    weights: dict[str, np.ndarray] = field(default_factory=dict)        # strat_name -> w
    subsets: dict[str, np.ndarray] = field(default_factory=dict)        # strat_name -> (K, S) idx
    marginals: Optional[dict] = None
    kl_per_strategy: dict[str, dict[str, float]] = field(default_factory=dict)


class StratifiedEvaluator:
    def __init__(
        self,
        *,
        n_subsets: int = 2000,
        subset_size: int = 80,
        seed: int = 42,
        max_turn_pmf: dict[int, int] = BLIND_MUSIC_TURNS_PMF,
        stratification_cols: tuple[str, ...] = (
            "pop_mean", "year_mean",
            # "preferred_musical_culture", "country_code", "top_tag",
        ),
        dirichlet_alpha: float = 0.5,
        n_bins_numeric: int = 4,
    ) -> None:
        self.n_subsets = n_subsets
        self.subset_size = subset_size
        self.seed = seed
        self.max_turn_pmf = dict(max_turn_pmf)
        self.stratification_cols = list(stratification_cols)
        self.dirichlet_alpha = dirichlet_alpha
        self.n_bins_numeric = n_bins_numeric
        self._prep: Optional[_Prepared] = None

    # ----- prepare -----

    def prepare_subsets(
        self,
        eval_df: pl.DataFrame,
        blind_features_df: Optional[pl.DataFrame] = None,
        density_ratio_df: Optional[pl.DataFrame] = None,
        strategies: tuple[str, ...] = ("density", "stratified", "calibrated"),
        blend_alpha: float = 0.5,
    ) -> None:
        """Prepare subset index arrays for each requested strategy.

        Available strategies:
            - "density"     : sample with prob ∝ V9 density-ratio weights
            - "stratified"  : sample with prob ∝ Dirichlet-smoothed marginal weights
            - "calibrated"  : raking — density weights calibrated to match blind
                              marginals exactly (RECOMMENDED — combines both)
            - "blend"       : geometric blend of density and stratified weights,
                              exponent = `blend_alpha`

        `density_ratio_df` required for {"density", "calibrated", "blend"}.
        `blind_features_df` required for {"stratified", "calibrated", "blend"}.
        """
        valid = {"density", "stratified", "calibrated", "blend"}
        bad = set(strategies) - valid
        if bad:
            raise ValueError(f"unknown strategies: {bad}")
        if "session_id" not in eval_df.columns:
            raise ValueError("eval_df must contain a session_id column")

        session_ids = eval_df["session_id"].to_numpy()
        max_turn = sample_max_turn_per_session(
            session_ids.tolist(), self.seed, self.max_turn_pmf
        )
        prep = _Prepared(
            eval_df=eval_df,
            session_ids=session_ids,
            max_turn_per_row=max_turn,
        )

        needs_density = bool({"density", "calibrated", "blend"} & set(strategies))
        needs_stratified = bool({"stratified", "calibrated", "blend"} & set(strategies))

        w_density = None
        w_stratified = None
        marginals = None

        if needs_density:
            if density_ratio_df is None:
                raise ValueError("density/calibrated/blend require density_ratio_df")
            wmap = dict(zip(
                density_ratio_df["session_id"].to_list(),
                density_ratio_df["weight_mean"].to_list(),
            ))
            w_density = np.array(
                [wmap.get(s, 0.0) for s in session_ids.tolist()], dtype=np.float64
            )

        if needs_stratified:
            if blind_features_df is None:
                raise ValueError("stratified/calibrated/blend require blind_features_df")
            missing = [c for c in self.stratification_cols if c not in eval_df.columns]
            if missing:
                raise ValueError(f"eval_df missing stratification cols: {missing}")
            marginals = extract_marginals(
                eval_df, blind_features_df,
                self.stratification_cols,
                n_bins=self.n_bins_numeric,
                alpha=self.dirichlet_alpha,
            )
            w_stratified = compute_strat_weights(eval_df, marginals)
            prep.marginals = marginals

        for s_idx, strat in enumerate(strategies):
            if strat == "density":
                w = w_density
            elif strat == "stratified":
                w = w_stratified
            elif strat == "calibrated":
                # Raking: density weights iteratively scaled to match blind marginals
                w = calibrate_weights_to_marginals(w_density, eval_df, marginals, n_iter=200)
            elif strat == "blend":
                w = geometric_blend(w_density, w_stratified, alpha=blend_alpha)
            prep.weights[strat] = w
            prep.subsets[strat] = density_weighted_subsets(
                w, n_subsets=self.n_subsets, subset_size=self.subset_size,
                seed=self.seed + s_idx,
            )
            if marginals is not None:
                prep.kl_per_strategy[strat] = kl_divergence_marginal(
                    w, eval_df, marginals
                )

        self._prep = prep

    # ----- evaluate -----

    def evaluate(
        self,
        recs_df: pl.DataFrame,
        *,
        gt_df: pl.DataFrame,
        metric: str = "ndcg@20",
        out_dir: Optional[Path] = None,
        label: str = "model",
    ) -> dict:
        """`recs_df` cols: session_id, turn_number, track_ids (list[str]), scores (list[float]).
        Must contain predictions at EVERY turn 1..7 per session so the sampled
        max_turn+1 lookup is always satisfiable.

        `gt_df` cols: session_id, turn_number, gt_track_id. (Typically derived
        from the splitK holdout_test parquet.)

        Returns a dict with per-strategy per-subset scores + summary stats.
        """
        if self._prep is None:
            raise RuntimeError("call prepare_subsets() first")
        if metric not in METRICS:
            raise ValueError(f"unknown metric {metric}; available: {sorted(METRICS)}")
        prep = self._prep
        metric_fn = METRICS[metric]

        # Build per-row prediction key: (session_id, target_turn = max_turn+1)
        eval_rows = pl.DataFrame({
            "session_id": prep.session_ids,
            "target_turn": prep.max_turn_per_row + 1,
            "row_idx": np.arange(len(prep.session_ids), dtype=np.int64),
        })
        # Join recs at the sampled target turn
        rec_cols = ["session_id", "turn_number", "track_ids", "scores"]
        recs_at_turn = (
            recs_df.select(rec_cols)
            .rename({"turn_number": "target_turn"})
        )
        joined = eval_rows.join(recs_at_turn, on=["session_id", "target_turn"], how="left")
        # Join GT at the same target turn
        gt_at_turn = (
            gt_df.select(["session_id", "turn_number", "gt_track_id"])
            .rename({"turn_number": "target_turn"})
        )
        joined = joined.join(gt_at_turn, on=["session_id", "target_turn"], how="left")

        # Per-row metric value
        joined = joined.sort("row_idx")
        track_ids_list = joined["track_ids"].to_list()
        scores_list = joined["scores"].to_list()
        gt_list = joined["gt_track_id"].to_list()
        per_row = np.array([
            metric_fn(
                t if t is not None else [],
                s if s is not None else [],
                g,
            )
            for t, s, g in zip(track_ids_list, scores_list, gt_list)
        ], dtype=np.float64)

        cov = float(np.mean(~np.isnan(per_row)))
        print(f"[evaluate] coverage = {cov:.3%}  (rows with both rec at turn + GT)")

        results: dict = {
            "metric": metric, "label": label,
            "n_subsets": self.n_subsets, "subset_size": self.subset_size,
            "coverage_at_target_turn": cov,
            "per_row_metric_count": int((~np.isnan(per_row)).sum()),
            "strategies": {},
        }

        for strat_name, subsets in prep.subsets.items():
            scores = self._score_per_subset(
                per_row, prep.max_turn_per_row, subsets,
            )
            summary = self._summarise(scores)
            kl = prep.kl_per_strategy.get(strat_name, {})
            results["strategies"][strat_name] = {
                **summary,
                "scores": scores.tolist(),
                "kl_marginal": kl,
                "kl_marginal_sum": float(sum(kl.values())) if kl else None,
            }

            if out_dir is not None:
                out_dir = Path(out_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                plot_distribution(
                    scores, summary,
                    out_dir / f"{label}__{metric}__{strat_name}.png",
                    title=f"{label} — {metric} ({strat_name})",
                )

        # Overlay plot if both strategies are present
        if out_dir is not None and len(results["strategies"]) > 1:
            scores_per = {k: np.array(v["scores"]) for k, v in results["strategies"].items()}
            sum_per = {k: v for k, v in results["strategies"].items()}
            plot_distribution_overlay(
                scores_per, sum_per,
                Path(out_dir) / f"{label}__{metric}__overlay.png",
                title=f"{label} — {metric} (overlay)",
            )
            # Strip the heavy `scores` list before JSON-dumping
            light = {
                "metric": results["metric"], "label": results["label"],
                "n_subsets": results["n_subsets"], "subset_size": results["subset_size"],
                "coverage_at_target_turn": cov,
                "strategies": {
                    k: {kk: vv for kk, vv in v.items() if kk != "scores"}
                    for k, v in results["strategies"].items()
                },
            }
            (Path(out_dir) / f"{label}__{metric}__summary.json").write_text(
                json.dumps(light, indent=2)
            )

        return results

    # ----- internals -----

    @staticmethod
    def _score_per_subset(
        per_row: np.ndarray, max_turn_per_row: np.ndarray, subsets: np.ndarray,
    ) -> np.ndarray:
        """For each subset row, group sampled sessions by their max_turn bucket,
        mean within bucket (ignoring NaN), then mean across buckets present.
        Returns shape (n_subsets,).
        """
        n_subsets = subsets.shape[0]
        out = np.empty(n_subsets, dtype=np.float64)
        for i in range(n_subsets):
            idx = subsets[i]
            buckets = max_turn_per_row[idx]
            vals = per_row[idx]
            mask = ~np.isnan(vals)
            if not mask.any():
                out[i] = np.nan
                continue
            # group by bucket
            uniq = np.unique(buckets[mask])
            bucket_means = []
            for b in uniq:
                m = mask & (buckets == b)
                if m.any():
                    bucket_means.append(float(np.nanmean(vals[m])))
            out[i] = float(np.mean(bucket_means)) if bucket_means else np.nan
        return out

    @staticmethod
    def _summarise(scores: np.ndarray) -> dict:
        clean = scores[~np.isnan(scores)]
        if clean.size == 0:
            nan = float("nan")
            return {"mean": nan, "std": nan, "median": nan,
                    "ci90": [nan, nan], "ci95": [nan, nan], "ci99": [nan, nan],
                    "n_valid": 0}
        return {
            "mean": float(np.mean(clean)),
            "std": float(np.std(clean, ddof=1)) if clean.size > 1 else 0.0,
            "median": float(np.median(clean)),
            "ci90": [float(np.percentile(clean, 5)), float(np.percentile(clean, 95))],
            "ci95": [float(np.percentile(clean, 2.5)), float(np.percentile(clean, 97.5))],
            "ci99": [float(np.percentile(clean, 0.5)), float(np.percentile(clean, 99.5))],
            "n_valid": int(clean.size),
        }
