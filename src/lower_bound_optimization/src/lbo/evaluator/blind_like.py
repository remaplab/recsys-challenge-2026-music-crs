"""Reusable evaluator that scores a model's recommendations on Blind-A-like
subsets of any GT split.

Wraps the lower-level `StratifiedEvaluator` (subset sampling + per-subset
metric aggregation) with: GT loading, density-ratio integration, default
stratification cols that work in cold-user split, subset caching, and an
`EvalResult` dataclass containing the full per-subset distribution plus
summary stats (mean, std, median, CI95, CVaR_α, KL).

Usage:
    from lbo.evaluator import BlindLikeEvaluator

    ev = BlindLikeEvaluator(
        gt_path="data/splitK/holdout_test.parquet",
        cache_path="models/.../eval_cache/holdout_test.npz",
    )
    result = ev.score(recs_df, metric="ndcg@20")  # strategy="calibrated"
    print(result.mean, result.std, result.cvar_score)
    arr = result.per_subset                       # np.ndarray, shape (n_subsets,)

Diagnostics across all strategies in one call:
    results = ev.score_all(recs_df, metric="ndcg@20")
    # {"calibrated": EvalResult, "density": EvalResult, "stratified": EvalResult}

The same `cache_path` shared across many models guarantees identical subsets →
the s08 paired-EB comparator can directly diff their per-subset arrays.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import polars as pl

from lbo.paths import BLIND_PARQUET, ROOT, SHIFT_V2_OUT, TRACKS_META
from lbo.evaluator.evaluator import StratifiedEvaluator, _Prepared
from lbo.evaluator.holdout_features import (
    build_blind_features,
    build_holdout_features,
    build_holdout_gt,
)


# Stratification axes that work in the cold-user-split regime:
#  - identity-laden cols (preferred_musical_culture, country_code, top_tag)
#    are dropped: eval pool doesn't span blind-A's demographic mix → KL > 5.
#  - max_turn / n_queries are dropped: the evaluator already samples a target
#    turn per session from BLIND_MUSIC_TURNS_PMF, so additionally stratifying
#    on session-level turn counts double-counts AND eval/blind have disjoint
#    full-session-length distributions → residual KL stays ~1.3.
DEFAULT_STRAT_COLS: tuple[str, ...] = (
    "specificity", "category",
    "pop_mean", "year_mean",
)

# Strategies the underlying StratifiedEvaluator implements. "calibrated"
# (raking — density weights iteratively scaled to match blind marginals) is
# the canonical choice per LBO docs B+C.
DEFAULT_STRATEGIES: tuple[str, ...] = ("calibrated", "density", "stratified")


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    """Per-(metric, strategy) outcome of one `BlindLikeEvaluator.score()` call.

    `per_subset` is the full vector (length = n_subsets) — keep it so callers
    can compute CVaR at any α, build paired-EB CIs against another model, or
    plot the distribution.
    """
    metric: str
    strategy: str
    per_subset: np.ndarray
    n_valid: int
    mean: float
    std: float
    median: float
    ci95_lo: float
    ci95_hi: float
    cvar_alpha: float
    cvar_score: float
    kl_marginal: dict[str, float] = field(default_factory=dict)
    kl_marginal_sum: float = 0.0

    def to_dict(self) -> dict:
        return {
            "metric": self.metric, "strategy": self.strategy,
            "n_valid": int(self.n_valid),
            "mean": float(self.mean), "std": float(self.std),
            "median": float(self.median),
            "ci95_lo": float(self.ci95_lo), "ci95_hi": float(self.ci95_hi),
            "cvar_alpha": float(self.cvar_alpha),
            "cvar_score": float(self.cvar_score),
            "kl_marginal": dict(self.kl_marginal),
            "kl_marginal_sum": float(self.kl_marginal_sum),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve(p: Path | str | None) -> Path | None:
    if p is None:
        return None
    p = Path(p)
    return p if p.is_absolute() else (ROOT / p).resolve()


def _normalize_recs(recs: pl.DataFrame) -> pl.DataFrame:
    """Accept either `turn` or `turn_number`; standardize to `turn_number`."""
    if "turn_number" in recs.columns:
        return recs
    if "turn" in recs.columns:
        return recs.rename({"turn": "turn_number"})
    raise ValueError(
        "recs must contain either 'turn_number' or 'turn'; got cols: "
        f"{recs.columns}"
    )


def _summarise(per_subset: np.ndarray, *, cvar_alpha: float) -> dict:
    clean = per_subset[~np.isnan(per_subset)]
    if clean.size == 0:
        nan = float("nan")
        return dict(n_valid=0, mean=nan, std=nan, median=nan,
                    ci95_lo=nan, ci95_hi=nan, cvar_score=nan)
    s = np.sort(clean)
    n = len(s)
    cvar_n = max(1, int(round((1.0 - cvar_alpha) * n)))
    return dict(
        n_valid=int(n),
        mean=float(s.mean()),
        std=float(s.std(ddof=1)) if n > 1 else 0.0,
        median=float(np.median(s)),
        ci95_lo=float(np.quantile(s, 0.025)),
        ci95_hi=float(np.quantile(s, 0.975)),
        cvar_score=float(s[:cvar_n].mean()),
    )


def _save_prep_cache(prep: _Prepared, path: Path) -> None:
    """Persist a `_Prepared` snapshot to .npz. KL stored as JSON string."""
    payload: dict = {
        "session_ids": np.asarray(prep.session_ids),
        "max_turn_per_row": np.asarray(prep.max_turn_per_row),
        "strategies": np.array(list(prep.subsets.keys()), dtype=object),
        "kl_per_strategy": np.array(json.dumps(prep.kl_per_strategy)),
    }
    for strat, subsets in prep.subsets.items():
        payload[f"subsets__{strat}"] = subsets
    for strat, w in prep.weights.items():
        payload[f"weights__{strat}"] = w
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **payload)


def _load_prep_cache(path: Path) -> _Prepared:
    data = np.load(path, allow_pickle=True)
    strategies = [str(s) for s in data["strategies"].tolist()]
    prep = _Prepared(
        eval_df=None,
        session_ids=data["session_ids"],
        max_turn_per_row=data["max_turn_per_row"],
    )
    for strat in strategies:
        prep.subsets[strat] = data[f"subsets__{strat}"]
        wkey = f"weights__{strat}"
        if wkey in data.files:
            prep.weights[strat] = data[wkey]
    if "kl_per_strategy" in data.files:
        prep.kl_per_strategy = json.loads(str(data["kl_per_strategy"]))
    return prep


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class BlindLikeEvaluator:
    """Score model recommendations on Blind-A-like subsets of any GT split.

    Construct once per (gt_path, blind, density_ratio, strat_cols, n_subsets,
    subset_size, seed) configuration; subsets are prepared up front and can
    be cached to disk (`cache_path`) so many models share the exact same
    sampling for fair comparison.
    """

    def __init__(
        self,
        *,
        gt_path: Path | str,
        blind_parquet: Path | str = BLIND_PARQUET,
        tracks_meta: Path | str = TRACKS_META,
        density_ratio: Path | str | pl.DataFrame = SHIFT_V2_OUT / "density_ratio.parquet",
        n_subsets: int = 2000,
        subset_size: int = 80,
        seed: int = 42,
        strat_cols: Iterable[str] = DEFAULT_STRAT_COLS,
        strategies: Iterable[str] = DEFAULT_STRATEGIES,
        cache_path: Path | str | None = None,
        cvar_alpha: float = 0.7,
        verbose: bool = True,
    ) -> None:
        self.gt_path = _resolve(Path(gt_path))
        self.blind_parquet = _resolve(Path(blind_parquet))
        self.tracks_meta = _resolve(Path(tracks_meta))
        self.cache_path = _resolve(Path(cache_path)) if cache_path is not None else None
        self.n_subsets = int(n_subsets)
        self.subset_size = int(subset_size)
        self.seed = int(seed)
        self.strat_cols = tuple(strat_cols)
        self.strategies = tuple(strategies)
        self.cvar_alpha = float(cvar_alpha)
        self.verbose = verbose

        # ── 1. eval features ──
        self._log(f"building eval features from {self.gt_path}…")
        self.eval_features = build_holdout_features(self.gt_path, self.tracks_meta)
        self._log(f"eval sessions: {self.eval_features.height}")

        # ── 2. blind features (for marginal reference) ──
        self._log(f"building blind features from {self.blind_parquet}…")
        self.blind_features = build_blind_features(
            self.blind_parquet, self.tracks_meta,
        )
        self._log(f"blind sessions: {self.blind_features.height}")

        # ── 3. density ratio (lazy — only loaded if a strategy needs it) ──
        needs_density = bool(
            {"density", "calibrated", "blend"} & set(self.strategies)
        )
        if needs_density:
            if isinstance(density_ratio, pl.DataFrame):
                dr = density_ratio
            else:
                dr_path = _resolve(Path(density_ratio))
                dr = pl.read_parquet(dr_path).select("session_id", "weight_mean")
            eval_sids = set(self.eval_features["session_id"].to_list())
            self.density_ratio = dr.filter(pl.col("session_id").is_in(eval_sids))
            coverage = self.density_ratio.height / max(self.eval_features.height, 1)
            self.density_coverage = coverage
            self._log(
                f"density-ratio coverage on eval: "
                f"{self.density_ratio.height}/{self.eval_features.height} "
                f"({coverage:.1%})"
            )
        else:
            self.density_ratio = None
            self.density_coverage = None
            self._log("density ratio not loaded (no density/calibrated/blend strategy)")

        # ── 4. GT ──
        self._log("building GT…")
        self.gt = build_holdout_gt(self.gt_path)

        # ── 5. validate strat cols against eval + blind feature schemas ──
        for name, df in (("eval_features", self.eval_features),
                          ("blind_features", self.blind_features)):
            missing = [c for c in self.strat_cols if c not in df.columns]
            if missing:
                raise ValueError(
                    f"strat_cols missing from {name}: {missing}. "
                    f"Available: {df.columns}"
                )

        # ── 6. underlying StratifiedEvaluator ──
        self.evaluator = StratifiedEvaluator(
            n_subsets=self.n_subsets,
            subset_size=self.subset_size,
            seed=self.seed,
            stratification_cols=self.strat_cols,
        )

        # ── 7. prepare subsets (or load cache) ──
        if self.cache_path is not None and self.cache_path.exists():
            self._log(f"loading subset cache from {self.cache_path}")
            self.evaluator._prep = _load_prep_cache(self.cache_path)
        else:
            self._log(
                f"preparing subsets "
                f"(n_subsets={self.n_subsets}, subset_size={self.subset_size})…"
            )
            self.evaluator.prepare_subsets(
                self.eval_features,
                blind_features_df=self.blind_features,
                density_ratio_df=self.density_ratio,
                strategies=self.strategies,
            )
            if self.cache_path is not None:
                _save_prep_cache(self.evaluator._prep, self.cache_path)
                self._log(f"saved subset cache to {self.cache_path}")

    # ----- public API -----

    def score(
        self,
        recs: pl.DataFrame,
        *,
        metric: str = "ndcg@20",
        strategy: str = "calibrated",
    ) -> EvalResult:
        """Evaluate `recs` on the prepared subsets under `strategy`.

        `recs` cols: session_id, (turn | turn_number), track_ids (list[str]),
                     scores (list[float]).
        Returns a single `EvalResult` with `per_subset` array + summary stats.
        """
        if strategy not in self.evaluator._prep.subsets:
            raise ValueError(
                f"strategy '{strategy}' not prepared; available: "
                f"{list(self.evaluator._prep.subsets.keys())}"
            )
        recs = _normalize_recs(recs)
        res = self.evaluator.evaluate(recs, gt_df=self.gt, metric=metric)
        strat_res = res["strategies"][strategy]
        per_subset = np.asarray(strat_res["scores"], dtype=np.float64)
        summary = _summarise(per_subset, cvar_alpha=self.cvar_alpha)
        kl = strat_res.get("kl_marginal", {}) or {}
        return EvalResult(
            metric=metric, strategy=strategy,
            per_subset=per_subset,
            **summary,
            cvar_alpha=self.cvar_alpha,
            kl_marginal=dict(kl),
            kl_marginal_sum=float(sum(kl.values())) if kl else 0.0,
        )

    def score_all(
        self,
        recs: pl.DataFrame,
        *,
        metric: str = "ndcg@20",
    ) -> dict[str, EvalResult]:
        """Score all prepared strategies in one shot.

        Returns {strategy_name: EvalResult}. Single pass through `evaluate`
        internally; cheaper than calling `score()` n_strategies times.
        """
        recs = _normalize_recs(recs)
        res = self.evaluator.evaluate(recs, gt_df=self.gt, metric=metric)
        out: dict[str, EvalResult] = {}
        for strategy, strat_res in res["strategies"].items():
            per_subset = np.asarray(strat_res["scores"], dtype=np.float64)
            summary = _summarise(per_subset, cvar_alpha=self.cvar_alpha)
            kl = strat_res.get("kl_marginal", {}) or {}
            out[strategy] = EvalResult(
                metric=metric, strategy=strategy,
                per_subset=per_subset,
                **summary,
                cvar_alpha=self.cvar_alpha,
                kl_marginal=dict(kl),
                kl_marginal_sum=float(sum(kl.values())) if kl else 0.0,
            )
        return out

    # ----- internals -----

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[BlindLikeEvaluator] {msg}")
