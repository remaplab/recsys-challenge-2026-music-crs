"""HeuristicV3 — heuristic_v0 fused (RRF) with two precomputed small CGs.

v2 used HybridAllQwen only as a *shortfall* fallback. v3 instead promotes two
already-tuned, already-exported CGs to first-class fusion partners of v0:

    rrf_oneshot       (turn-1 only; the oneshot RRF ensemble)
    tower_cf_ensemble (all turns; session-DRO export)

Outer reciprocal-rank fusion at the ranked-list level. v0 produces a deep
ranked list; the two cached CGs contribute their own ranked lists for the same
(session_id, turn). Tracks proposed only by the cached CGs ARE added to the
pool. The fused score is

    score(t) = 1/(rrf_k + r_v0) + w_rrf_oneshot/(rrf_k + r_rrf)
                                + w_tower_cf/(rrf_k + r_tower)

where r_* is the 0-based rank of t in each source (a source missing t — or, for
rrf_oneshot on turns > 1, missing entirely — simply drops out of the sum).
v0's own weight is fixed at 1.0; the two cached-CG weights and rrf_k are tuned.

No GPU, no recompute: the cached export parquets (`fold_*_oof_cg_val`,
`fold_*_oof_reranker_val`, `holdout_candidates`, `blind_candidates`) cover every
split the DRO pipeline infers on and are loaded once at fit, indexed by
(session_id, turn).

Fallback source: if after fusion a row still falls short of
`min(top_k, fallback_out_n)`, it is padded from the same cached lists
(tower_cf first, then rrf_oneshot) — these CGs are the only external source.
"""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Any

import polars as pl

from .heuristic_v0 import HeuristicRecommender

# parquets to merge from each cached export dir
_CACHE_GLOBS = (
    "fold_*_oof_cg_val.parquet",
    "fold_*_oof_reranker_val.parquet",
    "holdout_candidates.parquet",
    "blind_candidates.parquet",
    "blind_b_candidates.parquet",
)


def _load_cache(datasets_dir: str | None, name: str) -> dict[tuple[str, int], list[str]]:
    """(session_id, turn) -> ranked list[track_id] from an export dir."""
    out: dict[tuple[str, int], list[str]] = {}
    if not datasets_dir:
        print(f"[HeuristicV3] no {name} dir — source disabled.")
        return out
    root = Path(datasets_dir)
    files: list[str] = []
    for pat in _CACHE_GLOBS:
        files.extend(sorted(glob.glob(str(root / pat))))
    if not files:
        raise FileNotFoundError(f"[HeuristicV3] no {name} parquets under {root}")
    for fp in files:
        df = pl.read_parquet(fp, columns=["session_id", "turn", "track_ids"])
        for sid, turn, tids in zip(
            df["session_id"].to_list(),
            df["turn"].to_list(),
            df["track_ids"].to_list(),
        ):
            out[(sid, int(turn))] = list(tids or [])
    print(f"[HeuristicV3] loaded {name} for {len(out)} (session, turn) keys "
          f"from {len(files)} parquet(s).")
    return out


class HeuristicV3(HeuristicRecommender):
    RECOMMENDER_NAME = "HeuristicV3"

    def __init__(
        self,
        rrf_oneshot_datasets_dir: str | None = None,
        tower_cf_datasets_dir: str | None = None,
        w_rrf_oneshot: float = 1.0,
        w_tower_cf: float = 1.0,
        rrf_k: int = 60,
        fallback_out_n: int = 200,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.rrf_oneshot_datasets_dir = rrf_oneshot_datasets_dir
        self.tower_cf_datasets_dir = tower_cf_datasets_dir
        self.w_rrf_oneshot = float(w_rrf_oneshot)
        self.w_tower_cf = float(w_tower_cf)
        self.rrf_k = int(rrf_k)
        self.fallback_out_n = int(fallback_out_n)
        # (session_id, turn) -> ranked list[track_id]
        self._rrf: dict[tuple[str, int], list[str]] = {}
        self._tower: dict[tuple[str, int], list[str]] = {}

    # ------------------------------------------------------------------
    # fit: v0 fit, then load the two cached CG exports
    # ------------------------------------------------------------------

    def fit(self, train_df, track_metadata=None, **kwargs: Any) -> None:
        super().fit(train_df, track_metadata=track_metadata, **kwargs)
        self._load_caches()

    def _load_caches(self) -> None:
        self._rrf = _load_cache(self.rrf_oneshot_datasets_dir, "rrf_oneshot")
        self._tower = _load_cache(self.tower_cf_datasets_dir, "tower_cf")

    # ------------------------------------------------------------------
    # recommend: deep v0 list, then RRF fusion with the two cached CGs
    # ------------------------------------------------------------------

    def recommend(
        self,
        context_df: pl.DataFrame,
        top_k: int = 20,
        remove_seen: bool = True,
        **kwargs: Any,
    ) -> pl.DataFrame:
        depth = max(int(top_k), self.fallback_out_n)
        recs = super().recommend(
            context_df, top_k=depth, remove_seen=remove_seen, **kwargs
        )

        target = min(int(top_k), self.fallback_out_n)
        k = self.rrf_k
        out_tracks: list[list[str]] = []
        out_scores: list[list[float]] = []
        n_fused = 0
        n_padded = 0

        for row in recs.iter_rows(named=True):
            sid = row["session_id"]
            turn = int(row["turn"])
            v0_tids = list(row["track_ids"] or [])
            rrf_tids = self._rrf.get((sid, turn), [])[: self.fallback_out_n]
            tower_tids = self._tower.get((sid, turn), [])[: self.fallback_out_n]

            fused: dict[str, float] = {}
            for r, t in enumerate(v0_tids):
                fused[t] = fused.get(t, 0.0) + 1.0 / (k + r)
            if rrf_tids and self.w_rrf_oneshot != 0.0:
                for r, t in enumerate(rrf_tids):
                    fused[t] = fused.get(t, 0.0) + self.w_rrf_oneshot / (k + r)
            if tower_tids and self.w_tower_cf != 0.0:
                for r, t in enumerate(tower_tids):
                    fused[t] = fused.get(t, 0.0) + self.w_tower_cf / (k + r)

            if rrf_tids or tower_tids:
                n_fused += 1

            ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:target]
            tids = [t for t, _ in ranked]
            scs = [s for _, s in ranked]

            # fallback safety: pad from cached lists if still short
            if len(tids) < target:
                seen = set(tids)
                base = scs[-1] if scs else 0.0
                pad_rank = 0
                for src in (tower_tids, rrf_tids):
                    for t in src:
                        if len(tids) >= target:
                            break
                        if t in seen:
                            continue
                        seen.add(t)
                        tids.append(t)
                        pad_rank += 1
                        scs.append(base - pad_rank)
                if pad_rank:
                    n_padded += 1

            out_tracks.append(tids)
            out_scores.append(scs)

        print(f"[{self.RECOMMENDER_NAME}] RRF-fused {n_fused} rows; "
              f"padded {n_padded} rows.")

        return recs.with_columns(
            pl.Series("track_ids", out_tracks, dtype=pl.List(pl.Utf8)),
            pl.Series("scores", out_scores, dtype=pl.List(pl.Float64)),
        )

    # ------------------------------------------------------------------
    # persistence (extends v0; reload caches on restore)
    # ------------------------------------------------------------------

    def _get_model_state(self) -> dict:
        st = super()._get_model_state()
        st.update({
            "rrf_oneshot_datasets_dir": self.rrf_oneshot_datasets_dir,
            "tower_cf_datasets_dir": self.tower_cf_datasets_dir,
            "w_rrf_oneshot": self.w_rrf_oneshot,
            "w_tower_cf": self.w_tower_cf,
            "rrf_k": self.rrf_k,
            "fallback_out_n": self.fallback_out_n,
        })
        return st

    def _set_model_state(self, state: dict) -> None:
        super()._set_model_state(state)
        self.rrf_oneshot_datasets_dir = state.get("rrf_oneshot_datasets_dir")
        self.tower_cf_datasets_dir = state.get("tower_cf_datasets_dir")
        self.w_rrf_oneshot = float(state.get("w_rrf_oneshot", 1.0))
        self.w_tower_cf = float(state.get("w_tower_cf", 1.0))
        self.rrf_k = int(state.get("rrf_k", 60))
        self.fallback_out_n = int(state.get("fallback_out_n", 200))
        self._load_caches()
