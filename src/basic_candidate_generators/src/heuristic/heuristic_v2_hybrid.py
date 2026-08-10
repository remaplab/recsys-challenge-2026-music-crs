"""HeuristicV2Hybrid — heuristic_v0 with a precomputed HybridAllQwen fallback.

v0 is the strongest CG for NDCG@20 on non-edge-case sessions/turns, but it
returns short / empty lists on its three give-up branches (no usable prior,
empty candidate pool, fewer than top_k after fusion). This subclass keeps v0's
logic *verbatim* and only fills those shortfalls with the already-tuned,
already-exported HybridAllQwen recommendations.

No GPU, no recompute at tune time: HybridAllQwen's exported candidate parquets
(`fold_*_oof_cg_val`, `fold_*_oof_reranker_val`, `holdout_candidates`,
`blind_candidates`) are loaded once at fit and indexed by (session_id, turn).
Those parquets already cover every split the DRO pipeline infers on — tuning
(cg_val), PoSI (reranker_val), export (holdout + blind) — and were generated
with the same per-fold / deployment-like training regime, so the fallback is
out-of-fold-consistent for free.

Fill policy (design A — shortfall only): v0 candidates come first; the hybrid
list for the same (session_id, turn) is appended (deduped against v0's output)
until `min(top_k, fallback_out_n)`. The hybrid parquets were generated with
remove_seen=True over the same session, so they carry no seen leakage.
"""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Any

import polars as pl

from .heuristic_v0 import HeuristicRecommender

# parquets to merge from the fallback datasets dir (HybridAllQwen export)
_FALLBACK_GLOBS = (
    "fold_*_oof_cg_val.parquet",
    "fold_*_oof_reranker_val.parquet",
    "holdout_candidates.parquet",
    "blind_candidates.parquet",
    "blind_b_candidates.parquet",
)


class HeuristicV2Hybrid(HeuristicRecommender):
    RECOMMENDER_NAME = "HeuristicV2Hybrid"

    def __init__(
        self,
        fallback_datasets_dir: str | None = None,
        fallback_out_n: int = 200,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.fallback_datasets_dir = fallback_datasets_dir
        self.fallback_out_n = int(fallback_out_n)
        # (session_id, turn) -> list[track_id]
        self._fb: dict[tuple[str, int], list[str]] = {}

    # ------------------------------------------------------------------
    # fit: v0 fit, then load the precomputed hybrid fallback
    # ------------------------------------------------------------------

    def fit(self, train_df, track_metadata=None, **kwargs: Any) -> None:
        super().fit(train_df, track_metadata=track_metadata, **kwargs)
        self._load_fallback()

    def _load_fallback(self) -> None:
        self._fb = {}
        if not self.fallback_datasets_dir:
            print(f"[{self.RECOMMENDER_NAME}] no fallback_datasets_dir — v0-only.")
            return
        root = Path(self.fallback_datasets_dir)
        files: list[str] = []
        for pat in _FALLBACK_GLOBS:
            files.extend(sorted(glob.glob(str(root / pat))))
        if not files:
            raise FileNotFoundError(
                f"[{self.RECOMMENDER_NAME}] no fallback parquets under {root}"
            )
        for fp in files:
            df = pl.read_parquet(fp, columns=["session_id", "turn", "track_ids"])
            for sid, turn, tids in zip(
                df["session_id"].to_list(),
                df["turn"].to_list(),
                df["track_ids"].to_list(),
            ):
                self._fb[(sid, int(turn))] = list(tids or [])
        print(
            f"[{self.RECOMMENDER_NAME}] loaded fallback for {len(self._fb)} "
            f"(session, turn) keys from {len(files)} parquet(s)."
        )

    # ------------------------------------------------------------------
    # recommend: v0 recs, then shortfall-fill from hybrid fallback
    # ------------------------------------------------------------------

    def recommend(
        self,
        context_df: pl.DataFrame,
        top_k: int = 20,
        remove_seen: bool = True,
        **kwargs: Any,
    ) -> pl.DataFrame:
        recs = super().recommend(
            context_df, top_k=top_k, remove_seen=remove_seen, **kwargs
        )

        target = min(int(top_k), self.fallback_out_n)
        out_tracks: list[list[str]] = []
        out_scores: list[list[float]] = []
        n_filled = 0

        for row in recs.iter_rows(named=True):
            tids = list(row["track_ids"] or [])
            scs = list(row["scores"] or [])
            if len(tids) < target:
                fb = self._fb.get((row["session_id"], int(row["turn"])))
                if fb:
                    seen = set(tids)
                    base = scs[-1] if scs else 0.0
                    rank = 0
                    for t in fb:
                        if len(tids) >= target:
                            break
                        if t in seen:
                            continue
                        seen.add(t)
                        tids.append(t)
                        rank += 1
                        scs.append(base - rank)  # strictly below v0's last
                    if rank:
                        n_filled += 1
            out_tracks.append(tids)
            out_scores.append(scs)

        if n_filled:
            print(f"[{self.RECOMMENDER_NAME}] fallback-filled {n_filled} rows.")

        return recs.with_columns(
            pl.Series("track_ids", out_tracks, dtype=pl.List(pl.Utf8)),
            pl.Series("scores", out_scores, dtype=pl.List(pl.Float64)),
        )

    # ------------------------------------------------------------------
    # persistence (extends v0; reload fallback on restore)
    # ------------------------------------------------------------------

    def _get_model_state(self) -> dict:
        st = super()._get_model_state()
        st.update({
            "fallback_datasets_dir": self.fallback_datasets_dir,
            "fallback_out_n": self.fallback_out_n,
        })
        return st

    def _set_model_state(self, state: dict) -> None:
        super()._set_model_state(state)
        self.fallback_datasets_dir = state.get("fallback_datasets_dir")
        self.fallback_out_n = int(state.get("fallback_out_n", 200))
        self._load_fallback()
