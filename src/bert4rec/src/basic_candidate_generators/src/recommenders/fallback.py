from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date

import numpy as np
import polars as pl

from .interactions import explode_music_turns, parse_date

class AbstractFallback(ABC):

    NAME = "AbstractFallback"

    @abstractmethod
    def fit(self, train_df: pl.DataFrame, track_metadata: pl.DataFrame | None = None) -> None: ...

    @abstractmethod
    def recommend_one(
        self,
        session_id: str,
        turn: int,
        session_date: date | None,
        top_k: int,
    ) -> tuple[list[str], list[float]]: ...

class PopularityFallback(AbstractFallback):

    NAME = "PopularityFallback"

    def __init__(self, log_scale: bool = True):
        self.log_scale = log_scale
        self._track_ids: list[str] = []
        self._popularity: np.ndarray | None = None
        self._release_dates: np.ndarray | None = None

    def fit(self, train_df: pl.DataFrame, track_metadata: pl.DataFrame | None = None) -> None:
        long = explode_music_turns(train_df)
        counts = long.group_by("track_id").len(name="cnt").sort("cnt", descending=True)
        self._track_ids = counts["track_id"].to_list()
        c = counts["cnt"].to_numpy().astype(np.float64)
        self._popularity = np.log1p(c) if self.log_scale else c

        if track_metadata is not None:
            md = (
                track_metadata.select(["track_id", "release_date"])
                .unique(subset=["track_id"])
                .with_columns(pl.col("release_date").str.strptime(pl.Date, "%Y-%m-%d", strict=False))
                .with_columns(
                    pl.when(pl.col("release_date").dt.year() > 0)
                    .then(pl.col("release_date"))
                    .otherwise(pl.lit(None, dtype=pl.Date))
                    .alias("release_date")
                )
            )
            lookup = dict(zip(md["track_id"].to_list(), md["release_date"].to_list()))
            self._release_dates = np.array(
                [np.datetime64(lookup[tid], "D") if lookup.get(tid) else np.datetime64("NaT", "D")
                 for tid in self._track_ids]
            )
        else:
            self._release_dates = None

    def recommend_one(
        self, session_id: str, turn: int, session_date: date | None, top_k: int
    ) -> tuple[list[str], list[float]]:
        if self._popularity is None:
            return [], []
        scores = self._popularity.copy()
        sd = parse_date(session_date)
        if self._release_dates is not None and sd is not None:
            sd64 = np.datetime64(sd, "D")
            mask = (self._release_dates > sd64) | np.isnat(self._release_dates)
            scores[mask] = -np.inf
        k = min(top_k, len(scores))
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        idx = [i for i in idx if scores[i] > -np.inf]
        return [self._track_ids[i] for i in idx], [float(scores[i]) for i in idx]
