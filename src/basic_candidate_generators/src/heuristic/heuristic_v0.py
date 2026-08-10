"""HeuristicRecommender — modular content-based CG.

Architecture:
  - TrackIndex: per-track artist/album/year/popularity + inverted maps.
  - SimpleCG: atomic single-signal candidate generator.
      Seed CGs build the candidate pool (artist-last, album-last, artist-any,
      album-any, decade-popular).
      Dense CGs score the pool (year proximity, pop-bucket match, global pop_z).
  - LinearSumAggregator: weighted-sum fusion of per-CG proposals → top-K.
  - HeuristicRecommender: framework-facing BaseRecommender wrapper. Hparams are
    flat kwargs so Optuna search_space + cv_best yamls work unchanged.

Per-session category/specificity multipliers boost the four artist/album CGs.
"""

from __future__ import annotations

import sys as _sys
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from numba import njit

from tqdm import tqdm

_HERE = Path(__file__).resolve()
_sys.path.insert(0, str(_HERE.parent.parent))

from BaseRecommender import BaseRecommender  # noqa: E402

@njit
def numba_accumulate_scores(indices: np.ndarray, scores: np.ndarray, n_tracks: int) -> np.ndarray:
    """Numba-compiled dense accumulation to replace the dictionary updates."""
    acc = np.zeros(n_tracks, dtype=np.float64)
    for i in range(len(indices)):
        acc[indices[i]] += scores[i]
    return acc

# ---------------------------------------------------------------------------
# TrackIndex
# ---------------------------------------------------------------------------

class TrackIndex:
    """Per-track metadata + inverted indexes for fast candidate generation."""

    def __init__(self, df: pl.DataFrame):
        n = df.shape[0]
        self.track_ids: list[str] = df["track_id"].to_list()
        self.id_to_idx = {tid: i for i, tid in enumerate(self.track_ids)}

        self.artist: list[str | None] = [None] * n
        self.album: list[str | None] = [None] * n
        self.year = np.zeros(n, dtype=np.int32)
        self.popularity = np.zeros(n, dtype=np.float32)
        self.pop_bucket = np.full(n, -1, dtype=np.int8)

        for i, r in enumerate(df.iter_rows(named=True)):
            aid_list = r.get("artist_id") or []
            artist = aid_list[0] if aid_list else None
            self.artist[i] = artist

            album_raw = r.get("album_name")
            if isinstance(album_raw, (list, tuple)):
                album_raw = album_raw[0] if album_raw else None
            if album_raw:
                an = str(album_raw).strip()
                if an and an.lower() not in {"", "unknown", "none"}:
                    self.album[i] = f"{artist or ''}::{an}"

            rd = r.get("release_date") or ""
            try:
                if rd and len(str(rd)) >= 4:
                    self.year[i] = int(str(rd)[:4])
            except (TypeError, ValueError):
                pass

            pop = r.get("popularity")
            if pop is not None:
                self.popularity[i] = float(pop)

        valid_pop = self.popularity[self.popularity > 0]
        if len(valid_pop) > 0:
            q33 = float(np.percentile(valid_pop, 33))
            q67 = float(np.percentile(valid_pop, 67))
            for i in range(n):
                p = self.popularity[i]
                if p <= 0:
                    self.pop_bucket[i] = -1
                elif p < q33:
                    self.pop_bucket[i] = 0
                elif p < q67:
                    self.pop_bucket[i] = 1
                else:
                    self.pop_bucket[i] = 2
            mu = float(valid_pop.mean())
            sigma = float(valid_pop.std()) or 1.0
            self.pop_z = (self.popularity - mu) / sigma
        else:
            self.pop_z = np.zeros(n, dtype=np.float32)

        a2i: dict[str, list[int]] = defaultdict(list)
        b2i: dict[str, list[int]] = defaultdict(list)
        for i in range(n):
            if self.artist[i]: a2i[self.artist[i]].append(i)
            if self.album[i]:  b2i[self.album[i]].append(i)
        self.artist_to_idxs = {k: np.asarray(v, dtype=np.int64) for k, v in a2i.items()}
        self.album_to_idxs  = {k: np.asarray(v, dtype=np.int64) for k, v in b2i.items()}

        decade: dict[int, list[int]] = defaultdict(list)
        for i in range(n):
            if self.year[i] > 0:
                decade[(int(self.year[i]) // 10) * 10].append(i)
        self.decade_to_top_idxs: dict[int, np.ndarray] = {}
        for d, idxs in decade.items():
            arr = np.asarray(idxs, dtype=np.int64)
            self.decade_to_top_idxs[d] = arr[np.argsort(-self.popularity[arr])]


# ---------------------------------------------------------------------------
# Resources bag + abstract bases
# ---------------------------------------------------------------------------

@dataclass
class CGResources:
    """Bundle of optional artefacts a SimpleCG may consume at fit time."""
    track_index: TrackIndex | None = None
    train_df: pl.DataFrame | None = None
    track_metadata: pl.DataFrame | None = None
    track_embeddings: np.ndarray | None = None
    track_emb_id_to_idx: dict[str, int] | None = None
    query_embeddings: dict[str, np.ndarray] | None = None
    extras: dict[str, Any] = field(default_factory=dict)


class BaseSimpleCG(ABC):
    """One narrow scoring signal over the track catalogue."""
    NAME: str = "BaseSimpleCG"
    REQUIRES: tuple[str, ...] = ("track_index",)
    IS_SEED:     bool = False
    IS_BOOSTABLE: bool = False

    def check_resources(self, res: CGResources) -> None:
        missing = [r for r in self.REQUIRES if getattr(res, r, None) is None]
        if missing:
            raise ValueError(f"[{self.NAME}] missing resources: {missing}")

    @abstractmethod
    def fit(self, resources: CGResources) -> None: ...

    @abstractmethod
    def propose(
        self,
        prior_idxs: list[int],
        cat: str,
        spec: str,
        session_ctx: dict | None = None,
    ) -> tuple[np.ndarray, np.ndarray]: ...


class BaseAggregator(ABC):
    """Combine per-CG proposals into a ranked top-K list."""
    NAME: str = "BaseAggregator"

    def fit(self, *args, **kwargs) -> None:
        return None

    @abstractmethod
    def combine(
        self,
        proposals: list[tuple[np.ndarray, np.ndarray] | None],
        top_k: int,
        n_tracks: int,
        forbidden: set[int] | None = None,
        weights: list[float] | None = None,
        candidate_features: dict[int, dict[str, float]] | None = None,
    ) -> tuple[list[int], list[float]]: ...


# ---------------------------------------------------------------------------
# Concrete SimpleCGs
# ---------------------------------------------------------------------------

class _CGBase(BaseSimpleCG):
    REQUIRES = ("track_index",)

    def __init__(self, base_weight: float = 1.0):
        self.base_weight = float(base_weight)
        self.idx: TrackIndex | None = None

    def fit(self, resources: CGResources) -> None:
        self.check_resources(resources)
        self.idx = resources.track_index


class ArtistLastCG(_CGBase):
    NAME = "artist_last"; IS_SEED = True; IS_BOOSTABLE = True
    def propose(self, prior_idxs, cat, spec, session_ctx=None):
        a = self.idx.artist[prior_idxs[-1]]
        if a is None or a not in self.idx.artist_to_idxs: return np.array([], dtype=np.int64), np.array([], dtype=np.float64)
        idxs = self.idx.artist_to_idxs[a]
        return idxs, np.ones(len(idxs), dtype=np.float64)


class AlbumLastCG(_CGBase):
    NAME = "album_last"; IS_SEED = True; IS_BOOSTABLE = True
    def propose(self, prior_idxs, cat, spec, session_ctx=None):
        al = self.idx.album[prior_idxs[-1]]
        if al is None or al not in self.idx.album_to_idxs: return np.array([], dtype=np.int64), np.array([], dtype=np.float64)
        idxs = self.idx.album_to_idxs[al]
        return idxs, np.ones(len(idxs), dtype=np.float64)


class ArtistAnyCG(_CGBase):
    NAME = "artist_any"; IS_SEED = True; IS_BOOSTABLE = True
    def propose(self, prior_idxs, cat, spec, session_ctx=None):
        prior_artists = session_ctx["prior_artists"] if session_ctx else None
        if not prior_artists: return np.array([], dtype=np.int64), np.array([], dtype=np.float64)
        out_idxs = [self.idx.artist_to_idxs[a] for a in prior_artists if a in self.idx.artist_to_idxs]
        if not out_idxs: return np.array([], dtype=np.int64), np.array([], dtype=np.float64)
        merged = np.concatenate(out_idxs)
        return merged, np.ones(len(merged), dtype=np.float64)


class AlbumAnyCG(_CGBase):
    NAME = "album_any"; IS_SEED = True; IS_BOOSTABLE = True
    def propose(self, prior_idxs, cat, spec, session_ctx=None):
        prior_albums = session_ctx["prior_albums"] if session_ctx else None
        if not prior_albums: return np.array([], dtype=np.int64), np.array([], dtype=np.float64)
        out_idxs = [self.idx.album_to_idxs[al] for al in prior_albums if al in self.idx.album_to_idxs]
        if not out_idxs: return np.array([], dtype=np.int64), np.array([], dtype=np.float64)
        merged = np.concatenate(out_idxs)
        return merged, np.ones(len(merged), dtype=np.float64)


class DecadeSeedCG(_CGBase):
    """Inject top-N popular tracks from decade ±10 into pool. Score 0.0."""
    NAME = "decade_seed"; IS_SEED = True; IS_BOOSTABLE = False
    def __init__(self, top_n: int = 150):
        super().__init__(base_weight=0.0)
        self.top_n = int(top_n)
    def propose(self, prior_idxs, cat, spec, session_ctx=None):
        last_year = int(self.idx.year[prior_idxs[-1]])
        if last_year <= 0: return np.array([], dtype=np.int64), np.array([], dtype=np.float64)
        decade = (last_year // 10) * 10
        out_idxs = []
        for d in (decade - 10, decade, decade + 10):
            arr = self.idx.decade_to_top_idxs.get(d)
            if arr is None: continue
            out_idxs.append(arr[: self.top_n])
        if not out_idxs: return np.array([], dtype=np.int64), np.array([], dtype=np.float64)
        merged = np.concatenate(out_idxs)
        return merged, np.zeros(len(merged), dtype=np.float64)


class YearProxCG(_CGBase):
    """Dense: exp(-|Δyear|/5) wrt last track."""
    NAME = "year"
    def propose(self, prior_idxs, cat, spec, session_ctx=None):
        pool = session_ctx.get("pool_idxs") if session_ctx else None
        if pool is None or len(pool) == 0: return np.array([], dtype=np.int64), np.array([], dtype=np.float64)
        last_year = int(self.idx.year[prior_idxs[-1]])
        if last_year <= 0: return np.array([], dtype=np.int64), np.array([], dtype=np.float64)
        cand_year = self.idx.year[pool]
        valid = cand_year > 0
        if not valid.any(): return np.array([], dtype=np.int64), np.array([], dtype=np.float64)
        scores = np.zeros(len(pool), dtype=np.float64)
        dy = np.abs(cand_year[valid].astype(np.int32) - last_year).astype(np.float64)
        scores[valid] = np.exp(-dy / 5.0)
        nz = scores > 0
        return pool[nz], scores[nz]


class PopMatchCG(_CGBase):
    """Dense: +1 if candidate shares last track's popularity bucket."""
    NAME = "pop_match"
    def propose(self, prior_idxs, cat, spec, session_ctx=None):
        pool = session_ctx.get("pool_idxs") if session_ctx else None
        if pool is None or len(pool) == 0: return np.array([], dtype=np.int64), np.array([], dtype=np.float64)
        last_b = int(self.idx.pop_bucket[prior_idxs[-1]])
        if last_b < 0: return np.array([], dtype=np.int64), np.array([], dtype=np.float64)
        mask = self.idx.pop_bucket[pool] == last_b
        return pool[mask], np.ones(mask.sum(), dtype=np.float64)


class PopZCG(_CGBase):
    """Dense: global popularity prior, +pop_z per candidate."""
    NAME = "pop_z"
    def propose(self, prior_idxs, cat, spec, session_ctx=None):
        pool = session_ctx.get("pool_idxs") if session_ctx else None
        if pool is None or len(pool) == 0: return np.array([], dtype=np.int64), np.array([], dtype=np.float64)
        z = self.idx.pop_z[pool].astype(np.float64)
        nz = z != 0
        return pool[nz], z[nz]


# ---------------------------------------------------------------------------
# Aggregator: weighted linear sum
# ---------------------------------------------------------------------------

class LinearSumAggregator(BaseAggregator):
    NAME = "LinearSum"

    def combine(self, proposals, top_k, n_tracks, forbidden=None, weights=None, candidate_features=None):
        if weights is None or len(weights) != len(proposals):
            raise ValueError("LinearSumAggregator needs per-CG weights matching proposals length")

        flat_indices = []
        flat_scores = []

        for idxs_scores, w in zip(proposals, weights):
            if idxs_scores is None or w == 0.0:
                continue
            idxs, scs = idxs_scores
            if len(idxs) > 0:
                flat_indices.append(idxs)
                flat_scores.append(scs * w)

        if not flat_indices:
            return [], []

        all_indices = np.concatenate(flat_indices)
        all_scores = np.concatenate(flat_scores)

        acc = numba_accumulate_scores(all_indices, all_scores, n_tracks)

        pool_arr = np.unique(all_indices)

        if forbidden:
            valid_mask = np.isin(pool_arr, list(forbidden), invert=True)
            pool_arr = pool_arr[valid_mask]

        if len(pool_arr) == 0:
            return [], []

        pool_scores = acc[pool_arr]

        k = min(int(top_k), len(pool_arr))
        if k < len(pool_arr):
            order = np.argpartition(-pool_scores, k - 1)[:k]
        else:
            order = np.arange(len(pool_arr))

        order = order[np.argsort(-pool_scores[order])]

        return pool_arr[order].tolist(), pool_scores[order].tolist()


# ---------------------------------------------------------------------------
# Framework-facing recommender
# ---------------------------------------------------------------------------

class HeuristicRecommender(BaseRecommender):
    """Modular heuristic CG. Flat hparams preserved for Optuna + cv_best yamls."""
    RECOMMENDER_NAME = "Heuristic"

    def __init__(
        self,
        # Feature weights
        album_last: float = 3.0,
        artist_last: float = 2.5,
        album_any: float = 1.5,
        artist_any: float = 1.2,
        year: float = 0.6,
        pop_match: float = 0.4,
        pop_z: float = 0.05,
        # Category multipliers
        cat_A: float = 1.10, cat_B: float = 1.00, cat_C: float = 1.20,
        cat_D: float = 1.10, cat_E: float = 1.10, cat_F: float = 1.15,
        cat_G: float = 1.00, cat_H: float = 1.15, cat_I: float = 0.65,
        cat_J: float = 0.75, cat_K: float = 0.85,
        # Specificity multipliers
        spec_HH: float = 1.20, spec_LH: float = 1.05,
        spec_HL: float = 1.00, spec_LL: float = 1.00,
        # Pool size
        decade_top_n: int = 150,
        # Framework
        urm_mode: str = "session",
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        # Persistable hparam dicts
        self.weights = {
            "album_last":  album_last,  "artist_last": artist_last,
            "album_any":   album_any,   "artist_any":  artist_any,
            "year":        year,        "pop_match":   pop_match,
            "pop_z":       pop_z,
        }
        self.cat_mult = {
            "A": cat_A, "B": cat_B, "C": cat_C, "D": cat_D, "E": cat_E,
            "F": cat_F, "G": cat_G, "H": cat_H, "I": cat_I, "J": cat_J,
            "K": cat_K,
        }
        self.spec_mult = {
            "HH": spec_HH, "LH": spec_LH, "HL": spec_HL, "LL": spec_LL,
            "": 1.0, "?": 1.0,
        }
        self.decade_top_n = int(decade_top_n)
        self.urm_mode = urm_mode

        self.resources: CGResources | None = None
        self.simple_cgs: list[BaseSimpleCG] = []
        self.aggregator: BaseAggregator = LinearSumAggregator()
        self._build_cgs()

    def _build_cgs(self) -> None:
        w = self.weights
        self.simple_cgs = [
            AlbumLastCG(base_weight=w["album_last"]),
            ArtistLastCG(base_weight=w["artist_last"]),
            AlbumAnyCG(base_weight=w["album_any"]),
            ArtistAnyCG(base_weight=w["artist_any"]),
            DecadeSeedCG(top_n=self.decade_top_n),
            YearProxCG(base_weight=w["year"]),
            PopMatchCG(base_weight=w["pop_match"]),
            PopZCG(base_weight=w["pop_z"]),
        ]

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(
        self,
        train_df: pl.DataFrame | None,
        track_metadata: pl.DataFrame | None = None,
        track_embeddings: np.ndarray | None = None,
        track_emb_id_to_idx: dict[str, int] | None = None,
        query_embeddings: dict[str, np.ndarray] | None = None,
        **kwargs: Any,
    ) -> None:
        if track_metadata is None:
            raise ValueError(f"[{self.RECOMMENDER_NAME}] track_metadata required to fit.")
        print(f"[{self.RECOMMENDER_NAME}] Building TrackIndex from metadata...")
        idx = TrackIndex(track_metadata)
        print(
            f"[{self.RECOMMENDER_NAME}] indexed {len(idx.track_ids)} tracks, "
            f"{len(idx.artist_to_idxs)} artists, {len(idx.album_to_idxs)} albums."
        )
        self.resources = CGResources(
            track_index=idx, train_df=train_df, track_metadata=track_metadata,
            track_embeddings=track_embeddings, track_emb_id_to_idx=track_emb_id_to_idx,
            query_embeddings=query_embeddings, extras=kwargs,
        )
        for cg in tqdm(self.simple_cgs, desc="Fitting", leave=False):
            cg.check_resources(self.resources)
            cg.fit(self.resources)

    # ------------------------------------------------------------------
    # recommend
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_cat_spec(goal) -> tuple[str, str]:
        if isinstance(goal, dict):
            return (
                str(goal.get("category", "?") or "?"),
                str(goal.get("specificity", "?") or "?"),
            )
        return "?", "?"

    def recommend(
        self,
        context_df: pl.DataFrame,
        top_k: int = 20,
        remove_seen: bool = True,
        max_future_years: float | None = None,  # noqa: ARG002 (framework signature)
        **kwargs: Any,
    ) -> pl.DataFrame:
        if self.resources is None or self.resources.track_index is None:
            raise RuntimeError("Recommender not fitted")
        if "target_turn" not in context_df.columns:
            raise ValueError("context_df missing 'target_turn'. Use build_context_df().")

        idx = self.resources.track_index
        has_goal = "conversation_goal" in context_df.columns

        meta_cols = ["session_id", "user_id", "target_turn"]
        if has_goal: meta_cols.append("conversation_goal")
        session_meta = context_df.select(meta_cols).unique(subset=["session_id"])

        ctx_map: dict[str, list[str]] = {}
        if "track_id" in context_df.columns and context_df.height > 0:
            grouped = (
                context_df.filter(pl.col("track_id").is_not_null())
                .group_by("session_id")
                .agg(pl.col("track_id"))
            )
            ctx_map = dict(zip(grouped["session_id"].to_list(), grouped["track_id"].to_list()))

        out_sid: list[str] = []; out_uid: list[str] = []
        out_turn: list[int] = []; out_tracks: list[list[str]] = []; out_scores: list[list[float]] = []

        n_tracks = len(idx.track_ids)

        #for row in tqdm(session_meta.iter_rows(named=True), desc="Recommending", leave=False, total=session_meta.shape[0]):
        for row in session_meta.iter_rows(named=True):
            sid = row["session_id"]; uid = row["user_id"]; T = int(row["target_turn"])
            cat, spec = (
                self._extract_cat_spec(row["conversation_goal"]) if has_goal else ("?", "?")
            )
            prior_tids = ctx_map.get(sid, [])
            prior_idxs = [idx.id_to_idx[t] for t in prior_tids if t in idx.id_to_idx]

            if not prior_idxs:
                out_sid.append(sid); out_uid.append(uid); out_turn.append(T)
                out_tracks.append([]); out_scores.append([])
                continue

            prior_artists = {idx.artist[i] for i in prior_idxs if idx.artist[i] is not None}
            prior_albums  = {idx.album[i]  for i in prior_idxs if idx.album[i]  is not None}
            sess_ctx = {
                "last_idx": prior_idxs[-1],
                "prior_artists": prior_artists,
                "prior_albums": prior_albums,
                "pool_idxs": None,
            }

            # Stage 1: seed CGs → pool
            proposals: list[tuple[np.ndarray, np.ndarray] | None] = [None] * len(self.simple_cgs)
            for i, cg in enumerate(self.simple_cgs):
                if cg.IS_SEED:
                    proposals[i] = cg.propose(prior_idxs, cat, spec, sess_ctx)

            pool_sub_arrays = [p[0] for p in proposals if p is not None and len(p[0]) > 0]
            if not pool_sub_arrays:
                out_sid.append(sid); out_uid.append(uid); out_turn.append(T)
                out_tracks.append([]); out_scores.append([])
                continue

            pool_arr = np.unique(np.concatenate(pool_sub_arrays))
            if remove_seen:
                pool_arr = np.setdiff1d(pool_arr, prior_idxs, assume_unique=True)

            if len(pool_arr) == 0:
                out_sid.append(sid); out_uid.append(uid); out_turn.append(T)
                out_tracks.append([]); out_scores.append([])
                continue

            sess_ctx["pool_idxs"] = pool_arr

            # Stage 2: dense CGs over pool
            for i, cg in enumerate(self.simple_cgs):
                if not cg.IS_SEED:
                    proposals[i] = cg.propose(prior_idxs, cat, spec, sess_ctx)

            boost = self.cat_mult.get(cat, 1.0) * self.spec_mult.get(spec, 1.0)
            weights = [
                (cg.base_weight * boost if cg.IS_BOOSTABLE else cg.base_weight)
                for cg in self.simple_cgs
            ]
            forbid = set(prior_idxs) if remove_seen else None
            ranked, scores = self.aggregator.combine(
                proposals, top_k=top_k, n_tracks=n_tracks, forbidden=forbid, weights=weights,
            )
            out_sid.append(sid); out_uid.append(uid); out_turn.append(T)
            out_tracks.append([idx.track_ids[i] for i in ranked])
            out_scores.append(scores)

        return pl.DataFrame(
            {"session_id": out_sid, "user_id": out_uid, "turn": out_turn,
             "track_ids": out_tracks, "scores": out_scores},
            schema={"session_id": pl.Utf8, "user_id": pl.Utf8, "turn": pl.Int64,
                    "track_ids": pl.List(pl.Utf8), "scores": pl.List(pl.Float64)},
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _get_model_state(self) -> dict:
        return {
            "track_index": self.resources.track_index if self.resources else None,
            "weights":     self.weights,
            "cat_mult":    self.cat_mult,
            "spec_mult":   self.spec_mult,
            "decade_top_n": self.decade_top_n,
            "urm_mode":    self.urm_mode,
        }

    def _set_model_state(self, state: dict) -> None:
        self.weights = state.get("weights", {})
        self.cat_mult = state.get("cat_mult", {})
        self.spec_mult = state.get("spec_mult", {})
        self.decade_top_n = int(state.get("decade_top_n", 150))
        self.urm_mode = state.get("urm_mode", "session")
        self.aggregator = LinearSumAggregator()
        self._build_cgs()
        track_index = state.get("track_index") or state.get("idx")  # back-compat key
        if track_index is not None:
            self.resources = CGResources(track_index=track_index)
            for cg in self.simple_cgs:
                cg.fit(self.resources)
        else:
            self.resources = None
