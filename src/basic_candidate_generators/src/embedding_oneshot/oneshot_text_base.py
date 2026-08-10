"""OneShotTextCG — clean turn-1 lexical retrieval (query_text → track docs).

Standalone reimplementation of the TF-IDF / BM25 candidate generators, stripped
of the UserRecommender CF-blend / URM machinery that is dead at turn 1 (no
session history). The index is built over track-metadata documents; queries are
the precomputed rich `query_text` pulled from the splitK query caches (keyed by
(session_id, turn_number)), so this needs no text-mode bundle plumbing — it runs
under standard inference like the other one-shot CGs.

Subclasses implement:
    _build_index(docs)           — fit the retrieval index over track docs
    _score_batch(query_texts)    — (n_queries, n_tracks) float32, one BLAS call
"""
from __future__ import annotations

import glob
import time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import scipy.sparse as sp

from BaseRecommender import BaseRecommender

from recommenders.interactions import parse_date

from embedding_based.dense_query_cg import _build_release_dates

# query_text is identical across encoder sizes (it is the assembled text), so a
# single (8B) cache root suffices as the source.
_QUERY_TEXT_ROOT = "models/retrieval_text_towers/Qwen__Qwen3-Embedding-8B"

# track-metadata fields used to build retrieval documents, in doc order. The
# flat doc (`_track_doc`) concatenates them; BM25F-lite indexes them separately.
_DOC_FIELDS = ["artist_name", "track_name", "tag_list", "album_name"]


def _field_text(row: dict, col: str) -> str:
    """Whitespace-joined text of one metadata field (list- or str-valued)."""
    v = row.get(col) or []
    if isinstance(v, list):
        return " ".join(str(x) for x in v if x)
    return str(v) if v else ""


class OneShotTextCG(BaseRecommender):
    RECOMMENDER_NAME = "OneShotTextCG"

    def __init__(
        self,
        query_text_root: str = _QUERY_TEXT_ROOT,
        max_features: int = 50_000,
        max_future_years: float = 2.0,
        urm_mode: str = "session",
        template_stopwords: bool = False,   # E3.4: drop high-DF query boilerplate
        stopword_df_q: float = 0.5,         #       term-DF fraction → stopword
        colisten_top_n: int = 0,            # E3.6: append top-N co-listen artists
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.query_text_root = query_text_root
        self.max_features = int(max_features)
        self.max_future_years = float(max_future_years)
        self.urm_mode = urm_mode
        self.template_stopwords = bool(template_stopwords)
        self.stopword_df_q = float(stopword_df_q)
        self.colisten_top_n = int(colisten_top_n)

        self.track_ids: np.ndarray | None = None
        self.track_to_idx: dict[str, int] = {}
        self.release_dates: np.ndarray | None = None
        self.query_text_by_key: dict[tuple[str, int], str] = {}
        self.field_docs: dict[str, list[str]] = {}   # per-field docs (BM25F-lite)
        self._stop_words: list[str] | None = None

    # ------------------------------------------------------------------
    # subclass hooks
    # ------------------------------------------------------------------
    def _build_index(self, docs: list[str]) -> None:
        raise NotImplementedError

    def _score_batch(self, query_texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------
    def fit(self, train_df, track_metadata: pl.DataFrame | None = None,
            colisten_df: pl.DataFrame | None = None, **kwargs: Any) -> None:
        if track_metadata is None:
            raise ValueError(f"[{self.RECOMMENDER_NAME}] track_metadata required (catalogue)")
        t0 = time.time()
        self.track_ids = track_metadata["track_id"].to_numpy()
        self.track_to_idx = {t: i for i, t in enumerate(self.track_ids)}
        self.release_dates = _build_release_dates(self.track_ids, track_metadata)
        self._load_query_text()

        # Per-field + flat docs. The flat doc reproduces `_track_doc` exactly
        # (same field order) so unenriched runs match the legacy CGs.
        n = len(self.track_ids)
        field_docs = {c: [""] * n for c in _DOC_FIELDS}
        flat_docs = [""] * n
        for row in track_metadata.iter_rows(named=True):
            i = self.track_to_idx[row["track_id"]]
            parts = []
            for c in _DOC_FIELDS:
                txt = _field_text(row, c)
                field_docs[c][i] = txt
                if txt:
                    parts.append(txt)
            flat_docs[i] = " ".join(parts)

        if self.colisten_top_n > 0 and colisten_df is not None:
            extra = self._build_colisten_docs(colisten_df, field_docs["artist_name"])
            for i, e in extra.items():
                if e:
                    flat_docs[i] = (flat_docs[i] + " " + e).strip()
                    field_docs["artist_name"][i] = (
                        field_docs["artist_name"][i] + " " + e).strip()

        self.field_docs = field_docs
        self._compute_query_stopwords(train_df)
        self._build_index(flat_docs)
        print(f"[{self.RECOMMENDER_NAME}] fit in {time.time()-t0:.1f}s — "
              f"{n} tracks, {len(self.query_text_by_key)} cached queries, "
              f"{len(self._stop_words or [])} stopwords, colisten_top_n={self.colisten_top_n}")

    def _compute_query_stopwords(self, train_df) -> None:
        """E3.4 — terms appearing in ≥ `stopword_df_q` of training queries.

        Computed only over the fold's training-session turn-1 queries (fold-safe);
        the resulting list is reused at inference. None when disabled."""
        self._stop_words = None
        if not self.template_stopwords or train_df is None or not self.query_text_by_key:
            return
        keys = set(zip(train_df["session_id"].to_list(), train_df["turn_number"].to_list()))
        texts = [self.query_text_by_key[k] for k in keys if self.query_text_by_key.get(k)]
        if len(texts) < 20:
            return
        from sklearn.feature_extraction.text import CountVectorizer
        cv = CountVectorizer()
        X = cv.fit_transform(texts)
        dfrac = np.asarray((X > 0).sum(axis=0)).ravel() / X.shape[0]
        vocab = cv.get_feature_names_out()
        sw = [str(vocab[i]) for i in np.where(dfrac >= self.stopword_df_q)[0]]
        self._stop_words = sw or None

    def _build_colisten_docs(self, colisten_df: pl.DataFrame,
                             artist_docs: list[str]) -> dict[int, str]:
        """E3.6 — per track, the artist names of its top-N co-listened tracks.

        Co-occurrence = tracks sharing a training session (all turns). Needs the
        UNFILTERED fold (multi-turn) — the launcher passes it as `colisten_df`."""
        rows = colisten_df.filter(pl.col("track_id").is_not_null())
        sids = rows["session_id"].to_list()
        tidx = np.fromiter((self.track_to_idx.get(t, -1) for t in rows["track_id"].to_list()),
                           dtype=np.int64, count=rows.height)
        sess_codes = {s: i for i, s in enumerate(dict.fromkeys(sids))}
        srow = np.fromiter((sess_codes[s] for s in sids), dtype=np.int64, count=len(sids))
        ok = tidx >= 0
        M = sp.csr_matrix(
            (np.ones(int(ok.sum()), dtype=np.float32), (srow[ok], tidx[ok])),
            shape=(len(sess_codes), len(self.track_ids)))
        co = (M.T @ M).tocsr()
        co.setdiag(0)
        co.eliminate_zeros()
        out: dict[int, str] = {}
        for i in range(co.shape[0]):
            s, e = co.indptr[i], co.indptr[i + 1]
            if e == s:
                continue
            cols, vals = co.indices[s:e], co.data[s:e]
            top = cols[np.argsort(-vals)[: self.colisten_top_n]]
            out[i] = " ".join(artist_docs[j] for j in top if artist_docs[j])
        return out

    def _load_query_text(self) -> None:
        root = Path(self.query_text_root)
        dirs = sorted(d for d in root.glob("dense_*query*")
                      if (Path(d) / "query_meta.parquet").exists())
        if not dirs:
            raise FileNotFoundError(
                f"[{self.RECOMMENDER_NAME}] no query caches under {root}")
        store: dict[tuple[str, int], str] = {}
        for d in dirs:
            meta = pl.read_parquet(Path(d) / "query_meta.parquet",
                                   columns=["session_id", "turn_number", "query_text"])
            for sid, tn, qt in zip(meta["session_id"].to_list(),
                                   meta["turn_number"].to_list(),
                                   meta["query_text"].to_list()):
                store[(sid, int(tn))] = qt or ""
        self.query_text_by_key = store

    # ------------------------------------------------------------------
    # recommend
    # ------------------------------------------------------------------
    def recommend(
        self, context_df: pl.DataFrame, top_k: int = 200, remove_seen: bool = True,
        max_future_years: float | None = None, **kwargs: Any,
    ) -> pl.DataFrame:
        if self.track_ids is None:
            raise RuntimeError("Recommender not fitted")
        if "target_turn" not in context_df.columns:
            raise ValueError("context_df missing 'target_turn'. Use build_context_df().")
        if max_future_years is None:
            max_future_years = self.max_future_years

        meta = context_df.select(
            ["session_id", "user_id", "session_date", "target_turn"]
        ).unique(subset=["session_id"])

        ctx_map: dict[str, list[str]] = {}
        if "track_id" in context_df.columns and context_df.height > 0:
            grp = (context_df.filter(pl.col("track_id").is_not_null())
                   .group_by("session_id").agg(pl.col("track_id")))
            ctx_map = dict(zip(grp["session_id"].to_list(), grp["track_id"].to_list()))

        rows = meta.to_dicts()
        hit_rows: list[int] = []
        queries: list[str] = []
        for ri, r in enumerate(rows):
            qt = self.query_text_by_key.get((r["session_id"], int(r["target_turn"])))
            if qt:
                hit_rows.append(ri); queries.append(qt)

        out_tracks: list[list[str]] = [[] for _ in rows]
        out_scores: list[list[float]] = [[] for _ in rows]
        if hit_rows:
            S = self._score_batch(queries)               # (H, n_tracks)
            n_tracks = len(self.track_ids)
            for li, ri in enumerate(hit_rows):
                r = rows[ri]
                scores = S[li].astype(np.float64)
                sd = parse_date(r["session_date"])
                if sd is not None:
                    cutoff = np.datetime64(sd, "D") + np.timedelta64(
                        int(float(max_future_years) * 365), "D")
                    bad = (self.release_dates > cutoff) & ~np.isnat(self.release_dates)
                    scores[bad] = -np.inf
                if remove_seen:
                    for t in ctx_map.get(r["session_id"], []):
                        j = self.track_to_idx.get(t)
                        if j is not None:
                            scores[j] = -np.inf
                out_tracks[ri], out_scores[ri] = self._topk(scores, top_k, n_tracks)

        return pl.DataFrame(
            {"session_id": [r["session_id"] for r in rows],
             "user_id": [r["user_id"] for r in rows],
             "turn": [r["target_turn"] for r in rows],
             "track_ids": out_tracks, "scores": out_scores},
            schema={"session_id": pl.Utf8, "user_id": pl.Utf8, "turn": pl.Int64,
                    "track_ids": pl.List(pl.Utf8), "scores": pl.List(pl.Float64)},
        )

    def _topk(self, scores: np.ndarray, top_k: int, n_tracks: int):
        finite = int(np.isfinite(scores).sum())
        if finite == 0:
            return [], []
        k = min(top_k, finite)
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        return [self.track_ids[i] for i in idx], [float(scores[i]) for i in idx]

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------
    def _get_model_state(self) -> dict:
        return {
            "query_text_root": self.query_text_root, "max_features": self.max_features,
            "max_future_years": self.max_future_years, "urm_mode": self.urm_mode,
            "template_stopwords": self.template_stopwords,
            "stopword_df_q": self.stopword_df_q, "colisten_top_n": self.colisten_top_n,
            "track_ids": self.track_ids, "track_to_idx": self.track_to_idx,
            "release_dates": self.release_dates,
            "query_text_by_key": self.query_text_by_key,
            "field_docs": self.field_docs, "_stop_words": self._stop_words,
        }

    def _set_model_state(self, state: dict) -> None:
        for k, v in state.items():
            setattr(self, k, v)
