"""Text-based Candidate Generators.

Use conversational signals (user_query, conversation_goal, user_profile) to
retrieve items via TF-IDF or BM25 against track metadata text documents.

Scoring is batched: one sparse matmul for all sessions → BLAS multi-threaded,
no per-session Python loop overhead.

Optional CF blend: text scores are combined with a lazy CF signal computed
from the session's context tracks (profile → URM fold-in similarity).

Classes:
  TextCGRecommender   — abstract base (batched scoring interface)
  TFIDFTextCG         — TF-IDF cosine similarity
  BM25TextCG          — BM25 retrieval (precomputed matrix)
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import polars as pl
import scipy.sparse as sp
from scipy.sparse import csr_matrix

from .interactions import parse_date
from .user_base import UserRecommender


# ---------------------------------------------------------------------------
# Document / query builders
# ---------------------------------------------------------------------------

def _track_doc(row: dict) -> str:
    """Build text document for one track from metadata row."""
    parts = []
    for col in ["artist_name", "track_name", "tag_list", "album_name"]:
        v = row.get(col) or []
        if isinstance(v, list):
            parts.extend(v)
        elif isinstance(v, str):
            parts.append(v)
    return " ".join(str(p) for p in parts if p)


def _session_query(row: dict) -> str:
    """Build query string from a session row (GT turn).

    Prefers a precomputed `query_text` (the rich, uniformly-assembled query
    attached from the query-tower bundle); falls back to assembling from the
    raw conversation fields when absent.
    """
    qt = row.get("query_text")
    if qt:
        return qt
    parts = [row.get("user_query") or ""]
    goal = row.get("conversation_goal") or {}
    if isinstance(goal, dict):
        parts.append(goal.get("listener_goal") or "")
    profile = row.get("user_profile") or {}
    if isinstance(profile, dict):
        parts.append(profile.get("preferred_musical_culture") or "")
        parts.append(profile.get("preferred_language") or "")
    return " ".join(p for p in parts if p)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class TextCGRecommender(UserRecommender):
    """Abstract base for text-based candidate generators.

    Subclasses implement:
      _build_text_index(docs)      — build retrieval index from track docs
      _text_scores_batch(queries)  — (n_queries, n_tracks) float32 via one BLAS call
    """

    def __init__(self, cf_weight: float = 0.0, **kwargs: Any):
        super().__init__(**kwargs)
        self.cf_weight = cf_weight

    def _fit_model(self, urm: csr_matrix) -> None:
        pass

    def fit(self, train_df, track_metadata=None, **kw):
        super().fit(train_df, track_metadata=track_metadata, **kw)
        if track_metadata is not None:
            t0 = time.time()
            n = self.id_map.n_tracks
            docs = [""] * n
            for row in track_metadata.iter_rows(named=True):
                idx = self.id_map.track_to_idx.get(row["track_id"])
                if idx is not None:
                    docs[idx] = _track_doc(row)
            self._build_text_index(docs)
            print(f"[{self.RECOMMENDER_NAME}] Text index built in {time.time() - t0:.1f}s")

    def _build_text_index(self, docs: list[str]) -> None:
        raise NotImplementedError

    def _text_scores_batch(self, queries: list[str]) -> np.ndarray:
        """Return (n_queries, n_tracks) float32 — one BLAS call for all sessions."""
        raise NotImplementedError

    def _cf_scores(self, ctx_tracks: list[str]) -> np.ndarray:
        profile = self._build_profile_vector(ctx_tracks)
        if profile.nnz == 0:
            return np.zeros(self.id_map.n_tracks, dtype=np.float32)
        sims = self.urm @ profile.T
        return np.asarray((self.urm.T @ sims).todense()).ravel().astype(np.float32)

    def recommend_text(
        self,
        sess_info: pl.DataFrame,
        top_k: int = 100,
        remove_seen: bool = True,
    ) -> pl.DataFrame:
        """Score all sessions in batch, return recommendations DataFrame."""
        rows = sess_info.to_dicts()
        queries = [_session_query(r) for r in rows]

        t0 = time.time()
        all_text_scores = self._text_scores_batch(queries)  # (n_sess, n_tracks)
        print(f"  [{self.RECOMMENDER_NAME}] batch text score: {time.time() - t0:.1f}s")

        out_sid, out_uid, out_turn, out_tracks, out_scores, out_gt = [], [], [], [], [], []

        for i, row in enumerate(rows):
            sd = parse_date(row.get("session_date"))
            mask = self._filter_candidate_mask(sd, self.max_future_years)
            ctx_tracks = [t for t in (row.get("ctx_tracks") or []) if t is not None]
            seen_idxs = {
                self.id_map.track_to_idx[t]
                for t in ctx_tracks
                if t in self.id_map.track_to_idx
            }

            text_s = all_text_scores[i].astype(np.float64)

            if self.cf_weight > 0 and ctx_tracks:
                cf_s = self._cf_scores(ctx_tracks).astype(np.float64)
                cf_max = cf_s.max() or 1.0
                txt_max = text_s.max() or 1.0
                scores = (
                    self.cf_weight * cf_s / cf_max
                    + (1.0 - self.cf_weight) * text_s / txt_max
                )
            else:
                scores = text_s

            recs, scs = self._topk_from_scores(scores, seen_idxs, top_k, mask, remove_seen)
            out_sid.append(row["session_id"])
            out_uid.append(row["user_id"])
            out_turn.append(row["turn_number"])
            out_tracks.append(recs)
            out_scores.append(scs)
            out_gt.append(row.get("track_id"))

        return pl.DataFrame({
            "session_id": out_sid,
            "user_id": out_uid,
            "turn": out_turn,
            "track_ids": out_tracks,
            "scores": out_scores,
            "gt_track_id": out_gt,
        })


# ---------------------------------------------------------------------------
# TF-IDF
# ---------------------------------------------------------------------------

class TFIDFTextCG(TextCGRecommender):
    """TF-IDF cosine similarity between session queries and track documents."""

    RECOMMENDER_NAME = "TFIDF_TextCG"

    def __init__(self, max_features: int = 50_000, ngram_max: int = 2, **kwargs: Any):
        super().__init__(**kwargs)
        self.max_features = max_features
        self.ngram_max = ngram_max
        self._vec = None
        self._item_mat: sp.csr_matrix | None = None  # (n_tracks, vocab)

    def _build_text_index(self, docs: list[str]) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        self._vec = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=(1, self.ngram_max),
            sublinear_tf=True,
        )
        self._item_mat = self._vec.fit_transform(docs)  # (n_tracks, vocab)

    def _text_scores_batch(self, queries: list[str]) -> np.ndarray:
        Q = self._vec.transform(queries)  # (n_sess, vocab) sparse
        return (Q @ self._item_mat.T).toarray().astype(np.float32)


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------

class BM25TextCG(TextCGRecommender):
    """BM25 retrieval — precomputed B matrix for O(1) per-token query scoring."""

    RECOMMENDER_NAME = "BM25_TextCG"

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        max_features: int = 50_000,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.k1 = k1
        self.b = b
        self.max_features = max_features
        self._B: sp.csr_matrix | None = None   # (n_tracks, vocab) precomputed BM25
        self._vec = None                         # CountVectorizer for query transform

    def _build_text_index(self, docs: list[str]) -> None:
        from sklearn.feature_extraction.text import CountVectorizer

        cv = CountVectorizer(max_features=self.max_features)
        tf = cv.fit_transform(docs).astype(np.float32)  # (n_tracks, vocab)
        n = tf.shape[0]

        df = np.asarray((tf > 0).sum(axis=0)).ravel()
        idf = np.log((n - df + 0.5) / (df + 0.5) + 1.0).astype(np.float32)

        dl = np.asarray(tf.sum(axis=1)).ravel()
        dl_norm = (1.0 - self.b + self.b * dl / (dl.mean() or 1.0)).astype(np.float32)

        # Precompute B[i,t] = tf*(k1+1)/(tf+k1*dl_norm_i) * idf[t] on sparse COO
        coo = tf.tocoo()
        dl_per_nnz = dl_norm[coo.row]
        bm25_vals = (
            coo.data * (self.k1 + 1.0)
            / (coo.data + self.k1 * dl_per_nnz)
            * idf[coo.col]
        )
        self._B = sp.csr_matrix(
            (bm25_vals, (coo.row, coo.col)), shape=tf.shape, dtype=np.float32
        )
        self._vec = cv

    def _text_scores_batch(self, queries: list[str]) -> np.ndarray:
        Q_raw = self._vec.transform(queries)          # (n_sess, vocab) count
        Q_bin = (Q_raw > 0).astype(np.float32)        # binary query (BM25 convention)
        return (Q_bin @ self._B.T).toarray().astype(np.float32)  # (n_sess, n_tracks)
