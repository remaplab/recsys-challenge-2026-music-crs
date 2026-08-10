"""Bm25FOneShot — field-weighted turn-1 BM25 retrieval (E3.1).

Instead of one flat doc, builds a separate BM25 index per metadata field
{artist_name, track_name, tag_list, album_name} and scores

    S(q, t) = Σ_field  w_field · BM25_field(q, t)

so an artist-name match can outweigh a tag hit (the flat-doc BM25 lets the
blend silently absorb field scale). Each field's index/query share the recipe of
`Bm25OneShot`; the four field weights + (k1, b) are tuned. Query stopwords (E3.4)
from the base apply to every field's vectorizer.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse as sp

from .oneshot_text_base import _DOC_FIELDS, OneShotTextCG


class Bm25FOneShot(OneShotTextCG):
    RECOMMENDER_NAME = "Bm25FOneShot"

    def __init__(self, k1: float = 1.5, b: float = 0.75,
                 w_artist: float = 1.0, w_track: float = 1.0,
                 w_tags: float = 1.0, w_album: float = 1.0, **kwargs: Any):
        super().__init__(**kwargs)
        self.k1 = float(k1)
        self.b = float(b)
        self.weights = {"artist_name": float(w_artist), "track_name": float(w_track),
                        "tag_list": float(w_tags), "album_name": float(w_album)}
        self._vecs: dict[str, Any] = {}
        self._B: dict[str, sp.csr_matrix] = {}

    def _build_field(self, docs: list[str]) -> tuple[Any, sp.csr_matrix]:
        from sklearn.feature_extraction.text import CountVectorizer
        cv = CountVectorizer(max_features=self.max_features, stop_words=self._stop_words)
        tf = cv.fit_transform(docs).astype(np.float32)
        n = tf.shape[0]
        df = np.asarray((tf > 0).sum(axis=0)).ravel()
        idf = np.log((n - df + 0.5) / (df + 0.5) + 1.0).astype(np.float32)
        dl = np.asarray(tf.sum(axis=1)).ravel()
        dl_norm = (1.0 - self.b + self.b * dl / (dl.mean() or 1.0)).astype(np.float32)
        coo = tf.tocoo()
        vals = (coo.data * (self.k1 + 1.0)
                / (coo.data + self.k1 * dl_norm[coo.row]) * idf[coo.col])
        B = sp.csr_matrix((vals, (coo.row, coo.col)), shape=tf.shape, dtype=np.float32)
        return cv, B

    def _build_index(self, docs: list[str]) -> None:
        # `docs` (flat) ignored — BM25F indexes the per-field docs built by base.
        for f in _DOC_FIELDS:
            if self.weights[f] == 0.0:
                continue
            try:
                self._vecs[f], self._B[f] = self._build_field(self.field_docs[f])
            except ValueError:
                # empty vocabulary (field empty across catalogue) → skip field
                continue

    def _score_batch(self, query_texts: list[str]) -> np.ndarray:
        n_tracks = len(self.track_ids)
        S = np.zeros((len(query_texts), n_tracks), dtype=np.float32)
        for f, cv in self._vecs.items():
            Q_bin = (cv.transform(query_texts) > 0).astype(np.float32)
            S += self.weights[f] * (Q_bin @ self._B[f].T).toarray().astype(np.float32)
        return S

    def _get_model_state(self) -> dict:
        st = super()._get_model_state()
        st.update({"k1": self.k1, "b": self.b, "weights": self.weights,
                   "_vecs": self._vecs, "_B": self._B})
        return st
