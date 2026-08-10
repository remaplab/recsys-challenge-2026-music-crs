"""Bm25OneShot — turn-1 BM25 retrieval (query_text → track docs).

Precomputes the BM25 weight matrix B[i,t] over track docs so each query is one
binary-query × B^T sparse matmul. Same recipe as the legacy BM25TextCG, minus
the CF/URM machinery.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse as sp

from .oneshot_text_base import OneShotTextCG


class Bm25OneShot(OneShotTextCG):
    RECOMMENDER_NAME = "Bm25OneShot"

    def __init__(self, k1: float = 1.5, b: float = 0.75, **kwargs: Any):
        super().__init__(**kwargs)
        self.k1 = float(k1)
        self.b = float(b)
        self._vec = None
        self._B: sp.csr_matrix | None = None          # (n_tracks, vocab)

    def _build_index(self, docs: list[str]) -> None:
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
        self._B = sp.csr_matrix((vals, (coo.row, coo.col)), shape=tf.shape, dtype=np.float32)
        self._vec = cv

    def _score_batch(self, query_texts: list[str]) -> np.ndarray:
        Q_bin = (self._vec.transform(query_texts) > 0).astype(np.float32)
        return (Q_bin @ self._B.T).toarray().astype(np.float32)

    def _get_model_state(self) -> dict:
        st = super()._get_model_state()
        st.update({"k1": self.k1, "b": self.b, "_vec": self._vec, "_B": self._B})
        return st
