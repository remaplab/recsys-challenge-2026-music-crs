"""TfidfOneShot — turn-1 TF-IDF cosine retrieval (query_text → track docs)."""
from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse as sp

from .oneshot_text_base import OneShotTextCG


class TfidfOneShot(OneShotTextCG):
    RECOMMENDER_NAME = "TfidfOneShot"

    def __init__(self, ngram_max: int = 2, ngram_min: int = 1,
                 analyzer: str = "word", **kwargs: Any):
        super().__init__(**kwargs)
        self.ngram_max = int(ngram_max)
        self.ngram_min = int(ngram_min)
        self.analyzer = str(analyzer)        # "word" (default) or "char_wb" (E3.2)
        self._vec = None
        self._item_mat: sp.csr_matrix | None = None   # (n_tracks, vocab)

    def _build_index(self, docs: list[str]) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        # stop_words apply to word-level analysis only (sklearn ignores them for
        # char analyzers); the char-ngram channel runs without them.
        stop = self._stop_words if self.analyzer == "word" else None
        self._vec = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=(self.ngram_min, self.ngram_max),
            analyzer=self.analyzer,
            stop_words=stop,
            sublinear_tf=True,
        )
        self._item_mat = self._vec.fit_transform(docs)

    def _score_batch(self, query_texts: list[str]) -> np.ndarray:
        Q = self._vec.transform(query_texts)
        return (Q @ self._item_mat.T).toarray().astype(np.float32)

    def _get_model_state(self) -> dict:
        st = super()._get_model_state()
        st.update({"ngram_max": self.ngram_max, "ngram_min": self.ngram_min,
                   "analyzer": self.analyzer, "_vec": self._vec,
                   "_item_mat": self._item_mat})
        return st
