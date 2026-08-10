"""LightFM hybrid matrix factorization (item content features via ICM).

Uses the lightfm library with hstack([eye(n_items), l2-normalised ICM]) as
item_features.  Cold-start inference: user embedding = mean of item_repr over
the session profile tracks.
"""

from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sps
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize as sk_normalize

from .user_base import UserRecommender


class LightFMRecommender(UserRecommender):
    RECOMMENDER_NAME = "LightFM"

    def __init__(
        self,
        no_components:  int   = 64,
        loss:           str   = "warp",
        learning_rate:  float = 0.05,
        item_alpha:     float = 1e-6,
        user_alpha:     float = 1e-6,
        epochs:         int   = 30,
        num_threads:    int   = 4,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.no_components = no_components
        self.loss          = loss
        self.learning_rate = learning_rate
        self.item_alpha    = item_alpha
        self.user_alpha    = user_alpha
        self.epochs        = epochs
        self.num_threads   = num_threads

        self._item_repr: np.ndarray | None = None
        self._item_bias: np.ndarray | None = None

    def _fit_model(self, urm: csr_matrix) -> None:
        from lightfm import LightFM

        n_items = urm.shape[1]

        icm_norm = sk_normalize(self.icm.astype(np.float32), norm="l2", axis=1)
        eye      = sps.eye(n_items, format="csr", dtype=np.float32)
        ICM_aug  = sps.hstack([eye, icm_norm], format="csr").astype(np.float32)

        model = LightFM(
            no_components=self.no_components,
            loss=self.loss,
            learning_rate=self.learning_rate,
            item_alpha=self.item_alpha,
            user_alpha=self.user_alpha,
        )

        t0 = time.time()
        model.fit(
            urm,
            item_features=ICM_aug,
            epochs=self.epochs,
            num_threads=self.num_threads,
            verbose=False,
        )

        self._item_repr = np.asarray(
            ICM_aug @ model.item_embeddings, dtype=np.float32
        )
        self._item_bias = np.asarray(
            (ICM_aug @ model.item_biases.reshape(-1, 1)).flatten(), dtype=np.float32
        )

        print(
            f"[{self.RECOMMENDER_NAME}] {self.epochs}ep in {time.time()-t0:.1f}s  "
            f"loss={self.loss}  d={self.no_components}  repr={self._item_repr.shape}"
        )

    def _score_session_profile(self, profile: csr_matrix) -> np.ndarray:
        item_idxs = profile.nonzero()[1]
        if len(item_idxs) == 0 or self._item_repr is None:
            n = self._item_repr.shape[0] if self._item_repr is not None else 0
            return np.zeros(n, dtype=np.float32)
        user_emb = self._item_repr[item_idxs].mean(axis=0)
        return self._item_repr @ user_emb + self._item_bias

    def _get_model_state(self) -> dict:
        st = super()._get_model_state()
        st.update({
            "no_components": self.no_components,
            "loss":          self.loss,
            "learning_rate": self.learning_rate,
            "item_alpha":    self.item_alpha,
            "user_alpha":    self.user_alpha,
            "epochs":        self.epochs,
            "num_threads":   self.num_threads,
            "_item_repr":    self._item_repr,
            "_item_bias":    self._item_bias,
        })
        return st

    def _set_model_state(self, state: dict) -> None:
        super()._set_model_state(state)
        for k in ("no_components", "loss", "learning_rate", "item_alpha",
                  "user_alpha", "epochs", "num_threads"):
            setattr(self, k, state.get(k, getattr(self, k, None)))
        self._item_repr = state.get("_item_repr")
        self._item_bias = state.get("_item_bias")
