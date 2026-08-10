"""HybridAllQwen — multi-scale evolution of HybridCG.

Same self-contained content hybrid (TFIDF + BM25 + ICM + last/prev/query-emb,
fused by weighted RRF), but the embedding signals run over THREE Qwen3 track
towers simultaneously — 0.6B, 4B and 8B — each contributing its own
last-track + prev-context ranked lists with independently tuned RRF weights.

Why RRF and not an additive blend: the three sizes live in different embedding
spaces with different cosine-score scales, so summing similarities would be
ill-posed. RRF fuses *ranks*, so each size's list contributes `w/(k_rrf+rank)`
regardless of its score magnitude — the scale-free property is exactly what
makes cross-size fusion well-defined.

Towers share the catalogue (metadata) index space, so context-track indices are
valid against all three. Each extra tower is loaded + cached per size (reusing
`HybridCG._cached_tower_for`) and uploaded once per fit.

Scope: the query-emb→track signal stays 8B-only (its per-set query embeddings
are attached by the inference dispatch from the 8B query tower); only the base
8B backend sets `use_sess_qemb=True`. The 0.6B/4B backends add last+prev. Per-
size query-emb would need multi-bundle dispatch plumbing — deferred.

Whether the extra sizes help over 8B alone is an empirical question answered by
the DRO tuner: the per-size weights can all collapse to 0 if they don't.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .hybrid_cg import HybridCG, _EmbBackend


class HybridAllQwen(HybridCG):
    RECOMMENDER_NAME = "HybridAllQwen"

    def __init__(
        self,
        track_emb_dir_4b: str | None = None,
        track_emb_dir_0p6b: str | None = None,
        model_size_4b: str = "qwen3_4b",
        model_size_0p6b: str = "qwen3_0p6b",
        w_last_4b: float = 1.0,
        w_prev_4b: float = 1.0,
        w_last_0p6b: float = 1.0,
        w_prev_0p6b: float = 1.0,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.track_emb_dir_4b = track_emb_dir_4b
        self.track_emb_dir_0p6b = track_emb_dir_0p6b
        self.model_size_4b = model_size_4b
        self.model_size_0p6b = model_size_0p6b
        self.w_last_4b = w_last_4b
        self.w_prev_4b = w_prev_4b
        self.w_last_0p6b = w_last_0p6b
        self.w_prev_0p6b = w_prev_0p6b

        # extra towers (id_map order) + their resident CUDA tensors
        self.track_emb_4b: np.ndarray | None = None
        self.track_emb_0p6b: np.ndarray | None = None
        self._tower_gpu_4b = None
        self._tower_gpu_0p6b = None

    # ------------------------------------------------------------------
    # fit: 8B path (super) + extra towers, then rebuild backends
    # ------------------------------------------------------------------

    def fit(self, train_df, track_metadata=None, **kwargs: Any) -> None:
        # Builds 8B tower + text/ICM artefacts and calls _to_gpu(), which sets
        # the (8B-only) backend list — the extra towers aren't loaded yet there.
        super().fit(train_df, track_metadata=track_metadata, **kwargs)

        sig = self._meta_sig()
        if self.track_emb_dir_4b:
            self.track_emb_4b = self._cached_tower_for(
                self.track_emb_dir_4b, self.model_size_4b, sig)
        if self.track_emb_dir_0p6b:
            self.track_emb_0p6b = self._cached_tower_for(
                self.track_emb_dir_0p6b, self.model_size_0p6b, sig)

        self._upload_extra_towers()
        self._emb_backends = self._build_emb_backends()
        sizes = [be.size for be in self._emb_backends]
        print(f"[{self.RECOMMENDER_NAME}] emb backends: {sizes}")

    def _upload_extra_towers(self) -> None:
        """Upload the extra-size towers to CUDA (no-op on CPU)."""
        torch = self._torch
        if torch is None:
            self._tower_gpu_4b = self._tower_gpu_0p6b = None
            return
        if self.track_emb_4b is not None:
            self._tower_gpu_4b = torch.from_numpy(
                np.ascontiguousarray(self.track_emb_4b)).to("cuda")
        if self.track_emb_0p6b is not None:
            self._tower_gpu_0p6b = torch.from_numpy(
                np.ascontiguousarray(self.track_emb_0p6b)).to("cuda")

    def _build_emb_backends(self) -> list[_EmbBackend]:
        backends = super()._build_emb_backends()  # 8B base (use_sess_qemb=True)
        if self.track_emb_4b is not None:
            backends.append(_EmbBackend(
                self.model_size_4b, self.track_emb_4b, self._tower_gpu_4b,
                self.w_last_4b, self.w_prev_4b, 0.0, use_sess_qemb=False))
        if self.track_emb_0p6b is not None:
            backends.append(_EmbBackend(
                self.model_size_0p6b, self.track_emb_0p6b, self._tower_gpu_0p6b,
                self.w_last_0p6b, self.w_prev_0p6b, 0.0, use_sess_qemb=False))
        return backends

    # ------------------------------------------------------------------
    # persistence (params only; refit rebuilds towers — like the parent)
    # ------------------------------------------------------------------

    def _get_model_state(self) -> dict:
        st = super()._get_model_state()
        st.update({
            "track_emb_dir_4b": self.track_emb_dir_4b,
            "track_emb_dir_0p6b": self.track_emb_dir_0p6b,
            "model_size_4b": self.model_size_4b,
            "model_size_0p6b": self.model_size_0p6b,
            "w_last_4b": self.w_last_4b, "w_prev_4b": self.w_prev_4b,
            "w_last_0p6b": self.w_last_0p6b, "w_prev_0p6b": self.w_prev_0p6b,
        })
        return st

    def _set_model_state(self, state: dict) -> None:
        super()._set_model_state(state)
        for k in ("track_emb_dir_4b", "track_emb_dir_0p6b",
                  "model_size_4b", "model_size_0p6b",
                  "w_last_4b", "w_prev_4b", "w_last_0p6b", "w_prev_0p6b"):
            if k in state:
                setattr(self, k, state[k])
        # extra towers aren't pickled (refit rebuilds them from cache, like
        # the parent) — init to None so load() matches __init__'s state and
        # the intermediate _to_gpu() call inside super().fit() doesn't
        # AttributeError before fit() reloads and re-uploads them.
        self.track_emb_4b = None
        self.track_emb_0p6b = None
        self._tower_gpu_4b = None
        self._tower_gpu_0p6b = None
