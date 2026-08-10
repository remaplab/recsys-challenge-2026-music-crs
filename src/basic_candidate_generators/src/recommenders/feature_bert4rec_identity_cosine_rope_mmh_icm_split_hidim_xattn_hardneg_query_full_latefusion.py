"""Design A (query at all positions) + LATE-FUSION query→track retrieval term.

Motivation
----------
In the parent `..._query_full` variant the per-turn query enters the model only
additively, through `emb + query_proj(q)` with `query_proj` zero-initialised
(see `_MMHICMXAttnQueryModel._apply_query`). The query is therefore a slow-
learned perturbation of the item stream, and the final score is *purely*

    score(item) = (h_n · item_n) / tau

i.e. the query never touches the candidate items directly. But this is a
conversational task: the turn query IS the intent ("something calm to study
to"), and the single most predictive signal is the direct match between the
query text and the track's own text.

This variant adds a LATE-FUSION (retrieval) term at inference only:

    score(item) = (h_n · item_n) / tau
                + beta * (q_target_n · item_text_n) / tau_q

where
  * q_target  = the frozen Qwen3 embedding of the TARGET-turn query
                (already cached as `query_table[target_query_idx]`),
  * item_text = the track's textual modality — the `metadata-qwen3_embedding_0.6b`
                feature (the FIRST modality, 1024d), which lives in the same
                Qwen3 space as the query so the dot product is a meaningful
                semantic similarity.

Both vectors are L2-normalised, so the second term is a cosine similarity
scaled by `1/tau_q` and mixed in with weight `beta`. With `beta = 0` the model
is exactly the parent `..._query_full`.

Notes
-----
  * Train is UNCHANGED — the fusion term is inference-only, so the learned
    weights are identical to the parent. `beta`/`tau_q` are pure inference
    hyper-parameters and can be tuned without retraining (though we still expose
    them to the CV/fold tuners so they are searched jointly with the rest).
  * The textual modality is always the first modality (qwen3) by construction of
    `_build_modality_feature_matrix` (dense modalities come first, ICM blocks are
    appended), and its dim equals the query dim (1024). We assert this on setup.
  * Cold tracks get the same treatment via `self._cold_feature_matrix`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F

from .feature_bert4rec_identity_cosine_rope_mmh_icm_split_hidim_xattn_hardneg_query_full import (
    FeatureBert4RecIdentityCosineRoPEMMHICMSplitHiDimXAttnHardNegQueryFullRecommender,
)


class FeatureBert4RecIdentityCosineRoPEMMHICMSplitHiDimXAttnHardNegQueryFullLateFusionRecommender(
    FeatureBert4RecIdentityCosineRoPEMMHICMSplitHiDimXAttnHardNegQueryFullRecommender,
):
    """query_full + an inference-time query↔track-text retrieval score."""

    RECOMMENDER_NAME = "FeatureBert4RecIdentityCosineRoPEMMHICMSplitHiDimXAttnHardNegQueryFullLateFusion"

    def __init__(
        self,
        *args: Any,
        fusion_beta: float = 0.5,
        fusion_tau_q: float = 0.1,
        text_modality_index: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        # Weight of the late-fusion retrieval term (0 == parent query_full).
        self.fusion_beta = float(fusion_beta)
        # Temperature for the query↔track-text cosine similarity.
        self.fusion_tau_q = float(fusion_tau_q)
        # Which feature modality is the text one (qwen3 metadata == 0).
        self.text_modality_index = int(text_modality_index)

        # Precomputed L2-normalised text-modality embeddings, aligned with the
        # warm/cold local index order (== warm_embs / cold_embs rows). Filled by
        # _prepare_text_modality() at the start of recommend().
        self._warm_text_n: torch.Tensor | None = None
        self._cold_text_n: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Setup: slice the text modality out of the (raw) feature matrices and
    # L2-normalise once per recommend() call.
    # ------------------------------------------------------------------

    def _text_modality_bounds(self) -> tuple[int, int]:
        """Return (start, end) column indices of the text modality in the
        concatenated feature matrix. The text modality (qwen3) is the first
        dense modality, so dims[:idx] precede it."""
        dims = getattr(self.model, "modality_dims", None) or self._modality_dims
        assert dims, "modality_dims unavailable — was the model fit/loaded?"
        idx = self.text_modality_index
        start = int(sum(dims[:idx]))
        end = start + int(dims[idx])
        return start, end

    def _prepare_text_modality(self) -> None:
        if self.model is None or self.fusion_beta == 0.0:
            self._warm_text_n = None
            self._cold_text_n = None
            return

        start, end = self._text_modality_bounds()
        text_dim = end - start
        query_dim = int(self.model.query_table.shape[1])
        assert query_dim == text_dim, (
            f"[{self.RECOMMENDER_NAME}] late fusion needs the query and the text "
            f"modality to share a space, but query_dim={query_dim} != "
            f"text_modality_dim={text_dim} (modality #{self.text_modality_index}). "
            f"Set text_modality_index to the qwen3 modality."
        )

        with torch.no_grad():
            warm_text = self.model.feature_matrix[:, start:end].to(self.device_).float()
            self._warm_text_n = F.normalize(warm_text, dim=-1)
            if self._cold_feature_matrix is not None and len(self._cold_global_indices) > 0:
                cold_text = torch.from_numpy(
                    np.ascontiguousarray(self._cold_feature_matrix[:, start:end])
                ).to(self.device_).float()
                self._cold_text_n = F.normalize(cold_text, dim=-1)
            else:
                self._cold_text_n = None

        print(
            f"[{self.RECOMMENDER_NAME}] late fusion ready: "
            f"beta={self.fusion_beta}, tau_q={self.fusion_tau_q}, "
            f"text modality #{self.text_modality_index} cols [{start}:{end}] ({text_dim}d), "
            f"warm_text={tuple(self._warm_text_n.shape)}, "
            f"cold_text={tuple(self._cold_text_n.shape) if self._cold_text_n is not None else None}"
        )

    def recommend(self, context_df: pl.DataFrame, *args: Any, **kwargs: Any) -> pl.DataFrame:
        # Build the per-session prior-turn maps (parent) AND the text-modality
        # tensors used by the fusion term, then run the shared inference loop.
        self._prepare_text_modality()
        return super().recommend(context_df, *args, **kwargs)

    # ------------------------------------------------------------------
    # Scoring: base model score + beta * (q_target · item_text) / tau_q
    # ------------------------------------------------------------------

    def _score_session_sequence(
        self,
        prior: list[str],
        warm_embs: torch.Tensor,
        cold_embs: torch.Tensor | None,
        all_embs: torch.Tensor,
        target_query_idx: int = 0,
        prior_query_idxs: list[int] | None = None,
    ) -> np.ndarray:
        # 1. Base transformer score (Design A: query injected at all positions).
        scores = super()._score_session_sequence(
            prior, warm_embs, cold_embs, all_embs,
            target_query_idx=target_query_idx,
            prior_query_idxs=prior_query_idxs,
        )

        # 2. Late-fusion retrieval term. Skipped when disabled (beta == 0) or
        # when there is no target query (idx 0 == reserved zero row).
        tqi = int(target_query_idx)
        if self.fusion_beta == 0.0 or tqi == 0 or self._warm_text_n is None:
            return scores

        with torch.no_grad():
            q_vec = self.model.query_table[tqi].to(self.device_).float()
            if float(q_vec.abs().sum()) == 0.0:   # zero/no-query row → no signal
                return scores
            q_n = F.normalize(q_vec, dim=-1)

            retr_warm = (q_n @ self._warm_text_n.T) / self.fusion_tau_q
            scores[self._warm_global_indices] += (
                self.fusion_beta * retr_warm
            ).cpu().numpy()

            if self._cold_text_n is not None:
                retr_cold = (q_n @ self._cold_text_n.T) / self.fusion_tau_q
                scores[self._cold_global_indices] += (
                    self.fusion_beta * retr_cold
                ).cpu().numpy()

        return scores

    # ------------------------------------------------------------------
    # save / load — persist the fusion hyper-parameters
    # ------------------------------------------------------------------

    def _get_model_state(self) -> dict:
        st = super()._get_model_state()
        st["fusion_beta"]         = self.fusion_beta
        st["fusion_tau_q"]        = self.fusion_tau_q
        st["text_modality_index"] = self.text_modality_index
        return st

    def _set_model_state(self, state: dict) -> None:
        super()._set_model_state(state)
        self.fusion_beta         = float(state.get("fusion_beta", self.fusion_beta))
        self.fusion_tau_q        = float(state.get("fusion_tau_q", self.fusion_tau_q))
        self.text_modality_index = int(state.get("text_modality_index", self.text_modality_index))