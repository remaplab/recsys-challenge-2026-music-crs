"""Design A query injection + late-fusion retrieval baked into TRAINING.

The inference-only `..._latefusion` variant showed a clean, interpretable
pattern on the CV beta-sweep (same checkpoint, noise-free):

    beta  ndcg@20   recall@200
    0.0   0.2513    0.6990
    0.5   0.2508    0.7015
    1.0   0.2485    0.7030

i.e. the query->track-text term `q . item_text` is a good RECALL signal but a
RANKING drag at the top: added post-hoc to a frozen model it double-counts
topicality and pushes generically-on-topic tracks up, costing ndcg@20.

That monotone "recall up / ndcg down" shape is the signature of an UN-adapted
additive term. The principled fix is to train WITH the term in the loss, so the
model learns to be complementary to it (stop duplicating what retrieval covers)
and the mixing weights calibrate themselves to the objective:

    logits[b,l,:] = (out_n . warm_n)/tau  +  beta * (q_n[b,l] . warm_text_n)/tau_q

with `beta` and `tau_q` LEARNABLE scalars (parameterised in log-space for
positivity). They are initialised small (~0.1) so training starts close to the
parent `query_full` (cf. the zero-init `query_proj` philosophy) and ramps the
fusion only if it lowers the MLM/hardneg loss.

Inference reuses the `..._latefusion` scoring path verbatim — same additive
term on warm AND cold candidates — but with the LEARNED `beta`/`tau_q` (synced
from the model into the recommender before the inference loop runs).

Training is otherwise identical to the parent (`_fit_model` is inherited): the
Design-A dataset already feeds per-position query indices, so the fusion term
sees the per-turn query at every position; the loss only counts masked ones.
"""

from __future__ import annotations

import math
from typing import Any

import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from .feature_bert4rec_identity_cosine_rope_mmh_icm_split_hidim_xattn_hardneg_query import (
    _MMHICMXAttnQueryModel,
)
from .feature_bert4rec_identity_cosine_rope_mmh_icm_split_hidim_xattn_hardneg_query_full_latefusion import (
    FeatureBert4RecIdentityCosineRoPEMMHICMSplitHiDimXAttnHardNegQueryFullLateFusionRecommender,
)


# ---------------------------------------------------------------------------
# Model — query model + a learnable query->text retrieval term in the logits
# ---------------------------------------------------------------------------

class _MMHICMXAttnQueryFusionModel(_MMHICMXAttnQueryModel):
    """Query model whose warm-logits also carry a (learnable) late-fusion term.

    The text modality is the first modality (qwen3 metadata), which shares the
    query's Qwen3 space, so `q . item_text` is a meaningful semantic match.
    """

    def __init__(
        self,
        *args: Any,
        fusion_beta_init: float = 0.1,
        fusion_tau_q_init: float = 0.1,
        fusion_learnable: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        # Text modality = first modality; its dim must equal the query dim.
        self._text_dim = int(self.modality_dims[0])
        q_dim = int(self.query_table.shape[1])
        assert q_dim == self._text_dim, (
            f"train-fusion needs query and text-modality in the same space, but "
            f"query_dim={q_dim} != text_modality_dim={self._text_dim}"
        )

        lb = math.log(max(float(fusion_beta_init), 1e-4))
        lt = math.log(max(float(fusion_tau_q_init), 1e-4))
        if fusion_learnable:
            self.fusion_log_beta  = nn.Parameter(torch.tensor(lb, dtype=torch.float32))
            self.fusion_log_tau_q = nn.Parameter(torch.tensor(lt, dtype=torch.float32))
        else:
            self.register_buffer("fusion_log_beta",  torch.tensor(lb, dtype=torch.float32))
            self.register_buffer("fusion_log_tau_q", torch.tensor(lt, dtype=torch.float32))

    # Positive-constrained scalars.
    def fusion_beta(self) -> torch.Tensor:
        return self.fusion_log_beta.exp()

    def fusion_tau_q(self) -> torch.Tensor:
        return self.fusion_log_tau_q.exp()

    def _fusion_warm_logits(self, query_idx_seq: torch.Tensor) -> torch.Tensor:
        """beta * (q_n . warm_text_n) / tau_q  → (B, L, n_warm).

        Positions with query_idx == 0 (no query) get the zero query row, whose
        normalisation is zero, so they contribute nothing — same semantics as
        the additive `_apply_query`."""
        warm_text_n = F.normalize(self.feature_matrix[:, :self._text_dim], dim=-1)  # (n_warm, td)
        q   = self.query_table[query_idx_seq]          # (B, L, td)
        q_n = F.normalize(q, dim=-1)
        retr = torch.matmul(q_n, warm_text_n.t())      # (B, L, n_warm)
        return self.fusion_beta() * (retr / self.fusion_tau_q())

    def forward(self, x: torch.Tensor, query_idx_seq: torch.Tensor | None = None) -> torch.Tensor:
        base = super().forward(x, query_idx_seq=query_idx_seq)   # (B, L, n_warm)
        if query_idx_seq is None:
            return base
        return base + self._fusion_warm_logits(query_idx_seq)


# ---------------------------------------------------------------------------
# Recommender
# ---------------------------------------------------------------------------

class FeatureBert4RecIdentityCosineRoPEMMHICMSplitHiDimXAttnHardNegQueryFullTrainFusionRecommender(
    FeatureBert4RecIdentityCosineRoPEMMHICMSplitHiDimXAttnHardNegQueryFullLateFusionRecommender,
):
    """query_full + late-fusion retrieval term learned jointly during training."""

    RECOMMENDER_NAME = "FeatureBert4RecIdentityCosineRoPEMMHICMSplitHiDimXAttnHardNegQueryFullTrainFusion"

    def __init__(
        self,
        *args: Any,
        fusion_learnable: bool = True,
        **kwargs: Any,
    ) -> None:
        # `fusion_beta` / `fusion_tau_q` (from the LateFusion parent) are reused
        # as the INITIAL values of the learnable scalars; after fit they are
        # overwritten on the recommender with the learned values for inference.
        super().__init__(*args, **kwargs)
        self.fusion_learnable = bool(fusion_learnable)

    # Build the fusion-aware model instead of the plain query model.
    def _make_model(self, warm_feature_matrix, modality_dims):
        assert self._query_emb_table is not None, "_load_query_cache must run before _make_model"
        return _MMHICMXAttnQueryFusionModel(
            warm_feature_matrix, self.hidden_size, self.max_seq_len,
            self.n_layers, self.n_heads, self.dropout,
            init_tau=self.init_tau,
            modality_dims=modality_dims,
            query_emb_table=self._query_emb_table,
            fusion_beta_init=self.fusion_beta,
            fusion_tau_q_init=self.fusion_tau_q,
            fusion_learnable=self.fusion_learnable,
        )

    def recommend(self, context_df: pl.DataFrame, *args: Any, **kwargs: Any) -> pl.DataFrame:
        # Sync the LEARNED fusion scalars from the model so the inherited
        # LateFusion inference path uses them (it reads self.fusion_beta/tau_q).
        if self.model is not None and hasattr(self.model, "fusion_beta"):
            with torch.no_grad():
                self.fusion_beta  = float(self.model.fusion_beta().item())
                self.fusion_tau_q = float(self.model.fusion_tau_q().item())
            print(
                f"[{self.RECOMMENDER_NAME}] learned fusion at inference: "
                f"beta={self.fusion_beta:.4f}, tau_q={self.fusion_tau_q:.4f}"
            )
        return super().recommend(context_df, *args, **kwargs)

    # ------------------------------------------------------------------
    # save / load — fusion_learnable flag (beta/tau_q live in the state_dict
    # as model params/buffers and are restored by super()._set_model_state).
    # ------------------------------------------------------------------

    def _get_model_state(self) -> dict:
        st = super()._get_model_state()
        st["fusion_learnable"] = self.fusion_learnable
        return st

    def _set_model_state(self, state: dict) -> None:
        self.fusion_learnable = bool(state.get("fusion_learnable", getattr(self, "fusion_learnable", True)))
        super()._set_model_state(state)