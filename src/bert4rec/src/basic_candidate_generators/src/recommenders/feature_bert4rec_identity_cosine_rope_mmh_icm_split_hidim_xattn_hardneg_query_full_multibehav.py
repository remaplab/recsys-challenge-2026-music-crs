from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .feature_bert4rec import ITEM_OFFSET, MASK_TOKEN, PAD_TOKEN
from .feature_bert4rec_identity_cosine_rope_mmh_icm_split_hidim_xattn_hardneg_query import (
    _MMHICMXAttnQueryModel,
)
from .feature_bert4rec_identity_cosine_rope_mmh_icm_split_hidim_xattn_hardneg_query_full import (
    FeatureBert4RecIdentityCosineRoPEMMHICMSplitHiDimXAttnHardNegQueryFullRecommender,
)

class _MMHICMXAttnQueryMultiBehavModel(_MMHICMXAttnQueryModel):

    def __init__(self, *args: Any, n_interests: int = 4, interest_da: int = 256,
                 select_mode: str = "argmax", mv_beta: float = 12.0,
                 **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.n_interests = int(n_interests)
        self.select_mode = str(select_mode)
        self.mv_beta = float(mv_beta)
        d = self.hidden_size
        da = int(interest_da)

        self.interest_w1 = nn.Linear(d, da, bias=False)
        self.interest_w2 = nn.Linear(da, self.n_interests, bias=False)

    def _route(self, x: torch.Tensor, H: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pool = (x != PAD_TOKEN) & (x != MASK_TOKEN)
        pre = self.interest_w2(torch.tanh(self.interest_w1(H)))
        pre = pre.masked_fill(~pool.unsqueeze(-1), float("-inf"))
        A = torch.softmax(pre, dim=1)
        A = torch.nan_to_num(A, nan=0.0)
        Vu = torch.einsum("bld,blk->bkd", H, A)
        return F.normalize(Vu, dim=-1), A

    def _reduce(self, s_k: torch.Tensor) -> torch.Tensor:
        if self.select_mode == "softor":
            return torch.logsumexp(self.mv_beta * s_k, dim=-2) / self.mv_beta
        return s_k.max(dim=-2).values

    def interest_scores(self, x: torch.Tensor, items_n: torch.Tensor,
                        query_idx_seq: torch.Tensor | None,
                        warm_embs: torch.Tensor | None = None) -> torch.Tensor:
        if warm_embs is None:
            warm_embs = self.item_encoder(self.feature_matrix)
        H = self.encode_hidden(x, items_table=warm_embs, query_idx_seq=query_idx_seq)
        Vu_n, _ = self._route(x, H)
        return torch.einsum("bkd,nd->bkn", Vu_n, items_n)

    def forward(self, x: torch.Tensor, query_idx_seq: torch.Tensor | None = None) -> torch.Tensor:
        warm_embs = self.item_encoder(self.feature_matrix)
        warm_n = F.normalize(warm_embs, dim=-1)
        s_k = self.interest_scores(x, warm_n, query_idx_seq, warm_embs=warm_embs)
        s = self._reduce(s_k) / self.tau
        B, L = x.shape
        return s.unsqueeze(1).expand(B, L, -1).contiguous()

class FeatureBert4RecIdentityCosineRoPEMMHICMSplitHiDimXAttnHardNegQueryFullMultiBehavRecommender(
    FeatureBert4RecIdentityCosineRoPEMMHICMSplitHiDimXAttnHardNegQueryFullRecommender,
):

    RECOMMENDER_NAME = "FeatureBert4RecIdentityCosineRoPEMMHICMSplitHiDimXAttnHardNegQueryFullMultiBehav"

    def __init__(self, *args: Any, n_interests: int = 4, interest_da: int = 256,
                 select_mode: str = "argmax", mv_beta: float = 12.0,
                 rr_weight: float = 0.0, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.n_interests = int(n_interests)
        self.interest_da = int(interest_da)
        self.select_mode = str(select_mode)
        self.mv_beta = float(mv_beta)
        self.rr_weight = float(rr_weight)

    def _make_model(self, warm_feature_matrix: np.ndarray, modality_dims: list[int]):
        assert self._query_emb_table is not None, "_load_query_cache must run before _make_model"
        return _MMHICMXAttnQueryMultiBehavModel(
            warm_feature_matrix, self.hidden_size, self.max_seq_len,
            self.n_layers, self.n_heads, self.dropout,
            init_tau=self.init_tau, modality_dims=modality_dims,
            query_emb_table=self._query_emb_table, query_prenorm=self.query_prenorm,
            n_interests=self.n_interests, interest_da=self.interest_da,
            select_mode=self.select_mode, mv_beta=self.mv_beta,
        )

    def _train_batch_loss(self, batch, epoch: int, n_warm: int, hard_neg_tensor: torch.Tensor):
        masked_seq, labels, q_idx = batch[0], batch[1], batch[2]
        masked_seq = masked_seq.to(self.device_)
        labels     = labels.to(self.device_)
        q_idx      = q_idx.to(self.device_)
        B, L = masked_seq.shape
        K = self.model.n_interests

        warm_embs = self.model.item_encoder(self.model.feature_matrix)
        warm_n = F.normalize(warm_embs, dim=-1)
        H = self.model.encode_hidden(masked_seq, items_table=warm_embs, query_idx_seq=q_idx)
        Vu_n, A = self.model._route(masked_seq, H)
        s_k = torch.einsum("bkd,nd->bkn", Vu_n, warm_n)
        tau = self.model.tau

        flat_labels = labels.view(-1)
        pos = (flat_labels != -100).nonzero(as_tuple=False).squeeze(-1)
        if pos.numel() > 0:
            b_idx = pos // L
            y = flat_labels[pos]
            sk_sel = s_k[b_idx]
            if self.select_mode == "softor":
                logits_m = self.model._reduce(sk_sel) / tau
            else:
                gt_per_k = sk_sel.gather(
                    2, y.view(-1, 1, 1).expand(-1, K, 1)
                ).squeeze(-1)
                kstar = gt_per_k.argmax(dim=1)
                logits_m = sk_sel[torch.arange(sk_sel.size(0), device=self.device_), kstar] / tau
            loss_mlm = F.cross_entropy(logits_m, y)

            anchor = logits_m.gather(1, y.unsqueeze(1))
            hn = hard_neg_tensor[y]
            neg = logits_m.gather(1, hn)
            hn_logits = torch.cat([anchor, neg], dim=1) / self.hardneg_tau
            hn_labels = torch.zeros(hn_logits.size(0), dtype=torch.long, device=self.device_)
            loss_hn = F.cross_entropy(hn_logits, hn_labels)
        else:
            loss_mlm = torch.zeros((), device=self.device_)
            loss_hn  = torch.zeros((), device=self.device_)

        Abar = A.mean(dim=1, keepdim=True)
        diag_C = ((A - Abar) ** 2).sum(dim=1)
        loss_rr = (diag_C ** 2).sum(dim=1).mean()

        loss = loss_mlm + self._get_hardneg_weight(epoch) * loss_hn + self.rr_weight * loss_rr
        return loss, loss_mlm, loss_hn

    def _eval_encode_hidden(self, x: torch.Tensor, warm_embs: torch.Tensor,
                            q_idx_seq: torch.Tensor) -> torch.Tensor:
        H = self.model.encode_hidden(x, items_table=warm_embs, query_idx_seq=q_idx_seq)
        Vu_n, _ = self.model._route(x, H)
        return Vu_n

    def _eval_scores(self, vu_n, q, warm_n, cold_n, tau):
        def _logits(items_n):
            s_k = torch.einsum("bkd,nd->bkn", vu_n, items_n)
            return self.model._reduce(s_k) / tau
        warm_logits = _logits(warm_n)
        cold_logits = _logits(cold_n) if cold_n is not None else None
        return warm_logits, cold_logits

    def _score_session_sequence(self, prior, warm_embs, cold_embs, all_embs,
                                target_query_idx: int = 0, prior_query_idxs=None) -> np.ndarray:
        assert self.model is not None and self.id_map is not None
        n_warm = warm_embs.shape[0]

        tokens: list[int] = []
        kept_qidxs: list[int] = []
        prior_qidxs = prior_query_idxs or [0] * len(prior)
        for j, t in enumerate(prior):
            if t not in self.id_map.track_to_idx:
                continue
            g = self.id_map.track_to_idx[t]
            qj = prior_qidxs[j] if j < len(prior_qidxs) else 0
            if g in self._global_to_warm_local:
                tokens.append(self._global_to_warm_local[g] + ITEM_OFFSET); kept_qidxs.append(qj)
            elif g in self._global_to_cold_local:
                tokens.append(self._global_to_cold_local[g] + n_warm + ITEM_OFFSET); kept_qidxs.append(qj)
        tokens.append(MASK_TOKEN); kept_qidxs.append(int(target_query_idx))
        tokens     = tokens[-self.max_seq_len:]
        kept_qidxs = kept_qidxs[-self.max_seq_len:]
        pad_len = self.max_seq_len - len(tokens)
        tokens    = [PAD_TOKEN] * pad_len + tokens
        q_idx_seq = [0]         * pad_len + kept_qidxs

        x = torch.tensor([tokens],    dtype=torch.long, device=self.device_)
        q = torch.tensor([q_idx_seq], dtype=torch.long, device=self.device_)

        with torch.no_grad():
            H = self.model.encode_hidden(x, items_table=all_embs, query_idx_seq=q)
            Vu_n, _ = self.model._route(x, H)
            tau = self.model.tau

            def _scores(items_n: torch.Tensor) -> np.ndarray:
                s_k = torch.einsum("bkd,nd->bkn", Vu_n, items_n)
                return (self.model._reduce(s_k)[0] / tau).cpu().numpy()

            warm_scores = _scores(F.normalize(warm_embs, dim=-1))
            cold_scores = _scores(F.normalize(cold_embs, dim=-1)) if cold_embs is not None else None

        scores = np.full(self.id_map.n_tracks, -np.inf, dtype=np.float32)
        scores[self._warm_global_indices] = warm_scores
        if cold_scores is not None:
            scores[self._cold_global_indices] = cold_scores
        return scores

    def _get_model_state(self) -> dict:
        st = super()._get_model_state()
        st["n_interests"] = self.n_interests
        st["interest_da"] = self.interest_da
        st["select_mode"] = self.select_mode
        st["mv_beta"]     = self.mv_beta
        st["rr_weight"]   = self.rr_weight
        return st

    def _set_model_state(self, state: dict) -> None:
        self.n_interests = int(state.get("n_interests", getattr(self, "n_interests", 4)))
        self.interest_da = int(state.get("interest_da", getattr(self, "interest_da", 256)))
        self.select_mode = str(state.get("select_mode", getattr(self, "select_mode", "argmax")))
        self.mv_beta     = float(state.get("mv_beta", getattr(self, "mv_beta", 12.0)))
        self.rr_weight   = float(state.get("rr_weight", getattr(self, "rr_weight", 0.0)))
        super()._set_model_state(state)
