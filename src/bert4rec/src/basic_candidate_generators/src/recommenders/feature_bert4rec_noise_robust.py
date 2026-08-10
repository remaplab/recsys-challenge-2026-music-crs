from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .feature_bert4rec_identity_cosine_rope_mmh_icm_hardneg import _build_hard_negatives
from .feature_bert4rec_identity_cosine_rope_mmh_icm_split_hidim_xattn_hardneg_query import (
    FeatureBert4RecIdentityCosineRoPEMMHICMSplitHiDimXAttnHardNegQueryRecommender,
)
from .feature_bert4rec_identity_cosine_rope_mmh_icm_split_hidim_xattn_hardneg_query_full import (
    FeatureBert4RecIdentityCosineRoPEMMHICMSplitHiDimXAttnHardNegQueryFullRecommender,
)
from .feature_bert4rec_identity_cosine_rope_mmh_icm_split_hidim_xattn_hardneg_query_full_dif import (
    FeatureBert4RecIdentityCosineRoPEMMHICMSplitHiDimXAttnHardNegQueryFullDIFRecommender,
)
from .feature_bert4rec_identity_cosine_rope_mmh_icm_split_hidim_xattn_hardneg_query_full_nova import (
    FeatureBert4RecIdentityCosineRoPEMMHICMSplitHiDimXAttnHardNegQueryFullNOVARecommender,
)

class _NoiseRobustLossMixin:
    def __init__(self, *args: Any,
                 soft_eps: float = 0.0,
                 soft_m: int = 10,
                 fn_skip: int = 0,
                 **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.soft_eps = float(soft_eps)
        self.soft_m = int(soft_m)
        self.fn_skip = int(fn_skip)
        self._nr_ready = False
        self._nr_softpos: torch.Tensor | None = None
        self._nr_hardneg: torch.Tensor | None = None

    def _nr_prepare(self) -> None:
        if self._nr_ready:
            return
        wfm = self._warm_feature_matrix
        dims = self._modality_dims
        chunks, s = [], 0
        for d in dims:
            c = wfm[:, s:s + d]
            c = c / np.linalg.norm(c, axis=1, keepdims=True).clip(min=1e-10)
            chunks.append(c); s += d
        feat_l2 = np.concatenate(chunks, axis=1).astype(np.float32)
        feat_l2 = feat_l2 / np.linalg.norm(feat_l2, axis=1, keepdims=True).clip(min=1e-10)

        soft_on = self.soft_eps > 0.0
        start = (self.soft_m if soft_on else 0) + self.fn_skip
        total = start + self.hardneg_k
        nbr = _build_hard_negatives(feat_l2, k=total)
        self._nr_hardneg = torch.from_numpy(
            np.ascontiguousarray(nbr[:, start:start + self.hardneg_k]).astype(np.int64)
        ).to(self.device_)
        self._nr_softpos = (
            torch.from_numpy(np.ascontiguousarray(nbr[:, :self.soft_m]).astype(np.int64)).to(self.device_)
            if soft_on else None
        )
        print(f"[{self.RECOMMENDER_NAME}] noise-robust loss ready: "
              f"soft_eps={self.soft_eps}, soft_m={self.soft_m}, fn_skip={self.fn_skip}, "
              f"hardneg_k={self.hardneg_k} (negs from neighbour #{start})")
        self._nr_ready = True

    def _train_batch_loss(self, batch, epoch: int, n_warm: int,
                          hard_neg_tensor: torch.Tensor):
        self._nr_prepare()
        dev = self.device_
        masked_seq, labels, q_idx = batch[0].to(dev), batch[1].to(dev), batch[2].to(dev)

        logits = self.model(masked_seq, query_idx_seq=q_idx)
        flat = logits.view(-1, n_warm)
        lab = labels.view(-1)
        valid = lab != -100
        if not bool(valid.any()):
            z = torch.zeros((), device=dev)
            return z, z, z

        fv = flat[valid]
        y = lab[valid]
        V = y.numel()
        ar = torch.arange(V, device=dev)

        logp = F.log_softmax(fv, dim=-1)
        nll_true = -logp[ar, y]
        if self._nr_softpos is not None:
            sp = self._nr_softpos[y]
            nll_soft = -logp.gather(1, sp).mean(dim=1)
            loss_mlm = ((1.0 - self.soft_eps) * nll_true + self.soft_eps * nll_soft).mean()
        else:
            loss_mlm = nll_true.mean()

        anchor = fv[ar, y].unsqueeze(1)
        neg = fv.gather(1, self._nr_hardneg[y])
        hn_logits = torch.cat([anchor, neg], dim=1) / self.hardneg_tau
        hn_labels = torch.zeros(V, dtype=torch.long, device=dev)
        loss_hn = F.cross_entropy(hn_logits, hn_labels)

        loss = loss_mlm + self._get_hardneg_weight(epoch) * loss_hn
        return loss, loss_mlm, loss_hn

class FeatureBert4RecIdentityCosineRoPEMMHICMSplitHiDimXAttnHardNegQueryNoiseRobustRecommender(
    _NoiseRobustLossMixin,
    FeatureBert4RecIdentityCosineRoPEMMHICMSplitHiDimXAttnHardNegQueryRecommender,
):
    RECOMMENDER_NAME = "FeatureBert4RecIdentityCosineRoPEMMHICMSplitHiDimXAttnHardNegQueryNoiseRobust"

class FeatureBert4RecIdentityCosineRoPEMMHICMSplitHiDimXAttnHardNegQueryFullNoiseRobustRecommender(
    _NoiseRobustLossMixin,
    FeatureBert4RecIdentityCosineRoPEMMHICMSplitHiDimXAttnHardNegQueryFullRecommender,
):
    RECOMMENDER_NAME = "FeatureBert4RecIdentityCosineRoPEMMHICMSplitHiDimXAttnHardNegQueryFullNoiseRobust"

class FeatureBert4RecIdentityCosineRoPEMMHICMSplitHiDimXAttnHardNegQueryFullDIFNoiseRobustRecommender(
    _NoiseRobustLossMixin,
    FeatureBert4RecIdentityCosineRoPEMMHICMSplitHiDimXAttnHardNegQueryFullDIFRecommender,
):
    RECOMMENDER_NAME = "FeatureBert4RecIdentityCosineRoPEMMHICMSplitHiDimXAttnHardNegQueryFullDIFNoiseRobust"

class FeatureBert4RecIdentityCosineRoPEMMHICMSplitHiDimXAttnHardNegQueryFullNOVANoiseRobustRecommender(
    _NoiseRobustLossMixin,
    FeatureBert4RecIdentityCosineRoPEMMHICMSplitHiDimXAttnHardNegQueryFullNOVARecommender,
):
    RECOMMENDER_NAME = "FeatureBert4RecIdentityCosineRoPEMMHICMSplitHiDimXAttnHardNegQueryFullNOVANoiseRobust"
