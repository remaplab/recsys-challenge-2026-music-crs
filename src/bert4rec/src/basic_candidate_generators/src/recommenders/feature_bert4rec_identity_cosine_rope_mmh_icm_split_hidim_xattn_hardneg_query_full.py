from __future__ import annotations

import random
from typing import Any

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .feature_bert4rec import ITEM_OFFSET, MASK_TOKEN, PAD_TOKEN
from .feature_bert4rec_identity_cosine_rope_mmh_icm_split_hidim_xattn_hardneg_query import (
    FeatureBert4RecIdentityCosineRoPEMMHICMSplitHiDimXAttnHardNegQueryRecommender,
    _FeatureBert4RecQueryDataset,
)

class _FeatureBert4RecQueryFullDataset(_FeatureBert4RecQueryDataset):

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rng = random.Random(idx + 1) if self.deterministic else random
        seq = self.sequences[idx][-self.max_seq_len:]
        tokens = [tok for tok, _ in seq]
        q_idxs = [qi  for _,  qi in seq]
        masked = list(tokens)
        labels = [-100] * len(tokens)

        to_mask = [i for i in range(len(tokens)) if rng.random() < self.mask_prob]
        if not to_mask:
            to_mask = [rng.randrange(len(tokens))]

        for i in to_mask:
            labels[i] = tokens[i] - ITEM_OFFSET
            r = rng.random()
            if r < 0.8:
                masked[i] = MASK_TOKEN
            elif r < 0.9:
                masked[i] = rng.randint(0, self.n_warm - 1) + ITEM_OFFSET

        pad_len = self.max_seq_len - len(tokens)
        masked   = [PAD_TOKEN] * pad_len + masked
        labels   = [-100]      * pad_len + labels
        q_at_all = [0]         * pad_len + list(q_idxs)

        return (
            torch.tensor(masked,   dtype=torch.long),
            torch.tensor(labels,   dtype=torch.long),
            torch.tensor(q_at_all, dtype=torch.long),
        )

class FeatureBert4RecIdentityCosineRoPEMMHICMSplitHiDimXAttnHardNegQueryFullRecommender(
    FeatureBert4RecIdentityCosineRoPEMMHICMSplitHiDimXAttnHardNegQueryRecommender,
):

    RECOMMENDER_NAME = "FeatureBert4RecIdentityCosineRoPEMMHICMSplitHiDimXAttnHardNegQueryFull"

    def __init__(
        self,
        *args: Any,
        infer_zero_prior_queries: bool = False,
        debug_session_n: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

        self._cur_prior_turn_list: dict[str, list[tuple[int, str]]] = {}

        self.infer_zero_prior_queries = bool(infer_zero_prior_queries)

        self.debug_session_n = int(debug_session_n)
        self._debug_session_count = 0

    def _make_query_dataset(
        self,
        sequences: list[list[tuple[int, int]]],
        n_warm: int,
        max_seq_len: int,
        mask_prob: float,
        is_train: bool = True,
    ) -> Dataset:
        return _FeatureBert4RecQueryFullDataset(sequences, n_warm, max_seq_len, mask_prob,
                                                deterministic=self._val_deterministic(is_train))

    def _build_val_eval_examples(self, val_sequences, val_dates=None) -> list[tuple]:
        out: list[tuple] = []
        min_turn = int(self.eval_min_turn)
        zero_prior = self.infer_zero_prior_queries
        if val_dates is None:
            val_dates = [None] * len(val_sequences)
        for seq, sdate in zip(val_sequences, val_dates):
            for p in range(1, len(seq)):
                if (p + 1) < min_turn:
                    continue
                hist  = [tok for tok, _ in seq[:p]]
                histq = [0] * p if zero_prior else [qi for _, qi in seq[:p]]
                out.append((hist, histq, int(seq[p][0]) - ITEM_OFFSET, int(seq[p][1]), sdate))
        return out

    def recommend(self, context_df: pl.DataFrame, *args: Any, **kwargs: Any) -> pl.DataFrame:
        if "track_id" not in context_df.columns:
            from .interactions import explode_music_turns
            context_df = explode_music_turns(context_df)

        if "target_turn" in context_df.columns:
            tt_df = context_df.select(["session_id", "target_turn"]).unique(subset=["session_id"])
            self._session_target_turn = {
                row["session_id"]: int(row["target_turn"])
                for row in tt_df.iter_rows(named=True)
            }
        else:
            self._session_target_turn = {}

        self._cur_prior_turn_list = {}
        if "turn_number" in context_df.columns and context_df.height > 0:
            ctx_sorted = context_df.sort(["session_id", "turn_number"])
            for sid, grp in ctx_sorted.group_by("session_id", maintain_order=True):
                sid_str = sid[0] if isinstance(sid, tuple) else sid
                tns   = grp["turn_number"].to_list()
                tids  = grp["track_id"].to_list()
                self._cur_prior_turn_list[sid_str] = [
                    (int(tn), tid) for tn, tid in zip(tns, tids)
                    if tid is not None and tn is not None
                ]

        if not self._cur_prior_turn_list and context_df.height > 0:
            n_with_track = context_df.filter(pl.col("track_id").is_not_null()).height
            if n_with_track > 0:
                print(
                    f"[{self.RECOMMENDER_NAME}] WARNING: prior turn map is EMPTY "
                    f"despite {n_with_track} context rows with track_id. "
                    f"context_df.columns = {context_df.columns}. "
                    f"All prior positions will get query=0 at inference — train/test mismatch!"
                )

        n_prior_total = 0
        n_prior_hit   = 0
        for sid, prior_list in self._cur_prior_turn_list.items():
            for tn, _tid in prior_list:
                n_prior_total += 1
                if self._query_lookup.get((sid, tn), 0) != 0:
                    n_prior_hit += 1
        n_tgt_total = 0
        n_tgt_hit   = 0
        for sid, tt in self._session_target_turn.items():
            n_tgt_total += 1
            if self._query_lookup.get((sid, int(tt)), 0) != 0:
                n_tgt_hit += 1
        if n_prior_total or n_tgt_total:
            pcov = (n_prior_hit / n_prior_total * 100) if n_prior_total else 0.0
            tcov = (n_tgt_hit   / n_tgt_total   * 100) if n_tgt_total   else 0.0
            print(
                f"[{self.RECOMMENDER_NAME}] inference query coverage: "
                f"prior = {n_prior_hit}/{n_prior_total} ({pcov:.1f}%), "
                f"target = {n_tgt_hit}/{n_tgt_total} ({tcov:.1f}%) "
                f"  |  n_sessions={len(self._session_target_turn)}, "
                f"_query_lookup keys={len(self._query_lookup)}"
            )

        return super().recommend(context_df, *args, **kwargs)

    def _extra_score_kwargs_for_session(self, sess_id: str, user_id: str) -> dict[str, Any]:
        extra = super()._extra_score_kwargs_for_session(sess_id, user_id)

        prior_qidxs: list[int] = []
        for tn, _tid in self._cur_prior_turn_list.get(sess_id, []):
            prior_qidxs.append(self._query_lookup.get((sess_id, tn), 0))
        extra["prior_query_idxs"] = prior_qidxs
        return extra

    def _score_session_sequence(
        self,
        prior: list[str],
        warm_embs: torch.Tensor,
        cold_embs: torch.Tensor | None,
        all_embs: torch.Tensor,
        target_query_idx: int = 0,
        prior_query_idxs: list[int] | None = None,
    ) -> np.ndarray:
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
                tokens.append(self._global_to_warm_local[g] + ITEM_OFFSET)
                kept_qidxs.append(qj)
            elif g in self._global_to_cold_local:
                tokens.append(self._global_to_cold_local[g] + n_warm + ITEM_OFFSET)
                kept_qidxs.append(qj)

        if self.infer_zero_prior_queries:
            kept_qidxs = [0] * len(kept_qidxs)

        tokens.append(MASK_TOKEN)
        kept_qidxs.append(int(target_query_idx))

        tokens     = tokens[-self.max_seq_len:]
        kept_qidxs = kept_qidxs[-self.max_seq_len:]
        pad_len = self.max_seq_len - len(tokens)
        tokens     = [PAD_TOKEN] * pad_len + tokens
        q_idx_seq  = [0]         * pad_len + kept_qidxs

        x = torch.tensor([tokens],    dtype=torch.long, device=self.device_)
        q = torch.tensor([q_idx_seq], dtype=torch.long, device=self.device_)

        with torch.no_grad():
            hidden = self.model.encode_hidden(x, items_table=all_embs, query_idx_seq=q)
            h = hidden[0, -1, :]
            h_n  = F.normalize(h, dim=-1)
            tau  = self.model.tau
            warm_n = F.normalize(warm_embs, dim=-1)
            warm_scores = ((h_n @ warm_n.T) / tau).cpu().numpy()
            if cold_embs is not None:
                cold_n = F.normalize(cold_embs, dim=-1)
                cold_scores = ((h_n @ cold_n.T) / tau).cpu().numpy()
            else:
                cold_scores = None

        if self._debug_session_count < self.debug_session_n:
            self._debug_session_count += 1
            n_nonpad = sum(1 for t in tokens if t != PAD_TOKEN)
            top5_warm_idx = np.argsort(-warm_scores)[:5]
            top5_global = [self._warm_global_indices[i] for i in top5_warm_idx]
            top5_tracks = [self.id_map.idx_to_track[g] for g in top5_global]
            top5_scores = warm_scores[top5_warm_idx].tolist()
            print(
                f"[{self.RECOMMENDER_NAME}] DEBUG session #{self._debug_session_count}: "
                f"prior_len={len(prior)} (in_vocab={n_nonpad-1}), "
                f"target_qidx={int(target_query_idx)}, "
                f"prior_qidxs={kept_qidxs[pad_len:pad_len+n_nonpad-1] if not self.infer_zero_prior_queries else 'ZEROED'}, "
                f"top5_scores={[f'{s:.3f}' for s in top5_scores]}, "
                f"top5_warm_local={top5_warm_idx.tolist()}"
            )

        scores = np.full(self.id_map.n_tracks, -np.inf, dtype=np.float32)
        scores[self._warm_global_indices] = warm_scores
        if cold_scores is not None:
            scores[self._cold_global_indices] = cold_scores
        return scores
