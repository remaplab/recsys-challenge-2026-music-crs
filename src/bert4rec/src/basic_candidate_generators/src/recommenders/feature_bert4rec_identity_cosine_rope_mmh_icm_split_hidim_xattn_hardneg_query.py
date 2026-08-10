from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .feature_bert4rec import (
    ITEM_OFFSET,
    MASK_TOKEN,
    PAD_TOKEN,
    _FeatureBert4RecDataset,
)
from .feature_bert4rec_identity_cosine_rope_mmh_icm_hardneg import (
    _build_hard_negatives,
)
from .feature_bert4rec_identity_cosine_rope_mmh_icm_xattn import (
    _MMHICMXAttnModel,
)
from .feature_bert4rec_identity_cosine_rope_mmh_icm_split_hidim_xattn_hardneg import (
    FeatureBert4RecIdentityCosineRoPEMMHICMSplitHiDimXAttnHardNegRecommender,
)

def _load_query_emb_cache(
    cache_dir: Path,
    splits: list[str] = ("train", "dev", "blind_a"),
) -> tuple[np.ndarray, dict[tuple[str, int], int]]:
    cache_dir = Path(cache_dir)
    embs: list[np.ndarray] = []
    lookup: dict[tuple[str, int], int] = {}
    offset = 1
    for sp in splits:
        npy = cache_dir / f"{sp}.npy"
        meta_p = cache_dir / f"{sp}_meta.parquet"
        if not npy.exists() or not meta_p.exists():
            print(f"  [query-cache:{sp}] missing (npy={npy.exists()}, meta={meta_p.exists()}) — skipping")
            continue
        emb = np.load(npy).astype(np.float32, copy=False)
        meta = pl.read_parquet(meta_p)
        sess_arr = meta["session_id"].to_list()
        turn_arr = meta["turn_number"].to_list()
        for i, (sid, tn) in enumerate(zip(sess_arr, turn_arr)):
            lookup[(sid, int(tn))] = offset + i
        embs.append(emb)
        offset += len(emb)
        print(f"  [query-cache:{sp}] {len(emb)} queries (offset now {offset})")
    if not embs:
        raise FileNotFoundError(f"No query cache found in {cache_dir} for splits {splits}")
    query_dim = embs[0].shape[1]
    zero_row = np.zeros((1, query_dim), dtype=np.float32)
    full = np.concatenate([zero_row] + embs, axis=0)
    return full, lookup

class _FeatureBert4RecQueryDataset(Dataset):

    def __init__(
        self,
        sequences: list[list[tuple[int, int]]],
        n_warm: int,
        max_seq_len: int,
        mask_prob: float,
        deterministic: bool = False,
    ) -> None:
        self.sequences = sequences
        self.n_warm = n_warm
        self.max_seq_len = max_seq_len
        self.mask_prob = mask_prob

        self.deterministic = bool(deterministic)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        import random
        rng = random.Random(idx + 1) if self.deterministic else random
        seq = self.sequences[idx][-self.max_seq_len:]
        tokens   = [tok for tok, _ in seq]
        q_idxs   = [qi  for _,  qi in seq]
        masked   = list(tokens)
        labels   = [-100] * len(tokens)
        q_at_mask = [0]   * len(tokens)

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
            q_at_mask[i] = q_idxs[i]

        pad_len = self.max_seq_len - len(tokens)
        masked    = [PAD_TOKEN] * pad_len + masked
        labels    = [-100]      * pad_len + labels
        q_at_mask = [0]         * pad_len + q_at_mask

        return (
            torch.tensor(masked,    dtype=torch.long),
            torch.tensor(labels,    dtype=torch.long),
            torch.tensor(q_at_mask, dtype=torch.long),
        )

class _MMHICMXAttnQueryModel(_MMHICMXAttnModel):

    def __init__(
        self,
        *args: Any,
        query_emb_table: np.ndarray,
        query_prenorm: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.register_buffer(
            "query_table",
            torch.from_numpy(query_emb_table),
            persistent=False,
        )
        query_dim = query_emb_table.shape[1]
        self.query_proj = nn.Linear(query_dim, self.hidden_size, bias=True)

        nn.init.zeros_(self.query_proj.weight)
        nn.init.zeros_(self.query_proj.bias)

        self.query_prenorm = bool(query_prenorm)

    def _query_add(self, query_idx_seq: torch.Tensor | None) -> torch.Tensor | None:
        if query_idx_seq is None:
            return None
        q = self.query_table[query_idx_seq]
        return self.query_proj(q)

    def _apply_query(self, emb: torch.Tensor, query_idx_seq: torch.Tensor) -> torch.Tensor:
        add = self._query_add(query_idx_seq)
        return emb if add is None else emb + add

    def forward(self, x: torch.Tensor, query_idx_seq: torch.Tensor | None = None) -> torch.Tensor:
        warm_embs = self.item_encoder(self.feature_matrix)
        pre_add = self._query_add(query_idx_seq) if self.query_prenorm else None
        emb, pad_mask = self._build_seq_emb(x, warm_embs, pre_norm_add=pre_add)
        if not self.query_prenorm:
            emb = self._apply_query(emb, query_idx_seq)
        out = self.encoder(emb, src_key_padding_mask=pad_mask)
        out = self.output_norm(out)
        out_n  = F.normalize(out,      dim=-1)
        warm_n = F.normalize(warm_embs, dim=-1)
        return (out_n @ warm_n.T) / self.tau

    def encode_hidden(
        self,
        x: torch.Tensor,
        items_table: torch.Tensor | None = None,
        query_idx_seq: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if items_table is None:
            items_table = self.item_encoder(self.feature_matrix)
        pre_add = self._query_add(query_idx_seq) if self.query_prenorm else None
        emb, pad_mask = self._build_seq_emb(x, items_table, pre_norm_add=pre_add)
        if not self.query_prenorm:
            emb = self._apply_query(emb, query_idx_seq)
        out = self.encoder(emb, src_key_padding_mask=pad_mask)
        return self.output_norm(out)

class FeatureBert4RecIdentityCosineRoPEMMHICMSplitHiDimXAttnHardNegQueryRecommender(
    FeatureBert4RecIdentityCosineRoPEMMHICMSplitHiDimXAttnHardNegRecommender,
):

    RECOMMENDER_NAME = "FeatureBert4RecIdentityCosineRoPEMMHICMSplitHiDimXAttnHardNegQuery"

    def __init__(
        self,
        *args: Any,
        query_emb_dir: str = "models/query_emb_cache/qwen3_frozen",
        query_cache_splits: list[str] | None = None,
        query_prenorm: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.query_emb_dir = str(query_emb_dir)
        self.query_cache_splits = list(query_cache_splits) if query_cache_splits else ["train", "dev", "blind_a"]

        self.query_prenorm = bool(query_prenorm)
        import os
        _qp = os.environ.get("RECSYS_QUERY_PRENORM")
        if _qp is not None:
            self.query_prenorm = _qp not in ("0", "false", "False", "no")
        self._query_emb_table: np.ndarray | None = None
        self._query_lookup: dict[tuple[str, int], int] = {}
        self._session_target_turn: dict[str, int] = {}

    def _resolve_query_emb_dir(self) -> Path:
        p = Path(self.query_emb_dir)
        if p.is_absolute():
            return p
        from _cv_utils import repo_path
        return repo_path(self.query_emb_dir)

    def _load_query_cache(self) -> None:
        cache_dir = self._resolve_query_emb_dir()
        print(f"[{self.RECOMMENDER_NAME}] Loading query cache from {cache_dir}")
        emb, lookup = _load_query_emb_cache(cache_dir, self.query_cache_splits)
        self._query_emb_table = emb
        self._query_lookup    = lookup
        print(f"  query_emb_table: {emb.shape}  (row 0 is zero); {len(lookup)} (sess, turn) keys")

    def _build_sequences_with_queries(self) -> list[list[tuple[int, int]]]:
        assert self.id_map is not None and self._train_long is not None
        has_date = "session_date" in self._train_long.columns
        seqs: list[list[tuple[int, int]]] = []
        dates: list = []
        n_missing = 0
        n_total   = 0
        for _, grp in (
            self._train_long
            .sort(["session_id", "turn_number"])
            .group_by("session_id", maintain_order=True)
        ):
            sid_col = grp["session_id"].to_list()
            tn_col  = grp["turn_number"].to_list()
            tk_col  = grp["track_id"].to_list()
            pairs: list[tuple[int, int]] = []
            for sid, tn, tid in zip(sid_col, tn_col, tk_col):
                gidx = self.id_map.track_to_idx.get(tid)
                if gidx is None or gidx not in self._global_to_warm_local:
                    continue
                token = self._global_to_warm_local[gidx] + ITEM_OFFSET
                qidx  = self._query_lookup.get((sid, int(tn)), 0)
                n_total += 1
                if qidx == 0:
                    n_missing += 1
                pairs.append((token, qidx))
            if len(pairs) >= 2:
                seqs.append(pairs)
                dates.append(grp["session_date"].to_list()[0] if has_date else None)
        self._seq_session_dates = dates
        cov = 1.0 - (n_missing / max(1, n_total))
        print(f"  built {len(seqs)} train sequences; query coverage: "
              f"{n_total - n_missing}/{n_total} = {cov:.1%}")
        return seqs

    def _build_train_val_sequences_with_queries(self) -> tuple[
        list[list[tuple[int, int]]],
        list[list[tuple[int, int]]],
    ]:
        import random
        sequences = self._build_sequences_with_queries()
        dates = getattr(self, "_seq_session_dates", None) or [None] * len(sequences)
        paired = list(zip(sequences, dates))
        random.shuffle(paired)
        n_val = max(1, int(len(paired) * self.val_ratio))
        self._val_session_dates = [d for _, d in paired[:n_val]]
        return [s for s, _ in paired[n_val:]], [s for s, _ in paired[:n_val]]

    def _make_query_dataset(
        self,
        sequences: list[list[tuple[int, int]]],
        n_warm: int,
        max_seq_len: int,
        mask_prob: float,
        is_train: bool = True,
    ) -> Dataset:
        return _FeatureBert4RecQueryDataset(sequences, n_warm, max_seq_len, mask_prob,
                                            deterministic=self._val_deterministic(is_train))

    def _build_val_eval_examples(self, val_sequences, val_dates=None) -> list[tuple]:
        out: list[tuple] = []
        min_turn = int(self.eval_min_turn)
        if val_dates is None:
            val_dates = [None] * len(val_sequences)
        for seq, sdate in zip(val_sequences, val_dates):
            for p in range(1, len(seq)):
                if (p + 1) < min_turn:
                    continue
                hist = [tok for tok, _ in seq[:p]]
                out.append((hist, [0] * len(hist), int(seq[p][0]) - ITEM_OFFSET, int(seq[p][1]), sdate))
        return out

    def _eval_encode_hidden(self, x: torch.Tensor, warm_embs: torch.Tensor,
                            q_idx_seq: torch.Tensor) -> torch.Tensor:
        return self.model.encode_hidden(x, items_table=warm_embs, query_idx_seq=q_idx_seq)

    def _make_model(self, warm_feature_matrix: np.ndarray, modality_dims: list[int]):
        assert self._query_emb_table is not None, "_load_query_cache must run before _make_model"
        return _MMHICMXAttnQueryModel(
            warm_feature_matrix, self.hidden_size, self.max_seq_len,
            self.n_layers, self.n_heads, self.dropout,
            init_tau=self.init_tau,
            modality_dims=modality_dims,
            query_emb_table=self._query_emb_table,
            query_prenorm=self.query_prenorm,
        )

    def _eval_batch_logits(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
        masked_seq, labels, q_idx = batch[0], batch[1], batch[2]
        masked_seq = masked_seq.to(self.device_)
        labels     = labels.to(self.device_)
        q_idx      = q_idx.to(self.device_)
        return self.model(masked_seq, query_idx_seq=q_idx), labels

    def _train_batch_loss(self, batch, epoch: int, n_warm: int,
                          hard_neg_tensor: torch.Tensor):
        masked_seq, labels, q_idx = batch[0], batch[1], batch[2]
        masked_seq = masked_seq.to(self.device_)
        labels     = labels.to(self.device_)
        q_idx      = q_idx.to(self.device_)

        logits = self.model(masked_seq, query_idx_seq=q_idx)
        loss_mlm = F.cross_entropy(
            logits.view(-1, n_warm), labels.view(-1), ignore_index=-100
        )

        with torch.no_grad():
            flat_logits = logits.view(-1, n_warm)
            flat_labels = labels.view(-1)
            valid = flat_labels != -100
        tgt_pos = valid.nonzero(as_tuple=False).squeeze(-1)
        if tgt_pos.numel() > 0:
            tgt_idx = flat_labels[tgt_pos]
            anchor = flat_logits[tgt_pos, tgt_idx].unsqueeze(1)
            hn = hard_neg_tensor[tgt_idx]
            neg_logits = flat_logits[tgt_pos.unsqueeze(1).expand_as(hn), hn]
            hn_logits = torch.cat([anchor, neg_logits], dim=1) / self.hardneg_tau
            hn_labels = torch.zeros(hn_logits.size(0), dtype=torch.long,
                                     device=hn_logits.device)
            loss_hn = F.cross_entropy(hn_logits, hn_labels)
            loss_pn = self._pivotneg_loss(flat_logits, tgt_pos, tgt_idx, masked_seq, n_warm)
        else:
            loss_hn = torch.zeros((), device=self.device_)
            loss_pn = torch.zeros((), device=self.device_)

        loss = loss_mlm + self._get_hardneg_weight(epoch) * loss_hn + loss_pn
        return loss, loss_mlm, loss_hn

    def _prepare_pivotneg(self) -> None:
        return None

    def _pivotneg_loss(self, flat_logits, tgt_pos, tgt_idx, masked_seq, n_warm):
        return flat_logits.new_zeros(())

    def _fit_model(self, urm: csr_matrix) -> None:
        assert self.id_map is not None and self._train_long is not None

        self._set_seeds()

        warm_track_ids: set[str] = set(self._train_long["track_id"].to_list())
        warm_track_ids &= set(self.id_map.track_to_idx.keys())
        self._warm_global_indices = sorted(self.id_map.track_to_idx[t] for t in warm_track_ids)
        self._cold_global_indices = sorted(
            idx for t, idx in self.id_map.track_to_idx.items() if t not in warm_track_ids
        )
        self._global_to_warm_local = {g: l for l, g in enumerate(self._warm_global_indices)}
        self._global_to_cold_local = {g: l for l, g in enumerate(self._cold_global_indices)}

        n_warm = len(self._warm_global_indices)
        n_cold = len(self._cold_global_indices)

        self._load_query_cache()

        print(f"[{self.RECOMMENDER_NAME}] Loading per-modality features (with ICM)...")
        full_matrix, modality_dims = self._build_modality_feature_matrix()
        self._feature_dim = full_matrix.shape[1]
        self._modality_dims = modality_dims
        warm_feature_matrix = full_matrix[self._warm_global_indices]
        self._cold_feature_matrix = full_matrix[self._cold_global_indices]

        self._warm_feature_matrix = warm_feature_matrix

        train_sequences, val_sequences = self._build_train_val_sequences_with_queries()
        print(
            f"[{self.RECOMMENDER_NAME}] warm={n_warm}, cold={n_cold}, "
            f"train_seqs={len(train_sequences)}, val_seqs={len(val_sequences)}, "
            f"device={self.device_}, init_tau={self.init_tau}, "
            f"hardneg_k={self.hardneg_k}, hardneg_weight={self.hardneg_weight}, "
            f"hardneg_tau={self.hardneg_tau}"
        )

        print(f"[{self.RECOMMENDER_NAME}] Pre-computing hard negatives (k={self.hardneg_k})...")
        feat_l2_chunks = []
        s = 0
        for d in modality_dims:
            chunk = warm_feature_matrix[:, s:s + d]
            norms = np.linalg.norm(chunk, axis=1, keepdims=True).clip(min=1e-10)
            feat_l2_chunks.append(chunk / norms)
            s += d
        feat_l2 = np.concatenate(feat_l2_chunks, axis=1).astype(np.float32)
        feat_l2 = feat_l2 / np.linalg.norm(feat_l2, axis=1, keepdims=True).clip(min=1e-10)
        hard_neg_idx = _build_hard_negatives(feat_l2, k=self.hardneg_k)
        hard_neg_tensor = torch.from_numpy(hard_neg_idx).to(self.device_)

        self._prepare_pivotneg()

        pin = self.device_.type == "cuda"
        train_loader = DataLoader(
            self._make_query_dataset(train_sequences, n_warm, self.max_seq_len, self.mask_prob, is_train=True),
            batch_size=self.batch_size, shuffle=True, num_workers=0, pin_memory=pin,
        )
        val_loader = DataLoader(
            self._make_query_dataset(val_sequences, n_warm, self.max_seq_len, self.mask_prob, is_train=False),
            batch_size=self.batch_size, shuffle=False, num_workers=0, pin_memory=pin,
        )

        self.model = self._make_model(warm_feature_matrix, modality_dims)
        self._pca_init_encoder_per_modality(warm_feature_matrix, modality_dims)
        self.model.to(self.device_)

        for head_name in ("query_proj", "query_fusion"):
            head = getattr(self.model, head_name, None)
            if head is not None:
                n_qp = sum(p.numel() for p in head.parameters())
                print(f"  {head_name} params: {n_qp:,}")
                break

        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        total_steps = self.epochs * len(train_loader)
        warmup_steps = max(1, int(total_steps * self.warmup_ratio))
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, self._make_cosine_lr_lambda(total_steps, warmup_steps)
        )

        epoch_bar = tqdm(range(1, self.epochs + 1), desc=f"[{self.RECOMMENDER_NAME}]",
                          unit="ep", dynamic_ncols=True, file=sys.stdout)

        best_val = float("inf")
        best_epoch = 0
        best_state: dict | None = None
        patience_left = self.early_stop_patience
        val_examples = (
            self._build_val_eval_examples(val_sequences, getattr(self, "_val_session_dates", None))
            if self.early_stop_metric in ("ndcg", "recall") else None
        ) or None

        ema_enabled = self.ema_decay > 0.0
        ema_params: dict[str, torch.Tensor] = {}
        ema_best_val = float("inf")
        ema_best_epoch = 0
        ema_best_state: dict | None = None
        if ema_enabled:
            for n, p in self.model.named_parameters():
                ema_params[n] = p.detach().clone()
            print(f"[{self.RECOMMENDER_NAME}] EMA enabled: decay={self.ema_decay}, "
                  f"start_epoch={self.ema_start_epoch}, n_params={len(ema_params)}")

        for epoch in epoch_bar:
            self.model.train()
            total_loss = 0.0
            total_mlm  = 0.0
            total_hn   = 0.0
            for batch in tqdm(
                train_loader, desc=f"  ep {epoch:3d}", leave=False,
                unit="batch", dynamic_ncols=True, file=sys.stdout,
            ):

                loss, loss_mlm, loss_hn = self._train_batch_loss(
                    batch, epoch, n_warm, hard_neg_tensor
                )

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()
                total_mlm  += loss_mlm.item()
                total_hn   += loss_hn.item()

                if ema_enabled and epoch >= self.ema_start_epoch:
                    with torch.no_grad():
                        d = self.ema_decay
                        for n, p in self.model.named_parameters():
                            ema_params[n].mul_(d).add_(p.detach(), alpha=1.0 - d)

            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    logits, labels = self._eval_batch_logits(batch)
                    val_loss += F.cross_entropy(
                        logits.view(-1, n_warm), labels.view(-1), ignore_index=-100
                    ).item()
            val_avg = val_loss / len(val_loader)
            train_avg = total_loss / len(train_loader)

            ema_val_avg: float | None = None
            ema_improved = False
            if ema_enabled and epoch >= self.ema_start_epoch:
                live_backup = {n: p.detach().clone() for n, p in self.model.named_parameters()}
                with torch.no_grad():
                    for n, p in self.model.named_parameters():
                        p.data.copy_(ema_params[n])
                    ema_val = 0.0
                    for batch in val_loader:
                        logits, labels = self._eval_batch_logits(batch)
                        ema_val += F.cross_entropy(
                            logits.view(-1, n_warm), labels.view(-1), ignore_index=-100
                        ).item()
                    ema_val_avg = ema_val / len(val_loader)

                    ema_metrics = self._compute_val_metrics(val_examples)
                    for n, p in self.model.named_parameters():
                        p.data.copy_(live_backup[n])
                ema_sel, _ = self._val_selection(ema_metrics, ema_val_avg)
                if ema_sel < ema_best_val:
                    ema_best_val = ema_sel
                    ema_best_epoch = epoch

                    snap = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                    for n in ema_params:
                        snap[n] = ema_params[n].detach().cpu().clone()
                    ema_best_state = snap
                    ema_improved = True

            val_metrics = self._compute_val_metrics(val_examples)
            sel, mdisp = self._val_selection(val_metrics, val_avg)
            _active = val_metrics is not None

            improved = sel < best_val
            if improved:
                best_val = sel
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}

            if improved or ema_improved:
                patience_left = self.early_stop_patience
            else:
                patience_left -= 1

            postfix = dict(
                loss=f"{train_avg:.4f}",
                mlm=f"{total_mlm/len(train_loader):.4f}",
                hn=f"{total_hn/len(train_loader):.4f}",
                val=f"{val_avg:.4f}",
                best=f"{(-best_val) if _active else best_val:.4f}@{best_epoch}",
                tau=f"{self.model.tau.item():.4f}",
                patience=patience_left,
            )
            postfix.update(mdisp)
            if ema_val_avg is not None:
                postfix["ema_val"]  = f"{ema_val_avg:.4f}"
                postfix["ema_best"] = f"{(-ema_best_val) if _active else ema_best_val:.4f}@{ema_best_epoch}"
            epoch_bar.set_postfix(**postfix)
            if patience_left <= 0:
                _lbl = self._val_metric_label(_active)
                _bl = f"{(-best_val) if _active else best_val:.4f}"
                _be = f"{(-ema_best_val) if _active else ema_best_val:.4f}"
                print(f"\n[{self.RECOMMENDER_NAME}] early stopping at epoch {epoch}; "
                      f"best live {_lbl}={_bl}@{best_epoch}, "
                      f"best ema {_lbl}={_be}@{ema_best_epoch}")
                break

        use_ema = (
            ema_enabled
            and ema_best_state is not None
            and ema_best_val < best_val
        )
        _active = val_examples is not None
        _m = self._val_metric_label(_active)
        _sgn = -1.0 if _active else 1.0
        if use_ema:
            self.model.load_state_dict(ema_best_state)
            print(f"[{self.RECOMMENDER_NAME}] restored EMA best "
                  f"(ema_{_m}={_sgn*ema_best_val:.4f}@{ema_best_epoch} better than "
                  f"live_{_m}={_sgn*best_val:.4f}@{best_epoch})")
        elif best_state is not None:
            self.model.load_state_dict(best_state)
            tag = "live best" if not ema_enabled else f"live best (EMA worse: {_sgn*ema_best_val:.4f})"
            print(f"[{self.RECOMMENDER_NAME}] restored {tag} "
                  f"({_m}={_sgn*best_val:.4f}, epoch={best_epoch}, tau={self.model.tau.item():.4f})")

    def recommend(self, context_df: pl.DataFrame, *args: Any, **kwargs: Any) -> pl.DataFrame:
        if "target_turn" in context_df.columns:
            unique_df = context_df.select(["session_id", "target_turn"]).unique(subset=["session_id"])
            self._session_target_turn = {
                row["session_id"]: int(row["target_turn"])
                for row in unique_df.iter_rows(named=True)
            }
        else:
            self._session_target_turn = {}
        return super().recommend(context_df, *args, **kwargs)

    def _extra_score_kwargs_for_session(self, sess_id: str, user_id: str) -> dict[str, Any]:
        target_turn = self._session_target_turn.get(sess_id)
        if target_turn is None:
            return {"target_query_idx": 0}
        qidx = self._query_lookup.get((sess_id, target_turn), 0)
        return {"target_query_idx": qidx}

    def _score_session_sequence(
        self,
        prior: list[str],
        warm_embs: torch.Tensor,
        cold_embs: torch.Tensor | None,
        all_embs: torch.Tensor,
        target_query_idx: int = 0,
    ) -> np.ndarray:
        assert self.model is not None and self.id_map is not None
        n_warm = warm_embs.shape[0]

        tokens: list[int] = []
        for t in prior:
            if t not in self.id_map.track_to_idx:
                continue
            g = self.id_map.track_to_idx[t]
            if g in self._global_to_warm_local:
                tokens.append(self._global_to_warm_local[g] + ITEM_OFFSET)
            elif g in self._global_to_cold_local:
                tokens.append(self._global_to_cold_local[g] + n_warm + ITEM_OFFSET)

        tokens = tokens + [MASK_TOKEN]
        tokens = tokens[-self.max_seq_len:]
        pad_len = self.max_seq_len - len(tokens)
        tokens = [PAD_TOKEN] * pad_len + tokens

        q_idx_seq = [0] * self.max_seq_len
        q_idx_seq[-1] = int(target_query_idx)

        x = torch.tensor([tokens],     dtype=torch.long, device=self.device_)
        q = torch.tensor([q_idx_seq],  dtype=torch.long, device=self.device_)

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

        scores = np.full(self.id_map.n_tracks, -np.inf, dtype=np.float32)
        scores[self._warm_global_indices] = warm_scores
        if cold_scores is not None:
            scores[self._cold_global_indices] = cold_scores
        return scores

    def _get_model_state(self) -> dict:
        st = super()._get_model_state()
        st["query_emb_dir"]      = self.query_emb_dir
        st["query_cache_splits"] = self.query_cache_splits
        st["query_prenorm"]      = self.query_prenorm
        return st

    def _set_model_state(self, state: dict) -> None:
        self.query_emb_dir      = state.get("query_emb_dir", self.query_emb_dir)
        self.query_cache_splits = state.get("query_cache_splits", self.query_cache_splits)

        self.query_prenorm      = bool(state.get("query_prenorm", False))

        self._load_query_cache()
        super()._set_model_state(state)

        if self.model is not None:
            self.model.query_table = torch.from_numpy(self._query_emb_table).to(self.device_)
