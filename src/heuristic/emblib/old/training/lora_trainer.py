"""Encoder-agnostic InfoNCE trainer.

Trains ANY encoder whose forward(texts, user_cf, is_cold) returns (B, D)
L2-normalized vectors. The track tower is supplied as a (T, D) np.ndarray +
optional (T,) bool mask — so it works with the organizer Qwen3 tower OR a
custom-encoded BERT/SBERT/ModernBERT tower.

Despite the name, this trainer is NOT specific to LoRA: it updates whatever
parameters have requires_grad=True. For frozen-backbone encoders that's just
the projection head + soft-prompt routing components; for LoRA-backbone
encoders that's the LoRA matrices on top.

Query text is built by `build_query_text_v2` to keep training and inference
text formats identical (see scripts/03_encode_queries.py).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Set

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from emblib.data.parsing import build_query_text_v2
from emblib.data.user_features import UserFeatures


@dataclass
class LoRATrainConfig:
    batch_size: int = 8
    grad_accum_steps: int = 16
    max_epochs: int = 2
    patience: int = 1
    lr: float = 1e-4
    weight_decay: float = 1e-2
    warmup_steps: int = 500
    grad_clip: float = 1.0
    log_every: int = 5
    eval_every_steps: int = 0
    temperature: float = 0.02
    n_hard_negatives: int = 16
    use_in_batch_negatives: bool = True
    eval_top_ks: tuple = (10, 100, 500, 1000)
    eval_subsample: int = 2000


class LoRATrainDataset(Dataset):
    """Dataset for any encoder family — works with the organizer Qwen3 tower
    or any per-backbone text tower (just pass the matching id_to_idx and
    tower_mask).

    Builds per-row query text via `build_query_text_v2`, identical to what
    inference uses, so the encoder learns on the same input distribution it
    will see at evaluation time.
    """

    def __init__(
        self,
        meta_df: pl.DataFrame,
        conv_parquet_path: Path,
        id_to_idx: dict[str, int],
        tower_mask: np.ndarray,
        users: UserFeatures,
        keep_session_ids: Set[str] | None,
        hard_negs_path: Path | None,
        track_lookup: dict[str, dict] | None = None,
        use_thoughts: bool = True,
    ):
        conv_df = pl.read_parquet(conv_parquet_path)
        conv_by_session = {r["session_id"]: r for r in conv_df.to_dicts()}

        meta_rows = meta_df.to_dicts()
        self.users = users
        self.hard_negs = None
        if hard_negs_path is not None and hard_negs_path.exists():
            self.hard_negs = np.load(hard_negs_path)
            assert self.hard_negs.shape[0] == meta_df.shape[0]

        kept = []
        for i, r in enumerate(meta_rows):
            if keep_session_ids is not None and r["session_id"] not in keep_session_ids:
                continue
            tid = r["gt_track_id"]
            if tid is None or tid not in id_to_idx:
                continue
            tidx = id_to_idx[tid]
            if not tower_mask[tidx]:
                continue
            if self.hard_negs is not None and self.hard_negs[i, 0] == -1:
                continue
            kept.append(i)
        print(f"  kept {len(kept)}/{len(meta_rows)} rows")

        self.row_indices = kept
        self.texts = []
        n = len(kept)
        self.gt_idx = np.zeros(n, dtype=np.int64)
        self.user_cf = np.zeros((n, users.cf.shape[1]), dtype=np.float32)
        self.is_cold = np.ones(n, dtype=bool)

        for new_i, orig_i in enumerate(kept):
            r = meta_rows[orig_i]
            self.gt_idx[new_i] = id_to_idx[r["gt_track_id"]]
            uid = r["user_id"]
            if uid in users.id_to_idx:
                u_idx = users.id_to_idx[uid]
                self.user_cf[new_i] = users.cf[u_idx]
                self.is_cold[new_i] = bool(users.is_cold[u_idx])
            sess = conv_by_session.get(r["session_id"])
            if sess is None:
                self.texts.append("")
                continue
            self.texts.append(self._build_text(
                sess, int(r["turn_number"]), use_thoughts, track_lookup,
            ))

    @staticmethod
    def _build_text(sess, target_turn, use_thoughts, track_lookup):
        """Reconstruct the v2 query text for one (session, turn) row.

        Note: prior_progress (goal_progress_assessments) is NOT extracted
        here — v2 drops those entirely. session_date is pulled from the
        session row so the [SESSION] year=YYYY line can be added.
        """
        convs = sess["conversations"]
        user_profile = sess.get("user_profile") or {}
        conv_goal = sess.get("conversation_goal") or {}
        session_date = sess.get("session_date")
        if session_date is not None:
            session_date = str(session_date)

        turns_by_number = defaultdict(list)
        for t in convs:
            turns_by_number[t["turn_number"]].append(t)
        user_msgs = [t for t in turns_by_number.get(target_turn, []) if t["role"] == "user"]
        if not user_msgs:
            return ""
        current = user_msgs[0]["content"]

        chat_history = []
        for t in sorted(turns_by_number.keys()):
            if t >= target_turn:
                break
            for prev in turns_by_number[t]:
                chat_history.append({
                    "role": prev.get("role"),
                    "content": prev.get("content") or "",
                    "thought": prev.get("thought") or "",
                })

        return build_query_text_v2(
            chat_history=chat_history,
            user_query=current,
            user_profile=user_profile,
            conversation_goal=conv_goal,
            session_date=session_date,
            track_lookup=track_lookup,
            use_thoughts=use_thoughts,
        )

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {
            "text": self.texts[idx],
            "user_cf": torch.from_numpy(self.user_cf[idx]),
            "is_cold": torch.tensor(self.is_cold[idx]),
            "gt_idx": torch.tensor(self.gt_idx[idx], dtype=torch.long),
            "row_idx": idx,
        }


def collate_lora(batch):
    return {
        "texts":   [b["text"] for b in batch],
        "user_cf": torch.stack([b["user_cf"] for b in batch]),
        "is_cold": torch.stack([b["is_cold"] for b in batch]),
        "gt_idx":  torch.stack([b["gt_idx"] for b in batch]),
        "row_indices": [b["row_idx"] for b in batch],
    }


def get_warmup_lr(step, warmup, base_lr):
    if step >= warmup:
        return base_lr
    return base_lr * (step + 1) / max(1, warmup)


@torch.no_grad()
def evaluate_recall(encoder, val_ds, track_emb_t, track_mask_t,
                    device, top_ks, batch_size, subsample):
    encoder.eval()
    n = len(val_ds)
    if subsample and subsample < n:
        rng = np.random.default_rng(0)
        idx_subset = rng.choice(n, size=subsample, replace=False)
    else:
        idx_subset = np.arange(n)
    max_k = max(top_ks)
    hit = {k: 0 for k in top_ks}
    total = 0
    for s in range(0, len(idx_subset), batch_size):
        sel = idx_subset[s:s + batch_size]
        texts = [val_ds.texts[i] for i in sel]
        u_cf = torch.from_numpy(val_ds.user_cf[sel]).float()
        is_cold = torch.from_numpy(val_ds.is_cold[sel]).bool()
        gt = torch.from_numpy(val_ds.gt_idx[sel]).long().to(device)
        emb = encoder(texts, u_cf, is_cold).to(torch.float32)
        scores = (emb @ track_emb_t.T).masked_fill(
            ~track_mask_t.unsqueeze(0), float("-inf"))
        top = torch.topk(scores, k=max_k, dim=1).indices
        match = (top == gt.unsqueeze(1))
        for k in top_ks:
            hit[k] += int(match[:, :k].any(dim=1).sum().item())
        total += len(sel)
    return {f"recall@{k}": hit[k] / max(total, 1) for k in top_ks} | {"n_eval": total}


def train_lora_encoder(encoder, train_ds, val_ds, track_emb, track_mask,
                       cfg, device, out_dir):
    """track_emb: (T, D) np.ndarray L2-normalized; track_mask: (T,) bool np.ndarray.

    Despite the function name, this is a generic InfoNCE trainer; it updates
    whatever the encoder has marked as `requires_grad=True`. For
    frozen-backbone encoders that's just the projection head + routing
    components; for LoRA-backbone encoders the LoRA matrices are also updated.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    track_emb_t = torch.from_numpy(track_emb).to(device, dtype=torch.float32)
    track_mask_t = torch.from_numpy(track_mask).to(device)

    loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                        num_workers=0, collate_fn=collate_lora, drop_last=True)
    optim = torch.optim.AdamW(
        encoder.trainable_parameters(),
        lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.98), eps=1e-6,
    )
    print(f"\nTraining: {len(train_ds)} train, {len(val_ds)} val, "
          f"effective_batch={cfg.batch_size * cfg.grad_accum_steps}")
    print(f"Trainable params: {encoder.n_trainable():,}")

    history, best_val, best_epoch, no_improve, global_step = [], -1.0, -1, 0, 0

    for epoch in range(1, cfg.max_epochs + 1):
        encoder.train()
        accum_loss, accum_count, running_loss, n_steps = 0.0, 0, 0.0, 0
        optim.zero_grad()
        for step, batch in enumerate(loader):
            row_indices = batch["row_indices"]
            if train_ds.hard_negs is not None:
                pool = train_ds.hard_negs[
                    [train_ds.row_indices[i] for i in row_indices]
                ]
                k_use = min(cfg.n_hard_negatives, pool.shape[1])
                rng = np.random.default_rng()
                cols = rng.integers(0, pool.shape[1], size=(pool.shape[0], k_use))
                neg_idx = np.take_along_axis(pool, cols, axis=1)
                neg_idx_t = torch.from_numpy(neg_idx.astype(np.int64)).to(device)
            else:
                neg_idx_t = None

            q_emb = encoder(batch["texts"], batch["user_cf"], batch["is_cold"])
            q_emb = q_emb.to(torch.float32)
            gt = batch["gt_idx"].to(device)
            B = q_emb.size(0)

            pos_emb = track_emb_t[gt]
            pos_score = (q_emb * pos_emb).sum(dim=-1)
            logits_parts = [pos_score.unsqueeze(1)]

            if neg_idx_t is not None:
                neg_emb = track_emb_t[neg_idx_t]
                neg_score = torch.einsum("bh,bkh->bk", q_emb, neg_emb)
                neg_valid = track_mask_t[neg_idx_t]
                neg_score = neg_score.masked_fill(~neg_valid, float("-inf"))
                logits_parts.append(neg_score)

            if cfg.use_in_batch_negatives and B > 1:
                ib = q_emb @ pos_emb.T
                eye = torch.eye(B, dtype=torch.bool, device=device)
                ib = ib.masked_fill(eye, float("-inf"))
                logits_parts.append(ib)

            logits = torch.cat(logits_parts, dim=1) / cfg.temperature
            target = torch.zeros(B, dtype=torch.long, device=device)
            loss = F.cross_entropy(logits, target)
            (loss / cfg.grad_accum_steps).backward()
            accum_loss += loss.item()
            accum_count += 1

            if accum_count >= cfg.grad_accum_steps:
                lr_now = get_warmup_lr(global_step, cfg.warmup_steps, cfg.lr)
                for g in optim.param_groups:
                    g["lr"] = lr_now
                torch.nn.utils.clip_grad_norm_(
                    encoder.trainable_parameters(), cfg.grad_clip)
                optim.step()
                optim.zero_grad()
                running_loss += accum_loss / accum_count
                n_steps += 1
                global_step += 1
                if n_steps % cfg.log_every == 0:
                    print(f"  ep{epoch} step {n_steps} | "
                          f"loss={running_loss/n_steps:.4f} lr={lr_now:.2e}")
                accum_loss = 0.0
                accum_count = 0

        avg_loss = running_loss / max(n_steps, 1)
        val_metrics = evaluate_recall(
            encoder, val_ds, track_emb_t, track_mask_t, device,
            cfg.eval_top_ks, max(cfg.batch_size, 8), cfg.eval_subsample,
        )
        history.append({"epoch": epoch, "loss": avg_loss, "val": val_metrics})
        score = val_metrics.get("recall@500", 0.0)
        if score > best_val:
            best_val, best_epoch, no_improve = score, epoch, 0
            encoder.save_adapter(out_dir)
            flag = " ★ new best"
        else:
            no_improve += 1
            flag = f"  (no improve {no_improve}/{cfg.patience})"
        rec_str = " ".join(f"{k}={v:.4f}" for k, v in val_metrics.items())
        print(f"\n[ep {epoch}] loss={avg_loss:.4f}  val: {rec_str}{flag}\n")
        if no_improve >= cfg.patience:
            break

    return {"history": history, "best_epoch": best_epoch, "best_recall500": best_val}