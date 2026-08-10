"""HF backbone, no training, no projection.

Same forward pipeline as the matching text track tower (mean-pool of last
hidden state at fp32, L2-norm). user_cf / is_cold are accepted by the
QueryEncoder API but ignored — there's nowhere to put a soft-prompt token
without a projector, and the whole point of this encoder is to be the
"identity baseline" against the same-backbone text tower:

    score(q, t) = encoder(q_text)  ·  tower(t_text)

with both sides produced by the SAME pipeline. Any deviation between this
encoder and `src/tracks/text_track_loader.py` would mean we're no longer
measuring pure cosine similarity.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from emblib.encoders.base import QueryEncoder
from emblib.tracks.text_track_loader import BACKBONE_MODELS


class NativeFrozenEncoder(QueryEncoder):
    def __init__(self, backbone: str):
        super().__init__()
        if backbone not in BACKBONE_MODELS:
            raise ValueError(
                f"unknown backbone {backbone!r}; "
                f"choose from {list(BACKBONE_MODELS)}"
            )
        model_name, _modality, hidden, max_length = BACKBONE_MODELS[backbone]
        self._backbone_name = backbone
        self._H = hidden
        self._max_length = max_length

        # truncation_side='left' so when prompts overflow max_length the bottom
        # block ([CURRENT USER] + [GOAL] + [SESSION]) survives intact.
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, truncation_side="left",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or "[PAD]"

        # fp16 backbone matches what `text_track_loader.py` used to build
        # the track tower, so geometry stays consistent. Pooling math is
        # promoted to fp32 just like there.
        self.backbone = AutoModel.from_pretrained(
            model_name, dtype=torch.float16,
        ).eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

    @property
    def output_dim(self) -> int:
        return self._H

    def train(self, mode: bool = True):
        # Backbone is permanently frozen — never let dropout / LayerNorm
        # train-mode running stats switch back on.
        super().train(mode)
        self.backbone.eval()
        return self

    @torch.no_grad()
    def forward(self, texts, user_cf, is_cold, device=None):
        if device is None:
            device = next(self.parameters()).device
        tok = self.tokenizer(
            list(texts),
            padding=True, truncation=True,
            max_length=self._max_length,
            return_tensors="pt",
        ).to(device)
        # Cast last hidden state to fp32 for the pooling math, matching
        # text_track_loader.py exactly.
        last = self.backbone(**tok).last_hidden_state.float()       # (B, L, H)
        mask = tok["attention_mask"].float().unsqueeze(-1)          # (B, L, 1)
        pooled = (last * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return F.normalize(pooled, p=2, dim=-1)                     # (B, H)