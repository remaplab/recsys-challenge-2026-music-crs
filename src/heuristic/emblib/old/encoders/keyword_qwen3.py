"""Keyword-extractor (BERT / SBERT / ModernBERT) + Qwen3 encoder.

Stage 1: pass query through `extractor_model` with output_attentions=True.
         Score each token by mean attention from CLS in the last layer; pick
         the top-K most attended tokens (POS-stripping handled by a regex
         filter that keeps alphabetic word-pieces only).
Stage 2: re-encode the resulting compact keyword string with frozen
         Qwen3-Embedding-0.6B. Output dim = 1024, so it dots directly
         against the organizer's metadata-qwen3 track tower.

`extractor_model` can be ANY HF AutoModel that returns attentions and uses
[CLS] at position 0 — i.e. BERT, SBERT/MPNet, ModernBERT.
"""
from __future__ import annotations

import re

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from emblib.encoders.base import QueryEncoder
from emblib.encoders.qwen3_frozen import Qwen3QueryEncoder


_KEEP_RE = re.compile(r"[A-Za-z][A-Za-z'\-]+")


class KeywordQwen3Encoder(QueryEncoder):
    def __init__(
        self,
        extractor_model: str = "bert-base-uncased",
        qwen3_model: str = "Qwen/Qwen3-Embedding-0.6B",
        top_k_tokens: int = 24,
        max_extractor_length: int = 256,
    ):
        super().__init__()
        self.ext_tok = AutoTokenizer.from_pretrained(extractor_model)
        # ModernBERT may need attn_implementation="eager" for output_attentions.
        try:
            self.ext = AutoModel.from_pretrained(
                extractor_model, dtype=torch.float16,
                output_attentions=True, attn_implementation="eager",
            ).eval()
        except (TypeError, ValueError):
            # Older HF versions don't accept attn_implementation
            self.ext = AutoModel.from_pretrained(
                extractor_model, dtype=torch.float16, output_attentions=True,
            ).eval()
        for p in self.ext.parameters():
            p.requires_grad = False

        self.qwen3 = Qwen3QueryEncoder(model_name=qwen3_model, batch_size=8)
        self.top_k_tokens = top_k_tokens
        self.max_extractor_length = max_extractor_length
        self._out_dim = 1024
        self._extractor_name = extractor_model

    @property
    def output_dim(self) -> int:
        return self._out_dim

    @torch.no_grad()
    def _extract_keywords(self, texts: list[str], device) -> list[str]:
        tok = self.ext_tok(
            texts, padding=True, truncation=True,
            max_length=self.max_extractor_length, return_tensors="pt",
        ).to(device)
        out = self.ext(**tok)
        # last layer, average over heads, take CLS (row 0) attention to all tokens.
        # Cast fp16 -> fp32: the masked_fill below uses -1e9 as a sentinel and
        # that value overflows fp16 ("value cannot be converted to type c10::Half
        # without overflow"). Pooling math also happens at fp32 in this codebase
        # (see text_track_loader.py), so we stay consistent.
        attn = out.attentions[-1].mean(dim=1)[:, 0, :].float()   # (B, L)
        keywords_per_row = []
        for b in range(attn.size(0)):
            ids = tok["input_ids"][b]
            mask = tok["attention_mask"][b].bool()
            scores = attn[b].masked_fill(~mask, -1e9)
            order = torch.argsort(scores, descending=True)
            seen, picked = set(), []
            for j in order.tolist():
                tk = self.ext_tok.convert_ids_to_tokens([int(ids[j])])[0]
                if tk in self.ext_tok.all_special_tokens:
                    continue
                # ModernBERT uses 'Ġ' word-start marker; BERT uses '##' subword marker
                tk_clean = tk.replace("##", "").replace("Ġ", "").lower().strip()
                if not _KEEP_RE.match(tk_clean) or tk_clean in seen:
                    continue
                seen.add(tk_clean)
                picked.append(tk_clean)
                if len(picked) >= self.top_k_tokens:
                    break
            keywords_per_row.append(" ".join(picked) if picked else " ")
        return keywords_per_row

    def forward(self, texts, user_cf, is_cold, device=None):
        if device is None:
            device = next(self.ext.parameters()).device
        kw = self._extract_keywords(list(texts), device)
        emb = self.qwen3.encode(kw, show_progress=False)
        t = torch.from_numpy(emb).to(device, dtype=torch.float32)
        return F.normalize(t, p=2, dim=-1)