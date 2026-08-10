"""Frozen Qwen3 query encoder — the query-side mirror of the Qwen3 track tower.

This is the "new way to encode" baseline: it tokenizes/encodes with the SAME
Qwen3 model, the SAME last-token pooling (`pool_last_token`), and fp32
L2-normalization that `src/tracks/qwen_track_loader.py` uses for tracks. The
only difference is the instruction prefix on the query side (default "catalog").

Output is 1024-d, so it dots directly against the Qwen3-native track tower.
user_cf / is_cold are accepted for QueryEncoder API uniformity and ignored
(there's no soft-prompt slot in a frozen encoder).

The HF model is loaded ONCE in __init__ and kept resident, so per-batch
encoding (via `encode_corpus`) is fast — unlike the one-shot `encode_qwen_texts`.
"""
from __future__ import annotations

import torch

from emblib.encoders.base import QueryEncoder
from emblib.qwen.qwen_embeddings import pool_last_token, qwen_query_prefix, resolve_torch


class Qwen3NativeFrozenEncoder(QueryEncoder):
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        instruction_name: str = "catalog",
        max_length: int = 512,
        device_arg: str = "auto",
        dtype_arg: str = "auto",
        local_files_only: bool = False,
        trust_remote_code: bool = False,
    ):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        self._out_dim = 1024
        self.instruction_name = instruction_name
        self.max_length = max_length

        _torch, device, dtype = resolve_torch(device_arg, dtype_arg)
        self._device = device

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code, local_files_only=local_files_only,
        )
        # Left padding + left truncation so the [CURRENT USER] tail always
        # survives and last-token pooling sees a real token at position -1.
        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = "left"

        try:
            model = AutoModel.from_pretrained(
                model_name, dtype=dtype,
                trust_remote_code=trust_remote_code, local_files_only=local_files_only,
            )
        except TypeError:
            model = AutoModel.from_pretrained(
                model_name, torch_dtype=dtype,
                trust_remote_code=trust_remote_code, local_files_only=local_files_only,
            )
        self.model = model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @property
    def output_dim(self) -> int:
        return self._out_dim

    def train(self, mode: bool = True):
        super().train(mode)
        self.model.eval()   # permanently frozen
        return self

    @torch.no_grad()
    def forward(self, texts, user_cf, is_cold, device=None):
        dev = next(self.model.parameters()).device
        prefixed = [qwen_query_prefix(t, self.instruction_name) for t in texts]
        tok = self.tokenizer(
            prefixed, padding=True, truncation=True,
            max_length=self.max_length, return_tensors="pt",
        ).to(dev)
        hidden = self.model(**tok).last_hidden_state
        return pool_last_token(hidden, tok["attention_mask"])   # fp32, L2-normalized