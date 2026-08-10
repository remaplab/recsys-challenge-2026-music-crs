"""Frozen Qwen3-Embedding-0.6B query encoder."""
from __future__ import annotations
from typing import List

import numpy as np
import torch


class Qwen3QueryEncoder:
    INSTRUCTION = (
        "Given a music recommendation conversation, retrieve the most relevant "
        "track from the catalog"
    )

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        device: str | None = None,
        batch_size: int = 8,
        max_length: int = 512,
        dtype: torch.dtype = torch.float16,
    ):
        from transformers import AutoModel, AutoTokenizer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading {model_name} on {self.device} (dtype={dtype})")
        # truncation_side='left' so when prompts overflow max_length, oldest
        # context is cut and the [CURRENT USER] line at the end always survives.
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, padding_side="left", truncation_side="left",
        )
        self.model = AutoModel.from_pretrained(model_name, torch_dtype=dtype).to(self.device).eval()
        self.batch_size = batch_size
        self.max_length = max_length

    @staticmethod
    def _last_token_pool(last_hidden, attention_mask):
        is_left_padded = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if is_left_padded:
            return last_hidden[:, -1]
        seq_lens = attention_mask.sum(dim=1) - 1
        return last_hidden[
            torch.arange(last_hidden.size(0), device=last_hidden.device), seq_lens
        ]

    def encode(self, queries: List[str], show_progress: bool = True) -> np.ndarray:
        from tqdm import tqdm

        prefixed = [f"Instruct: {self.INSTRUCTION}\nQuery: {q}" for q in queries]
        out = []
        iterator = range(0, len(prefixed), self.batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Qwen3 encode")

        for i in iterator:
            batch = prefixed[i : i + self.batch_size]
            tok = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                hidden = self.model(**tok).last_hidden_state
                pooled = self._last_token_pool(hidden, tok["attention_mask"])
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            out.append(pooled.float().cpu().numpy())
        return np.concatenate(out, axis=0).astype(np.float32)