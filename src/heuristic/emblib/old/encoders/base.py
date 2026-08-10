"""Common encoder interface.

Every query encoder implements:
    forward(texts: list[str], user_cf: Tensor, is_cold: Tensor) -> (B, D) L2-normalized

`user_cf` and `is_cold` are accepted by every encoder for API uniformity.
Encoders that don't use user info just ignore them.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn


class QueryEncoder(nn.Module, ABC):
    """Base class for all query encoders."""

    @property
    @abstractmethod
    def output_dim(self) -> int: ...

    @abstractmethod
    def forward(
        self,
        texts: list[str],
        user_cf: torch.Tensor,
        is_cold: torch.Tensor,
        device: torch.device | None = None,
    ) -> torch.Tensor: ...

    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        return (p for p in self.parameters() if p.requires_grad)

    def n_trainable(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())

    def save_adapter(self, path: Path) -> None:
        raise NotImplementedError("Frozen encoders don't save adapters.")


@torch.no_grad()
def encode_corpus(
    encoder: QueryEncoder,
    texts: list[str],
    user_cf_arr: np.ndarray,
    is_cold_arr: np.ndarray,
    batch_size: int = 16,
    show_progress: bool = True,
) -> np.ndarray:
    """Generic corpus-encoding helper used for both queries and tracks."""
    encoder.eval()
    out = []
    iterator = range(0, len(texts), batch_size)
    if show_progress:
        from tqdm import tqdm
        iterator = tqdm(iterator, desc=f"encode ({type(encoder).__name__})")
    for s in iterator:
        e = min(s + batch_size, len(texts))
        emb = encoder(
            texts[s:e],
            torch.from_numpy(user_cf_arr[s:e]).float(),
            torch.from_numpy(is_cold_arr[s:e]).bool(),
        )
        out.append(emb.cpu().numpy().astype(np.float32))
    return np.concatenate(out, axis=0)