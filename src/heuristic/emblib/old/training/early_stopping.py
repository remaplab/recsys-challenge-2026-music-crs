"""Tracks best metric across epochs and signals when to stop training."""
from __future__ import annotations
from pathlib import Path
import torch


class EarlyStopper:
    def __init__(self, patience: int = 3, mode: str = "max", min_delta: float = 0.0):
        assert mode in ("max", "min")
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best_value: float | None = None
        self.best_epoch: int = -1
        self.best_state: dict | None = None
        self.counter: int = 0
        self.should_stop: bool = False

    def _is_better(self, value: float) -> bool:
        if self.best_value is None:
            return True
        if self.mode == "max":
            return value > self.best_value + self.min_delta
        return value < self.best_value - self.min_delta

    def update(self, value: float, epoch: int, model_state: dict) -> bool:
        """Returns True if this is the new best."""
        if self._is_better(value):
            self.best_value = value
            self.best_epoch = epoch
            # Detach + move to CPU to avoid keeping GPU memory tied up
            self.best_state = {k: v.detach().cpu().clone() for k, v in model_state.items()}
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
            return False

    def save_best(self, path: Path, extra: dict | None = None) -> None:
        if self.best_state is None:
            raise RuntimeError("No best checkpoint to save yet")
        payload = {"model_state": self.best_state, "epoch": self.best_epoch, "value": self.best_value}
        if extra:
            payload.update(extra)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)