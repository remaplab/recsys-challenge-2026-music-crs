"""Pluggable per-(session, turn) metric registry.

Each metric is a callable with signature::

    fn(pred_track_ids: list[str], pred_scores: list[float], gt_track_id: str) -> float

Returns NaN when the metric is undefined (e.g. empty preds or gt None).
Register new metrics via `@register_metric("name")`.
"""
from __future__ import annotations

from typing import Callable

import math

MetricFn = Callable[[list[str], list[float], str], float]
METRICS: dict[str, MetricFn] = {}


def register_metric(name: str):
    def deco(fn: MetricFn) -> MetricFn:
        METRICS[name] = fn
        return fn
    return deco


def _make_ndcg(k: int) -> MetricFn:
    def _fn(pred_track_ids, pred_scores, gt_track_id) -> float:
        if gt_track_id is None:
            return float("nan")
        if not pred_track_ids:
            # Empty preds with valid GT = model failure → score 0 (matches blind-A scoring).
            return 0.0
        try:
            rank = pred_track_ids.index(gt_track_id)
        except ValueError:
            return 0.0
        if rank >= k:
            return 0.0
        return 1.0 / math.log2(rank + 2)
    return _fn


def _make_recall(k: int) -> MetricFn:
    def _fn(pred_track_ids, pred_scores, gt_track_id) -> float:
        if gt_track_id is None:
            return float("nan")
        if not pred_track_ids:
            return 0.0
        return 1.0 if gt_track_id in pred_track_ids[:k] else 0.0
    return _fn


def _make_hit(k: int) -> MetricFn:
    return _make_recall(k)


for _k in (1, 2, 3, 4, 5, 10, 15, 20, 50, 100, 200):
    METRICS[f"ndcg@{_k}"] = _make_ndcg(_k)
    METRICS[f"recall@{_k}"] = _make_recall(_k)
    METRICS[f"hit@{_k}"] = _make_hit(_k)


def mrr_at_k(pred_track_ids, pred_scores, gt_track_id, k: int = 20) -> float:
    if gt_track_id is None or not pred_track_ids:
        return float("nan")
    try:
        rank = pred_track_ids.index(gt_track_id)
    except ValueError:
        return 0.0
    if rank >= k:
        return 0.0
    return 1.0 / (rank + 1)


METRICS["mrr@20"] = lambda p, s, g: mrr_at_k(p, s, g, 20)
