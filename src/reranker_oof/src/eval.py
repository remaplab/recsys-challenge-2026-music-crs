"""Official challenge metric — nDCG@K and Recall@K, macro-averaged by turn.

This is a verbatim port of the evaluation helpers from
``src/basic_candidate_generators/CG_assembly.ipynb`` (Cell 0). It is also the
contract used by the project organizers, so it MUST NOT drift from the
notebook implementation.

Semantics
---------
1. For each (session_id, turn_number) the model emits an ordered list of
   track IDs (its top-K predictions).
2. Per (session, turn) we compute nDCG@K (only one relevant item, the ground
   truth) and Recall@K (1 if ground truth is in the top-K, else 0).
3. We average those per-(session, turn) numbers WITHIN each turn_number, then
   take the mean across turn_numbers (macro-by-turn). This is the so-called
   "fair-by-turn" averaging used by the challenge.

The contract for ``evaluate``
-----------------------------
- ``predictions`` : ``dict[(session_id, turn_number) -> list[track_id]]``
- ``ground_truth_df`` : a **pandas** DataFrame with columns
  ``session_id``, ``turn_number``, ``ground_truth``. Pandas is required because
  the original challenge eval code uses ``itertuples`` on it.
"""
from __future__ import annotations

from collections import defaultdict
from math import log2
from typing import Mapping, Sequence


def ndcg_at_k(ranked_list: Sequence[str], ground_truth: str, k: int = 20) -> float:
    """nDCG@K when there is exactly one relevant item (binary relevance).

    Returns ``1 / log2(rank + 2)`` if the ground truth is found inside the
    first ``k`` predictions (0-indexed rank), else ``0``.
    """
    for r, track_id in enumerate(ranked_list[:k]):
        if track_id == ground_truth:
            return 1.0 / log2(r + 2)
    return 0.0


def recall_at_k(ranked_list: Sequence[str], ground_truth: str, k: int = 200) -> float:
    """Binary recall: 1 if ground truth is in the top-K, else 0."""
    return 1.0 if ground_truth in ranked_list[:k] else 0.0


def evaluate(
    predictions: Mapping[tuple[str, int], Sequence[str]],
    ground_truth_df,                     # pandas.DataFrame
    k: int = 20,
) -> dict:
    """Macro-by-turn nDCG@k and Recall@k.

    Parameters
    ----------
    predictions
        Mapping ``(session_id, turn_number) -> ordered list of track_ids``.
    ground_truth_df
        Pandas DataFrame with columns ``session_id``, ``turn_number``,
        ``ground_truth``.
    k
        Cutoff for both nDCG and Recall.

    Returns
    -------
    dict with keys
        - ``ndcg@<k>``           : float, macro-by-turn nDCG
        - ``recall@<k>``         : float, macro-by-turn recall
        - ``per_turn_ndcg@<k>``  : dict ``turn_number -> nDCG@k``
        - ``per_turn_recall@<k>``: dict ``turn_number -> recall@k``
    """
    # Build a fast lookup (session, turn) -> ground_truth.
    gt = {
        (row.session_id, row.turn_number): row.ground_truth
        for row in ground_truth_df.itertuples()
    }

    # Group per-turn lists of metric values, then aggregate.
    turn_ndcg: dict[int, list[float]] = defaultdict(list)
    turn_recall: dict[int, list[float]] = defaultdict(list)
    for (sid, tn), ranked_list in predictions.items():
        if (sid, tn) not in gt:
            continue                                   # row missing from GT
        g = gt[(sid, tn)]
        turn_ndcg[tn].append(ndcg_at_k(ranked_list, g, k))
        turn_recall[tn].append(recall_at_k(ranked_list, g, k))

    per_turn_ndcg = {t: sum(v) / len(v) for t, v in turn_ndcg.items()}
    per_turn_recall = {t: sum(v) / len(v) for t, v in turn_recall.items()}

    return {
        f"ndcg@{k}": (sum(per_turn_ndcg.values()) / len(per_turn_ndcg)) if per_turn_ndcg else 0.0,
        f"recall@{k}": (sum(per_turn_recall.values()) / len(per_turn_recall)) if per_turn_recall else 0.0,
        f"per_turn_ndcg@{k}": per_turn_ndcg,
        f"per_turn_recall@{k}": per_turn_recall,
    }
