"""Holdout-subset generators for stratified reranker evaluation.

Why subsets?
------------
The Blind-A leaderboard score is computed on only 80 sessions, drawn from a
specific joint distribution of:

- conversation_goal.category × conversation_goal.specificity (the
  combinations sometimes called LL/LH/HL/HH ×  A..K)
- number of conversation turns (a discrete distribution observed in the
  Blind-A file)

If we measure offline performance only on the full holdout uniformly we may
optimize the wrong objective. We therefore build several families of holdout
subsets that approximate Blind-A's distribution and a few control distributions
(random, early/mid/late turns) so the team can diagnose where uplift comes
from and tune the reranker accordingly.

The three families exposed here
-------------------------------
- :func:`blind_a_like_subsets` — closest in spirit to Blind-A. Stratifies
  by (category, specificity), and samples ONE target turn per session from
  the Blind-A target-turn distribution.
- :func:`random_subsets` — control. Per session sample a max-turn
  uniformly in [1, 8] and keep every row with ``turn_number ≤ max_turn``.
- :func:`turn_bucket_subsets` — early / mid / late. Per session pick one
  turn uniformly inside the bucket and evaluate that single row.

Public contract
---------------
Every generator returns a list of ``(label, eval_df)`` pairs where:

- ``label`` is a human-readable seed/subset name (e.g. ``"seed0"``).
- ``eval_df`` is a polars DataFrame with columns
  ``(session_id, turn_number, ground_truth)`` — the rows to evaluate.

Downstream code uses ``eval_df`` to filter reranker predictions to the chosen
``(session_id, turn_number)`` pairs and to provide the ground-truth for the
``evaluate()`` helper.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl


# ---------------------------------------------------------------------------
# Blind-A target-turn distribution helper
# ---------------------------------------------------------------------------

def _blind_a_long(blind_a_path: Path) -> pl.DataFrame:
    """Compute the per-Blind-A-session ``(category, specificity, target_turn)``.

    Blind-A conversations contain a list of utterances (user / music /
    assistant). For sessions with ``n`` utterances the target music-turn to
    predict is ``(n + 2) // 3``: 1 utterance (user-only, no music yet) →
    turn 1; 4 utterances → turn 2; …; 22 utterances → turn 8.
    """
    blind = pl.read_parquet(blind_a_path)
    return blind.select(
        pl.col("conversation_goal").struct.field("category").alias("category"),
        pl.col("conversation_goal").struct.field("specificity").alias("specificity"),
        ((pl.col("conversations").list.len() + 2) // 3).alias("target_turn"),
    )


def _blind_a_target_turn_distribution(blind_a_path: Path) -> dict[int, int]:
    """Compute ``{target_turn -> count}`` from the Blind-A raw file.

    Marginal distribution over target_turn (ignores cat/spec). Kept for
    backwards-compat (used as the Dirichlet prior inside
    :func:`blind_a_like_subsets`).
    """
    long = _blind_a_long(blind_a_path)
    counts = (
        long.group_by("target_turn")
        .agg(pl.len().alias("count"))
        .sort("target_turn")
    )
    return {int(r["target_turn"]): int(r["count"]) for r in counts.iter_rows(named=True)}


def _blind_a_strata_distribution(blind_a_path: Path) -> dict[tuple[str, str], int]:
    """Compute ``{(category, specificity) -> count}`` (marginal over cells)."""
    long = _blind_a_long(blind_a_path)
    counts = long.group_by("category", "specificity").agg(pl.len().alias("count"))
    return {(r["category"], r["specificity"]): int(r["count"])
            for r in counts.iter_rows(named=True)}


def _blind_a_cell_turn_distribution(
    blind_a_path: Path,
) -> dict[tuple[str, str], dict[int, int]]:
    """Per-(category, specificity) target-turn histogram from Blind-A.

    Returns nested ``{(cat, spec): {turn: count}}``. Some cells have a
    single Blind-A row → singleton turn distribution. We smooth those via
    a Dirichlet prior in :func:`blind_a_like_subsets`.
    """
    long = _blind_a_long(blind_a_path)
    out: dict[tuple[str, str], dict[int, int]] = {}
    for r in long.group_by("category", "specificity", "target_turn").agg(
        pl.len().alias("count")
    ).iter_rows(named=True):
        key = (r["category"], r["specificity"])
        out.setdefault(key, {})[int(r["target_turn"])] = int(r["count"])
    return out


# ---------------------------------------------------------------------------
# (1) Blind-A-like subsets
# ---------------------------------------------------------------------------

def blind_a_like_subsets(
    holdout: pl.DataFrame,
    blind_a_path: Path,
    n_subsets: int = 10,
    seed_offset: int = 0,
    subset_size: int | None = None,
    dirichlet_alpha: float = 5.0,
    stratify_by_goal: bool = True,
) -> list[tuple[str, pl.DataFrame]]:
    """Generate holdout subsets that mimic Blind-A's joint distribution
    over ``(target_turn, category, specificity)``.

    Why a joint match?
    ------------------
    Blind-A's leaderboard scores a stratified mix of conversation goals
    (``category`` × ``specificity``) AND a stratified mix of target turns
    (sessions are cut at varying utterance counts). Holdout has all 8 turns
    per session and its ``(category, specificity)`` marginal differs from
    Blind-A's. A subset that only matches the turn marginal (the previous
    behaviour of this function) silently inherits holdout's cat/spec
    distribution — which is NOT Blind-A's. This new version matches the
    full joint via two stages.

    Sampling mechanic
    -----------------
    Pre-compute once from Blind-A:
      * ``P_blind_a(cat, spec)``  — cell marginal.
      * ``P_blind_a(t)``          — turn marginal.
      * ``c[cat, spec][t]``       — cell-conditional turn counts (sparse).
      * Smoothed cell-conditional turn distribution:
        ``P(t | cat, spec) ∝ c[cat, spec][t] + α · P_blind_a(t)``
        — a Dirichlet posterior with the Blind-A turn marginal as the
        prior. Cells with many Blind-A samples retain their empirical
        shape; singleton cells regress toward the population marginal
        (NOT uniform) so the early-turn bias of Blind-A is preserved
        even where no data exists. ``α`` is the prior strength in
        pseudo-samples (default 5).

    Per subset:
      1. **Cell-weighted session resample** (with replacement). Each
         holdout session gets a weight proportional to
         ``P_blind_a(cell) / P_holdout(cell)``. Sessions in cells absent
         from Blind-A get weight 0. Drawing ``subset_size`` sessions with
         these weights gives a sample whose ``(cat, spec)`` marginal
         matches Blind-A in expectation.
      2. **Per-session target turn**. For each sampled session, draw one
         target turn from the smoothed cell-conditional
         ``P(t | session.cat, session.spec)``.
      3. **Join back to holdout** on ``(session_id, turn_number)`` to
         attach the ground-truth track for metric computation.

    Different subsets differ in BOTH the session resample and the per-row
    turn draw → tighter mimicry of Blind-A's sampling noise.

    Parameters
    ----------
    holdout
        ``data/splitK/holdout_test.parquet`` (assembled, one row per turn).
    blind_a_path
        Path to the raw Blind-A parquet.
    n_subsets
        Number of subset variants to generate.
    seed_offset
        Added to the seed so the function can be called twice with disjoint
        RNG streams.
    subset_size
        Sessions per subset. ``None`` → use the holdout's session count
        (low-variance population estimator). Pass ``80`` (Blind-A size) to
        replicate the Monte-Carlo noise of the actual leaderboard sample.
    dirichlet_alpha
        Smoothing strength for the cell-conditional turn distribution.
        ``0`` = pure empirical (overfits sparse cells); ``∞`` = pure
        marginal prior (ignores cell info). Default 5 keeps strong empirical
        signal for cells with ≥5 samples and smoothly falls back to the
        marginal as the cell shrinks.
    stratify_by_goal
        Attach ``(category, specificity)`` columns to the output for
        downstream per-stratum breakdown plots.

    Returns
    -------
    list[(label, eval_df)]
        ``eval_df`` columns:
          - ``session_id``, ``turn_number``, ``ground_truth``
          - ``category``, ``specificity`` (if ``stratify_by_goal``)
    """
    # --- (1) Per-session (cat, spec) of holdout ---
    if "conversation_goal" in holdout.columns:
        strata_df = (
            holdout.unique(subset="session_id", keep="first")
            .select(
                "session_id",
                pl.col("conversation_goal").struct.field("category").alias("category"),
                pl.col("conversation_goal").struct.field("specificity").alias("specificity"),
            )
        )
    else:
        # No goal info → fall back to uniform cell so we degrade gracefully
        # to "turn-marginal-only" behaviour, identical to the prior version.
        strata_df = holdout.unique(subset="session_id", keep="first").select(
            "session_id",
            pl.lit("__unk__", dtype=pl.Utf8).alias("category"),
            pl.lit("__unk__", dtype=pl.Utf8).alias("specificity"),
        )

    # --- (2) Blind-A statistics ---
    blind_cells = _blind_a_strata_distribution(blind_a_path)
    blind_turn_marginal = _blind_a_target_turn_distribution(blind_a_path)
    blind_cell_turns = _blind_a_cell_turn_distribution(blind_a_path)
    total_blind = float(sum(blind_cells.values()))

    turns_axis = np.array(sorted(blind_turn_marginal.keys()), dtype=np.int64)
    marginal_w = np.array(
        [blind_turn_marginal[t] for t in turns_axis], dtype=np.float64
    )
    marginal_w = marginal_w / marginal_w.sum()                  # Dirichlet prior

    # Pre-compute smoothed per-cell turn distributions over ``turns_axis``.
    cell_turn_pmf: dict[tuple[str, str], np.ndarray] = {}
    for cell, freq in blind_cells.items():
        counts = blind_cell_turns.get(cell, {})
        empirical = np.array([counts.get(int(t), 0) for t in turns_axis], dtype=np.float64)
        smoothed = empirical + dirichlet_alpha * marginal_w
        cell_turn_pmf[cell] = smoothed / smoothed.sum()

    # --- (3) Per-session sampling weights ---
    # Count holdout sessions per cell.
    holdout_cell_count = (
        strata_df.group_by("category", "specificity")
        .agg(pl.len().alias("h_count"))
    )
    holdout_count_map = {
        (r["category"], r["specificity"]): int(r["h_count"])
        for r in holdout_cell_count.iter_rows(named=True)
    }

    # Per-session weight ∝ P_blind_a(cell) / P_holdout(cell). Sessions in
    # cells absent from Blind-A get 0 → never sampled.
    sessions_in_order = strata_df.sort("session_id")
    session_ids = sessions_in_order["session_id"].to_list()
    session_cats = sessions_in_order["category"].to_list()
    session_specs = sessions_in_order["specificity"].to_list()
    weights = np.zeros(len(session_ids), dtype=np.float64)
    for i, (cat, spec) in enumerate(zip(session_cats, session_specs)):
        b_freq = blind_cells.get((cat, spec), 0) / total_blind
        h_count = holdout_count_map.get((cat, spec), 0)
        if b_freq > 0 and h_count > 0:
            weights[i] = b_freq / h_count
    w_sum = weights.sum()
    if w_sum <= 0:
        raise ValueError(
            "No holdout sessions overlap with any Blind-A (cat, spec) cell"
        )
    weights = weights / w_sum

    n_per_subset = int(subset_size) if subset_size is not None else len(session_ids)

    # --- (4) Per subset: resample sessions, draw turn from cell-conditional ---
    # Use a polars-friendly mapping ``session_id → (cat, spec)`` for the
    # sample loop (Python dict is faster than per-row joins for 80–1000 rows).
    sid_to_cell: dict[str, tuple[str, str]] = {
        sid: (c, s) for sid, c, s in zip(session_ids, session_cats, session_specs)
    }
    turn_axis_list = turns_axis.tolist()

    out: list[tuple[str, pl.DataFrame]] = []
    for s_idx in range(n_subsets):
        seed = seed_offset + s_idx
        rng = np.random.default_rng(seed)

        # Resample sessions WITH replacement → Blind-A-like cat/spec marginal.
        chosen_idx = rng.choice(len(session_ids), size=n_per_subset, p=weights, replace=True)
        chosen_sids = [session_ids[i] for i in chosen_idx]
        chosen_cells = [(session_cats[i], session_specs[i]) for i in chosen_idx]

        # Draw a target turn per sampled session from its (smoothed) cell pmf.
        # Cells absent from cell_turn_pmf (shouldn't happen since weights would
        # be 0) fall back to the marginal.
        sampled_turns = np.empty(n_per_subset, dtype=np.int64)
        for j, cell in enumerate(chosen_cells):
            pmf = cell_turn_pmf.get(cell, marginal_w)
            sampled_turns[j] = int(rng.choice(turn_axis_list, p=pmf))

        # Build a DataFrame with a stable per-row identifier — sessions
        # can repeat under WR sampling, so we tag each row with its draw
        # index to avoid join blow-up. ``_draw`` is dropped in the final
        # select.
        draws = pl.DataFrame({
            "_draw":       np.arange(n_per_subset, dtype=np.int64),
            "session_id":  chosen_sids,
            "turn_number": sampled_turns.astype(np.int64),
        })

        # Inner-join holdout on (session_id, turn_number) to attach the
        # ground-truth track for scoring. Holdout has at most 1 row per (session, turn),
        # so the join cardinality is preserved.
        eval_df = (
            draws.join(
                holdout.select("session_id", "turn_number", "track_id"),
                on=["session_id", "turn_number"], how="left",
            )
            .filter(pl.col("track_id").is_not_null())     # session×turn missing → drop
            .select(
                "session_id", "turn_number",
                pl.col("track_id").alias("ground_truth"),
            )
        )
        if stratify_by_goal:
            eval_df = eval_df.join(strata_df, on="session_id", how="left")
        out.append((f"seed{seed}_n{n_per_subset}", eval_df))
    return out


# ---------------------------------------------------------------------------
# (2) Random subsets (truncate session at random max-turn)
# ---------------------------------------------------------------------------

def random_subsets(
    holdout: pl.DataFrame,
    n_subsets: int = 20,
    seed_offset: int = 1000,
) -> list[tuple[str, pl.DataFrame]]:
    """Random per-session truncation control.

    For each subset:
      1. Per session sample ``max_turn`` uniformly in [1, 8].
      2. Keep every holdout row whose ``turn_number ≤ max_turn``.

    Resulting metric is macro-by-turn nDCG@20 on the kept rows — gives an
    "if we stopped the conversation at a random point" signal.
    """
    sessions = holdout.select("session_id").unique().sort("session_id")
    session_ids: list[str] = sessions["session_id"].to_list()

    out: list[tuple[str, pl.DataFrame]] = []
    for s_idx in range(n_subsets):
        seed = seed_offset + s_idx
        rng = np.random.default_rng(seed)
        max_turns = rng.integers(low=1, high=9, size=len(session_ids))  # high is exclusive
        sel = pl.DataFrame({
            "session_id": session_ids,
            "_max_turn": max_turns.astype(np.int64),
        })
        eval_df = (
            holdout.join(sel, on="session_id", how="inner")
            .filter(pl.col("turn_number") <= pl.col("_max_turn"))
            .select(
                "session_id", "turn_number",
                pl.col("track_id").alias("ground_truth"),
            )
        )
        out.append((f"seed{seed}", eval_df))
    return out


# ---------------------------------------------------------------------------
# (3) Turn-bucket subsets (early / mid / late)
# ---------------------------------------------------------------------------

def turn_bucket_subsets(
    holdout: pl.DataFrame,
    bucket: tuple[int, int],
    n_subsets: int = 10,
    seed_offset: int = 2000,
) -> list[tuple[str, pl.DataFrame]]:
    """Per-session pick ONE turn uniformly inside ``bucket`` (inclusive).

    Used three times in the launcher:
      - early   = (1, 3)
      - mid     = (4, 6)
      - late    = (7, 8)

    Sessions whose ``max(turn_number) < bucket[0]`` are skipped (the session
    never reached the bucket). Sessions whose max turn falls inside the
    bucket use that max as the upper bound to avoid drawing a turn that
    doesn't exist.
    """
    low, high = bucket
    assert 1 <= low <= high <= 8, "bucket must lie in [1, 8]"

    # Per-session max turn (used to clamp the random draw).
    max_turn_per_session = (
        holdout.group_by("session_id")
        .agg(pl.col("turn_number").max().alias("_max_t"))
    )
    session_ids: list[str] = max_turn_per_session["session_id"].to_list()
    max_ts: list[int] = max_turn_per_session["_max_t"].to_list()

    out: list[tuple[str, pl.DataFrame]] = []
    for s_idx in range(n_subsets):
        seed = seed_offset + s_idx
        rng = np.random.default_rng(seed)
        sampled = np.empty(len(session_ids), dtype=np.int64)
        for i, mt in enumerate(max_ts):
            if mt < low:
                # Session doesn't reach this bucket — mark as -1, filtered out.
                sampled[i] = -1
            else:
                hi_clamped = min(high, mt)
                sampled[i] = int(rng.integers(low=low, high=hi_clamped + 1))
        sel = pl.DataFrame({
            "session_id":  session_ids,
            "turn_number": sampled,
        }).filter(pl.col("turn_number") > 0)
        eval_df = (
            holdout.join(sel, on=["session_id", "turn_number"], how="inner")
            .select(
                "session_id", "turn_number",
                pl.col("track_id").alias("ground_truth"),
            )
        )
        out.append((f"seed{seed}", eval_df))
    return out


# ---------------------------------------------------------------------------
# (4) Per-turn full holdout (one subset per turn ∈ {1..8})
# ---------------------------------------------------------------------------

def per_turn_full_holdout(holdout: pl.DataFrame) -> list[tuple[str, pl.DataFrame]]:
    """One subset per turn — the entire holdout filtered to ``turn_number == t``.

    Eight subsets total. Provides per-turn diagnostic without sampling.
    """
    out: list[tuple[str, pl.DataFrame]] = []
    for t in range(1, 9):
        eval_df = (
            holdout.filter(pl.col("turn_number") == t)
            .select(
                "session_id", "turn_number",
                pl.col("track_id").alias("ground_truth"),
            )
        )
        out.append((f"turn{t}", eval_df))
    return out
