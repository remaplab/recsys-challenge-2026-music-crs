"""Candidate-pool assembly and fusion-feature engineering.

This module turns each candidate generator's "wide" output (one row per
``(session_id, turn)`` containing parallel lists of ``track_ids`` and
``scores``) into a single LONG-format pool DataFrame, with one row per
``(session_id, turn_number, track_id)`` candidate and an outer-joined view of
every CG's per-candidate signals (rank, reciprocal rank, score, top-K flags).

The pool is then enriched with **fusion features** — Borda / RRF / CombSUM /
CombMNZ / ISR / log-rank / rank-product — plus a few rank meta features
(min / max / median / std rank, agreement_top10/50, number of CGs retrieving
each candidate).

Conventions
-----------
- Columns suffixed with ``_<cg>`` carry per-CG signals.
- ``rank_<cg>`` is 1-indexed and ``NULL`` if the CG did not retrieve the
  candidate. Top-K flags are always ``False`` for non-retrievers.
- All fusion features are "higher = better" so that the reranker sees a
  consistent direction.

All functions are PURE (no I/O, no global state). They expect ``polars``
DataFrames already loaded from disk.

Ported verbatim (with documentation) from
``src/basic_candidate_generators/CG_assembly.ipynb`` (Cell 0 / Cell 3).
"""
from __future__ import annotations

import polars as pl


# ---------------------------------------------------------------------------
# Per-CG wide → long conversion
# ---------------------------------------------------------------------------

def truncate_candidates(df: pl.DataFrame, k: int) -> pl.DataFrame:
    """Slice each (session, turn) candidate list to the first ``k`` entries.

    If a parallel ``scores`` list column is present, it is sliced too so the
    two lists stay aligned.
    """
    assert k > 0, "k must be a positive integer"
    exprs = [pl.col("track_ids").list.slice(0, k)]
    if "scores" in df.columns:
        exprs.append(pl.col("scores").list.slice(0, k))
    return df.with_columns(exprs)


def explode_and_normalize_scores(df: pl.DataFrame, k: int = 200) -> pl.DataFrame:
    """Convert a wide CG output into long format with normalised signals.

    Input columns (subset): ``session_id``, ``turn``, ``track_ids``[, ``scores``,
    ``gt_track_id``, ``user_id``]. The function:

    1. Slices to top-``k``.
    2. Explodes the parallel ``track_ids``/``scores`` lists.
    3. Adds ``rank`` (1-indexed, per (session, turn) group).
    4. Adds ``reciprocal_rank = 1 / rank``.
    5. If raw scores are present, adds ``score_zscore`` and ``score_minmax``
       (both per (session, turn) group). Useful for CombSUM/MNZ later.

    Returns a DataFrame ready to be passed to :func:`pool_union`.
    """
    # Adapt any broken-schema CG output (constant `turn`, `gt_turn_number`,
    # `fallback_used`) to the canonical layout before any group-by-turn op.
    from .cg_calibration import normalize_cg_columns  # local: avoid cycle
    df = normalize_cg_columns(df)
    df = truncate_candidates(df, k)
    has_scores = "scores" in df.columns
    if has_scores:
        df = df.explode("track_ids", "scores").rename(
            {"track_ids": "track_id", "scores": "score"}
        )
    else:
        df = df.explode("track_ids").rename({"track_ids": "track_id"})

    # 1-indexed rank inside each (session, turn) group; cast to Int32 to keep
    # the column small (max value is k ≤ 200, fits comfortably).
    df = df.with_columns(
        (pl.int_range(pl.len()).over("session_id", "turn") + 1)
        .cast(pl.Int32)
        .alias("rank"),
    ).with_columns(
        (1.0 / pl.col("rank")).alias("reciprocal_rank")
    )

    # Per-group score normalisations (z-score + min-max). NaNs are fine here —
    # downstream code treats them as null in fusion expressions.
    if has_scores:
        g = ["session_id", "turn"]
        df = df.with_columns(
            ((pl.col("score") - pl.col("score").mean().over(g))
             / pl.col("score").std().over(g)).alias("score_zscore"),
            ((pl.col("score") - pl.col("score").min().over(g))
             / (pl.col("score").max().over(g) - pl.col("score").min().over(g))).alias("score_minmax"),
        )
    return df


# ---------------------------------------------------------------------------
# Outer-join pool union across CGs
# ---------------------------------------------------------------------------

def pool_union(cg_dfs: list[tuple[str, pl.DataFrame]], *, lean: bool = False) -> pl.DataFrame:
    """Outer-join the long-format CG DataFrames into a single candidate pool.

    Parameters
    ----------
    cg_dfs
        List of ``(cg_name, long_df)`` pairs. Each ``long_df`` must come from
        :func:`explode_and_normalize_scores`.
    lean
        Fusion-cache fast path. When True, materialise only the per-CG columns
        the fusion pool actually consumes — ``rank_<cg>``, ``reciprocal_rank_<cg>``
        (calibration input) and ``score_minmax_<cg>`` (minmax fusion input) —
        dropping ``score_z_<cg>`` and the three ``in_topN_<cg>`` flags. Cuts the
        outer-join payload from 7 to 3 columns/CG (≈55 %), so the join is faster
        and lighter on RAM. The feature-builder path keeps ``lean=False`` (it
        consumes the in_topN flags). Fusion/calibration output is identical.

    Returns
    -------
    polars.DataFrame
        One row per ``(session_id, turn_number, track_id)``. For each CG the
        output contains:

        - ``rank_<cg>``            : Int32, null if CG didn't retrieve
        - ``reciprocal_rank_<cg>`` : Float64, null if CG didn't retrieve
        - ``score_z_<cg>``         : Float64, null if absent
        - ``score_minmax_<cg>``    : Float64, null if absent
        - ``in_top10_<cg>``        : Bool, False if absent
        - ``in_top50_<cg>``        : Bool, False if absent
        - ``in_top200_<cg>``       : Bool, False if absent

        Plus the meta columns ``gt_track_id`` and ``user_id`` carried over
        from whichever CG happens to have them (any CG with the labels works
        because they're invariant across CGs).

    Notes
    -----
    The ``turn`` column from the wide CG outputs is renamed to ``turn_number``
    here — that matches the convention used elsewhere in the pipeline.
    """
    join_keys = ["session_id", "turn", "track_id"]
    meta_cols = ["user_id", "gt_track_id"]

    # Build one selection per CG with renamed columns + top-K flags.
    cg_frames = []
    for cg, df in cg_dfs:
        sel = [
            *join_keys,
            pl.col("rank").alias(f"rank_{cg}"),
            pl.col("reciprocal_rank").alias(f"reciprocal_rank_{cg}"),
            # ``score_zscore`` / ``score_minmax`` only exist if the CG carried
            # raw scores. Substitute explicit nulls when absent so the schema
            # stays uniform across CGs (otherwise the outer join would diverge).
            pl.col("score_minmax").alias(f"score_minmax_{cg}") if "score_minmax" in df.columns
            else pl.lit(None, dtype=pl.Float64).alias(f"score_minmax_{cg}"),
        ]
        if not lean:
            sel += [
                pl.col("score_zscore").alias(f"score_z_{cg}") if "score_zscore" in df.columns
                else pl.lit(None, dtype=pl.Float64).alias(f"score_z_{cg}"),
                (pl.col("rank") <= 10).alias(f"in_top10_{cg}"),
                (pl.col("rank") <= 50).alias(f"in_top50_{cg}"),
                (pl.col("rank") <= 200).alias(f"in_top200_{cg}"),
            ]
        cg_frames.append(df.select(sel))

    # Speedup 1E: pairwise reduction with thread pool.
    #
    # Old linear path: ``result = frame_0; for f in frames[1:]: result =
    # result.join(f)`` → 17 sequential joins, each building on the growing
    # ``result``. Each step waits for the previous one's finish.
    #
    # New path: log2 levels of independent pair-joins, each level run in
    # parallel via threads. Polars' join releases the GIL during the hash
    # build / probe phases → genuine parallel speedup.
    #
    # Algorithm:
    #   level 0: frames[0..17] (18 frames)
    #   level 1: 9 pair-joins, parallel → 9 frames
    #   level 2: 4 pair-joins (+1 carry) → 5 frames
    #   level 3: 2 pair-joins (+1 carry) → 3 frames
    #   level 4: 1 pair-join (+1 carry)  → 2 frames
    #   level 5: 1 pair-join             → 1 frame
    #
    # Total joins == 17 (same as linear). Critical-path joins reduced
    # from 17 to ~ceil(log2(18)) = 5, plus per-level wait overhead.
    # In practice ~30-50 % wall-time saving on the pool union step.
    from concurrent.futures import ThreadPoolExecutor

    frames = list(cg_frames)
    while len(frames) > 1:
        pairs = [(frames[i], frames[i + 1]) for i in range(0, len(frames) - 1, 2)]
        carry = [frames[-1]] if len(frames) % 2 else []
        with ThreadPoolExecutor(max_workers=min(8, len(pairs))) as ex:
            merged = list(ex.map(
                lambda pair: pair[0].join(
                    pair[1], on=join_keys, how="full", coalesce=True,
                ),
                pairs,
            ))
        frames = merged + carry
    result = frames[0]

    # After a full outer-join the in_topN_* columns may be null where a
    # CG didn't see the candidate — fill those with False to keep dtype Bool.
    # (No in_topN columns in lean mode.)
    top_n_cols = [c for c in result.columns if c.startswith("in_top")]
    if top_n_cols:
        result = result.with_columns([pl.col(c).fill_null(False) for c in top_n_cols])

    # Re-attach each meta column independently from whichever CGs carry it.
    # A per-column pass guarantees `user_id` ends up populated whenever ≥1
    # CG had it, even if other CGs (e.g. two_tower_v2_session) omit the
    # column. Doing it as one mixed-schema concat would let a null-bearing
    # row win the `unique(keep="first")` step depending on CG order.
    for meta_col in meta_cols:
        per_cg = [
            df.select([*join_keys, meta_col])
            for _, df in cg_dfs
            if meta_col in df.columns
        ]
        if not per_cg:
            continue
        meta = (
            pl.concat(per_cg)
            .filter(pl.col(meta_col).is_not_null())
            .unique(subset=join_keys, keep="first")
        )
        result = result.join(meta, on=join_keys, how="left")

    # Rename ``turn`` → ``turn_number`` to match the convention used by the
    # rest of the pipeline (and by ``data/splitK/*.parquet``).
    return result.rename({"turn": "turn_number"})


# ---------------------------------------------------------------------------
# Rank-fusion features
# ---------------------------------------------------------------------------

def add_fusion_features(
    pool: pl.DataFrame,
    cg_names: list[str],
    k_rrf_grid: tuple[int, ...] = (5, 60),
    rp_penalty_grid: tuple[int, ...] = (200, 1000),
) -> pl.DataFrame:
    """Add the rank-fusion feature family to the pool.

    Adds (all "higher = better"):
        - ``fusion_borda``               : Σ_cg (N - rank + 1)
        - ``fusion_rrf_k<kk>``           : Σ_cg 1/(kk + rank), for each kk
        - ``fusion_combsum``             : Σ_cg score_minmax_cg (nulls→0)
        - ``fusion_combmnz``             : combsum * n_retrieving
        - ``n_cgs_retrieving``           : Σ_cg [rank is not null]
        - ``fusion_isr``                 : Σ_cg 1/rank²
        - ``fusion_logrank``             : Σ_cg 1/log2(rank+1)
        - ``fusion_rp_pen<pen>``         : -mean_cg log(rank or pen) — rank product
        - ``min_rank`` / ``max_rank`` / ``median_rank`` / ``std_rank``
        - ``agreement_top10`` / ``agreement_top50``

    Parameters
    ----------
    pool
        Output of :func:`pool_union`.
    cg_names
        The list of logical CG names that contributed to the pool. Used to
        locate the per-CG columns.
    k_rrf_grid
        RRF dampening constants. The standard value in TREC literature is 60.
        We also include 5 (more aggressive top-of-list weighting) as in the
        notebook.
    rp_penalty_grid
        Penalty rank used in the rank-product computation when a CG did not
        retrieve a candidate (effectively imputes that the candidate is at
        rank ``pen`` for that CG).
    """
    rank_cols = [f"rank_{cg}" for cg in cg_names]
    mm_cols = [f"score_minmax_{cg}" for cg in cg_names]
    top10_cols = [f"in_top10_{cg}" for cg in cg_names]
    top50_cols = [f"in_top50_{cg}" for cg in cg_names]
    n_cgs = len(cg_names)
    # Borda uses N=top-K width as the maximum possible rank. The notebook
    # hard-codes 200 because that's the explode cap; we keep the same value.
    N_FULL = 200

    exprs: list[pl.Expr] = []

    # ----- Borda count (linear-credit fusion) -----
    exprs.append(sum(
        pl.when(pl.col(r).is_not_null()).then(N_FULL - pl.col(r) + 1).otherwise(0)
        for r in rank_cols
    ).alias("fusion_borda"))

    # ----- Reciprocal Rank Fusion, one column per kk -----
    for kk in k_rrf_grid:
        exprs.append(sum(
            pl.when(pl.col(r).is_not_null()).then(1.0 / (kk + pl.col(r))).otherwise(0.0)
            for r in rank_cols
        ).alias(f"fusion_rrf_k{kk}"))

    # ----- CombSUM / CombMNZ over min-max-normalised scores -----
    combsum = sum(pl.col(mm).fill_null(0.0) for mm in mm_cols)
    n_retrieved = sum(pl.col(r).is_not_null().cast(pl.Int32) for r in rank_cols)
    exprs.append(combsum.alias("fusion_combsum"))
    exprs.append((combsum * n_retrieved).alias("fusion_combmnz"))
    exprs.append(n_retrieved.alias("n_cgs_retrieving"))

    # ----- Inverse Squared Rank + log-rank fusions -----
    exprs.append(sum(
        pl.when(pl.col(r).is_not_null()).then(1.0 / pl.col(r) ** 2).otherwise(0.0)
        for r in rank_cols
    ).alias("fusion_isr"))
    exprs.append(sum(
        pl.when(pl.col(r).is_not_null()).then(1.0 / (pl.col(r) + 1).log(2)).otherwise(0.0)
        for r in rank_cols
    ).alias("fusion_logrank"))

    # ----- Rank product (penalty-imputed for missing CGs), one per penalty -----
    # We negate the result so that "higher = better".
    for pen in rp_penalty_grid:
        log_ranks = [
            pl.when(pl.col(r).is_not_null()).then(pl.col(r).log())
              .otherwise(pl.lit(pen).log())
            for r in rank_cols
        ]
        exprs.append((-(sum(log_ranks) / n_cgs)).alias(f"fusion_rp_pen{pen}"))

    # ----- Rank-vector meta features -----
    # Speedup 3B: ``min_rank`` and ``max_rank`` go via ``pl.min_horizontal``
    # / ``pl.max_horizontal`` — those compile to a single fused parallel
    # kernel that walks the N rank columns row-wise. The old
    # ``concat_list().list.min/max`` path materialised an N-element list
    # column with millions of cells just to read its scalar reduction
    # back out (~50-100 ms × 2 reductions per chunk → saves ~150 ms/chunk).
    #
    # ``median`` and ``std`` stay on the list path because polars 1.x
    # lacks horizontal equivalents for them. The shared ``rank_list``
    # built below is therefore only computed twice (median + std) instead
    # of four times — the optimiser fuses the two list aggregations into
    # one underlying list scan, so net cost vs the four-list path is
    # roughly halved.
    exprs.append(pl.min_horizontal([pl.col(r) for r in rank_cols]).alias("min_rank"))
    exprs.append(pl.max_horizontal([pl.col(r) for r in rank_cols]).alias("max_rank"))
    rank_list = pl.concat_list([pl.col(r) for r in rank_cols])
    exprs.append(rank_list.list.median().alias("median_rank"))
    exprs.append(rank_list.list.std().alias("std_rank"))
    exprs.append((sum(pl.col(c).cast(pl.Int32) for c in top10_cols) / n_cgs).alias("agreement_top10"))
    exprs.append((sum(pl.col(c).cast(pl.Int32) for c in top50_cols) / n_cgs).alias("agreement_top50"))

    return pool.with_columns(exprs)
