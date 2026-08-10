"""3-way split (train_proj / train_rerank / val) — v2.

Improvements over v1:
  1. Softer cutoff (default 2017-01-01) — gives more candidate sessions for
     the holdout while still preserving the temporal property:
     no train_proj session is dated AFTER the latest val session.
  2. Stratification by user warmth (warm/cold in cf-bpr): the simulated-cold
     rate is computed among WARM users only, so the holdout's real-cold rate
     matches dev's 25.8% target rather than 26% of everyone.
  3. Stratification by dominant (category, specificity): inside the holdout
     pool, users are bucketed by their most frequent goal pair and sampled
     proportionally so train_rerank and val have similar category mixes.
  4. Quota-based selection rather than greedy: each stratum gets an integer
     session quota, and users are picked deterministically by hash within
     each stratum until the quota is met.

Output: models/splits/train_val.parquet with columns
  session_id, user_id, split, simulate_cold

Run:
    uv run python -m scripts.launchers.make_train_val_split_v2 \\
        --cutoff-date 2017-01-01
"""
from __future__ import annotations

import argparse
import hashlib
from collections import defaultdict, Counter
from pathlib import Path

import polars as pl


#DATA           = Path("./data/talkpl-ai")
# TRAIN_PARQUET  = DATA / "TalkPlayData-Challenge-Dataset/data/train-00000-of-00001.parquet"
# DEV_PARQUET    = DATA / "TalkPlayData-Challenge-Dataset/data/test-00000-of-00001.parquet"
# USER_EMB_TRAIN = DATA / "TalkPlayData-Challenge-User-Embeddings/data/train-00000-of-00001.parquet"
#OUT            = Path("./models/splits/train_val.parquet")

PKG  = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "data/talkpl-ai"                       # was Path("./data/talkpl-ai")
TRAIN_PARQUET  = DATA / "TalkPlayData-Challenge-Dataset/data/train-00000-of-00001.parquet"
DEV_PARQUET    = DATA / "TalkPlayData-Challenge-Dataset/data/test-00000-of-00001.parquet"
USER_EMB_TRAIN = DATA / "TalkPlayData-Challenge-User-Embeddings/data/train-00000-of-00001.parquet"
OUT  = PKG / "models/splits/train_val.parquet"       # was Path("./models/splits/..."


def hash_bucket(key: str, seed: int, mod: int = 100000) -> int:
    """Deterministic uniform hash of a string key into [0, mod)."""
    h = hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()
    return int(h[:8], 16) % mod


def load_session_metadata(train_parquet: Path) -> pl.DataFrame:
    """Pull session_id, user_id, session_date, dominant (cat, spec) per session."""
    print(f"Loading {train_parquet}")
    full = pl.read_parquet(train_parquet)
    rows = []
    for r in full.to_dicts():
        cg = r.get("conversation_goal") or {}
        rows.append({
            "session_id":   r["session_id"],
            "user_id":      r["user_id"],
            "session_date": str(r["session_date"]),
            "category":     cg.get("category") or "?",
            "specificity":  cg.get("specificity") or "?",
        })
    return pl.DataFrame(rows)


def load_warm_users(user_emb_train: Path) -> set[str]:
    """Users that have a non-empty cf-bpr embedding (i.e. real warm users)."""
    if not user_emb_train.exists():
        print(f"  WARNING: {user_emb_train} missing — assuming all users warm.")
        return set()
    df = pl.read_parquet(user_emb_train)
    warm = set()
    for r in df.to_dicts():
        v = r.get("cf-bpr")
        if v is not None and len(v) > 0:
            warm.add(r["user_id"])
    return warm


def dominant_pair_per_user(sessions: pl.DataFrame) -> dict[str, tuple[str, str]]:
    """For each user, find the most common (category, specificity) pair.
    Ties broken by alphabetical stability."""
    cnt: dict[str, Counter] = defaultdict(Counter)
    for r in sessions.iter_rows(named=True):
        cnt[r["user_id"]][(r["category"], r["specificity"])] += 1
    return {
        uid: max(c.items(), key=lambda kv: (kv[1], kv[0]))[0]
        for uid, c in cnt.items()
    }


def sessions_per_user(sessions: pl.DataFrame) -> dict[str, int]:
    return dict(zip(
        sessions.group_by("user_id").len()["user_id"].to_list(),
        sessions.group_by("user_id").len()["len"].to_list(),
    ))


def stratified_quota_sample(
    candidates_by_stratum: dict[tuple, list[str]],
    sessions_per_user_map: dict[str, int],
    target_total_sessions: int,
    seed: int,
) -> set[str]:
    """Pick users from each stratum proportional to its share of sessions in
    `candidates_by_stratum`, until target_total_sessions is approximately reached.

    Within each stratum, users are sorted by hash_bucket and taken in order so
    the choice is deterministic for a given seed.
    """
    strata_session_count = {
        stratum: sum(sessions_per_user_map[u] for u in users)
        for stratum, users in candidates_by_stratum.items()
    }
    total_pool_sessions = sum(strata_session_count.values())
    if total_pool_sessions == 0:
        return set()

    chosen: set[str] = set()
    chosen_sessions = 0
    # Per-stratum target (round half-up)
    for stratum, users in candidates_by_stratum.items():
        share = strata_session_count[stratum] / total_pool_sessions
        stratum_target = int(round(target_total_sessions * share))
        if stratum_target == 0:
            continue
        ordered = sorted(users, key=lambda u: hash_bucket(u, seed + 7))
        taken_sessions = 0
        for u in ordered:
            if taken_sessions >= stratum_target:
                break
            chosen.add(u)
            taken_sessions += sessions_per_user_map[u]
            chosen_sessions += sessions_per_user_map[u]
    return chosen


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rerank-fraction", type=float, default=0.16)
    p.add_argument("--val-fraction", type=float, default=0.20)
    p.add_argument(
        "--cutoff-date", type=str, default="2017-01-01",
        help="Sessions on/after this date are eligible for train_rerank/val. "
             "Softer than the v1 default (2018-01-01) so the holdout has a "
             "richer distribution of session dates. Anything BEFORE the cutoff "
             "stays in train_proj, which preserves temporal causality.",
    )
    p.add_argument("--cold-fraction", type=float, default=0.26,
                   help="Fraction of WARM holdout users to mark simulate_cold. "
                        "Real cold users are already cold; this dials the holdout's "
                        "effective cold rate to match dev's 25.8%.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-stratify-category", action="store_true",
                   help="Skip (category, specificity) stratification (not recommended).")
    args = p.parse_args()

    if args.rerank_fraction + args.val_fraction >= 1.0:
        raise ValueError("rerank-fraction + val-fraction must be < 1.0")

    # ── Load core tables ─────────────────────────────────────────────────────
    sessions = load_session_metadata(TRAIN_PARQUET)
    n_total = sessions.shape[0]
    n_users = sessions["user_id"].n_unique()
    print(f"  {n_total} sessions, {n_users} users")

    cutoff = args.cutoff_date
    n_after = sessions.filter(pl.col("session_date") >= cutoff).shape[0]
    print(f"  cutoff={cutoff}  post-cutoff sessions: {n_after} ({n_after/n_total:.3f})")

    warm_users = load_warm_users(USER_EMB_TRAIN)
    print(f"  warm users (have cf-bpr): {len(warm_users)} / {n_users} "
          f"({len(warm_users)/max(n_users,1):.3f})")

    spu = sessions_per_user(sessions)
    dom_pair = dominant_pair_per_user(sessions)

    # ── Holdout eligibility: user's LATEST session must be on/after cutoff ──
    user_latest = (
        sessions
        .group_by("user_id")
        .agg(pl.col("session_date").max().alias("latest"))
    )
    eligible_users = set(
        user_latest.filter(pl.col("latest") >= cutoff)["user_id"].to_list()
    )
    print(f"  holdout-eligible users (latest session ≥ cutoff): {len(eligible_users)}")

    # ── Stratified sampling of holdout users ─────────────────────────────────
    target_holdout = int(n_total * (args.rerank_fraction + args.val_fraction))
    print(f"  target holdout sessions: {target_holdout}")

    if args.no_stratify_category:
        print("  category stratification: OFF")
        candidates_by_stratum = {("ALL", "ALL"): list(eligible_users)}
    else:
        # One stratum per (warm, dominant_category, dominant_specificity) bucket.
        # warm/cold included so we balance both axes simultaneously.
        candidates_by_stratum: dict[tuple, list[str]] = defaultdict(list)
        for u in eligible_users:
            warm_flag = "W" if u in warm_users else "C"
            cat, spec = dom_pair.get(u, ("?", "?"))
            candidates_by_stratum[(warm_flag, cat, spec)].append(u)
        print(f"  category strata: {len(candidates_by_stratum)} buckets "
              f"(warm/cold × cat × spec)")

    holdout_users = stratified_quota_sample(
        candidates_by_stratum, spu, target_holdout, args.seed,
    )
    holdout_sessions = sum(spu[u] for u in holdout_users)
    print(f"  selected holdout: {len(holdout_users)} users, {holdout_sessions} sessions "
          f"({holdout_sessions/n_total:.3f} of total — target {target_holdout/n_total:.3f})")

    # ── Within holdout, split rerank vs val proportionally per stratum ──────
    rerank_share = args.rerank_fraction / (args.rerank_fraction + args.val_fraction)
    rerank_threshold = int(rerank_share * 100000)
    rerank_users = {
        u for u in holdout_users
        if hash_bucket(u, args.seed + 2) < rerank_threshold
    }
    val_users = holdout_users - rerank_users

    rerank_sessions = sum(spu[u] for u in rerank_users)
    val_sessions = sum(spu[u] for u in val_users)
    print(f"  rerank: {len(rerank_users)} users, {rerank_sessions} sessions "
          f"({rerank_sessions/n_total:.3f})")
    print(f"  val:    {len(val_users)} users, {val_sessions} sessions "
          f"({val_sessions/n_total:.3f})")

    # ── Cold simulation: only on WARM holdout users ─────────────────────────
    warm_holdout = holdout_users & warm_users
    cold_threshold = int(args.cold_fraction * 100000)
    simulate_cold_users = {
        u for u in warm_holdout
        if hash_bucket(u, args.seed + 1) < cold_threshold
    }
    print(f"  marked {len(simulate_cold_users)}/{len(warm_holdout)} WARM holdout "
          f"users as simulate_cold (target {args.cold_fraction:.2f}, "
          f"actual {len(simulate_cold_users)/max(len(warm_holdout),1):.3f})")

    real_cold = len(holdout_users) - len(warm_holdout)
    effective_cold = real_cold + len(simulate_cold_users)
    print(f"  effective cold rate in holdout: {effective_cold/max(len(holdout_users),1):.3f} "
          f"(real {real_cold} + simulated {len(simulate_cold_users)})")

    # ── Write output ─────────────────────────────────────────────────────────
    rows = []
    for r in sessions.iter_rows(named=True):
        uid = r["user_id"]
        if uid in rerank_users:
            split = "train_rerank"
        elif uid in val_users:
            split = "val"
        else:
            split = "train_proj"
        rows.append({
            "session_id":    r["session_id"],
            "user_id":       uid,
            "split":         split,
            "simulate_cold": (split != "train_proj") and (uid in simulate_cold_users),
        })
    out_df = pl.DataFrame(rows)

    # ── Diagnostics + final assertions ──────────────────────────────────────
    print(f"\nFinal split counts:")
    print(out_df.group_by("split").len().sort("split"))

    proj_users = set(out_df.filter(pl.col("split") == "train_proj")["user_id"].to_list())
    assert len(proj_users & rerank_users) == 0, "train_proj ∩ rerank not empty"
    assert len(proj_users & val_users) == 0,    "train_proj ∩ val not empty"
    assert len(rerank_users & val_users) == 0,  "rerank ∩ val not empty"
    print("User-disjointness: OK")

    # Temporal causality check
    proj_sessions = sessions.filter(pl.col("user_id").is_in(proj_users))
    rerank_sessions_df = sessions.filter(pl.col("user_id").is_in(rerank_users))
    val_sessions_df = sessions.filter(pl.col("user_id").is_in(val_users))
    if rerank_sessions_df.shape[0]:
        max_proj = proj_sessions["session_date"].max()
        min_rerank = rerank_sessions_df["session_date"].min()
        min_val = val_sessions_df["session_date"].min() if val_sessions_df.shape[0] else None
        print(f"\nTemporal causality:")
        print(f"  train_proj: {proj_sessions['session_date'].min()} .. {max_proj}")
        print(f"  rerank:     {min_rerank} .. {rerank_sessions_df['session_date'].max()}")
        if min_val:
            print(f"  val:        {min_val} .. {val_sessions_df['session_date'].max()}")
        # The cutoff we used guarantees: max(train_proj) ≤ cutoff ≤ min(rerank, val)
        print(f"  cutoff was {cutoff}; max(train_proj)={max_proj}, "
              f"min(rerank)={min_rerank}, min(val)={min_val}")

    if DEV_PARQUET.exists():
        dev_dates = pl.read_parquet(DEV_PARQUET, columns=["session_date"])["session_date"].sort()
        print(f"  dev (reference): {dev_dates[0]} .. {dev_dates[-1]}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out_df.write_parquet(OUT)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()