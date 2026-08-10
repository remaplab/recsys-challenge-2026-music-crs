"""splitK — 5-fold cross-validation split (session-stratified, user-coherent).

  Unifies full_dataset (train+test) and submission_dataset (blind-A) into a single
  assembled DataFrame (one row per complete turn), then:

  1. Extracts ~HOLDOUT_TARGET sessions as holdout_test (never touched until final eval).
  2. Splits remaining sessions into 5 folds, each with 3 parts:
       cg_train      ~80%  (4 groups) — CG training
       cg_val        ~16%  (80% of held-out group) — CG HP tuning + OOF reranker data
       reranker_val  ~ 4%  (20% of held-out group) — reranker HP tuning / validation

  All splits are user-coherent: every session of a user lands in the same part.

  Output: data/splitK/
    holdout_test.parquet
    fold_{k}_cg_train.parquet       k = 0..4
    fold_{k}_cg_val.parquet
    fold_{k}_reranker_val.parquet
  Total: 16 files.

  Columns: session_id, user_id, session_date, user_profile, conversation_goal,
           user_query, user_thought, turn_number, track_id, assistant_thought,
           goal_progress_assessments, assistant_response, is_submission
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data" / "talkpl-ai"
TRAIN = DATA / "TalkPlayData-Challenge-Dataset/data/train-00000-of-00001.parquet"
TEST = DATA / "TalkPlayData-Challenge-Dataset/data/test-00000-of-00001.parquet"
BLIND = DATA / "TalkPlayData-Challenge-Blind-A/data/test-00000-of-00001.parquet"
OUT = ROOT / "data" / "splitK"

N_FOLDS = 5
SEED = 42
HOLDOUT_TARGET = 1000
SCHEMA_VERSION = 1

# Persisted single source of truth (committed to git). Drives every run.
# RNG path is reserved for the very first generation OR a deliberate
# regenerate.
ASSIGNMENT_PARQUET = OUT / "splitK_assignment.parquet"
ASSIGNMENT_MANIFEST = OUT / "splitK_assignment_manifest.json"


# ---------------------------------------------------------------------------
# Assembly helpers
# ---------------------------------------------------------------------------

def _assemble(df: pl.DataFrame, is_submission: bool) -> pl.DataFrame:
    """Explode conversations into one row per complete turn (user+music+assistant).

    For submission sessions the last user-only turn has no music/assistant pair
    and is therefore dropped by the inner join (incomplete → not a valid training row).
    """
    queries = (
        df.explode("conversations").unnest("conversations")
        .filter(pl.col("role") == "user").drop("role")
        .rename({"thought": "user_thought", "content": "user_query"})
        .drop("session_date", "goal_progress_assessments", "conversation_goal")
    )
    recs = (
        df.explode("conversations").unnest("conversations")
        .filter(pl.col("role") == "music").drop("role")
        .rename({"thought": "assistant_thought", "content": "track_id"})
        .drop("user_profile")
    )
    asst = (
        df.explode("conversations").unnest("conversations")
        .filter(pl.col("role") == "assistant").drop("role", "thought")
        .rename({"content": "assistant_response"})
        .drop("session_date", "goal_progress_assessments", "conversation_goal", "user_profile")
    )
    return (
        queries
        .join(recs, on=["session_id", "user_id", "turn_number"])
        .join(asst, on=["session_id", "user_id", "turn_number"])
        .with_columns(pl.lit(is_submission).alias("is_submission"))
    )


# ---------------------------------------------------------------------------
# Holdout extraction: deterministic hash-based ordering, greedy accumulation
# ---------------------------------------------------------------------------

def _extract_holdout(
    df: pl.DataFrame, target_sessions: int, seed: int
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return (holdout_df, remaining_df) with ~target_sessions in holdout.

    Users are ordered by sha256(user_id + str(seed)) for determinism, then
    greedily accumulated until their total session count reaches target_sessions.
    Stops at a user boundary — actual holdout may slightly exceed target.
    """
    user_sessions = (
        df.group_by("user_id")
        .agg(pl.col("session_id").n_unique().alias("n_sessions"))
    )
    users = user_sessions["user_id"].to_list()
    counts = dict(zip(user_sessions["user_id"].to_list(), user_sessions["n_sessions"].to_list()))

    salt = str(seed).encode()
    users.sort(key=lambda u: hashlib.sha256(u.encode() + salt).digest())

    holdout_users: list[str] = []
    total = 0
    for u in users:
        if total >= target_sessions:
            break
        holdout_users.append(u)
        total += counts[u]

    holdout_set = set(holdout_users)
    holdout_df = df.filter(pl.col("user_id").is_in(holdout_set))
    remaining_df = df.filter(~pl.col("user_id").is_in(holdout_set))
    print(f"Holdout: {holdout_df['session_id'].n_unique():,} sessions, "
          f"{len(holdout_users):,} users  (target {target_sessions})")
    return holdout_df, remaining_df


# ---------------------------------------------------------------------------
# Fold assignment: greedy bin packing to equalise session counts across groups
# ---------------------------------------------------------------------------

def _assign_groups(df: pl.DataFrame, n_folds: int, seed: int) -> dict[str, int]:
    """Return {user_id: group_index} with balanced session counts per group.

    The `user_id` tie-breaker in the sort is REQUIRED for determinism:
    polars `group_by` emits rows in non-deterministic order across runs;
    sorting only on `n_sessions` leaves ties at the mercy of polars'
    emission order → the rng below would shuffle a non-deterministic
    baseline → different fold assignments across runs of the same SEED.
    """
    user_sessions = (
        df.group_by("user_id")
        .agg(pl.col("session_id").n_unique().alias("n_sessions"))
        .sort(["n_sessions", "user_id"], descending=[True, False])
    )
    users = user_sessions["user_id"].to_numpy()
    counts = user_sessions["n_sessions"].to_numpy()

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(users))
    users, counts = users[perm], counts[perm]
    order = np.argsort(-counts, kind="stable")
    users, counts = users[order], counts[order]

    totals = np.zeros(n_folds, dtype=np.int64)
    assignment: dict[str, int] = {}
    for u, c in zip(users, counts):
        g = int(np.argmin(totals))
        assignment[u] = g
        totals[g] += c

    print(f"Group session counts: {totals}  (total {totals.sum()})")
    return assignment


# ---------------------------------------------------------------------------
# Split one group's users into cg_val (80%) and reranker_val (20%) by sessions
# ---------------------------------------------------------------------------

def _split_group_two(
    group_users: list[str], counts_map: dict[str, int], seed: int
) -> tuple[set[str], set[str]]:
    """Split group_users into two user-coherent sets with ~80/20 session ratio.

    Returns (cg_val_users, reranker_val_users).
    Uses the same greedy bin-packing as _assign_groups with 2 bins.

    `group_users` is sorted lexicographically first so the rng below
    operates on a baseline that does NOT depend on the upstream dict
    insertion order from `_assign_groups`.
    """
    group_users = sorted(group_users)
    users = np.array(group_users)
    counts = np.array([counts_map[u] for u in users])

    rng = np.random.default_rng(seed + 1)
    perm = rng.permutation(len(users))
    users, counts = users[perm], counts[perm]
    order = np.argsort(-counts, kind="stable")
    users, counts = users[order], counts[order]

    totals = np.zeros(2, dtype=np.int64)
    bin0: list[str] = []
    bin1: list[str] = []
    for u, c in zip(users, counts):
        if totals[0] / max(totals.sum(), 1) < 0.8:
            bin0.append(u)
            totals[0] += c
        else:
            bin1.append(u)
            totals[1] += c

    return set(bin0), set(bin1)


# ---------------------------------------------------------------------------
# Assignment parquet — single source of truth (committed to git)
# ---------------------------------------------------------------------------

def _expected_bucket_vocab(n_folds: int) -> set[str]:
    v = {"holdout"}
    for k in range(n_folds):
        v.add(f"fold_{k}_cg_val")
        v.add(f"fold_{k}_reranker_val")
    return v


def _sha256_of_sids(sids: list[str]) -> str:
    h = hashlib.sha256()
    for s in sorted(sids):
        h.update(s.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def _build_assignments_via_rng(
    unified: pl.DataFrame,
) -> dict[str, str]:
    """Apply the deterministic RNG pipeline → {session_id: bucket}.

    Both blind and non-blind rows are kept (historical design choice in
    splitK_bak: blind sessions can land in any non-holdout bucket; only
    the holdout slot intentionally excludes blind).
    """
    holdout_df, remaining = _extract_holdout(unified, HOLDOUT_TARGET, SEED)
    assignment_user_to_group = _assign_groups(remaining, N_FOLDS, SEED)
    groups: list[list[str]] = [[] for _ in range(N_FOLDS)]
    for u, g in assignment_user_to_group.items():
        groups[g].append(u)

    counts_map: dict[str, int] = (
        remaining.group_by("user_id")
        .agg(pl.col("session_id").n_unique().alias("n"))
        .select(["user_id", "n"])
        .to_pandas().set_index("user_id")["n"].to_dict()
    )

    sid_to_bucket: dict[str, str] = {}
    # All holdout sessions, including any blind ones whose user happened to
    # land in holdout (matches splitK_bak design — blind sessions are NOT
    # excluded from splitK).
    for sid in holdout_df["session_id"].unique().to_list():
        sid_to_bucket[sid] = "holdout"

    for k in range(N_FOLDS):
        cg_val_users, reranker_val_users = _split_group_two(
            groups[k], counts_map, SEED + k,
        )
        cg_val_sids = (
            remaining.filter(pl.col("user_id").is_in(cg_val_users))
            ["session_id"].unique().to_list()
        )
        rr_val_sids = (
            remaining.filter(pl.col("user_id").is_in(reranker_val_users))
            ["session_id"].unique().to_list()
        )
        for sid in cg_val_sids:
            sid_to_bucket[sid] = f"fold_{k}_cg_val"
        for sid in rr_val_sids:
            sid_to_bucket[sid] = f"fold_{k}_reranker_val"
    return sid_to_bucket


def _check_determinism(unified: pl.DataFrame) -> None:
    """Run the RNG pipeline twice and assert identical assignments.

    Cheap (a few seconds). Runs only on the regenerate path; fails fast if
    the patches regress.
    """
    print("[check] running determinism check ...")
    a1 = _build_assignments_via_rng(unified)
    a2 = _build_assignments_via_rng(unified)
    if a1 != a2:
        diff = [s for s in a1 if a1[s] != a2.get(s)]
        raise AssertionError(
            f"[check] assignment NOT deterministic: "
            f"{len(diff)} session(s) differ across two runs. "
            f"sample={diff[:5]}"
        )
    print("[check] OK — assignment is deterministic")


def _write_assignment_artifacts(
    sid_to_bucket: dict[str, str], unified: pl.DataFrame,
) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sids = list(sid_to_bucket.keys())
    buckets = [sid_to_bucket[s] for s in sids]
    df = pl.DataFrame({"session_id": sids, "bucket": buckets}).sort("session_id")
    df.write_parquet(ASSIGNMENT_PARQUET)
    print(f"  assignment parquet → {ASSIGNMENT_PARQUET.name}  "
          f"sessions={df.height}")

    counts = (
        df.group_by("bucket").len().sort("bucket")
        .to_pandas().set_index("bucket")["len"].to_dict()
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "n_sessions": int(df.height),
        "n_folds": int(N_FOLDS),
        "holdout_target": int(HOLDOUT_TARGET),
        "seed_used_at_generation": int(SEED),
        "sha256_of_session_ids_sorted": _sha256_of_sids(sids),
        "generated_at": _dt.datetime.utcnow().isoformat() + "Z",
        "bucket_counts": {k: int(v) for k, v in counts.items()},
    }
    ASSIGNMENT_MANIFEST.write_text(json.dumps(manifest, indent=2))
    print(f"  manifest          → {ASSIGNMENT_MANIFEST.name}")


def _load_assignment_artifacts() -> dict[str, str]:
    """Read + verify the on-disk parquet + manifest."""
    df = pl.read_parquet(ASSIGNMENT_PARQUET)
    if set(df.columns) != {"session_id", "bucket"}:
        sys.exit(
            f"[load] {ASSIGNMENT_PARQUET} has unexpected columns: {df.columns}"
        )
    manifest = json.loads(ASSIGNMENT_MANIFEST.read_text())
    if manifest.get("schema_version") != SCHEMA_VERSION:
        sys.exit(
            f"[load] manifest schema_version={manifest.get('schema_version')} "
            f"!= current {SCHEMA_VERSION}. Regenerate or migrate."
        )
    sids = df["session_id"].to_list()
    if len(set(sids)) != len(sids):
        sys.exit("[load] duplicate session_id in assignment parquet")
    if manifest.get("n_sessions") != len(sids):
        sys.exit(
            f"[load] manifest n_sessions={manifest.get('n_sessions')} "
            f"!= parquet rows={len(sids)}. Parquet has been edited."
        )
    sha_now = _sha256_of_sids(sids)
    if manifest.get("sha256_of_session_ids_sorted") != sha_now:
        sys.exit(
            f"[load] sha256 mismatch — assignment parquet edited without "
            f"updating the manifest. Run --regenerate to refresh."
        )
    expected = _expected_bucket_vocab(N_FOLDS)
    actual = set(df["bucket"].unique().to_list())
    extra = actual - expected
    if extra:
        sys.exit(f"[load] unknown bucket label(s) in parquet: {extra}")
    return dict(zip(sids, df["bucket"].to_list()))


def _validate_against_data(
    sid_to_bucket: dict[str, str], unified: pl.DataFrame,
) -> None:
    """Raw data ↔ assignment integrity check.

    splitK by design covers BOTH non-blind and blind sessions (blind ones
    can be assigned to any fold's cg_val / reranker_val bucket; only the
    `holdout` bucket excludes blind). The check therefore just verifies
    that the parquet and the assembled session set are the same — no
    is_submission constraint.
    """
    assembled = set(unified["session_id"].unique().to_list())
    blind = set(
        unified.filter(pl.col("is_submission"))["session_id"]
        .unique().to_list()
    )
    parquet_sids = set(sid_to_bucket.keys())

    missing_in_parquet = assembled - parquet_sids
    extra_in_parquet = parquet_sids - assembled

    if missing_in_parquet:
        sys.exit(
            f"[validate] {len(missing_in_parquet)} session(s) in raw data "
            f"are NOT in {ASSIGNMENT_PARQUET.name}. Raw data has grown or "
            f"drifted — run --regenerate. Examples: "
            f"{sorted(missing_in_parquet)[:5]}"
        )
    if extra_in_parquet:
        sys.exit(
            f"[validate] {len(extra_in_parquet)} parquet session(s) are NOT "
            f"in raw data. Stale assignment. Examples: "
            f"{sorted(extra_in_parquet)[:5]}"
        )

    holdout_sids = {s for s, b in sid_to_bucket.items() if b == "holdout"}
    holdout_blind = holdout_sids & blind
    print(
        f"[validate] OK  total={len(assembled)}  blind={len(blind)}  "
        f"parquet={len(parquet_sids)}  holdout={len(holdout_sids)} "
        f"(of which {len(holdout_blind)} blind — by design)"
    )


def _write_per_fold_parquets(
    unified: pl.DataFrame, sid_to_bucket: dict[str, str],
) -> None:
    """Materialise the 16 per-fold parquet files from the assignment map."""
    OUT.mkdir(parents=True, exist_ok=True)
    sid_df = pl.DataFrame({
        "session_id": list(sid_to_bucket.keys()),
        "bucket":     list(sid_to_bucket.values()),
    })
    tagged = unified.join(sid_df, on="session_id", how="inner")

    # Holdout.
    ho = tagged.filter(pl.col("bucket") == "holdout").drop("bucket")
    path = OUT / "holdout_test.parquet"
    ho.write_parquet(path)
    print(f"  holdout_test : {ho.shape[0]:>8,} rows, "
          f"{ho['session_id'].n_unique():>5,} sessions → {path.name}")

    # Per fold.
    for k in range(N_FOLDS):
        cg_val_label = f"fold_{k}_cg_val"
        rr_val_label = f"fold_{k}_reranker_val"
        # cg_train = sessions whose bucket is fold_j_(cg_val|reranker_val) for j != k.
        train_labels = [
            f"fold_{j}_{s}"
            for j in range(N_FOLDS) if j != k
            for s in ("cg_val", "reranker_val")
        ]
        splits = {
            "cg_train":     tagged.filter(pl.col("bucket").is_in(train_labels)),
            "cg_val":       tagged.filter(pl.col("bucket") == cg_val_label),
            "reranker_val": tagged.filter(pl.col("bucket") == rr_val_label),
        }
        for name, df in splits.items():
            df = df.drop("bucket")
            path = OUT / f"fold_{k}_{name}.parquet"
            df.write_parquet(path)
            n_sess = df["session_id"].n_unique()
            print(
                f"  fold {k} {name:14s}: {df.shape[0]:>8,} rows, "
                f"{n_sess:>5,} sessions → {path.name}"
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--regenerate", action="store_true",
        help="Force RNG-based regeneration of the assignment parquet "
             "(overwrites existing). Use only on deliberate splitK refresh.",
    )
    p.add_argument(
        "--validate", action="store_true",
        help="Read the existing assignment parquet + manifest, verify "
             "integrity vs raw data, do NOT write any per-fold parquets.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    print("Loading raw data…")
    full_dataset = pl.concat([pl.read_parquet(TRAIN), pl.read_parquet(TEST)])
    submission = pl.read_parquet(BLIND)

    print("Assembling unified df…")
    unified = pl.concat([
        _assemble(full_dataset, is_submission=False),
        _assemble(submission, is_submission=True),
    ])
    print(f"Unified: {unified.shape[0]:,} rows, "
          f"{unified['session_id'].n_unique():,} sessions, "
          f"{unified['user_id'].n_unique():,} users")

    have_parquet = ASSIGNMENT_PARQUET.exists() and ASSIGNMENT_MANIFEST.exists()

    if args.validate:
        if not have_parquet:
            sys.exit(
                f"[validate] missing {ASSIGNMENT_PARQUET} or "
                f"{ASSIGNMENT_MANIFEST}. Nothing to validate."
            )
        sid_to_bucket = _load_assignment_artifacts()
        _validate_against_data(sid_to_bucket, unified)
        print("[validate] DONE — no per-fold parquets written")
        return

    # ── Decide source of truth ─────────────────────────────────────────────
    if args.regenerate or not have_parquet:
        if args.regenerate:
            print("[regenerate] forced via CLI — running RNG pipeline")
        else:
            print(
                f"[regenerate] no {ASSIGNMENT_PARQUET.name} on disk — "
                f"running RNG pipeline"
            )
        _check_determinism(unified)
        sid_to_bucket = _build_assignments_via_rng(unified)
        _validate_against_data(sid_to_bucket, unified)
        _write_assignment_artifacts(sid_to_bucket, unified)
    else:
        print(f"[load] using {ASSIGNMENT_PARQUET.name} as source of truth")
        sid_to_bucket = _load_assignment_artifacts()
        _validate_against_data(sid_to_bucket, unified)

    # ── Materialise the 16 per-fold parquets ───────────────────────────────
    _write_per_fold_parquets(unified, sid_to_bucket)


if __name__ == "__main__":
    main()
