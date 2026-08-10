"""3-way split (train / val / test) — v4.


OUTPUT
──────
models/splits/train_val_test.parquet:
    session_id           — primary key
    user_id
    split                — 'train' | 'val' | 'test' | 'discarded'
    n_turns              — train: -1 (use all turns)
                           val/test: the truncated evaluation length
                           discarded: -1 (not used)
    predict_turn_number  — train: -1 (every turn trains)
                           val/test: the turn_number to predict
                           (= n_turns; turns 1..n_turns-1 are history)
                           discarded: -1
    simulate_cold        — bool. True for warm holdout users we tag as
                           cold to match BlindA's cold rate.
    category, specificity, last_session_date  — diagnostics

Run:
    uv run python scripts/01_make_split_v4.py
    uv run python scripts/01_make_split_v4.py --train-frac 0.70 --val-frac 0.15
    uv run python scripts/01_make_split_v4.py --no-stratify   # sanity baseline
"""
from __future__ import annotations

import argparse
import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl


# ── PATHS ─────────────────────────────────────────────────────────────────
DATA            = Path("./data/talkpl-ai")
TRAIN_PARQUET   = DATA / "TalkPlayData-Challenge-Dataset/data/train-00000-of-00001.parquet"
TEST_PARQUET    = DATA / "TalkPlayData-Challenge-Dataset/data/test-00000-of-00001.parquet"
BLIND_PARQUET   = DATA / "TalkPlayData-Challenge-Blind-A/data/test-00000-of-00001.parquet"
USER_EMB_TRAIN  = DATA / "TalkPlayData-Challenge-User-Embeddings/data/train-00000-of-00001.parquet"
USER_EMB_WARM   = DATA / "TalkPlayData-Challenge-User-Embeddings/data/test_warm-00000-of-00001.parquet"
USER_EMB_COLD   = DATA / "TalkPlayData-Challenge-User-Embeddings/data/test_cold-00000-of-00001.parquet"

OUT_PATH        = Path("./models/splits/train_val_test.parquet")


# ── HELPERS ───────────────────────────────────────────────────────────────
def hash_bucket(key: str, seed: int, mod: int = 100_000) -> int:
    """Deterministic uniform hash of a string into [0, mod)."""
    h = hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()
    return int(h[:8], 16) % mod


def hash_unit(key: str, seed: int) -> float:
    """Deterministic uniform float in [0, 1)."""
    return hash_bucket(key, seed, mod=10**9) / 10**9


@dataclass
class SessionRow:
    session_id: str
    user_id: str
    n_turns_source: int       # n music turns in the source data
    last_turn_number: int     # turn_number of the last music turn (≤ n_turns_source's max)
    category: str
    specificity: str
    session_date: str         # ISO date string


# ── INGESTION ─────────────────────────────────────────────────────────────
def load_warm_user_set() -> set[str]:
    """Users with a non-empty cf-bpr in ANY user_emb file."""
    warm = set()
    for path, label in [
        (USER_EMB_TRAIN, "train"),
        (USER_EMB_WARM,  "warm"),
        (USER_EMB_COLD,  "cold"),
    ]:
        if not path.exists():
            print(f"  [warn] {path} missing — skipping {label} cf source")
            continue
        df = pl.read_parquet(path)
        for r in df.iter_rows(named=True):
            v = r.get("cf-bpr")
            if v is not None and len(v) > 0:
                warm.add(r["user_id"])
    return warm


def parse_sessions(parquet_path: Path, *, source: str) -> list[SessionRow]:
    """Parse one organizer parquet into per-session records."""
    print(f"\nLoading {source}: {parquet_path}")
    df = pl.read_parquet(parquet_path)

    rows: list[SessionRow] = []
    for r in df.iter_rows(named=True):
        convs = r.get("conversations") or []
        sid   = r["session_id"]
        uid   = r["user_id"]
        cg    = r.get("conversation_goal") or {}
        cat   = cg.get("category") or "?"
        spec  = cg.get("specificity") or "?"
        sdate = str(r.get("session_date") or "")

        by_turn: dict[int, list[dict]] = defaultdict(list)
        for t in convs:
            by_turn[t["turn_number"]].append(t)

        # Music turns = prediction targets. For BlindA (no GT) fall back to user turns.
        music_tns = sorted(
            tn for tn, ts in by_turn.items()
            if any(t.get("role") == "music" for t in ts)
        )
        if music_tns:
            n_turns = len(music_tns)
            last_tn = music_tns[-1]
        else:
            user_tns = sorted(
                tn for tn, ts in by_turn.items()
                if any(t.get("role") == "user" for t in ts)
            )
            n_turns = len(user_tns)
            last_tn = user_tns[-1] if user_tns else 0

        if n_turns == 0:
            continue

        rows.append(SessionRow(
            session_id=sid, user_id=uid,
            n_turns_source=n_turns, last_turn_number=last_tn,
            category=cat, specificity=spec, session_date=sdate,
        ))

    print(f"  {len(rows)} sessions parsed from {source}")

    # Length-distribution sanity print
    length_hist = Counter(r.n_turns_source for r in rows)
    print(f"  length distribution: "
          f"{dict(sorted(length_hist.items()))}")
    return rows


# ── BLIND-A TARGET DISTRIBUTION ──────────────────────────────────────────
@dataclass
class BlindATargets:
    p_cat:     dict[str, float]
    p_spec:    dict[str, float]
    p_length:  dict[int, float]
    p_joint:   dict[tuple[str, str, int], float]   # (cat, spec, length) → prob
    cold_rate: float

    def __repr__(self) -> str:
        return (
            f"BlindATargets(\n"
            f"  P(cat)        = {self.p_cat}\n"
            f"  P(spec)       = {self.p_spec}\n"
            f"  P(length)     = {self.p_length}\n"
            f"  P(joint) has  = {len(self.p_joint)} non-empty cells\n"
            f"  cold_rate     = {self.cold_rate:.4f}\n"
            f")"
        )


def compute_blind_targets(blind_rows: list[SessionRow], warm_users: set[str]) -> BlindATargets:
    cat_c    = Counter(r.category for r in blind_rows)
    spec_c   = Counter(r.specificity for r in blind_rows)
    length_c = Counter(r.n_turns_source for r in blind_rows)
    joint_c  = Counter((r.category, r.specificity, r.n_turns_source) for r in blind_rows)
    n        = len(blind_rows)
    cold_n   = sum(1 for r in blind_rows if r.user_id not in warm_users)

    return BlindATargets(
        p_cat={k: v / n for k, v in cat_c.items()},
        p_spec={k: v / n for k, v in spec_c.items()},
        p_length={k: v / n for k, v in length_c.items()},
        p_joint={k: v / n for k, v in joint_c.items()},
        cold_rate=cold_n / max(n, 1),
    )


# ── LENGTH-TRUNCATION (the mechanism that makes length-matching possible) ─
def sample_target_lengths(
    n_sessions: int,
    p_length: dict[int, float],
    seed: int,
    max_source_length: int,
) -> np.ndarray:
    """Sample n_sessions target lengths from BlindA's P(length).

    Every sampled length is clamped to ≤ max_source_length (= 8 in our
    pool). Returns int array of shape (n_sessions,).

    Determinism: uses np.random.default_rng(seed) so the same seed →
    same length assignment → reproducible.
    """
    rng = np.random.default_rng(seed)
    lengths = np.array(sorted(p_length.keys()), dtype=np.int64)
    probs   = np.array([p_length[L] for L in lengths], dtype=np.float64)
    probs   = probs / probs.sum()

    # Clamp at max_source_length: any length > pool max is impossible to satisfy
    # with our 8-turn data, so its probability mass is redistributed proportionally
    # over the achievable lengths.
    achievable = lengths <= max_source_length
    if not achievable.all():
        dropped = lengths[~achievable]
        print(f"  [length-sampling] BlindA has lengths {dropped.tolist()} > "
              f"pool max {max_source_length}; their {probs[~achievable].sum():.4f} "
              f"probability mass is redistributed over achievable lengths")
        lengths = lengths[achievable]
        probs   = probs[achievable]
        probs   = probs / probs.sum()

    samples = rng.choice(lengths, size=n_sessions, p=probs)
    return samples.astype(np.int64)


# ── IPF on (cat, spec) only — length is handled by truncation ──────────
def ipf_session_weights_2d(
    sessions: list[SessionRow],
    p_cat: dict[str, float],
    p_spec: dict[str, float],
    max_iter: int = 200,
    tol: float = 1e-5,
) -> np.ndarray:
    """Weight sessions so that weighted (cat) and (spec) marginals match.

    Length is NOT IPF'd here because the pool has degenerate length
    distribution (all 8s). Length matching happens via truncation below.
    """
    N = len(sessions)
    if N == 0:
        return np.array([], dtype=np.float64)
    w = np.ones(N, dtype=np.float64)

    by_cat:  dict[str, list[int]] = defaultdict(list)
    by_spec: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(sessions):
        by_cat[r.category].append(i)
        by_spec[r.specificity].append(i)

    def _rescale(buckets: dict, p_target: dict) -> float:
        total_w = w.sum()
        max_drift = 0.0
        for key, idx_list in buckets.items():
            current = w[idx_list].sum()
            target  = p_target.get(key, 0.0) * total_w
            if current <= 0 or target <= 0:
                continue
            ratio = target / current
            max_drift = max(max_drift, abs(ratio - 1.0))
            for i in idx_list:
                w[i] *= ratio
        return max_drift

    for it in range(max_iter):
        d1 = _rescale(by_cat,  p_cat)
        d2 = _rescale(by_spec, p_spec)
        if max(d1, d2) < tol:
            print(f"  IPF converged at iter {it+1}  drift={max(d1, d2):.2e}")
            break
    else:
        print(f"  IPF hit {max_iter} iters  drift={max(d1, d2):.2e}")

    w *= N / w.sum()
    return w


# ── HOLDOUT SELECTION ────────────────────────────────────────────────────
def select_holdout_users_by_temporal_position(
    sessions: list[SessionRow],
    eligible_users: set[str],
    train_frac: float,
    val_frac: float,
    test_frac: float,
) -> tuple[set[str], set[str], set[str]]:
    """Sort eligible users by last-session-date; partition into train/val/test."""
    last_date: dict[str, str] = {}
    for r in sessions:
        if r.user_id not in eligible_users:
            continue
        if r.user_id not in last_date or r.session_date > last_date[r.user_id]:
            last_date[r.user_id] = r.session_date

    ordered = sorted(
        last_date.keys(),
        key=lambda u: (last_date[u], hash_bucket(u, 0, mod=10**9)),
    )
    n = len(ordered)
    n_train = int(n * train_frac)
    n_val   = int(n * val_frac)
    train_users = set(ordered[:n_train])
    val_users   = set(ordered[n_train : n_train + n_val])
    test_users  = set(ordered[n_train + n_val :])

    print(f"  Temporal partition by user-last-date:")
    if n_train:
        print(f"    train: {len(train_users)} users  "
              f"[{last_date[ordered[0]]} .. {last_date[ordered[n_train-1]]}]")
    if n_val:
        print(f"    val:   {len(val_users)} users  "
              f"[{last_date[ordered[n_train]]} .. {last_date[ordered[n_train+n_val-1]]}]")
    if test_users:
        print(f"    test:  {len(test_users)} users  "
              f"[{last_date[ordered[n_train+n_val]]} .. {last_date[ordered[-1]]}]")

    return train_users, val_users, test_users


def reweighted_user_subsample(
    sessions_in_bucket: list[SessionRow],
    target_n_sessions: int,
    p_cat: dict[str, float],
    p_spec: dict[str, float],
    seed: int,
) -> set[str]:
    """Pick users from a temporal bucket so that aggregate (cat, spec)
    marginals match BlindA targets.

    Greedy by IPF-derived user score, with hash tiebreaker. Drops
    low-scoring users — those users go to 'discarded', NOT back to train,
    so temporal causality is preserved.
    """
    if not sessions_in_bucket:
        return set()
    weights = ipf_session_weights_2d(sessions_in_bucket, p_cat, p_spec)

    user_score: dict[str, float] = defaultdict(float)
    user_count: dict[str, int]   = defaultdict(int)
    for i, r in enumerate(sessions_in_bucket):
        user_score[r.user_id] += weights[i]
        user_count[r.user_id] += 1

    ordered = sorted(
        user_score.keys(),
        key=lambda u: (-user_score[u], hash_bucket(u, seed + 11, mod=10**9)),
    )
    chosen: set[str] = set()
    n_taken = 0
    for u in ordered:
        if n_taken >= target_n_sessions:
            break
        chosen.add(u)
        n_taken += user_count[u]
    return chosen


def assign_simulate_cold(
    holdout_users: set[str],
    warm_users: set[str],
    target_cold_rate: float,
    seed: int,
    label: str,
) -> set[str]:
    """Tag warm holdout users as simulate_cold to hit BlindA's cold rate."""
    real_cold = holdout_users - warm_users
    real_rate = len(real_cold) / max(len(holdout_users), 1)

    if real_rate >= target_cold_rate:
        print(f"  [{label}] real-cold {real_rate:.4f} ≥ target "
              f"{target_cold_rate:.4f} — no simulation")
        return set()

    deficit = target_cold_rate - real_rate
    n_simulate = int(round(deficit * len(holdout_users)))
    warm_in = sorted(holdout_users & warm_users,
                     key=lambda u: hash_bucket(u, seed + 23, mod=10**9))
    simulated = set(warm_in[:n_simulate])
    eff = (len(real_cold) + len(simulated)) / len(holdout_users)
    print(f"  [{label}] real-cold {real_rate:.4f}, target {target_cold_rate:.4f} "
          f"→ simulating {len(simulated)} → effective {eff:.4f}")
    return simulated


# ── DIAGNOSTICS ──────────────────────────────────────────────────────────
def report_marginals(
    label: str,
    rows: list[SessionRow],
    truncated_lengths: dict[str, int],   # session_id → eval length (BlindA-sampled)
    targets: BlindATargets,
    simulate_cold_users: set[str],
    warm_users: set[str],
) -> None:
    if not rows:
        print(f"  [{label}] empty"); return
    n = len(rows)

    cat_c  = Counter(r.category for r in rows)
    spec_c = Counter(r.specificity for r in rows)
    # For length, use truncated length (what eval will actually see), not source length
    if truncated_lengths:
        len_c = Counter(truncated_lengths[r.session_id] for r in rows)
    else:
        len_c = Counter(r.n_turns_source for r in rows)
    joint_c = Counter(
        (r.category, r.specificity,
         truncated_lengths.get(r.session_id, r.n_turns_source))
        for r in rows
    )

    real_cold = sum(1 for r in rows if r.user_id not in warm_users)
    sim_cold  = sum(1 for r in rows if r.user_id in simulate_cold_users)
    eff_cold  = (real_cold + sim_cold) / n

    def _drift(observed: Counter, target: dict) -> float:
        keys = set(observed) | set(target)
        return max(abs(observed.get(k, 0) / n - target.get(k, 0.0)) for k in keys)

    def _l1_total(observed: Counter, target: dict) -> float:
        keys = set(observed) | set(target)
        return sum(abs(observed.get(k, 0) / n - target.get(k, 0.0)) for k in keys)

    print(f"  [{label}] n_sessions={n}  cold_eff={eff_cold:.4f} "
          f"(target {targets.cold_rate:.4f})")
    print(f"    category    L∞={_drift(cat_c,  targets.p_cat):.4f}  "
          f"L1={_l1_total(cat_c,  targets.p_cat):.4f}")
    print(f"    specificity L∞={_drift(spec_c, targets.p_spec):.4f}  "
          f"L1={_l1_total(spec_c, targets.p_spec):.4f}")
    print(f"    length      L∞={_drift(len_c,  targets.p_length):.4f}  "
          f"L1={_l1_total(len_c,  targets.p_length):.4f}")
    print(f"    joint(cat,spec,length)  L∞={_drift(joint_c, targets.p_joint):.4f}  "
          f"L1={_l1_total(joint_c, targets.p_joint):.4f}")


# ── MAIN ─────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-frac", type=float, default=0.70)
    p.add_argument("--val-frac",   type=float, default=0.15)
    p.add_argument("--test-frac",  type=float, default=0.15)
    p.add_argument("--no-stratify", action="store_true",
                   help="Skip IPF; raw temporal partition only (sanity baseline)")
    p.add_argument("--keep-frac", type=float, default=0.95,
                   help="Fraction of each holdout bucket to KEEP after IPF "
                        "stratification. The rest are discarded (NOT moved to "
                        "train, to preserve temporal causality). Lower = better "
                        "marginal fit, less data; higher = more data, looser fit.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=OUT_PATH)
    args = p.parse_args()

    if abs(args.train_frac + args.val_frac + args.test_frac - 1.0) > 1e-6:
        raise ValueError(
            f"fractions must sum to 1.0, got "
            f"{args.train_frac + args.val_frac + args.test_frac}")

    # ── 1. Ingest ───────────────────────────────────────────────────────
    train_rows = parse_sessions(TRAIN_PARQUET, source="organizer-train")
    test_rows  = parse_sessions(TEST_PARQUET,  source="organizer-test")
    blind_rows = parse_sessions(BLIND_PARQUET, source="BlindA")
    pool       = train_rows + test_rows
    print(f"\nMerged pool: {len(pool)} sessions "
          f"({len(train_rows)} + {len(test_rows)})")

    pool_users = {r.user_id for r in pool}
    blind_users = {r.user_id for r in blind_rows}
    overlap = pool_users & blind_users
    if overlap:
        print(f"  [warn] {len(overlap)} pool users overlap BlindA — excluding from val/test")

    # ── 2. BlindA targets ───────────────────────────────────────────────
    warm_users = load_warm_user_set()
    print(f"\nWarm users (have cf-bpr): {len(warm_users)}")
    targets = compute_blind_targets(blind_rows, warm_users)
    print(targets)

    pool_max_length = max(r.n_turns_source for r in pool)
    print(f"\nPool max session length: {pool_max_length}")
    if pool_max_length < max(targets.p_length):
        print(f"  [info] BlindA has lengths up to {max(targets.p_length)}; "
              f"these are unrepresentable in our pool (will be redistributed "
              f"during length sampling).")

    # ── 3. Temporal partition of pool users (excl. BlindA-overlap) ──────
    print(f"\nPartitioning users by last-session-date...")
    eligible_users = pool_users - overlap
    train_users, val_users, test_users = select_holdout_users_by_temporal_position(
        pool, eligible_users, args.train_frac, args.val_frac, args.test_frac,
    )

    # ── 4. Stratified subselection (cat × spec marginals) ──────────────
    discarded_users: set[str] = set()
    if not args.no_stratify:
        print("\nIPF stratification within val and test (cat × spec only — "
              "length is handled by truncation below)...")
        val_sessions  = [r for r in pool if r.user_id in val_users]
        test_sessions = [r for r in pool if r.user_id in test_users]

        target_val_n  = int(args.keep_frac * len(val_sessions))
        target_test_n = int(args.keep_frac * len(test_sessions))

        kept_val  = reweighted_user_subsample(
            val_sessions,  target_val_n,
            targets.p_cat, targets.p_spec, args.seed,
        )
        kept_test = reweighted_user_subsample(
            test_sessions, target_test_n,
            targets.p_cat, targets.p_spec, args.seed + 1,
        )

        # Bug-fix vs v3: dropped users go to 'discarded', NOT to train.
        # This preserves the temporal property max(train.last) ≤ min(val.last).
        discarded_users = (val_users - kept_val) | (test_users - kept_test)
        val_users  = kept_val
        test_users = kept_test
        print(f"  After stratification: train={len(train_users)} "
              f"val={len(val_users)} test={len(test_users)} "
              f"discarded={len(discarded_users)}")
    else:
        print("\n[--no-stratify] using raw temporal partition")

    assert not (train_users & val_users)
    assert not (train_users & test_users)
    assert not (val_users  & test_users)
    assert not (train_users & discarded_users)

    # ── 5. Cold-rate calibration ───────────────────────────────────────
    print("\nCalibrating cold rate to match BlindA...")
    val_simulate  = assign_simulate_cold(val_users,  warm_users,
                                          targets.cold_rate, args.seed + 7,  "val")
    test_simulate = assign_simulate_cold(test_users, warm_users,
                                          targets.cold_rate, args.seed + 13, "test")
    simulate_all = val_simulate | test_simulate

    # ── 6. Length truncation: sample target length per holdout session ─
    # This is the v4 mechanism that finally makes length-matching work.
    print("\nSampling truncated evaluation lengths from BlindA P(length)...")
    val_session_rows  = sorted(
        [r for r in pool if r.user_id in val_users],
        key=lambda r: r.session_id,
    )
    test_session_rows = sorted(
        [r for r in pool if r.user_id in test_users],
        key=lambda r: r.session_id,
    )

    val_truncated_lens  = sample_target_lengths(
        len(val_session_rows),  targets.p_length,
        args.seed + 31, max_source_length=pool_max_length,
    )
    test_truncated_lens = sample_target_lengths(
        len(test_session_rows), targets.p_length,
        args.seed + 41, max_source_length=pool_max_length,
    )

    truncated_by_sid: dict[str, int] = {}
    for r, L in zip(val_session_rows,  val_truncated_lens):
        truncated_by_sid[r.session_id] = int(L)
    for r, L in zip(test_session_rows, test_truncated_lens):
        truncated_by_sid[r.session_id] = int(L)

    # ── 7. Build output dataframe ──────────────────────────────────────
    print("\nBuilding output dataframe...")
    out_rows = []
    for r in pool:
        if r.user_id in train_users:
            split, n_turns_out, predict_tn = "train", -1, -1
        elif r.user_id in val_users:
            L = truncated_by_sid.get(r.session_id, r.last_turn_number)
            split, n_turns_out, predict_tn = "val", L, L
        elif r.user_id in test_users:
            L = truncated_by_sid.get(r.session_id, r.last_turn_number)
            split, n_turns_out, predict_tn = "test", L, L
        elif r.user_id in discarded_users:
            split, n_turns_out, predict_tn = "discarded", -1, -1
        else:
            # User in the pool overlap with BlindA — exclude entirely
            split, n_turns_out, predict_tn = "discarded", -1, -1
        out_rows.append({
            "session_id":          r.session_id,
            "user_id":             r.user_id,
            "split":               split,
            "n_turns":             n_turns_out,
            "predict_turn_number": predict_tn,
            "simulate_cold":       r.user_id in simulate_all,
            "category":            r.category,
            "specificity":         r.specificity,
            "last_session_date":   r.session_date,
        })

    out_df = pl.DataFrame(out_rows)
    print(f"\nFinal split counts (sessions):")
    print(out_df.group_by("split").len().sort("split"))

    # ── 8. Marginal fit ────────────────────────────────────────────────
    print("\nMarginal-fit report (lower drift = better):")
    train_session_rows = [r for r in pool if r.user_id in train_users]
    report_marginals("train", train_session_rows, {},                 targets, simulate_all, warm_users)
    report_marginals("val",   val_session_rows,   truncated_by_sid,   targets, simulate_all, warm_users)
    report_marginals("test",  test_session_rows,  truncated_by_sid,   targets, simulate_all, warm_users)

    # ── 9. Temporal causality check ────────────────────────────────────
    print("\nTemporal causality check (last-session-date per bucket):")

    def _user_last_date(rows: list[SessionRow]) -> dict[str, str]:
        out: dict[str, str] = {}
        for r in rows:
            if r.user_id not in out or r.session_date > out[r.user_id]:
                out[r.user_id] = r.session_date
        return out

    train_last = _user_last_date(train_session_rows)
    val_last   = _user_last_date(val_session_rows)
    test_last  = _user_last_date(test_session_rows)
    max_train_last = max(train_last.values()) if train_last else "-"
    min_val_last   = min(val_last.values())   if val_last   else "-"
    max_val_last   = max(val_last.values())   if val_last   else "-"
    min_test_last  = min(test_last.values())  if test_last  else "-"

    print(f"  train  (max last_session_date): {max_train_last}")
    print(f"  val    (min .. max):             {min_val_last} .. {max_val_last}")
    print(f"  test   (min last_session_date):  {min_test_last}")

    if train_last and val_last:
        ok = max_train_last <= min_val_last
        print(f"  train ≤ val   : {'✓' if ok else '✗ LEAK'}")
    if val_last and test_last:
        ok = max_val_last <= min_test_last
        print(f"  val   ≤ test  : {'✓' if ok else '✗ LEAK'}")

    # ── 10. Predict-turn distribution ──────────────────────────────────
    print("\nTruncated evaluation length distribution (val + test):")
    holdout_df = (
        out_df.filter(pl.col("split").is_in(["val", "test"]))
              .group_by("split", "n_turns").len()
              .sort("split", "n_turns")
    )
    print(holdout_df)

    # ── 11. Write ──────────────────────────────────────────────────────
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.write_parquet(args.out)
    print(f"\nWrote {args.out}  ({out_df.shape[0]} rows)")


if __name__ == "__main__":
    main()