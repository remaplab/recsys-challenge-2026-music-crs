"""Rebuild train/val/test so that VAL and TEST match Blind A on the covariates
that drive the macro-by-turn NDCG@20 metric — so an offline delta predicts an
online delta.  (Drop-in launcher for the embedding-lab pipeline.)

This is the script you provided, unchanged in logic; only the argparse defaults
now point at the standard embedding-lab data layout so it can be run with no
flags:

    uv run python scripts/01b_rebuild_split.py                 # write the split
    uv run python scripts/01b_rebuild_split.py --report-only   # inspect only
    uv run python scripts/01b_rebuild_split.py --test-size 0   # skip the test fold

Output: models/splits/train_val_test_blinda_matched.parquet with columns
  session_id, split ("train"|"val"|"test"), predict_turn_number

WHY / WHAT IT MATCHES / LEAK SAFETY: see the long note in your original file.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import polars as pl

PKG  = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "data/talkpl-ai"                       # was Path("./data/talkpl-ai")
OUT  = PKG / "models/splits/train_val.parquet"       # was Path("./models/splits/...")
#DATA = Path("./data/talkpl-ai")
DEFAULT_TRACK_META = DATA / "TalkPlayData-Challenge-Track-Metadata/data/all_tracks-00000-of-00001.parquet"
DEFAULT_TRAIN_CONV = DATA / "TalkPlayData-Challenge-Dataset/data/train-00000-of-00001.parquet"
DEFAULT_TEST_CONV = DATA / "TalkPlayData-Challenge-Dataset/data/test-00000-of-00001.parquet"
DEFAULT_BLIND_CONV = DATA / "TalkPlayData-Challenge-Blind-A/data/test-00000-of-00001.parquet"
DEFAULT_SPLIT_PARQUET = Path("./models/splits/train_val.parquet")
DEFAULT_OUT_SPLIT = Path("./models/splits/train_val_test_blinda_matched.parquet")


# ─── covariate buckets (identical to the analyzer) ───────────────────────────
def cat_size_bucket(n):
    if n == 0:
        return "0"
    if n <= 5:
        return "1-5"
    if n <= 20:
        return "6-20"
    if n <= 100:
        return "21-100"
    return "100+"


def us_bucket(n):
    if n <= 0:
        return "0"
    if n == 1:
        return "1"
    if n <= 3:
        return "2-3"
    if n <= 7:
        return "4-7"
    return "8+"


CAT_ORDER = ["0", "1-5", "6-20", "21-100", "100+"]
US_ORDER = ["0", "1", "2-3", "4-7", "8+"]
MONO_ORDER = ["turn1", "mono", "multi"]
DEC_ORDER = ["turn1", "decyes", "decno"]


# ─── minimal loaders (decoupled from the heuristic) ──────────────────────────
def _parse_year(rd):
    if rd is None:
        return 0
    s = str(rd)
    try:
        if len(s) >= 4:
            y = int(s[:4])
            if 1900 <= y <= 2035:
                return y
    except (TypeError, ValueError):
        pass
    return 0


def load_track_index(path):
    df = pl.read_parquet(path)
    has_tid = "track_id" in df.columns
    tracks = set()
    id_to_artist = {}
    artist_n = Counter()
    id_to_year = {}

    def first(v):
        if isinstance(v, (list, tuple)):
            return v[0] if v else None
        return v

    for r in df.iter_rows(named=True):
        tid = r.get("track_id") if has_tid else r.get("id")
        if tid is None:
            continue
        tid = str(tid)
        tracks.add(tid)
        a = first(r.get("artist_id"))
        if a is not None:
            id_to_artist[tid] = str(a)
            artist_n[str(a)] += 1
        id_to_year[tid] = _parse_year(r.get("release_date") if "release_date" in df.columns else r.get("year"))
    yr_known = sum(1 for y in id_to_year.values() if y > 0)
    print(f"[meta] tracks={len(tracks)}  with_artist={len(id_to_artist)}  "
          f"artists={len(artist_n)}  with_year={yr_known} ({yr_known/max(len(tracks),1)*100:.1f}%)")
    return {"tracks": tracks, "artist": id_to_artist, "an": artist_n, "year": id_to_year}


def parse_pool(path):
    out = {}
    for s in pl.read_parquet(path).to_dicts():
        convs = s.get("conversations") or []
        music = sorted(
            [(int(t["turn_number"]), str(t["content"]).strip())
             for t in convs
             if t.get("role") == "music" and t.get("content") and t.get("turn_number") is not None],
            key=lambda x: x[0])
        out[str(s["session_id"])] = {"user_id": str(s.get("user_id")), "music": music}
    return out


def covariates(meta, music, K, n_other):
    resolved = [tid for tn, tid in music if tn < K and tid in meta["tracks"]]
    if K == 1 or not resolved:                                  # v2 turn-1 path
        return {"K": K, "cat": "0", "mono": "turn1", "us": us_bucket(n_other), "dec": "turn1"}
    arts = {meta["artist"][t] for t in resolved if t in meta["artist"]}
    catalog = sum(meta["an"].get(a, 0) for a in arts)
    mono = "mono" if len(arts) <= 1 else "multi"
    dec = "decyes" if meta["year"].get(resolved[-1], 0) > 0 else "decno"
    return {"K": K, "cat": cat_size_bucket(catalog), "mono": mono,
            "us": us_bucket(n_other), "dec": dec}


# ─── reporting ───────────────────────────────────────────────────────────────
def marginal(covs, key, order=None):
    c = Counter(cv[key] for cv in covs)
    tot = max(sum(c.values()), 1)
    keys = order or sorted(c, key=str)
    return {k: c.get(k, 0) / tot for k in keys}


def tv(p, q):
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0) - q.get(k, 0)) for k in keys)


def print_marginal(name, off, blind, order):
    print(f"\n  {name}  (TV distance offline vs BlindA = {tv(off, blind):.3f})")
    print(f"    {'bucket':<10}{'offline':>10}{'BlindA':>10}{'gap':>9}")
    for k in order:
        o, b = off.get(k, 0.0), blind.get(k, 0.0)
        print(f"    {str(k):<10}{o*100:>9.1f}%{b*100:>9.1f}%{(o-b)*100:>+8.1f}%")


def report(title, off_covs, blind_covs):
    print("\n" + "=" * 80)
    print(f"  {title}   (n_offline={len(off_covs)}  n_blind={len(blind_covs)})")
    print("=" * 80)
    dists = {}
    for key, order in [("K", list(range(1, 9))), ("cat", CAT_ORDER),
                       ("us", US_ORDER), ("mono", MONO_ORDER), ("dec", DEC_ORDER)]:
        off = marginal(off_covs, key, order)
        bl = marginal(blind_covs, key, order)
        print_marginal({"K": "K (predict turn)", "cat": "prior-artist catalog",
                        "us": "user-history #other-sessions",
                        "mono": "mono/multi",
                        "dec": "decade-branch fires (year of last prior)"}[key], off, bl, order)
        dists[key] = tv(off, bl)
    overall = float(np.mean(list(dists.values())))
    print(f"\n  >> mean covariate TV distance = {overall:.3f}  "
          f"(0 = identical; lower is better)")
    return overall


# ─── targets ─────────────────────────────────────────────────────────────────
def apportion(fracs, total):
    raw = {k: f * total for k, f in fracs.items()}
    out = {k: int(np.floor(v)) for k, v in raw.items()}
    rem = total - sum(out.values())
    for k in sorted(raw, key=lambda k: -(raw[k] - out[k]))[:max(rem, 0)]:
        out[k] += 1
    return out


def build_targets(blind_covs, size):
    n = max(len(blind_covs), 1)
    Kc = Counter(cv["K"] for cv in blind_covs)
    K_target = apportion({k: v / n for k, v in Kc.items()}, size)
    cat_within = {}
    for K, kt in K_target.items():
        rows = [cv for cv in blind_covs if cv["K"] == K]
        if not rows or kt == 0:
            cat_within[K] = {}
            continue
        cc = Counter(cv["cat"] for cv in rows)
        cat_within[K] = apportion({c: v / len(rows) for c, v in cc.items()}, kt)
    return K_target, cat_within


# ─── joint allocator (K hard, catalog best-effort, val/test share scarce cells) ─
def select_all(specs, universe, sessions, pool_user_counts, meta,
               blind_covs, blind_us, blind_mono, blind_dec, rng):
    sess_cand = {}
    for sid in universe:
        s = sessions[sid]
        nm = len(s["music"])
        if nm == 0:
            continue
        n_other = max(pool_user_counts.get(s["user_id"], 0) - 1, 0)
        sess_cand[sid] = {K: covariates(meta, s["music"], K, n_other)
                          for K in range(1, nm + 1)}
    cand_sids = defaultdict(list)
    for sid, ks in sess_cand.items():
        for K, cov in ks.items():
            cand_sids[(K, cov["cat"])].append(sid)

    def sec_w(cov):
        return (max(blind_us.get(cov["us"], 1e-6), 1e-6)
                * max(blind_mono.get(cov["mono"], 1e-6), 1e-6)
                * max(blind_dec.get(cov["dec"], 1e-6), 1e-6))

    targets = {name: build_targets(blind_covs, size) for name, size in specs}
    need = {name: dict(cw) for name in targets for cw in [
        {(K, c): cnt for K, cc in targets[name][1].items() for c, cnt in cc.items()}]}
    K_target = {name: targets[name][0] for name in targets}
    selected = {name: [] for name, _ in specs}
    selK = {name: Counter() for name, _ in specs}
    used = set()

    # PHASE 1 — exact (K,cat), scarcest stratum first, shared across splits
    all_strata = set()
    for name in need:
        all_strata |= set(need[name].keys())
    demand = {st: sum(need[name].get(st, 0) for name in need) for st in all_strata}
    supply = {st: len(cand_sids.get(st, [])) for st in all_strata}
    for st in sorted(all_strata, key=lambda s: -(demand[s] / max(supply[s], 1))):
        avail = [sid for sid in cand_sids.get(st, []) if sid not in used]
        if not avail:
            continue
        total_need = sum(need[name].get(st, 0) for name in need)
        if total_need <= 0:
            continue
        w = np.array([sec_w(sess_cand[sid][st[0]]) for sid in avail], dtype=float)
        w = w / w.sum()
        perm = rng.choice(len(avail), size=len(avail), replace=False, p=w)
        take = min(len(avail), total_need)
        alloc = apportion({name: need[name].get(st, 0) / total_need
                           for name in need if need[name].get(st, 0) > 0}, take)
        idx = 0
        for name in sorted(alloc, key=lambda n: -alloc[n]):
            want = min(alloc[name], need[name].get(st, 0))
            got = 0
            while idx < len(perm) and got < want:
                sid = avail[perm[idx]]; idx += 1
                if sid in used:
                    continue
                used.add(sid)
                selected[name].append((sid, st[0], sess_cand[sid][st[0]]))
                need[name][st] -= 1
                selK[name][st[0]] += 1
                got += 1

    # PHASE 2 — fill each split's per-K deficit from ANY unused session at that K
    for name, _ in specs:
        for K in sorted(K_target[name]):
            deficit = K_target[name][K] - selK[name][K]
            if deficit <= 0:
                continue
            cands = [sid for sid in sess_cand if sid not in used and K in sess_cand[sid]]
            if not cands:
                continue
            def w_of(sid):
                cov = sess_cand[sid][K]
                boost = 2.0 if need[name].get((K, cov["cat"]), 0) > 0 else 1.0
                return sec_w(cov) * boost
            w = np.array([w_of(sid) for sid in cands], dtype=float)
            w = w / w.sum()
            perm = rng.choice(len(cands), size=len(cands), replace=False, p=w)
            got = 0
            for j in perm:
                if got >= deficit:
                    break
                sid = cands[j]
                if sid in used:
                    continue
                used.add(sid)
                cov = sess_cand[sid][K]
                selected[name].append((sid, K, cov))
                if need[name].get((K, cov["cat"]), 0) > 0:
                    need[name][(K, cov["cat"])] -= 1
                selK[name][K] += 1
                got += 1
    return selected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track-meta", type=Path, default=DEFAULT_TRACK_META)
    ap.add_argument("--train-conv", type=Path, default=DEFAULT_TRAIN_CONV)
    ap.add_argument("--test-conv", type=Path, default=DEFAULT_TEST_CONV)
    ap.add_argument("--blind-conv", type=Path, default=DEFAULT_BLIND_CONV)
    ap.add_argument("--split-parquet", type=Path, default=DEFAULT_SPLIT_PARQUET,
                    help="Existing split — read for val/test SIZES and the BEFORE report.")
    ap.add_argument("--out-split", type=Path, default=DEFAULT_OUT_SPLIT)
    ap.add_argument("--val-size", type=int, default=None, help="Default: current #val rows.")
    ap.add_argument("--test-size", type=int, default=600,
                    help="Keep this SMALL. Use 0 to skip the test fold entirely.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    print("#" * 80)
    print("  REBUILD SPLIT — Blind-A-matched val/test")
    print("#" * 80)

    meta = load_track_index(args.track_meta)

    print("[pool] parsing organizer train + test ...")
    sessions = parse_pool(args.train_conv)
    sessions.update(parse_pool(args.test_conv))
    pool_user_counts = Counter(s["user_id"] for s in sessions.values())
    blind = parse_pool(args.blind_conv)
    print(f"  pool sessions={len(sessions)}  blind sessions={len(blind)}")

    split = pl.read_parquet(args.split_parquet)
    split_sids = set(str(x) for x in split["session_id"].to_list())
    universe = [sid for sid in sessions if sid in split_sids and sessions[sid]["music"]]
    print(f"  universe (split ∩ pool, has music) = {len(universe)}")

    val_size = args.val_size or int((split["split"] == "val").sum())
    test_size = args.test_size if args.test_size is not None else int((split["split"] == "test").sum())
    print(f"  target sizes: val={val_size}  test={test_size}")

    blind_covs = []
    for sid, s in blind.items():
        K = (s["music"][-1][0] + 1) if s["music"] else 1
        n_other = pool_user_counts.get(s["user_id"], 0)
        blind_covs.append(covariates(meta, s["music"], K, n_other))
    blind_us = marginal(blind_covs, "us", US_ORDER)
    blind_mono = marginal(blind_covs, "mono", MONO_ORDER)
    blind_dec = marginal(blind_covs, "dec", DEC_ORDER)

    cur_val = dict(zip([str(x) for x in split.filter(pl.col("split") == "val")["session_id"].to_list()],
                       split.filter(pl.col("split") == "val")["predict_turn_number"].to_list()
                       if "predict_turn_number" in split.columns
                       else [1] * int((split["split"] == "val").sum())))
    before_covs = []
    for sid, K in cur_val.items():
        s = sessions.get(sid)
        if s is None:
            continue
        n_other = max(pool_user_counts.get(s["user_id"], 0) - 1, 0)
        before_covs.append(covariates(meta, s["music"], int(K), n_other))
    before_d = report("BEFORE — current val vs Blind A", before_covs, blind_covs)

    sel = select_all([("val", val_size), ("test", test_size)], universe, sessions,
                     pool_user_counts, meta, blind_covs, blind_us, blind_mono, blind_dec, rng)
    val_sel, test_sel = sel["val"], sel["test"]

    after_val_d = report("AFTER — rebuilt val vs Blind A", [c for _, _, c in val_sel], blind_covs)
    if test_sel:
        report("AFTER — rebuilt test vs Blind A", [c for _, _, c in test_sel], blind_covs)

    print("\n" + "=" * 80)
    print(f"  CONVERGENCE: mean covariate TV  val  {before_d:.3f} -> {after_val_d:.3f}  "
          f"({'improved' if after_val_d < before_d else 'NO improvement'})")
    print("=" * 80)

    if args.report_only:
        print("\n[report-only] not writing a new split.")
        return

    rows = []
    for sid, K, _ in val_sel:
        rows.append({"session_id": sid, "split": "val", "predict_turn_number": int(K)})
    for sid, K, _ in test_sel:
        rows.append({"session_id": sid, "split": "test", "predict_turn_number": int(K)})
    val_test = {sid for sid, _, _ in val_sel} | {sid for sid, _, _ in test_sel}
    for sid in universe:                                  # everything else -> train
        if sid in val_test:
            continue
        n_music = len(sessions[sid]["music"])
        rows.append({"session_id": sid, "split": "train", "predict_turn_number": int(n_music)})

    out = pl.DataFrame(rows)
    args.out_split.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(args.out_split)
    n_train = int((out["split"] == "train").sum())
    print(f"\n[write] {args.out_split}")
    print(f"  train={n_train}  val={len(val_sel)}  test={len(test_sel)}  total={out.height}")
    print("\nNEXT: re-encode queries/tracks and re-run 05/06 against this split.")


if __name__ == "__main__":
    main()