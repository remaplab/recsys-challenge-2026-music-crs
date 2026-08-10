"""scripts/98_top1_graft.py  — CPU, no GPU.

Build a new submission that, per session, takes the TOP-1 prediction from submission A and
places it at rank 1, then fills ranks 2..N from submission B in B's order, skipping any track
already used (so the A top-1 is never duplicated). The result keeps A's confident first pick
but B's ranking for the rest.

Per session:
  rank 1            = A.predicted_track_ids[0]            (A's top pick)
  ranks 2..N        = B.predicted_track_ids, in order, EXCLUDING the rank-1 track (dedup)
Truncated to N (default 20).

If a session is missing from A, the entry falls back to B's full list (and vice-versa); if both
missing, the session is skipped.

OUTPUT: a complete submission JSON (same schema as the inputs).

USAGE:
  python scripts/98_top1_graft.py --a subA.json --b subB.json --out combined.json
  # optional: --n 20  (final list length)
"""
from __future__ import annotations
import argparse, json
from pathlib import Path


def load_sub(path):
    d=json.load(open(path))
    if isinstance(d,dict): d=list(d.values()) if all(isinstance(v,dict) for v in d.values()) else d
    by={}
    for e in d: by[str(e["session_id"])]=e
    return by


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--a",required=True,type=Path,help="submission whose TOP-1 goes to rank 1.")
    ap.add_argument("--b",required=True,type=Path,help="submission that fills ranks 2..N (ordered).")
    ap.add_argument("--out",required=True,type=Path)
    ap.add_argument("--n",type=int,default=20,help="final list length (default 20).")
    a=ap.parse_args()

    A=load_sub(a.a); B=load_sub(a.b)
    all_sids=sorted(set(A)|set(B))
    out=[]; both=a_only=b_only=0
    for sid in all_sids:
        ea=A.get(sid); eb=B.get(sid)
        a_list=[str(x) for x in (ea.get("predicted_track_ids") or [])] if ea else []
        b_list=[str(x) for x in (eb.get("predicted_track_ids") or [])] if eb else []
        meta=ea or eb

        if a_list and b_list:
            both+=1
            top1=a_list[0]
            combined=[top1]
            for t in b_list:
                if t==top1: continue           # skip the A top-1 to avoid duplicate
                combined.append(t)
                if len(combined)>=a.n: break
            # if B was too short to fill N, top up from A's remainder (still deduped)
            if len(combined)<a.n:
                seen=set(combined)
                for t in a_list[1:]:
                    if t not in seen:
                        combined.append(t); seen.add(t)
                        if len(combined)>=a.n: break
        elif a_list:
            a_only+=1; combined=a_list[:a.n]    # B missing -> use A as-is
        else:
            b_only+=1; combined=b_list[:a.n]    # A missing -> use B as-is

        out.append({"session_id":sid,
                    "user_id":(meta or {}).get("user_id"),
                    "turn_number":(meta or {}).get("turn_number"),
                    "predicted_track_ids":combined[:a.n]})

    a.out.write_text(json.dumps(out,indent=2))
    print(f"  A (top-1 source): {a.a.name}")
    print(f"  B (rank 2..{a.n} source): {a.b.name}")
    print(f"  sessions: {len(out)}  | both present: {both}  | A-only: {a_only}  | B-only: {b_only}")
    # sanity: confirm no duplicates within any session
    dup=sum(1 for e in out if len(e["predicted_track_ids"])!=len(set(e["predicted_track_ids"])))
    print(f"  sessions with duplicate track ids: {dup}  (should be 0)")
    print(f"  wrote -> {a.out}")


if __name__=="__main__":
    main()