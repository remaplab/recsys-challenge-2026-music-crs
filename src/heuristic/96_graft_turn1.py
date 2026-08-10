"""scripts/96_graft_turn1.py  — CPU, no GPU.

Take a fused TURN-1 file (from script 95) and SUBSTITUTE its turn-1 predictions into one or
more complete submissions. Every non-turn-1 entry is left untouched; each turn-1 entry's
predicted_track_ids is replaced by the fused turn-1 ranking for that session.

This lets a strong-on-the-rest submission inherit the better turn-1 recall/nDCG from the RRF
fusion, without changing anything at turns >= 2.

OUTPUT: for each input submission X.json, writes X_t1graft.json (same schema, turn-1 swapped).

USAGE:
  python scripts/96_graft_turn1.py --fused-turn1 fused_turn1.json \
      sub_blindb_cands_3.json sub_blindb_0p6276.json ...
  # optional: --suffix _t1graft  --outdir grafted/
"""
from __future__ import annotations
import argparse, json
from pathlib import Path


def load_list(path):
    d=json.load(open(path))
    if isinstance(d,dict): d=list(d.values()) if all(isinstance(v,dict) for v in d.values()) else d
    return d


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--fused-turn1",required=True,type=Path,
                    help="fused turn-1 JSON from script 95 (entries with turn_number=1).")
    ap.add_argument("subs",nargs="+",help="complete submissions to graft turn-1 into.")
    ap.add_argument("--suffix",default="_t1graft",help="output filename suffix.")
    ap.add_argument("--outdir",type=Path,default=None,help="output dir (default: alongside input).")
    a=ap.parse_args()

    fused=load_list(a.fused_turn1)
    # map session_id -> fused turn-1 predictions
    fused_by={str(e["session_id"]):[str(x) for x in (e.get("predicted_track_ids") or [])]
              for e in fused if e.get("turn_number")==1}
    print(f"  fused turn-1 sessions available: {len(fused_by)}")

    for p in a.subs:
        p=Path(p); entries=load_list(p)
        swapped=0; t1_total=0; missing=[]
        for e in entries:
            if e.get("turn_number")==1:
                t1_total+=1
                sid=str(e["session_id"])
                if sid in fused_by:
                    e["predicted_track_ids"]=list(fused_by[sid]); swapped+=1
                else:
                    missing.append(sid)
        outdir=a.outdir or p.parent
        outdir.mkdir(parents=True,exist_ok=True)
        outp=outdir/f"{p.stem}{a.suffix}{p.suffix}"
        outp.write_text(json.dumps(entries,indent=2))
        msg=f"  {p.name}: grafted {swapped}/{t1_total} turn-1 entries -> {outp.name}"
        if missing: msg+=f"   (no fused for {len(missing)}: {[s[:8] for s in missing[:5]]})"
        print(msg)
    print("\n  done. Non-turn-1 entries left untouched; only turn-1 predicted_track_ids replaced.")


if __name__=="__main__":
    main()