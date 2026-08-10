"""Sweep encoders end-to-end and produce a single comparison table.

For every encoder in --encoders:
  1. encode queries (skip if cache already exists)
  2. encode tracks  (skip for qwen3_*; skip if cache exists)
  3. evaluate recall + heatmaps on dev
Then print a side-by-side recall table and save it as CSV.
"""
from __future__ import annotations
import argparse
import json
import subprocess
from pathlib import Path

import polars as pl


EVAL_OUT = Path("./models/eval_results")


def run(cmd: list[str]) -> None:
    print(f"\n>>> {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--encoders", nargs="+", required=True,
                   help="Cache names to compare, e.g. qwen3_frozen "
                        "qwen3_lora__routing qwen3_lora__no_routing bert_frozen ...")
    p.add_argument("--split", default="dev")
    args = p.parse_args()

    rows = []
    for enc in args.encoders:
        report_path = EVAL_OUT / enc / args.split / "recall.json"
        if not report_path.exists():
            print(f"[warn] no report at {report_path} -- run 06_evaluate first")
            continue
        data = json.loads(report_path.read_text())
        row = {"encoder": enc, **{k: v for k, v in data["overall"].items()
                                  if k.startswith("recall@")}}
        row["n"] = data["overall"]["n_eval"]
        rows.append(row)

    if not rows:
        print("no reports found"); return
    df = pl.DataFrame(rows)
    print("\n=== COMPARISON ===")
    print(df.to_pandas().to_string(index=False))
    out_csv = EVAL_OUT / f"comparison_{args.split}.csv"
    df.write_csv(out_csv)
    print(f"\nSaved {out_csv}")


if __name__ == "__main__":
    main()