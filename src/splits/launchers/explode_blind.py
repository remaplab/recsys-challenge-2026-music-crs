"""explode_blind — turn the raw Blind-A / Blind-B parquets into the readable
one-row-per-turn format used across the pipeline (same shape as splitK).

Default (no args) explodes both blinds and writes four files to
data/exploded_blind/:
  blind-a.parquet              one row per COMPLETE turn (user+music+assistant)
  blind-b.parquet
  submission-blind-a.parquet   the withheld final turns (user-only, track null)
  submission-blind-b.parquet

Single dataset by path:
  python explode_blind.py --input PATH --name blind-a

Session-level fields (session_date, user_profile, conversation_goal,
goal_progress_assessments) are kept on every row — bridge keys to
listening-history-filtered.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data" / "talkpl-ai"
BLIND_A = DATA / "TalkPlayData-Challenge-Blind-A/data/test-00000-of-00001.parquet"
BLIND_B = DATA / "TalkPlayData-Challenge-Blind-B/data/test-00000-of-00001.parquet"
OUT = ROOT / "data" / "exploded_blind"

SESS_COLS = ["session_id", "user_id", "session_date", "user_profile",
             "conversation_goal", "goal_progress_assessments"]


def _explode(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return (complete_turns, target_turns).

    complete_turns : one row per turn that has a user query AND a music answer.
    target_turns   : user turns with no music answer (the withheld GT to predict).
    """
    base = df.explode("conversations").unnest("conversations")
    users = (
        base.filter(pl.col("role") == "user")
        .select(*SESS_COLS, "turn_number",
                pl.col("thought").alias("user_thought"),
                pl.col("content").alias("user_query"))
    )
    music = (
        base.filter(pl.col("role") == "music")
        .select("session_id", "turn_number",
                pl.col("thought").alias("assistant_thought"),
                pl.col("content").alias("track_id"))
    )
    asst = (
        base.filter(pl.col("role") == "assistant")
        .select("session_id", "turn_number",
                pl.col("content").alias("assistant_response"))
    )
    complete = (
        users.join(music, on=["session_id", "turn_number"], how="inner")
        .join(asst, on=["session_id", "turn_number"], how="left")
        .sort("session_id", "turn_number")
    )
    target = (
        users.join(music.select("session_id", "turn_number"),
                   on=["session_id", "turn_number"], how="anti")
        .sort("session_id", "turn_number")
    )
    return complete, target


def _write(name: str, complete: pl.DataFrame, target: pl.DataFrame) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"{name}.parquet"
    complete.write_parquet(p)
    print(f"  {p.name:28s} {complete.height:>6,} rows  "
          f"{complete['session_id'].n_unique():>4} sessions")
    if target.height:
        sp = OUT / f"submission-{name}.parquet"
        target.write_parquet(sp)
        print(f"  {sp.name:28s} {target.height:>6,} rows  "
              f"{target['session_id'].n_unique():>4} sessions (to predict)")


def _explode_path(src: Path, name: str) -> None:
    print(f"[explode] {name} <- {src}")
    c, t = _explode(pl.read_parquet(src))
    _write(name, c, t)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--input", type=Path, help="parquet to explode")
    p.add_argument("--name", help="output basename (e.g. blind-a)")
    args = p.parse_args()

    if args.input or args.name:
        if not (args.input and args.name):
            p.error("provide BOTH --input PATH and --name NAME (or neither to explode both blinds)")
        _explode_path(args.input, args.name)
        return

    # Default: explode both raw blinds.
    _explode_path(BLIND_A, "blind-a")
    _explode_path(BLIND_B, "blind-b")


if __name__ == "__main__":
    main()
