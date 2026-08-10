"""Export / simulate the embeddings consumed by TwoTowerV2Recommender.

TwoTower v2 builds one text per (session_id, turn_number):

    listener_goal + " [SEP] " + user_message_for_that_turn

It does this for every observed music turn and, for blind-style sessions, for
the last unanswered user turn. In its default mode this script mirrors that
logic and exports:

  - meta.parquet: row metadata and the exact encoded text
  - embeddings.npy: float32 matrix aligned row-by-row with meta.parquet
  - lookup.pkl: dict[(session_id, turn_number)] -> embedding

If --checkpoint points to a fitted TwoTowerV2 .pkl, it also simulates the
trained model and exports:

  - predictions.parquet / submission_like.json
  - user_vectors.npy + user_vectors_meta.parquet
  - item_vectors.npy + item_vectors_meta.parquet
  - user_*_input.npy intermediate inputs to the trained user tower
  - item_*_input.npy intermediate inputs to the trained item tower

Example:
    cd src/basic_candidate_generators
    python -m launchers_crossvalidation.export_two_tower_v2_text_embeddings \
      --input ../../data/talkpl-ai/TalkPlayData-Challenge-Dataset/data/train-00000-of-00001.parquet \
      --input ../../data/talkpl-ai/TalkPlayData-Challenge-Dataset/data/test-00000-of-00001.parquet \
      --input ../../data/talkpl-ai/TalkPlayData-Challenge-Blind-A/data/test-00000-of-00001.parquet \
      --output-dir ../../models/two_tower_v2_text_embeddings
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import polars as pl


_PKG_ROOT = Path(__file__).resolve().parent.parent
_SRC_ROOT = _PKG_ROOT / "src"
_REPO_ROOT = _PKG_ROOT.parent.parent
sys.path.insert(0, str(_SRC_ROOT))


def _repo_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else _REPO_ROOT / p


def _default_inputs() -> list[Path]:
    return [
        _repo_path("data/talkpl-ai/TalkPlayData-Challenge-Dataset/data/train-00000-of-00001.parquet"),
        _repo_path("data/talkpl-ai/TalkPlayData-Challenge-Dataset/data/test-00000-of-00001.parquet"),
        _repo_path("data/talkpl-ai/TalkPlayData-Challenge-Blind-A/data/test-00000-of-00001.parquet"),
    ]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export SentenceTransformer text embeddings used by two_tower_v2."
    )
    p.add_argument(
        "--input",
        action="append",
        default=None,
        help="Raw TalkPlay parquet. Can be passed multiple times. Defaults to train+test+blind-A.",
    )
    p.add_argument(
        "--output-dir",
        default="models/two_tower_v2_text_embeddings",
        help="Output directory, resolved from the repository root unless absolute.",
    )
    p.add_argument(
        "--text-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model name. Keep this equal to two_tower_v2.text_model.",
    )
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, or any SentenceTransformer-supported device string.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing outputs if present.",
    )
    p.add_argument(
        "--include-cold-start-unanswered",
        action="store_true",
        help=(
            "Also encode the last user turn when a session has no previous music "
            "turns. This is useful for blind cold-start sessions, but it is a "
            "small extension beyond the current two_tower_v2 implementation."
        ),
    )
    p.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "Optional fitted TwoTowerV2 checkpoint (.pkl). If provided, the "
            "script also exports final user/item tower vectors and top-k predictions."
        ),
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of predictions to export when --checkpoint is provided.",
    )
    p.add_argument(
        "--skip-text-encode",
        action="store_true",
        help=(
            "With --checkpoint, skip standalone SBERT files and only run the "
            "loaded two-tower simulation."
        ),
    )
    return p.parse_args()


def build_rows(
    raw_df: pl.DataFrame,
    source_path: Path,
    include_cold_start_unanswered: bool,
) -> list[dict]:
    rows: list[dict] = []
    for row_idx, row in enumerate(raw_df.iter_rows(named=True)):
        sid = row["session_id"]
        convs = row.get("conversations") or []
        goal = (row.get("conversation_goal") or {}).get("listener_goal") or ""
        user_by_turn = {
            c["turn_number"]: c["content"]
            for c in convs
            if c.get("role") == "user"
        }

        for c in convs:
            if c.get("role") != "music":
                continue
            turn_number = c["turn_number"]
            user_text = user_by_turn.get(turn_number, "")
            rows.append({
                "session_id": sid,
                "turn_number": int(turn_number),
                "kind": "music_turn",
                "source_path": str(source_path),
                "source_row": row_idx,
                "text": goal + " [SEP] " + user_text,
            })

        music_turns = [
            c["turn_number"]
            for c in convs
            if c.get("role") == "music"
        ]
        last_conv_turn = max((c["turn_number"] for c in convs), default=0)
        has_unanswered_music_context = music_turns and last_conv_turn > max(music_turns)
        has_cold_start_unanswered = (
            include_cold_start_unanswered
            and not music_turns
            and last_conv_turn > 0
        )
        if has_unanswered_music_context or has_cold_start_unanswered:
            sorted_convs = sorted(convs, key=lambda c: c["turn_number"])
            last_user = next(
                (
                    c["content"]
                    for c in reversed(sorted_convs)
                    if c.get("role") == "user"
                ),
                "",
            )
            rows.append({
                "session_id": sid,
                "turn_number": int(last_conv_turn),
                "kind": (
                    "cold_start_unanswered_user_turn"
                    if has_cold_start_unanswered
                    else "last_unanswered_user_turn"
                ),
                "source_path": str(source_path),
                "source_row": row_idx,
                "text": goal + " [SEP] " + last_user,
            })
    return rows


def build_last_turn_context(raw_df: pl.DataFrame) -> pl.DataFrame:
    """Build the context_df consumed by TwoTowerV2.recommend for raw sessions."""
    rows: list[dict] = []
    for row in raw_df.iter_rows(named=True):
        sid = row["session_id"]
        uid = row["user_id"]
        session_date = row.get("session_date")
        convs = sorted(row.get("conversations") or [], key=lambda c: c["turn_number"])
        last_turn = max((c["turn_number"] for c in convs), default=0)
        added_context = False

        for c in convs:
            if c.get("role") != "music":
                continue
            if c["turn_number"] >= last_turn:
                continue
            rows.append({
                "session_id": sid,
                "user_id": uid,
                "session_date": session_date,
                "track_id": c["content"],
                "turn_number": int(c["turn_number"]),
                "target_turn": int(last_turn),
            })
            added_context = True

        if not added_context:
            rows.append({
                "session_id": sid,
                "user_id": uid,
                "session_date": session_date,
                "track_id": None,
                "turn_number": None,
                "target_turn": int(last_turn),
            })

    return pl.DataFrame(rows, schema={
        "session_id": pl.Utf8,
        "user_id": pl.Utf8,
        "session_date": pl.Utf8,
        "track_id": pl.Utf8,
        "turn_number": pl.Int64,
        "target_turn": pl.Int64,
    })


def recs_to_submission(recs: pl.DataFrame, top_k: int) -> list[dict]:
    rows: list[dict] = []
    for row in recs.iter_rows(named=True):
        rows.append({
            "session_id": row["session_id"],
            "user_id": row.get("user_id"),
            "turn_number": row["turn"],
            "predicted_track_ids": (row["track_ids"] or [])[:top_k],
            "predicted_response": "",
        })
    return rows


def export_user_tower_inputs(rec, sessions: list[dict], output_dir: Path) -> None:
    import torch

    from recommenders.two_tower_v2 import MAX_TAGS, TEXT_DIM

    B, L, MT = len(sessions), rec.max_ctx, MAX_TAGS
    ctx_ti = np.zeros((B, L), np.int64)
    ctx_ai = np.zeros((B, L), np.int64)
    ctx_tg = np.zeros((B, L, MT), np.int64)
    ctx_di = np.zeros((B, L), np.int64)
    ctx_pi = np.zeros((B, L), np.int64)
    ctx_ui = np.zeros((B, L), np.int64)
    ctx_mk = np.zeros((B, L), bool)
    u_age = np.zeros(B, np.int64)
    u_ctr = np.zeros(B, np.int64)
    u_gen = np.zeros(B, np.int64)
    text_sbert = np.zeros((B, TEXT_DIM), np.float32)

    for b, s in enumerate(sessions):
        ctx = s["context"][-L:]
        for j, tid in enumerate(ctx):
            f = rec._track_features.get(tid, rec._unk_track)
            ctx_ti[b, j] = f["track_idx"]
            ctx_ai[b, j] = f["artist_idx"]
            tags = f["tag_idxs"][:MT]
            ctx_tg[b, j, :len(tags)] = tags
            ctx_di[b, j] = f["decade_idx"]
            ctx_pi[b, j] = f["pop_bin"]
            ctx_ui[b, j] = f["dur_bin"]
            ctx_mk[b, j] = True
        uf = rec._user_features.get(s["user_id"], rec._unk_user)
        u_age[b] = uf["age_idx"]
        u_ctr[b] = uf["country_idx"]
        u_gen[b] = uf["gender_idx"]
        text_sbert[b] = rec._text_embeds.get(
            (s["session_id"], s["turn_number"]),
            np.zeros(TEXT_DIM, np.float32),
        )

    T = lambda a: torch.tensor(a).to(rec._device)
    ctx_t = {
        "track_idx": T(ctx_ti),
        "artist_idx": T(ctx_ai),
        "tag_idxs": T(ctx_tg),
        "decade_idx": T(ctx_di),
        "pop_bin": T(ctx_pi),
        "dur_bin": T(ctx_ui),
    }
    with torch.no_grad():
        rec._model.eval()
        user_tower = rec._model.user_tower
        text_projected = user_tower.text_proj(T(text_sbert)).cpu().numpy()
        B_t, L_t = ctx_t["track_idx"].shape
        flat = {k: v.reshape(B_t * L_t, *v.shape[2:]) for k, v in ctx_t.items()}
        item_context_vectors = (
            user_tower.item_tower(**flat)
            .view(B_t, L_t, rec.D)
            .cpu()
            .numpy()
        )
        mask = T(ctx_mk).float().unsqueeze(-1)
        if user_tower.recency_decay < 0.999:
            weights = torch.tensor(
                [user_tower.recency_decay ** (L_t - 1 - i) for i in range(L_t)],
                dtype=mask.dtype,
                device=rec._device,
            ).view(1, L_t, 1)
            mask = mask * weights
        ctx_vec = (
            torch.tensor(item_context_vectors, device=rec._device) * mask
        ).sum(1) / mask.sum(1).clamp(min=1)
        context_aggregated = ctx_vec.cpu().numpy()
        age_emb = user_tower.age_emb(T(u_age)).cpu().numpy()
        country_emb = user_tower.country_emb(T(u_ctr)).cpu().numpy()
        gender_emb = user_tower.gender_emb(T(u_gen)).cpu().numpy()
        demo_concat = np.concatenate([age_emb, country_emb, gender_emb], axis=1)
        user_mlp_input = np.concatenate(
            [text_projected, context_aggregated, demo_concat],
            axis=1,
        ).astype(np.float32, copy=False)

    np.save(output_dir / "user_text_sbert_input.npy", text_sbert)
    np.save(output_dir / "user_text_projected_input.npy", text_projected.astype(np.float32, copy=False))
    np.save(output_dir / "user_context_item_vectors_input.npy", item_context_vectors.astype(np.float32, copy=False))
    np.save(output_dir / "user_context_aggregated_input.npy", context_aggregated.astype(np.float32, copy=False))
    np.save(output_dir / "user_demo_embedding_input.npy", demo_concat.astype(np.float32, copy=False))
    np.save(output_dir / "user_tower_mlp_input.npy", user_mlp_input)


def export_item_tower_inputs(rec, output_dir: Path) -> None:
    import torch

    from recommenders.two_tower_v2 import _item_tensors_batch

    track_parts: list[np.ndarray] = []
    artist_parts: list[np.ndarray] = []
    tag_parts: list[np.ndarray] = []
    decade_parts: list[np.ndarray] = []
    pop_parts: list[np.ndarray] = []
    dur_parts: list[np.ndarray] = []
    concat_parts: list[np.ndarray] = []

    with torch.no_grad():
        rec._model.eval()
        item_tower = rec._model.item_tower
        for start in range(0, len(rec._tids), 512):
            chunk = rec._tids[start:start + 512]
            feats = {
                k: v.to(rec._device)
                for k, v in _item_tensors_batch(chunk, rec._track_features, rec._unk_track).items()
            }
            track_emb = item_tower.track_emb(feats["track_idx"])
            artist_emb = item_tower.artist_emb(feats["artist_idx"])
            tag_v = item_tower.tag_emb(feats["tag_idxs"])
            tag_mask = (feats["tag_idxs"] != 0).float().unsqueeze(-1)
            tag_emb = (tag_v * tag_mask).sum(-2) / tag_mask.sum(-2).clamp(min=1)
            decade_emb = item_tower.decade_emb(feats["decade_idx"])
            pop_emb = item_tower.pop_emb(feats["pop_bin"])
            dur_emb = item_tower.dur_emb(feats["dur_bin"])
            concat = torch.cat(
                [track_emb, artist_emb, tag_emb, decade_emb, pop_emb, dur_emb],
                dim=-1,
            )
            track_parts.append(track_emb.cpu().numpy())
            artist_parts.append(artist_emb.cpu().numpy())
            tag_parts.append(tag_emb.cpu().numpy())
            decade_parts.append(decade_emb.cpu().numpy())
            pop_parts.append(pop_emb.cpu().numpy())
            dur_parts.append(dur_emb.cpu().numpy())
            concat_parts.append(concat.cpu().numpy())

    np.save(output_dir / "item_track_id_embedding_input.npy", np.vstack(track_parts).astype(np.float32, copy=False))
    np.save(output_dir / "item_artist_embedding_input.npy", np.vstack(artist_parts).astype(np.float32, copy=False))
    np.save(output_dir / "item_tag_embedding_input.npy", np.vstack(tag_parts).astype(np.float32, copy=False))
    np.save(output_dir / "item_decade_embedding_input.npy", np.vstack(decade_parts).astype(np.float32, copy=False))
    np.save(output_dir / "item_pop_embedding_input.npy", np.vstack(pop_parts).astype(np.float32, copy=False))
    np.save(output_dir / "item_duration_embedding_input.npy", np.vstack(dur_parts).astype(np.float32, copy=False))
    np.save(output_dir / "item_tower_mlp_input.npy", np.vstack(concat_parts).astype(np.float32, copy=False))


def export_full_model_outputs(
    checkpoint: Path,
    raw_df: pl.DataFrame,
    output_dir: Path,
    top_k: int,
    text_model: str | None,
) -> None:
    import pickle

    import torch
    import torch.nn.functional as F

    from recommenders.two_tower_v2 import (
        TwoTowerV2Recommender,
        _encode_users,
        _item_tensors_batch,
    )

    print(f"[two_tower_v2 full] loading checkpoint {checkpoint}")
    # BaseRecommender.load() bypasses __init__ via __new__. Older copies of
    # TwoTowerV2Recommender._set_model_state then fail because _device is not
    # initialized yet. Instantiate normally, then restore state.
    with open(checkpoint, "rb") as f:
        state = pickle.load(f)
    rec = TwoTowerV2Recommender()
    rec._set_model_state(state)
    print(f"    loaded {state.get('recommender_name', 'TwoTowerV2')} from {checkpoint}")

    if text_model == "all-MiniLM-L6-v2":
        text_model = "sentence-transformers/all-MiniLM-L6-v2"
    if text_model:
        print(f"[two_tower_v2 full] using text_model={text_model}")
        rec.text_model = text_model
    elif rec.text_model == "all-MiniLM-L6-v2":
        rec.text_model = "sentence-transformers/all-MiniLM-L6-v2"
        print(f"[two_tower_v2 full] normalized text_model={rec.text_model}")

    print("[two_tower_v2 full] encoding missing raw-session texts")
    rec.encode_additional(raw_df)

    context_df = build_last_turn_context(raw_df)
    recs = rec.recommend(context_df, top_k=top_k, remove_seen=True)
    if "user_id" not in recs.columns:
        user_meta = raw_df.select(["session_id", "user_id"]).unique(subset=["session_id"])
        recs = recs.join(user_meta, on="session_id", how="left")
    recs.write_parquet(output_dir / "predictions.parquet")
    with open(output_dir / "submission_like.json", "w") as f:
        json.dump(recs_to_submission(recs, top_k), f, indent=2)

    sessions: list[dict] = []
    for session_key, group in context_df.group_by("session_id", maintain_order=True):
        sid = session_key[0] if isinstance(session_key, tuple) else session_key
        group = group.sort("turn_number", nulls_last=True)
        first = group.row(0, named=True)
        sessions.append({
            "session_id": sid,
            "user_id": first["user_id"],
            "turn_number": int(first["target_turn"]),
            "context": [
                r["track_id"]
                for r in group.iter_rows(named=True)
                if r.get("track_id") is not None
            ],
        })

    print(f"[two_tower_v2 full] exporting {len(sessions):,} final user vectors")
    user_vectors = _encode_users(
        rec._model,
        sessions,
        rec.max_ctx,
        rec._track_features,
        rec._user_features,
        rec._text_embeds,
        rec._unk_track,
        rec._unk_user,
        rec._device,
    ).astype(np.float32, copy=False)
    pl.DataFrame([
        {
            "session_id": s["session_id"],
            "user_id": s["user_id"],
            "turn_number": s["turn_number"],
            "context_len": len(s["context"]),
        }
        for s in sessions
    ]).write_parquet(output_dir / "user_vectors_meta.parquet")
    np.save(output_dir / "user_vectors.npy", user_vectors)

    print("[two_tower_v2 full] exporting user tower input embeddings")
    export_user_tower_inputs(rec, sessions, output_dir)

    print(f"[two_tower_v2 full] exporting {len(rec._tids):,} final item vectors")
    item_chunks: list[np.ndarray] = []
    with torch.no_grad():
        rec._model.eval()
        for start in range(0, len(rec._tids), 512):
            chunk = rec._tids[start:start + 512]
            feats = {
                k: v.to(rec._device)
                for k, v in _item_tensors_batch(chunk, rec._track_features, rec._unk_track).items()
            }
            item_chunks.append(
                F.normalize(rec._model.item_tower(**feats), dim=-1).cpu().numpy()
            )
    item_vectors = np.vstack(item_chunks).astype(np.float32, copy=False)
    pl.DataFrame({"track_id": rec._tids}).write_parquet(output_dir / "item_vectors_meta.parquet")
    np.save(output_dir / "item_vectors.npy", item_vectors)

    print("[two_tower_v2 full] exporting item tower input embeddings")
    export_item_tower_inputs(rec, output_dir)

    print(f"[two_tower_v2 full] wrote {output_dir / 'predictions.parquet'}")
    print(f"[two_tower_v2 full] wrote {output_dir / 'submission_like.json'}")
    print(f"[two_tower_v2 full] wrote user_vectors.npy shape={user_vectors.shape}")
    print(f"[two_tower_v2 full] wrote item_vectors.npy shape={item_vectors.shape}")


def main() -> None:
    args = parse_args()
    if args.skip_text_encode and not args.checkpoint:
        raise SystemExit("--skip-text-encode is only useful together with --checkpoint.")
    input_paths = [_repo_path(p) for p in args.input] if args.input else _default_inputs()
    output_dir = _repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    meta_path = output_dir / "meta.parquet"
    npy_path = output_dir / "embeddings.npy"
    lookup_path = output_dir / "lookup.pkl"
    full_paths = [
        output_dir / "predictions.parquet",
        output_dir / "submission_like.json",
        output_dir / "user_vectors_meta.parquet",
        output_dir / "user_vectors.npy",
        output_dir / "user_text_sbert_input.npy",
        output_dir / "user_text_projected_input.npy",
        output_dir / "user_context_item_vectors_input.npy",
        output_dir / "user_context_aggregated_input.npy",
        output_dir / "user_demo_embedding_input.npy",
        output_dir / "user_tower_mlp_input.npy",
        output_dir / "item_vectors_meta.parquet",
        output_dir / "item_vectors.npy",
        output_dir / "item_track_id_embedding_input.npy",
        output_dir / "item_artist_embedding_input.npy",
        output_dir / "item_tag_embedding_input.npy",
        output_dir / "item_decade_embedding_input.npy",
        output_dir / "item_pop_embedding_input.npy",
        output_dir / "item_duration_embedding_input.npy",
        output_dir / "item_tower_mlp_input.npy",
    ]
    checked_paths = [] if args.skip_text_encode else [meta_path, npy_path, lookup_path]
    if args.checkpoint:
        checked_paths.extend(full_paths)
    existing = [p for p in checked_paths if p.exists()]
    if existing and not args.overwrite:
        names = ", ".join(str(p) for p in existing)
        raise SystemExit(f"Output already exists: {names}. Pass --overwrite to replace.")

    raw_frames: list[pl.DataFrame] = []
    all_rows: list[dict] = []
    for input_path in input_paths:
        print(f"[two_tower_v2 embeddings] reading {input_path}")
        raw_df = pl.read_parquet(input_path)
        raw_frames.append(raw_df)
        if not args.skip_text_encode:
            all_rows.extend(
                build_rows(
                    raw_df,
                    input_path,
                    include_cold_start_unanswered=args.include_cold_start_unanswered,
                )
            )

    raw_all = pl.concat(raw_frames) if len(raw_frames) > 1 else raw_frames[0]

    if not args.skip_text_encode:
        if not all_rows:
            raise SystemExit("No rows to encode. Check that inputs contain a conversations column.")

        meta = pl.DataFrame(all_rows).unique(
            subset=["session_id", "turn_number"],
            keep="first",
            maintain_order=True,
        )
        texts = meta["text"].to_list()

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise SystemExit(
                "Missing dependency: sentence-transformers. Install it in the "
                "basic_candidate_generators environment, then rerun this script."
            ) from exc

        device = args.device
        if device == "auto":
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"

        print(
            f"[two_tower_v2 embeddings] encoding {len(texts):,} texts "
            f"with {args.text_model} on {device}"
        )
        encoder = SentenceTransformer(args.text_model, device=device)
        embeddings = encoder.encode(
            texts,
            batch_size=args.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32, copy=False)

        meta.write_parquet(meta_path)
        np.save(npy_path, embeddings)

        lookup = {
            (row["session_id"], int(row["turn_number"])): embeddings[i]
            for i, row in enumerate(meta.iter_rows(named=True))
        }
        with open(lookup_path, "wb") as f:
            pickle.dump(lookup, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"[two_tower_v2 embeddings] wrote {meta_path}")
        print(f"[two_tower_v2 embeddings] wrote {npy_path} shape={embeddings.shape}")
        print(f"[two_tower_v2 embeddings] wrote {lookup_path}")

    if args.checkpoint:
        export_full_model_outputs(
            checkpoint=_repo_path(args.checkpoint),
            raw_df=raw_all,
            output_dir=output_dir,
            top_k=args.top_k,
            text_model=args.text_model,
        )


if __name__ == "__main__":
    sys.exit(main())
