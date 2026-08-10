# RecSys 2026 ACM challenge, Team: Hallucinated

## Step 1: Download the full challenge datasets to obtail the following structure

```
data/talkpl-ai/TalkPlayData-Challenge-Blind-A
data/talkpl-ai/TalkPlayData-Challenge-Blind-B
data/talkpl-ai/TalkPlayData-Challenge-Dataset
data/talkpl-ai/TalkPlayData-Challenge-Track-Embeddings
data/talkpl-ai/TalkPlayData-Challenge-Track-Metadata # Both all_tracks and test_tracks
data/talkpl-ai/TalkPlayData-Challenge-User-Embeddings
data/talkpl-ai/TalkPlayData-Challenge-User-Metadata
```

## Step 2: Generate split K

```bash
cd src/splits
uv run python -u -m launchers.splitK_crossvalidation
uv run python -u -m launchers.explode_blind
```

## Step 3: Compute Qwen embeddings

```bash
cd src/heuristic
uv run python scripts/12_encode_gambling_caches.py \
    --stages tracks blind blindb_all splitk \
    --models qwen3_0p6b qwen3_4b qwen3_8b --instruction-prompts none \
    --query-batch-size 2 --track-batch-size 2 --allow-downloads --trust-remote-code
    
uv run scripts/03_encode_queries.py --encoder qwen3_frozen --splits train dev blind_a blind_b
```

## Step 4: Candidate Generators

Check readme at `src/basic_candidate_generators/README_REGEN_CG_DATASETS.md` and `src/bert4rec/README.md`

## Step 5: Reranker

Check readme at `src/reranker_oof/RERANKER_README.md`

## Step 6: Heuristic

Check readme at `src/heuristic/README.md`

## Step 7: LLM response generation

Check readme at `src/resp_blind_b/README.md`

## Step 8: Merge responses to prediction json

```bash
uv run merge_submission.py -p path/to/prediction.json -r path/to/response.json
```

For example:
```bash
uv run merge_submission.py -p heuristic/LAST.json -r ../data/blind_b_responses/final_pipeline/responses.json
```