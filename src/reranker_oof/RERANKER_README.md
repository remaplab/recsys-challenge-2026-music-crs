# How to run the reranker pipeline

## Requirements

- CG candidate outputs at `models/CG_crossvalidation/<CG name>/datasets/` for
  **every** CG listed under `cgs:` in `configs/blind_no_filter/dataset.yaml`
  (currently 14 — the 8 reranker CGs plus 6 `query_full_*`/`split_hidim_*`
  BERT4Rec members): `fold_{0..4}_oof_cg_val.parquet` / `fold_{0..4}_oof_reranker_val.parquet`,
  `holdout_candidates.parquet`, `blind_b_candidates.parquet`, `blind_a_all_turns_candidates.parquet`.
  Blind-B-only inference (see below) needs just `blind_b_candidates.parquet`.
- Saved calibrators for Blind-B-only inference: `models/reranker_oof/calibrators/<name>/calibrators.pkl`
  (`train_cg_calibrators.py` — fits from `fold_{0..4}_oof_cg_val.parquet`, needs all 14 CGs).
- `data/splitK/` (folds + `holdout_test.parquet`), plus `data/exploded_blind/blind-{a,b}.parquet` for ground truth.
- Talkpl-ai catalog: Track-Metadata, User-Metadata, User-Embeddings (warm-user flag), Track-Embeddings (SigLIP2 + LAION-CLAP modality embeddings).
- Qwen3-8B towers at `models/retrieval_text_towers/Qwen__Qwen3-Embedding-8B/` — track tower + per-split query caches (incl. blind).
- For retrain (`s06_retrain_submit`) only: optuna DB at `models/reranker_oof/optuna/blind_b/no_filter_v5`.

```bash
cd src/reranker_oof
```
## Inference

No retrain, no optuna DB — only loads boosters already saved under
`<out_dir>/boosters/booster_*.json` (from a prior `s06` run) and rewrites `submissions/`, `scored_*.parquet`,
`metrics_*.csv`, `candidates/` in place. Skips SHAP.

**Blind-B-only fast resubmit (assemble + resubmit in one script):**
Calibrators checkpoint is required at `models/reranker_oof/calibrators/blind_no_filter/calibrators.pkl`

```bash
uv run python -m launchers_overfit_blind_b.s06c_blind_b_only \
    --dataset_config configs/blind_no_filter/dataset.yaml \
    --config configs/blind_no_filter/xgb_v5.yaml \
    --variants v2_blind_last
```


## Retrain reranker - no tuning

Store the calibrators for Blind-B-only inference (from the 14 CGs' fold OOF parquets):
```bash
uv run python -m launchers_overfit_blind_b.train_cg_calibrators \
    --config configs/blind_no_filter/dataset.yaml
```

**Assemble the dataset:**
```bash
uv run python -m launchers_overfit_blind_b.s03_assemble_dataset --config configs/blind_no_filter/dataset.yaml
```

To disable the test_tracks filtering just put to `false` the following in `src/reranker_oof/configs/blind_no_filter/dataset.yaml`:
```yaml
test_tracks:
  enabled: false
  path: data/talkpl-ai/TalkPlayData-Challenge-Track-Metadata/data/test_tracks-00000-of-00001.parquet
```

**Retrain reranker and make inferece:**
```bash
uv run python -u -m launchers_overfit_blind_b.s06_retrain_submit --config configs/blind_no_filter/xgb_v5.yaml
```
Note: The optuna database is required at `models/reranker_oof/optuna/blind_b/no_filter_v5`

**Find your outputs:**
You can find your outputs in the `models/reranker_oof/blind_b_retrain/no_filter_v5/v2_blind_last/` folder.


## COMPLETE TUNING PIPELINE (not needed, just for reference)

Given the above prerequisites, you can run the complete tuning pipeline as follows:
*(you can point the config files also at your own paths, if you have a different setup)*

```bash
# Tuning of weighted rrf aggregation
uv run python -m launchers_overfit_blind_b.s01_tune_rrf --config configs/blind_no_filter/dataset.yaml

# Export the best rrf weights to a yaml file
uv run python -m launchers_overfit_blind_b.s02_export_best_rrf --config configs/blind_no_filter/dataset.yaml

# Assemble the dataset for reranker training
uv run python -m launchers_overfit_blind_b.s03_assemble_dataset --config configs/blind_no_filter/dataset.yaml

# Tune the reranker hyperparameters with optuna
uv run python -u -m launchers_overfit_blind_b.s05_tune_reranker --config configs/blind_no_filter/xgb_v5.yaml

# Retrain the reranker with the best hyperparameters and make inference on blind-B
uv run python -u -m launchers_overfit_blind_b.s06_retrain_submit --config configs/blind_no_filter/xgb_v5.yaml
```
