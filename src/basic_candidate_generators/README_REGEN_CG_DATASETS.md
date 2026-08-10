# Regenerate CG datasets for the reranker

The 8 reranker CGs: `bm25_colisten_oneshot`, `tower_a_oneshot`,
`emb_item_knn_8b_session_dro`, `hybrid_all_qwen_session_dro`,
`tower_ensemble_session_dro`, `tower_cf_ensemble_session_dro`,
`heuristic_v2_hybrid_session_dro`, `heuristic_v3_session_dro`.
Oneshot components feed `rrf_oneshot`; DRO fusion heuristics (stage 3) need
stage 2's outputs on disk first.

## Requirements

- `data/splitK/` (fold + holdout splits).
- Qwen query caches per tower folder: `dense_*` (splitK/holdout/blind-A) +
  `dense_blindb_all_*` (blind-B) — without the latter, no `blind_b_candidates`.
- GPU (towers + ensembles).
- Oneshot: `checkpoints/{non_holdout,full}.pkl.gz` (+ per-fold) under each
  `<cg>_oneshot/`. DRO: `checkpoints/*.pkl` under each `<cg>_session_dro/`.
- Fusion configs: `rrf_oneshot/best_params_rrf_oneshot_cvar70.yaml`.

```bash
cd src/basic_candidate_generators
```

## Inference (checkpoints only)

```bash
# oneshot components (13, checkpoint-only, no fit) — all splits (folds +
# holdout + blind-A + blind-B), not just blind-B: downstream (rrf_oneshot
# fusion, then the DRO fusion heuristics) reads these CGs' full dataset, so
# validate all of it here rather than risk a partial/blind-B-only rewrite
# silently breaking the more complex recommenders that consume it next.
uv run python -u -m launchers_dro_oneshot.infer_oneshot_from_checkpoints --model dense_text_8b --splits folds holdout full blind_b
uv run python -u -m launchers_dro_oneshot.infer_oneshot_from_checkpoints --model dense_text_4b --splits folds holdout full blind_b
uv run python -u -m launchers_dro_oneshot.infer_oneshot_from_checkpoints --model dense_text_0p6b --splits folds holdout full blind_b
uv run python -u -m launchers_dro_oneshot.infer_oneshot_from_checkpoints --model tfidf --splits folds holdout full blind_b
uv run python -u -m launchers_dro_oneshot.infer_oneshot_from_checkpoints --model bm25 --splits folds holdout full blind_b
uv run python -u -m launchers_dro_oneshot.infer_oneshot_from_checkpoints --model bm25f --splits folds holdout full blind_b
uv run python -u -m launchers_dro_oneshot.infer_oneshot_from_checkpoints --model char_ngram --splits folds holdout full blind_b
uv run python -u -m launchers_dro_oneshot.infer_oneshot_from_checkpoints --model entity_match --splits folds holdout full blind_b
uv run python -u -m launchers_dro_oneshot.infer_oneshot_from_checkpoints --model bm25_colisten --splits folds holdout full blind_b
uv run python -u -m launchers_dro_oneshot.infer_oneshot_from_checkpoints --model tower_a --splits folds holdout full blind_b
uv run python -u -m launchers_dro_oneshot.infer_oneshot_from_checkpoints --model tower_b --splits folds holdout full blind_b
uv run python -u -m launchers_dro_oneshot.infer_oneshot_from_checkpoints --model tower_c --splits folds holdout full blind_b
uv run python -u -m launchers_dro_oneshot.infer_oneshot_from_checkpoints --model tower_audioclap --splits folds holdout full blind_b

# rrf_oneshot fusion — auto-scopes to whichever splits the components actually
# have (skips + prints "no component candidates" for any split missing on disk;
# with only blind_b_candidates.parquet present above, this writes only
# rrf_oneshot/datasets/blind_b_candidates.parquet)
uv run python -u -m launchers_dro_oneshot.recreate_rrf_datasets \
  --best models/CG_crossvalidation/rrf_oneshot/best_params_rrf_oneshot_cvar70.yaml

# DRO CGs, checkpoint-loaded (state only — folds/holdout/blind-A get rewritten as a side effect, ignore them)
uv run python -u -m launchers_dro.infer_from_checkpoints --model emb_item_knn_8b --urm_mode session
uv run python -u -m launchers_dro.infer_from_checkpoints --model hybrid_all_qwen --urm_mode session
uv run python -u -m launchers_dro.infer_from_checkpoints --model tower_ensemble --urm_mode session
uv run python -u -m launchers_dro.infer_from_checkpoints --model tower_cf_ensemble --urm_mode session
uv run python -u -m launchers_dro.infer_from_checkpoints --model heuristic_v2_hybrid --urm_mode session
uv run python -u -m launchers_dro.infer_from_checkpoints --model heuristic_v3 --urm_mode session

# DRO Blind-B, checkpoint-loaded
# Order matters: fusion heuristics need their upstreams' blind_b_candidates
# on disk first.
uv run python -u -m launchers_dro.infer_blind_b_from_checkpoint --model emb_item_knn_8b --urm_mode session
uv run python -u -m launchers_dro.infer_blind_b_from_checkpoint --model hybrid_all_qwen --urm_mode session
uv run python -u -m launchers_dro.infer_blind_b_from_checkpoint --model tower_ensemble --urm_mode session
uv run python -u -m launchers_dro.infer_blind_b_from_checkpoint --model tower_cf_ensemble --urm_mode session
uv run python -u -m launchers_dro.infer_blind_b_from_checkpoint --model heuristic_v2_hybrid --urm_mode session
uv run python -u -m launchers_dro.infer_blind_b_from_checkpoint --model heuristic_v3 --urm_mode session

# Blind-A all-turns assembly (the 8 reranker CGs)
uv run python -u -m launchers_crossvalidation.assemble_blind_a --only \
  bm25_colisten_oneshot tower_a_oneshot \
  emb_item_knn_8b_session_dro hybrid_all_qwen_session_dro \
  tower_ensemble_session_dro tower_cf_ensemble_session_dro \
  heuristic_v2_hybrid_session_dro heuristic_v3_session_dro
```

## Training (retrain from scratch, all splits)

```bash
# oneshot components (13) — folds + holdout + blind-A, then blind-B (reuses full.pkl.gz)
uv run python -u -m launchers_dro_oneshot.retrain_and_export_oneshot_ckpt --model dense_text_8b
uv run python -u -m launchers_dro_oneshot.retrain_and_export_oneshot_ckpt --model dense_text_8b --blind_b_only
uv run python -u -m launchers_dro_oneshot.retrain_and_export_oneshot_ckpt --model dense_text_4b
uv run python -u -m launchers_dro_oneshot.retrain_and_export_oneshot_ckpt --model dense_text_4b --blind_b_only
uv run python -u -m launchers_dro_oneshot.retrain_and_export_oneshot_ckpt --model dense_text_0p6b
uv run python -u -m launchers_dro_oneshot.retrain_and_export_oneshot_ckpt --model dense_text_0p6b --blind_b_only
uv run python -u -m launchers_dro_oneshot.retrain_and_export_oneshot_ckpt --model tfidf
uv run python -u -m launchers_dro_oneshot.retrain_and_export_oneshot_ckpt --model tfidf --blind_b_only
uv run python -u -m launchers_dro_oneshot.retrain_and_export_oneshot_ckpt --model bm25
uv run python -u -m launchers_dro_oneshot.retrain_and_export_oneshot_ckpt --model bm25 --blind_b_only
uv run python -u -m launchers_dro_oneshot.retrain_and_export_oneshot_ckpt --model bm25f
uv run python -u -m launchers_dro_oneshot.retrain_and_export_oneshot_ckpt --model bm25f --blind_b_only
uv run python -u -m launchers_dro_oneshot.retrain_and_export_oneshot_ckpt --model char_ngram
uv run python -u -m launchers_dro_oneshot.retrain_and_export_oneshot_ckpt --model char_ngram --blind_b_only
uv run python -u -m launchers_dro_oneshot.retrain_and_export_oneshot_ckpt --model entity_match
uv run python -u -m launchers_dro_oneshot.retrain_and_export_oneshot_ckpt --model entity_match --blind_b_only
uv run python -u -m launchers_dro_oneshot.retrain_and_export_oneshot_ckpt --model bm25_colisten
uv run python -u -m launchers_dro_oneshot.retrain_and_export_oneshot_ckpt --model bm25_colisten --blind_b_only
uv run python -u -m launchers_dro_oneshot.retrain_and_export_oneshot_ckpt --model tower_a
uv run python -u -m launchers_dro_oneshot.retrain_and_export_oneshot_ckpt --model tower_a --blind_b_only
uv run python -u -m launchers_dro_oneshot.retrain_and_export_oneshot_ckpt --model tower_b
uv run python -u -m launchers_dro_oneshot.retrain_and_export_oneshot_ckpt --model tower_b --blind_b_only
uv run python -u -m launchers_dro_oneshot.retrain_and_export_oneshot_ckpt --model tower_c
uv run python -u -m launchers_dro_oneshot.retrain_and_export_oneshot_ckpt --model tower_c --blind_b_only
uv run python -u -m launchers_dro_oneshot.retrain_and_export_oneshot_ckpt --model tower_audioclap
uv run python -u -m launchers_dro_oneshot.retrain_and_export_oneshot_ckpt --model tower_audioclap --blind_b_only

# rrf_oneshot fusion (needs stage above on disk first)
uv run python -u -m launchers_dro_oneshot.recreate_rrf_datasets \
  --best models/CG_crossvalidation/rrf_oneshot/best_params_rrf_oneshot_cvar70.yaml

# DRO base CGs
uv run python -u -m launchers_dro.retrain_and_export_dro --model emb_item_knn_8b --urm_mode session
uv run python -u -m launchers_dro.export_blind_b_dro --model emb_item_knn_8b
uv run python -u -m launchers_dro.retrain_and_export_dro --model hybrid_all_qwen --urm_mode session
uv run python -u -m launchers_dro.export_blind_b_dro --model hybrid_all_qwen
uv run python -u -m launchers_dro.retrain_and_export_dro --model tower_ensemble --urm_mode session
uv run python -u -m launchers_dro.export_blind_b_dro --model tower_ensemble
uv run python -u -m launchers_dro.retrain_and_export_dro --model tower_cf_ensemble --urm_mode session
uv run python -u -m launchers_dro.export_blind_b_dro --model tower_cf_ensemble

# DRO fusion heuristics (needs DRO base CGs above on disk first)
uv run python -u -m launchers_dro.retrain_and_export_dro --model heuristic_v2_hybrid --urm_mode session
uv run python -u -m launchers_dro.export_blind_b_dro --model heuristic_v2_hybrid
uv run python -u -m launchers_dro.retrain_and_export_dro --model heuristic_v3 --urm_mode session
uv run python -u -m launchers_dro.export_blind_b_dro --model heuristic_v3

# Blind-A all-turns assembly (the 8 reranker CGs)
uv run python -u -m launchers_crossvalidation.assemble_blind_a --only \
  bm25_colisten_oneshot tower_a_oneshot \
  emb_item_knn_8b_session_dro hybrid_all_qwen_session_dro \
  tower_ensemble_session_dro tower_cf_ensemble_session_dro \
  heuristic_v2_hybrid_session_dro heuristic_v3_session_dro
```
