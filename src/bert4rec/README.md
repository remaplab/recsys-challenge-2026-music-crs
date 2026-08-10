# Candidate Generators

Six candidate-generation models (the `feature_bert4rec` family) and the
`retrain_and_export` pipeline, with the per-model hyperparameters under `configs/`.

## Layout

```
src/basic_candidate_generators/
  launchers_crossvalidation/
    retrain_and_export.py     # model retrain + export (checkpoint + submission)
    _cv_utils.py              # shared utilities (data loaders, inference, scoring)
  src/
    BaseRecommender.py
    recommenders/             # the 6 models + the inheritance chain they import
  configs/                    # hyperparameters per (model, objective)
```

## Requirements

- Python via **uv** (`uv run --no-sync`).
- The official challenge track embeddings + track metadata + the cross-validation
  splits under `data/splitK/`, at the paths in each config's `fixed_params`.
- A GPU for training.

## How to run

For each model, run the four steps below **in order**: retrain + export, then
Blind-B export, then the column fix, then Blind-A assembly last — assembly
unions on the raw `turn` column, so it must run *after* the fix or it keys
history rows to the wrong turn. All commands are one model at a time, passing
its config via `--config`, and use `--model` / `--objective` / `--objective_k`
from the table to fill in `<MODEL_KEY>`, `<YAML>`, `<OBJECTIVE>`, `<OBJECTIVE_K>`
below.

### Model → config → objective

| MODEL_KEY | --config | --objective | --objective_k | output folder |
|---|---|---|---|---|
| `query_full_multibehav_alltext_softor` | `configs/query_full_multibehav_alltext_softor_ndcg.yaml` | ndcg | 20 | `query_full_multibehav_alltext_softor_session_ndcg/` |
| `query_full_multibehav_alltext_softor` | `configs/query_full_multibehav_alltext_softor_recall.yaml` | recall | 200 | `query_full_multibehav_alltext_softor_session/` |
| `split_hidim_xattn_hardneg_query_full_dif` | `configs/split_hidim_xattn_hardneg_query_full_dif_ndcg.yaml` | ndcg | 20 | `split_hidim_xattn_hardneg_query_full_dif_session_ndcg/` |
| `split_hidim_xattn_hardneg_query_full_nova` | `configs/split_hidim_xattn_hardneg_query_full_nova_ndcg.yaml` | ndcg | 20 | `split_hidim_xattn_hardneg_query_full_nova_session_ndcg/` |
| `split_hidim_xattn_hardneg_query_full_noiserobust` | `configs/split_hidim_xattn_hardneg_query_full_noiserobust_recall.yaml` | recall | 200 | `split_hidim_xattn_hardneg_query_full_noiserobust_session/` |
| `split_hidim_xattn_hardneg` | `configs/split_hidim_xattn_hardneg_ndcg.yaml` | ndcg | 20 | `split_hidim_xattn_hardneg_session_ndcg/` |

`--model` must match the `model:` field inside the config; `class` and
`module` are read from the config. The output folder is `<MODEL_KEY>_session/`
with `_ndcg` **auto-appended** when `--objective ndcg` (pass `--folder_suffix ''` to force the bare name, or any other
string to override).

### Step 1 — retrain + export

Fits every fold, the non-holdout model (→ `holdout_candidates.parquet`), and
the full model (→ `checkpoints/full.pkl`, Blind-A per-turn candidates,
`submission/blind_A_<folder>.json`).

```bash
cd src/bert4rec/src/basic_candidate_generators # From the root of the repository
uv run --no-sync python -m launchers_crossvalidation.retrain_and_export \
    --model <MODEL_KEY> --urm_mode session \
    --config configs/<YAML> \
    --objective <OBJECTIVE> --objective_k <OBJECTIVE_K> \
    --storage_dir models/CG_crossvalidation --top_k 500
```

Output under `--storage_dir`/`<folder>/`:

- `checkpoints/full.pkl`, `checkpoints/non_holdout.pkl`, `checkpoints/fold_*_cg_train{,_val}.pkl`
- `datasets/fold_*_oof_cg_val.parquet`, `datasets/fold_*_oof_reranker_val.parquet`
- `datasets/holdout_candidates.parquet`
- `datasets/blind_candidates.parquet` (Blind-A, submission-turn only — see step 4)
- `submission/blind_A_<folder>.json`

Add `--skip_datasets --skip_holdout_candidates` to write only the checkpoint +
submission json.

**Re-exporting from checkpoints, without refitting:** once step 1 has run at
least once, three flags reload an existing checkpoint and skip training
entirely — combinable in any combination in one call:

- `--refresh_fold_datasets` — reloads the fold checkpoints, re-writes
  `fold_*_oof_{cg_val,reranker_val}.parquet` (e.g. at a different `--top_k`).
- `--refresh_holdout_candidates` — reloads `non_holdout.pkl`, re-writes
  `holdout_candidates.parquet` (same score report as a full run).
- `--refresh_blind_candidates` — reloads `full.pkl`, re-writes
  `blind_candidates.parquet` and `submission/blind_A_<folder>.json` (unless
  `--skip_blind_candidates`/`--skip_submission`).

```bash
uv run --no-sync python -m launchers_crossvalidation.retrain_and_export \
    --model <MODEL_KEY> --urm_mode session --config configs/<YAML> \
    --objective <OBJECTIVE> --objective_k <OBJECTIVE_K> \
    --storage_dir models/CG_crossvalidation --top_k 500 \
    --refresh_fold_datasets --refresh_holdout_candidates --refresh_blind_candidates
```

### Step 2 — Blind-B export

No retraining: reuse `checkpoints/full.pkl` from step 1 and run inference only
on the Blind-B set.

```bash
cd src/bert4rec/src/basic_candidate_generators # From the root of the repository
uv run --no-sync python -m launchers_crossvalidation.retrain_and_export \
    --model <MODEL_KEY> --urm_mode session --config configs/<YAML> \
    --objective <OBJECTIVE> --objective_k <OBJECTIVE_K> --storage_dir models/CG_crossvalidation --top_k 500 \
    --predict_blindB --blindB_path <blind_b_parquet> --blindB_query_split <split>
```

Writes `datasets/blind_candidates_B.parquet`, `datasets/blind_b_candidates.parquet`
(the per-turn file `dataset.yaml`'s `blind_b` split expects), and
`submission/blind_B_<folder>.json`. For query-aware models, `--blindB_query_split`
must name a query-cache split present under the checkpoint's `query_emb_dir`.

### Step 3 — fix dataset columns

Align turn numbers across the parquets written in steps 1–2, **before**
Blind-A assembly reads them:

```bash
cd src/basic_candidate_generators # From the root of the repository
uv run python -u -m launchers_crossvalidation.fix_dataset_columns \
        --path models/CG_crossvalidation/<folder>/datasets \
        --apply --drop_fallback_used --apply
```

### Step 4 — Blind-A assembly

`s03_assemble_dataset` (in `src/reranker_oof`) reads `blind_a` from
`blind_a_all_turns_candidates.parquet`, not `blind_candidates.parquet` — it's
the union of the history turns (from the fold OOF parquets, keyed on
`["session_id", "turn"]`) with the submission turn. Must run **after** step 3:
if `turn` is still the pre-fix constant value, the union keys history rows to
the wrong turn. Assemble from the **root** `basic_candidate_generators`
package (shared `models/CG_crossvalidation` store, not the bert4rec copy):

```bash
cd src/basic_candidate_generators # From the root of the repository
uv run python -m launchers_crossvalidation.assemble_blind_a --only <folder>
```

Omit `--only` to (re-)assemble it for every CG under `--cg-store` (default
`models/CG_crossvalidation`). Its output schema
(`session_id, user_id, turn, track_ids, scores, gt_track_id`) has no
`gt_turn_number`/`fallback_used` columns, so it doesn't need a second pass
through `fix_dataset_columns`.



## TRAINING

Full retrain (from scratch) of all 6 `feature_bert4rec` CGs, producing
`checkpoints/` + everything `src/reranker_oof`'s `dataset.yaml` needs.

**Do steps 1+2 (GPU, retrain)**, run every model's block below in sequence. **Then do step 3 for
every model, and only after that run step 4 — never interleaved per model.**
`assemble_blind_a`'s ground-truth backfill (`gt_map`) scans **every** CG
folder under `models/CG_crossvalidation`, not just the one being assembled;
if any other folder in the store still has the pre-fix constant-`turn`
schema at assembly time, its rows get mis-keyed.

### Steps 1+2 — per model, one at a time (GPU, retrain)

#### `query_full_multibehav_alltext_softor` — ndcg → `query_full_multibehav_alltext_softor_session_ndcg`

```bash
cd src/bert4rec/src/basic_candidate_generators # From the root of the repository
uv run --no-sync python -m launchers_crossvalidation.retrain_and_export \
    --model query_full_multibehav_alltext_softor --urm_mode session \
    --config configs/query_full_multibehav_alltext_softor_ndcg.yaml \
    --objective ndcg --objective_k 20 --storage_dir models/CG_crossvalidation --top_k 500

uv run --no-sync python -m launchers_crossvalidation.retrain_and_export \
    --model query_full_multibehav_alltext_softor --urm_mode session \
    --config configs/query_full_multibehav_alltext_softor_ndcg.yaml \
    --objective ndcg --objective_k 20 --storage_dir models/CG_crossvalidation --top_k 500 \
    --predict_blindB
```

#### `query_full_multibehav_alltext_softor` — recall → `query_full_multibehav_alltext_softor_session`

```bash
cd src/bert4rec/src/basic_candidate_generators # From the root of the repository
uv run --no-sync python -m launchers_crossvalidation.retrain_and_export \
    --model query_full_multibehav_alltext_softor --urm_mode session \
    --config configs/query_full_multibehav_alltext_softor_recall.yaml \
    --objective recall --objective_k 200 --storage_dir models/CG_crossvalidation --top_k 500

uv run --no-sync python -m launchers_crossvalidation.retrain_and_export \
    --model query_full_multibehav_alltext_softor --urm_mode session \
    --config configs/query_full_multibehav_alltext_softor_recall.yaml \
    --objective recall --objective_k 200 --storage_dir models/CG_crossvalidation --top_k 500 \
    --predict_blindB
```

#### `split_hidim_xattn_hardneg_query_full_dif` — ndcg → `split_hidim_xattn_hardneg_query_full_dif_session_ndcg`

```bash
cd src/bert4rec/src/basic_candidate_generators # From the root of the repository
uv run --no-sync python -m launchers_crossvalidation.retrain_and_export \
    --model split_hidim_xattn_hardneg_query_full_dif --urm_mode session \
    --config configs/split_hidim_xattn_hardneg_query_full_dif_ndcg.yaml \
    --objective ndcg --objective_k 20 --storage_dir models/CG_crossvalidation --top_k 500

uv run --no-sync python -m launchers_crossvalidation.retrain_and_export \
    --model split_hidim_xattn_hardneg_query_full_dif --urm_mode session \
    --config configs/split_hidim_xattn_hardneg_query_full_dif_ndcg.yaml \
    --objective ndcg --objective_k 20 --storage_dir models/CG_crossvalidation --top_k 500 \
    --predict_blindB
```

#### `split_hidim_xattn_hardneg_query_full_nova` — ndcg → `split_hidim_xattn_hardneg_query_full_nova_session_ndcg`

```bash
cd src/bert4rec/src/basic_candidate_generators # From the root of the repository
uv run --no-sync python -m launchers_crossvalidation.retrain_and_export \
    --model split_hidim_xattn_hardneg_query_full_nova --urm_mode session \
    --config configs/split_hidim_xattn_hardneg_query_full_nova_ndcg.yaml \
    --objective ndcg --objective_k 20 --storage_dir models/CG_crossvalidation --top_k 500

uv run --no-sync python -m launchers_crossvalidation.retrain_and_export \
    --model split_hidim_xattn_hardneg_query_full_nova --urm_mode session \
    --config configs/split_hidim_xattn_hardneg_query_full_nova_ndcg.yaml \
    --objective ndcg --objective_k 20 --storage_dir models/CG_crossvalidation --top_k 500 \
    --predict_blindB
```

#### `split_hidim_xattn_hardneg_query_full_noiserobust` — recall → `split_hidim_xattn_hardneg_query_full_noiserobust_session`

```bash
cd src/bert4rec/src/basic_candidate_generators # From the root of the repository
uv run --no-sync python -m launchers_crossvalidation.retrain_and_export \
    --model split_hidim_xattn_hardneg_query_full_noiserobust --urm_mode session \
    --config configs/split_hidim_xattn_hardneg_query_full_noiserobust_recall.yaml \
    --objective recall --objective_k 200 --storage_dir models/CG_crossvalidation --top_k 500

uv run --no-sync python -m launchers_crossvalidation.retrain_and_export \
    --model split_hidim_xattn_hardneg_query_full_noiserobust --urm_mode session \
    --config configs/split_hidim_xattn_hardneg_query_full_noiserobust_recall.yaml \
    --objective recall --objective_k 200 --storage_dir models/CG_crossvalidation --top_k 500 \
    --predict_blindB
```

#### `split_hidim_xattn_hardneg` — ndcg → `split_hidim_xattn_hardneg_session_ndcg`

```bash
cd src/bert4rec/src/basic_candidate_generators # From the root of the repository
uv run --no-sync python -m launchers_crossvalidation.retrain_and_export \
    --model split_hidim_xattn_hardneg --urm_mode session \
    --config configs/split_hidim_xattn_hardneg_ndcg.yaml \
    --objective ndcg --objective_k 20 --storage_dir models/CG_crossvalidation --top_k 500

uv run --no-sync python -m launchers_crossvalidation.retrain_and_export \
    --model split_hidim_xattn_hardneg --urm_mode session \
    --config configs/split_hidim_xattn_hardneg_ndcg.yaml \
    --objective ndcg --objective_k 20 --storage_dir models/CG_crossvalidation --top_k 500 \
    --predict_blindB
```

### Step 3 — fix columns for all 6 (run only after every model above has finished)

```bash
cd src/basic_candidate_generators # From the root of the repository
uv run python -u -m launchers_crossvalidation.fix_dataset_columns \
    --path "models/CG_crossvalidation/query_full_multibehav_alltext_softor_session_ndcg/datasets" \
    --apply --drop_fallback_used --apply
uv run python -u -m launchers_crossvalidation.fix_dataset_columns \
    --path "models/CG_crossvalidation/query_full_multibehav_alltext_softor_session/datasets" \
    --apply --drop_fallback_used --apply
uv run python -u -m launchers_crossvalidation.fix_dataset_columns \
    --path "models/CG_crossvalidation/split_hidim_xattn_hardneg_query_full_dif_session_ndcg/datasets" \
    --apply --drop_fallback_used --apply
uv run python -u -m launchers_crossvalidation.fix_dataset_columns \
    --path "models/CG_crossvalidation/split_hidim_xattn_hardneg_query_full_nova_session_ndcg/datasets" \
    --apply --drop_fallback_used --apply
uv run python -u -m launchers_crossvalidation.fix_dataset_columns \
    --path "models/CG_crossvalidation/split_hidim_xattn_hardneg_query_full_noiserobust_session/datasets" \
    --apply --drop_fallback_used --apply
uv run python -u -m launchers_crossvalidation.fix_dataset_columns \
    --path "models/CG_crossvalidation/split_hidim_xattn_hardneg_session_ndcg/datasets" \
    --apply --drop_fallback_used --apply
```

### Step 4 — Blind-A assembly for all 6 (one call, only after step 3 above is done for all 6)

```bash
cd src/basic_candidate_generators # From the root of the repository
uv run python -m launchers_crossvalidation.assemble_blind_a --only \
    query_full_multibehav_alltext_softor_session_ndcg \
    query_full_multibehav_alltext_softor_session \
    split_hidim_xattn_hardneg_query_full_dif_session_ndcg \
    split_hidim_xattn_hardneg_query_full_nova_session_ndcg \
    split_hidim_xattn_hardneg_query_full_noiserobust_session \
    split_hidim_xattn_hardneg_session_ndcg
```

## INFERENCE (checkpoint-only)

Checkpoint-only re-export of everything `src/reranker_oof`'s `dataset.yaml`
needs, for all 6 `feature_bert4rec` CGs, assuming `checkpoints/` already
exist from a prior full Training run above (no refitting).

**Do steps 1+2 (GPU, checkpoint reload)**, run every model's block below in sequence. **Then do step 3 for
every model, and only after that run step 4 — never interleaved per model.**
Same `gt_map`-scans-every-folder caveat as Training applies here too.

### Steps 1+2 — per model, one at a time (GPU, checkpoint reload)

#### `query_full_multibehav_alltext_softor` — ndcg → `query_full_multibehav_alltext_softor_session_ndcg`

```bash
cd src/bert4rec/src/basic_candidate_generators # From the root of the repository
uv run --no-sync python -m launchers_crossvalidation.retrain_and_export \
    --model query_full_multibehav_alltext_softor --urm_mode session \
    --config configs/query_full_multibehav_alltext_softor_ndcg.yaml \
    --objective ndcg --objective_k 20 --storage_dir models/CG_crossvalidation --top_k 500 \
    --refresh_fold_datasets --refresh_holdout_candidates --refresh_blind_candidates

uv run --no-sync python -m launchers_crossvalidation.retrain_and_export \
    --model query_full_multibehav_alltext_softor --urm_mode session \
    --config configs/query_full_multibehav_alltext_softor_ndcg.yaml \
    --objective ndcg --objective_k 20 --storage_dir models/CG_crossvalidation --top_k 500 \
    --predict_blindB
```

#### `query_full_multibehav_alltext_softor` — recall → `query_full_multibehav_alltext_softor_session`

```bash
cd src/bert4rec/src/basic_candidate_generators # From the root of the repository
uv run --no-sync python -m launchers_crossvalidation.retrain_and_export \
    --model query_full_multibehav_alltext_softor --urm_mode session \
    --config configs/query_full_multibehav_alltext_softor_recall.yaml \
    --objective recall --objective_k 200 --storage_dir models/CG_crossvalidation --top_k 500 \
    --refresh_fold_datasets --refresh_holdout_candidates --refresh_blind_candidates

uv run --no-sync python -m launchers_crossvalidation.retrain_and_export \
    --model query_full_multibehav_alltext_softor --urm_mode session \
    --config configs/query_full_multibehav_alltext_softor_recall.yaml \
    --objective recall --objective_k 200 --storage_dir models/CG_crossvalidation --top_k 500 \
    --predict_blindB
```

#### `split_hidim_xattn_hardneg_query_full_dif` — ndcg → `split_hidim_xattn_hardneg_query_full_dif_session_ndcg`

```bash
cd src/bert4rec/src/basic_candidate_generators # From the root of the repository
uv run --no-sync python -m launchers_crossvalidation.retrain_and_export \
    --model split_hidim_xattn_hardneg_query_full_dif --urm_mode session \
    --config configs/split_hidim_xattn_hardneg_query_full_dif_ndcg.yaml \
    --objective ndcg --objective_k 20 --storage_dir models/CG_crossvalidation --top_k 500 \
    --refresh_fold_datasets --refresh_holdout_candidates --refresh_blind_candidates

uv run --no-sync python -m launchers_crossvalidation.retrain_and_export \
    --model split_hidim_xattn_hardneg_query_full_dif --urm_mode session \
    --config configs/split_hidim_xattn_hardneg_query_full_dif_ndcg.yaml \
    --objective ndcg --objective_k 20 --storage_dir models/CG_crossvalidation --top_k 500 \
    --predict_blindB
```

#### `split_hidim_xattn_hardneg_query_full_nova` — ndcg → `split_hidim_xattn_hardneg_query_full_nova_session_ndcg`

```bash
cd src/bert4rec/src/basic_candidate_generators # From the root of the repository
uv run --no-sync python -m launchers_crossvalidation.retrain_and_export \
    --model split_hidim_xattn_hardneg_query_full_nova --urm_mode session \
    --config configs/split_hidim_xattn_hardneg_query_full_nova_ndcg.yaml \
    --objective ndcg --objective_k 20 --storage_dir models/CG_crossvalidation --top_k 500 \
    --refresh_fold_datasets --refresh_holdout_candidates --refresh_blind_candidates

uv run --no-sync python -m launchers_crossvalidation.retrain_and_export \
    --model split_hidim_xattn_hardneg_query_full_nova --urm_mode session \
    --config configs/split_hidim_xattn_hardneg_query_full_nova_ndcg.yaml \
    --objective ndcg --objective_k 20 --storage_dir models/CG_crossvalidation --top_k 500 \
    --predict_blindB
```

#### `split_hidim_xattn_hardneg_query_full_noiserobust` — recall → `split_hidim_xattn_hardneg_query_full_noiserobust_session`

```bash
cd src/bert4rec/src/basic_candidate_generators # From the root of the repository
uv run --no-sync python -m launchers_crossvalidation.retrain_and_export \
    --model split_hidim_xattn_hardneg_query_full_noiserobust --urm_mode session \
    --config configs/split_hidim_xattn_hardneg_query_full_noiserobust_recall.yaml \
    --objective recall --objective_k 200 --storage_dir models/CG_crossvalidation --top_k 500 \
    --refresh_fold_datasets --refresh_holdout_candidates --refresh_blind_candidates

uv run --no-sync python -m launchers_crossvalidation.retrain_and_export \
    --model split_hidim_xattn_hardneg_query_full_noiserobust --urm_mode session \
    --config configs/split_hidim_xattn_hardneg_query_full_noiserobust_recall.yaml \
    --objective recall --objective_k 200 --storage_dir models/CG_crossvalidation --top_k 500 \
    --predict_blindB
```

#### `split_hidim_xattn_hardneg` — ndcg → `split_hidim_xattn_hardneg_session_ndcg`

```bash
cd src/bert4rec/src/basic_candidate_generators # From the root of the repository
uv run --no-sync python -m launchers_crossvalidation.retrain_and_export \
    --model split_hidim_xattn_hardneg --urm_mode session \
    --config configs/split_hidim_xattn_hardneg_ndcg.yaml \
    --objective ndcg --objective_k 20 --storage_dir models/CG_crossvalidation --top_k 500 \
    --refresh_fold_datasets --refresh_holdout_candidates --refresh_blind_candidates

uv run --no-sync python -m launchers_crossvalidation.retrain_and_export \
    --model split_hidim_xattn_hardneg --urm_mode session \
    --config configs/split_hidim_xattn_hardneg_ndcg.yaml \
    --objective ndcg --objective_k 20 --storage_dir models/CG_crossvalidation --top_k 500 \
    --predict_blindB
```

### Step 3 — fix columns for all 6 (run only after every model above has finished)

```bash
cd src/basic_candidate_generators # From the root of the repository
uv run python -u -m launchers_crossvalidation.fix_dataset_columns \
    --path "models/CG_crossvalidation/query_full_multibehav_alltext_softor_session_ndcg/datasets" \
    --apply --drop_fallback_used --apply
uv run python -u -m launchers_crossvalidation.fix_dataset_columns \
    --path "models/CG_crossvalidation/query_full_multibehav_alltext_softor_session/datasets" \
    --apply --drop_fallback_used --apply
uv run python -u -m launchers_crossvalidation.fix_dataset_columns \
    --path "models/CG_crossvalidation/split_hidim_xattn_hardneg_query_full_dif_session_ndcg/datasets" \
    --apply --drop_fallback_used --apply
uv run python -u -m launchers_crossvalidation.fix_dataset_columns \
    --path "models/CG_crossvalidation/split_hidim_xattn_hardneg_query_full_nova_session_ndcg/datasets" \
    --apply --drop_fallback_used --apply
uv run python -u -m launchers_crossvalidation.fix_dataset_columns \
    --path "models/CG_crossvalidation/split_hidim_xattn_hardneg_query_full_noiserobust_session/datasets" \
    --apply --drop_fallback_used --apply
uv run python -u -m launchers_crossvalidation.fix_dataset_columns \
    --path "models/CG_crossvalidation/split_hidim_xattn_hardneg_session_ndcg/datasets" \
    --apply --drop_fallback_used --apply
```

### Step 4 — Blind-A assembly for all 6 (one call, only after step 3 above is done for all 6)

```bash
cd src/basic_candidate_generators # From the root of the repository
uv run python -m launchers_crossvalidation.assemble_blind_a --only \
    query_full_multibehav_alltext_softor_session_ndcg \
    query_full_multibehav_alltext_softor_session \
    split_hidim_xattn_hardneg_query_full_dif_session_ndcg \
    split_hidim_xattn_hardneg_query_full_nova_session_ndcg \
    split_hidim_xattn_hardneg_query_full_noiserobust_session \
    split_hidim_xattn_hardneg_session_ndcg
```