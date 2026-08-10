"""Pre-build + cache the similarity matrices for the embedding-based CGs.

HP-tuning trials should never pay the one-off cost of building W_emb / W_img /
W_cbf. This script primes those caches up front by fitting each EmbeddingItemKNN
model once per fold with every signal forced on — `_fit_model` then builds and
writes any missing cache as a side effect.

Cache shapes:
  * W_emb (Qwen tower)   — fold-independent, one .npz, built on the first fold.
  * W_img (SigLIP2)      — fold-independent, one .npz, built on the first fold.
  * W_cbf (tag-CBF/ICM)  — fold-DEPENDENT (interaction-popularity), content-hash
                           keyed → one .npz per fold.
All are built at their `*_cache_k` and trimmed to the search-time k at load, so
re-running this script is idempotent (existing caches are loaded, not rebuilt).

Run:
    cd src/basic_candidate_generators
    uv run python -m launchers_dro.init_cg_caches            # all emb models in cfg
    uv run python -m launchers_dro.init_cg_caches --model emb_item_knn_8b
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parent.parent   # src/basic_candidate_generators/
_SRC_ROOT = _PKG_ROOT / "src"
_CV_ROOT = _PKG_ROOT / "launchers_crossvalidation"
_DRO_ROOT = _PKG_ROOT / "launchers_dro"

for p in (_SRC_ROOT, _CV_ROOT, _DRO_ROOT):
    sys.path.insert(0, str(p))

import polars as pl   # noqa: E402
import yaml           # noqa: E402

from _cv_utils import (  # noqa: E402
    instantiate_rec,
    load_fold,
    pkg_path,
    repo_path,
    resolve_param_paths,
)

_TARGET_CLASS = "EmbeddingItemKNN"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prime W_emb/W_img/W_cbf caches for CG tuning.")
    p.add_argument("--model", default=None,
                   help="Single model key. Default: every EmbeddingItemKNN model in the config.")
    p.add_argument("--config", default="launchers_dro/configs/tune_crossvalidation_dro.yaml")
    p.add_argument("--n_folds", type=int, default=None, help="Override data.n_folds.")
    return p.parse_args()


def _prime_model(model_name: str, mcfg: dict, cfg: dict,
                 splitk_dir: Path, track_meta, n_folds: int) -> None:
    fixed = resolve_param_paths(mcfg.get("fixed_params") or {})
    urm_mode = (mcfg.get("urm_modes") or ["session"])[0]
    # Force every cached signal on so each branch builds + writes its cache;
    # CF is fold-dependent and cheap, so leave it off.
    params = {**fixed, "icm_weight": 1.0, "img_weight": 1.0, "cf_weight": 0.0}

    print(f"\n{'=' * 60}")
    print(f"[init-cache] model={model_name}  urm_mode={urm_mode}  folds={n_folds}")
    for fold in range(n_folds):
        t0 = time.time()
        train_df = load_fold(splitk_dir, fold)
        rec = instantiate_rec(mcfg["class"], mcfg["module"], params, urm_mode)
        rec.fit(train_df, track_metadata=track_meta)
        print(f"[init-cache] {model_name} fold {fold} primed in {time.time() - t0:.1f}s")


def main() -> None:
    args = parse_args()
    with open(pkg_path(args.config)) as f:
        cfg = yaml.safe_load(f)

    models = cfg["models"]
    if args.model:
        if args.model not in models:
            sys.exit(f"[init-cache] unknown model '{args.model}'. Available: {list(models)}")
        targets = [args.model]
    else:
        targets = [m for m, mc in models.items() if mc.get("class") == _TARGET_CLASS]
        if not targets:
            sys.exit(f"[init-cache] no {_TARGET_CLASS} models in config")

    data_cfg = cfg["data"]
    splitk_dir = repo_path(data_cfg["splitk_dir"])
    n_folds = args.n_folds or int(data_cfg.get("n_folds", 5))
    track_meta = pl.read_parquet(repo_path(data_cfg["track_metadata_path"]))

    print(f"[init-cache] priming {len(targets)} model(s): {targets}")
    for model_name in targets:
        _prime_model(model_name, models[model_name], cfg, splitk_dir, track_meta, n_folds)
    print("\n[init-cache] done")


if __name__ == "__main__":
    main()
