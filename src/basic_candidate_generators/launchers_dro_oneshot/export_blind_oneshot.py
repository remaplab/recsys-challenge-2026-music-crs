"""Export ALL-turn blind candidates for one-shot components and their rrf
aggregator, for an arbitrary blind parquet (e.g. blind-B).

``export_oneshot_candidates.py`` already exports OOF/holdout at every turn, but
its *blind* path predicts only the target turn. This launcher fixes that for new
blinds: it runs the component (or the fused rrf_oneshot) over EVERY turn of the
blind sessions — history + withheld submission turn — via the same
``run_inference_dispatch`` / ``FusionIndex`` used elsewhere.

Two modes:

    # one component → <model>_oneshot/datasets/<out_name>
    uv run python -m launchers_dro_oneshot.export_blind_oneshot \\
        --model dense_text_8b --blind path/to/blind_b.parquet \\
        --out_name blind_b_all_turns_candidates.parquet

    # fuse the components → rrf_oneshot/datasets/<out_name>
    # (run the per-component exports with the SAME --out_name first)
    uv run python -m launchers_dro_oneshot.export_blind_oneshot --rrf \\
        --blind path/to/blind_b.parquet --out_name blind_b_all_turns_candidates.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parent.parent
_SRC_ROOT = _PKG_ROOT / "src"
_CV_ROOT = _PKG_ROOT / "launchers_crossvalidation"
_REPO_ROOT = _PKG_ROOT.parent.parent
_LBO_SRC = _REPO_ROOT / "src" / "lower_bound_optimization" / "src"
for _p in (_SRC_ROOT, _CV_ROOT, _PKG_ROOT, str(_LBO_SRC)):
    sys.path.insert(0, str(_p))

import polars as pl    # noqa: E402
import yaml            # noqa: E402

from _cv_utils import (  # noqa: E402
    instantiate_rec, load_fold, pkg_path, repo_path, resolve_param_paths,
    run_inference_dispatch,
)

from launchers_crossvalidation.export_blind import _assemble_blind_full   # noqa: E402
from launchers_dro_oneshot.export_oneshot_candidates import _best_params  # noqa: E402
from launchers_dro_oneshot.tune_rrf_oneshot import FusionIndex, _load_comp  # noqa: E402


def _component_blind(model: str, cfg: dict, storage_dir: Path, blind: Path,
                     out_name: str, top_n: int, alpha: float) -> Path:
    mcfg = cfg["models"][model]
    data_cfg = cfg["data"]
    n_folds = int(data_cfg.get("n_folds", 5))
    splitk_dir = repo_path(data_cfg["splitk_dir"])
    track_meta = pl.read_parquet(repo_path(data_cfg["track_metadata_path"]))
    inf_mode = mcfg.get("inference_mode", "standard")
    fixed = resolve_param_paths(mcfg.get("fixed_params") or {})
    params = {**_best_params(model, storage_dir, alpha), **fixed}
    uses_colisten = mcfg.get("uses_colisten", False)

    full_train = pl.concat(
        [load_fold(splitk_dir, f) for f in range(n_folds)]
    ).unique(subset=["session_id", "turn_number"])
    rec = instantiate_rec(mcfg["class"], mcfg["module"], params, "session")
    fit_kwargs = {"track_metadata": track_meta}
    if uses_colisten:
        fit_kwargs["colisten_df"] = full_train
    rec.fit(full_train, **fit_kwargs)

    blind_df = pl.read_parquet(blind if blind.is_absolute() else repo_path(str(blind)))
    full_df = _assemble_blind_full(blind_df)
    recs = run_inference_dispatch(rec, full_df, top_n, inf_mode, track_meta)
    out = storage_dir / f"{model}_oneshot" / "datasets" / out_name
    out.parent.mkdir(parents=True, exist_ok=True)
    recs.write_parquet(out)
    print(f"[export_blind_oneshot] {model}: {recs.shape[0]} rows "
          f"({recs['session_id'].n_unique()} sess, turns {recs['turn'].min()}-{recs['turn'].max()}) → {out}")
    return out


def _rrf_blind(cfg: dict, storage_dir: Path, out_name: str, top_n: int, alpha: float) -> Path:
    import optuna
    from _cv_utils import make_storage

    rrf_cfg = cfg["rrf"]
    components = [m for m in rrf_cfg["components"]
                  if (storage_dir / f"{m}_oneshot" / "datasets" / out_name).exists()]
    if len(components) < 2:
        sys.exit(f"[rrf] need >=2 components with {out_name}; have {components}. "
                 "Run the per-component export first.")
    anchor = rrf_cfg.get("anchor_component") or components[0]
    print(f"[rrf] fusing {len(components)} components (anchor={anchor})")

    comp_lists, user_by_key = [], None
    for m in components:
        lists, users = _load_comp(storage_dir / f"{m}_oneshot" / "datasets" / out_name)
        comp_lists.append(lists)
        if m == anchor:
            user_by_key = users
    idx = FusionIndex(comp_lists, user_by_key)

    # Best weights/k from the rrf study (reconstruct its space-tagged name).
    w_log = bool(rrf_cfg["weight"].get("log", False))
    k_log = bool(rrf_cfg["k_rrf"].get("log", False))
    space_tag = f"{'klog' if k_log else 'klin'}{'_wlog' if w_log else ''}_anc-{anchor}"
    study = optuna.load_study(
        study_name=f"rrf_oneshot_cvar{int(round(alpha*100))}_{space_tag}",
        storage=make_storage(storage_dir / "rrf_oneshot" / "optuna_rrf_oneshot.db"))
    best = study.best_params
    weights = [1.0 if m == anchor else float(best[f"w_{m}"]) for m in components]
    k_rrf = float(best["k_rrf"])
    print(f"[rrf] best k_rrf={k_rrf}  weights={dict(zip(components, weights))}")

    recs = idx.fuse(weights, k_rrf, top_n)
    out = storage_dir / "rrf_oneshot" / "datasets" / out_name
    out.parent.mkdir(parents=True, exist_ok=True)
    recs.write_parquet(out)
    print(f"[export_blind_oneshot] rrf_oneshot: {recs.shape[0]} rows "
          f"({recs['session_id'].n_unique()} sess) → {out}")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default=None, help="One-shot component (component mode).")
    p.add_argument("--rrf", action="store_true", help="Fuse components → rrf_oneshot.")
    p.add_argument("--blind", type=Path, required=True)
    p.add_argument("--out_name", default="blind_b_all_turns_candidates.parquet")
    p.add_argument("--config", default="launchers_dro_oneshot/configs/tune_oneshot.yaml")
    p.add_argument("--storage_dir", default="models/CG_crossvalidation")
    p.add_argument("--top_n", type=int, default=300)
    p.add_argument("--robust_alpha", type=float, default=0.7)
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config if Path(args.config).is_absolute()
                              else pkg_path(args.config)))
    storage_dir = repo_path(args.storage_dir)

    if args.rrf:
        _rrf_blind(cfg, storage_dir, args.out_name, args.top_n, args.robust_alpha)
    elif args.model:
        if args.model not in cfg["models"]:
            sys.exit(f"[export_blind_oneshot] unknown model {args.model!r}")
        _component_blind(args.model, cfg, storage_dir, args.blind,
                         args.out_name, args.top_n, args.robust_alpha)
    else:
        sys.exit("[export_blind_oneshot] provide --model <component> or --rrf")


if __name__ == "__main__":
    main()
