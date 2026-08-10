"""Export all-turn candidates for an arbitrary blind (DRO variant).

Thin wrapper over ``launchers_crossvalidation.export_blind.run_export_blind`` —
same all-turns inference, but resolves the DRO folder (``{model}_{urm}_dro``) and
its ``best_params_..._dro{suffix}.yaml`` (next to the optuna DB, else the legacy
package config). The full.pkl checkpoint already encodes the fitted model, so
only ``class`` / ``module`` / ``inference_mode`` are read from the YAML.

Usage:
    cd src/basic_candidate_generators
    uv run python -m launchers_dro.export_blind_dro \\
        --model heuristic --urm_mode session --robust_mode cvar --robust_alpha 0.7 \\
        --blind path/to/blind_b.parquet --out_name blind_b_all_turns_candidates.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PKG_ROOT / "src"))
sys.path.insert(0, str(_PKG_ROOT / "launchers_crossvalidation"))

from _cv_utils import pkg_path   # noqa: E402

from launchers_crossvalidation.export_blind import run_export_blind   # noqa: E402


def _suffix(robust_mode: str, robust_alpha: float) -> str:
    if robust_mode == "cvar":
        return f"_cvar{int(round(robust_alpha * 100))}"
    return f"_{robust_mode}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", required=True)
    p.add_argument("--urm_mode", required=True, choices=["session", "user"])
    p.add_argument("--robust_mode", default="cvar", choices=["mean", "cvar", "group_dro"])
    p.add_argument("--robust_alpha", type=float, default=0.7)
    p.add_argument("--blind", type=Path, required=True)
    p.add_argument("--out_name", default="blind_b_all_turns_candidates.parquet")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--top_k", type=int, default=200)
    p.add_argument("--storage_dir", default="models/CG_crossvalidation")
    args = p.parse_args()

    suffix = _suffix(args.robust_mode, args.robust_alpha)
    folder_key = f"{args.model}_{args.urm_mode}_dro"
    _REPO_ROOT = Path(__file__).resolve().parents[3]
    storage_yaml = (_REPO_ROOT / args.storage_dir / folder_key
                    / f"best_params_{folder_key}{suffix}.yaml")
    legacy_yaml = pkg_path(f"configs/cv_best_{folder_key}{suffix}.yaml")
    cfg_path = storage_yaml if storage_yaml.exists() else legacy_yaml

    run_export_blind(
        model=args.model, urm_mode=args.urm_mode, folder_key=folder_key,
        cfg_path=cfg_path, blind=args.blind, out_name=args.out_name,
        checkpoint=args.checkpoint, top_k=args.top_k, storage_dir=args.storage_dir,
    )


if __name__ == "__main__":
    main()
