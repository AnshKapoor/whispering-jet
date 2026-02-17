"""Check which preprocessed files from a preprocess grid already exist.

Inputs:
- Grid YAML with ``variants[].preprocessed_id`` and optional ``base_config``.
- Optional explicit preprocessed directory.

Output:
- Console summary (existing/missing IDs).
- CSV status table with one row per variant.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load YAML as a dict; return empty dict when file is empty."""

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data or {}


def _default_preprocessed_dir(grid_cfg: dict[str, Any], grid_path: Path) -> Path:
    """Resolve default preprocessed dir from base config referenced by grid."""

    base_rel = Path(str(grid_cfg.get("base_config", "config/backbone_full.yaml")))
    if base_rel.is_absolute():
        base_path = base_rel
    else:
        # Grids in this repo usually store paths relative to project root.
        from_cwd = (Path.cwd() / base_rel).resolve()
        from_grid = (grid_path.parent / base_rel).resolve()
        base_path = from_cwd if from_cwd.exists() else from_grid
    base_cfg = _load_yaml(base_path)
    root_out = Path(str((base_cfg.get("output", {}) or {}).get("dir", "data")))
    return root_out / "preprocessed"


def _variant_row(variant: dict[str, Any], preprocessed_dir: Path) -> dict[str, Any] | None:
    """Build one status row from a single preprocess-grid variant."""

    pre_id = variant.get("preprocessed_id")
    if pre_id is None:
        return None

    preprocessing = variant.get("preprocessing", {}) or {}
    resampling = preprocessing.get("resampling", {}) or {}
    smoothing = preprocessing.get("smoothing", {}) or {}

    out_path = preprocessed_dir / f"preprocessed_{int(pre_id)}.csv"
    return {
        "preprocessed_id": int(pre_id),
        "exists": out_path.exists(),
        "path": str(out_path),
        "resampling_method": str(resampling.get("method", "")),
        "n_points": resampling.get("n_points"),
        "smoothing_enabled": bool(smoothing.get("enabled", False)),
        "smoothing_method": str(smoothing.get("method", "none")),
    }


def main() -> None:
    """CLI entrypoint for grid-status check."""

    parser = argparse.ArgumentParser(description="Check preprocessed file presence for variants in a grid.")
    parser.add_argument("--grid", type=Path, default=Path("config/preprocess_grid.yaml"))
    parser.add_argument(
        "--preprocessed-dir",
        type=Path,
        default=None,
        help="Optional override for preprocessed directory (default: from base config output.dir/preprocessed).",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("output/eda/preprocessed_status/preprocess_grid_status.csv"),
    )
    args = parser.parse_args()

    grid_cfg = _load_yaml(args.grid)
    variants = list(grid_cfg.get("variants", []) or [])
    if not variants:
        raise ValueError(f"No variants found in {args.grid}")

    preprocessed_dir = args.preprocessed_dir or _default_preprocessed_dir(grid_cfg, args.grid)
    preprocessed_dir = preprocessed_dir.resolve()

    rows = []
    for variant in variants:
        row = _variant_row(variant, preprocessed_dir)
        if row is not None:
            rows.append(row)
    if not rows:
        raise ValueError("No variants with preprocessed_id found.")

    status = pd.DataFrame(rows).sort_values("preprocessed_id").reset_index(drop=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    status.to_csv(args.out_csv, index=False)

    existing = status[status["exists"]]
    missing = status[~status["exists"]]

    print(f"Grid: {args.grid}")
    print(f"Preprocessed dir: {preprocessed_dir}")
    print(f"Variants: total={len(status)} existing={len(existing)} missing={len(missing)}")
    if not missing.empty:
        missing_ids = ",".join(str(x) for x in missing["preprocessed_id"].tolist())
        print(f"Missing preprocessed_id: {missing_ids}")
    print(f"Status CSV: {args.out_csv}")


if __name__ == "__main__":
    main()
