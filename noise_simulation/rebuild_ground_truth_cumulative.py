"""Rebuild ground-truth cumulative files from existing Doc29 batch outputs.

This utility fixes dB-domain aggregation mistakes by recomputing aggregation
in energy domain from already generated:
  groups/<A_D>_<Runway>/<NPD_ID>/groundtruth_batch_*.csv

Usage:
  python noise_simulation/rebuild_ground_truth_cumulative.py \
    --output-root noise_simulation/results_ground_truth/preprocessed_1

Input format (batch CSV):
  - semicolon separated
  - columns include x, y, z, cumulative_res

Outputs:
  - overwrites each group_cumulative.csv
  - overwrites ground_truth_cumulative.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from noise_simulation.receiver_points import annotate_measuring_points


def _add_energy(accum: Optional[pd.DataFrame], df: pd.DataFrame) -> pd.DataFrame:
    """Add cumulative_res in energy domain on (x, y, z) keys."""
    req = {"x", "y", "z", "cumulative_res"}
    if not req.issubset(df.columns):
        raise ValueError(f"Missing required columns: {sorted(req - set(df.columns))}")
    new = df[["x", "y", "z", "cumulative_res"]].copy()
    new["energy"] = np.power(10.0, new["cumulative_res"].to_numpy() / 10.0)
    new = new.drop(columns=["cumulative_res"])
    if accum is None:
        return new
    merged = accum.merge(new, on=["x", "y", "z"], how="outer", suffixes=("_a", "_b"))
    merged["energy"] = merged["energy_a"].fillna(0.0) + merged["energy_b"].fillna(0.0)
    return merged[["x", "y", "z", "energy"]]


def _energy_to_db(df_energy: pd.DataFrame) -> pd.DataFrame:
    """Convert energy-domain frame to x,y,z,cumulative_res."""
    out = df_energy.copy()
    out["cumulative_res"] = 10.0 * np.log10(np.maximum(out["energy"].to_numpy(), 1e-12))
    out = out[["x", "y", "z", "cumulative_res"]]
    return annotate_measuring_points(out)


def _merge_energy(accum: Optional[pd.DataFrame], new: pd.DataFrame) -> pd.DataFrame:
    """Merge two energy-domain frames on x,y,z."""
    if accum is None:
        return new
    merged = accum.merge(new, on=["x", "y", "z"], how="outer", suffixes=("_a", "_b"))
    merged["energy"] = merged["energy_a"].fillna(0.0) + merged["energy_b"].fillna(0.0)
    return merged[["x", "y", "z", "energy"]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild ground-truth cumulative files from existing batches.")
    parser.add_argument("--output-root", required=True, help="Ground-truth output root (contains groups/).")
    args = parser.parse_args()

    root = Path(args.output_root).resolve()
    groups_root = root / "groups"
    if not groups_root.exists():
        raise FileNotFoundError(f"Missing groups folder: {groups_root}")

    global_accum: Optional[pd.DataFrame] = None
    group_dirs = [p for p in groups_root.rglob("*") if p.is_dir()]
    rebuilt = 0

    for group_dir in sorted(group_dirs):
        batch_files = sorted(group_dir.glob("groundtruth_batch_*.csv"))
        if not batch_files:
            continue

        group_accum: Optional[pd.DataFrame] = None
        for batch_file in batch_files:
            batch_df = pd.read_csv(batch_file, sep=";")
            group_accum = _add_energy(group_accum, batch_df)

        if group_accum is None:
            continue

        group_out = group_dir / "group_cumulative.csv"
        _energy_to_db(group_accum).to_csv(group_out, index=False)
        global_accum = _merge_energy(global_accum, group_accum)
        rebuilt += 1

    if global_accum is None:
        raise RuntimeError("No group batch files found to rebuild.")

    global_out = root / "ground_truth_cumulative.csv"
    _energy_to_db(global_accum).to_csv(global_out, index=False)

    print(f"Rebuilt groups: {rebuilt}")
    print(f"Wrote: {global_out}")


if __name__ == "__main__":
    main()
