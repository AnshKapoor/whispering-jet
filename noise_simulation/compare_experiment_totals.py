"""Compare cluster-based predictions against retained-flight reference totals.

This script sits on top of Doc29 automation outputs and aggregates both:
1) cluster-based predictions (`subtracks.csv`)
2) retained-flight reference totals (`groundtruth.csv`)
in the energy domain, then compares them per category and overall.

Usage:
  python noise_simulation/compare_experiment_totals.py \
    --summary noise_simulation/results/EXP001/summary_mse.csv \
    --out noise_simulation/results/EXP001/aggregate_totals

Input format:
  - summary CSV rows from `run_doc29_experiment.py`, with at least:
    A/D, Runway, aircraft_type, n_flights, subtracks_csv, groundtruth_csv

Output files:
  - category_summary.csv
      One row per category with:
      mae_cluster, n_receivers
  - category_aligned_receivers.csv
      Receiver-level aligned values per category:
      measuring_point, L_eq_pred, L_eq_cluster, delta_cluster, abs_err_cluster
  - overall_aligned_9points.csv
      One row per receiver point (typically 9) after aggregating all categories:
      measuring_point, L_eq_pred, L_eq_cluster, delta_cluster, abs_err_cluster
  - overall_summary.json
      Same metrics after aggregating all categories together.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from noise_simulation.receiver_points import annotate_measuring_points


def _to_energy(level_db: np.ndarray) -> np.ndarray:
    """Convert dB to linear energy."""
    return np.power(10.0, level_db / 10.0)


def _to_db(energy: np.ndarray) -> np.ndarray:
    """Convert linear energy to dB with numeric floor."""
    return 10.0 * np.log10(np.maximum(energy, 1e-12))


def _load_doc29_cumulative(path: Path) -> pd.DataFrame:
    """Load Doc29 cumulative output and keep x,y,z,cumulative_res columns."""
    df = pd.read_csv(path, sep=";")
    needed = {"x", "y", "z", "cumulative_res"}
    if not needed.issubset(df.columns):
        missing = sorted(needed - set(df.columns))
        raise ValueError(f"{path} missing required columns: {missing}")
    return df[["x", "y", "z", "cumulative_res"]].copy()


def _accumulate_energy(
    accum: Optional[pd.DataFrame],
    new: pd.DataFrame,
    scale: float,
) -> pd.DataFrame:
    """Accumulate cumulative_res in energy domain with optional scaling."""
    tmp = new.copy()
    tmp["energy"] = _to_energy(tmp["cumulative_res"].to_numpy()) * float(scale)
    tmp = tmp.drop(columns=["cumulative_res"])
    if accum is None:
        return tmp
    merged = accum.merge(tmp, on=["x", "y", "z"], how="outer", suffixes=("_a", "_b"))
    merged["energy"] = merged["energy_a"].fillna(0.0) + merged["energy_b"].fillna(0.0)
    return merged[["x", "y", "z", "energy"]]


def _compare_energy_maps(
    pred_energy: pd.DataFrame,
    gt_energy: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Align prediction and retained-flight reference at receiver level."""
    pred = pred_energy.copy()
    gt = gt_energy.copy()
    pred["L_eq_pred"] = _to_db(pred["energy"].to_numpy())
    gt["L_eq_cluster"] = _to_db(gt["energy"].to_numpy())
    merged = pred[["x", "y", "z", "L_eq_pred"]].merge(
        gt[["x", "y", "z", "L_eq_cluster"]],
        on=["x", "y", "z"],
        how="inner",
    )
    if merged.empty:
        raise ValueError("No overlapping receiver rows between prediction and ground truth.")

    diff = merged["L_eq_pred"] - merged["L_eq_cluster"]
    merged["delta_cluster"] = diff
    merged["abs_err_cluster"] = np.abs(diff)

    mae = float(np.mean(merged["abs_err_cluster"]))

    metrics = {
        "mae_cluster": mae,
        "n_receivers": int(len(merged)),
    }
    return merged, metrics


def _parse_group_by(raw: str) -> List[str]:
    cols = [c.strip() for c in raw.split(",") if c.strip()]
    if not cols:
        raise ValueError("group-by must contain at least one column name.")
    return cols


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute fair category-wise noise comparison from Doc29 outputs.")
    parser.add_argument("--summary", required=True, help="Path to summary_mse.csv from run_doc29_experiment.py.")
    parser.add_argument("--out", required=True, help="Output directory for comparison files.")
    parser.add_argument(
        "--group-by",
        default="A/D,Runway,aircraft_type",
        help="Comma-separated category columns from summary (default: A/D,Runway,aircraft_type).",
    )
    parser.add_argument("--tracks-per-cluster", type=int, default=7, help="Used only for unweighted subtracks mode.")
    parser.add_argument(
        "--subtracks-weighting",
        choices=["weighted", "unweighted"],
        default="weighted",
        help=(
            "'weighted': Flight_subtracks.csv already encodes flight counts in Nr.day; "
            "'unweighted': each track treated as 1 and scaled by n_flights/tracks_per_cluster."
        ),
    )
    args = parser.parse_args()

    summary_path = Path(args.summary)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    group_cols = _parse_group_by(args.group_by)

    summary = pd.read_csv(summary_path)
    required = set(group_cols) | {"n_flights", "subtracks_csv", "groundtruth_csv"}
    if not required.issubset(summary.columns):
        missing = sorted(required - set(summary.columns))
        raise ValueError(f"summary CSV missing columns: {missing}")

    pred_by_group: Dict[Tuple[object, ...], Optional[pd.DataFrame]] = {}
    gt_by_group: Dict[Tuple[object, ...], Optional[pd.DataFrame]] = {}
    overall_pred: Optional[pd.DataFrame] = None
    overall_gt: Optional[pd.DataFrame] = None
    skipped_rows = 0

    for _, row in summary.iterrows():
        sub_path = Path(str(row["subtracks_csv"]))
        gt_path = Path(str(row["groundtruth_csv"]))
        if not sub_path.exists() or not gt_path.exists():
            skipped_rows += 1
            continue

        n_flights = int(row["n_flights"])
        if args.subtracks_weighting == "weighted":
            scale = 1.0
        else:
            scale = n_flights / float(args.tracks_per_cluster)

        sub_df = _load_doc29_cumulative(sub_path)
        gt_df = _load_doc29_cumulative(gt_path)
        key = tuple(row[col] for col in group_cols)

        pred_by_group[key] = _accumulate_energy(pred_by_group.get(key), sub_df, scale=scale)
        gt_by_group[key] = _accumulate_energy(gt_by_group.get(key), gt_df, scale=1.0)
        overall_pred = _accumulate_energy(overall_pred, sub_df, scale=scale)
        overall_gt = _accumulate_energy(overall_gt, gt_df, scale=1.0)

    category_rows: List[Dict[str, object]] = []
    aligned_frames: List[pd.DataFrame] = []
    all_keys = sorted(set(pred_by_group.keys()) | set(gt_by_group.keys()))
    for key in all_keys:
        pred_energy = pred_by_group.get(key)
        gt_energy = gt_by_group.get(key)
        if pred_energy is None or gt_energy is None:
            continue
        aligned, metrics = _compare_energy_maps(pred_energy, gt_energy)
        row = {col: key[idx] for idx, col in enumerate(group_cols)}
        row.update(metrics)
        category_rows.append(row)
        for idx, col in enumerate(group_cols):
            aligned[col] = key[idx]
        aligned_frames.append(aligned)

    if not category_rows:
        raise RuntimeError("No comparable categories found. Check summary paths and grouping columns.")

    category_summary = pd.DataFrame(category_rows).sort_values(group_cols).reset_index(drop=True)
    category_summary_path = out_dir / "category_summary.csv"
    category_summary.to_csv(category_summary_path, index=False)

    aligned_all = pd.concat(aligned_frames, ignore_index=True)
    aligned_all = annotate_measuring_points(aligned_all)
    aligned_cols = group_cols + [
        "L_eq_pred",
        "L_eq_cluster",
        "delta_cluster",
        "abs_err_cluster",
    ]
    aligned_cols = ["measuring_point"] + aligned_cols
    aligned_path = out_dir / "category_aligned_receivers.csv"
    aligned_all[aligned_cols].to_csv(aligned_path, index=False)

    if overall_pred is None or overall_gt is None:
        raise RuntimeError("No overall aggregates produced. Check input summary.")
    overall_aligned, overall_metrics = _compare_energy_maps(overall_pred, overall_gt)
    overall_aligned = annotate_measuring_points(overall_aligned)
    overall_aligned = overall_aligned[
        ["measuring_point", "L_eq_pred", "L_eq_cluster", "delta_cluster", "abs_err_cluster"]
    ].copy()
    overall_aligned_path = out_dir / "overall_aligned_9points.csv"
    overall_aligned.to_csv(overall_aligned_path, index=False)

    overall_summary = {
        "summary_csv": str(summary_path),
        "group_by": group_cols,
        "subtracks_weighting": args.subtracks_weighting,
        "tracks_per_cluster": args.tracks_per_cluster,
        "skipped_rows": int(skipped_rows),
        "n_categories": int(len(category_summary)),
        "overall_aligned_receivers_csv": str(overall_aligned_path),
        **overall_metrics,
    }
    overall_path = out_dir / "overall_summary.json"
    overall_path.write_text(json.dumps(overall_summary, indent=2), encoding="utf-8")

    print(f"Wrote {category_summary_path}")
    print(f"Wrote {aligned_path}")
    print(f"Wrote {overall_aligned_path}")
    print(f"Wrote {overall_path}")


if __name__ == "__main__":
    main()
