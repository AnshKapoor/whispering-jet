"""Attach global all-flight ground truth to experiment aggregate totals.

Usage:
  python scripts/attach_global_ground_truth_to_experiment_totals.py \
    --ground-truth-csv noise_simulation/results_ground_truth/preprocessed_1_final/ground_truth_cumulative.csv \
    --start 1 --end 62

Behavior:
  - Adds cumulative_res_gt_all_flights and error columns to
    noise_simulation/results/EXP###/aggregate_totals/overall_aligned_9points.csv
  - Updates overall_summary.json with all-flights metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


def _to_energy(level_db: np.ndarray) -> np.ndarray:
    return np.power(10.0, level_db / 10.0)


def _to_db(energy: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(energy, 1e-12))


def _avg_db(level_db: np.ndarray) -> float:
    if level_db.size == 0:
        return float("nan")
    return float(_to_db(np.array([np.mean(_to_energy(level_db))]))[0])


def _exp_name(exp_num: int) -> str:
    return f"EXP{exp_num:03d}"


def _merge_keys(overall: pd.DataFrame, gt: pd.DataFrame) -> List[str]:
    if {"x", "y", "z"}.issubset(overall.columns) and {"x", "y", "z"}.issubset(gt.columns):
        return ["x", "y", "z"]
    if "measuring_point" in overall.columns and "measuring_point" in gt.columns:
        return ["measuring_point"]
    raise ValueError("No compatible join keys between overall_aligned_9points.csv and ground truth CSV.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Attach global all-flight ground truth to existing experiment aggregate totals."
    )
    parser.add_argument(
        "--ground-truth-csv",
        required=True,
        help="Path to the global all-flight ground_truth_cumulative.csv.",
    )
    parser.add_argument(
        "--results-root",
        default="noise_simulation/results",
        help="Root directory containing EXP### folders.",
    )
    parser.add_argument("--start", type=int, default=1, help="Start experiment number (default: 1).")
    parser.add_argument("--end", type=int, default=62, help="End experiment number (default: 62).")
    args = parser.parse_args()

    gt_path = Path(args.ground_truth_csv).resolve()
    if not gt_path.exists():
        raise FileNotFoundError(f"Global ground-truth CSV not found: {gt_path}")

    gt_df = pd.read_csv(gt_path)
    if "cumulative_res" not in gt_df.columns:
        raise ValueError("ground_truth_cumulative.csv missing cumulative_res column.")

    gt_df = gt_df.rename(columns={"cumulative_res": "cumulative_res_gt_all_flights"})

    results_root = Path(args.results_root).resolve()
    updated: List[str] = []
    skipped: List[str] = []

    for exp_num in range(args.start, args.end + 1):
        exp = _exp_name(exp_num)
        agg_dir = results_root / exp / "aggregate_totals"
        overall_path = agg_dir / "overall_aligned_9points.csv"
        summary_path = agg_dir / "overall_summary.json"
        if not overall_path.exists():
            skipped.append(exp)
            continue

        overall = pd.read_csv(overall_path)
        join_cols = _merge_keys(overall, gt_df)
        merged = overall.merge(
            gt_df[join_cols + ["cumulative_res_gt_all_flights"]],
            on=join_cols,
            how="inner",
        )
        if merged.empty:
            skipped.append(exp)
            continue

        merged["delta_all_flights"] = (
            merged["cumulative_res_pred"] - merged["cumulative_res_gt_all_flights"]
        )
        merged["abs_err_all_flights"] = merged["delta_all_flights"].abs()
        merged["sq_err_all_flights"] = merged["delta_all_flights"] ** 2

        if {"x", "y", "z"}.issubset(merged.columns):
            merged = merged.sort_values(["x", "y", "z"]).reset_index(drop=True)
        else:
            merged = merged.sort_values(join_cols).reset_index(drop=True)

        merged.to_csv(overall_path, index=False)

        summary = {}
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))

        avg_pred = _avg_db(merged["cumulative_res_pred"].to_numpy(dtype=float))
        avg_gt_all = _avg_db(merged["cumulative_res_gt_all_flights"].to_numpy(dtype=float))
        mse_all = float(merged["sq_err_all_flights"].mean())
        mae_all = float(merged["abs_err_all_flights"].mean())

        summary.update(
            {
                "global_ground_truth_csv": str(gt_path),
                "avg_cumulative_res_gt_all_flights": avg_gt_all,
                "delta_avg_cumulative_res_all_flights": float(avg_pred - avg_gt_all),
                "mae_cumulative_res_all_flights": mae_all,
                "mse_cumulative_res_all_flights": mse_all,
                "rmse_cumulative_res_all_flights": float(np.sqrt(mse_all)),
                "n_receivers_all_flights": int(len(merged)),
            }
        )
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        updated.append(exp)
        print(f"Updated {overall_path}")

    print(f"Done. updated={len(updated)} skipped={len(skipped)}")
    if skipped:
        print(
            "Skipped experiments without aggregate_totals/overall_aligned_9points.csv: "
            + ", ".join(skipped)
        )


if __name__ == "__main__":
    main()
