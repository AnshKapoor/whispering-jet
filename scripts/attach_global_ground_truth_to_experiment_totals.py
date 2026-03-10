"""Attach all-flight ground truth to experiment aggregate-total outputs.

This utility runs on top of existing `aggregate_totals` folders created by
`noise_simulation/compare_experiment_totals.py`. It enriches each
`overall_aligned_9points.csv` with a second ground-truth reference taken from a
global all-flight ground-truth run, then updates `overall_summary.json` with
metrics against that full-flight reference.

Usage:
  python scripts/attach_global_ground_truth_to_experiment_totals.py \
    --ground-truth-csv noise_simulation/results_ground_truth/preprocessed_1_final/ground_truth_cumulative.csv \
    --start 1 --end 62
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _to_energy(level_db: np.ndarray) -> np.ndarray:
    return np.power(10.0, level_db / 10.0)


def _to_db(energy: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(energy, 1e-12))


def _load_csv_auto(path: Path) -> pd.DataFrame:
    """Load a CSV while tolerating comma or semicolon separators."""
    return pd.read_csv(path, sep=None, engine="python")


def _prepare_ground_truth(df: pd.DataFrame) -> pd.DataFrame:
    needed = {"cumulative_res"}
    if not needed.issubset(df.columns):
        raise ValueError(f"Ground-truth CSV missing columns: {sorted(needed - set(df.columns))}")

    out = df.copy()
    if "measuring_point" not in out.columns:
        key_cols = {"x", "y", "z"}
        if not key_cols.issubset(out.columns):
            raise ValueError("Ground-truth CSV needs either measuring_point or x/y/z columns.")
        from noise_simulation.receiver_points import annotate_measuring_points

        out = annotate_measuring_points(out)

    out = out.rename(columns={"cumulative_res": "cumulative_res_gt_all_flights"})
    keep_cols = ["measuring_point", "cumulative_res_gt_all_flights"]
    if "x" in out.columns:
        keep_cols.append("x")
    if "y" in out.columns:
        keep_cols.append("y")
    if "z" in out.columns:
        keep_cols.append("z")
    return out[keep_cols].copy()


def _log_energy_average(levels_db: np.ndarray) -> float:
    return float(_to_db(np.array([np.mean(_to_energy(levels_db))]))[0])


def _compute_all_flights_metrics(aligned: pd.DataFrame) -> Dict[str, float]:
    diff = aligned["delta_all_flights"].to_numpy()
    pred = aligned["cumulative_res_pred"].to_numpy()
    gt = aligned["cumulative_res_gt_all_flights"].to_numpy()
    mse = float(np.mean(np.square(diff)))
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(mse))
    return {
        "avg_cumulative_res_pred": _log_energy_average(pred),
        "avg_cumulative_res_gt_all_flights": _log_energy_average(gt),
        "delta_avg_cumulative_res_all_flights": float(_log_energy_average(pred) - _log_energy_average(gt)),
        "mae_cumulative_res_all_flights": mae,
        "mse_cumulative_res_all_flights": mse,
        "rmse_cumulative_res_all_flights": rmse,
        "n_receivers_all_flights": int(len(aligned)),
    }


def _enrich_overall_csv(overall_path: Path, gt_df: pd.DataFrame) -> Dict[str, float]:
    overall = _load_csv_auto(overall_path)
    if "measuring_point" not in overall.columns:
        from noise_simulation.receiver_points import annotate_measuring_points

        overall = annotate_measuring_points(overall)

    # Allow safe re-runs by dropping any previously attached all-flight columns.
    stale_cols = [
        "cumulative_res_gt_all_flights",
        "delta_all_flights",
        "abs_err_all_flights",
        "sq_err_all_flights",
    ]
    overall = overall.drop(columns=[col for col in stale_cols if col in overall.columns], errors="ignore")

    required = {"measuring_point", "cumulative_res_pred"}
    if not required.issubset(overall.columns):
        raise ValueError(f"{overall_path} missing columns: {sorted(required - set(overall.columns))}")

    aligned = overall.merge(
        gt_df[["measuring_point", "cumulative_res_gt_all_flights"]],
        on="measuring_point",
        how="left",
        validate="one_to_one",
    )
    if aligned["cumulative_res_gt_all_flights"].isna().any():
        missing = aligned.loc[aligned["cumulative_res_gt_all_flights"].isna(), "measuring_point"].tolist()
        raise ValueError(f"{overall_path} missing global ground truth for: {missing}")

    aligned["delta_all_flights"] = (
        aligned["cumulative_res_pred"] - aligned["cumulative_res_gt_all_flights"]
    )
    aligned["abs_err_all_flights"] = np.abs(aligned["delta_all_flights"])
    aligned["sq_err_all_flights"] = np.square(aligned["delta_all_flights"])

    preferred_order = [
        "measuring_point",
        "x",
        "y",
        "z",
        "cumulative_res_pred",
        "cumulative_res_gt",
        "cumulative_res_gt_all_flights",
        "delta",
        "delta_all_flights",
        "abs_err",
        "abs_err_all_flights",
        "sq_err",
        "sq_err_all_flights",
    ]
    ordered_cols = [col for col in preferred_order if col in aligned.columns] + [
        col for col in aligned.columns if col not in preferred_order
    ]
    aligned = aligned[ordered_cols].copy()
    aligned.to_csv(overall_path, index=False)
    return _compute_all_flights_metrics(aligned)


def _update_summary_json(summary_path: Path, gt_csv: Path, metrics: Dict[str, float]) -> None:
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        summary = {}
    summary["global_ground_truth_csv"] = str(gt_csv.resolve())
    summary.update(metrics)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _iter_experiment_dirs(results_root: Path, start: int, end: int) -> Iterable[Path]:
    for n in range(start, end + 1):
        yield results_root / f"EXP{n:03d}"


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

    results_root = (REPO_ROOT / args.results_root).resolve()
    gt_csv = Path(args.ground_truth_csv)
    if not gt_csv.is_absolute():
        gt_csv = (REPO_ROOT / gt_csv).resolve()
    if not gt_csv.exists():
        raise FileNotFoundError(f"Global ground-truth CSV not found: {gt_csv}")

    gt_df = _prepare_ground_truth(_load_csv_auto(gt_csv))
    updated = 0
    skipped: List[str] = []

    for exp_dir in _iter_experiment_dirs(results_root, args.start, args.end):
        overall_path = exp_dir / "aggregate_totals" / "overall_aligned_9points.csv"
        summary_path = exp_dir / "aggregate_totals" / "overall_summary.json"
        if not overall_path.exists():
            skipped.append(exp_dir.name)
            continue

        metrics = _enrich_overall_csv(overall_path, gt_df)
        _update_summary_json(summary_path, gt_csv, metrics)
        updated += 1
        print(f"Updated {overall_path}")

    print(f"Done. updated={updated} skipped={len(skipped)}")
    if skipped:
        print("Skipped experiments without aggregate_totals/overall_aligned_9points.csv:")
        print(", ".join(skipped))


if __name__ == "__main__":
    main()
