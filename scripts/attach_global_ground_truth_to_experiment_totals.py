"""Attach all-flights ground truth to experiment aggregate totals.

Usage:
  python scripts/attach_global_ground_truth_to_experiment_totals.py \
    --ground-truth-root noise_simulation/results_ground_truth \
    --start 5 --end 20

Behavior:
  - Resolves the matching preprocessed-specific all-flights ground truth per experiment.
  - Updates `overall_aligned_9points.csv` with the final slim receiver-level columns:
      measuring_point, L_eq_pred, L_eq_cluster, delta_cluster, abs_err_cluster,
      L_eq_ground_truth, delta_ground_truth, abs_err_ground_truth
  - Updates `overall_summary.json` with only the requested aggregate metrics:
      MAE against clustered and all-flights ground truth.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import pandas as pd
import yaml

import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from noise_simulation.receiver_points import annotate_measuring_points


def _exp_name(exp_num: int) -> str:
    return f"EXP{exp_num:03d}"


def _read_csv_auto(path: Path) -> pd.DataFrame:
    """Read CSV with auto-detected delimiter for ground-truth exports."""

    return pd.read_csv(path, sep=None, engine="python")


def _merge_keys(overall: pd.DataFrame, gt: pd.DataFrame) -> List[str]:
    if "measuring_point" in overall.columns and "measuring_point" in gt.columns:
        return ["measuring_point"]
    if {"x", "y", "z"}.issubset(overall.columns) and {"x", "y", "z"}.issubset(gt.columns):
        return ["x", "y", "z"]
    raise ValueError("No compatible join keys between overall_aligned_9points.csv and ground truth CSV.")


def _load_experiment_preprocessed_stem(exp_dir: Path) -> str:
    """Return the preprocessed CSV stem declared in config_resolved.yaml."""

    cfg_path = exp_dir / "config_resolved.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing experiment config: {cfg_path}")
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    preprocessed_rel = (
        cfg.get("input", {}).get("preprocessed_csv") if isinstance(cfg.get("input"), dict) else None
    )
    if not preprocessed_rel:
        raise ValueError(f"Missing input.preprocessed_csv in {cfg_path}")
    return Path(str(preprocessed_rel)).stem


def _resolve_ground_truth_csv(
    exp_dir: Path,
    ground_truth_root: Path,
    fallback_path: Path | None,
) -> Path:
    """Resolve the all-flights ground truth CSV for one experiment."""

    stem = _load_experiment_preprocessed_stem(exp_dir)
    candidate = ground_truth_root / f"{stem}_final" / "ground_truth_cumulative.csv"
    if candidate.exists():
        return candidate
    if fallback_path is not None and fallback_path.exists():
        return fallback_path
    raise FileNotFoundError(
        f"Missing ground truth for {exp_dir.name}: expected {candidate}"
        + (f" and fallback {fallback_path}" if fallback_path is not None else "")
    )


def _normalize_ground_truth(gt_path: Path) -> pd.DataFrame:
    """Return ground truth with measuring_point and L_eq_ground_truth columns."""

    gt_df = _read_csv_auto(gt_path)
    if "cumulative_res" not in gt_df.columns:
        raise ValueError(f"{gt_path} missing cumulative_res column.")
    if "measuring_point" not in gt_df.columns:
        gt_df = annotate_measuring_points(gt_df)
    gt_df = gt_df.rename(columns={"cumulative_res": "L_eq_ground_truth"})
    return gt_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Attach all-flights ground truth to existing experiment aggregate totals."
    )
    parser.add_argument(
        "--ground-truth-root",
        default="noise_simulation/results_ground_truth",
        help="Root directory containing preprocessed_*_final ground-truth folders.",
    )
    parser.add_argument(
        "--fallback-ground-truth-csv",
        default="noise_simulation/results_ground_truth/preprocessed_1_final/ground_truth_cumulative.csv",
        help="Fallback all-flights ground truth CSV used when the exact preprocessed-specific file is missing.",
    )
    parser.add_argument(
        "--results-root",
        default="noise_simulation/results",
        help="Root directory containing EXP### folders.",
    )
    parser.add_argument("--start", type=int, default=1, help="Start experiment number (default: 1).")
    parser.add_argument("--end", type=int, default=62, help="End experiment number (default: 62).")
    args = parser.parse_args()

    ground_truth_root = Path(args.ground_truth_root)
    if not ground_truth_root.is_absolute():
        ground_truth_root = (REPO_ROOT / ground_truth_root).resolve()
    if not ground_truth_root.exists():
        raise FileNotFoundError(f"Ground-truth root not found: {ground_truth_root}")

    fallback_path = Path(args.fallback_ground_truth_csv)
    if not fallback_path.is_absolute():
        fallback_path = (REPO_ROOT / fallback_path).resolve()
    if not fallback_path.exists():
        fallback_path = None

    results_root = Path(args.results_root)
    if not results_root.is_absolute():
        results_root = (REPO_ROOT / results_root).resolve()

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

        exp_output_dir = REPO_ROOT / "output" / "experiments" / exp
        gt_path = _resolve_ground_truth_csv(exp_output_dir, ground_truth_root, fallback_path)
        gt_df = _normalize_ground_truth(gt_path)

        overall = pd.read_csv(overall_path)
        join_cols = _merge_keys(overall, gt_df)
        merged = overall.merge(
            gt_df[join_cols + ["L_eq_ground_truth"]],
            on=join_cols,
            how="inner",
        )
        if merged.empty:
            skipped.append(exp)
            continue

        if "measuring_point" not in merged.columns:
            merged = annotate_measuring_points(merged)

        merged["delta_ground_truth"] = merged["L_eq_pred"] - merged["L_eq_ground_truth"]
        merged["abs_err_ground_truth"] = merged["delta_ground_truth"].abs()
        merged = merged.sort_values("measuring_point").reset_index(drop=True)

        final_cols = [
            "measuring_point",
            "L_eq_pred",
            "L_eq_ground_truth",
            "delta_ground_truth",
            "abs_err_ground_truth",
            "L_eq_cluster",
            "delta_cluster",
            "abs_err_cluster",
        ]
        merged = merged[final_cols]
        merged.to_csv(overall_path, index=False)

        summary = {}
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))

        final_summary = {
            "summary_csv": summary.get("summary_csv"),
            "group_by": summary.get("group_by"),
            "subtracks_weighting": summary.get("subtracks_weighting"),
            "tracks_per_cluster": summary.get("tracks_per_cluster"),
            "skipped_rows": summary.get("skipped_rows"),
            "n_categories": summary.get("n_categories"),
            "overall_aligned_receivers_csv": summary.get("overall_aligned_receivers_csv"),
            "all_flights_ground_truth_csv": str(gt_path),
            "mae_cluster": float(merged["abs_err_cluster"].mean()),
            "mae_ground_truth": float(merged["abs_err_ground_truth"].mean()),
            "n_receivers": int(len(merged)),
        }
        summary_path.write_text(json.dumps(final_summary, indent=2), encoding="utf-8")

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
