"""Rank DTW pilot runs and optionally promote winners into EXP021-EXP024.

Composite score:
  0.35*z(silhouette) - 0.25*z(DB) + 0.25*z(CH) - 0.15*z(noise_frac)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml


TARGET_MAP = {
    "hdbscan": "EXP021",
    "dbscan": "EXP022",
    "optics": "EXP023",
    "kmeans": "EXP024",
}

METHOD_KEYS = [
    "optics",
    "dbscan",
    "hdbscan",
    "kmeans",
    "minibatch_kmeans",
    "agglomerative",
    "birch",
    "gmm",
    "meanshift",
    "affinity_propagation",
    "two_stage",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank DTW pilot runs and promote winners.")
    parser.add_argument("--output-root", default="output/experiments", help="Experiments output root.")
    parser.add_argument("--exp-start", type=int, default=89, help="Pilot start experiment number.")
    parser.add_argument("--exp-end", type=int, default=100, help="Pilot end experiment number.")
    parser.add_argument("--outdir", default="output/eda/dtw_dense_pilot_ranking", help="Ranking output directory.")
    parser.add_argument(
        "--target-grid",
        default="experiments/experiment_grid.yaml",
        help="Grid file to rewrite EXP021-EXP024 when --apply-promotion is set.",
    )
    parser.add_argument(
        "--apply-promotion",
        action="store_true",
        help="Rewrite EXP021-EXP024 in target grid using selected winners.",
    )
    return parser.parse_args()


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(mask):
        return float("nan")
    return float(np.average(values[mask], weights=weights[mask]))


def _zscore(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    mu = float(s.mean(skipna=True))
    sigma = float(s.std(skipna=True, ddof=0))
    if not np.isfinite(sigma) or sigma == 0:
        return pd.Series(np.zeros(len(s), dtype=float), index=s.index)
    return (s - mu) / sigma


def _load_run_summary(exp_dir: Path) -> dict | None:
    by_flow_path = exp_dir / "metrics_by_flow.csv"
    cfg_path = exp_dir / "config_resolved.yaml"
    if not by_flow_path.exists() or not cfg_path.exists():
        return None

    by_flow = pd.read_csv(by_flow_path)
    if by_flow.empty:
        return None
    weights = pd.to_numeric(by_flow.get("n_flights_used_for_fit"), errors="coerce").fillna(0).to_numpy(dtype=float)
    silhouette = pd.to_numeric(by_flow.get("silhouette"), errors="coerce").to_numpy(dtype=float)
    db = pd.to_numeric(by_flow.get("davies_bouldin"), errors="coerce").to_numpy(dtype=float)
    ch = pd.to_numeric(by_flow.get("calinski_harabasz"), errors="coerce").to_numpy(dtype=float)
    noise = pd.to_numeric(by_flow.get("noise_frac"), errors="coerce").to_numpy(dtype=float)
    n_clusters = pd.to_numeric(by_flow.get("n_clusters"), errors="coerce").fillna(0).to_numpy(dtype=float)

    valid_cluster_flows = int(np.sum(n_clusters >= 2))
    flow_count = int(len(by_flow))
    noise_weighted = _weighted_mean(noise, weights)
    eligible = bool(valid_cluster_flows >= 6 and np.isfinite(noise_weighted) and noise_weighted <= 0.45)

    with cfg_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    clustering_cfg = cfg.get("clustering", {}) or {}
    method = str(clustering_cfg.get("method", "")).lower()

    return {
        "experiment_name": exp_dir.name,
        "method": method,
        "n_flows": flow_count,
        "flows_with_ge2_clusters": valid_cluster_flows,
        "noise_frac": noise_weighted,
        "silhouette": _weighted_mean(silhouette, weights),
        "davies_bouldin": _weighted_mean(db, weights),
        "calinski_harabasz": _weighted_mean(ch, weights),
        "eligible": eligible,
        "config_resolved_path": str(cfg_path),
    }


def _extract_promotion_payload(exp_dir: Path) -> dict:
    cfg_path = exp_dir / "config_resolved.yaml"
    with cfg_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    clustering_cfg = cfg.get("clustering", {}) or {}
    method = str(clustering_cfg.get("method", "")).lower()
    return {
        "method": method,
        "distance_metric": "dtw",
        "distance_params": dict(clustering_cfg.get("distance_params", {}) or {}),
        "method_params": dict(clustering_cfg.get(method, {}) or {}),
        "input": dict(cfg.get("input", {}) or {}),
        "features": dict(cfg.get("features", {}) or {}),
        "evaluation": dict(clustering_cfg.get("evaluation", {}) or {}),
    }


def _apply_promotion_to_grid(
    target_grid_path: Path,
    winners: Dict[str, dict],
) -> None:
    with target_grid_path.open("r", encoding="utf-8") as fh:
        grid = yaml.safe_load(fh) or {}
    experiments: List[dict] = grid.get("experiments", [])

    exp_index: Dict[str, int] = {}
    for i, exp in enumerate(experiments):
        name = str(exp.get("experiment_name") or exp.get("name") or "")
        if name:
            exp_index[name] = i

    for method, target_exp in TARGET_MAP.items():
        if method not in winners:
            raise ValueError(f"Missing winner for method={method}; cannot promote {target_exp}.")
        if target_exp not in exp_index:
            raise ValueError(f"Target experiment {target_exp} not found in {target_grid_path}.")

        idx = exp_index[target_exp]
        entry = dict(experiments[idx])
        payload = winners[method]

        # Remove legacy algo parameter blocks to avoid ambiguity.
        for key in METHOD_KEYS:
            entry.pop(key, None)

        entry["name"] = target_exp
        entry["experiment_name"] = target_exp
        entry["method"] = payload["method"]
        entry["distance_metric"] = "dtw"
        entry["distance_params"] = payload["distance_params"]
        entry[payload["method"]] = payload["method_params"]
        entry["input"] = payload["input"]
        entry["features"] = payload["features"]
        entry["evaluation"] = payload["evaluation"]
        # Full DTW final runs: disable fit sampling.
        entry["sample_for_fit"] = {
            "enabled": False,
            "max_flights_per_flow": 900,
            "random_state": 11,
            "mode": "sample_only",
        }

        experiments[idx] = entry

    grid["experiments"] = experiments
    with target_grid_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(grid, fh, sort_keys=False)


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    output_root = Path(args.output_root)
    rows: List[dict] = []
    for exp_num in range(int(args.exp_start), int(args.exp_end) + 1):
        exp_name = f"EXP{exp_num:03d}"
        exp_dir = output_root / exp_name
        row = _load_run_summary(exp_dir)
        if row is not None:
            rows.append(row)

    if not rows:
        raise RuntimeError("No pilot experiments found with metrics_by_flow.csv + config_resolved.yaml.")

    df = pd.DataFrame(rows).sort_values("experiment_name").reset_index(drop=True)
    df["z_silhouette"] = _zscore(df["silhouette"])
    df["z_db"] = _zscore(df["davies_bouldin"])
    df["z_ch"] = _zscore(df["calinski_harabasz"])
    df["z_noise"] = _zscore(df["noise_frac"])
    df["composite_score"] = (
        0.35 * df["z_silhouette"]
        - 0.25 * df["z_db"]
        + 0.25 * df["z_ch"]
        - 0.15 * df["z_noise"]
    )

    df_rank = df.sort_values(["eligible", "composite_score"], ascending=[False, False]).reset_index(drop=True)
    df_rank.to_csv(outdir / "pilot_ranking.csv", index=False)

    winners_rows = []
    winners_payload: Dict[str, dict] = {}
    for method in ["kmeans", "dbscan", "hdbscan", "optics"]:
        subset = df_rank[df_rank["method"] == method].copy()
        if subset.empty:
            continue
        eligible_subset = subset[subset["eligible"] == True]
        chosen = eligible_subset.iloc[0] if not eligible_subset.empty else subset.iloc[0]
        exp_name = str(chosen["experiment_name"])
        exp_dir = output_root / exp_name
        payload = _extract_promotion_payload(exp_dir)
        winners_payload[method] = payload
        winners_rows.append(
            {
                "method": method,
                "winner_experiment": exp_name,
                "eligible": bool(chosen["eligible"]),
                "composite_score": float(chosen["composite_score"]),
                "promote_to": TARGET_MAP[method],
            }
        )

    winners_df = pd.DataFrame(winners_rows).sort_values("method")
    winners_df.to_csv(outdir / "winners.csv", index=False)
    (outdir / "winners_payload.json").write_text(json.dumps(winners_payload, indent=2), encoding="utf-8")

    if args.apply_promotion:
        _apply_promotion_to_grid(Path(args.target_grid), winners_payload)

    print(f"Saved ranking: {outdir / 'pilot_ranking.csv'}", flush=True)
    print(f"Saved winners: {outdir / 'winners.csv'}", flush=True)
    if args.apply_promotion:
        print(f"Updated grid with promotions: {args.target_grid}", flush=True)


if __name__ == "__main__":
    main()
