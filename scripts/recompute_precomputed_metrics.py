"""Recompute internal metrics for precomputed DTW/Frechet runs from cached distances.

This is a post-hoc utility to fill silhouette/DB/CH when the original run
reported `reason=sparse_precomputed_distances`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import List

import numpy as np
import pandas as pd
from scipy.sparse import load_npz

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clustering.evaluation import compute_internal_metrics


def _hash_config(flow_name: str, metric: str, params: dict, cache_ids: List[int]) -> str:
    payload = json.dumps(
        {"flow": flow_name, "metric": metric, "params": params, "flight_ids": cache_ids},
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _flow_from_label_path(path: Path) -> str:
    name = path.stem
    if name.startswith("labels_"):
        return name.replace("labels_", "", 1)
    return name


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(mask):
        return float("nan")
    return float(np.average(values[mask], weights=weights[mask]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recompute DTW/Frechet internal metrics from cached distances.")
    parser.add_argument("--experiment", required=True, help="Experiment name (e.g., EXP023).")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: output/experiments/<EXP>/recomputed_metrics).",
    )
    parser.add_argument("--embed-components", type=int, default=3, help="MDS components for DB/CH.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exp_dir = Path("output") / "experiments" / args.experiment
    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment folder not found: {exp_dir}")

    cfg_path = exp_dir / "config_resolved.yaml"
    runtime_log = exp_dir / "runtime_log.txt"
    cache_dir = exp_dir / "cache"
    if not (cfg_path.exists() and runtime_log.exists() and cache_dir.exists()):
        raise FileNotFoundError("Missing config_resolved.yaml, runtime_log.txt, or cache directory.")

    cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.suffix == ".json" else None
    if cfg is None:
        import yaml

        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}

    clustering_cfg = cfg.get("clustering", {}) or {}
    metric = str(clustering_cfg.get("distance_metric", "euclidean")).lower()
    if metric not in {"dtw", "frechet"}:
        raise ValueError(f"Only dtw/frechet supported for this tool; got {metric}.")
    method = str(clustering_cfg.get("method", "optics")).lower()
    params = {
        "distance_metric": metric,
        "n_points": (cfg.get("preprocessing", {}) or {}).get("resampling", {}).get("n_points"),
    }
    params.update(dict(clustering_cfg.get("distance_params", {}) or {}))
    if method == "optics":
        min_req = int((clustering_cfg.get("optics", {}) or {}).get("min_samples", 5))
        params["min_required_neighbors"] = max(1, min_req)

    runtime = json.loads(runtime_log.read_text(encoding="utf-8"))
    label_paths = [Path(p) for p in runtime.get("labels", [])]
    if not label_paths:
        raise ValueError("No label paths found in runtime_log.txt.")

    out_dir = Path(args.output_dir) if args.output_dir else exp_dir / "recomputed_metrics"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for label_path in label_paths:
        flow_name = _flow_from_label_path(label_path)
        labels_df = pd.read_csv(label_path)
        if "cluster_id" not in labels_df.columns or "flight_id" not in labels_df.columns:
            raise ValueError(f"Missing columns in {label_path}")
        labels = labels_df["cluster_id"].to_numpy()
        flight_ids = labels_df["flight_id"].astype(int).tolist()

        key = _hash_config(flow_name, metric, params, flight_ids)
        cache_path = cache_dir / f"dist_{key}.npz"
        if not cache_path.exists():
            cache_ids = list(range(len(flight_ids)))
            key = _hash_config(flow_name, metric, params, cache_ids)
            cache_path = cache_dir / f"dist_{key}.npz"
        if not cache_path.exists():
            raise FileNotFoundError(f"Cache not found for {flow_name}: {cache_path}")

        D_sparse = load_npz(cache_path)
        D = D_sparse.toarray()
        D = 0.5 * (D + D.T)
        np.fill_diagonal(D, 0.0)

        metrics = compute_internal_metrics(
            D,
            labels,
            metric_mode="precomputed",
            include_noise=False,
            sparse_precomputed_policy="dense_if_small",
            sparse_precomputed_max_n=10_000,
            precomputed_embed_for_dbch=True,
            precomputed_embed_components=int(args.embed_components),
        )
        metrics.update(
            {
                "flow": flow_name,
                "n_flights": int(len(labels)),
                "n_noise_flights": int(np.sum(labels == -1)),
                "n_clustered_flights": int(np.sum(labels != -1)),
            }
        )
        rows.append(metrics)

    df = pd.DataFrame(rows).sort_values("flow").reset_index(drop=True)
    df.to_csv(out_dir / "metrics_by_flow_recomputed.csv", index=False)

    weights = df["n_flights"].to_numpy(dtype=float)
    global_row = {
        "total_flights": int(df["n_flights"].sum()),
        "total_noise_flights": int(df["n_noise_flights"].sum()),
        "total_clustered_flights": int(df["n_clustered_flights"].sum()),
        "noise_frac": float(df["n_noise_flights"].sum() / df["n_flights"].sum()),
        "silhouette": _weighted_mean(df["silhouette"].to_numpy(dtype=float), weights),
        "davies_bouldin": _weighted_mean(df["davies_bouldin"].to_numpy(dtype=float), weights),
        "calinski_harabasz": _weighted_mean(df["calinski_harabasz"].to_numpy(dtype=float), weights),
    }
    pd.DataFrame([global_row]).to_csv(out_dir / "metrics_global_recomputed.csv", index=False)

    print(f"Wrote: {out_dir / 'metrics_by_flow_recomputed.csv'}", flush=True)
    print(f"Wrote: {out_dir / 'metrics_global_recomputed.csv'}", flush=True)


if __name__ == "__main__":
    main()
