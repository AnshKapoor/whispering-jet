"""Probe DBSCAN eps for dense DTW distance matrices on sampled flights per flow.

The script mirrors experiment sampling (`sample_only`) and computes a k-distance
curve per flow (k = min_samples). The knee is estimated as the point with
maximum distance to the line joining the first and last sorted k-distances.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clustering.distances import build_feature_matrix, pairwise_distance_matrix


FLOW_ORDER = [
    ("Start", "09L"),
    ("Start", "09R"),
    ("Start", "27L"),
    ("Start", "27R"),
    ("Landung", "09L"),
    ("Landung", "09R"),
    ("Landung", "27L"),
    ("Landung", "27R"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe DTW eps for DBSCAN via k-distance knee.")
    parser.add_argument("--preprocessed", required=True, help="Path to preprocessed CSV.")
    parser.add_argument("--sample-per-flow", type=int, default=900, help="Sample size per flow.")
    parser.add_argument("--min-samples", type=int, default=8, help="DBSCAN min_samples (k-distance rank).")
    parser.add_argument("--random-state", type=int, default=11, help="Random seed for flow sampling.")
    parser.add_argument("--candidate-k", type=int, default=999999, help="DTW candidate_k for distance build.")
    parser.add_argument("--dtw-window-size", type=int, default=8, help="DTW window size.")
    parser.add_argument(
        "--use-lb-keogh",
        action="store_true",
        help="Enable LB-Keogh pruning (default disabled for dense exact probe).",
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="Output directory (default: output/eda/dtw_dbscan_eps_probe/<preprocessed_stem>_n<sample>_k<k>).",
    )
    return parser.parse_args()


def _sorted_k_distance(D: np.ndarray, k: int) -> np.ndarray:
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("Distance matrix must be square.")
    if D.shape[0] < 3:
        raise ValueError("At least 3 flights required for knee detection.")

    # Exclude self-distance and take kth nearest neighbor distance (k excludes self).
    row_sorted = np.sort(D, axis=1)[:, 1:]
    if k < 1 or k > row_sorted.shape[1]:
        raise ValueError(f"k={k} is out of range for n={D.shape[0]}.")
    kdist = row_sorted[:, k - 1]
    return np.sort(kdist)


def _knee_max_dist_to_line(y: np.ndarray) -> tuple[int, float]:
    if y.ndim != 1 or y.size < 3:
        raise ValueError("Need at least 3 sorted values for knee detection.")
    x = np.arange(y.size, dtype=float)

    x1, y1 = x[0], y[0]
    x2, y2 = x[-1], y[-1]
    denom = np.hypot(y2 - y1, x2 - x1)
    if denom == 0:
        idx = int(y.size // 2)
        return idx, float(y[idx])

    # Distance from each point to line through endpoints.
    distances = np.abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1) / denom
    idx = int(np.argmax(distances))
    return idx, float(y[idx])


def _sample_flow_ids(flow_df: pd.DataFrame, sample_n: int, rng: np.random.Generator) -> Iterable[int]:
    ids = flow_df["flight_id"].drop_duplicates().to_numpy(dtype=int)
    if ids.size <= sample_n:
        return ids.tolist()
    sampled = rng.choice(ids, size=sample_n, replace=False)
    return np.sort(sampled).tolist()


def main() -> None:
    args = parse_args()
    preprocessed_path = Path(args.preprocessed)
    if not preprocessed_path.exists():
        raise FileNotFoundError(f"Preprocessed CSV not found: {preprocessed_path}")

    outdir = Path(args.outdir) if args.outdir else (
        Path("output")
        / "eda"
        / "dtw_dbscan_eps_probe"
        / f"{preprocessed_path.stem}_n{args.sample_per_flow}_k{args.min_samples}"
    )
    outdir.mkdir(parents=True, exist_ok=True)

    usecols = ["flight_id", "step", "x_utm", "y_utm", "A/D", "Runway"]
    df = pd.read_csv(preprocessed_path, usecols=usecols)
    rng = np.random.default_rng(args.random_state)

    rows = []
    params = {
        "candidate_k": int(args.candidate_k),
        "dtw_window_size": int(args.dtw_window_size),
        "use_lb_keogh": bool(args.use_lb_keogh),
    }

    for ad, runway in FLOW_ORDER:
        flow_df = df[(df["A/D"] == ad) & (df["Runway"] == runway)].copy()
        if flow_df.empty:
            continue

        flow_ids = _sample_flow_ids(flow_df, int(args.sample_per_flow), rng)
        flow_fit = flow_df[flow_df["flight_id"].isin(flow_ids)].copy()
        X, trajs = build_feature_matrix(flow_fit, vector_cols=["x_utm", "y_utm"], allow_ragged=True)
        n = X.shape[0]
        if n <= args.min_samples:
            continue

        D = pairwise_distance_matrix(
            trajs,
            metric="dtw",
            cache_dir=outdir / "cache",
            flow_name=f"{ad}_{runway}",
            params=params,
            cache_ids=sorted(flow_ids),
        )
        if hasattr(D, "toarray"):
            D = D.toarray()
        D = np.asarray(D, dtype=float)
        D = 0.5 * (D + D.T)
        np.fill_diagonal(D, 0.0)

        density = float(np.count_nonzero(D) / (n * (n - 1))) if n > 1 else 0.0
        kdist_sorted = _sorted_k_distance(D, int(args.min_samples))
        knee_idx, knee_eps = _knee_max_dist_to_line(kdist_sorted)

        rows.append(
            {
                "flow": f"{ad}_{runway}",
                "n_flights": int(n),
                "matrix_density_offdiag": density,
                "k": int(args.min_samples),
                "knee_index": int(knee_idx),
                "knee_eps": float(knee_eps),
                "kdist_min": float(kdist_sorted[0]),
                "kdist_p50": float(np.percentile(kdist_sorted, 50)),
                "kdist_p75": float(np.percentile(kdist_sorted, 75)),
                "kdist_p90": float(np.percentile(kdist_sorted, 90)),
                "kdist_max": float(kdist_sorted[-1]),
            }
        )

    if not rows:
        raise RuntimeError("No flow had enough sampled flights to compute eps probe.")

    per_flow = pd.DataFrame(rows).sort_values("flow").reset_index(drop=True)
    eps_base = float(np.median(per_flow["knee_eps"].to_numpy(dtype=float)))
    recommendations = {
        "eps_base": eps_base,
        "eps_0_9x": float(0.9 * eps_base),
        "eps_1_0x": float(1.0 * eps_base),
        "eps_1_1x": float(1.1 * eps_base),
    }

    per_flow.to_csv(outdir / "per_flow_knee_eps.csv", index=False)
    summary = {
        "preprocessed": str(preprocessed_path),
        "sample_per_flow": int(args.sample_per_flow),
        "min_samples": int(args.min_samples),
        "distance_params": params,
        "recommendations": recommendations,
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved per-flow knees: {outdir / 'per_flow_knee_eps.csv'}", flush=True)
    print(f"Saved summary: {outdir / 'summary.json'}", flush=True)
    print(
        "Recommended eps values: "
        f"{recommendations['eps_0_9x']:.3f}, {recommendations['eps_1_0x']:.3f}, {recommendations['eps_1_1x']:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
