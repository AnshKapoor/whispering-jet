"""Plot side-by-side backbone comparisons for one flow across two experiments.

Usage:
  python scripts/plot_backbone_side_by_side.py \
    --left EXP016 \
    --right EXP017 \
    --flow Start_27L \
    --top-k 9

Inputs:
  - output/experiments/<EXP>/backbone_tracks.csv if present
  - else labels_*.csv + config_resolved.yaml + preprocessed CSV

Outputs:
  - output/eda/backbone_comparisons/<flow>/<left>_vs_<right>_top<k>.png
  - output/eda/backbone_comparisons/<flow>/<left>_vs_<right>_top<k>_cluster_ranks.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backbone_tracks.backbone import compute_backbones


def _load_config(exp_dir: Path) -> dict:
    cfg_path = exp_dir / "config_resolved.yaml"
    if not cfg_path.exists():
        return {}
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}


def _parse_flow_name(flow_name: str) -> Tuple[str, str]:
    parts = flow_name.split("_", 1)
    if len(parts) != 2:
        raise ValueError(f"Flow must look like Start_27L or Landung_09R, got: {flow_name}")
    return parts[0], parts[1]


def _load_labels_for_flow(exp_dir: Path, flow_name: str) -> pd.DataFrame:
    flow_path = exp_dir / f"labels_{flow_name}.csv"
    cols = ["flight_id", "cluster_id", "A/D", "Runway"]
    if flow_path.exists():
        return pd.read_csv(flow_path, usecols=lambda c: c in cols)

    parts: List[pd.DataFrame] = []
    for path in sorted(exp_dir.glob("labels_*.csv")):
        parts.append(pd.read_csv(path, usecols=lambda c: c in cols))
    if not parts:
        raise FileNotFoundError(f"No labels_*.csv found in {exp_dir}")

    labels = pd.concat(parts, ignore_index=True)
    ad, runway = _parse_flow_name(flow_name)
    if {"A/D", "Runway"}.issubset(labels.columns):
        labels = labels[(labels["A/D"] == ad) & (labels["Runway"] == runway)].copy()
    return labels


def _compute_backbones_for_experiment(exp_dir: Path, flow_name: str) -> pd.DataFrame:
    cfg = _load_config(exp_dir)
    preprocessed_path = (cfg.get("input", {}) or {}).get("preprocessed_csv")
    if not preprocessed_path:
        raise FileNotFoundError(f"Missing input.preprocessed_csv in {exp_dir / 'config_resolved.yaml'}")

    preprocessed_path = Path(preprocessed_path)
    if not preprocessed_path.exists():
        raise FileNotFoundError(f"Preprocessed CSV not found: {preprocessed_path}")

    labels = _load_labels_for_flow(exp_dir, flow_name)
    if labels.empty:
        raise RuntimeError(f"No labels found for {flow_name} in {exp_dir.name}")

    usecols = ["flight_id", "step", "x_utm", "y_utm", "latitude", "longitude", "A/D", "Runway"]
    pre = pd.read_csv(preprocessed_path, usecols=lambda c: c in set(usecols))
    merged = pre.merge(labels[["flight_id", "cluster_id"]], on="flight_id", how="inner")

    percentiles = (cfg.get("backbone", {}) or {}).get("percentiles", [10, 50, 90])
    min_flights = int((cfg.get("backbone", {}) or {}).get("min_flights_per_cluster", 10))
    backbones = compute_backbones(
        merged,
        percentiles=list(percentiles),
        min_flights_per_cluster=min_flights,
        use_utm=False,
    )
    return backbones


def _load_backbones(exp_dir: Path, flow_name: str) -> pd.DataFrame:
    ad, runway = _parse_flow_name(flow_name)
    backbone_csv = exp_dir / "backbone_tracks.csv"
    if backbone_csv.exists():
        df = pd.read_csv(backbone_csv)
    else:
        df = _compute_backbones_for_experiment(exp_dir, flow_name)

    needed = {"A/D", "Runway", "cluster_id", "step", "percentile", "latitude", "longitude", "n_flights"}
    missing = sorted(needed - set(df.columns))
    if missing:
        raise ValueError(f"{exp_dir.name} backbone data missing columns: {missing}")

    df = df[(df["A/D"] == ad) & (df["Runway"] == runway) & (df["cluster_id"] != -1)].copy()
    if df.empty:
        raise RuntimeError(f"No retained backbones found for {flow_name} in {exp_dir.name}")
    return df


def _cluster_palette(cluster_ids: Iterable[int]) -> Dict[int, Tuple[float, float, float, float]]:
    cmap = plt.get_cmap("tab20")
    colors: Dict[int, Tuple[float, float, float, float]] = {}
    for idx, cid in enumerate(sorted(cluster_ids)):
        colors[cid] = cmap(idx % 20)
    return colors


def _lighten(color: Tuple[float, float, float, float], factor: float) -> Tuple[float, float, float, float]:
    rgb = tuple(1 - (1 - c) * factor for c in color[:3])
    return rgb + (1.0,)


def _cluster_ranks(backbones: pd.DataFrame, experiment: str, top_k: int) -> pd.DataFrame:
    ranks = (
        backbones.groupby("cluster_id", as_index=False)["n_flights"]
        .max()
        .sort_values(["n_flights", "cluster_id"], ascending=[False, True])
        .reset_index(drop=True)
    )
    ranks["rank"] = np.arange(1, len(ranks) + 1)
    ranks["experiment"] = experiment
    if top_k > 0:
        ranks = ranks.head(top_k).copy()
    return ranks[["experiment", "rank", "cluster_id", "n_flights"]]


def _bounds(frames: Iterable[pd.DataFrame]) -> Tuple[float, float, float, float]:
    lon = pd.concat([df["longitude"] for df in frames], ignore_index=True)
    lat = pd.concat([df["latitude"] for df in frames], ignore_index=True)
    min_lon, max_lon = float(lon.min()), float(lon.max())
    min_lat, max_lat = float(lat.min()), float(lat.max())
    pad_lon = max((max_lon - min_lon) * 0.08, 0.0025)
    pad_lat = max((max_lat - min_lat) * 0.08, 0.0025)
    return min_lon - pad_lon, max_lon + pad_lon, min_lat - pad_lat, max_lat + pad_lat


def _style_axis(ax: plt.Axes, title: str, bounds: Tuple[float, float, float, float]) -> None:
    x0, x1, y0, y1 = bounds
    ax.set_title(title, fontsize=10)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#e9e9e9", linewidth=0.7)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")


def _plot_all_clusters(ax: plt.Axes, backbones: pd.DataFrame, title: str, bounds: Tuple[float, float, float, float]) -> None:
    palette = _cluster_palette(backbones["cluster_id"].unique())
    for cluster_id, subset in backbones.groupby("cluster_id"):
        p50 = subset[subset["percentile"] == 50].sort_values("step")
        if p50.empty:
            continue
        ax.plot(
            p50["longitude"],
            p50["latitude"],
            color=palette[cluster_id],
            lw=1.8,
            alpha=0.95,
        )
    _style_axis(ax, title, bounds)


def _plot_ranked_cluster(
    ax: plt.Axes,
    backbones: pd.DataFrame,
    rank_row: pd.Series | None,
    title_prefix: str,
    bounds: Tuple[float, float, float, float],
) -> None:
    if rank_row is None:
        ax.axis("off")
        ax.text(0.5, 0.5, "No cluster at this rank", ha="center", va="center", fontsize=10)
        return

    cluster_id = int(rank_row["cluster_id"])
    n_flights = int(rank_row["n_flights"])
    subset = backbones[backbones["cluster_id"] == cluster_id].copy()
    p50 = subset[subset["percentile"] == 50].sort_values("step")
    p10 = subset[subset["percentile"] == 10].sort_values("step")
    p90 = subset[subset["percentile"] == 90].sort_values("step")

    base = plt.get_cmap("tab20")(cluster_id % 20)
    if not p10.empty:
        ax.plot(p10["longitude"], p10["latitude"], color=_lighten(base, 0.6), lw=1.0, ls="--")
    if not p90.empty:
        ax.plot(p90["longitude"], p90["latitude"], color=_lighten(base, 0.6), lw=1.0, ls="--")
    if not p50.empty:
        ax.plot(p50["longitude"], p50["latitude"], color=base, lw=2.0)

    title = f"{title_prefix}: rank {int(rank_row['rank'])}, cluster {cluster_id}, n={n_flights}"
    _style_axis(ax, title, bounds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare one flow's backbones side by side across two experiments.")
    parser.add_argument("--left", required=True, help="Left experiment, e.g. EXP016")
    parser.add_argument("--right", required=True, help="Right experiment, e.g. EXP017")
    parser.add_argument("--flow", required=True, help="Flow like Start_27L or Landung_09L")
    parser.add_argument("--experiments-root", default="output/experiments", help="Experiment root folder")
    parser.add_argument("--output-root", default="output/eda/backbone_comparisons", help="Output folder root")
    parser.add_argument("--top-k", type=int, default=9, help="Maximum number of ranked cluster rows to plot")
    args = parser.parse_args()

    exp_root = Path(args.experiments_root)
    left_dir = exp_root / args.left
    right_dir = exp_root / args.right
    if not left_dir.exists():
        raise FileNotFoundError(f"Experiment folder not found: {left_dir}")
    if not right_dir.exists():
        raise FileNotFoundError(f"Experiment folder not found: {right_dir}")
    if args.top_k < 1:
        raise ValueError("--top-k must be >= 1")

    left_backbones = _load_backbones(left_dir, args.flow)
    right_backbones = _load_backbones(right_dir, args.flow)
    left_ranks = _cluster_ranks(left_backbones, args.left, args.top_k)
    right_ranks = _cluster_ranks(right_backbones, args.right, args.top_k)

    n_rows = 1 + max(len(left_ranks), len(right_ranks))
    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=2,
        figsize=(14, max(4.5 * n_rows, 8)),
        constrained_layout=True,
    )
    if n_rows == 1:
        axes = np.array([axes])

    bounds = _bounds([left_backbones, right_backbones])
    _plot_all_clusters(
        axes[0, 0],
        left_backbones,
        f"{args.left} {args.flow}: all retained backbones",
        bounds,
    )
    _plot_all_clusters(
        axes[0, 1],
        right_backbones,
        f"{args.right} {args.flow}: all retained backbones",
        bounds,
    )

    for row_idx in range(1, n_rows):
        left_rank = left_ranks.iloc[row_idx - 1] if row_idx - 1 < len(left_ranks) else None
        right_rank = right_ranks.iloc[row_idx - 1] if row_idx - 1 < len(right_ranks) else None
        _plot_ranked_cluster(axes[row_idx, 0], left_backbones, left_rank, args.left, bounds)
        _plot_ranked_cluster(axes[row_idx, 1], right_backbones, right_rank, args.right, bounds)

    fig.suptitle(
        f"Backbone comparison for {args.flow}: {args.left} vs {args.right}",
        fontsize=14,
        y=1.01,
    )

    flow_dir = Path(args.output_root) / args.flow
    flow_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.left}_vs_{args.right}_top{args.top_k}"
    fig_path = flow_dir / f"{stem}.png"
    rank_path = flow_dir / f"{stem}_cluster_ranks.csv"
    fig.savefig(fig_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    rank_df = pd.concat([left_ranks, right_ranks], ignore_index=True)
    rank_df.to_csv(rank_path, index=False)

    print(f"Saved {fig_path}")
    print(f"Saved {rank_path}")


if __name__ == "__main__":
    main()
