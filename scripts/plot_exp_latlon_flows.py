"""Plot clustered trajectories per flow in lat/lon for a given experiment.

Usage:
  python scripts/plot_exp_latlon_flows.py EXP77_optics_ms8_xi03_mcs04_utm ^
    --flows Start_09L Start_09R Start_27L ^
    --max-flights-per-cluster 120

  python scripts/plot_exp_latlon_flows.py EXP023 ^
    --flows Landung_09L ^
    --cluster-ids 0

Outputs:
  output/experiments/<experiment>/graphs/clustered_trajectories_<flow>.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def _load_config(exp_dir: Path) -> dict:
    cfg_path = exp_dir / "config_resolved.yaml"
    if not cfg_path.exists():
        return {}
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}


def _read_label_csv(path: Path, cols: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        return df
    return df[cols]


def _load_labels_for_flow(exp_dir: Path, flow_name: str) -> pd.DataFrame:
    """Load labels for a specific flow if present, else filter from all labels."""

    flow_path = exp_dir / f"labels_{flow_name}.csv"
    cols = ["flight_id", "cluster_id", "A/D", "Runway"]
    if flow_path.exists():
        labels = _read_label_csv(flow_path, cols)
        return labels

    parts: list[pd.DataFrame] = []
    for p in sorted(exp_dir.glob("labels_*.csv")):
        parts.append(_read_label_csv(p, cols))
    if not parts:
        raise FileNotFoundError("No labels_*.csv found in experiment folder.")
    labels = pd.concat(parts, ignore_index=True)

    if "A/D" in labels.columns and "Runway" in labels.columns and "_" in flow_name:
        ad, runway = flow_name.split("_", 1)
        labels = labels[(labels["A/D"] == ad) & (labels["Runway"] == runway)].copy()

    return labels


def _sample_flight_ids(labels: pd.DataFrame, max_per_cluster: int, seed: int) -> set[int]:
    rng = np.random.default_rng(seed)
    sampled: list[int] = []
    for cid in sorted(labels["cluster_id"].unique()):
        ids = labels.loc[labels["cluster_id"] == cid, "flight_id"].to_numpy()
        if len(ids) > max_per_cluster:
            ids = rng.choice(ids, size=max_per_cluster, replace=False)
        sampled.extend(ids.tolist())
    return set(sampled)


def _read_preprocessed(preprocessed_path: Path, flight_ids: set[int]) -> pd.DataFrame:
    usecols = ["flight_id", "step", "latitude", "longitude"]
    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(preprocessed_path, usecols=usecols, chunksize=200_000):
        chunk["flight_id"] = chunk["flight_id"].astype(int)
        chunk = chunk[chunk["flight_id"].isin(flight_ids)]
        if not chunk.empty:
            chunks.append(chunk)
    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True)


def _cluster_palette(cluster_ids: Iterable[int]) -> dict[int, str]:
    palette: dict[int, str] = {}
    # High-contrast qualitative palette (stable ordering by sorted cluster id).
    # First 8 are intentionally very distinct for dense overlays.
    base = [
        "#4E79A7",  # blue
        "#F28E2B",  # orange
        "#E15759",  # red
        "#76B7B2",  # teal
        "#59A14F",  # green
        "#EDC948",  # yellow
        "#B07AA1",  # purple
        "#FF9DA7",  # pink
        "#9C755F",  # brown
        "#BAB0AC",  # gray
    ]

    for idx, cid in enumerate(sorted(cluster_ids)):
        # Keep cluster 4 slightly lighter for readability in dense overlays.

        if idx < len(base):
            palette[cid] = base[idx]
        else:
            # Keep deterministic fallback for >10 clusters.
            palette[cid] = plt.get_cmap("tab20")(idx % 20)
    return palette


def _plot_flow(
    df: pd.DataFrame,
    flow_name: str,
    out_path: Path,
    max_flights_per_cluster: int,
    include_noise: bool = True,
) -> None:
    cluster_ids = sorted(df["cluster_id"].unique())
    noise_color = "#B0B0B0"
    if -1 in cluster_ids:
        cluster_ids = [cid for cid in cluster_ids if cid != -1]
    palette = _cluster_palette(cluster_ids)

    plt.style.use("seaborn-v0_8-white")
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot noise first in gray so clusters stand out.
    noise_df = df[df["cluster_id"] == -1] if include_noise else pd.DataFrame(columns=df.columns)
    if include_noise:
        for _, flight in noise_df.groupby("flight_id"):
            flight = flight.sort_values("step")
            ax.plot(
                flight["longitude"],
                flight["latitude"],
                color=noise_color,
                alpha=0.2,
                lw=0.6,
            )

    for cid in cluster_ids:
        subset = df[df["cluster_id"] == cid]
        for _, flight in subset.groupby("flight_id"):
            flight = flight.sort_values("step")
            ax.plot(
                flight["longitude"],
                flight["latitude"],
                color=palette[cid],
                alpha=0.35,
                lw=0.8,
            )

    ax.set_title(
        f"{flow_name} clusters (lat/lon)\n"
        f"Sampled trajectories per cluster (max {max_flights_per_cluster})"
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#e9e9e9", linewidth=0.7)

    if len(cluster_ids) <= 20:
        handles = [
            Line2D([0], [0], color=palette[cid], lw=2, label=f"cluster {cid}")
            for cid in cluster_ids
        ]
        if not noise_df.empty:
            handles.insert(0, Line2D([0], [0], color=noise_color, lw=2, label="noise"))
        legend_loc = "upper left" if len(cluster_ids) == 8 else "upper right"
        ax.legend(handles=handles, loc=legend_loc, frameon=True, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot clustered trajectories per flow in lat/lon.")
    parser.add_argument("experiment", help="Experiment folder name under output/experiments/")
    parser.add_argument("--output-root", default="output/experiments", help="Root experiments directory")
    parser.add_argument("--preprocessed", default=None, help="Override preprocessed CSV path")
    parser.add_argument("--flows", nargs="+", required=True, help="Flows like Start_09L")
    parser.add_argument(
        "--cluster-ids",
        nargs="+",
        type=int,
        default=None,
        help="Optional cluster IDs to plot. Example: --cluster-ids 0 or --cluster-ids 0 1 -1",
    )
    parser.add_argument("--max-flights-per-cluster", type=int, default=200)
    parser.add_argument("--min-points-per-flight", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--exclude-noise",
        action="store_true",
        help="Exclude noise flights (cluster_id = -1) from the plot.",
    )
    args = parser.parse_args()

    exp_dir = Path(args.output_root) / args.experiment
    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment folder not found: {exp_dir}")

    cfg = _load_config(exp_dir)
    preprocessed_path = args.preprocessed or (cfg.get("input", {}) or {}).get("preprocessed_csv")
    if preprocessed_path is None:
        raise FileNotFoundError(
            "Missing preprocessed CSV path. Pass --preprocessed or ensure config_resolved.yaml contains input.preprocessed_csv."
        )
    preprocessed_path = Path(preprocessed_path)
    if not preprocessed_path.exists():
        raise FileNotFoundError(f"Preprocessed CSV not found: {preprocessed_path}")

    graphs_dir = exp_dir / "graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)

    for flow_name in args.flows:
        labels = _load_labels_for_flow(exp_dir, flow_name)
        if labels.empty:
            print(f"No labels found for flow {flow_name}; skipping.")
            continue

        labels = labels.copy()
        labels.loc[:, "flight_id"] = labels["flight_id"].astype(int)
        labels.loc[:, "cluster_id"] = labels["cluster_id"].astype(int)
        sampled_ids = _sample_flight_ids(labels, args.max_flights_per_cluster, args.seed)
        df = _read_preprocessed(preprocessed_path, sampled_ids)
        if df.empty:
            print(f"No rows loaded for flow {flow_name}; skipping.")
            continue

        df = df.merge(labels[["flight_id", "cluster_id"]], on="flight_id", how="left")
        df = df[df["cluster_id"].notna()]
        if args.exclude_noise:
            df = df[df["cluster_id"] != -1]
            if df.empty:
                print(f"Flow {flow_name}: only noise flights present; skipping due to --exclude-noise.")
                continue

        if args.min_points_per_flight > 1:
            counts = df.groupby("flight_id").size()
            keep_ids = counts[counts >= args.min_points_per_flight].index
            df = df[df["flight_id"].isin(keep_ids)]
            if df.empty:
                print(
                    f"Flow {flow_name}: no flights left after min-points filter; "
                    f"lower --min-points-per-flight."
                )
                continue

        suffix = "_no_noise" if args.exclude_noise else ""
        out_path = graphs_dir / f"clusters_{flow_name}_latlon{suffix}.png"
        _plot_flow(
            df,
            flow_name,
            out_path,
            args.max_flights_per_cluster,
            include_noise=not args.exclude_noise,
        )
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
