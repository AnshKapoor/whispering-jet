"""Plot retained flights for one experiment/flow/cluster/aircraft subset.

Usage:
  python scripts/plot_noise_subset_overlay.py --experiment EXP011 --ad Landung --runway 09L --cluster-id 0 --aircraft-type A20N

Output:
  output/eda/noise_subset_audits/<EXP>_<A_D>_<Runway>_cluster<id>_<type>_overlay.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from noise_simulation.automation.aircraft_types import (  # noqa: E402
    build_flight_meta,
    build_flight_type_map,
    build_icao_type_map,
)


TRACK_DEFS = [
    ("center", 0.0, "#111111", 2.0),
    ("inner_left", 0.71, "#4E79A7", 1.3),
    ("inner_right", -0.71, "#4E79A7", 1.3),
    ("middle_left", 1.43, "#F28E2B", 1.1),
    ("middle_right", -1.43, "#F28E2B", 1.1),
    ("outer_left", 2.14, "#59A14F", 0.9),
    ("outer_right", -2.14, "#59A14F", 0.9),
]


def _load_config(exp_dir: Path) -> tuple[dict, Path]:
    cfg_path = exp_dir / "config_resolved.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    preprocessed = Path(cfg["input"]["preprocessed_csv"])
    if not preprocessed.is_absolute():
        preprocessed = (ROOT / preprocessed).resolve()
    return cfg, preprocessed


def _cluster_tracks_utm(df_flow: pd.DataFrame) -> dict[str, np.ndarray]:
    bb = df_flow.groupby("step")[["x_utm", "y_utm"]].median().reset_index().sort_values("step")
    coords = bb[["x_utm", "y_utm"]].to_numpy()
    forward = np.vstack([coords[1:], coords[-1]])
    backward = np.vstack([coords[0], coords[:-1]])
    tangents = forward - backward
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    tangents = tangents / norms
    perps = np.stack([-tangents[:, 1], tangents[:, 0]], axis=1)

    merged = df_flow.merge(bb, on="step", suffixes=("", "_bb"))
    deltas = merged[["x_utm", "y_utm"]].to_numpy() - merged[["x_utm_bb", "y_utm_bb"]].to_numpy()
    offsets = np.einsum("ij,ij->i", deltas, perps[merged["step"].to_numpy()])
    sigma = float(pd.Series(offsets).std())

    tracks: dict[str, np.ndarray] = {}
    for name, mult, _color, _lw in TRACK_DEFS:
        tracks[name] = coords + mult * sigma * perps
    return tracks


def _to_lonlat(
    tracks_utm: dict[str, np.ndarray],
    transformer: Transformer,
) -> dict[str, np.ndarray]:
    """Convert UTM track coordinates to lon/lat for plotting."""

    out: dict[str, np.ndarray] = {}
    for name, coords in tracks_utm.items():
        lon, lat = transformer.transform(coords[:, 0], coords[:, 1])
        out[name] = np.column_stack([lon, lat])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot retained subset flights with cluster backbone and 7-track layout.")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--ad", required=True)
    parser.add_argument("--runway", required=True)
    parser.add_argument("--cluster-id", type=int, required=True)
    parser.add_argument("--aircraft-type", required=True)
    parser.add_argument(
        "--output",
        default=None,
        help="Optional explicit output PNG path. Defaults to output/eda/noise_subset_audits/<subset>_overlay.png",
    )
    args = parser.parse_args()

    exp_dir = ROOT / "output" / "experiments" / args.experiment
    _, preprocessed_path = _load_config(exp_dir)
    matched_paths = sorted((ROOT / "matched_trajectories").glob("matched_trajs_*.csv"))

    flight_meta = build_flight_meta(preprocessed_path)
    icao_set = {meta["icao24"] for meta in flight_meta.values() if meta["icao24"] != "UNKNOWN"}
    icao_map = build_icao_type_map(matched_paths, icao_set)
    flight_type_map = build_flight_type_map(flight_meta, icao_map)

    labels_path = exp_dir / f"labels_{args.ad}_{args.runway}.csv"
    labels = pd.read_csv(labels_path)
    labels = labels[labels["cluster_id"] == args.cluster_id].copy()
    labels["aircraft_type"] = labels["flight_id"].map(flight_type_map).fillna("UNKNOWN")

    subset_ids = set(labels.loc[labels["aircraft_type"] == args.aircraft_type, "flight_id"].astype(int))
    if not subset_ids:
        raise ValueError("No retained flights found for the requested subset.")

    usecols = ["flight_id", "step", "x_utm", "y_utm", "latitude", "longitude", "A/D", "Runway"]
    pre = pd.read_csv(preprocessed_path, usecols=usecols)
    pre = pre.merge(labels[["flight_id", "cluster_id"]], on="flight_id", how="inner")

    cluster_df = pre[(pre["A/D"] == args.ad) & (pre["Runway"] == args.runway) & (pre["cluster_id"] == args.cluster_id)].copy()
    subset_df = cluster_df[cluster_df["flight_id"].isin(subset_ids)].copy()
    if subset_df.empty:
        raise ValueError("Subset trajectory dataframe is empty after filtering.")

    transformer = Transformer.from_crs("EPSG:32632", "EPSG:4326", always_xy=True)

    bb = pd.read_csv(exp_dir / "backbone_tracks.csv")
    bb = bb[(bb["A/D"] == args.ad) & (bb["Runway"] == args.runway) & (bb["cluster_id"] == args.cluster_id)].copy()
    p50 = bb[bb["percentile"] == 50].sort_values("step")

    tracks = _to_lonlat(_cluster_tracks_utm(cluster_df), transformer)

    out_path = Path(args.output) if args.output else (
        ROOT
        / "output"
        / "eda"
        / "noise_subset_audits"
        / f"{args.experiment}_{args.ad}_{args.runway}_cluster{args.cluster_id}_{args.aircraft_type}_overlay.png"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), constrained_layout=True)

    for ax in axes:
        for fid, traj in subset_df.groupby("flight_id"):
            ax.plot(traj["longitude"], traj["latitude"], color="#b8b8b8", alpha=0.25, linewidth=0.7)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, color="#ececec", linewidth=0.8)

    axes[0].plot(p50["longitude"], p50["latitude"], color="#c1121f", linewidth=2.2, label="Backbone p50")
    axes[0].set_title("Retained A20N Flights + Cluster Backbone (cluster-level)")
    axes[0].legend(loc="best", fontsize=8)

    for name, _mult, color, lw in TRACK_DEFS:
        coords = tracks[name]
        axes[1].plot(coords[:, 0], coords[:, 1], color=color, linewidth=lw, label=name.replace("_", " "))
    axes[1].set_title("Retained A20N Flights + Doc29 7-Track Layout (cluster-level)")
    axes[1].legend(loc="best", fontsize=8)

    all_lon = subset_df["longitude"].to_numpy()
    all_lat = subset_df["latitude"].to_numpy()
    if len(all_lon):
        min_lon, max_lon = float(np.min(all_lon)), float(np.max(all_lon))
        min_lat, max_lat = float(np.min(all_lat)), float(np.max(all_lat))
        pad_lon = max((max_lon - min_lon) * 0.2, 0.005)
        pad_lat = max((max_lat - min_lat) * 0.2, 0.005)
        for ax in axes:
            ax.set_xlim(min_lon - pad_lon, max_lon + pad_lon)
            ax.set_ylim(min_lat - pad_lat, max_lat + pad_lat)

    fig.suptitle(
        f"{args.experiment} | {args.ad}_{args.runway} | cluster {args.cluster_id} | {args.aircraft_type} | flights={len(subset_ids)}",
        fontsize=12,
    )
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    print(out_path)


if __name__ == "__main__":
    main()
