"""Create a diagnostic 4-panel figure for raw vs final Doc29 7-track geometry."""

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
from scripts.plot_doc29_transform_comparison import (  # noqa: E402
    TRACK_DEFS,
    _cluster_tracks_utm,
    _load_or_build_doc29_tracks_lonlat,
    _load_flight_subtracks_map,
    _resolve_raw_track_dir,
    _to_lonlat,
)
from noise_simulation import generate_doc29_inputs as doc29_inputs  # noqa: E402


def _load_config(exp_dir: Path) -> Path:
    cfg_path = exp_dir / "config_resolved.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    preprocessed = Path(cfg["input"]["preprocessed_csv"])
    if not preprocessed.is_absolute():
        preprocessed = (ROOT / preprocessed).resolve()
    return preprocessed


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot raw vs final Doc29 7-track geometry with clear markers.")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--ad", required=True)
    parser.add_argument("--runway", required=True)
    parser.add_argument("--cluster-id", type=int, required=True)
    parser.add_argument("--aircraft-type", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    exp_dir = ROOT / "output" / "experiments" / args.experiment
    preprocessed_path = _load_config(exp_dir)
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
        raise ValueError("No retained flights found for the requested aircraft subset.")

    usecols = ["flight_id", "step", "x_utm", "y_utm", "latitude", "longitude", "A/D", "Runway"]
    pre = pd.read_csv(preprocessed_path, usecols=usecols)
    pre = pre.merge(labels[["flight_id", "cluster_id"]], on="flight_id", how="inner")
    cluster_df = pre[
        (pre["A/D"] == args.ad)
        & (pre["Runway"] == args.runway)
        & (pre["cluster_id"] == args.cluster_id)
    ].copy()
    subset_df = cluster_df[cluster_df["flight_id"].isin(subset_ids)].copy()

    transformer = Transformer.from_crs("EPSG:32632", "EPSG:4326", always_xy=True)
    raw_tracks = _to_lonlat(_cluster_tracks_utm(cluster_df), transformer)

    subset_dir = (
        ROOT
        / "noise_simulation"
        / "results"
        / args.experiment
        / f"{args.ad}_{args.runway}"
        / f"cluster_{args.cluster_id}"
    )
    matching_dirs = [p for p in subset_dir.iterdir() if p.is_dir() and p.name.startswith(f"{args.aircraft_type}__")]
    if not matching_dirs:
        raise FileNotFoundError(f"No noise-simulation subset folder found for {args.aircraft_type} in {subset_dir}")
    aircraft_dir = matching_dirs[0]
    doc29_track_paths = _load_flight_subtracks_map(aircraft_dir / "Flight_subtracks.csv")
    doc29_tracks = _load_or_build_doc29_tracks_lonlat(
        doc29_track_paths,
        transformer,
        runway=args.runway,
        mode=doc29_inputs.MODE_MAP[args.ad],
        raw_track_dir=_resolve_raw_track_dir(args.experiment, args.ad, args.runway, args.cluster_id),
    )

    out_path = Path(args.output) if args.output else (
        ROOT
        / "output"
        / "eda"
        / "noise_subset_audits"
        / f"{args.experiment}_{args.ad}_{args.runway}_cluster{args.cluster_id}_{args.aircraft_type}_doc29_diagnostic.png"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 4, figsize=(24, 6.5), constrained_layout=True)
    for ax in axes:
        for _fid, traj in subset_df.groupby("flight_id"):
            ax.plot(traj["longitude"], traj["latitude"], color="#b8b8b8", alpha=0.18, linewidth=0.6)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, color="#ececec", linewidth=0.8)

    for name, _mult, color, lw in TRACK_DEFS:
        coords = raw_tracks[name]
        axes[0].plot(coords[:, 0], coords[:, 1], color=color, linewidth=lw, linestyle="--", label=name.replace("_", " "))
    axes[0].set_title("Raw 7-track input")
    axes[0].legend(loc="best", fontsize=8)

    for name, _mult, color, lw in TRACK_DEFS:
        coords = doc29_tracks[name]
        axes[1].plot(coords[:, 0], coords[:, 1], color=color, linewidth=lw, marker="o", markersize=2.5, label=name.replace("_", " "))
    axes[1].set_title("Final Doc29 7-track points")
    axes[1].legend(loc="best", fontsize=8)

    center_raw = raw_tracks["center"]
    center_final = doc29_tracks["center"]
    axes[2].plot(center_raw[:, 0], center_raw[:, 1], color="#d62828", linewidth=2.0, linestyle="--", label="raw center")
    axes[2].plot(center_final[:, 0], center_final[:, 1], color="#1d3557", linewidth=2.0, marker="o", markersize=2.8, label="final center")
    axes[2].set_title("Center track: raw vs final")
    axes[2].legend(loc="best", fontsize=8)

    for name, _mult, color, lw in TRACK_DEFS:
        raw = raw_tracks[name]
        final = doc29_tracks[name]
        axes[3].plot(raw[:, 0], raw[:, 1], color=color, linewidth=1.0, linestyle="--", alpha=0.9)
        axes[3].plot(final[:, 0], final[:, 1], color=color, linewidth=lw, marker="o", markersize=2.2, alpha=0.95)
    axes[3].scatter(center_final[0, 0], center_final[0, 1], s=28, color="#111111", zorder=5, label="runway anchor")
    axes[3].set_title("Raw vs final all tracks")
    axes[3].legend(loc="best", fontsize=8)

    all_lon = subset_df["longitude"].to_numpy()
    all_lat = subset_df["latitude"].to_numpy()
    for track_set in (raw_tracks, doc29_tracks):
        for coords in track_set.values():
            all_lon = np.concatenate([all_lon, coords[:, 0]])
            all_lat = np.concatenate([all_lat, coords[:, 1]])
    min_lon, max_lon = float(np.min(all_lon)), float(np.max(all_lon))
    min_lat, max_lat = float(np.min(all_lat)), float(np.max(all_lat))
    pad_lon = max((max_lon - min_lon) * 0.12, 0.004)
    pad_lat = max((max_lat - min_lat) * 0.12, 0.004)
    for ax in axes:
        ax.set_xlim(min_lon - pad_lon, max_lon + pad_lon)
        ax.set_ylim(min_lat - pad_lat, max_lat + pad_lat)

    fig.suptitle(
        f"{args.experiment} | {args.ad}_{args.runway} | cluster {args.cluster_id} | {args.aircraft_type}",
        fontsize=12,
    )
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    print(out_path)


if __name__ == "__main__":
    main()
