"""Generate backbone tracks and plots per experiment (lat/lon).

Usage:
  python scripts/plot_backbone_tracks.py --experiment EXP77

Options:
  --arrival-scheme median   -> plot only median (p50) for arrivals
  --arrival-scheme seven    -> plot 7-track layout (Doc29-style) for arrivals

Inputs (required):
  output/experiments/<EXP>/labels_*.csv
  config_resolved.yaml (for preprocessed path + percentiles)

Outputs:
  output/experiments/<EXP>/backbone_tracks.csv
  output/experiments/<EXP>/plots/backbone/<flow>/cluster_<id>_backbone.png
  output/experiments/<EXP>/plots/backbone/<flow>/cluster_<id>_arrival_doc29.png (if enabled)
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
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backbone_tracks.backbone import compute_backbones


def _load_config(exp_dir: Path) -> dict:
    cfg_path = exp_dir / "config_resolved.yaml"
    if not cfg_path.exists():
        return {}
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}


def _load_labels(exp_dir: Path) -> pd.DataFrame:
    parts = []
    for p in sorted(exp_dir.glob("labels_*.csv")):
        parts.append(pd.read_csv(p))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _lighten(color: Tuple[float, float, float], factor: float) -> Tuple[float, float, float]:
    return tuple(1 - (1 - c) * factor for c in color)


def _cluster_colors(cluster_ids: Iterable[int]) -> Dict[int, Tuple[float, float, float]]:
    cmap = plt.get_cmap("tab20")
    colors = {}
    for idx, cid in enumerate(sorted(cluster_ids)):
        colors[cid] = cmap(idx % 20)[:3]
    return colors


def _apply_latlon_padding(
    ax: plt.Axes,
    longitude: Iterable[float],
    latitude: Iterable[float],
    pad_ratio: float = 0.35,
    min_pad: float = 0.005,
) -> None:
    """Set padded lon/lat limits so tracks do not fill the full frame."""

    lon = np.asarray(list(longitude), dtype=float)
    lat = np.asarray(list(latitude), dtype=float)
    if lon.size == 0 or lat.size == 0:
        return

    min_lon, max_lon = float(np.min(lon)), float(np.max(lon))
    min_lat, max_lat = float(np.min(lat)), float(np.max(lat))
    pad_lon = max((max_lon - min_lon) * pad_ratio, min_pad)
    pad_lat = max((max_lat - min_lat) * pad_ratio, min_pad)
    ax.set_xlim(min_lon - pad_lon, max_lon + pad_lon)
    ax.set_ylim(min_lat - pad_lat, max_lat + pad_lat)


def _doc29_tracks_utm(df_flow: pd.DataFrame) -> Dict[str, np.ndarray]:
    # Median backbone per step in UTM
    bb = df_flow.groupby("step")[["x_utm", "y_utm"]].median().reset_index().sort_values("step")
    coords = bb[["x_utm", "y_utm"]].to_numpy()

    # Tangents (central diff)
    forward = np.vstack([coords[1:], coords[-1]])
    backward = np.vstack([coords[0], coords[:-1]])
    tangents = forward - backward
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    tangents = tangents / norms
    perps = np.stack([-tangents[:, 1], tangents[:, 0]], axis=1)

    # Lateral offsets relative to backbone
    merged = df_flow.merge(bb, on="step", suffixes=("", "_bb"))
    deltas = merged[["x_utm", "y_utm"]].to_numpy() - merged[["x_utm_bb", "y_utm_bb"]].to_numpy()
    offsets = np.einsum("ij,ij->i", deltas, perps[merged["step"].to_numpy()])
    sigma = float(pd.Series(offsets).std())

    track_defs = [
        ("center", 0.0),
        ("inner_left", 0.71),
        ("inner_right", -0.71),
        ("middle_left", 1.43),
        ("middle_right", -1.43),
        ("outer_left", 2.14),
        ("outer_right", -2.14),
    ]
    tracks = {}
    for name, mult in track_defs:
        shift = mult * sigma * perps
        shifted = coords + shift
        tracks[name] = shifted
    return tracks


def _plot_backbone_cluster(
    subset: pd.DataFrame,
    out_path: Path,
    color: Tuple[float, float, float],
) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    p50 = subset[subset["percentile"] == 50].sort_values("step")
    p10 = subset[subset["percentile"] == 10].sort_values("step")
    p90 = subset[subset["percentile"] == 90].sort_values("step")

    ax.plot(p50["longitude"], p50["latitude"], color=color, lw=2.0, label="p50")
    ax.plot(p10["longitude"], p10["latitude"], color=_lighten(color, 0.6), lw=1.2, ls="--", label="p10")
    ax.plot(p90["longitude"], p90["latitude"], color=_lighten(color, 0.6), lw=1.2, ls="--", label="p90")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal", adjustable="box")
    all_lon = pd.concat([p10["longitude"], p50["longitude"], p90["longitude"]], ignore_index=True)
    all_lat = pd.concat([p10["latitude"], p50["latitude"], p90["latitude"]], ignore_index=True)
    _apply_latlon_padding(ax, all_lon, all_lat)
    ax.grid(True, color="#ececec", lw=0.8)
    ax.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_doc29_arrival(
    df_flow: pd.DataFrame,
    out_path: Path,
    transformer: Transformer,
) -> None:
    tracks = _doc29_tracks_utm(df_flow)
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    colors = {
        "center": "#000000",
        "inner_left": "#4E79A7",
        "inner_right": "#4E79A7",
        "middle_left": "#F28E2B",
        "middle_right": "#F28E2B",
        "outer_left": "#59A14F",
        "outer_right": "#59A14F",
    }

    # Fill bands between tracks for visual separation.
    band_pairs = [
        ("outer_left", "middle_left", "#59A14F"),
        ("middle_left", "inner_left", "#F28E2B"),
        ("inner_left", "center", "#4E79A7"),
        ("center", "inner_right", "#4E79A7"),
        ("inner_right", "middle_right", "#F28E2B"),
        ("middle_right", "outer_right", "#59A14F"),
    ]
    for left_name, right_name, fill_color in band_pairs:
        if left_name not in tracks or right_name not in tracks:
            continue
        left = tracks[left_name]
        right = tracks[right_name]
        lon_l, lat_l = transformer.transform(left[:, 0], left[:, 1])
        lon_r, lat_r = transformer.transform(right[:, 0], right[:, 1])
        poly_lon = list(lon_l) + list(lon_r[::-1])
        poly_lat = list(lat_l) + list(lat_r[::-1])
        ax.fill(poly_lon, poly_lat, color=fill_color, alpha=0.08, linewidth=0)

    for name, coords in tracks.items():
        lon, lat = transformer.transform(coords[:, 0], coords[:, 1])
        ax.plot(lon, lat, color=colors.get(name, "#333333"), lw=1.4, label=name)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal", adjustable="box")
    all_lon = []
    all_lat = []
    for coords in tracks.values():
        lon, lat = transformer.transform(coords[:, 0], coords[:, 1])
        all_lon.extend(lon)
        all_lat.extend(lat)
    _apply_latlon_padding(ax, all_lon, all_lat)
    ax.grid(True, color="#ececec", lw=0.8)
    ax.legend(loc="best", fontsize=8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate backbone tracks and plots for an experiment.")
    parser.add_argument("--experiment", required=True, help="Experiment name (e.g., EXP77).")
    parser.add_argument("--output-root", default="output/experiments", help="Root experiments directory.")
    parser.add_argument("--arrival-scheme", choices=["median", "seven"], default="median")
    args = parser.parse_args()

    exp_dir = Path(args.output_root) / args.experiment
    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment folder not found: {exp_dir}")

    cfg = _load_config(exp_dir)
    preprocessed_path = (cfg.get("input", {}) or {}).get("preprocessed_csv")
    if not preprocessed_path:
        raise FileNotFoundError("Missing input.preprocessed_csv in config_resolved.yaml")
    preprocessed_path = Path(preprocessed_path)
    if not preprocessed_path.exists():
        raise FileNotFoundError(f"Preprocessed CSV not found: {preprocessed_path}")

    labels = _load_labels(exp_dir)
    if labels.empty:
        raise FileNotFoundError("No labels_*.csv found in experiment folder.")

    needed_cols = [
        "flight_id",
        "step",
        "x_utm",
        "y_utm",
        "latitude",
        "longitude",
        "A/D",
        "Runway",
    ]
    pre = pd.read_csv(preprocessed_path, usecols=lambda c: c in set(needed_cols))
    pre = pre.merge(labels[["flight_id", "cluster_id"]], on="flight_id", how="inner")

    percentiles = (cfg.get("backbone", {}) or {}).get("percentiles", [10, 50, 90])
    min_flights = int((cfg.get("backbone", {}) or {}).get("min_flights_per_cluster", 10))
    backbones = compute_backbones(
        pre,
        percentiles=list(percentiles),
        min_flights_per_cluster=min_flights,
        use_utm=False,
    )
    if backbones.empty:
        raise RuntimeError("No backbone tracks generated (check inputs).")

    backbone_csv = exp_dir / "backbone_tracks.csv"
    backbones.to_csv(backbone_csv, index=False)

    plots_dir = exp_dir / "plots" / "backbone"
    transformer = Transformer.from_crs("epsg:32632", "epsg:4326", always_xy=True)

    for (ad, runway, cluster_id), subset in backbones.groupby(["A/D", "Runway", "cluster_id"]):
        color_map = _cluster_colors(backbones["cluster_id"].unique())
        color = color_map.get(cluster_id, (0.2, 0.2, 0.2))
        flow_name = f"{ad}_{runway}"
        out_path = plots_dir / flow_name / f"cluster_{cluster_id}_backbone.png"
        _plot_backbone_cluster(subset, out_path, color)

        if args.arrival_scheme == "seven":
            df_flow = pre[(pre["A/D"] == ad) & (pre["Runway"] == runway) & (pre["cluster_id"] == cluster_id)]
            if not df_flow.empty:
                out_path = plots_dir / flow_name / f"cluster_{cluster_id}_doc29.png"
                _plot_doc29_arrival(df_flow, out_path, transformer)

    print(f"Backbone tracks written to {backbone_csv}")


if __name__ == "__main__":
    main()
