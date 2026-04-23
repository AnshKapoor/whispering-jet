"""Plot how a cluster-level backbone becomes the final Doc29 groundtracks.

This script creates a three-panel comparison for one experiment/flow/cluster/
aircraft subset:
1. retained aircraft flights + cluster p50 backbone
2. retained aircraft flights + raw 7-track layout derived from the cluster
3. retained aircraft flights + final discretized Doc29 groundtracks actually
   referenced by Flight_subtracks.csv
"""

from __future__ import annotations

import argparse
import csv
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
from noise_simulation import generate_doc29_inputs as doc29_inputs  # noqa: E402


TRACK_DEFS = [
    ("center", 0.0, "#111111", 2.0),
    ("inner_left", 0.71, "#4E79A7", 1.3),
    ("inner_right", -0.71, "#4E79A7", 1.3),
    ("middle_left", 1.43, "#F28E2B", 1.1),
    ("middle_right", -1.43, "#F28E2B", 1.1),
    ("outer_left", 2.14, "#59A14F", 0.9),
    ("outer_right", -2.14, "#59A14F", 0.9),
]


def _load_config(exp_dir: Path) -> Path:
    cfg_path = exp_dir / "config_resolved.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    preprocessed = Path(cfg["input"]["preprocessed_csv"])
    if not preprocessed.is_absolute():
        preprocessed = (ROOT / preprocessed).resolve()
    return preprocessed


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


def _to_lonlat(tracks_utm: dict[str, np.ndarray], transformer: Transformer) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for name, coords in tracks_utm.items():
        lon, lat = transformer.transform(coords[:, 0], coords[:, 1])
        out[name] = np.column_stack([lon, lat])
    return out


def _load_flight_subtracks_map(flight_subtracks_csv: Path) -> dict[str, Path]:
    rows: list[dict[str, str]] = []
    with flight_subtracks_csv.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter=";")
        for row in reader:
            rows.append(row)
    if not rows:
        raise ValueError(f"No rows found in {flight_subtracks_csv}")

    groundtrack_row = next((r for r in rows if r["Unnamed: 0"] == "Groundtrack"), None)
    if groundtrack_row is None:
        raise ValueError(f"Missing Groundtrack row in {flight_subtracks_csv}")

    out: dict[str, Path] = {}
    base = ROOT / "noise_simulation" / "doc-29-implementation"
    for header, rel_path in groundtrack_row.items():
        if header == "Unnamed: 0":
            continue
        track_name = header.replace("Track ", "").strip().replace(" ", "_")
        out[track_name] = (base / rel_path).resolve()
    return out


def _load_doc29_tracks_lonlat(track_paths: dict[str, Path], transformer: Transformer) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for name, path in track_paths.items():
        df = pd.read_csv(path, sep=";")
        lon, lat = transformer.transform(df["easting"].to_numpy(), df["northing"].to_numpy())
        out[name] = np.column_stack([lon, lat])
    return out


def _generate_doc29_track_df(raw_df: pd.DataFrame, runway: str, mode: str, interpolation_length: float = 200.0) -> pd.DataFrame:
    """Return final Doc29-ready track points from a raw UTM groundtrack."""

    df = raw_df[["easting", "northing"]].copy()
    if len(df) > 0:
        df = df.iloc[1:].reset_index(drop=True)
    start_x, start_y = doc29_inputs.STARTPOINTS[(runway, mode)]
    start_row = pd.DataFrame({"easting": [start_x], "northing": [start_y]})
    df = pd.concat([start_row, df], ignore_index=True)

    interpolator = doc29_inputs.groundtrack_interpolation()
    interp = interpolator.interpolation(df, interpolation_length)
    interp = interpolator.calculate_turn_radius(interp)
    interp.rename(columns={"turn_radius": "radius"}, inplace=True)
    distances = (interp["easting"].diff() ** 2 + interp["northing"].diff() ** 2) ** 0.5
    interp["s"] = distances.cumsum().fillna(0)
    return interp


def _resolve_raw_track_dir(experiment: str, ad: str, runway: str, cluster_id: int) -> Path:
    """Return the raw 7-track directory exported with the noise-simulation inputs."""

    return (
        ROOT
        / "noise_simulation"
        / "results"
        / experiment
        / "raw_tracks"
        / f"{ad}_{runway}"
        / f"cluster_{cluster_id}"
    )


def _load_or_build_doc29_tracks_lonlat(
    track_paths: dict[str, Path],
    transformer: Transformer,
    *,
    runway: str,
    mode: str,
    raw_track_dir: Path,
    interpolation_length: float = 200.0,
) -> dict[str, np.ndarray]:
    """Load cached final Doc29 tracks or generate them from raw 7-track files."""

    if all(path.exists() for path in track_paths.values()):
        return _load_doc29_tracks_lonlat(track_paths, transformer)

    out: dict[str, np.ndarray] = {}
    for name in [item[0] for item in TRACK_DEFS]:
        raw_path = raw_track_dir / f"groundtrack_{name}.csv"
        if not raw_path.exists():
            raise FileNotFoundError(
                f"Missing cached Doc29 groundtrack and raw fallback for {name}: {raw_path}"
            )
        raw_df = pd.read_csv(raw_path, sep=";")
        interp = _generate_doc29_track_df(raw_df, runway=runway, mode=mode, interpolation_length=interpolation_length)
        lon, lat = transformer.transform(interp["easting"].to_numpy(), interp["northing"].to_numpy())
        out[name] = np.column_stack([lon, lat])
    return out


def _load_doc29_flight_tracks_lonlat(
    flight_dir: Path,
    transformer: Transformer,
) -> list[np.ndarray]:
    tracks: list[np.ndarray] = []
    for path in sorted(flight_dir.glob("flight_*.csv")):
        df = pd.read_csv(path, sep=";")
        lon, lat = transformer.transform(df["easting"].to_numpy(), df["northing"].to_numpy())
        tracks.append(np.column_stack([lon, lat]))
    return tracks


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare backbone-derived tracks with final Doc29 tracks.")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--ad", required=True)
    parser.add_argument("--runway", required=True)
    parser.add_argument("--cluster-id", type=int, required=True)
    parser.add_argument("--aircraft-type", required=True)
    parser.add_argument(
        "--output",
        default=None,
        help="Optional explicit output PNG path. Defaults to output/eda/noise_subset_audits/<subset>_doc29_transform.png",
    )
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
    bb = pd.read_csv(exp_dir / "backbone_tracks.csv")
    bb = bb[
        (bb["A/D"] == args.ad)
        & (bb["Runway"] == args.runway)
        & (bb["cluster_id"] == args.cluster_id)
    ].copy()
    p50 = bb[bb["percentile"] == 50].sort_values("step")

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
    mode = doc29_inputs.MODE_MAP[args.ad]
    raw_track_dir = _resolve_raw_track_dir(args.experiment, args.ad, args.runway, args.cluster_id)
    doc29_tracks = _load_or_build_doc29_tracks_lonlat(
        doc29_track_paths,
        transformer,
        runway=args.runway,
        mode=mode,
        raw_track_dir=raw_track_dir,
    )
    flight_doc29_dir = (
        ROOT
        / "noise_simulation"
        / "doc-29-implementation"
        / "Groundtracks"
        / args.experiment
        / f"{args.ad}_{args.runway}"
        / f"cluster_{args.cluster_id}"
        / "groundtruth"
        / args.aircraft_type
    )
    if flight_doc29_dir.exists():
        per_flight_doc29_tracks = _load_doc29_flight_tracks_lonlat(flight_doc29_dir, transformer)
    else:
        per_flight_doc29_tracks = []
        for _fid, traj in subset_df.groupby("flight_id"):
            raw_df = traj.sort_values("step")[["x_utm", "y_utm"]].rename(
                columns={"x_utm": "easting", "y_utm": "northing"}
            )
            interp = _generate_doc29_track_df(raw_df, runway=args.runway, mode=mode)
            lon, lat = transformer.transform(interp["easting"].to_numpy(), interp["northing"].to_numpy())
            per_flight_doc29_tracks.append(np.column_stack([lon, lat]))

    out_path = Path(args.output) if args.output else (
        ROOT
        / "output"
        / "eda"
        / "noise_subset_audits"
        / f"{args.experiment}_{args.ad}_{args.runway}_cluster{args.cluster_id}_{args.aircraft_type}_doc29_transform.png"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 4, figsize=(24, 6.5), constrained_layout=True)
    for ax in axes:
        for _fid, traj in subset_df.groupby("flight_id"):
            ax.plot(traj["longitude"], traj["latitude"], color="#b8b8b8", alpha=0.20, linewidth=0.7)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, color="#ececec", linewidth=0.8)

    axes[0].plot(p50["longitude"], p50["latitude"], color="#c1121f", linewidth=2.2, label="Backbone p50")
    axes[0].set_title("Retained flights + backbone input")
    axes[0].legend(loc="best", fontsize=8)

    for name, _mult, color, lw in TRACK_DEFS:
        coords = raw_tracks[name]
        axes[1].plot(coords[:, 0], coords[:, 1], color=color, linewidth=lw, label=name.replace("_", " "))
    axes[1].set_title("Raw 7-track layout passed to Doc29")
    axes[1].legend(loc="best", fontsize=8)

    for coords in per_flight_doc29_tracks:
        axes[2].plot(coords[:, 0], coords[:, 1], color="#6a1b9a", alpha=0.18, linewidth=0.7)
    axes[2].set_title(f"Final Doc29 per-flight tracks ({len(per_flight_doc29_tracks)})")

    for name, _mult, color, lw in TRACK_DEFS:
        coords = doc29_tracks[name]
        axes[3].plot(coords[:, 0], coords[:, 1], color=color, linewidth=lw, label=name.replace("_", " "))
    axes[3].set_title("Final Doc29 7-track groundtracks")
    axes[3].legend(loc="best", fontsize=8)

    all_lon = subset_df["longitude"].to_numpy()
    all_lat = subset_df["latitude"].to_numpy()
    for track_set in (raw_tracks, doc29_tracks):
        for coords in track_set.values():
            all_lon = np.concatenate([all_lon, coords[:, 0]])
            all_lat = np.concatenate([all_lat, coords[:, 1]])
    for coords in per_flight_doc29_tracks:
        all_lon = np.concatenate([all_lon, coords[:, 0]])
        all_lat = np.concatenate([all_lat, coords[:, 1]])
    all_lon = np.concatenate([all_lon, p50["longitude"].to_numpy()])
    all_lat = np.concatenate([all_lat, p50["latitude"].to_numpy()])
    min_lon, max_lon = float(np.min(all_lon)), float(np.max(all_lon))
    min_lat, max_lat = float(np.min(all_lat)), float(np.max(all_lat))
    pad_lon = max((max_lon - min_lon) * 0.12, 0.004)
    pad_lat = max((max_lat - min_lat) * 0.12, 0.004)
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
