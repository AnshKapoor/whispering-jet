"""Plot per-flight Doc29 groundtruth track transformation for one subset.

Creates a two-panel comparison:
1. one selected actual flight vs its final Doc29 groundtrack
2. all retained flights in the subset vs all per-flight Doc29 groundtracks
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
from noise_simulation import generate_doc29_inputs as doc29_inputs  # noqa: E402
from scripts.plot_doc29_transform_comparison import _generate_doc29_track_df  # noqa: E402


def _load_config(exp_dir: Path) -> Path:
    cfg_path = exp_dir / "config_resolved.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    preprocessed = Path(cfg["input"]["preprocessed_csv"])
    if not preprocessed.is_absolute():
        preprocessed = (ROOT / preprocessed).resolve()
    return preprocessed


def _load_subset_df(experiment: str, ad: str, runway: str, cluster_id: int, aircraft_type: str) -> tuple[pd.DataFrame, list[int]]:
    exp_dir = ROOT / "output" / "experiments" / experiment
    preprocessed_path = _load_config(exp_dir)
    matched_paths = sorted((ROOT / "matched_trajectories").glob("matched_trajs_*.csv"))

    flight_meta = build_flight_meta(preprocessed_path)
    icao_set = {meta["icao24"] for meta in flight_meta.values() if meta["icao24"] != "UNKNOWN"}
    icao_map = build_icao_type_map(matched_paths, icao_set)
    flight_type_map = build_flight_type_map(flight_meta, icao_map)

    labels_path = exp_dir / f"labels_{ad}_{runway}.csv"
    labels = pd.read_csv(labels_path)
    labels = labels[labels["cluster_id"] == cluster_id].copy()
    labels["aircraft_type"] = labels["flight_id"].map(flight_type_map).fillna("UNKNOWN")
    subset_ids = sorted(labels.loc[labels["aircraft_type"] == aircraft_type, "flight_id"].astype(int).tolist())
    if not subset_ids:
        raise ValueError("No retained flights found for the requested subset.")

    usecols = ["flight_id", "step", "x_utm", "y_utm", "latitude", "longitude", "A/D", "Runway"]
    pre = pd.read_csv(preprocessed_path, usecols=usecols)
    pre = pre[pre["flight_id"].isin(subset_ids)].copy()
    pre = pre[(pre["A/D"] == ad) & (pre["Runway"] == runway)].copy()
    return pre, subset_ids


def _load_doc29_flight(track_path: Path, transformer: Transformer) -> pd.DataFrame:
    df = pd.read_csv(track_path, sep=";")
    lon, lat = transformer.transform(df["easting"].to_numpy(), df["northing"].to_numpy())
    return pd.DataFrame({"longitude": lon, "latitude": lat, "s": df["s"].to_numpy()})


def _build_doc29_flight_from_subset(traj: pd.DataFrame, runway: str, mode: str, transformer: Transformer) -> pd.DataFrame:
    raw_df = traj.sort_values("step")[["x_utm", "y_utm"]].rename(columns={"x_utm": "easting", "y_utm": "northing"})
    interp = _generate_doc29_track_df(raw_df, runway=runway, mode=mode)
    lon, lat = transformer.transform(interp["easting"].to_numpy(), interp["northing"].to_numpy())
    return pd.DataFrame({"longitude": lon, "latitude": lat, "s": interp["s"].to_numpy()})


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot actual-flight vs per-flight Doc29 groundtruth tracks.")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--ad", required=True)
    parser.add_argument("--runway", required=True)
    parser.add_argument("--cluster-id", type=int, required=True)
    parser.add_argument("--aircraft-type", required=True)
    parser.add_argument("--flight-id", type=int, default=None, help="Optional specific flight id to highlight.")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    subset_df, subset_ids = _load_subset_df(
        args.experiment, args.ad, args.runway, args.cluster_id, args.aircraft_type
    )
    flight_id = args.flight_id if args.flight_id is not None else subset_ids[0]
    if flight_id not in subset_ids:
        raise ValueError(f"Flight {flight_id} is not in the requested subset.")

    gt_dir = (
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
    transformer = Transformer.from_crs("EPSG:32632", "EPSG:4326", always_xy=True)
    mode = doc29_inputs.MODE_MAP[args.ad]

    sample_flight_df = subset_df[subset_df["flight_id"] == flight_id].sort_values("step")
    sample_track_path = gt_dir / f"flight_{flight_id}.csv"
    if sample_track_path.exists():
        sample_doc29_df = _load_doc29_flight(sample_track_path, transformer)
    else:
        sample_doc29_df = _build_doc29_flight_from_subset(sample_flight_df, runway=args.runway, mode=mode, transformer=transformer)

    all_doc29: dict[int, pd.DataFrame] = {}
    for fid, traj in subset_df.groupby("flight_id"):
        path = gt_dir / f"flight_{fid}.csv"
        if path.exists():
            all_doc29[int(fid)] = _load_doc29_flight(path, transformer)
        else:
            all_doc29[int(fid)] = _build_doc29_flight_from_subset(
                traj, runway=args.runway, mode=mode, transformer=transformer
            )

    out_path = Path(args.output) if args.output else (
        ROOT
        / "output"
        / "eda"
        / "noise_subset_audits"
        / f"{args.experiment}_{args.ad}_{args.runway}_cluster{args.cluster_id}_{args.aircraft_type}_groundtruth_doc29.png"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2), constrained_layout=True)
    for ax in axes:
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, color="#ececec", linewidth=0.8)

    # Left: one actual flight vs one final Doc29 flight track.
    axes[0].plot(
        sample_flight_df["longitude"],
        sample_flight_df["latitude"],
        color="#9d0208",
        linewidth=2.0,
        label=f"Actual flight {flight_id}",
    )
    axes[0].plot(
        sample_doc29_df["longitude"],
        sample_doc29_df["latitude"],
        color="#1d3557",
        linewidth=2.0,
        label="Final Doc29 track",
    )
    axes[0].set_title("One flight transformed to Doc29")
    axes[0].legend(loc="best", fontsize=8)

    # Right: all actual flights vs all per-flight Doc29 tracks.
    for _, traj in subset_df.groupby("flight_id"):
        axes[1].plot(traj["longitude"], traj["latitude"], color="#b8b8b8", alpha=0.20, linewidth=0.6)
    for fid, df in all_doc29.items():
        color = "#1d3557" if fid != flight_id else "#457b9d"
        lw = 0.8 if fid != flight_id else 1.6
        alpha = 0.25 if fid != flight_id else 0.9
        axes[1].plot(df["longitude"], df["latitude"], color=color, linewidth=lw, alpha=alpha)
    axes[1].set_title("All retained flights vs all per-flight Doc29 tracks")

    all_lon = np.concatenate(
        [
            subset_df["longitude"].to_numpy(),
            sample_doc29_df["longitude"].to_numpy(),
            *[df["longitude"].to_numpy() for df in all_doc29.values()],
        ]
    )
    all_lat = np.concatenate(
        [
            subset_df["latitude"].to_numpy(),
            sample_doc29_df["latitude"].to_numpy(),
            *[df["latitude"].to_numpy() for df in all_doc29.values()],
        ]
    )
    min_lon, max_lon = float(np.min(all_lon)), float(np.max(all_lon))
    min_lat, max_lat = float(np.min(all_lat)), float(np.max(all_lat))
    pad_lon = max((max_lon - min_lon) * 0.12, 0.004)
    pad_lat = max((max_lat - min_lat) * 0.12, 0.004)
    for ax in axes:
        ax.set_xlim(min_lon - pad_lon, max_lon + pad_lon)
        ax.set_ylim(min_lat - pad_lat, max_lat + pad_lat)

    fig.suptitle(
        f"{args.experiment} | {args.ad}_{args.runway} | cluster {args.cluster_id} | {args.aircraft_type} | flight {flight_id}",
        fontsize=12,
    )
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    print(out_path)


if __name__ == "__main__":
    main()
