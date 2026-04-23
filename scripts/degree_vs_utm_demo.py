"""Quick demo: degree-space vs UTM-space distance behavior on sample flights.

Inputs
- preprocessed CSV with at least:
  flight_id, step, latitude, longitude, x_utm, y_utm

Outputs
- Console summary with:
  1) sampled flights and local cos(latitude) scaling
  2) pairwise endpoint distance comparison
  3) pole sanity table for cos(latitude)
- Optional CSV:
  output/eda/degree_vs_utm_demo/pairwise_endpoint_distances.csv

Usage
  python scripts/degree_vs_utm_demo.py --preprocessed output/preprocessed/preprocessed_1.csv --n-flights 3
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


M_PER_DEG_LAT = 111_132.0
M_PER_DEG_LON_EQ = 111_320.0


def _plot_two_flights(sample: pd.DataFrame, picked: list[int], out_path: Path) -> None:
    """Plot two sampled flights in both degree and UTM spaces."""
    if len(picked) < 2:
        return
    fids = picked[:2]
    sub = sample[sample["flight_id"].isin(fids)].copy()
    if sub.empty:
        return

    colors = ["#1f77b4", "#d62728"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax_deg, ax_utm = axes

    for idx, fid in enumerate(fids):
        g = sub[sub["flight_id"] == fid].sort_values("step")
        c = colors[idx % len(colors)]
        ax_deg.plot(g["longitude"], g["latitude"], color=c, linewidth=2, label=f"flight {fid}")
        ax_deg.scatter(g["longitude"].iloc[0], g["latitude"].iloc[0], color=c, s=24, marker="o")
        ax_deg.scatter(g["longitude"].iloc[-1], g["latitude"].iloc[-1], color=c, s=24, marker="x")

        ax_utm.plot(g["x_utm"], g["y_utm"], color=c, linewidth=2, label=f"flight {fid}")
        ax_utm.scatter(g["x_utm"].iloc[0], g["y_utm"].iloc[0], color=c, s=24, marker="o")
        ax_utm.scatter(g["x_utm"].iloc[-1], g["y_utm"].iloc[-1], color=c, s=24, marker="x")

    ax_deg.set_title("Two Flights in Degree Space")
    ax_deg.set_xlabel("Longitude")
    ax_deg.set_ylabel("Latitude")
    ax_deg.grid(True, alpha=0.3)
    ax_deg.legend(loc="best")

    ax_utm.set_title("Two Flights in UTM Space")
    ax_utm.set_xlabel("x_utm (m)")
    ax_utm.set_ylabel("y_utm (m)")
    ax_utm.grid(True, alpha=0.3)
    ax_utm.legend(loc="best")
    ax_utm.set_aspect("equal", adjustable="box")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _pick_flights(df: pd.DataFrame, n_flights: int, seed: int) -> list[int]:
    ids = np.array(sorted(df["flight_id"].dropna().astype(int).unique()))
    if ids.size == 0:
        raise ValueError("No flight_id values found.")
    if ids.size <= n_flights:
        return ids.tolist()
    rng = np.random.default_rng(seed)
    picked = rng.choice(ids, size=n_flights, replace=False)
    return sorted(int(x) for x in picked)


def _flight_endpoint_row(df_f: pd.DataFrame) -> dict[str, float]:
    first = df_f.iloc[0]
    last = df_f.iloc[-1]
    mean_lat = float(df_f["latitude"].mean())
    cos_lat = math.cos(math.radians(mean_lat))
    dlat = float(last["latitude"] - first["latitude"])
    dlon = float(last["longitude"] - first["longitude"])
    dx = float(last["x_utm"] - first["x_utm"])
    dy = float(last["y_utm"] - first["y_utm"])

    utm_m = math.hypot(dx, dy)
    deg_euclid = math.hypot(dlat, dlon)
    naive_deg_m = deg_euclid * M_PER_DEG_LON_EQ
    local_scaled_m = math.hypot(dlat * M_PER_DEG_LAT, dlon * M_PER_DEG_LON_EQ * cos_lat)

    err_naive = abs(naive_deg_m - utm_m) / utm_m if utm_m > 0 else float("nan")
    err_local = abs(local_scaled_m - utm_m) / utm_m if utm_m > 0 else float("nan")

    return {
        "mean_lat": mean_lat,
        "cos_lat": cos_lat,
        "dlat_deg": dlat,
        "dlon_deg": dlon,
        "utm_endpoint_m": utm_m,
        "deg_euclid_raw": deg_euclid,
        "deg_naive_to_m": naive_deg_m,
        "deg_local_scaled_m": local_scaled_m,
        "rel_err_naive_pct": 100.0 * err_naive,
        "rel_err_local_pct": 100.0 * err_local,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Degree vs UTM distance demo on sampled flights.")
    parser.add_argument(
        "--preprocessed",
        default="output/preprocessed/preprocessed_1.csv",
        help="Path to preprocessed CSV.",
    )
    parser.add_argument("--n-flights", type=int, default=3, help="Number of sample flights (2-3 recommended).")
    parser.add_argument("--seed", type=int, default=11, help="Random seed for sampling.")
    parser.add_argument(
        "--outdir",
        default="output/eda/degree_vs_utm_demo",
        help="Output directory for CSV summary.",
    )
    args = parser.parse_args()

    if args.n_flights < 2:
        raise ValueError("--n-flights must be >= 2")

    usecols = ["flight_id", "step", "latitude", "longitude", "x_utm", "y_utm"]
    df = pd.read_csv(args.preprocessed, usecols=usecols)
    df = df.dropna(subset=usecols).copy()
    df["flight_id"] = df["flight_id"].astype(int)
    df = df.sort_values(["flight_id", "step"])

    picked = _pick_flights(df, n_flights=args.n_flights, seed=args.seed)
    sample = df[df["flight_id"].isin(picked)].copy()

    flight_rows: list[dict[str, float | int]] = []
    for fid, g in sample.groupby("flight_id", sort=True):
        g = g.sort_values("step")
        row = {"flight_id": int(fid), "n_points": int(len(g))}
        row.update(_flight_endpoint_row(g))
        flight_rows.append(row)

    flight_df = pd.DataFrame(flight_rows)

    # Pairwise endpoint comparison between sampled flights.
    pair_rows: list[dict[str, float | int]] = []
    ids = list(flight_df["flight_id"].astype(int))
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            fi = sample[sample["flight_id"] == ids[i]].sort_values("step").iloc[-1]
            fj = sample[sample["flight_id"] == ids[j]].sort_values("step").iloc[-1]
            mean_lat = float((fi["latitude"] + fj["latitude"]) / 2.0)
            cos_lat = math.cos(math.radians(mean_lat))
            dlat = float(fi["latitude"] - fj["latitude"])
            dlon = float(fi["longitude"] - fj["longitude"])
            dx = float(fi["x_utm"] - fj["x_utm"])
            dy = float(fi["y_utm"] - fj["y_utm"])
            utm_m = math.hypot(dx, dy)
            deg_raw = math.hypot(dlat, dlon)
            deg_naive_m = deg_raw * M_PER_DEG_LON_EQ
            deg_local_m = math.hypot(dlat * M_PER_DEG_LAT, dlon * M_PER_DEG_LON_EQ * cos_lat)
            err_naive = abs(deg_naive_m - utm_m) / utm_m if utm_m > 0 else float("nan")
            err_local = abs(deg_local_m - utm_m) / utm_m if utm_m > 0 else float("nan")
            pair_rows.append(
                {
                    "flight_a": int(ids[i]),
                    "flight_b": int(ids[j]),
                    "mean_lat": mean_lat,
                    "cos_lat": cos_lat,
                    "utm_m": utm_m,
                    "deg_raw": deg_raw,
                    "deg_naive_to_m": deg_naive_m,
                    "deg_local_scaled_m": deg_local_m,
                    "rel_err_naive_pct": 100.0 * err_naive,
                    "rel_err_local_pct": 100.0 * err_local,
                }
            )

    pair_df = pd.DataFrame(pair_rows)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    pair_df.to_csv(outdir / "pairwise_endpoint_distances.csv", index=False)
    flights_plot_path = outdir / "two_flights_deg_vs_utm.png"
    _plot_two_flights(sample=sample, picked=picked, out_path=flights_plot_path)

    print("Sampled flights:", picked)
    print("\nPer-flight endpoint comparison:")
    print(
        flight_df[
            [
                "flight_id",
                "n_points",
                "mean_lat",
                "cos_lat",
                "utm_endpoint_m",
                "deg_naive_to_m",
                "deg_local_scaled_m",
                "rel_err_naive_pct",
                "rel_err_local_pct",
            ]
        ].round(4).to_string(index=False)
    )

    if not pair_df.empty:
        print("\nPairwise endpoint comparison:")
        print(
            pair_df[
                [
                    "flight_a",
                    "flight_b",
                    "mean_lat",
                    "cos_lat",
                    "utm_m",
                    "deg_naive_to_m",
                    "deg_local_scaled_m",
                    "rel_err_naive_pct",
                    "rel_err_local_pct",
                ]
            ].round(4).to_string(index=False)
        )

    pole_demo = pd.DataFrame({"latitude_deg": [0, 30, 52, 54, 56, 80, 89, 90]})
    pole_demo["cos_lat"] = pole_demo["latitude_deg"].map(lambda x: math.cos(math.radians(float(x))))
    pole_demo["meters_per_deg_lon"] = pole_demo["cos_lat"] * M_PER_DEG_LON_EQ
    print("\nPole sanity table:")
    print(pole_demo.round(4).to_string(index=False))
    print(f"\nSaved: {outdir / 'pairwise_endpoint_distances.csv'}")
    print(f"Saved: {flights_plot_path}")


if __name__ == "__main__":
    main()
