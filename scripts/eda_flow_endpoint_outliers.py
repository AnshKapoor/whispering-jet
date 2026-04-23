#!/usr/bin/env python
"""
Assess per-flow endpoint consistency and flag odd-looking flights.

The script computes one row per flight from a preprocessed CSV, then derives:
- raw start-point summaries by flow
- raw end-point summaries by flow
- operation-aware anchor summaries by flow

The operation-aware anchor is:
- Departure (`Start`): first point of the trajectory
- Arrival (`Landung`): last point of the trajectory

Outliers are flagged using robust radial distance in UTM space relative to the
per-flow median endpoint:

    d_i = sqrt((x_i - x_med)^2 + (y_i - y_med)^2)
    tau = median(d) + k * 1.4826 * MAD(d)

Flights with `d_i > tau` are flagged as endpoint outliers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FLOW_ORDER = [
    "Start_09L",
    "Start_09R",
    "Start_27L",
    "Start_27R",
    "Landung_09L",
    "Landung_09R",
    "Landung_27L",
    "Landung_27R",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit preprocessed flow endpoints and flag odd-looking flights.")
    parser.add_argument(
        "--preprocessed",
        required=True,
        help="Path to preprocessed CSV.",
    )
    parser.add_argument(
        "--aircraft-type-match-filter",
        choices=["all", "matched", "unmatched"],
        default="all",
        help="Filter by aircraft_type_match. 'unmatched' includes false and missing.",
    )
    parser.add_argument(
        "--robust-k",
        type=float,
        default=3.0,
        help="Multiplier for robust MAD-based outlier threshold.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("output/eda/flow_endpoint_outliers"),
        help="Output directory.",
    )
    return parser.parse_args()


def _is_aircraft_type_match(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    truthy_numeric = numeric.eq(1.0)
    truthy_text = (
        series.astype("string")
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes"})
    )
    return truthy_numeric | truthy_text


def _mad(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    med = np.median(arr)
    return float(np.median(np.abs(arr - med)))


def _robust_threshold(distances: np.ndarray, k: float) -> tuple[float, float, float]:
    d = np.asarray(distances, dtype=float)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return float("nan"), float("nan"), float("nan")
    med = float(np.median(d))
    mad = _mad(d)
    scale = 1.4826 * mad
    tau = med + float(k) * scale
    return med, scale, tau


def _normalize_launch_path(raw: str) -> Path:
    return Path(str(raw).split("|", 1)[0].strip())


def _load_flights(preprocessed_path: Path, aircraft_type_match_filter: str) -> pd.DataFrame:
    usecols = [
        "flight_id",
        "step",
        "A/D",
        "Runway",
        "latitude",
        "longitude",
        "x_utm",
        "y_utm",
        "aircraft_type_match",
    ]
    df = pd.read_csv(preprocessed_path, usecols=usecols)
    df["flight_id"] = df["flight_id"].astype(int)

    if aircraft_type_match_filter != "all":
        flight_match = (
            df.groupby("flight_id", sort=False)["aircraft_type_match"]
            .first()
            .pipe(_is_aircraft_type_match)
        )
        if aircraft_type_match_filter == "matched":
            keep_ids = set(flight_match[flight_match].index.astype(int).tolist())
        else:
            keep_ids = set(flight_match[~flight_match].index.astype(int).tolist())
        df = df[df["flight_id"].isin(keep_ids)].copy()

    if df.empty:
        raise ValueError("No flights left after filtering.")

    df["flow"] = df["A/D"].astype(str).str.strip() + "_" + df["Runway"].astype(str).str.strip()
    return df


def _build_flight_endpoints(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for flight_id, grp in df.groupby("flight_id", sort=True):
        g = grp.sort_values("step")
        first = g.iloc[0]
        last = g.iloc[-1]
        ad = str(first["A/D"]).strip()
        runway = str(first["Runway"]).strip()
        flow = f"{ad}_{runway}"
        start_x = float(first["x_utm"])
        start_y = float(first["y_utm"])
        end_x = float(last["x_utm"])
        end_y = float(last["y_utm"])
        if ad == "Start":
            anchor_x, anchor_y = start_x, start_y
            anchor_lat, anchor_lon = float(first["latitude"]), float(first["longitude"])
        else:
            anchor_x, anchor_y = end_x, end_y
            anchor_lat, anchor_lon = float(last["latitude"]), float(last["longitude"])
        rows.append(
            {
                "flight_id": int(flight_id),
                "A/D": ad,
                "Runway": runway,
                "flow": flow,
                "n_points": int(len(g)),
                "start_latitude": float(first["latitude"]),
                "start_longitude": float(first["longitude"]),
                "start_x_utm": start_x,
                "start_y_utm": start_y,
                "end_latitude": float(last["latitude"]),
                "end_longitude": float(last["longitude"]),
                "end_x_utm": end_x,
                "end_y_utm": end_y,
                "anchor_latitude": anchor_lat,
                "anchor_longitude": anchor_lon,
                "anchor_x_utm": anchor_x,
                "anchor_y_utm": anchor_y,
            }
        )
    return pd.DataFrame(rows)


def _summarize_endpoint(flights: pd.DataFrame, prefix: str, robust_k: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    work = flights.copy()
    outlier_flag_col = f"{prefix}_is_outlier"
    dist_col = f"{prefix}_dist_from_median_m"
    med_x_col = f"{prefix}_median_x_utm"
    med_y_col = f"{prefix}_median_y_utm"
    tau_col = f"{prefix}_threshold_m"

    work[outlier_flag_col] = False
    work[dist_col] = np.nan
    work[med_x_col] = np.nan
    work[med_y_col] = np.nan
    work[tau_col] = np.nan

    x_col = f"{prefix}_x_utm"
    y_col = f"{prefix}_y_utm"
    lat_col = f"{prefix}_latitude"
    lon_col = f"{prefix}_longitude"

    for flow in FLOW_ORDER:
        grp = work[work["flow"] == flow].copy()
        if grp.empty:
            continue
        med_x = float(np.median(grp[x_col].to_numpy(dtype=float)))
        med_y = float(np.median(grp[y_col].to_numpy(dtype=float)))
        med_lat = float(np.median(grp[lat_col].to_numpy(dtype=float)))
        med_lon = float(np.median(grp[lon_col].to_numpy(dtype=float)))
        dist = np.sqrt((grp[x_col] - med_x) ** 2 + (grp[y_col] - med_y) ** 2)
        med_d, robust_sigma, tau = _robust_threshold(dist.to_numpy(dtype=float), robust_k)
        p95 = float(np.percentile(dist, 95))
        p99 = float(np.percentile(dist, 99))
        flags = dist > tau
        work.loc[grp.index, outlier_flag_col] = flags.to_numpy()
        work.loc[grp.index, dist_col] = dist.to_numpy()
        work.loc[grp.index, med_x_col] = med_x
        work.loc[grp.index, med_y_col] = med_y
        work.loc[grp.index, tau_col] = tau
        rows.append(
            {
                "flow": flow,
                "n_flights": int(len(grp)),
                f"{prefix}_median_latitude": med_lat,
                f"{prefix}_median_longitude": med_lon,
                med_x_col: med_x,
                med_y_col: med_y,
                f"{prefix}_dist_median_m": med_d,
                f"{prefix}_dist_robust_sigma_m": robust_sigma,
                tau_col: tau,
                f"{prefix}_dist_p95_m": p95,
                f"{prefix}_dist_p99_m": p99,
                f"{prefix}_n_outliers": int(flags.sum()),
                f"{prefix}_outlier_frac": float(flags.mean()),
            }
        )
    return pd.DataFrame(rows), work


def _plot_endpoint_scatter(
    flights: pd.DataFrame,
    prefix: str,
    out_path: Path,
    title: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=False, sharey=False)
    groups = [("Landung", axes[0]), ("Start", axes[1])]
    cmap = plt.get_cmap("tab10")
    flow_colors = {flow: cmap(i % 10) for i, flow in enumerate(FLOW_ORDER)}
    x_col = f"{prefix}_longitude"
    y_col = f"{prefix}_latitude"
    flag_col = f"{prefix}_is_outlier"

    for ad, ax in groups:
        sub = flights[flights["A/D"] == ad].copy()
        sub_flows = [f for f in FLOW_ORDER if f.startswith(ad + "_")]
        for flow in sub_flows:
            part = sub[sub["flow"] == flow]
            if part.empty:
                continue
            normal = part[~part[flag_col]]
            outliers = part[part[flag_col]]
            ax.scatter(
                normal[x_col],
                normal[y_col],
                s=10,
                alpha=0.35,
                color=flow_colors[flow],
                label=f"{flow} (n={len(part)})",
            )
            if not outliers.empty:
                ax.scatter(
                    outliers[x_col],
                    outliers[y_col],
                    s=20,
                    alpha=0.9,
                    color=flow_colors[flow],
                    marker="x",
                )
        ax.set_title("Arrivals" if ad == "Landung" else "Departures")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8, frameon=True)

    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    preprocessed_path = _normalize_launch_path(args.preprocessed)
    flights_raw = _load_flights(preprocessed_path, args.aircraft_type_match_filter)
    flights = _build_flight_endpoints(flights_raw)

    start_summary, start_work = _summarize_endpoint(flights, prefix="start", robust_k=args.robust_k)
    end_summary, end_work = _summarize_endpoint(start_work, prefix="end", robust_k=args.robust_k)
    anchor_summary, anchor_work = _summarize_endpoint(end_work, prefix="anchor", robust_k=args.robust_k)

    stem = preprocessed_path.stem
    suffix = args.aircraft_type_match_filter
    outdir = args.outdir / f"{stem}_{suffix}"
    outdir.mkdir(parents=True, exist_ok=True)

    start_summary.to_csv(outdir / "flow_start_summary.csv", index=False)
    end_summary.to_csv(outdir / "flow_end_summary.csv", index=False)
    anchor_summary.to_csv(outdir / "flow_anchor_summary.csv", index=False)
    anchor_work.to_csv(outdir / "flight_endpoint_assessment.csv", index=False)

    start_outliers = anchor_work[anchor_work["start_is_outlier"]].copy()
    anchor_outliers = anchor_work[anchor_work["anchor_is_outlier"]].copy()
    start_outliers.to_csv(outdir / "flight_start_outliers.csv", index=False)
    anchor_outliers.to_csv(outdir / "flight_anchor_outliers.csv", index=False)

    _plot_endpoint_scatter(
        anchor_work,
        prefix="start",
        out_path=outdir / "start_point_scatter.png",
        title=f"Start-point consistency by flow ({stem}, filter={suffix})",
    )
    _plot_endpoint_scatter(
        anchor_work,
        prefix="anchor",
        out_path=outdir / "anchor_point_scatter.png",
        title=f"Operation-aware anchor consistency by flow ({stem}, filter={suffix})",
    )

    metadata = {
        "preprocessed_csv": str(preprocessed_path),
        "aircraft_type_match_filter": args.aircraft_type_match_filter,
        "n_flights": int(len(flights)),
        "robust_k": float(args.robust_k),
        "start_summary_csv": str(outdir / "flow_start_summary.csv"),
        "end_summary_csv": str(outdir / "flow_end_summary.csv"),
        "anchor_summary_csv": str(outdir / "flow_anchor_summary.csv"),
        "flight_endpoint_assessment_csv": str(outdir / "flight_endpoint_assessment.csv"),
        "flight_start_outliers_csv": str(outdir / "flight_start_outliers.csv"),
        "flight_anchor_outliers_csv": str(outdir / "flight_anchor_outliers.csv"),
        "start_scatter_png": str(outdir / "start_point_scatter.png"),
        "anchor_scatter_png": str(outdir / "anchor_point_scatter.png"),
    }
    (outdir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved endpoint audit to {outdir}")


if __name__ == "__main__":
    main()
