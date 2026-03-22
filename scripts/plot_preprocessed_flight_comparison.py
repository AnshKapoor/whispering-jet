#!/usr/bin/env python
"""
Plot the same flight across multiple preprocessed CSV variants.

Inputs:
- One or more preprocessed CSVs containing at least:
  `flight_id`, coordinate columns, `A/D`, `Runway`, `icao24`, `callsign`

Outputs under `output/eda/preprocessed_flight_comparison/` by default:
- A 2x5 small-multiples plot for the selected flight
- A single overlay plot across all selected preprocessed files
- A compact CSV summary of the selected flight metadata per file
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_csvs() -> list[Path]:
    root = _repo_root() / "output" / "preprocessed"
    return [root / f"preprocessed_{i}.csv" for i in range(1, 11)]


def _load_flight(csv_path: Path, flight_id: int, xcol: str, ycol: str) -> pd.DataFrame:
    cols = ["flight_id", "step", xcol, ycol, "A/D", "Runway", "icao24", "callsign", "trajectory_serial"]
    df = pd.read_csv(csv_path, usecols=cols)
    out = df[df["flight_id"] == flight_id].copy()
    if out.empty:
        return out
    return out.sort_values("step")


def _load_flight_by_identity(
    csv_path: Path,
    identity: dict[str, object],
    xcol: str,
    ycol: str,
) -> pd.DataFrame:
    cols = ["flight_id", "step", xcol, ycol, "A/D", "Runway", "icao24", "callsign", "trajectory_serial"]
    df = pd.read_csv(csv_path, usecols=cols)
    mask = (
        df["icao24"].astype(str).eq(str(identity["icao24"]))
        & df["callsign"].astype(str).eq(str(identity["callsign"]))
        & df["A/D"].astype(str).eq(str(identity["A/D"]))
        & df["Runway"].astype(str).eq(str(identity["Runway"]))
    )
    out = df[mask].copy()
    if out.empty:
        return out
    # If multiple flights match the same identity tuple, keep the first flight_id deterministically.
    flight_ids = sorted(out["flight_id"].dropna().unique().tolist())
    out = out[out["flight_id"] == flight_ids[0]].copy()
    return out.sort_values("step")


def _metadata_row(csv_name: str, flight: pd.DataFrame) -> dict[str, object]:
    first = flight.iloc[0]
    return {
        "preprocessed_file": csv_name,
        "flight_id": int(first["flight_id"]),
        "trajectory_serial": int(first["trajectory_serial"]),
        "icao24": str(first["icao24"]),
        "callsign": str(first["callsign"]),
        "A/D": str(first["A/D"]),
        "Runway": str(first["Runway"]),
        "n_points": int(len(flight)),
    }


def _nice_label(csv_path: Path) -> str:
    return csv_path.stem.replace("preprocessed_", "pre ")


def _plot_grid(
    flights: Sequence[tuple[Path, pd.DataFrame]],
    xcol: str,
    ycol: str,
    out_path: Path,
    title: str,
) -> None:
    n = len(flights)
    ncols = 5
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 4 * nrows), sharex=True, sharey=True)
    if hasattr(axes, "ravel"):
        axes_flat = axes.ravel()
    else:
        axes_flat = [axes]

    for ax, (csv_path, flight) in zip(axes_flat, flights):
        if flight.empty:
            ax.set_title(f"{_nice_label(csv_path)}\nmissing")
            ax.axis("off")
            continue
        ax.plot(
            flight[xcol].to_numpy(),
            flight[ycol].to_numpy(),
            color="#1f77b4",
            linewidth=1.2,
            linestyle=":",
            marker="o",
            markersize=3.2,
            markerfacecolor="white",
            markeredgewidth=0.8,
        )
        ax.scatter(
            [flight[xcol].iloc[0], flight[xcol].iloc[-1]],
            [flight[ycol].iloc[0], flight[ycol].iloc[-1]],
            s=18,
            c=["#2ca02c", "#d62728"],
            zorder=3,
        )
        ax.set_title(f"{_nice_label(csv_path)}\n{len(flight)} pts", fontsize=10)
        ax.grid(True, alpha=0.25)

    for ax in axes_flat[len(flights):]:
        ax.axis("off")

    fig.suptitle(title, fontsize=15)
    fig.supxlabel(xcol)
    fig.supylabel(ycol)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_overlay(
    flights: Sequence[tuple[Path, pd.DataFrame]],
    xcol: str,
    ycol: str,
    out_path: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = plt.get_cmap("tab10", len(flights))

    for idx, (csv_path, flight) in enumerate(flights):
        if flight.empty:
            continue
        ax.plot(
            flight[xcol].to_numpy(),
            flight[ycol].to_numpy(),
            color=cmap(idx),
            linewidth=1.1,
            linestyle=":",
            marker="o",
            markersize=2.8,
            markerfacecolor="white",
            markeredgewidth=0.7,
            alpha=0.9,
            label=_nice_label(csv_path),
        )

    ax.set_title(title)
    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=9, ncol=2)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare one flight across preprocessed CSV variants.")
    parser.add_argument("--flight-id", type=int, default=1, help="Flight ID to compare across files.")
    parser.add_argument(
        "--coord-system",
        choices=["latlon", "utm"],
        default="latlon",
        help="Coordinate system to plot.",
    )
    parser.add_argument(
        "--csvs",
        nargs="*",
        default=None,
        help="Optional explicit list of preprocessed CSVs. Defaults to preprocessed_1..10.",
    )
    parser.add_argument(
        "--include-pre11",
        action="store_true",
        help="Also include preprocessed_11.csv, matched by identity fallback if needed.",
    )
    parser.add_argument(
        "--outdir",
        default="output/eda/preprocessed_flight_comparison",
        help="Output directory for plots and summary CSV.",
    )
    args = parser.parse_args()

    csvs = [Path(p) for p in args.csvs] if args.csvs else _default_csvs()
    if args.include_pre11:
        csvs.append(_repo_root() / "output" / "preprocessed" / "preprocessed_11.csv")
    missing_inputs = [p for p in csvs if not p.exists()]
    if missing_inputs:
        raise FileNotFoundError(f"Missing preprocessed inputs: {missing_inputs}")

    xcol, ycol = ("longitude", "latitude") if args.coord_system == "latlon" else ("x_utm", "y_utm")
    flights: list[tuple[Path, pd.DataFrame]] = []
    meta_rows: list[dict[str, object]] = []
    reference_csv = csvs[0]
    reference_flight = _load_flight(reference_csv, args.flight_id, xcol, ycol)
    if reference_flight.empty:
        raise ValueError(f"Flight ID {args.flight_id} was not found in {reference_csv}.")
    ref_first = reference_flight.iloc[0]
    identity = {
        "icao24": ref_first["icao24"],
        "callsign": ref_first["callsign"],
        "A/D": ref_first["A/D"],
        "Runway": ref_first["Runway"],
    }

    for idx, csv_path in enumerate(csvs):
        if idx == 0:
            flight = reference_flight.copy()
        else:
            flight = _load_flight(csv_path, args.flight_id, xcol, ycol)
            if flight.empty:
                flight = _load_flight_by_identity(csv_path, identity, xcol, ycol)
        flights.append((csv_path, flight))
        if not flight.empty:
            meta_rows.append(_metadata_row(csv_path.name, flight))

    if not meta_rows:
        raise ValueError(f"Flight ID {args.flight_id} was not found in the selected preprocessed files.")

    meta = pd.DataFrame(meta_rows)
    first = meta.iloc[0]
    title = (
        f"Flight {args.flight_id} across preprocessed_1..10 "
        f"({first['A/D']} {first['Runway']} | {first['icao24']} | {first['callsign']})"
    )

    outdir = Path(args.outdir)
    if not outdir.is_absolute():
        outdir = (_repo_root() / outdir).resolve()
    stem = f"flight_{args.flight_id:06d}_{args.coord_system}"

    _plot_grid(flights, xcol, ycol, outdir / f"{stem}_grid.png", title)
    _plot_overlay(flights, xcol, ycol, outdir / f"{stem}_overlay.png", title)
    meta.to_csv(outdir / f"{stem}_summary.csv", index=False)

    print(f"Saved {outdir / f'{stem}_grid.png'}")
    print(f"Saved {outdir / f'{stem}_overlay.png'}")
    print(f"Saved {outdir / f'{stem}_summary.csv'}")


if __name__ == "__main__":
    main()
