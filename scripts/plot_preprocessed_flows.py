#!/usr/bin/env python
"""
Plot sampled trajectories from a preprocessed CSV, colored by flow.

Flow is defined as "{A/D}_{Runway}" (e.g., "Start_09L", "Landung_27R").
Plots use latitude/longitude axes.

Outputs:
  - PNG to output/plots by default.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _is_aircraft_type_match(series: pd.Series) -> pd.Series:
    """Return a boolean mask for truthy aircraft-type matches."""

    numeric = pd.to_numeric(series, errors="coerce")
    truthy_numeric = numeric.eq(1.0)
    truthy_text = (
        series.astype("string")
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes"})
    )
    return truthy_numeric | truthy_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot preprocessed trajectories by flow.")
    parser.add_argument("--csv", required=True, help="Preprocessed CSV path.")
    parser.add_argument("--per-flow", type=int, default=200, help="Flights per flow to plot.")
    parser.add_argument("--seed", type=int, default=11, help="Random seed for sampling.")
    parser.add_argument("--out", default=None, help="Output PNG path.")
    parser.add_argument(
        "--aircraft-type-match-filter",
        choices=["all", "matched", "unmatched"],
        default="all",
        help="Filter flights by aircraft_type_match status. "
        "'matched' keeps only true/1, 'unmatched' keeps false/0 and missing.",
    )
    parser.add_argument(
        "--separate-only",
        action="store_true",
        help="Write only arrivals/departures figures, not the mixed overview.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(
        csv_path,
        usecols=["flight_id", "A/D", "Runway", "latitude", "longitude", "aircraft_type_match"],
    )
    flight_match = (
        df.groupby("flight_id", sort=False)["aircraft_type_match"]
        .first()
        .pipe(_is_aircraft_type_match)
    )
    if args.aircraft_type_match_filter == "matched":
        keep_ids = set(flight_match[flight_match].index.astype(int).tolist())
        df = df[df["flight_id"].isin(keep_ids)].copy()
        if df.empty:
            raise ValueError("No flights left after filtering aircraft_type_match == true.")
    elif args.aircraft_type_match_filter == "unmatched":
        keep_ids = set(flight_match[~flight_match].index.astype(int).tolist())
        df = df[df["flight_id"].isin(keep_ids)].copy()
        if df.empty:
            raise ValueError("No flights left after filtering aircraft_type_match == false/missing.")

    df["flow_label"] = df["A/D"].astype(str).str.strip() + "_" + df["Runway"].astype(str).str.strip()

    # Sample flight IDs per flow
    sampled_ids = []
    for flow, grp in df.groupby("flow_label"):
        ids = grp["flight_id"].dropna().unique()
        if len(ids) == 0:
            continue
        take = min(args.per_flow, len(ids))
        sampled = pd.Series(ids).sample(take, random_state=args.seed).tolist()
        sampled_ids.extend(sampled)

    sample_df = df[df["flight_id"].isin(sampled_ids)].copy()
    if sample_df.empty:
        raise ValueError("No trajectories found for plotting.")

    flows = sorted(sample_df["flow_label"].unique())
    # Four direction-consistent groups:
    # Start 09L == Landung 27R, Landung 09L == Start 27R,
    # Start 09R == Landung 27L, Landung 09R == Start 27L.
    group_colors = [
        (0.9, 0.1, 0.1),   # red
        (0.1, 0.7, 0.2),   # green
        (0.1, 0.4, 0.9),   # blue
        (0.95, 0.8, 0.1),  # yellow
    ]
    group_map = {
        ("Start", "09L"): 0,
        ("Landung", "27R"): 0,
        ("Landung", "09L"): 1,
        ("Start", "27R"): 1,
        ("Start", "09R"): 2,
        ("Landung", "27L"): 2,
        ("Landung", "09R"): 3,
        ("Start", "27L"): 3,
    }

    color_map = {}
    for flow in flows:
        ad, runway = flow.split("_", 1)
        idx = group_map.get((ad, runway))
        if idx is None:
            color_map[flow] = (0.2, 0.2, 0.2)
        else:
            color_map[flow] = group_colors[idx]

    if args.aircraft_type_match_filter == "matched":
        title_filter = "matched aircraft types only"
    elif args.aircraft_type_match_filter == "unmatched":
        title_filter = "unmatched or missing aircraft types only"
    else:
        title_filter = "all flights"

    def plot_subset(subset_df: pd.DataFrame, title_suffix: str, out_path: Path) -> None:
        fig, ax = plt.subplots(figsize=(10, 8))
        subset_flows = sorted(subset_df["flow_label"].unique())
        for flow in subset_flows:
            flow_df = subset_df[subset_df["flow_label"] == flow]
            for _, traj in flow_df.groupby("flight_id"):
                ax.plot(
                    traj["longitude"].to_numpy(),
                    traj["latitude"].to_numpy(),
                    color=color_map.get(flow, (0.2, 0.2, 0.2)),
                    alpha=0.25,
                    linewidth=0.7,
                )
            ax.plot([], [], color=color_map.get(flow, (0.2, 0.2, 0.2)),
                    label=f"{flow} (n~{len(flow_df['flight_id'].unique())})")

        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(
            f"Preprocessed trajectories {title_suffix} ({title_filter})\n"
            f"sampled {args.per_flow}/flow"
        )
        ax.legend(loc="upper right", fontsize=8, frameon=True)
        ax.grid(True, alpha=0.3)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"Saved {out_path}")

    # Build subsets
    arrivals = sample_df[sample_df["A/D"].astype(str).str.strip().eq("Landung")]
    departures = sample_df[sample_df["A/D"].astype(str).str.strip().eq("Start")]

    out_dir = _repo_root() / "output" / "plots"
    if args.out:
        base = Path(args.out)
        base_stem = base.stem
        out_dir = base.parent
    else:
        base_stem = f"{csv_path.stem}_flows_{args.per_flow}"

    plot_subset(arrivals, "arrivals only", out_dir / f"{base_stem}_arrivals.png")
    plot_subset(departures, "departures only", out_dir / f"{base_stem}_departures.png")
    if not args.separate_only:
        plot_subset(sample_df, "arrivals + departures", out_dir / f"{base_stem}_mixed.png")


if __name__ == "__main__":
    main()
