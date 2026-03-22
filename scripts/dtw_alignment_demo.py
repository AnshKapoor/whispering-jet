"""Interactive DTW alignment demo for two flights from a preprocessed CSV.

This utility uses the same DTW configuration as the project distance layer:
- local cost: Euclidean
- step pattern: symmetric2

Inputs:
- preprocessed CSV with at least: flight_id, step, x_utm, y_utm

Outputs under output/eda/dtw_demo/ by default:
- flight_a_points.csv / flight_b_points.csv: ordered trajectory point tables
- local_cost_matrix.csv: pairwise Euclidean cost between every point pair
- accumulated_cost_matrix.csv: DTW cumulative cost matrix
- direction_matrix.csv: DTW predecessor encoding from dtw-python
- matched_points.csv: warping-path table with local costs
- euclidean_diagonal_match.csv: index-aligned point matching and costs
- path_comparison_summary.json: DTW-path vs diagonal-path cost summary
- summary.json: Euclidean, DTW, and flight metadata
- alignment_plot.png: both trajectories with matched-point line segments
- local_cost_heatmap.png / accumulated_cost_heatmap.png: matrix visualizations
- local_cost_with_paths.png: local-cost matrix with DTW path and Euclidean diagonal

Example:
    python scripts/dtw_alignment_demo.py --preprocessed output/preprocessed/preprocessed_1.csv --flight-a 11 --flight-b 201
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from distance_metrics import _dtw_python, dtw_distance, euclidean_distance


@dataclass
class FlightRecord:
    flight_id: str
    flow: str
    icao24: str
    callsign: str
    xy: np.ndarray


def _flight_points_table(record: FlightRecord) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "point_idx": np.arange(record.xy.shape[0], dtype=int),
            "x_utm": record.xy[:, 0],
            "y_utm": record.xy[:, 1],
        }
    )


def _load_flights(path: Path) -> dict[str, FlightRecord]:
    df = pd.read_csv(path)
    required = {"flight_id", "step", "x_utm", "y_utm"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    records: dict[str, FlightRecord] = {}
    sort_cols = ["flight_id", "step"]
    df = df.sort_values(sort_cols).copy()

    for fid, grp in df.groupby("flight_id", sort=True):
        xy = grp[["x_utm", "y_utm"]].to_numpy(dtype=float)
        if xy.shape[0] < 2:
            continue
        ad = str(grp["A/D"].iloc[0]) if "A/D" in grp.columns else ""
        runway = str(grp["Runway"].iloc[0]) if "Runway" in grp.columns else ""
        flow = f"{ad}_{runway}".strip("_")
        icao24 = str(grp["icao24"].iloc[0]) if "icao24" in grp.columns else ""
        callsign = str(grp["callsign"].iloc[0]) if "callsign" in grp.columns else ""
        records[str(fid)] = FlightRecord(
            flight_id=str(fid),
            flow=flow,
            icao24=icao24,
            callsign=callsign,
            xy=xy,
        )
    if not records:
        raise ValueError(f"No valid flights found in {path}")
    return records


def _choose_flights(
    flights: dict[str, FlightRecord],
    flight_a: str | None,
    flight_b: str | None,
    seed: int,
) -> tuple[FlightRecord, FlightRecord]:
    if flight_a and flight_b:
        try:
            return flights[str(flight_a)], flights[str(flight_b)]
        except KeyError as exc:
            raise ValueError(f"Flight id not found: {exc}") from exc

    keys = sorted(flights.keys())
    if len(keys) < 2:
        raise ValueError("Need at least two flights for DTW demo.")
    rng = np.random.default_rng(seed)
    chosen = rng.choice(keys, size=2, replace=False)
    return flights[str(chosen[0])], flights[str(chosen[1])]


def _alignment_demo(
    traj_a: np.ndarray, traj_b: np.ndarray, window: int | None
) -> tuple[float, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    if _dtw_python is None:
        raise ImportError("dtw-python is required for alignment path inspection.")

    kwargs = {
        "dist_method": "euclidean",
        "step_pattern": "symmetric2",
        "distance_only": False,
        "keep_internals": True,
    }
    if window is not None:
        kwargs["window_type"] = "sakoechiba"
        kwargs["window_args"] = {"window_size": int(window)}

    alignment = _dtw_python(traj_a, traj_b, **kwargs)
    idx_a = np.asarray(alignment.index1, dtype=int)
    idx_b = np.asarray(alignment.index2, dtype=int)
    local_cost = np.linalg.norm(traj_a[idx_a] - traj_b[idx_b], axis=1)

    cumulative = np.cumsum(local_cost)
    matched = pd.DataFrame(
        {
            "path_step": np.arange(idx_a.size, dtype=int),
            "flight_a_point_idx": idx_a,
            "flight_b_point_idx": idx_b,
            "x_a": traj_a[idx_a, 0],
            "y_a": traj_a[idx_a, 1],
            "x_b": traj_b[idx_b, 0],
            "y_b": traj_b[idx_b, 1],
            "local_cost": local_cost,
            "cumulative_path_cost": cumulative,
        }
    )
    return (
        float(alignment.distance),
        matched,
        np.asarray(alignment.localCostMatrix, dtype=float),
        np.asarray(alignment.costMatrix, dtype=float),
        np.asarray(alignment.directionMatrix),
    )


def _plot_alignment(
    record_a: FlightRecord,
    record_b: FlightRecord,
    matched: pd.DataFrame,
    out_path: Path,
    max_segments: int = 60,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(record_a.xy[:, 0], record_a.xy[:, 1], color="#1f77b4", linewidth=2, label=f"A {record_a.flight_id} ({record_a.flow})")
    ax.plot(record_b.xy[:, 0], record_b.xy[:, 1], color="#d62728", linewidth=2, label=f"B {record_b.flight_id} ({record_b.flow})")

    if len(matched) > 0:
        segment_idx = np.linspace(0, len(matched) - 1, num=min(max_segments, len(matched)), dtype=int)
        sample = matched.iloc[segment_idx]
        for _, row in sample.iterrows():
            ax.plot([row["x_a"], row["x_b"]], [row["y_a"], row["y_b"]], color="0.6", alpha=0.25, linewidth=0.8)

    ax.scatter(record_a.xy[:, 0], record_a.xy[:, 1], s=12, color="#1f77b4")
    ax.scatter(record_b.xy[:, 0], record_b.xy[:, 1], s=12, color="#d62728")
    ax.set_xlabel("x_utm")
    ax.set_ylabel("y_utm")
    ax.set_title("DTW alignment demo (symmetric2)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_matrix(matrix: np.ndarray, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, aspect="auto", origin="lower", cmap="viridis")
    ax.set_xlabel("Flight B point index")
    ax.set_ylabel("Flight A point index")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_matrix_with_paths(
    matrix: np.ndarray,
    matched: pd.DataFrame,
    diagonal: pd.DataFrame,
    title: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, aspect="auto", origin="lower", cmap="viridis")
    ax.plot(
        diagonal["flight_b_point_idx"].to_numpy(),
        diagonal["flight_a_point_idx"].to_numpy(),
        color="white",
        linewidth=1.5,
        linestyle="--",
        label="Euclidean diagonal",
    )
    ax.plot(
        matched["flight_b_point_idx"].to_numpy(),
        matched["flight_a_point_idx"].to_numpy(),
        color="#ff4d4d",
        linewidth=1.8,
        label="DTW path",
    )
    ax.set_xlabel("Flight B point index")
    ax.set_ylabel("Flight A point index")
    ax.set_title(title)
    ax.legend(loc="upper left")
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _euclidean_diagonal_match(traj_a: np.ndarray, traj_b: np.ndarray) -> pd.DataFrame:
    if traj_a.shape != traj_b.shape:
        raise ValueError("Euclidean diagonal comparison requires equal-length trajectories.")
    idx = np.arange(traj_a.shape[0], dtype=int)
    local_cost = np.linalg.norm(traj_a - traj_b, axis=1)
    cumulative = np.cumsum(local_cost)
    return pd.DataFrame(
        {
            "path_step": idx,
            "flight_a_point_idx": idx,
            "flight_b_point_idx": idx,
            "x_a": traj_a[:, 0],
            "y_a": traj_a[:, 1],
            "x_b": traj_b[:, 0],
            "y_b": traj_b[:, 1],
            "local_cost": local_cost,
            "cumulative_path_cost": cumulative,
        }
    )


def _write_log(
    out_path: Path,
    summary: dict[str, object],
    flight_a_points: pd.DataFrame,
    flight_b_points: pd.DataFrame,
    matched: pd.DataFrame,
    diagonal: pd.DataFrame,
    local_cost_matrix: np.ndarray,
    accumulated_cost_matrix: np.ndarray,
) -> None:
    lines: list[str] = []
    lines.append("DTW alignment demo log")
    lines.append("")
    lines.append(json.dumps(summary, indent=2))
    lines.append("")
    lines.append("First 10 points from flight A:")
    lines.append(flight_a_points.head(10).to_string(index=False))
    lines.append("")
    lines.append("First 10 points from flight B:")
    lines.append(flight_b_points.head(10).to_string(index=False))
    lines.append("Matched path preview (first 20 rows):")
    lines.append(matched.head(20).to_string(index=False))
    lines.append("")
    lines.append("Euclidean diagonal preview (first 20 rows):")
    lines.append(diagonal.head(20).to_string(index=False))
    lines.append("")
    lines.append("Local cost matrix preview (top-left 8x8):")
    lines.append(np.array2string(local_cost_matrix[:8, :8], precision=2, suppress_small=False))
    lines.append("")
    lines.append("Accumulated cost matrix preview (top-left 8x8):")
    lines.append(np.array2string(accumulated_cost_matrix[:8, :8], precision=2, suppress_small=False))
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _default_outdir(preprocessed: Path, a: str, b: str) -> Path:
    stem = preprocessed.stem
    return Path("output/eda/dtw_demo") / f"{stem}_{a}_vs_{b}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect DTW point matching for two flights.")
    parser.add_argument("--preprocessed", required=True, help="Path to preprocessed CSV.")
    parser.add_argument("--flight-a", help="First flight_id. If omitted with --flight-b, random flights are chosen.")
    parser.add_argument("--flight-b", help="Second flight_id. If omitted with --flight-a, random flights are chosen.")
    parser.add_argument("--seed", type=int, default=11, help="Random seed when choosing flights automatically.")
    parser.add_argument("--window-size", type=int, default=None, help="Optional Sakoe-Chiba window size.")
    parser.add_argument("--outdir", help="Optional explicit output directory.")
    parser.add_argument("--list-random", type=int, default=0, help="Print N random flight ids from the file and exit.")
    args = parser.parse_args()

    preprocessed = Path(args.preprocessed)
    flights = _load_flights(preprocessed)

    if args.list_random > 0:
        rng = np.random.default_rng(args.seed)
        keys = sorted(flights.keys())
        sample_n = min(int(args.list_random), len(keys))
        chosen = rng.choice(keys, size=sample_n, replace=False)
        print("Random flight_id sample:")
        for fid in chosen:
            rec = flights[str(fid)]
            print(f"  {rec.flight_id}: flow={rec.flow} icao24={rec.icao24} callsign={rec.callsign}")
        return

    record_a, record_b = _choose_flights(flights, args.flight_a, args.flight_b, args.seed)
    outdir = Path(args.outdir) if args.outdir else _default_outdir(preprocessed, record_a.flight_id, record_b.flight_id)
    outdir.mkdir(parents=True, exist_ok=True)

    flight_a_points = _flight_points_table(record_a)
    flight_b_points = _flight_points_table(record_b)
    euclid = euclidean_distance(record_a.xy, record_b.xy)
    dtw_val, matched, local_cost_matrix, accumulated_cost_matrix, direction_matrix = _alignment_demo(
        record_a.xy, record_b.xy, args.window_size
    )
    dtw_project = dtw_distance(record_a.xy, record_b.xy, window_size=args.window_size)
    diagonal = _euclidean_diagonal_match(record_a.xy, record_b.xy)

    flight_a_points_csv = outdir / "flight_a_points.csv"
    flight_b_points_csv = outdir / "flight_b_points.csv"
    matched_csv = outdir / "matched_points.csv"
    diagonal_csv = outdir / "euclidean_diagonal_match.csv"
    local_cost_csv = outdir / "local_cost_matrix.csv"
    accumulated_csv = outdir / "accumulated_cost_matrix.csv"
    direction_csv = outdir / "direction_matrix.csv"
    flight_a_points.to_csv(flight_a_points_csv, index=False)
    flight_b_points.to_csv(flight_b_points_csv, index=False)
    matched.to_csv(matched_csv, index=False)
    diagonal.to_csv(diagonal_csv, index=False)
    pd.DataFrame(local_cost_matrix).to_csv(local_cost_csv, index=False)
    pd.DataFrame(accumulated_cost_matrix).to_csv(accumulated_csv, index=False)
    pd.DataFrame(direction_matrix).to_csv(direction_csv, index=False)

    diagonal_path_cost = float(diagonal["local_cost"].sum())
    euclidean_norm_from_diagonal = float(np.sqrt(np.sum(np.square(diagonal["local_cost"].to_numpy()))))
    path_comparison = {
        "dtw_path_steps": int(len(matched)),
        "euclidean_diagonal_steps": int(len(diagonal)),
        "dtw_path_cost_sum": float(matched["local_cost"].sum()),
        "euclidean_diagonal_cost_sum": diagonal_path_cost,
        "euclidean_distance_l2": euclid,
        "euclidean_distance_from_diagonal_l2": euclidean_norm_from_diagonal,
        "dtw_minus_diagonal_sum": float(matched["local_cost"].sum() - diagonal_path_cost),
    }
    (outdir / "path_comparison_summary.json").write_text(json.dumps(path_comparison, indent=2), encoding="utf-8")

    summary = {
        "preprocessed_csv": str(preprocessed),
        "flight_a": {
            "flight_id": record_a.flight_id,
            "flow": record_a.flow,
            "icao24": record_a.icao24,
            "callsign": record_a.callsign,
            "n_points": int(record_a.xy.shape[0]),
        },
        "flight_b": {
            "flight_id": record_b.flight_id,
            "flow": record_b.flow,
            "icao24": record_b.icao24,
            "callsign": record_b.callsign,
            "n_points": int(record_b.xy.shape[0]),
        },
        "euclidean_distance": euclid,
        "dtw_distance_alignment": dtw_val,
        "dtw_distance_project_fn": dtw_project,
        "window_size": args.window_size,
        "matched_path_length": int(len(matched)),
        "flight_a_points_csv": str(flight_a_points_csv),
        "flight_b_points_csv": str(flight_b_points_csv),
        "euclidean_diagonal_match_csv": str(diagonal_csv),
        "local_cost_matrix_csv": str(local_cost_csv),
        "accumulated_cost_matrix_csv": str(accumulated_csv),
        "direction_matrix_csv": str(direction_csv),
        "matched_points_csv": str(matched_csv),
        "path_comparison_summary_json": str(outdir / "path_comparison_summary.json"),
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _plot_alignment(record_a, record_b, matched, outdir / "alignment_plot.png")
    _plot_matrix(local_cost_matrix, "Local DTW cost matrix", outdir / "local_cost_heatmap.png")
    _plot_matrix(accumulated_cost_matrix, "Accumulated DTW cost matrix", outdir / "accumulated_cost_heatmap.png")
    _plot_matrix_with_paths(
        local_cost_matrix,
        matched,
        diagonal,
        "Local cost matrix with DTW path vs Euclidean diagonal",
        outdir / "local_cost_with_paths.png",
    )
    _write_log(
        outdir / "alignment_log.txt",
        summary,
        flight_a_points,
        flight_b_points,
        matched,
        diagonal,
        local_cost_matrix,
        accumulated_cost_matrix,
    )

    print(json.dumps(summary, indent=2))
    print("\nFirst 15 matched steps:")
    print(matched.head(15).to_string(index=False))
    print("\nFirst 15 Euclidean diagonal steps:")
    print(diagonal.head(15).to_string(index=False))
    print("\nLocal cost matrix preview (top-left 8x8):")
    print(np.array2string(local_cost_matrix[:8, :8], precision=2, suppress_small=False))
    print("\nAccumulated cost matrix preview (top-left 8x8):")
    print(np.array2string(accumulated_cost_matrix[:8, :8], precision=2, suppress_small=False))
    print("\nPath comparison summary:")
    print(json.dumps(path_comparison, indent=2))


if __name__ == "__main__":
    main()
