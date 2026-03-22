"""Interactive discrete Frechet distance demo for two flights from a preprocessed CSV.

This utility mirrors the project Frechet distance semantics and exposes:
- trajectory point tables
- local pairwise Euclidean distance matrix
- discrete Frechet dynamic-programming matrix
- a backtracked bottleneck path
- the critical pair that determines the final Frechet value
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

from distance_metrics import discrete_frechet_distance, euclidean_distance


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
    df = df.sort_values(["flight_id", "step"]).copy()
    for fid, grp in df.groupby("flight_id", sort=True):
        xy = grp[["x_utm", "y_utm"]].to_numpy(dtype=float)
        if xy.shape[0] < 2:
            continue
        ad = str(grp["A/D"].iloc[0]) if "A/D" in grp.columns else ""
        runway = str(grp["Runway"].iloc[0]) if "Runway" in grp.columns else ""
        flow = f"{ad}_{runway}".strip("_")
        icao24 = str(grp["icao24"].iloc[0]) if "icao24" in grp.columns else ""
        callsign = str(grp["callsign"].iloc[0]) if "callsign" in grp.columns else ""
        records[str(fid)] = FlightRecord(str(fid), flow, icao24, callsign, xy)
    if not records:
        raise ValueError(f"No valid flights found in {path}")
    return records


def _choose_flights(
    flights: dict[str, FlightRecord], flight_a: str | None, flight_b: str | None, seed: int
) -> tuple[FlightRecord, FlightRecord]:
    if flight_a and flight_b:
        try:
            return flights[str(flight_a)], flights[str(flight_b)]
        except KeyError as exc:
            raise ValueError(f"Flight id not found: {exc}") from exc
    keys = sorted(flights.keys())
    if len(keys) < 2:
        raise ValueError("Need at least two flights for Frechet demo.")
    rng = np.random.default_rng(seed)
    chosen = rng.choice(keys, size=2, replace=False)
    return flights[str(chosen[0])], flights[str(chosen[1])]


def _compute_local_matrix(traj_a: np.ndarray, traj_b: np.ndarray) -> np.ndarray:
    diff = traj_a[:, None, :] - traj_b[None, :, :]
    return np.linalg.norm(diff, axis=2)


def _compute_frechet_tables(traj_a: np.ndarray, traj_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    local = _compute_local_matrix(traj_a, traj_b)
    n, m = local.shape
    ca = np.full((n, m), np.inf, dtype=float)
    ca[0, 0] = local[0, 0]
    for i in range(1, n):
        ca[i, 0] = max(ca[i - 1, 0], local[i, 0])
    for j in range(1, m):
        ca[0, j] = max(ca[0, j - 1], local[0, j])
    for i in range(1, n):
        for j in range(1, m):
            ca[i, j] = max(min(ca[i - 1, j], ca[i - 1, j - 1], ca[i, j - 1]), local[i, j])
    return local, ca


def _backtrack_frechet_path(local: np.ndarray, ca: np.ndarray) -> pd.DataFrame:
    i, j = local.shape[0] - 1, local.shape[1] - 1
    rows: list[dict[str, float | int]] = []
    while True:
        rows.append(
            {
                "path_step_rev": len(rows),
                "flight_a_point_idx": i,
                "flight_b_point_idx": j,
                "local_cost": float(local[i, j]),
                "frechet_dp_value": float(ca[i, j]),
            }
        )
        if i == 0 and j == 0:
            break
        candidates: list[tuple[float, int, int]] = []
        if i > 0:
            candidates.append((ca[i - 1, j], i - 1, j))
        if j > 0:
            candidates.append((ca[i, j - 1], i, j - 1))
        if i > 0 and j > 0:
            candidates.append((ca[i - 1, j - 1], i - 1, j - 1))
        _, i, j = min(candidates, key=lambda x: x[0])
    rows.reverse()
    path = pd.DataFrame(rows)
    path["path_step"] = np.arange(len(path), dtype=int)
    return path[["path_step", "flight_a_point_idx", "flight_b_point_idx", "local_cost", "frechet_dp_value"]]


def _plot_alignment(
    record_a: FlightRecord,
    record_b: FlightRecord,
    path: pd.DataFrame,
    out_path: Path,
    max_segments: int = 60,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(record_a.xy[:, 0], record_a.xy[:, 1], color="#1f77b4", linewidth=2, label=f"A {record_a.flight_id} ({record_a.flow})")
    ax.plot(record_b.xy[:, 0], record_b.xy[:, 1], color="#d62728", linewidth=2, label=f"B {record_b.flight_id} ({record_b.flow})")
    if len(path) > 0:
        sample_idx = np.linspace(0, len(path) - 1, num=min(max_segments, len(path)), dtype=int)
        sample = path.iloc[sample_idx]
        for _, row in sample.iterrows():
            ia = int(row["flight_a_point_idx"])
            ib = int(row["flight_b_point_idx"])
            ax.plot(
                [record_a.xy[ia, 0], record_b.xy[ib, 0]],
                [record_a.xy[ia, 1], record_b.xy[ib, 1]],
                color="0.6",
                alpha=0.25,
                linewidth=0.8,
            )
        critical = path.loc[path["local_cost"].idxmax()]
        ia = int(critical["flight_a_point_idx"])
        ib = int(critical["flight_b_point_idx"])
        ax.plot(
            [record_a.xy[ia, 0], record_b.xy[ib, 0]],
            [record_a.xy[ia, 1], record_b.xy[ib, 1]],
            color="#ffbf00",
            linewidth=2.5,
            label="Critical Frechet pair",
        )
    ax.scatter(record_a.xy[:, 0], record_a.xy[:, 1], s=12, color="#1f77b4")
    ax.scatter(record_b.xy[:, 0], record_b.xy[:, 1], s=12, color="#d62728")
    ax.set_xlabel("x_utm")
    ax.set_ylabel("y_utm")
    ax.set_title("Discrete Frechet distance demo")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_matrix_with_path(matrix: np.ndarray, path: pd.DataFrame, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, aspect="auto", origin="lower", cmap="viridis")
    ax.plot(
        path["flight_b_point_idx"].to_numpy(),
        path["flight_a_point_idx"].to_numpy(),
        color="#ff4d4d",
        linewidth=1.8,
        label="Frechet backtrack path",
    )
    ax.set_xlabel("Flight B point index")
    ax.set_ylabel("Flight A point index")
    ax.set_title(title)
    ax.legend(loc="upper left")
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _default_outdir(preprocessed: Path, a: str, b: str) -> Path:
    return Path("output/eda/frechet_demo") / f"{preprocessed.stem}_{a}_vs_{b}"


def _generate_toy_example(outdir: Path) -> None:
    """Write a small synthetic Frechet example for intuition."""
    toy_a = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 1.0],
            [4.0, 2.0],
        ],
        dtype=float,
    )
    toy_b = np.array(
        [
            [0.0, 0.5],
            [1.0, 0.5],
            [2.0, 0.6],
            [3.0, 1.7],
            [4.0, 2.8],
        ],
        dtype=float,
    )
    local, ca = _compute_frechet_tables(toy_a, toy_b)
    path = _backtrack_frechet_path(local, ca)
    toy_frechet = float(discrete_frechet_distance(toy_a, toy_b))

    critical = path.loc[path["local_cost"].idxmax()]
    critical_pair = {
        "flight_a_point_idx": int(critical["flight_a_point_idx"]),
        "flight_b_point_idx": int(critical["flight_b_point_idx"]),
        "local_cost": float(critical["local_cost"]),
        "frechet_distance": toy_frechet,
    }

    pd.DataFrame(local).to_csv(outdir / "toy_local_cost_matrix.csv", index=False)
    pd.DataFrame(ca).to_csv(outdir / "toy_frechet_dp_matrix.csv", index=False)
    path.to_csv(outdir / "toy_frechet_backtrack_path.csv", index=False)

    toy_summary = {
        "description": "Small synthetic pair to visualize Frechet bottleneck behavior.",
        "toy_frechet_distance": toy_frechet,
        "critical_pair": critical_pair,
    }
    (outdir / "toy_summary.json").write_text(json.dumps(toy_summary, indent=2), encoding="utf-8")

    rec_a = FlightRecord("toy_A", "toy", "", "", toy_a)
    rec_b = FlightRecord("toy_B", "toy", "", "", toy_b)
    _plot_alignment(rec_a, rec_b, path, outdir / "toy_alignment_plot.png", max_segments=30)
    _plot_matrix_with_path(local, path, "Toy local matrix with Frechet path", outdir / "toy_local_cost_with_path.png")
    _plot_matrix_with_path(ca, path, "Toy Frechet DP matrix with path", outdir / "toy_frechet_dp_with_path.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect discrete Frechet distance for two flights.")
    parser.add_argument("--preprocessed", required=True, help="Path to preprocessed CSV.")
    parser.add_argument("--flight-a", help="First flight_id. If omitted with --flight-b, random flights are chosen.")
    parser.add_argument("--flight-b", help="Second flight_id. If omitted with --flight-a, random flights are chosen.")
    parser.add_argument("--seed", type=int, default=11, help="Random seed when choosing flights automatically.")
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
    frechet_val = discrete_frechet_distance(record_a.xy, record_b.xy)
    local, ca = _compute_frechet_tables(record_a.xy, record_b.xy)
    path = _backtrack_frechet_path(local, ca)

    critical = path.loc[path["local_cost"].idxmax()]
    ia = int(critical["flight_a_point_idx"])
    ib = int(critical["flight_b_point_idx"])
    critical_pair = {
        "flight_a_point_idx": ia,
        "flight_b_point_idx": ib,
        "x_a": float(record_a.xy[ia, 0]),
        "y_a": float(record_a.xy[ia, 1]),
        "x_b": float(record_b.xy[ib, 0]),
        "y_b": float(record_b.xy[ib, 1]),
        "local_cost": float(critical["local_cost"]),
    }

    flight_a_points_csv = outdir / "flight_a_points.csv"
    flight_b_points_csv = outdir / "flight_b_points.csv"
    local_csv = outdir / "local_cost_matrix.csv"
    dp_csv = outdir / "frechet_dp_matrix.csv"
    path_csv = outdir / "frechet_backtrack_path.csv"

    flight_a_points.to_csv(flight_a_points_csv, index=False)
    flight_b_points.to_csv(flight_b_points_csv, index=False)
    pd.DataFrame(local).to_csv(local_csv, index=False)
    pd.DataFrame(ca).to_csv(dp_csv, index=False)
    path.to_csv(path_csv, index=False)

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
        "frechet_distance": frechet_val,
        "critical_pair": critical_pair,
        "flight_a_points_csv": str(flight_a_points_csv),
        "flight_b_points_csv": str(flight_b_points_csv),
        "local_cost_matrix_csv": str(local_csv),
        "frechet_dp_matrix_csv": str(dp_csv),
        "frechet_backtrack_path_csv": str(path_csv),
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    _plot_alignment(record_a, record_b, path, outdir / "alignment_plot.png")
    _plot_matrix_with_path(local, path, "Local distance matrix with Frechet path", outdir / "local_cost_with_path.png")
    _plot_matrix_with_path(ca, path, "Frechet DP matrix with backtrack path", outdir / "frechet_dp_with_path.png")

    log_lines = [
        "Discrete Frechet demo log",
        "",
        json.dumps(summary, indent=2),
        "",
        "First 10 points from flight A:",
        flight_a_points.head(10).to_string(index=False),
        "",
        "First 10 points from flight B:",
        flight_b_points.head(10).to_string(index=False),
        "",
        "Backtracked path preview (first 20 rows):",
        path.head(20).to_string(index=False),
        "",
        "Local cost matrix preview (top-left 8x8):",
        np.array2string(local[:8, :8], precision=2, suppress_small=False),
        "",
        "Frechet DP matrix preview (top-left 8x8):",
        np.array2string(ca[:8, :8], precision=2, suppress_small=False),
    ]
    (outdir / "frechet_log.txt").write_text("\n".join(log_lines), encoding="utf-8")
    _generate_toy_example(outdir)

    print(json.dumps(summary, indent=2))
    print("\nBacktracked Frechet path (first 15 rows):")
    print(path.head(15).to_string(index=False))
    print("\nCritical pair:")
    print(json.dumps(critical_pair, indent=2))


if __name__ == "__main__":
    main()
