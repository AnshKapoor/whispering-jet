"""Pairwise distance scale audit across Euclidean, DTW, and Frechet.

This script helps compare raw distance scales per flow by sampling flights from
preprocessed trajectory files and computing pairwise distances.

Outputs:
- summary_by_flow.csv: mean/median (and counts) per metric and flow
- sampled_flights.csv: actual sampled flight IDs per flow
- pairwise_distances.csv: per-pair distances for sampled flights

Assumed input columns (minimum):
- flight_id, step, x_utm, y_utm, A/D, Runway
Optional metadata columns used in sampled_flights.csv:
- icao24, callsign
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from distance_metrics import dtw_distance, euclidean_distance, discrete_frechet_distance


@dataclass
class FlightTraj:
    flight_id: str
    flow: str
    xy: np.ndarray
    icao24: str
    callsign: str


def _flow_label(ad: str, runway: str) -> str:
    ad = str(ad).strip()
    runway = str(runway).strip()
    return f"{ad}_{runway}"


def _load_flights(preprocessed_csv: Path, allowed_flows: set[str] | None) -> list[FlightTraj]:
    df = pd.read_csv(preprocessed_csv)
    required = {"flight_id", "step", "x_utm", "y_utm", "A/D", "Runway"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{preprocessed_csv}: missing required columns: {sorted(missing)}")

    work = df.copy()
    work["flow"] = work.apply(lambda r: _flow_label(r["A/D"], r["Runway"]), axis=1)
    if allowed_flows is not None:
        work = work[work["flow"].isin(allowed_flows)]
    work = work.sort_values(["flow", "flight_id", "step"])

    flights: list[FlightTraj] = []
    for (flow, fid), grp in work.groupby(["flow", "flight_id"], sort=True):
        xy = grp[["x_utm", "y_utm"]].to_numpy(dtype=float)
        if xy.shape[0] < 2:
            continue
        if not np.isfinite(xy).all():
            continue
        icao24 = str(grp["icao24"].iloc[0]) if "icao24" in grp.columns else ""
        callsign = str(grp["callsign"].iloc[0]) if "callsign" in grp.columns else ""
        flights.append(
            FlightTraj(
                flight_id=str(fid),
                flow=str(flow),
                xy=xy,
                icao24=icao24,
                callsign=callsign,
            )
        )
    return flights


def _sample_per_flow(
    flights: list[FlightTraj], sample_per_flow: int, seed: int
) -> dict[str, list[FlightTraj]]:
    by_flow: dict[str, list[FlightTraj]] = {}
    for f in flights:
        by_flow.setdefault(f.flow, []).append(f)

    rng = np.random.default_rng(seed)
    sampled: dict[str, list[FlightTraj]] = {}
    for flow, items in sorted(by_flow.items()):
        if len(items) <= sample_per_flow:
            sampled[flow] = items
            continue
        idx = rng.choice(len(items), size=sample_per_flow, replace=False)
        sampled[flow] = [items[i] for i in sorted(idx)]
    return sampled


def _iter_pairs(n: int, max_pairs: int | None, seed: int) -> Iterable[tuple[int, int]]:
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    if max_pairs is None or max_pairs >= len(all_pairs):
        return all_pairs
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(all_pairs), size=max_pairs, replace=False)
    idx.sort()
    return [all_pairs[k] for k in idx]


def _safe_euclidean(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return float("nan")
    return float(euclidean_distance(a, b))


def _compute_flow_pairs(
    flights: list[FlightTraj], dtw_window: int | None, max_pairs: int | None, seed: int
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    n = len(flights)
    if n < 2:
        empty = pd.DataFrame(
            columns=[
                "flight_id_a",
                "flight_id_b",
                "euclidean",
                "dtw",
                "frechet",
                "len_a",
                "len_b",
            ]
        )
        return empty, {
            "n_pairs_total": 0,
            "n_pairs_used": 0,
            "euclidean_mean": float("nan"),
            "euclidean_median": float("nan"),
            "dtw_mean": float("nan"),
            "dtw_median": float("nan"),
            "frechet_mean": float("nan"),
            "frechet_median": float("nan"),
        }

    pairs = _iter_pairs(n, max_pairs=max_pairs, seed=seed)
    rows: list[dict[str, object]] = []
    total = len(pairs)
    t0 = time.perf_counter()
    for k, (i, j) in enumerate(pairs, start=1):
        fa, fb = flights[i], flights[j]
        rows.append(
            {
                "flight_id_a": fa.flight_id,
                "flight_id_b": fb.flight_id,
                "euclidean": _safe_euclidean(fa.xy, fb.xy),
                "dtw": float(dtw_distance(fa.xy, fb.xy, window_size=dtw_window)),
                "frechet": float(discrete_frechet_distance(fa.xy, fb.xy)),
                "len_a": int(fa.xy.shape[0]),
                "len_b": int(fb.xy.shape[0]),
            }
        )
        if total >= 200 and (k % max(1, total // 10) == 0):
            elapsed = time.perf_counter() - t0
            print(f"    progress {k}/{total} ({elapsed:.1f}s)", flush=True)

    pair_df = pd.DataFrame(rows)
    summary = {
        "n_pairs_total": n * (n - 1) // 2,
        "n_pairs_used": int(len(pair_df)),
        "euclidean_mean": float(pair_df["euclidean"].mean(skipna=True)),
        "euclidean_median": float(pair_df["euclidean"].median(skipna=True)),
        "dtw_mean": float(pair_df["dtw"].mean(skipna=True)),
        "dtw_median": float(pair_df["dtw"].median(skipna=True)),
        "frechet_mean": float(pair_df["frechet"].mean(skipna=True)),
        "frechet_median": float(pair_df["frechet"].median(skipna=True)),
    }
    return pair_df, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Pairwise distance scale audit by flow.")
    parser.add_argument(
        "--preprocessed",
        nargs="+",
        required=True,
        help="One or more preprocessed CSV paths.",
    )
    parser.add_argument(
        "--sample-per-flow",
        type=int,
        default=250,
        help="Flights sampled per flow (typical range 200-300).",
    )
    parser.add_argument(
        "--max-pairs-per-flow",
        type=int,
        default=None,
        help="Optional cap on number of evaluated pairs per flow for runtime control.",
    )
    parser.add_argument(
        "--dtw-window",
        type=int,
        default=None,
        help="Optional Sakoe-Chiba DTW window size.",
    )
    parser.add_argument(
        "--flows",
        nargs="*",
        default=None,
        help="Optional flow whitelist, e.g. Landung_09L Start_27R",
    )
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument(
        "--outdir",
        default="output/eda/pairwise_distance_scales",
        help="Output directory.",
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    allowed_flows = set(args.flows) if args.flows else None

    summary_rows: list[dict[str, object]] = []
    sampled_rows: list[dict[str, object]] = []
    pair_rows: list[pd.DataFrame] = []

    for p in args.preprocessed:
        path = Path(p)
        print(f"[load] {path}", flush=True)
        flights = _load_flights(path, allowed_flows=allowed_flows)
        sampled_by_flow = _sample_per_flow(
            flights,
            sample_per_flow=args.sample_per_flow,
            seed=args.seed,
        )

        for flow, flow_flights in sampled_by_flow.items():
            print(
                f"[flow] {path.name} {flow} sampled={len(flow_flights)} max_pairs={args.max_pairs_per_flow}",
                flush=True,
            )
            for f in flow_flights:
                sampled_rows.append(
                    {
                        "preprocessed_file": path.name,
                        "preprocessed_path": str(path),
                        "flow": flow,
                        "flight_id": f.flight_id,
                        "icao24": f.icao24,
                        "callsign": f.callsign,
                        "n_points": int(f.xy.shape[0]),
                    }
                )

            pair_df, stats = _compute_flow_pairs(
                flow_flights,
                dtw_window=args.dtw_window,
                max_pairs=args.max_pairs_per_flow,
                seed=args.seed,
            )
            if not pair_df.empty:
                pair_df.insert(0, "flow", flow)
                pair_df.insert(0, "preprocessed_path", str(path))
                pair_df.insert(0, "preprocessed_file", path.name)
                pair_rows.append(pair_df)

            summary_rows.append(
                {
                    "preprocessed_file": path.name,
                    "preprocessed_path": str(path),
                    "flow": flow,
                    "n_flights_sampled": int(len(flow_flights)),
                    **stats,
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    sampled_df = pd.DataFrame(sampled_rows)
    pairs_df = pd.concat(pair_rows, ignore_index=True) if pair_rows else pd.DataFrame()

    summary_path = outdir / "summary_by_flow.csv"
    sampled_path = outdir / "sampled_flights.csv"
    pairs_path = outdir / "pairwise_distances.csv"
    summary_df.to_csv(summary_path, index=False)
    sampled_df.to_csv(sampled_path, index=False)
    pairs_df.to_csv(pairs_path, index=False)

    print(f"[done] summary: {summary_path}")
    print(f"[done] sampled flights: {sampled_path}")
    print(f"[done] pairwise distances: {pairs_path}")


if __name__ == "__main__":
    main()
