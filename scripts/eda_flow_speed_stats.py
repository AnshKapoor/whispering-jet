"""Summarize per-flow stepwise speed proxies from a preprocessed trajectory CSV.

The script is intended as an ad-hoc diagnostic for velocity leakage concerns in
index-based Euclidean clustering. Preprocessed trajectory CSVs in this
repository do not carry true timestamps, so the primary quantity reported is
segment length per step in meters:

    d_step[i] = sqrt((x[i+1]-x[i])^2 + (y[i+1]-y[i])^2)

If ``--dt-seconds`` is supplied, the script additionally reports a derived
speed-like quantity ``d_step / dt`` in m/s. The underlying diagnostic remains
the same: large within-flow variation in segment lengths over normalized step
index suggests that index-based Euclidean distance may be conflating path shape
with traversal rate.

Outputs:
- ``flow_summary.csv``: flow-level summary over sampled flights
- ``flight_summary.csv``: per-flight summary statistics
- ``flow_step_profile.csv``: percentile bands by flow and step segment index
- ``metadata.json``: run metadata and key assumptions
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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

PERCENTILES = [5, 10, 25, 50, 75, 90, 95]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute per-flow stepwise speed proxy statistics from a preprocessed CSV."
    )
    parser.add_argument(
        "--preprocessed",
        required=True,
        help="Preprocessed CSV path. Launch-config labels may append ' | ...'; the script strips that suffix.",
    )
    parser.add_argument(
        "--sample-per-flow",
        type=int,
        default=1000,
        help="Maximum flights sampled per flow for detailed statistics.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when downsampling flights within a flow.",
    )
    parser.add_argument(
        "--dt-seconds",
        type=float,
        default=None,
        help="Optional constant seconds per step to derive a speed-like m/s metric.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Optional output directory. Defaults to output/eda/flow_speed_stats/<stem>/",
    )
    return parser.parse_args()


def _normalize_launch_path(raw: str) -> Path:
    path_text = raw.split("|", 1)[0].strip()
    return Path(path_text)


def _require_columns(df: pd.DataFrame, required: list[str]) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _sample_flow_flights(df: pd.DataFrame, sample_per_flow: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    keep_ids: list[int] = []
    dedup = df[["flow", "flight_id"]].drop_duplicates()

    for flow in FLOW_ORDER:
        flow_ids = dedup.loc[dedup["flow"] == flow, "flight_id"].to_numpy()
        if flow_ids.size == 0:
            continue
        if flow_ids.size <= sample_per_flow:
            chosen = flow_ids
        else:
            chosen = np.sort(rng.choice(flow_ids, size=sample_per_flow, replace=False))
        keep_ids.extend(chosen.tolist())

    return df[df["flight_id"].isin(keep_ids)].copy()


def _compute_segment_rows(df: pd.DataFrame, dt_seconds: float | None) -> pd.DataFrame:
    work = df.sort_values(["flow", "flight_id", "step"]).copy()
    grouped = work.groupby(["flow", "flight_id"], sort=False)
    work["dx"] = grouped["x_utm"].diff()
    work["dy"] = grouped["y_utm"].diff()
    work["segment_index"] = grouped.cumcount() - 1
    work["segment_length_m"] = np.sqrt(work["dx"] ** 2 + work["dy"] ** 2)
    work = work[work["segment_index"] >= 0].copy()
    work["segment_index"] = work["segment_index"].astype(int)

    if dt_seconds is not None:
        work["speed_mps"] = work["segment_length_m"] / float(dt_seconds)
        work["metric_value"] = work["speed_mps"]
        work["metric_name"] = "speed_mps"
    else:
        work["metric_value"] = work["segment_length_m"]
        work["metric_name"] = "segment_length_m"
    return work


def _percentile_stats(values: np.ndarray, prefix: str) -> dict[str, float]:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        return {f"{prefix}_p{p}": np.nan for p in PERCENTILES}
    out = {}
    for p in PERCENTILES:
        out[f"{prefix}_p{p}"] = float(np.percentile(clean, p))
    return out


def _build_flight_summary(segments: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for (flow, flight_id), grp in segments.groupby(["flow", "flight_id"], sort=True):
        vals = grp["metric_value"].to_numpy(dtype=float)
        row: dict[str, float | int | str] = {
            "flow": flow,
            "flight_id": int(flight_id),
            "n_segments": int(vals.size),
            "metric_mean": float(np.mean(vals)),
            "metric_std": float(np.std(vals, ddof=0)),
            "metric_min": float(np.min(vals)),
            "metric_max": float(np.max(vals)),
        }
        row.update(_percentile_stats(vals, "metric"))
        rows.append(row)
    return pd.DataFrame(rows)


def _build_flow_summary(
    original_df: pd.DataFrame,
    sampled_df: pd.DataFrame,
    segments: pd.DataFrame,
    flight_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    total_flights_by_flow = (
        original_df[["flow", "flight_id"]].drop_duplicates().groupby("flow")["flight_id"].nunique().to_dict()
    )
    sampled_flights_by_flow = (
        sampled_df[["flow", "flight_id"]].drop_duplicates().groupby("flow")["flight_id"].nunique().to_dict()
    )

    for flow in FLOW_ORDER:
        flight_grp = flight_summary[flight_summary["flow"] == flow]
        seg_grp = segments[segments["flow"] == flow]
        if flight_grp.empty or seg_grp.empty:
            continue

        flight_medians = flight_grp["metric_p50"].to_numpy(dtype=float)
        flight_means = flight_grp["metric_mean"].to_numpy(dtype=float)
        pooled = seg_grp["metric_value"].to_numpy(dtype=float)

        row: dict[str, float | int | str] = {
            "flow": flow,
            "n_flights_total": int(total_flights_by_flow.get(flow, 0)),
            "n_flights_sampled": int(sampled_flights_by_flow.get(flow, 0)),
            "n_segments_pooled": int(pooled.size),
            "flight_median_of_medians": float(np.median(flight_medians)),
            "flight_mean_of_medians": float(np.mean(flight_medians)),
            "flight_median_of_means": float(np.median(flight_means)),
            "flight_mean_of_means": float(np.mean(flight_means)),
            "pooled_metric_mean": float(np.mean(pooled)),
            "pooled_metric_std": float(np.std(pooled, ddof=0)),
        }
        row.update(_percentile_stats(flight_medians, "flight_median"))
        row.update(_percentile_stats(flight_means, "flight_mean"))
        row.update(_percentile_stats(pooled, "pooled_metric"))
        rows.append(row)

    return pd.DataFrame(rows)


def _build_flow_step_profile(segments: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for (flow, segment_index), grp in segments.groupby(["flow", "segment_index"], sort=True):
        vals = grp["metric_value"].to_numpy(dtype=float)
        row: dict[str, float | int | str] = {
            "flow": flow,
            "segment_index": int(segment_index),
            "n_flights": int(grp["flight_id"].nunique()),
            "metric_mean": float(np.mean(vals)),
            "metric_std": float(np.std(vals, ddof=0)),
        }
        row.update(_percentile_stats(vals, "metric"))
        rows.append(row)
    return pd.DataFrame(rows)


def _write_metadata(
    outdir: Path,
    input_path: Path,
    original_df: pd.DataFrame,
    sampled_df: pd.DataFrame,
    segments: pd.DataFrame,
    dt_seconds: float | None,
    sample_per_flow: int,
    seed: int,
) -> None:
    total_flights = int(original_df["flight_id"].nunique())
    sampled_flights = int(sampled_df["flight_id"].nunique())
    metadata = {
        "input_path": str(input_path),
        "sample_per_flow": int(sample_per_flow),
        "seed": int(seed),
        "dt_seconds": None if dt_seconds is None else float(dt_seconds),
        "reported_metric": "speed_mps" if dt_seconds is not None else "segment_length_m",
        "note": (
            "Without timestamps in the preprocessed CSV, the default diagnostic uses segment length per step "
            "as a speed proxy. If a constant dt is known externally, --dt-seconds converts it to m/s."
        ),
        "n_flights_total": total_flights,
        "n_flights_sampled": sampled_flights,
        "flows_present": sorted(sampled_df["flow"].dropna().unique().tolist()),
        "max_segment_index": int(segments["segment_index"].max()) if not segments.empty else None,
    }
    (outdir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    args = _parse_args()
    input_path = _normalize_launch_path(args.preprocessed)
    if not input_path.exists():
        raise FileNotFoundError(f"Preprocessed CSV not found: {input_path}")

    outdir = args.outdir
    if outdir is None:
        outdir = Path("output") / "eda" / "flow_speed_stats" / input_path.stem
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path, usecols=["step", "x_utm", "y_utm", "A/D", "Runway", "flight_id"])
    _require_columns(df, ["step", "x_utm", "y_utm", "A/D", "Runway", "flight_id"])

    df["flow"] = df["A/D"].astype(str) + "_" + df["Runway"].astype(str)
    df = df[df["flow"].isin(FLOW_ORDER)].copy()
    df["flight_id"] = pd.to_numeric(df["flight_id"], errors="coerce")
    df["step"] = pd.to_numeric(df["step"], errors="coerce")
    df = df.dropna(subset=["flight_id", "step", "x_utm", "y_utm"]).copy()
    df["flight_id"] = df["flight_id"].astype(int)
    df["step"] = df["step"].astype(int)

    sampled_df = _sample_flow_flights(df, sample_per_flow=args.sample_per_flow, seed=args.seed)
    segments = _compute_segment_rows(sampled_df, dt_seconds=args.dt_seconds)
    if segments.empty:
        raise RuntimeError("No valid segment rows were produced from the sampled flights.")

    flight_summary = _build_flight_summary(segments)
    flow_summary = _build_flow_summary(df, sampled_df, segments, flight_summary)
    flow_step_profile = _build_flow_step_profile(segments)

    flow_summary.to_csv(outdir / "flow_summary.csv", index=False)
    flight_summary.to_csv(outdir / "flight_summary.csv", index=False)
    flow_step_profile.to_csv(outdir / "flow_step_profile.csv", index=False)
    _write_metadata(
        outdir=outdir,
        input_path=input_path,
        original_df=df,
        sampled_df=sampled_df,
        segments=segments,
        dt_seconds=args.dt_seconds,
        sample_per_flow=args.sample_per_flow,
        seed=args.seed,
    )

    print(f"Saved flow speed statistics to {outdir}", flush=True)
    print(f"Reported metric: {'speed_mps' if args.dt_seconds is not None else 'segment_length_m'}", flush=True)


if __name__ == "__main__":
    main()
