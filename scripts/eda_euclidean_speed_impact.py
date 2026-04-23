"""Quantify speed/progression impact on index-based Euclidean distances.

This EDA script compares the current clustering-style Euclidean trajectory
representation against two alternatives:

- equal-arc-length Euclidean geometry
- DTW on the current fixed-step representation

It also builds speed profiles from matched trajectories and measures whether the
distortion between index-based and arc-length Euclidean distances tracks speed
differences. Outputs are written under ``output/eda/euclidean_speed_impact/``.

Key outputs:
- ``flow_summary.csv``: one row per flow with distance and correlation summaries
- ``overall_summary.json``: weighted/global summary across all flows
- ``pair_metrics.csv``: per-pair diagnostics for sampled flights
- ``flight_summary.csv``: per-flight raw/profile metadata
- ``metadata.json``: run metadata and pipeline validation diagnostics
- plots:
  - ``speed_vs_delta_shape.png``
  - ``euclidean_vs_dtw.png``
  - ``delta_shape_by_flow.png``
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backbone_tracks.config import load_config
from backbone_tracks.io import add_utm_coordinates, ensure_required_columns, load_monthly_csvs
from backbone_tracks.preprocessing import preprocess_flights
from backbone_tracks.segmentation import segment_flights
from distance_metrics import dtw_distance
from save_preprocessed import apply_repetition_dedup


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
KNOT_TO_MPS = 0.514444


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assess whether speed/progression effects distort Euclidean trajectory distances."
    )
    parser.add_argument(
        "--preprocessed",
        required=True,
        help="Preprocessed CSV path. Launch-config labels may append ' | ...'; the script strips that suffix.",
    )
    parser.add_argument(
        "--config",
        default="config/backbone_full.yaml",
        help="Config used to generate the target preprocessed CSV.",
    )
    parser.add_argument(
        "--sample-per-flow",
        type=int,
        default=200,
        help="Maximum number of flights sampled per flow for pairwise analysis.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for flight and optional pair sampling.",
    )
    parser.add_argument(
        "--dtw-window-size",
        type=int,
        default=8,
        help="Sakoe-Chiba window size used for DTW pairwise diagnostics.",
    )
    parser.add_argument(
        "--max-pairs-per-flow",
        type=int,
        default=None,
        help="Optional cap on analyzed pairs per flow. When omitted, all sampled pairs are used.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Optional output directory. Defaults to output/eda/euclidean_speed_impact/<stem>_n<sample>_w<dtw>/.",
    )
    return parser.parse_args()


def _normalize_launch_path(raw: str) -> Path:
    return Path(raw.split("|", 1)[0].strip())


def _percentile_stats(values: np.ndarray, prefix: str) -> dict[str, float]:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        return {f"{prefix}_p{p}": np.nan for p in PERCENTILES}
    return {f"{prefix}_p{p}": float(np.percentile(clean, p)) for p in PERCENTILES}


def _corr(series_a: pd.Series, series_b: pd.Series, method: str) -> float:
    frame = pd.concat([series_a, series_b], axis=1).dropna()
    if len(frame) < 3:
        return float("nan")
    if frame.iloc[:, 0].nunique() < 2 or frame.iloc[:, 1].nunique() < 2:
        return float("nan")
    return float(frame.iloc[:, 0].corr(frame.iloc[:, 1], method=method))


def _sample_flight_ids(df: pd.DataFrame, sample_per_flow: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    dedup = df[["flow", "flight_id"]].drop_duplicates()
    keep_ids: list[int] = []
    for flow in FLOW_ORDER:
        flow_ids = dedup.loc[dedup["flow"] == flow, "flight_id"].to_numpy(dtype=int)
        if flow_ids.size == 0:
            continue
        if flow_ids.size <= sample_per_flow:
            chosen = np.sort(flow_ids)
        else:
            chosen = np.sort(rng.choice(flow_ids, size=sample_per_flow, replace=False))
        keep_ids.extend(chosen.tolist())
    return keep_ids


def _load_actual_preprocessed(path: Path) -> pd.DataFrame:
    usecols = ["step", "x_utm", "y_utm", "A/D", "Runway", "flight_id", "icao24", "callsign"]
    df = pd.read_csv(path, usecols=usecols)
    df["flight_id"] = pd.to_numeric(df["flight_id"], errors="coerce").astype("Int64")
    df["step"] = pd.to_numeric(df["step"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["flight_id", "step", "x_utm", "y_utm", "A/D", "Runway"]).copy()
    df["flight_id"] = df["flight_id"].astype(int)
    df["step"] = df["step"].astype(int)
    df["flow"] = df["A/D"].astype(str) + "_" + df["Runway"].astype(str)
    df = df[df["flow"].isin(FLOW_ORDER)].copy()
    return df


def _rebuild_pipeline(
    config_path: Path,
    preprocessed_id: int,
    effective_n_points: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = load_config(config_path)
    input_cfg = cfg.get("input", {}) or {}
    preprocessing_cfg = cfg.get("preprocessing", {}) or {}
    seg_cfg = cfg.get("segmentation", {}) or {}
    coords_cfg = cfg.get("coordinates", {}) or {}
    flows_cfg = cfg.get("flows", {}) or {}
    testing_cfg = cfg.get("testing", {}) or {}

    csv_glob = str(input_cfg.get("csv_glob", "matched_trajectories/matched_trajs_*.csv"))
    parse_dates = list(input_cfg.get("parse_dates", ["timestamp"]))
    repetition_cfg = preprocessing_cfg.get("repetition_check", {}) or {}
    filter_cfg = preprocessing_cfg.get("filter", {}) or {}
    smoothing_cfg = preprocessing_cfg.get("smoothing", {}) or {}
    resampling_cfg = dict(preprocessing_cfg.get("resampling", {}) or {})
    # The stored preprocessed artifact is the ground truth for the current
    # Euclidean representation. Some historical files were generated with a
    # different n_points value than the current config default, so force the
    # rebuild to use the effective point count observed in the file.
    resampling_cfg["n_points"] = int(effective_n_points)
    flow_keys = tuple(flows_cfg.get("flow_keys", ["A/D", "Runway"]))
    use_utm = bool(coords_cfg.get("use_utm", True))
    utm_crs = str(coords_cfg.get("utm_crs", "epsg:32632"))
    max_rows = testing_cfg.get("max_rows_total")

    logging.info("Loading matched trajectories from %s", csv_glob)
    df = load_monthly_csvs(csv_glob=csv_glob, parse_dates=parse_dates, max_rows_total=max_rows)
    df = ensure_required_columns(df)
    df = apply_repetition_dedup(df, repetition_cfg=repetition_cfg, preprocessed_id=preprocessed_id)
    if use_utm:
        df = add_utm_coordinates(df, utm_crs=utm_crs)

    segmented = segment_flights(
        df,
        time_gap_sec=float(seg_cfg.get("time_gap_sec", 120)),
        distance_jump_m=float(seg_cfg.get("distance_jump_m", 600)),
        min_points_per_flight=int(seg_cfg.get("min_points_per_flight", 15)),
        split_on_identity=bool(seg_cfg.get("split_on_identity", True)),
    )

    rebuilt_current = preprocess_flights(
        segmented,
        smoothing_cfg=smoothing_cfg,
        resampling_cfg=resampling_cfg,
        filter_cfg=filter_cfg,
        use_utm=use_utm,
        flow_keys=flow_keys,
    )
    raw_mode_cfg = {"enabled": False, "method": "none"}
    rebuilt_raw = preprocess_flights(
        segmented,
        smoothing_cfg=smoothing_cfg,
        resampling_cfg=raw_mode_cfg,
        filter_cfg=filter_cfg,
        use_utm=use_utm,
        flow_keys=flow_keys,
    )

    survivor_ids = sorted(pd.to_numeric(rebuilt_current["flight_id"], errors="coerce").dropna().astype(int).unique())
    mapping = {fid: idx + 1 for idx, fid in enumerate(survivor_ids)}

    rebuilt_current = rebuilt_current[rebuilt_current["flight_id"].isin(mapping)].copy()
    rebuilt_raw = rebuilt_raw[rebuilt_raw["flight_id"].isin(mapping)].copy()
    rebuilt_current["flight_id"] = rebuilt_current["flight_id"].map(mapping).astype(int)
    rebuilt_raw["flight_id"] = rebuilt_raw["flight_id"].map(mapping).astype(int)

    rebuilt_current["flow"] = rebuilt_current["A/D"].astype(str) + "_" + rebuilt_current["Runway"].astype(str)
    rebuilt_raw["flow"] = rebuilt_raw["A/D"].astype(str) + "_" + rebuilt_raw["Runway"].astype(str)
    rebuilt_current = rebuilt_current[rebuilt_current["flow"].isin(FLOW_ORDER)].copy()
    rebuilt_raw = rebuilt_raw[rebuilt_raw["flow"].isin(FLOW_ORDER)].copy()
    return rebuilt_current, rebuilt_raw


def _validate_rebuild(actual: pd.DataFrame, rebuilt: pd.DataFrame) -> dict[str, float | int]:
    actual_view = (
        actual[["flight_id", "step", "A/D", "Runway", "x_utm", "y_utm"]]
        .sort_values(["flight_id", "step"])
        .reset_index(drop=True)
    )
    rebuilt_view = (
        rebuilt[["flight_id", "step", "A/D", "Runway", "x_utm", "y_utm"]]
        .sort_values(["flight_id", "step"])
        .reset_index(drop=True)
    )

    if len(actual_view) != len(rebuilt_view):
        raise RuntimeError(
            f"Rebuilt preprocessing row count {len(rebuilt_view)} does not match actual preprocessed row count {len(actual_view)}."
        )

    merged = actual_view.merge(
        rebuilt_view,
        on=["flight_id", "step"],
        how="outer",
        suffixes=("_actual", "_rebuilt"),
        indicator=True,
    )
    if not (merged["_merge"] == "both").all():
        bad = int((merged["_merge"] != "both").sum())
        raise RuntimeError(f"Rebuilt preprocessing could not align {bad} rows with the actual preprocessed file.")

    same_flow = (
        merged["A/D_actual"].astype(str).eq(merged["A/D_rebuilt"].astype(str))
        & merged["Runway_actual"].astype(str).eq(merged["Runway_rebuilt"].astype(str))
    )
    if not same_flow.all():
        bad = int((~same_flow).sum())
        raise RuntimeError(f"Rebuilt preprocessing disagrees on flow labels for {bad} aligned rows.")

    dx = np.abs(merged["x_utm_actual"] - merged["x_utm_rebuilt"])
    dy = np.abs(merged["y_utm_actual"] - merged["y_utm_rebuilt"])
    max_abs_dx = float(dx.max())
    max_abs_dy = float(dy.max())
    mean_abs_dx = float(dx.mean())
    mean_abs_dy = float(dy.mean())
    if max(max_abs_dx, max_abs_dy) > 1e-3:
        raise RuntimeError(
            "Rebuilt preprocessing diverges from the actual preprocessed file "
            f"(max_abs_dx={max_abs_dx:.6f}, max_abs_dy={max_abs_dy:.6f})."
        )

    return {
        "rows_checked": int(len(merged)),
        "max_abs_dx_m": max_abs_dx,
        "max_abs_dy_m": max_abs_dy,
        "mean_abs_dx_m": mean_abs_dx,
        "mean_abs_dy_m": mean_abs_dy,
        "n_flights": int(actual["flight_id"].nunique()),
    }


def _build_current_arrays(df: pd.DataFrame, flight_ids: Iterable[int]) -> dict[int, np.ndarray]:
    keep = set(int(fid) for fid in flight_ids)
    arrays: dict[int, np.ndarray] = {}
    for flight_id, grp in df[df["flight_id"].isin(keep)].groupby("flight_id", sort=True):
        traj = grp.sort_values("step")[["x_utm", "y_utm"]].to_numpy(dtype=float)
        if len(traj) >= 2 and np.isfinite(traj).all():
            arrays[int(flight_id)] = traj
    return arrays


def _resample_arclength(coords: np.ndarray, n_points: int) -> np.ndarray | None:
    if coords.ndim != 2 or coords.shape[0] < 2:
        return None
    diffs = np.diff(coords, axis=0)
    seg = np.linalg.norm(diffs, axis=1)
    cum = np.concatenate(([0.0], np.cumsum(seg)))
    total = float(cum[-1])
    if not np.isfinite(total) or total <= 0:
        return None
    target = np.linspace(0.0, total, n_points)
    x = np.interp(target, cum, coords[:, 0])
    y = np.interp(target, cum, coords[:, 1])
    return np.column_stack([x, y])


def _derive_speed_mps(df: pd.DataFrame) -> np.ndarray:
    coords = df[["x_utm", "y_utm"]].to_numpy(dtype=float)
    times = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    t = times.view("int64").to_numpy(dtype=np.int64) / 1e9
    point_speed = np.full(len(df), np.nan, dtype=float)
    if len(df) < 2:
        return point_speed
    seg_dist = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    seg_dt = np.diff(t)
    with np.errstate(divide="ignore", invalid="ignore"):
        seg_speed = np.where(seg_dt > 0, seg_dist / seg_dt, np.nan)
    if seg_speed.size == 0:
        return point_speed
    point_speed[0] = seg_speed[0]
    point_speed[-1] = seg_speed[-1]
    if len(df) > 2:
        left = seg_speed[:-1]
        right = seg_speed[1:]
        with np.errstate(invalid="ignore"):
            point_speed[1:-1] = np.nanmean(np.vstack([left, right]), axis=0)
    return point_speed


def _build_speed_profile(df: pd.DataFrame, n_points: int) -> tuple[np.ndarray | None, dict[str, float | int | str]]:
    coords = df[["x_utm", "y_utm"]].to_numpy(dtype=float)
    diffs = np.diff(coords, axis=0)
    seg = np.linalg.norm(diffs, axis=1)
    cum = np.concatenate(([0.0], np.cumsum(seg)))
    total = float(cum[-1])
    if not np.isfinite(total) or total <= 0:
        return None, {
            "raw_points": int(len(df)),
            "valid_speed_points": 0,
            "speed_source": "none",
            "groundspeed_valid_fraction": 0.0,
        }

    gs = pd.to_numeric(df.get("groundspeed"), errors="coerce").to_numpy(dtype=float)
    direct = gs * KNOT_TO_MPS
    direct_valid = np.isfinite(direct)
    derived = _derive_speed_mps(df)
    speed = direct.copy()
    speed[~direct_valid] = derived[~direct_valid]
    valid = np.isfinite(speed)
    if valid.sum() < 2:
        return None, {
            "raw_points": int(len(df)),
            "valid_speed_points": int(valid.sum()),
            "speed_source": "insufficient",
            "groundspeed_valid_fraction": float(direct_valid.mean()) if len(direct_valid) else 0.0,
        }

    progress = cum / total
    valid_progress = progress[valid]
    valid_speed = speed[valid]
    if valid_progress.size == 1:
        profile = np.full(n_points, float(valid_speed[0]), dtype=float)
    else:
        target = np.linspace(0.0, 1.0, n_points)
        profile = np.interp(target, valid_progress, valid_speed)

    source = "direct_groundspeed" if direct_valid.all() else ("mixed_groundspeed_derived" if direct_valid.any() else "derived_xy_dt")
    return profile, {
        "raw_points": int(len(df)),
        "valid_speed_points": int(valid.sum()),
        "speed_source": source,
        "groundspeed_valid_fraction": float(direct_valid.mean()) if len(direct_valid) else 0.0,
    }


def _build_raw_profiles(
    raw_df: pd.DataFrame,
    flight_ids: Iterable[int],
    n_points: int,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], pd.DataFrame]:
    keep = set(int(fid) for fid in flight_ids)
    arclength_xy: dict[int, np.ndarray] = {}
    speed_profiles: dict[int, np.ndarray] = {}
    rows: list[dict[str, float | int | str]] = []

    work = raw_df[raw_df["flight_id"].isin(keep)].copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True, errors="coerce")

    for flight_id, grp in work.groupby("flight_id", sort=True):
        grp = grp.sort_values("timestamp")
        coords = grp[["x_utm", "y_utm"]].to_numpy(dtype=float)
        arc = _resample_arclength(coords, n_points=n_points)
        speed, meta = _build_speed_profile(grp, n_points=n_points)
        flow = str(grp["A/D"].iloc[0]) + "_" + str(grp["Runway"].iloc[0])
        row: dict[str, float | int | str] = {
            "flight_id": int(flight_id),
            "flow": flow,
            "raw_points": int(meta["raw_points"]),
            "valid_speed_points": int(meta["valid_speed_points"]),
            "groundspeed_valid_fraction": float(meta["groundspeed_valid_fraction"]),
            "speed_source": str(meta["speed_source"]),
            "arc_profile_available": bool(arc is not None),
            "speed_profile_available": bool(speed is not None),
        }
        rows.append(row)
        if arc is not None:
            arclength_xy[int(flight_id)] = arc
        if speed is not None:
            speed_profiles[int(flight_id)] = speed

    return arclength_xy, speed_profiles, pd.DataFrame(rows)


def _sample_pair_indices(n_items: int, max_pairs: int, seed: int) -> set[tuple[int, int]]:
    all_pairs = [(i, j) for i in range(n_items) for j in range(i + 1, n_items)]
    if len(all_pairs) <= max_pairs:
        return set(all_pairs)
    rng = np.random.default_rng(seed)
    chosen_idx = rng.choice(len(all_pairs), size=max_pairs, replace=False)
    return {all_pairs[int(idx)] for idx in chosen_idx}


def _pair_metrics_for_flow(
    flow: str,
    flight_ids: list[int],
    current_arrays: dict[int, np.ndarray],
    arclength_xy: dict[int, np.ndarray],
    speed_profiles: dict[int, np.ndarray],
    dtw_window_size: int,
    max_pairs: int | None,
    seed: int,
) -> pd.DataFrame:
    usable_ids = [fid for fid in flight_ids if fid in current_arrays and fid in arclength_xy and fid in speed_profiles]
    if len(usable_ids) < 2:
        return pd.DataFrame()

    pair_filter: set[tuple[int, int]] | None = None
    if max_pairs is not None and max_pairs > 0:
        pair_filter = _sample_pair_indices(len(usable_ids), max_pairs=max_pairs, seed=seed)

    rows: list[dict[str, float | int | str]] = []
    for i in range(len(usable_ids)):
        fid_i = usable_ids[i]
        current_i = current_arrays[fid_i]
        arc_i = arclength_xy[fid_i]
        speed_i = speed_profiles[fid_i]
        current_i_flat = current_i.ravel()
        arc_i_flat = arc_i.ravel()
        for j in range(i + 1, len(usable_ids)):
            if pair_filter is not None and (i, j) not in pair_filter:
                continue
            fid_j = usable_ids[j]
            current_j = current_arrays[fid_j]
            arc_j = arclength_xy[fid_j]
            speed_j = speed_profiles[fid_j]

            d_index = float(np.linalg.norm(current_i_flat - current_j.ravel()))
            d_arc = float(np.linalg.norm(arc_i_flat - arc_j.ravel()))
            d_speed = float(np.linalg.norm(speed_i - speed_j))
            d_dtw = float(dtw_distance(current_i, current_j, window_size=dtw_window_size))
            delta = d_index - d_arc
            rows.append(
                {
                    "flow": flow,
                    "flight_id_a": int(fid_i),
                    "flight_id_b": int(fid_j),
                    "D_euc_index": d_index,
                    "D_euc_arclength": d_arc,
                    "delta_shape": delta,
                    "abs_delta_shape": abs(delta),
                    "D_speed": d_speed,
                    "D_dtw": d_dtw,
                }
            )
    return pd.DataFrame(rows)


def _build_flow_summary(
    pair_df: pd.DataFrame,
    sampled_actual: pd.DataFrame,
    flight_summary: pd.DataFrame,
) -> pd.DataFrame:
    total_flights_by_flow = (
        sampled_actual[["flow", "flight_id"]]
        .drop_duplicates()
        .groupby("flow")["flight_id"]
        .nunique()
        .to_dict()
    )
    usable_flights_by_flow = (
        flight_summary.loc[
            flight_summary["arc_profile_available"] & flight_summary["speed_profile_available"],
            ["flow", "flight_id"],
        ]
        .drop_duplicates()
        .groupby("flow")["flight_id"]
        .nunique()
        .to_dict()
    )

    rows: list[dict[str, float | int | str]] = []
    for flow in FLOW_ORDER:
        grp = pair_df[pair_df["flow"] == flow]
        if grp.empty:
            continue
        row: dict[str, float | int | str] = {
            "flow": flow,
            "n_flights_sampled": int(total_flights_by_flow.get(flow, 0)),
            "n_flights_used": int(usable_flights_by_flow.get(flow, 0)),
            "n_pairs": int(len(grp)),
            "mean_D_euc_index": float(grp["D_euc_index"].mean()),
            "median_D_euc_index": float(grp["D_euc_index"].median()),
            "mean_D_euc_arclength": float(grp["D_euc_arclength"].mean()),
            "median_D_euc_arclength": float(grp["D_euc_arclength"].median()),
            "mean_D_dtw": float(grp["D_dtw"].mean()),
            "median_D_dtw": float(grp["D_dtw"].median()),
            "mean_D_speed": float(grp["D_speed"].mean()),
            "median_D_speed": float(grp["D_speed"].median()),
            "mean_delta_shape": float(grp["delta_shape"].mean()),
            "median_delta_shape": float(grp["delta_shape"].median()),
            "mean_abs_delta_shape": float(grp["abs_delta_shape"].mean()),
            "median_abs_delta_shape": float(grp["abs_delta_shape"].median()),
            "pearson_delta_vs_speed": _corr(grp["delta_shape"], grp["D_speed"], method="pearson"),
            "spearman_delta_vs_speed": _corr(grp["delta_shape"], grp["D_speed"], method="spearman"),
            "pearson_index_vs_dtw": _corr(grp["D_euc_index"], grp["D_dtw"], method="pearson"),
            "spearman_index_vs_dtw": _corr(grp["D_euc_index"], grp["D_dtw"], method="spearman"),
            "pearson_arclength_vs_dtw": _corr(grp["D_euc_arclength"], grp["D_dtw"], method="pearson"),
            "spearman_arclength_vs_dtw": _corr(grp["D_euc_arclength"], grp["D_dtw"], method="spearman"),
        }
        row.update(_percentile_stats(grp["delta_shape"].to_numpy(dtype=float), "delta_shape"))
        row.update(_percentile_stats(grp["abs_delta_shape"].to_numpy(dtype=float), "abs_delta_shape"))
        rows.append(row)

    return pd.DataFrame(rows)


def _build_overall_summary(pair_df: pd.DataFrame, flow_summary: pd.DataFrame) -> dict[str, object]:
    if pair_df.empty:
        return {"status": "no_pairs"}
    weighted_order = (
        flow_summary.sort_values("median_abs_delta_shape", ascending=False)[["flow", "median_abs_delta_shape"]]
        .to_dict(orient="records")
    )
    return {
        "status": "ok",
        "n_pairs_total": int(len(pair_df)),
        "n_flows_with_pairs": int(flow_summary["flow"].nunique()),
        "mean_delta_shape": float(pair_df["delta_shape"].mean()),
        "median_delta_shape": float(pair_df["delta_shape"].median()),
        "mean_abs_delta_shape": float(pair_df["abs_delta_shape"].mean()),
        "median_abs_delta_shape": float(pair_df["abs_delta_shape"].median()),
        "pearson_delta_vs_speed": _corr(pair_df["delta_shape"], pair_df["D_speed"], method="pearson"),
        "spearman_delta_vs_speed": _corr(pair_df["delta_shape"], pair_df["D_speed"], method="spearman"),
        "pearson_index_vs_dtw": _corr(pair_df["D_euc_index"], pair_df["D_dtw"], method="pearson"),
        "spearman_index_vs_dtw": _corr(pair_df["D_euc_index"], pair_df["D_dtw"], method="spearman"),
        "pearson_arclength_vs_dtw": _corr(pair_df["D_euc_arclength"], pair_df["D_dtw"], method="pearson"),
        "spearman_arclength_vs_dtw": _corr(pair_df["D_euc_arclength"], pair_df["D_dtw"], method="spearman"),
        "flows_ranked_by_median_abs_delta_shape": weighted_order,
    }


def _sample_for_plot(df: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df
    return df.sample(n=max_rows, random_state=seed).copy()


def _plot_speed_vs_delta(pair_df: pd.DataFrame, out_path: Path, seed: int) -> None:
    plot_df = _sample_for_plot(pair_df, max_rows=60_000, seed=seed)
    fig, ax = plt.subplots(figsize=(9.5, 7.0))
    for flow in FLOW_ORDER:
        grp = plot_df[plot_df["flow"] == flow]
        if grp.empty:
            continue
        ax.scatter(grp["D_speed"], grp["delta_shape"], s=10, alpha=0.35, label=flow)
    ax.set_xlabel("Speed-profile distance (m/s, L2)")
    ax.set_ylabel("Euclidean distortion: D_euc_index - D_euc_arclength")
    ax.set_title("Speed Difference vs Euclidean Distortion")
    ax.grid(True, color="#e8e8e8", linewidth=0.7)
    ax.legend(loc="best", fontsize=8, ncol=2, frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_euclidean_vs_dtw(pair_df: pd.DataFrame, out_path: Path, seed: int) -> None:
    plot_df = _sample_for_plot(pair_df, max_rows=60_000, seed=seed)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), sharey=True)
    panels = [
        ("D_euc_index", "Index-based Euclidean vs DTW"),
        ("D_euc_arclength", "Arc-length Euclidean vs DTW"),
    ]
    for ax, (xcol, title) in zip(axes, panels):
        for flow in FLOW_ORDER:
            grp = plot_df[plot_df["flow"] == flow]
            if grp.empty:
                continue
            ax.scatter(grp[xcol], grp["D_dtw"], s=10, alpha=0.35, label=flow)
        ax.set_xlabel(xcol)
        ax.set_title(title)
        ax.grid(True, color="#e8e8e8", linewidth=0.7)
    axes[0].set_ylabel("DTW distance")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[1].legend(handles, labels, loc="best", fontsize=8, ncol=2, frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_delta_by_flow(pair_df: pd.DataFrame, out_path: Path) -> None:
    flows = [flow for flow in FLOW_ORDER if flow in set(pair_df["flow"])]
    data = [pair_df.loc[pair_df["flow"] == flow, "delta_shape"].to_numpy(dtype=float) for flow in flows]
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.boxplot(data, labels=flows, showfliers=False)
    ax.set_ylabel("delta_shape = D_euc_index - D_euc_arclength")
    ax.set_title("Per-flow Distribution of Euclidean Distortion")
    ax.grid(True, axis="y", color="#e8e8e8", linewidth=0.7)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    input_path = _normalize_launch_path(args.preprocessed)
    if not input_path.exists():
        raise FileNotFoundError(f"Preprocessed CSV not found: {input_path}")
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    outdir = args.outdir
    if outdir is None:
        outdir = (
            Path("output")
            / "eda"
            / "euclidean_speed_impact"
            / f"{input_path.stem}_n{int(args.sample_per_flow)}_w{int(args.dtw_window_size)}"
        )
    outdir.mkdir(parents=True, exist_ok=True)

    actual = _load_actual_preprocessed(input_path)
    n_points = int(actual.groupby("flight_id")["step"].count().median())
    if n_points < 2:
        raise RuntimeError("Preprocessed file does not contain enough points per flight.")

    stem_tail = input_path.stem.split("_")[-1]
    preprocessed_id = int(stem_tail) if stem_tail.isdigit() else 0
    rebuilt_current, rebuilt_raw = _rebuild_pipeline(
        config_path=config_path,
        preprocessed_id=preprocessed_id,
        effective_n_points=n_points,
    )
    rebuild_check = _validate_rebuild(actual=actual, rebuilt=rebuilt_current)

    sampled_ids = _sample_flight_ids(actual, sample_per_flow=args.sample_per_flow, seed=args.seed)
    sampled_actual = actual[actual["flight_id"].isin(sampled_ids)].copy()
    current_arrays = _build_current_arrays(actual, sampled_ids)
    arclength_xy, speed_profiles, flight_summary = _build_raw_profiles(rebuilt_raw, sampled_ids, n_points=n_points)

    pair_frames: list[pd.DataFrame] = []
    for flow in FLOW_ORDER:
        flow_ids = sorted(
            sampled_actual.loc[sampled_actual["flow"] == flow, "flight_id"].drop_duplicates().astype(int).tolist()
        )
        if len(flow_ids) < 2:
            continue
        logging.info("Computing pairwise diagnostics for %s (%d sampled flights)", flow, len(flow_ids))
        flow_pairs = _pair_metrics_for_flow(
            flow=flow,
            flight_ids=flow_ids,
            current_arrays=current_arrays,
            arclength_xy=arclength_xy,
            speed_profiles=speed_profiles,
            dtw_window_size=args.dtw_window_size,
            max_pairs=args.max_pairs_per_flow,
            seed=args.seed + FLOW_ORDER.index(flow),
        )
        if not flow_pairs.empty:
            pair_frames.append(flow_pairs)

    if not pair_frames:
        raise RuntimeError("No pairwise diagnostics were produced. Check the sampled/raw flight linkage.")

    pair_df = pd.concat(pair_frames, ignore_index=True)
    flow_summary = _build_flow_summary(pair_df=pair_df, sampled_actual=sampled_actual, flight_summary=flight_summary)
    overall_summary = _build_overall_summary(pair_df=pair_df, flow_summary=flow_summary)

    pair_df.to_csv(outdir / "pair_metrics.csv", index=False)
    flight_summary.sort_values(["flow", "flight_id"]).to_csv(outdir / "flight_summary.csv", index=False)
    flow_summary.sort_values("flow").to_csv(outdir / "flow_summary.csv", index=False)
    (outdir / "overall_summary.json").write_text(json.dumps(overall_summary, indent=2), encoding="utf-8")

    metadata = {
        "input_preprocessed": str(input_path),
        "config_path": str(config_path),
        "sample_per_flow": int(args.sample_per_flow),
        "seed": int(args.seed),
        "dtw_window_size": int(args.dtw_window_size),
        "max_pairs_per_flow": None if args.max_pairs_per_flow is None else int(args.max_pairs_per_flow),
        "n_points_current_representation": int(n_points),
        "flows_present_in_preprocessed": sorted(actual["flow"].dropna().unique().tolist()),
        "n_flights_preprocessed_total": int(actual["flight_id"].nunique()),
        "n_flights_sampled_total": int(sampled_actual["flight_id"].nunique()),
        "pair_rows_written": int(len(pair_df)),
        "rebuild_validation": rebuild_check,
        "note": (
            "Current Euclidean distances are taken from the actual preprocessed CSV. "
            "Arc-length geometry and speed profiles are rebuilt from matched trajectories "
            "using the same config and validated against the stored preprocessed representation."
        ),
    }
    (outdir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    _plot_speed_vs_delta(pair_df, outdir / "speed_vs_delta_shape.png", seed=args.seed)
    _plot_euclidean_vs_dtw(pair_df, outdir / "euclidean_vs_dtw.png", seed=args.seed + 1)
    _plot_delta_by_flow(pair_df, outdir / "delta_shape_by_flow.png")

    print(f"Saved Euclidean speed-impact analysis to {outdir}", flush=True)
    print(f"Pair rows: {len(pair_df)}", flush=True)


if __name__ == "__main__":
    main()
