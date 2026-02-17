"""Trajectory smoothing and resampling for backbone clustering.

Handles per-flight smoothing (Savitzky–Golay or moving average fallback) and
interpolation to fixed-length trajectories in either lat/lon or UTM space.
"""

from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

try:  # Savitzky-Golay is preferred; fall back to moving average if unavailable.
    from scipy.signal import savgol_filter
except ImportError:  # pragma: no cover - defensive
    savgol_filter = None


SMOOTH_DEFAULT_COLUMNS = ["latitude", "longitude", "altitude"]


def smooth_series(
    values: pd.Series,
    window_length: int,
    polyorder: int,
    method: str = "auto",
    audit: Dict[str, int] | None = None,
) -> pd.Series:
    """Smooth a numeric series using the configured method.

    Supported methods:
      - ``auto`` / ``savgol``: Savitzky-Golay if SciPy is available, otherwise moving average
      - ``moving_average``: centred rolling mean
      - ``median``: centred rolling median
      - ``ewm``: exponential weighted moving average
      - ``none``: no smoothing
    """

    method_norm = str(method or "auto").strip().lower().replace("-", "_")
    if method_norm in {"none", "off", "disabled"}:
        if audit is not None:
            audit["none"] = audit.get("none", 0) + 1
        return values

    if method_norm in {"ewm", "ema", "exponential"}:
        if audit is not None:
            audit["ewm"] = audit.get("ewm", 0) + 1
        span = max(int(window_length), 2)
        return values.ewm(span=span, adjust=False).mean()

    if method_norm in {"moving_average", "rolling_mean", "mean"}:
        if window_length < 2:
            if audit is not None:
                audit["none"] = audit.get("none", 0) + 1
            return values
        if audit is not None:
            audit["moving_average"] = audit.get("moving_average", 0) + 1
        return values.rolling(window=int(window_length), center=True, min_periods=1).mean()

    if method_norm in {"median", "rolling_median"}:
        if window_length < 2:
            if audit is not None:
                audit["none"] = audit.get("none", 0) + 1
            return values
        if audit is not None:
            audit["median"] = audit.get("median", 0) + 1
        return values.rolling(window=int(window_length), center=True, min_periods=1).median()

    if method_norm not in {"auto", "savgol", "savitzky_golay", "savitzkygolay"}:
        raise ValueError(f"Unsupported smoothing method: {method}")

    if window_length < 3 or len(values) < window_length:
        if audit is not None:
            audit["none"] = audit.get("none", 0) + 1
        return values
    if window_length % 2 == 0:
        window_length -= 1
    if window_length < 3:
        if audit is not None:
            audit["none"] = audit.get("none", 0) + 1
        return values

    if savgol_filter:
        if audit is not None:
            audit["savgol"] = audit.get("savgol", 0) + 1
        poly = min(polyorder, window_length - 1)
        return pd.Series(savgol_filter(values, window_length=window_length, polyorder=poly), index=values.index)

    if audit is not None:
        audit["moving_average_fallback"] = audit.get("moving_average_fallback", 0) + 1
    return values.rolling(window=window_length, center=True, min_periods=1).mean()


def resample_trajectory(
    df: pd.DataFrame,
    n_points: int,
    method: str,
    columns: Sequence[str],
) -> pd.DataFrame:
    """Resample a trajectory to exactly n_points using time or index as the domain."""

    if method not in {"time", "index"}:
        raise ValueError(f"Unsupported resampling method: {method}")

    traj = df.copy()
    if method == "time":
        t = (traj["timestamp"] - traj["timestamp"].min()).dt.total_seconds()
    else:
        t = np.linspace(0.0, 1.0, len(traj))

    target = np.linspace(t.min(), t.max(), n_points)
    resampled: Dict[str, Iterable[float]] = {"step": np.arange(n_points, dtype=int)}

    for col in columns:
        if col in traj.columns:
            resampled[col] = np.interp(target, t, traj[col].astype(float))

    meta_cols = ["A/D", "Runway", "flight_id", "icao24", "callsign", "aircraft_type_match"]
    for col in meta_cols:
        if col in traj.columns:
            resampled[col] = traj[col].iloc[0]

    return pd.DataFrame(resampled)


def preprocess_flights(
    df: pd.DataFrame,
    smoothing_cfg: Dict[str, object],
    resampling_cfg: Dict[str, object],
    filter_cfg: Dict[str, object] | None = None,
    use_utm: bool = False,
    flow_keys: Sequence[str] = ("A/D", "Runway"),
) -> pd.DataFrame:
    """
    For each (flight_id, flow), sort by timestamp, optionally smooth, and resample to fixed points.
    Returns concatenated resampled trajectories.
    """

    filter_cfg = filter_cfg or {}
    max_airport_distance_m = filter_cfg.get("max_airport_distance_m")
    if max_airport_distance_m is None and filter_cfg.get("max_airport_distance_km") is not None:
        max_airport_distance_m = float(filter_cfg["max_airport_distance_km"]) * 1000.0
    if max_airport_distance_m is not None:
        max_airport_distance_m = float(max_airport_distance_m)
    min_length_m = None
    if filter_cfg.get("min_length_km") is not None:
        min_length_m = float(filter_cfg["min_length_km"]) * 1000.0
    allowed_runways = filter_cfg.get("allowed_runways")
    if allowed_runways:
        allowed_runways = {str(r).strip() for r in allowed_runways}

    logger = logging.getLogger(__name__)

    smooth_enabled = smoothing_cfg.get("enabled", True)
    smooth_method = str(smoothing_cfg.get("method", "auto"))
    window_length = int(smoothing_cfg.get("window_length", 7))
    polyorder = int(smoothing_cfg.get("polyorder", 2))
    smooth_cols = list(smoothing_cfg.get("columns", SMOOTH_DEFAULT_COLUMNS))
    if use_utm:
        smooth_cols = list({*smooth_cols, "x_utm", "y_utm"})

    simplify_cfg = filter_cfg.get("simplify", {}) or {}
    rdp_enabled = bool(simplify_cfg.get("rdp_enabled", False))
    rdp_epsilon_m = float(simplify_cfg.get("rdp_epsilon_m", 0.0))
    rdp_min_points = int(simplify_cfg.get("rdp_min_points", 10))

    n_points = int(resampling_cfg.get("n_points", 40))
    method = str(resampling_cfg.get("method", "time")).lower()
    resampling_enabled = bool(resampling_cfg.get("enabled", True))
    raw_mode = (not resampling_enabled) or method in {"none", "raw", "off", "disabled"}
    resample_cols = ["altitude", "dist_to_airport_m"]
    if use_utm:
        resample_cols = ["x_utm", "y_utm", *resample_cols, "latitude", "longitude"]
    else:
        resample_cols = ["latitude", "longitude", *resample_cols]

    outputs: List[pd.DataFrame] = []
    smooth_audit: Dict[str, int] = {}
    rdp_applied = 0
    group_cols = [*flow_keys, "flight_id"]
    for group_values, flight in df.groupby(group_cols):
        flight_id = group_values[-1] if isinstance(group_values, tuple) else group_values
        flight_sorted = flight.sort_values("timestamp")
        if len(flight_sorted) < 2:
            continue
        if allowed_runways and "Runway" in flight_sorted.columns:
            runway = str(flight_sorted["Runway"].iloc[0]).strip()
            if runway not in allowed_runways:
                continue

        if max_airport_distance_m is not None and "dist_to_airport_m" in flight_sorted.columns:
            flight_sorted = flight_sorted[
                pd.to_numeric(flight_sorted["dist_to_airport_m"], errors="coerce")
                <= max_airport_distance_m
            ].copy()
            if len(flight_sorted) < 2:
                continue
        if min_length_m is not None and use_utm and {"x_utm", "y_utm"}.issubset(flight_sorted.columns):
            dx = flight_sorted["x_utm"].diff()
            dy = flight_sorted["y_utm"].diff()
            length = float(np.sqrt(dx * dx + dy * dy).sum(skipna=True))
            if length < min_length_m:
                continue

        if rdp_enabled and use_utm and {"x_utm", "y_utm"}.issubset(flight_sorted.columns):
            if len(flight_sorted) >= rdp_min_points and rdp_epsilon_m > 0:
                try:
                    from distance_metrics import rdp  # local helper

                    coords = flight_sorted[["x_utm", "y_utm"]].to_numpy(dtype=float)
                    simplified = rdp(coords, rdp_epsilon_m)
                    # Keep only rows that match simplified coordinates.
                    key = {(float(x), float(y)) for x, y in simplified}
                    mask = flight_sorted[["x_utm", "y_utm"]].apply(
                        lambda row: (float(row["x_utm"]), float(row["y_utm"])) in key, axis=1
                    )
                    if mask.any():
                        flight_sorted = flight_sorted[mask].copy()
                        rdp_applied += 1
                except Exception as exc:  # pragma: no cover - defensive
                    logger.warning("RDP simplification failed for flight %s: %s", flight_id, exc)

        if smooth_enabled:
            for col in smooth_cols:
                if col in flight_sorted.columns:
                    flight_sorted[col] = smooth_series(
                        flight_sorted[col].astype(float),
                        window_length,
                        polyorder,
                        method=smooth_method,
                        audit=smooth_audit,
                    )

        if raw_mode:
            # Keep original points in timestamp order and assign sequential step index.
            raw = flight_sorted.reset_index(drop=True).copy()
            raw["step"] = np.arange(len(raw), dtype=int)
            outputs.append(raw)
        else:
            try:
                resampled = resample_trajectory(
                    flight_sorted,
                    n_points=n_points,
                    method=method,
                    columns=resample_cols,
                )
                outputs.append(resampled)
            except Exception as exc:  # pragma: no cover - defensive
                logging.warning("Skipping flight %s due to resampling error: %s", flight_id, exc)
                continue

    if not outputs:
        return pd.DataFrame()

    result = pd.concat(outputs, ignore_index=True)
    if smooth_enabled:
        logger.info("Smoothing usage counts: %s", smooth_audit)
    if raw_mode:
        logger.info("Resampling disabled: keeping raw per-flight points with sequential step.")
    if rdp_enabled:
        logger.info("RDP simplification applied to %d flights (epsilon=%.2f m)", rdp_applied, rdp_epsilon_m)
    return result
