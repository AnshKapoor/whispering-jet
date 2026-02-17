"""Flight segmentation based on time/distance rules.

Implements multi-rule flight boundary detection and optional caps on flights per
flow for test-mode runs.
"""

from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import pandas as pd


def segment_flights(
    df: pd.DataFrame,
    time_gap_sec: float,
    distance_jump_m: float,
    min_points_per_flight: int,
    split_on_identity: bool = True,
) -> pd.DataFrame:
    """
    Add a ``flight_id`` column using rule-based boundaries per (A/D, Runway).

    Rule order:
    1. Identity split: if ``icao24`` or ``callsign`` changes, start a new flight.
    2. Otherwise apply:
       - Rule A: time gap
       - Rule B: direction reset (sign flip in airport-distance deltas)
       - Rule C: distance jump

    Flights shorter than ``min_points_per_flight`` are dropped.
    """

    df_sorted = df.sort_values(["A/D", "Runway", "timestamp"]).reset_index(drop=True)
    flight_ids = np.full(len(df_sorted), -1, dtype=int)

    next_flight_id = 0
    for _, group_idx in df_sorted.groupby(["A/D", "Runway"]).groups.items():
        indices = list(group_idx)
        prev_ts: pd.Timestamp | None = None
        prev_dist: float | None = None
        prev_prev_dist: float | None = None
        prev_icao: str | None = None
        prev_callsign: str | None = None
        current_id = next_flight_id

        for pos, row_idx in enumerate(indices):
            row = df_sorted.loc[row_idx]
            ts_curr = pd.to_datetime(row["timestamp"])
            dist_curr = float(row["dist_to_airport_m"])
            icao_curr = str(row["icao24"]).strip().upper() if "icao24" in row and pd.notna(row["icao24"]) else None
            callsign_curr = (
                str(row["callsign"]).strip().upper() if "callsign" in row and pd.notna(row["callsign"]) else None
            )

            new_flight = False
            identity_changed = False
            if split_on_identity and pos > 0:
                if icao_curr and prev_icao and icao_curr != prev_icao:
                    identity_changed = True
                if callsign_curr and prev_callsign and callsign_curr != prev_callsign:
                    identity_changed = True

            # Prioritize identity split. For unchanged identity, use geometric/time rules.
            if identity_changed:
                new_flight = True
            else:
                # Rule A: time gap
                if prev_ts is not None and (ts_curr - prev_ts).total_seconds() > time_gap_sec:
                    new_flight = True

                # Rule B: direction change (sign flip in distance deltas)
                if prev_dist is not None and prev_prev_dist is not None:
                    prev_diff = prev_dist - prev_prev_dist
                    curr_diff = dist_curr - prev_dist
                    if curr_diff != 0 and prev_diff != 0 and np.sign(curr_diff) != np.sign(prev_diff):
                        new_flight = True

                # Rule C: distance jump
                if prev_dist is not None and abs(dist_curr - prev_dist) > distance_jump_m:
                    new_flight = True

            if new_flight and pos > 0:
                next_flight_id += 1
                current_id = next_flight_id
                # Reset local context so Rule B does not use deltas across segment boundaries.
                prev_ts = None
                prev_dist = None
                prev_prev_dist = None
                prev_icao = None
                prev_callsign = None

            flight_ids[row_idx] = current_id

            prev_prev_dist = prev_dist
            prev_dist = dist_curr
            prev_ts = ts_curr
            prev_icao = icao_curr
            prev_callsign = callsign_curr

        next_flight_id += 1

    df_sorted["flight_id"] = flight_ids

    counts = df_sorted.groupby("flight_id").size()
    valid_ids = counts[counts >= min_points_per_flight].index
    filtered = df_sorted[df_sorted["flight_id"].isin(valid_ids)].reset_index(drop=True)
    dropped = len(counts) - len(valid_ids)
    if dropped:
        logging.info("Dropped %d flights shorter than %d points", dropped, min_points_per_flight)
    return filtered


def limit_flights_per_flow(
    df: pd.DataFrame,
    max_flights_per_flow: int,
    flow_keys: Iterable[str],
) -> pd.DataFrame:
    """Restrict to the first N flights per flow (deterministic order)."""

    if max_flights_per_flow is None or max_flights_per_flow <= 0:
        return df

    keep_ids: set[int] = set()
    for _, group in df.groupby(list(flow_keys)):
        ids = group["flight_id"].drop_duplicates().sort_values().head(max_flights_per_flow)
        keep_ids.update(ids.tolist())

    limited = df[df["flight_id"].isin(keep_ids)].reset_index(drop=True)
    logging.info(
        "Restricted flights per flow: kept %d flights (<= %d per flow)",
        limited["flight_id"].nunique(),
        max_flights_per_flow,
    )
    return limited
