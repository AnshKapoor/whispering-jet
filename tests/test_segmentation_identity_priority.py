"""Tests for segmentation identity-priority boundary behavior."""

from __future__ import annotations

import pandas as pd

from backbone_tracks.segmentation import segment_flights


def _base_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "A/D": ["Start", "Start", "Start", "Start"],
            "Runway": ["09L", "09L", "09L", "09L"],
            "timestamp": pd.to_datetime(
                [
                    "2022-01-01T00:00:00Z",
                    "2022-01-01T00:00:01Z",
                    "2022-01-01T00:00:02Z",
                    "2022-01-01T00:00:03Z",
                ],
                utc=True,
            ),
            "dist_to_airport_m": [1000.0, 999.0, 998.0, 997.0],
            "icao24": ["AAA001", "AAA001", "BBB001", "BBB001"],
            "callsign": ["CALL1", "CALL1", "CALL2", "CALL2"],
        }
    )


def test_identity_change_starts_new_flight() -> None:
    """Changing identity should split flight even without time/distance breaks."""

    df = _base_df()
    out = segment_flights(
        df,
        time_gap_sec=120.0,
        distance_jump_m=600.0,
        min_points_per_flight=1,
        split_on_identity=True,
    )
    ids = out["flight_id"].tolist()
    assert ids[0] == ids[1]
    assert ids[2] == ids[3]
    assert ids[1] != ids[2]


def test_same_identity_uses_other_rules() -> None:
    """When identity stays the same, time-gap rule must still split flights."""

    df = pd.DataFrame(
        {
            "A/D": ["Start", "Start", "Start"],
            "Runway": ["09L", "09L", "09L"],
            "timestamp": pd.to_datetime(
                [
                    "2022-01-01T00:00:00Z",
                    "2022-01-01T00:00:01Z",
                    "2022-01-01T00:10:30Z",
                ],
                utc=True,
            ),
            "dist_to_airport_m": [1000.0, 999.0, 998.0],
            "icao24": ["AAA001", "AAA001", "AAA001"],
            "callsign": ["CALL1", "CALL1", "CALL1"],
        }
    )
    out = segment_flights(
        df,
        time_gap_sec=60.0,
        distance_jump_m=600.0,
        min_points_per_flight=1,
        split_on_identity=True,
    )
    ids = out["flight_id"].tolist()
    assert ids[0] == ids[1]
    assert ids[1] != ids[2]
