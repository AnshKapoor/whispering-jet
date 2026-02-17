from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.save_preprocessed import apply_repetition_dedup


def _cfg(tmp_path) -> dict:
    return {
        "enabled": True,
        "window_minutes": 10,
        "timezone": "UTC",
        "require_same_date": True,
        "identity": "icao24_callsign",
        "action": "drop",
        "keep_policy": "earliest_t_ref",
        "output_dir": str(tmp_path),
    }


def _row(
    timestamp: str,
    t_ref: str,
    mp: str,
    icao24: str,
    callsign: str,
) -> dict:
    return {
        "timestamp": timestamp,
        "t_ref": t_ref,
        "MP": mp,
        "icao24": icao24,
        "callsign": callsign,
        "latitude": 52.0,
        "longitude": 9.0,
        "altitude": 1000.0,
        "dist_to_airport_m": 5000.0,
        "A/D": "Start",
        "Runway": "09L",
    }


def test_no_repeat_keeps_rows(tmp_path):
    df = pd.DataFrame(
        [
            _row("2022-01-01T10:00:01Z", "2022-01-01T10:00:00Z", "M01", "ABC123", "DLH001"),
            _row("2022-01-01T10:00:03Z", "2022-01-01T10:00:00Z", "M01", "ABC123", "DLH001"),
            _row("2022-01-01T10:30:01Z", "2022-01-01T10:30:00Z", "M02", "ABC123", "DLH001"),
            _row("2022-01-01T10:30:03Z", "2022-01-01T10:30:00Z", "M02", "ABC123", "DLH001"),
        ]
    )
    out = apply_repetition_dedup(df, _cfg(tmp_path), preprocessed_id=999)
    assert len(out) == len(df)


def test_repeat_within_10_minutes_drops_later_event(tmp_path):
    df = pd.DataFrame(
        [
            _row("2022-01-01T10:00:01Z", "2022-01-01T10:00:00Z", "M01", "ABC123", "DLH001"),
            _row("2022-01-01T10:00:03Z", "2022-01-01T10:00:00Z", "M01", "ABC123", "DLH001"),
            _row("2022-01-01T10:07:01Z", "2022-01-01T10:07:00Z", "M02", "ABC123", "DLH001"),
            _row("2022-01-01T10:07:03Z", "2022-01-01T10:07:00Z", "M02", "ABC123", "DLH001"),
        ]
    )
    out = apply_repetition_dedup(df, _cfg(tmp_path), preprocessed_id=999)
    assert len(out) == 2
    assert set(out["MP"]) == {"M01"}


def test_gap_greater_than_10_minutes_keeps_both_events(tmp_path):
    df = pd.DataFrame(
        [
            _row("2022-01-01T10:00:01Z", "2022-01-01T10:00:00Z", "M01", "ABC123", "DLH001"),
            _row("2022-01-01T10:00:03Z", "2022-01-01T10:00:00Z", "M01", "ABC123", "DLH001"),
            _row("2022-01-01T10:20:01Z", "2022-01-01T10:20:00Z", "M02", "ABC123", "DLH001"),
            _row("2022-01-01T10:20:03Z", "2022-01-01T10:20:00Z", "M02", "ABC123", "DLH001"),
        ]
    )
    out = apply_repetition_dedup(df, _cfg(tmp_path), preprocessed_id=999)
    assert len(out) == len(df)


def test_same_identity_across_date_boundary_not_merged(tmp_path):
    df = pd.DataFrame(
        [
            _row("2022-01-01T23:58:01Z", "2022-01-01T23:58:00Z", "M01", "ABC123", "DLH001"),
            _row("2022-01-01T23:58:03Z", "2022-01-01T23:58:00Z", "M01", "ABC123", "DLH001"),
            _row("2022-01-02T00:03:01Z", "2022-01-02T00:03:00Z", "M02", "ABC123", "DLH001"),
            _row("2022-01-02T00:03:03Z", "2022-01-02T00:03:00Z", "M02", "ABC123", "DLH001"),
        ]
    )
    out = apply_repetition_dedup(df, _cfg(tmp_path), preprocessed_id=999)
    assert len(out) == len(df)


def test_missing_keys_are_retained_and_counted(tmp_path):
    df = pd.DataFrame(
        [
            _row("2022-01-01T10:00:01Z", "2022-01-01T10:00:00Z", "M01", "ABC123", "DLH001"),
            _row("2022-01-01T10:07:01Z", "2022-01-01T10:07:00Z", "M02", "ABC123", "DLH001"),
            _row("2022-01-01T10:07:03Z", "2022-01-01T10:07:00Z", "M02", "ABC123", ""),
        ]
    )
    out = apply_repetition_dedup(df, _cfg(tmp_path), preprocessed_id=999)
    # One of the valid repeated events should be dropped; missing-key row should remain.
    assert len(out) == 2
    assert (out["callsign"] == "").any()


def test_repetition_dedup_is_deterministic(tmp_path):
    df = pd.DataFrame(
        [
            _row("2022-01-01T10:00:01Z", "2022-01-01T10:00:00Z", "M01", "ABC123", "DLH001"),
            _row("2022-01-01T10:07:01Z", "2022-01-01T10:07:00Z", "M02", "ABC123", "DLH001"),
            _row("2022-01-01T10:08:01Z", "2022-01-01T10:08:00Z", "M03", "ABC123", "DLH001"),
        ]
    )
    out1 = apply_repetition_dedup(df, _cfg(tmp_path), preprocessed_id=999)
    out2 = apply_repetition_dedup(df, _cfg(tmp_path), preprocessed_id=999)
    pd.testing.assert_frame_equal(
        out1.sort_values(["timestamp", "MP"]).reset_index(drop=True),
        out2.sort_values(["timestamp", "MP"]).reset_index(drop=True),
    )


def test_missing_required_columns_skips_without_changes(tmp_path):
    df = pd.DataFrame(
        [
            {
                "timestamp": "2022-01-01T10:00:01Z",
                "icao24": "ABC123",
                "callsign": "DLH001",
                "t_ref": "2022-01-01T10:00:00Z",
            }
        ]
    )
    out = apply_repetition_dedup(df, _cfg(tmp_path), preprocessed_id=999)
    pd.testing.assert_frame_equal(df, out)
