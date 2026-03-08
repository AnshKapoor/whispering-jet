"""Compute repetition-aware unique flight-event counts from matched trajectories.

This script applies the same MP repetition rule used in preprocessing:
- identity: (icao24, callsign)
- optional same UTC date requirement
- events within window_minutes are grouped as one physical flight pass
- earliest event in each repetition cluster is kept

Outputs (under --outdir):
- unique_flights_summary.json
- unique_flights_overview.csv
- unique_flights_by_aircraft_type.csv
- unique_flights_by_runway.csv
- unique_flights_by_flow.csv
- unique_flights_by_operation.csv
- unique_flights_by_operation_runway.csv
- unique_flights_by_operation_aircraft_type.csv
- unique_flights_by_runway_aircraft_type.csv
- unique_flights_output_index.csv
- dropped_repetition_events.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import pandas as pd

REQUIRED_COLS = ["icao24", "callsign", "MP", "t_ref"]
OPTIONAL_COLS = [
    "A/D",
    "Runway",
    "aircraft_type_noise",
    "aircraft_type_adsb",
    "aircraft_type",
    "typecode",
    "icao_type",
    "aircraft",
]


def _norm_identifier(s: pd.Series) -> pd.Series:
    out = s.astype("string").str.strip().str.upper()
    out = out.replace({"": pd.NA, "NAN": pd.NA, "NONE": pd.NA, "<NA>": pd.NA})
    return out


def _coalesce_type(df: pd.DataFrame) -> pd.Series:
    # Priority: noise label -> ADS-B typecode -> generic type columns.
    candidates = [
        "aircraft_type_noise",
        "aircraft_type_adsb",
        "aircraft_type",
        "typecode",
        "icao_type",
        "aircraft",
    ]
    result = pd.Series(pd.NA, index=df.index, dtype="string")
    for col in candidates:
        if col in df.columns:
            cur = _norm_identifier(df[col])
            result = result.fillna(cur)
    return result.fillna("UNKNOWN")


def _available_columns(csv_path: Path, cols: Iterable[str]) -> list[str]:
    header = pd.read_csv(csv_path, nrows=0)
    return [c for c in cols if c in header.columns]


def _build_event_table(csv_glob: str, chunk_size: int) -> tuple[pd.DataFrame, int, int]:
    files = sorted(Path(".").glob(csv_glob))
    if not files:
        raise FileNotFoundError(f"No files matched: {csv_glob}")

    seen_event_keys: set[tuple[object, object, object, object]] = set()
    records: list[dict] = []
    rows_scanned = 0
    files_scanned = 0

    for csv_path in files:
        files_scanned += 1
        cols = _available_columns(csv_path, REQUIRED_COLS + OPTIONAL_COLS)
        missing_required = [c for c in REQUIRED_COLS if c not in cols]
        if missing_required:
            raise ValueError(f"{csv_path} missing required columns: {missing_required}")

        dtype_map = {
            c: "string"
            for c in [
                "icao24",
                "callsign",
                "MP",
                "t_ref",
                "A/D",
                "Runway",
                "aircraft_type_noise",
                "aircraft_type_adsb",
                "aircraft_type",
                "typecode",
                "icao_type",
                "aircraft",
            ]
            if c in cols
        }

        for chunk in pd.read_csv(
            csv_path,
            usecols=cols,
            dtype=dtype_map,
            chunksize=chunk_size,
            low_memory=False,
        ):
            rows_scanned += len(chunk)
            chunk["_icao24"] = _norm_identifier(chunk["icao24"])
            chunk["_callsign"] = _norm_identifier(chunk["callsign"])
            chunk["_mp"] = _norm_identifier(chunk["MP"])
            chunk["_t_ref_utc"] = pd.to_datetime(chunk["t_ref"], errors="coerce", utc=True)
            chunk["_date_utc"] = chunk["_t_ref_utc"].dt.normalize()
            chunk["_ad"] = (
                _norm_identifier(chunk["A/D"])
                if "A/D" in chunk.columns
                else pd.Series(pd.NA, index=chunk.index, dtype="string")
            )
            chunk["_runway"] = (
                _norm_identifier(chunk["Runway"])
                if "Runway" in chunk.columns
                else pd.Series(pd.NA, index=chunk.index, dtype="string")
            )
            chunk["_aircraft_type"] = _coalesce_type(chunk)

            event_view = chunk[
                [
                    "_icao24",
                    "_callsign",
                    "_mp",
                    "_t_ref_utc",
                    "_date_utc",
                    "_ad",
                    "_runway",
                    "_aircraft_type",
                ]
            ].drop_duplicates(subset=["_icao24", "_callsign", "_mp", "_t_ref_utc"], keep="first")

            for row in event_view.itertuples(index=False):
                t_ref_val = row._3  # _t_ref_utc
                t_ref_key = int(t_ref_val.value) if pd.notna(t_ref_val) else None
                key = (row._0, row._1, row._2, t_ref_key)
                if key in seen_event_keys:
                    continue
                seen_event_keys.add(key)
                records.append(
                    {
                        "icao24": row._0,
                        "callsign": row._1,
                        "MP": row._2,
                        "t_ref_utc": t_ref_val,
                        "date_utc": row._4,
                        "A/D": row._5,
                        "Runway": row._6,
                        "aircraft_type": row._7,
                    }
                )

    events = pd.DataFrame(records)
    return events, rows_scanned, files_scanned


def _counts(df: pd.DataFrame, by: list[str], count_col: str = "n_unique_flights") -> pd.DataFrame:
    out = (
        df.groupby(by, dropna=False)
        .size()
        .reset_index(name=count_col)
        .sort_values(count_col, ascending=False)
        .reset_index(drop=True)
    )
    out["percentage"] = (out[count_col] / out[count_col].sum() * 100.0).round(3)
    return out


def _write_output_index(outdir: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(outdir / "unique_flights_output_index.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Repetition-aware unique trajectory stats from matched trajectories.")
    parser.add_argument("--csv-glob", type=str, default="matched_trajectories/matched_trajs_*.csv")
    parser.add_argument("--window-minutes", type=int, default=10)
    parser.add_argument("--chunk-size", type=int, default=300_000)
    parser.add_argument("--require-same-date", action="store_true", default=True)
    parser.add_argument("--outdir", type=Path, default=Path("output/eda/matched_unique_flights"))
    args = parser.parse_args()

    if args.window_minutes <= 0:
        raise ValueError("--window-minutes must be > 0.")

    args.outdir.mkdir(parents=True, exist_ok=True)
    events, rows_scanned, files_scanned = _build_event_table(args.csv_glob, args.chunk_size)

    if events.empty:
        raise RuntimeError("No events found after reading matched trajectory CSVs.")

    valid_mask = events["icao24"].notna() & events["callsign"].notna() & events["t_ref_utc"].notna()
    valid = events.loc[valid_mask].copy()
    missing = events.loc[~valid_mask].copy()

    group_keys = ["icao24", "callsign", "date_utc"] if args.require_same_date else ["icao24", "callsign"]
    valid = valid.sort_values(group_keys + ["t_ref_utc", "MP"], kind="mergesort").reset_index(drop=True)
    gap = valid.groupby(group_keys)["t_ref_utc"].diff()
    boundary = gap.isna() | (gap > pd.Timedelta(minutes=args.window_minutes))
    valid["_boundary"] = boundary
    valid["_rep_cluster"] = valid.groupby(group_keys, sort=False)["_boundary"].cumsum().astype("int64")

    cluster_cols = group_keys + ["_rep_cluster"]
    cluster_sizes = valid.groupby(cluster_cols, dropna=False).size().rename("cluster_size").reset_index()
    repeat_clusters = int((cluster_sizes["cluster_size"] > 1).sum())

    kept = valid.drop_duplicates(subset=cluster_cols, keep="first").copy()
    dropped = valid.loc[~valid.index.isin(kept.index)].copy()

    dedup = pd.concat([kept, missing], ignore_index=True, sort=False)
    dedup["A/D"] = dedup["A/D"].fillna("UNKNOWN")
    dedup["Runway"] = dedup["Runway"].fillna("UNKNOWN")
    dedup["aircraft_type"] = dedup["aircraft_type"].fillna("UNKNOWN")
    dedup["flow_label"] = dedup["A/D"].astype(str) + "_" + dedup["Runway"].astype(str)

    by_aircraft = _counts(dedup, ["aircraft_type"])
    by_runway = _counts(dedup, ["Runway"])
    by_flow = _counts(dedup, ["A/D", "Runway", "flow_label"])
    by_operation = _counts(dedup, ["A/D"])
    by_operation_runway = _counts(dedup, ["A/D", "Runway"])
    by_operation_aircraft = _counts(dedup, ["A/D", "aircraft_type"])
    by_runway_aircraft = _counts(dedup, ["Runway", "aircraft_type"])

    by_aircraft.to_csv(args.outdir / "unique_flights_by_aircraft_type.csv", index=False)
    by_runway.to_csv(args.outdir / "unique_flights_by_runway.csv", index=False)
    by_flow.to_csv(args.outdir / "unique_flights_by_flow.csv", index=False)
    by_operation.to_csv(args.outdir / "unique_flights_by_operation.csv", index=False)
    by_operation_runway.to_csv(args.outdir / "unique_flights_by_operation_runway.csv", index=False)
    by_operation_aircraft.to_csv(args.outdir / "unique_flights_by_operation_aircraft_type.csv", index=False)
    by_runway_aircraft.to_csv(args.outdir / "unique_flights_by_runway_aircraft_type.csv", index=False)
    dropped.to_csv(args.outdir / "dropped_repetition_events.csv", index=False)

    summary = {
        "csv_glob": args.csv_glob,
        "files_scanned": int(files_scanned),
        "rows_scanned": int(rows_scanned),
        "event_rows_unique_raw": int(len(events)),
        "events_valid_for_dedup": int(len(valid)),
        "events_missing_identity_or_time": int(len(missing)),
        "repeat_clusters": int(repeat_clusters),
        "events_dropped_as_repetition": int(len(dropped)),
        "unique_trajectories_after_dedup": int(len(dedup)),
        "window_minutes": int(args.window_minutes),
        "require_same_date": bool(args.require_same_date),
    }
    (args.outdir / "unique_flights_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    overview_rows = [
        {"metric": "files_scanned", "value": summary["files_scanned"]},
        {"metric": "rows_scanned", "value": summary["rows_scanned"]},
        {"metric": "event_rows_unique_raw", "value": summary["event_rows_unique_raw"]},
        {"metric": "events_valid_for_dedup", "value": summary["events_valid_for_dedup"]},
        {"metric": "events_missing_identity_or_time", "value": summary["events_missing_identity_or_time"]},
        {"metric": "repeat_clusters", "value": summary["repeat_clusters"]},
        {"metric": "events_dropped_as_repetition", "value": summary["events_dropped_as_repetition"]},
        {"metric": "unique_trajectories_after_dedup", "value": summary["unique_trajectories_after_dedup"]},
        {"metric": "window_minutes", "value": summary["window_minutes"]},
        {"metric": "require_same_date", "value": summary["require_same_date"]},
    ]
    pd.DataFrame(overview_rows).to_csv(args.outdir / "unique_flights_overview.csv", index=False)

    _write_output_index(
        args.outdir,
        [
            {"file": "unique_flights_overview.csv", "description": "Top-level repetition-aware count summary."},
            {"file": "unique_flights_by_aircraft_type.csv", "description": "Unique trajectory counts and percentages by aircraft type."},
            {"file": "unique_flights_by_runway.csv", "description": "Unique trajectory counts and percentages by runway."},
            {"file": "unique_flights_by_operation.csv", "description": "Unique trajectory counts and percentages by A/D (Start/Landung)."},
            {"file": "unique_flights_by_flow.csv", "description": "Unique trajectory counts and percentages by A/D + Runway."},
            {"file": "unique_flights_by_operation_runway.csv", "description": "Unique trajectory counts and percentages by operation-runway pairs."},
            {"file": "unique_flights_by_operation_aircraft_type.csv", "description": "Unique trajectory counts and percentages by operation-aircraft pairs."},
            {"file": "unique_flights_by_runway_aircraft_type.csv", "description": "Unique trajectory counts and percentages by runway-aircraft pairs."},
            {"file": "dropped_repetition_events.csv", "description": "Events removed by repetition dedup rule."},
            {"file": "unique_flights_summary.json", "description": "Machine-readable full summary metadata."},
        ],
    )

    print(f"Summary: {args.outdir / 'unique_flights_summary.json'}")
    print(f"Overview CSV: {args.outdir / 'unique_flights_overview.csv'}")
    print(f"Aircraft type: {args.outdir / 'unique_flights_by_aircraft_type.csv'}")
    print(f"Runway: {args.outdir / 'unique_flights_by_runway.csv'}")
    print(f"Flow: {args.outdir / 'unique_flights_by_flow.csv'}")
    print(f"Operation: {args.outdir / 'unique_flights_by_operation.csv'}")
    print(f"Operation+Runway: {args.outdir / 'unique_flights_by_operation_runway.csv'}")
    print(f"Operation+Aircraft: {args.outdir / 'unique_flights_by_operation_aircraft_type.csv'}")
    print(f"Runway+Aircraft: {args.outdir / 'unique_flights_by_runway_aircraft_type.csv'}")
    print(f"Output index: {args.outdir / 'unique_flights_output_index.csv'}")
    print(f"Dropped repetition events: {args.outdir / 'dropped_repetition_events.csv'}")


if __name__ == "__main__":
    main()
