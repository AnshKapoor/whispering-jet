"""Save preprocessed flight data using the backbone pipeline preprocessing stage."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backbone_tracks.config import load_config
from backbone_tracks.io import add_utm_coordinates, ensure_required_columns, load_monthly_csvs, save_dataframe
from backbone_tracks.preprocessing import preprocess_flights
from backbone_tracks.segmentation import segment_flights


def _normalise_identifier(series: pd.Series) -> pd.Series:
    """Return uppercase identifiers with common null-like tokens mapped to NA."""

    normalized = series.astype("string").str.strip().str.upper()
    return normalized.mask(normalized.isin({"", "NA", "NAN", "NONE", "<NA>"}))


def _write_repetition_reports(
    output_dir: Path,
    preprocessed_id: int,
    summary: dict,
    dropped_events: pd.DataFrame,
) -> None:
    """Persist repetition-check diagnostics to JSON and CSV artefacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"preprocessed_{preprocessed_id}_mp_repeat"

    summary_path_json = output_dir / f"{stem}_summary.json"
    summary_path_csv = output_dir / f"{stem}_summary.csv"
    dropped_path_csv = output_dir / f"{stem}_dropped_events.csv"

    summary_path_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame([summary]).to_csv(summary_path_csv, index=False)
    dropped_events.to_csv(dropped_path_csv, index=False)


def apply_repetition_dedup(
    df: pd.DataFrame,
    repetition_cfg: dict,
    preprocessed_id: int,
) -> pd.DataFrame:
    """Drop repeated MP measurement events within identity/date time windows.

    Repetition is defined for rows sharing the same (icao24, callsign, UTC date)
    where distinct event timestamps ``t_ref`` are within ``window_minutes``.
    The earliest event in each close-time cluster is kept and later events are dropped.
    """

    enabled = bool(repetition_cfg.get("enabled", False))
    if not enabled:
        return df

    required_cols = ["icao24", "callsign", "MP", "t_ref"]
    missing_cols = [col for col in required_cols if col not in df.columns]

    action = str(repetition_cfg.get("action", "drop")).strip().lower()
    keep_policy = str(repetition_cfg.get("keep_policy", "earliest_t_ref")).strip().lower()
    timezone = str(repetition_cfg.get("timezone", "UTC")).strip().upper()
    window_minutes = int(repetition_cfg.get("window_minutes", 10))
    require_same_date = bool(repetition_cfg.get("require_same_date", True))
    identity = str(repetition_cfg.get("identity", "icao24_callsign")).strip().lower()
    output_dir = Path(repetition_cfg.get("output_dir", "output/eda/mp_repetition_checks"))

    if action not in {"drop", "audit"}:
        raise ValueError(f"Unsupported repetition_check.action: {action}")
    if keep_policy not in {"earliest_t_ref"}:
        raise ValueError(f"Unsupported repetition_check.keep_policy: {keep_policy}")
    if timezone != "UTC":
        raise ValueError("repetition_check.timezone currently supports only UTC.")
    if identity != "icao24_callsign":
        raise ValueError("repetition_check.identity currently supports only 'icao24_callsign'.")
    if window_minutes <= 0:
        raise ValueError("repetition_check.window_minutes must be > 0.")

    rows_in = int(len(df))
    summary = {
        "rows_in": rows_in,
        "rows_out": rows_in,
        "rows_dropped": 0,
        "events_in": 0,
        "events_kept": 0,
        "events_dropped": 0,
        "identity_day_groups": 0,
        "repeat_clusters": 0,
        "missing_key_rows": 0,
        "window_minutes": window_minutes,
        "timezone": timezone,
        "keep_policy": keep_policy,
        "action": action,
        "enabled": enabled,
        "require_same_date": require_same_date,
        "identity": identity,
        "status": "ok",
    }

    if missing_cols:
        summary["status"] = "skipped_missing_columns"
        summary["missing_columns"] = ",".join(missing_cols)
        _write_repetition_reports(output_dir, preprocessed_id, summary, pd.DataFrame())
        logging.warning(
            "Repetition check skipped: missing columns %s",
            missing_cols,
        )
        return df

    work = df.copy()
    work["_rep_icao24"] = _normalise_identifier(work["icao24"])
    work["_rep_callsign"] = _normalise_identifier(work["callsign"])
    work["_rep_mp"] = _normalise_identifier(work["MP"])
    work["_rep_t_ref_utc"] = pd.to_datetime(work["t_ref"], utc=True, errors="coerce")
    work["_rep_date"] = work["_rep_t_ref_utc"].dt.date

    valid_mask = (
        work["_rep_icao24"].notna()
        & work["_rep_callsign"].notna()
        & work["_rep_mp"].notna()
        & work["_rep_t_ref_utc"].notna()
    )
    if require_same_date:
        valid_mask &= work["_rep_date"].notna()

    summary["missing_key_rows"] = int((~valid_mask).sum())

    event_cols = ["_rep_icao24", "_rep_callsign", "_rep_mp", "_rep_t_ref_utc", "_rep_date"]
    events = work.loc[valid_mask, event_cols].drop_duplicates().copy()
    summary["events_in"] = int(len(events))

    dropped_event_records: list[dict] = []
    group_keys = ["_rep_icao24", "_rep_callsign", "_rep_date"] if require_same_date else ["_rep_icao24", "_rep_callsign"]
    grouped = events.groupby(group_keys, dropna=False)
    summary["identity_day_groups"] = int(grouped.ngroups)

    window_delta = pd.Timedelta(minutes=window_minutes)
    repeat_clusters = 0
    for group_vals, sub in grouped:
        sub_sorted = sub.sort_values("_rep_t_ref_utc").reset_index(drop=True)
        if len(sub_sorted) <= 1:
            continue

        cluster_start = 0
        for idx in range(1, len(sub_sorted) + 1):
            boundary = idx == len(sub_sorted)
            if not boundary:
                gap = sub_sorted.loc[idx, "_rep_t_ref_utc"] - sub_sorted.loc[idx - 1, "_rep_t_ref_utc"]
                boundary = gap > window_delta
            if not boundary:
                continue

            cluster = sub_sorted.iloc[cluster_start:idx].copy()
            if len(cluster) > 1:
                repeat_clusters += 1
                kept = cluster.iloc[0]
                dropped = cluster.iloc[1:]
                for _, row in dropped.iterrows():
                    record = {
                        "icao24": row["_rep_icao24"],
                        "callsign": row["_rep_callsign"],
                        "mp": row["_rep_mp"],
                        "t_ref_utc": row["_rep_t_ref_utc"].isoformat(),
                        "date_utc": str(row["_rep_date"]),
                        "kept_t_ref_utc": kept["_rep_t_ref_utc"].isoformat(),
                    }
                    if require_same_date:
                        if isinstance(group_vals, tuple):
                            record["group_key"] = "|".join(str(v) for v in group_vals)
                        else:
                            record["group_key"] = str(group_vals)
                    dropped_event_records.append(record)

            cluster_start = idx

    summary["repeat_clusters"] = int(repeat_clusters)
    dropped_events = pd.DataFrame(dropped_event_records)
    summary["events_dropped"] = int(len(dropped_events))
    summary["events_kept"] = int(summary["events_in"] - summary["events_dropped"])

    if action == "drop" and not dropped_events.empty:
        drop_keys_df = dropped_events[["icao24", "callsign", "mp", "t_ref_utc"]].copy()
        drop_keys_df["t_ref_utc"] = pd.to_datetime(drop_keys_df["t_ref_utc"], utc=True)
        drop_keys_df = drop_keys_df.drop_duplicates()
        drop_keys_df["__drop"] = True
        marked = work.merge(
            drop_keys_df,
            left_on=["_rep_icao24", "_rep_callsign", "_rep_mp", "_rep_t_ref_utc"],
            right_on=["icao24", "callsign", "mp", "t_ref_utc"],
            how="left",
            sort=False,
        )
        drop_mask = marked["__drop"].notna() & valid_mask
        filtered = work.loc[~drop_mask].copy()
        summary["rows_out"] = int(len(filtered))
        summary["rows_dropped"] = int(drop_mask.sum())
    else:
        filtered = work.copy()

    _write_repetition_reports(output_dir, preprocessed_id, summary, dropped_events)

    logging.info(
        "MP repetition check: rows_in=%d rows_out=%d rows_dropped=%d events_in=%d events_dropped=%d repeat_clusters=%d",
        summary["rows_in"],
        summary["rows_out"],
        summary["rows_dropped"],
        summary["events_in"],
        summary["events_dropped"],
        summary["repeat_clusters"],
    )

    drop_cols = [col for col in filtered.columns if col.startswith("_rep_")]
    if drop_cols:
        filtered = filtered.drop(columns=drop_cols)
    return filtered


def configure_logging(cfg: dict, log_name_override: str | None = None) -> None:
    """Configure logging based on config settings."""

    logging_cfg = cfg.get("logging", {}) or {}
    level_name = str(logging_cfg.get("level", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)
    fmt = "%(levelname)s | %(message)s"

    handlers = []
    log_dir = logging_cfg.get("dir")
    log_file = log_name_override or logging_cfg.get("filename")
    if log_dir and log_file:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(Path(log_dir) / log_file, mode="a", encoding="utf-8"))
    handlers.append(logging.StreamHandler())

    logging.basicConfig(level=level, format=fmt, handlers=handlers)


def _next_preprocessed_id(preprocessed_dir: Path) -> int:
    existing = sorted(preprocessed_dir.glob("preprocessed_*.csv"))
    ids = []
    for path in existing:
        stem = path.stem.replace("preprocessed_", "")
        if stem.isdigit():
            ids.append(int(stem))
    return max(ids, default=0) + 1


def _append_registry(
    registry_path: Path,
    preprocessed_path: Path,
    cfg: dict,
    n_flights: int,
    n_rows: int,
) -> None:
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    preprocessing_cfg = cfg.get("preprocessing", {}) or {}
    smoothing_cfg = preprocessing_cfg.get("smoothing", {}) or {}
    resampling_cfg = preprocessing_cfg.get("resampling", {}) or {}
    interp_method = resampling_cfg.get("method", "time")
    n_points = resampling_cfg.get("n_points")
    smoothing = (
        smoothing_cfg.get("method", "none") if smoothing_cfg.get("enabled", False) else "none"
    )
    include_altitude = bool(cfg.get("features", {}).get("include_altitude", False))
    flow_keys = (cfg.get("flows", {}) or {}).get("flow_keys", [])

    header = (
        "| id | file | n_flights | n_rows | n_points | interpolation | smoothing | include_altitude | flow_keys |"
    )
    sep = "|---|---|---:|---:|---:|---|---|---|---|"
    line = (
        f"| {preprocessed_path.stem.replace('preprocessed_', '')} | {preprocessed_path} | "
        f"{n_flights} | {n_rows} | {n_points} | {interp_method} | {smoothing} | "
        f"{'yes' if include_altitude else 'no'} | {flow_keys} |"
    )

    if not registry_path.exists():
        registry_path.write_text(header + "\n" + sep + "\n" + line + "\n", encoding="utf-8")
        return

    with registry_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    configure_logging(cfg, log_name_override="preprocess.log")

    input_cfg = cfg.get("input", {}) or {}
    csv_glob = input_cfg.get("csv_glob", "Enhanced/matched_*.csv")
    parse_dates = input_cfg.get("parse_dates", ["timestamp"])
    max_rows = None
    testing_cfg = cfg.get("testing", {}) or {}
    if testing_cfg.get("enabled", False):
        max_rows = testing_cfg.get("max_rows_total")

    coord_cfg = cfg.get("coordinates", {}) or {}
    use_utm = bool(coord_cfg.get("use_utm", False))
    utm_crs = coord_cfg.get("utm_crs", "epsg:32632")

    flows_cfg = cfg.get("flows", {}) or {}
    flow_keys = flows_cfg.get("flow_keys", ["Runway"])

    seg_cfg = cfg.get("segmentation", {}) or {}
    preprocessing_cfg = cfg.get("preprocessing", {}) or {}
    smoothing_cfg = preprocessing_cfg.get("smoothing", {})
    resampling_cfg = preprocessing_cfg.get("resampling", {})
    filter_cfg = preprocessing_cfg.get("filter", {}) or {}
    repetition_cfg = preprocessing_cfg.get("repetition_check", {}) or {}
    serial_column = preprocessing_cfg.get("serial_column")
    serial_start = int(preprocessing_cfg.get("serial_start", 1))

    output_cfg = cfg.get("output", {}) or {}
    output_dir = Path(output_cfg.get("dir", "data")) / "preprocessed"
    output_dir.mkdir(parents=True, exist_ok=True)
    preprocessed_id = output_cfg.get("preprocessed_id")
    if preprocessed_id is None:
        preprocessed_id = _next_preprocessed_id(output_dir)
    preprocessed_id_int = int(preprocessed_id)
    out_path = output_dir / f"preprocessed_{preprocessed_id_int}.csv"

    df = load_monthly_csvs(csv_glob=csv_glob, parse_dates=parse_dates, max_rows_total=max_rows)
    df = ensure_required_columns(df)
    df = apply_repetition_dedup(df, repetition_cfg=repetition_cfg, preprocessed_id=preprocessed_id_int)
    if use_utm:
        df = add_utm_coordinates(df, utm_crs=utm_crs)

    df = segment_flights(
        df,
        time_gap_sec=float(seg_cfg.get("time_gap_sec", 60)),
        distance_jump_m=float(seg_cfg.get("distance_jump_m", 600)),
        min_points_per_flight=int(seg_cfg.get("min_points_per_flight", 10)),
        split_on_identity=bool(seg_cfg.get("split_on_identity", True)),
    )

    preprocessed = preprocess_flights(
        df,
        smoothing_cfg=smoothing_cfg,
        resampling_cfg=resampling_cfg,
        filter_cfg=filter_cfg,
        use_utm=use_utm,
        flow_keys=flow_keys,
    )

    if "flight_id" in preprocessed.columns:
        unique_ids = sorted(preprocessed["flight_id"].dropna().unique())
        mapping = {fid: idx + 1 for idx, fid in enumerate(unique_ids)}
        preprocessed["flight_id"] = preprocessed["flight_id"].map(mapping).astype(int)

    if serial_column:
        if "flight_id" not in preprocessed.columns:
            raise ValueError("Cannot assign serial numbers without a flight_id column.")
        unique_ids = sorted(preprocessed["flight_id"].dropna().unique())
        mapping = {fid: idx + serial_start for idx, fid in enumerate(unique_ids)}
        preprocessed[serial_column] = preprocessed["flight_id"].map(mapping).astype(int)

    save_dataframe(preprocessed, out_path)

    n_flights = int(preprocessed["flight_id"].nunique()) if "flight_id" in preprocessed.columns else 0
    n_rows = int(len(preprocessed))
    registry_path = Path("thesis") / "docs" / "preprocessed_registry.md"
    _append_registry(registry_path, out_path, cfg, n_flights, n_rows)
    print(f"Saved preprocessed data to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save preprocessed flight data.")
    parser.add_argument(
        "-c",
        "--config",
        default="config/backbone_full.yaml",
        help="Path to YAML config file.",
    )
    args = parser.parse_args()
    main(args.config)
