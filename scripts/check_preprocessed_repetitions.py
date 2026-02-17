"""Audit repeated-flight measurement events for existing preprocessed outputs.

The repetition rule matches the preprocessing logic:
- identity: (icao24, callsign)
- same UTC date from t_ref
- consecutive event times within ``window_minutes`` are grouped
- earliest t_ref in each group is the kept representative

If a preprocessed CSV does not contain required columns
(`icao24`, `callsign`, `MP`, `t_ref`), the script can fall back to
`matched_trajectories` source CSVs so existing preprocessed IDs can still be audited.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any

import pandas as pd

REQUIRED_COLS = ["icao24", "callsign", "MP", "t_ref"]


def _normalise_identifier(series: pd.Series) -> pd.Series:
    """Uppercase, strip, and map common null-like tokens to missing."""

    normalized = series.astype("string").str.strip().str.upper()
    return normalized.mask(normalized.isin({"", "NA", "NAN", "NONE", "<NA>"}))


def _extract_preprocessed_id(path: Path) -> int | None:
    """Extract integer ID from filename `preprocessed_<id>.csv`."""

    stem = path.stem
    if not stem.startswith("preprocessed_"):
        return None
    token = stem.replace("preprocessed_", "", 1)
    return int(token) if token.isdigit() else None


def _load_matched_required_cols(csv_glob: str) -> pd.DataFrame:
    """Load only required columns from matched trajectory CSV files."""

    paths = sorted(glob.glob(csv_glob))
    if not paths:
        raise FileNotFoundError(f"No files matched: {csv_glob}")

    frames: list[pd.DataFrame] = []
    for path in paths:
        cols = pd.read_csv(path, nrows=0).columns
        missing = [c for c in REQUIRED_COLS if c not in cols]
        if missing:
            raise ValueError(f"Missing required columns in {path}: {missing}")
        frames.append(pd.read_csv(path, usecols=REQUIRED_COLS, low_memory=False))
    return pd.concat(frames, ignore_index=True)


def detect_repetitions(
    df: pd.DataFrame,
    window_minutes: int,
    require_same_date: bool = True,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Return audit summary and dropped-event table for repetition windows."""

    if window_minutes <= 0:
        raise ValueError("window_minutes must be > 0")

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

    summary: dict[str, Any] = {
        "rows_in": int(len(work)),
        "rows_out": int(len(work)),
        "rows_dropped": 0,
        "events_in": 0,
        "events_kept": 0,
        "events_dropped": 0,
        "identity_day_groups": 0,
        "repeat_clusters": 0,
        "missing_key_rows": int((~valid_mask).sum()),
        "window_minutes": int(window_minutes),
        "require_same_date": bool(require_same_date),
    }

    event_cols = ["_rep_icao24", "_rep_callsign", "_rep_mp", "_rep_t_ref_utc", "_rep_date"]
    events = work.loc[valid_mask, event_cols].drop_duplicates().copy()
    summary["events_in"] = int(len(events))

    group_keys = ["_rep_icao24", "_rep_callsign", "_rep_date"] if require_same_date else ["_rep_icao24", "_rep_callsign"]
    grouped = events.groupby(group_keys, dropna=False)
    summary["identity_day_groups"] = int(grouped.ngroups)

    window_delta = pd.Timedelta(minutes=window_minutes)
    dropped_event_records: list[dict[str, Any]] = []
    repeat_clusters = 0

    for _, sub in grouped:
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

            cluster = sub_sorted.iloc[cluster_start:idx]
            if len(cluster) > 1:
                repeat_clusters += 1
                kept = cluster.iloc[0]
                for _, row in cluster.iloc[1:].iterrows():
                    dropped_event_records.append(
                        {
                            "icao24": row["_rep_icao24"],
                            "callsign": row["_rep_callsign"],
                            "MP": row["_rep_mp"],
                            "t_ref_utc": row["_rep_t_ref_utc"].isoformat(),
                            "date_utc": str(row["_rep_date"]),
                            "kept_t_ref_utc": kept["_rep_t_ref_utc"].isoformat(),
                        }
                    )
            cluster_start = idx

    dropped_events = pd.DataFrame(dropped_event_records)
    summary["repeat_clusters"] = int(repeat_clusters)
    summary["events_dropped"] = int(len(dropped_events))
    summary["events_kept"] = int(summary["events_in"] - summary["events_dropped"])

    if not dropped_events.empty:
        drop_keys_df = dropped_events[["icao24", "callsign", "MP", "t_ref_utc"]].copy()
        drop_keys_df["t_ref_utc"] = pd.to_datetime(drop_keys_df["t_ref_utc"], utc=True)
        drop_keys_df = drop_keys_df.drop_duplicates()
        drop_keys_df["__drop"] = True
        marked = work.merge(
            drop_keys_df,
            left_on=["_rep_icao24", "_rep_callsign", "_rep_mp", "_rep_t_ref_utc"],
            right_on=["icao24", "callsign", "MP", "t_ref_utc"],
            how="left",
            sort=False,
        )
        summary["rows_dropped"] = int((marked["__drop"].notna() & valid_mask).sum())

    return summary, dropped_events


def _write_report(
    outdir: Path,
    stem: str,
    summary: dict[str, Any],
    dropped_events: pd.DataFrame,
) -> None:
    """Write summary and dropped-event reports."""

    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"{stem}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame([summary]).to_csv(outdir / f"{stem}_summary.csv", index=False)
    dropped_events.to_csv(outdir / f"{stem}_dropped_events.csv", index=False)


def main() -> None:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(description="Audit repeated flights for existing preprocessed CSVs.")
    parser.add_argument("--preprocessed-dir", type=Path, default=Path("output/preprocessed"))
    parser.add_argument("--matched-glob", type=str, default="matched_trajectories/matched_trajs_*.csv")
    parser.add_argument("--window-minutes", type=int, default=10)
    parser.add_argument("--id-start", type=int, default=None)
    parser.add_argument("--id-end", type=int, default=None)
    parser.add_argument("--outdir", type=Path, default=Path("output/eda/mp_repetition_checks_existing"))
    parser.add_argument(
        "--no-fallback-matched",
        action="store_true",
        help="Disable fallback to matched trajectories when preprocessed CSV lacks required columns.",
    )
    args = parser.parse_args()

    files = sorted(args.preprocessed_dir.glob("preprocessed_*.csv"))
    if args.id_start is not None:
        files = [p for p in files if (_extract_preprocessed_id(p) or -1) >= args.id_start]
    if args.id_end is not None:
        files = [p for p in files if (_extract_preprocessed_id(p) or 10**9) <= args.id_end]

    if not files:
        raise FileNotFoundError(f"No preprocessed files found in range under {args.preprocessed_dir}")

    fallback_df: pd.DataFrame | None = None
    fallback_enabled = not args.no_fallback_matched
    all_rows: list[dict[str, Any]] = []

    for path in files:
        stem = path.stem
        header_cols = set(pd.read_csv(path, nrows=0).columns.tolist())
        missing_cols = [c for c in REQUIRED_COLS if c not in header_cols]

        source = "preprocessed_csv"
        note = ""
        if missing_cols:
            if not fallback_enabled:
                summary = {
                    "preprocessed_file": str(path),
                    "source": "unavailable",
                    "status": "skipped_missing_columns",
                    "missing_columns": ",".join(missing_cols),
                }
                _write_report(args.outdir, f"{stem}_mp_repeat_audit", summary, pd.DataFrame())
                all_rows.append(summary)
                continue

            if fallback_df is None:
                fallback_df = _load_matched_required_cols(args.matched_glob)
            df_use = fallback_df
            source = "matched_trajectories_fallback"
            note = f"preprocessed_missing={','.join(missing_cols)}"
        else:
            df_use = pd.read_csv(path, usecols=REQUIRED_COLS, low_memory=False)

        summary, dropped_events = detect_repetitions(df_use, window_minutes=args.window_minutes)
        summary["preprocessed_file"] = str(path)
        summary["preprocessed_id"] = _extract_preprocessed_id(path)
        summary["source"] = source
        summary["status"] = "ok"
        if note:
            summary["note"] = note

        _write_report(args.outdir, f"{stem}_mp_repeat_audit", summary, dropped_events)
        all_rows.append(summary)

    df_all = pd.DataFrame(all_rows)
    args.outdir.mkdir(parents=True, exist_ok=True)
    df_all.to_csv(args.outdir / "preprocessed_mp_repeat_audit_summary.csv", index=False)

    ok = int((df_all["status"] == "ok").sum()) if "status" in df_all.columns else 0
    print(f"Audited files: {len(df_all)} (ok={ok})")
    print(f"Summary CSV: {args.outdir / 'preprocessed_mp_repeat_audit_summary.csv'}")


if __name__ == "__main__":
    main()
