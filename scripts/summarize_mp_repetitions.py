"""Summarize MP repetition-check diagnostics across preprocessed IDs.

Reads per-run summary files created by scripts/save_preprocessed.py:
  output/eda/mp_repetition_checks/preprocessed_<id>_mp_repeat_summary.csv

Writes a consolidated CSV and prints a compact table to stdout.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _read_one_summary(path: Path, preprocessed_id: int) -> dict:
    if not path.exists():
        return {
            "preprocessed_id": preprocessed_id,
            "status": "missing_report",
            "rows_in": pd.NA,
            "rows_out": pd.NA,
            "rows_dropped": pd.NA,
            "events_in": pd.NA,
            "events_kept": pd.NA,
            "events_dropped": pd.NA,
            "identity_day_groups": pd.NA,
            "repeat_clusters": pd.NA,
            "missing_key_rows": pd.NA,
            "window_minutes": pd.NA,
            "timezone": pd.NA,
            "keep_policy": pd.NA,
            "action": pd.NA,
        }

    df = pd.read_csv(path)
    if df.empty:
        return {
            "preprocessed_id": preprocessed_id,
            "status": "empty_report",
            "rows_in": pd.NA,
            "rows_out": pd.NA,
            "rows_dropped": pd.NA,
            "events_in": pd.NA,
            "events_kept": pd.NA,
            "events_dropped": pd.NA,
            "identity_day_groups": pd.NA,
            "repeat_clusters": pd.NA,
            "missing_key_rows": pd.NA,
            "window_minutes": pd.NA,
            "timezone": pd.NA,
            "keep_policy": pd.NA,
            "action": pd.NA,
        }

    row = df.iloc[0].to_dict()
    row["preprocessed_id"] = preprocessed_id
    row.setdefault("status", "ok")
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize MP repetition reports for preprocessed files.")
    parser.add_argument(
        "--reports-dir",
        default="output/eda/mp_repetition_checks",
        help="Directory containing preprocessed_<id>_mp_repeat_summary.csv files.",
    )
    parser.add_argument("--id-start", type=int, default=1, help="Start preprocessed ID (inclusive).")
    parser.add_argument("--id-end", type=int, default=10, help="End preprocessed ID (inclusive).")
    parser.add_argument(
        "--out",
        default=None,
        help=(
            "Output CSV path. Default: <reports-dir>/repetition_counts_preprocessed_<start>_<end>.csv"
        ),
    )
    args = parser.parse_args()

    if args.id_end < args.id_start:
        raise ValueError("--id-end must be >= --id-start")

    reports_dir = Path(args.reports_dir)
    out_path = (
        Path(args.out)
        if args.out
        else reports_dir / f"repetition_counts_preprocessed_{args.id_start}_{args.id_end}.csv"
    )

    rows: list[dict] = []
    for pre_id in range(args.id_start, args.id_end + 1):
        summary_path = reports_dir / f"preprocessed_{pre_id}_mp_repeat_summary.csv"
        rows.append(_read_one_summary(summary_path, pre_id))

    result = pd.DataFrame(rows)
    ordered_cols = [
        "preprocessed_id",
        "status",
        "rows_in",
        "rows_out",
        "rows_dropped",
        "events_in",
        "events_kept",
        "events_dropped",
        "identity_day_groups",
        "repeat_clusters",
        "missing_key_rows",
        "window_minutes",
        "timezone",
        "keep_policy",
        "action",
    ]
    existing_cols = [c for c in ordered_cols if c in result.columns]
    extra_cols = [c for c in result.columns if c not in existing_cols]
    result = result[existing_cols + extra_cols].sort_values("preprocessed_id")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False)

    print(f"Saved summary: {out_path}")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
