#!/usr/bin/env python
"""
Filter a preprocessed CSV using flight-level endpoint outlier flags.

Typical workflow:
1. Run `eda_flow_endpoint_outliers.py` to create `flight_endpoint_assessment.csv`
2. Use this script to drop flights flagged as outliers

The filter can be based on:
- start point only
- operation-aware anchor only
- either start or anchor
- both start and anchor
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Drop endpoint outlier flights from a preprocessed CSV.")
    parser.add_argument("--preprocessed", required=True, help="Input preprocessed CSV path.")
    parser.add_argument(
        "--assessment",
        required=True,
        help="Path to flight_endpoint_assessment.csv from eda_flow_endpoint_outliers.py",
    )
    parser.add_argument(
        "--mode",
        choices=["start", "anchor", "either", "both"],
        default="either",
        help="Which outlier flags to apply.",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output filtered CSV path.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    preprocessed_path = Path(args.preprocessed)
    assessment_path = Path(args.assessment)
    out_path = Path(args.out)

    assess = pd.read_csv(assessment_path, usecols=["flight_id", "start_is_outlier", "anchor_is_outlier"])
    assess["flight_id"] = assess["flight_id"].astype(int)
    assess["start_is_outlier"] = assess["start_is_outlier"].astype(bool)
    assess["anchor_is_outlier"] = assess["anchor_is_outlier"].astype(bool)

    if args.mode == "start":
        drop_ids = set(assess.loc[assess["start_is_outlier"], "flight_id"].tolist())
    elif args.mode == "anchor":
        drop_ids = set(assess.loc[assess["anchor_is_outlier"], "flight_id"].tolist())
    elif args.mode == "both":
        mask = assess["start_is_outlier"] & assess["anchor_is_outlier"]
        drop_ids = set(assess.loc[mask, "flight_id"].tolist())
    else:
        mask = assess["start_is_outlier"] | assess["anchor_is_outlier"]
        drop_ids = set(assess.loc[mask, "flight_id"].tolist())

    total_flights = 0
    kept_flights = 0
    rows_in = 0
    rows_out = 0
    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(preprocessed_path, chunksize=200_000):
        rows_in += len(chunk)
        chunk["flight_id"] = chunk["flight_id"].astype(int)
        total_flights += chunk["flight_id"].nunique()
        filtered = chunk[~chunk["flight_id"].isin(drop_ids)].copy()
        rows_out += len(filtered)
        kept_flights += filtered["flight_id"].nunique()
        chunks.append(filtered)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(chunks, ignore_index=True).to_csv(out_path, index=False)

    meta = {
        "preprocessed_csv": str(preprocessed_path),
        "assessment_csv": str(assessment_path),
        "mode": args.mode,
        "rows_in": int(rows_in),
        "rows_out": int(rows_out),
        "dropped_flights": int(len(drop_ids)),
        "output_csv": str(out_path),
    }
    (out_path.with_suffix(".json")).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
