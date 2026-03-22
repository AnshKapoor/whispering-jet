"""Compute ground-truth cumulative_res for all flights in batches.

This script runs Doc29 simulations for every flight in a preprocessed dataset,
grouped by aircraft type, runway, and A/D. It aggregates Doc29 output using the
`cumulative_res` column and writes a single ground-truth result that can be
reused across experiments that share the same preprocessed file.

Usage:
  python noise_simulation/run_doc29_ground_truth.py \
    --preprocessed data/preprocessed/preprocessed_1.csv \
    --matched-dir matched_trajectories \
    --output-root noise_simulation/results_ground_truth/exp5_n40

Example input formats:
  - Preprocessed CSV columns: flight_id, step, x_utm, y_utm, A/D, Runway, icao24
  - Matched CSV columns: icao24, aircraft_type_adsb, aircraft_type_noise

Key outputs:
  <output-root>/ground_truth_cumulative.csv
  <output-root>/groups/<A_D>_<Runway>/<AircraftType>/group_cumulative.csv
  <output-root>/summary.json
"""

from __future__ import annotations

import argparse
import json
import logging
import numpy as np
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from noise_simulation import generate_doc29_inputs as doc29_inputs
from noise_simulation.automation.aircraft_types import (
    NPD_TYPE_MAP,
    build_flight_meta,
    build_flight_type_map,
    build_icao_type_map,
    FLIGHT_PROFILE_MAP,
    SPECTRAL_CLASS_MAP,
)
from noise_simulation.automation.doc29_runner import run_doc29
from noise_simulation.automation.flight_csv import (
    FlightCsvConfig,
    FlightEntry,
    build_columns,
    write_flight_csv,
)
from noise_simulation.automation.groundtruth_tracks import generate_groundtruth_tracks
from noise_simulation.receiver_points import annotate_measuring_points


def _npd_suffix(ad: str) -> str:
    """Return NPD suffix for A/D.

    Usage:
      suffix = _npd_suffix("Landung")  # "A"
    """
    if ad == "Landung":
        return "A"
    if ad == "Start":
        return "D"
    raise ValueError(f"Unsupported A/D label: {ad}")


def _safe_name(value: str) -> str:
    """Return a filesystem-friendly name.

    Usage:
      safe = _safe_name("B738/CFM")
    """
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)


def _type_folder_name(icao_type: str, npd_id: str) -> str:
    """Return a collision-safe folder name for type outputs.

    Usage:
      folder = _type_folder_name("E195", "CF3410E")  # E195__CF3410E
    """
    return _safe_name(f"{icao_type}__{npd_id}")


def _relative_to_doc29(doc29_root: Path, path: Path) -> str:
    """Return POSIX-style path relative to doc29_root.

    Usage:
      rel = _relative_to_doc29(doc29_root, doc29_root / "NPD_data" / "B734_A.csv")
    """
    return path.relative_to(doc29_root).as_posix()


def _warn_missing_profiles(logger: logging.Logger, doc29_root: Path) -> None:
    profiles_dir = doc29_root / "Flight_profiles" / "Profiles for Ansh"
    for icao, acft_id in FLIGHT_PROFILE_MAP.items():
        for suffix in ("A", "D"):
            path = profiles_dir / f"{acft_id}_{suffix}.csv"
            if not path.exists():
                logger.warning("Missing height profile for %s: %s", icao, path)


def _resolve_height_profile(doc29_root: Path, aircraft_type: str, ad: str) -> str:
    """Resolve aircraft-specific height profile, fallback to reference.

    Expects profiles at:
      Flight_profiles/Profiles for Ansh/{ACFT_ID}_{A|D}.csv
    """
    suffix = "A" if ad == "Landung" else "D"
    acft_id = FLIGHT_PROFILE_MAP.get(aircraft_type)
    if acft_id:
        candidate = doc29_root / "Flight_profiles" / "Profiles for Ansh" / f"{acft_id}_{suffix}.csv"
        if candidate.exists():
            return _relative_to_doc29(doc29_root, candidate)
    ref = doc29_root / "Flight_profiles" / ("reference_arrival.csv" if ad == "Landung" else "reference_departure.csv")
    return _relative_to_doc29(doc29_root, ref)


def _resolve_spectral_class(doc29_root: Path, aircraft_type: str, ad: str) -> str:
    """Resolve spectral class file by aircraft + operation, fallback to default."""
    key = "approach" if ad == "Landung" else "departure"
    class_id = SPECTRAL_CLASS_MAP.get(aircraft_type, {}).get(key)
    if class_id is not None:
        candidate = doc29_root / "Atmosphere_model" / f"Spectral_class_{class_id}.csv"
        if candidate.exists():
            return _relative_to_doc29(doc29_root, candidate)
    return _relative_to_doc29(doc29_root, doc29_root / "Atmosphere_model" / "Spectral_classes.csv")


def _build_flight_cfg(
    ad: str,
    runway: str,
    npd_table: str,
    height_profile: str,
    spectral_class: str,
    reference_speed: str,
    engine_type: str,
    engine_position: str,
    default_startpoint: str,
) -> FlightCsvConfig:
    """Build shared flight.csv config for a single A/D + runway.

    Usage:
      cfg = _build_flight_cfg("Start", "09L", "NPD_data/B738_D.csv", ...)
    """
    mode = doc29_inputs.MODE_MAP[ad]
    return FlightCsvConfig(
        mode=mode,
        default_startpoint=default_startpoint,
        first_point=doc29_inputs._format_first_point(runway, mode),
        reference_speed=reference_speed,
        engine_type=engine_type,
        engine_position=engine_position,
        npd_table=npd_table,
        runway_file=doc29_inputs.RUNWAY_FILE_MAP[runway],
        height_profile=height_profile,
        nr_night="0",
        spectral_class=spectral_class,
    )


def _configure_logging(log_file: Path) -> logging.Logger:
    """Configure a file + console logger with timestamps.

    Usage:
      logger = _configure_logging(Path("run_ground_truth.log"))
    """
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("doc29_ground_truth")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = "%(asctime)s | %(levelname)s | %(message)s"
    formatter = logging.Formatter(fmt=fmt, datefmt="%Y-%m-%d %H:%M:%S")

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def _chunk_list(values: List[int], batch_size: int) -> Iterable[List[int]]:
    """Yield successive batches from a list.

    Usage:
      for batch in _chunk_list(ids, 500):
          ...
    """
    for idx in range(0, len(values), batch_size):
        yield values[idx : idx + batch_size]


def _add_cumulative(
    accum: Optional[pd.DataFrame],
    new: pd.DataFrame,
) -> pd.DataFrame:
    """Sum cumulative_res by (x, y, z) in energy domain across Doc29 outputs.

    Input format (semicolon CSV):
      x;y;z;...;cumulative_res

    Returns:
      DataFrame with columns: x, y, z, energy
    """
    key_cols = ["x", "y", "z"]
    if "energy" in new.columns and set(key_cols).issubset(new.columns):
        new = new[key_cols + ["energy"]].copy()
    else:
        cols = ["x", "y", "z", "cumulative_res"]
        if not set(cols).issubset(new.columns):
            raise ValueError("Doc29 output missing required columns: x, y, z, cumulative_res")
        new = new[cols].copy()
        new["energy"] = np.power(10.0, new["cumulative_res"].to_numpy() / 10.0)
        new = new.drop(columns=["cumulative_res"])
    if accum is None:
        return new

    # Backward compatibility if accum was loaded in dB-domain.
    if "energy" not in accum.columns and "cumulative_res" in accum.columns:
        accum = accum[["x", "y", "z", "cumulative_res"]].copy()
        accum["energy"] = np.power(10.0, accum["cumulative_res"].to_numpy() / 10.0)
        accum = accum.drop(columns=["cumulative_res"])

    merged = accum.merge(new, on=["x", "y", "z"], how="outer", suffixes=("_a", "_b"))
    merged["energy"] = merged["energy_a"].fillna(0.0) + merged["energy_b"].fillna(0.0)
    return merged[["x", "y", "z", "energy"]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute ground-truth cumulative_res in batches.")
    parser.add_argument("--preprocessed", required=True, help="Preprocessed CSV with flight_id, A/D, Runway, icao24.")
    parser.add_argument("--matched-dir", default="matched_trajectories", help="Directory containing matched_trajs_*.csv.")
    parser.add_argument("--doc29-root", default="noise_simulation/doc-29-implementation", help="Doc29 implementation root.")
    parser.add_argument("--output-root", required=True, help="Output folder for ground-truth results.")
    parser.add_argument("--batch-size", type=int, default=1000, help="Flights per Doc29 batch run.")
    parser.add_argument("--interpolation-length", type=float, default=200.0, help="Groundtrack interpolation spacing.")
    parser.add_argument("--cumulative-metric", default="Leq_day", help="Doc29 cumulative metric.")
    parser.add_argument("--time-in-s", type=float, default=86400 * 365, help="Time horizon in seconds.")
    parser.add_argument("--reference-speed", default="160", help="Reference speed for flight.csv.")
    parser.add_argument("--engine-type", default="turbofan", help="Engine type for flight.csv.")
    parser.add_argument("--engine-position", default="wing-mounted", help="Engine position for flight.csv.")
    parser.add_argument("--default-startpoint", default="True", help="Default startpoint (True/False).")
    parser.add_argument("--keep-tracks", action="store_true", help="Keep generated groundtracks on disk.")
    parser.add_argument("--log-file", default=None, help="Optional log file path.")
    args = parser.parse_args()

    preprocessed_path = Path(args.preprocessed)
    if not preprocessed_path.is_absolute():
        preprocessed_path = (REPO_ROOT / preprocessed_path).resolve()
    if not preprocessed_path.exists():
        # Backward-compatibility for legacy paths that still point to data/preprocessed.
        alt = (REPO_ROOT / "output" / "preprocessed" / preprocessed_path.name).resolve()
        if alt.exists():
            preprocessed_path = alt
        else:
            raise FileNotFoundError(f"Preprocessed CSV not found: {preprocessed_path}")
    matched_dir = Path(args.matched_dir)
    if not matched_dir.is_absolute():
        matched_dir = (REPO_ROOT / matched_dir).resolve()
    matched_paths = sorted(matched_dir.glob("matched_trajs_*.csv"))
    if not matched_paths:
        raise FileNotFoundError(f"No matched_trajs_*.csv in {matched_dir}")

    doc29_root = (REPO_ROOT / args.doc29_root).resolve()
    if not doc29_root.exists():
        raise FileNotFoundError(f"Doc29 root not found: {doc29_root}")

    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = (REPO_ROOT / output_root).resolve()
    log_path = Path(args.log_file) if args.log_file else (output_root / "run_ground_truth.log")
    logger = _configure_logging(log_path)
    _warn_missing_profiles(logger, doc29_root)

    logger.info("Loading flight metadata and aircraft types...")
    flight_meta = build_flight_meta(preprocessed_path)
    icao_set = {meta["icao24"] for meta in flight_meta.values() if meta["icao24"] != "UNKNOWN"}
    icao_map = build_icao_type_map(matched_paths, icao_set)
    flight_type_map = build_flight_type_map(flight_meta, icao_map)

    grouped: Dict[Tuple[str, str, str], List[int]] = {}
    allowed_types = set(NPD_TYPE_MAP.keys())
    for fid, meta in flight_meta.items():
        ad = meta.get("A/D", "UNKNOWN")
        runway = meta.get("Runway", "UNKNOWN")
        aircraft_type = flight_type_map.get(fid, "UNKNOWN")
        if aircraft_type not in allowed_types:
            continue
        key = (ad, runway, aircraft_type)
        grouped.setdefault(key, []).append(fid)

    output_root.mkdir(parents=True, exist_ok=True)
    groups_root = output_root / "groups"
    groundtracks_root = doc29_root / "Groundtracks" / "ground_truth"
    flight_csv_root = output_root / "flight_csv"

    global_accum: Optional[pd.DataFrame] = None
    group_summaries = []

    for (ad, runway, aircraft_type), flight_ids in sorted(grouped.items()):
        if ad not in doc29_inputs.MODE_MAP or runway not in doc29_inputs.RUNWAY_FILE_MAP:
            logger.info("Skipping unsupported flow: %s %s", ad, runway)
            continue
        if aircraft_type == "UNKNOWN":
            logger.info("Skipping UNKNOWN aircraft type for %s %s", ad, runway)
            continue

        npd_id = NPD_TYPE_MAP.get(aircraft_type, aircraft_type)
        npd_suffix = _npd_suffix(ad)
        npd_path = doc29_root / "NPD_data" / f"{npd_id}_{npd_suffix}.csv"
        if not npd_path.exists():
            logger.info("Skipping %s: missing NPD table %s", npd_id, npd_path)
            continue

        height_profile = _resolve_height_profile(doc29_root, aircraft_type, ad)
        spectral_class = _resolve_spectral_class(doc29_root, aircraft_type, ad)
        cfg = _build_flight_cfg(
            ad=ad,
            runway=runway,
            npd_table=_relative_to_doc29(doc29_root, npd_path),
            height_profile=height_profile,
            spectral_class=spectral_class,
            reference_speed=args.reference_speed,
            engine_type=args.engine_type,
            engine_position=args.engine_position,
            default_startpoint=args.default_startpoint,
        )

        flow_name = f"{ad}_{runway}"
        # Keep ICAO + NPD in folder name to avoid collisions when multiple ICAO
        # types share the same NPD ID (e.g., E170/E75L -> CF348E).
        safe_type = _type_folder_name(aircraft_type, npd_id)
        group_dir = groups_root / flow_name / safe_type
        group_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Processing group %s / %s (%s) (%d flights)",
            flow_name,
            aircraft_type,
            npd_id,
            len(flight_ids),
        )

        group_accum: Optional[pd.DataFrame] = None
        batches = list(_chunk_list(sorted(flight_ids), args.batch_size))
        for batch_idx, batch_ids in enumerate(batches, start=1):
            batch_tag = f"batch_{batch_idx:04d}"
            logger.info("  Batch %s with %d flights", batch_tag, len(batch_ids))

            tracks_out = groundtracks_root / flow_name / safe_type / batch_tag
            flights_by_type = {aircraft_type: batch_ids}
            tracks = generate_groundtruth_tracks(
                preprocessed_path,
                flights_by_type,
                tracks_out,
                runway=runway,
                mode=doc29_inputs.MODE_MAP[ad],
                interpolation_length=args.interpolation_length,
            )

            entries: List[FlightEntry] = []
            for col_idx, (fid, track_path) in enumerate(tracks.get(aircraft_type, []), start=1):
                rel = _relative_to_doc29(doc29_root, track_path)
                entries.append(FlightEntry(f"Flight {col_idx}", rel, "1"))

            if not entries:
                logger.info("  No tracks generated for batch %s, skipping.", batch_tag)
                continue

            flight_csv = flight_csv_root / flow_name / safe_type / f"Flight_groundtruth_{batch_tag}.csv"
            columns = build_columns(entries, cfg)
            write_flight_csv(columns, flight_csv)

            output_csv = group_dir / f"groundtruth_{batch_tag}.csv"
            run_doc29(
                doc29_root,
                doc29_root / "Input_Airport.csv",
                flight_csv,
                args.cumulative_metric,
                args.time_in_s,
                output_csv,
            )

            batch_df = pd.read_csv(output_csv, sep=";")
            group_accum = _add_cumulative(group_accum, batch_df)

            if not args.keep_tracks:
                for _, track_path in tracks.get(aircraft_type, []):
                    track_path.unlink(missing_ok=True)
                if tracks_out.exists():
                    try:
                        tracks_out.rmdir()
                    except OSError:
                        pass

        if group_accum is None:
            logger.info("  No output for group %s / %s", flow_name, npd_id)
            continue

        group_out = group_dir / "group_cumulative.csv"
        group_out_df = group_accum.copy()
        group_out_df["cumulative_res"] = 10.0 * np.log10(
            np.maximum(group_out_df["energy"].to_numpy(), 1e-12)
        )
        group_out_df = group_out_df[["x", "y", "z", "cumulative_res"]]
        group_out_df = annotate_measuring_points(group_out_df)
        group_out_df.to_csv(group_out, index=False)
        global_accum = _add_cumulative(global_accum, group_accum)

        group_summaries.append(
            {
                "flow": flow_name,
                "aircraft_type": aircraft_type,
                "npd_id": npd_id,
                "type_folder": safe_type,
                "n_flights": len(flight_ids),
                "group_output": str(group_out),
            }
        )

    if global_accum is None:
        raise RuntimeError("No ground-truth outputs were generated. Check inputs and NPD tables.")

    global_out = output_root / "ground_truth_cumulative.csv"
    global_out_df = global_accum.copy()
    global_out_df["cumulative_res"] = 10.0 * np.log10(
        np.maximum(global_out_df["energy"].to_numpy(), 1e-12)
    )
    global_out_df = global_out_df[["x", "y", "z", "cumulative_res"]]
    global_out_df = annotate_measuring_points(global_out_df)
    global_out_df.to_csv(global_out, index=False)

    summary = {
        "preprocessed": str(preprocessed_path),
        "output": str(global_out),
        "groups": group_summaries,
        "batch_size": args.batch_size,
        "cumulative_metric": args.cumulative_metric,
        "time_in_s": args.time_in_s,
    }
    with (output_root / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    logger.info("Ground truth complete. Output: %s", global_out)


if __name__ == "__main__":
    main()
