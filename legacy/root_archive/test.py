"""Utility script to match aircraft noise measurements with ADS-B trajectories."""

import argparse
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from pyproj import Transformer

try:
    from traffic.data import aircraft as aircraft_db
except ImportError:  # pragma: no cover - optional dependency
    aircraft_db = None

# ---------------------------
# Helpers for robust Excel IO
# ---------------------------

HEADER_TOKENS = {
    "mp", "ata/atd", "a/d", "runway", "sid/star", "flugzeugtyp", "mtom",
    "triebwerk", "tlasmax", "abstand", "hohenwinkel", "höhe", "lasmax",
    "leq", "lae", "t10", "tgesamt", "azb-klasse"
}


logger = logging.getLogger(__name__)


def setup_logging(log_file: Optional[str] = None) -> Path:
    """Configure logging for the script and return the log file path.

    Parameters
    ----------
    log_file:
        Optional explicit log file path supplied via CLI. When omitted a
        timestamped file will be created inside ``logs/python``.

    Returns
    -------
    Path
        Path to the log file that captures the execution details.
    """

    # Determine the log file destination while ensuring the directory exists.
    if log_file:
        log_path = Path(log_file)
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        log_dir = Path("logs") / "python"
        log_path = log_dir / f"{timestamp}.log"

    log_path.parent.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # Remove stale handlers to prevent duplicate log entries on repeated runs.
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(
        logging.Formatter("%(levelname)s | %(message)s")
    )
    root_logger.addHandler(console_handler)

    root_logger.debug("Logging initialised. Writing detailed log to %s", log_path)
    return log_path


def _norm(value: Any) -> str:
    """Normalize header tokens by removing whitespace and harmonising accents."""

    string_value: str = "" if pd.isna(value) else str(value)
    string_value = re.sub(r"[\u00A0\s]+", " ", string_value).strip().lower()
    # unify common diacritics/variants that appear in headers
    string_value = (
        string_value
        .replace("ö", "o").replace("ä", "a").replace("ü", "u")
        .replace("[", "").replace("]", "")
    )
    return string_value


def _looks_like_header(cells: list[str]) -> bool:
    """Return ``True`` when a row resembles a header based on known tokens."""

    normed = [_norm(x) for x in cells]
    normed = [x for x in normed if x]
    if not normed:
        return False

    tokens = set()
    for x in normed:
        x2 = x.replace(" [m]", "")
        if "abstand" in x2:
            tokens.add("abstand")
        if "tlas" in x2:
            tokens.add("tlasmax")
        if "sid" in x2 or "star" in x2:
            tokens.add("sid/star")
        if "hohenwinkel" in x2 or "höhenwinkel" in x2:
            tokens.add("hohenwinkel")
        if x2 in HEADER_TOKENS:
            tokens.add(x2)
    return ("tlasmax" in tokens) and (len(tokens) >= 5)


def _read_noise_sheet_autoheader(excel_path: str, sheet_name: str) -> pd.DataFrame:
    """Read a sheet without assuming header row; detect it automatically."""

    logger.debug("Reading sheet '%s' with automatic header detection", sheet_name)
    raw = pd.read_excel(excel_path, sheet_name=sheet_name, header=None, dtype=object)

    scan_rows = min(20, len(raw))
    header_idx = None
    for r in range(scan_rows):
        row_vals = raw.iloc[r].tolist()
        if _looks_like_header(row_vals):
            header_idx = r
            break
    if header_idx is None:
        logger.debug("No explicit header row found in sheet '%s'; using first row", sheet_name)
        header_idx = 0  # fallback

    cols = raw.iloc[header_idx].astype(str).tolist()
    cols = [re.sub(r"[\u00A0\s]+", " ", c).strip() for c in cols]

    df = raw.iloc[header_idx + 1:].copy()
    df.columns = cols

    # drop fully empty columns/rows
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")

    # If MP missing, backfill with sheet name
    if not any(_norm(c) == "mp" for c in df.columns):
        df["MP"] = sheet_name

    return df


# ---------------------------
# Core matching functionality
# ---------------------------

# Maximum number of matches to generate when ``test_mode`` is enabled.
TEST_MODE_MATCH_LIMIT: int = 5


def match_noise_to_adsb(
    noise_xlsx: str,
    adsb_joblib: str,
    out_traj_parquet: Optional[str] = None,
    tol_sec: int = 10,
    buffer_frac: float = 0.5,
    window_min: int = 3,
    test_mode: bool = False,
    dedupe_traj: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Match Excel noise measurements to ADS-B trajectories from a joblib file.

    Parameters
    ----------
    noise_xlsx:
        Path to the Excel workbook that contains the noise measurements.
    adsb_joblib:
        Path to the joblib bundle storing ADS-B trajectories.
    out_traj_parquet:
        Optional destination for the extracted trajectory slices.
    tol_sec:
        Temporal tolerance (in seconds) applied while searching ADS-B hits.
    buffer_frac:
        Spatial buffer fraction multiplied with the measurement distance.
    window_min:
        Half width (in minutes) of the trajectory extraction window.
    test_mode:
        When ``True`` the workflow exits after :data:`TEST_MODE_MATCH_LIMIT`
        matches are found so that the early output can be inspected quickly.
    dedupe_traj:
        When ``True`` identical trajectory points keyed by ``icao24`` and
        ``timestamp`` will be dropped to produce a clean, deduplicated output.

    Notes
    -----
    The extracted trajectory slices are automatically trimmed to a 5 km radius
    around the Hanover Airport centre and annotated with noise/ADS-B aircraft
    type metadata to support downstream clustering tasks.
    """

    logger.info("Reading Excel workbook with automatic header detection: %s", noise_xlsx)
    xls = pd.ExcelFile(noise_xlsx)
    frames: list[pd.DataFrame] = []
    for sheet in xls.sheet_names:
        df = _read_noise_sheet_autoheader(noise_xlsx, sheet)
        logger.info("Detected %d columns in sheet '%s'", len(df.columns), sheet)
        logger.debug("Sheet '%s' columns: %s", sheet, list(df.columns))
        frames.append(df)

    df_noise = pd.concat(frames, ignore_index=True)

    # Retain aircraft type column from Excel and normalise into ``aircraft_type_noise``.
    if "Flugzeugtyp" in df_noise.columns:
        df_noise["aircraft_type_noise"] = df_noise["Flugzeugtyp"].astype("string")
    else:
        df_noise["aircraft_type_noise"] = pd.NA
    df_noise["aircraft_type_noise"] = df_noise["aircraft_type_noise"].astype("string")

    # Normalize MP labels like "M 01" -> "MP1"
    def _normalize_mp(value: Any) -> str:
        """Return MP identifiers in canonical ``MP#`` form."""

        string_value = str(value)
        string_value = re.sub(r"[\u00A0\s]+", " ", string_value).strip().upper()
        string_value = (
            string_value
            .replace("M ", "MP")
            .replace("M0", "MP0")
            .replace("M", "MP")
            .replace(" ", "")
        )
        if not string_value.startswith("MP"):
            string_value = "MP" + string_value
        string_value = re.sub(r"^MP0*([1-9]\d*)$", r"MP\1", string_value)
        return string_value

    mp_candidates = [c for c in df_noise.columns if _norm(c) == "mp"]
    mp_col = mp_candidates[0] if mp_candidates else "MP"
    logger.debug("Using column '%s' for MP normalisation", mp_col)
    df_noise[mp_col] = df_noise[mp_col].apply(_normalize_mp)
    df_noise.rename(columns={mp_col: "MP"}, inplace=True)

    # Identify TLASmax column robustly
    tlas_cols = [c for c in df_noise.columns if re.search(r"tlas", _norm(c))]
    if not tlas_cols:
        tlas_col = df_noise.columns[min(8, len(df_noise.columns) - 1)]
        logger.warning(
            "No column containing 'tlas' detected; defaulting to column #%d ('%s')",
            min(8, len(df_noise.columns) - 1),
            tlas_col,
        )
    else:
        tlas_col = tlas_cols[0]
    df_noise.rename(columns={tlas_col: "TLASmax"}, inplace=True)

    # Distance column
    dist_cols = [c for c in df_noise.columns if re.search(r"abstand", _norm(c))]
    dist_col = dist_cols[0] if dist_cols else "Abstand [m]"
    logger.debug("Using distance column '%s'", dist_col)
    df_noise["Abstand [m]"] = pd.to_numeric(df_noise.get(dist_col, np.nan), errors="coerce")

    # ------------------------------
    # Parse timestamps (DST-safe)
    # ------------------------------
    logger.info("Parsing TLASmax timestamps with DST-safe logic")

    # Parse TLASmax as naive datetime (no timezone yet)
    t_raw = pd.to_datetime(df_noise["TLASmax"], errors="coerce", dayfirst=True)

    # 1) Localize to Europe/Berlin
    t_loc = t_raw.dt.tz_localize(
        "Europe/Berlin",
        ambiguous="NaT",  # ambiguous fall-back hour -> NaT
        nonexistent="shift_forward",  # spring-forward gap -> shift
    )

    # 2) Retry ambiguous rows as DST (summer time)
    amb_mask = t_loc.isna() & t_raw.notna()
    if amb_mask.any():
        logger.debug("Retrying %d ambiguous timestamps with DST assumption", int(amb_mask.sum()))
        t_retry = t_raw[amb_mask].dt.tz_localize(
            "Europe/Berlin",
            ambiguous=True,
            nonexistent="shift_forward",
        )
        t_loc.loc[amb_mask] = t_retry

    # 3) Convert to UTC
    df_noise["t_ref"] = t_loc.dt.tz_convert("UTC")

    num_bad = int(df_noise["t_ref"].isna().sum())
    if num_bad:
        logger.warning("%d TLASmax rows could not be parsed/localized (NaT).", num_bad)

    # Prepare match fields
    df_noise["icao24"] = pd.NA
    df_noise["callsign"] = pd.NA
    df_noise["aircraft_type_adsb"] = pd.NA
    df_noise["aircraft_type_match"] = pd.NA

    # -------------------------
    # Load ADS-B (joblib DF) - *more* robust coercion & introspection
    # -------------------------
    logger.info("Loading ADS-B joblib bundle: %s", adsb_joblib)
    raw_ads_payload: Any = joblib.load(adsb_joblib)
    ads_obj: Any = getattr(raw_ads_payload, "data", raw_ads_payload)

    def _inspect(obj: Any) -> str:
        """Return a human readable description of arbitrary joblib payloads."""

        try:
            import numpy as _np
            obj_type = type(obj)
            if isinstance(obj, dict):
                keys = list(obj.keys())[:20]
                return f"type={obj_type.__name__}, dict_keys={keys}"
            if isinstance(obj, (list, tuple)):
                first_type = type(obj[0]).__name__ if obj else None
                return f"type={obj_type.__name__}, len={len(obj)}, first_type={first_type}"
            if "DataFrame" in obj_type.__name__:
                try:
                    shape = getattr(obj, "shape", None)
                    return f"type={obj_type.__name__}, shape={shape}"
                except Exception:
                    return f"type={obj_type.__name__}"
            if isinstance(obj, _np.ndarray):
                return f"type=np.ndarray, shape={obj.shape}, dtype={obj.dtype}"
            return f"type={obj_type.__name__}"
        except Exception:
            return f"type={type(obj)}"

    def _as_dataframe(obj: Any) -> pd.DataFrame:
        """Convert a joblib payload into a Pandas DataFrame when possible."""

        # 0) traffic objects (duck-typed)
        try:
            if hasattr(obj, "data") and isinstance(getattr(obj, "data"), pd.DataFrame):
                return getattr(obj, "data")
        except Exception:
            logger.debug("Failed to interpret payload as traffic object", exc_info=True)

        # 1) direct pandas
        if isinstance(obj, pd.DataFrame):
            return obj

        # 2) duck types from other libs
        try:
            import pyarrow as pa  # type: ignore
            if isinstance(obj, pa.Table):
                return obj.to_pandas()
        except Exception:
            logger.debug("pyarrow conversion unavailable", exc_info=True)
        try:
            import polars as pl  # type: ignore
            if isinstance(obj, pl.DataFrame):
                return obj.to_pandas()
        except Exception:
            logger.debug("polars conversion unavailable", exc_info=True)
        try:
            import dask.dataframe as dd  # type: ignore
            if isinstance(obj, dd.DataFrame):
                return obj.compute()
        except Exception:
            logger.debug("dask conversion unavailable", exc_info=True)

        # 3) dict containers
        if isinstance(obj, dict):
            for key in ["df", "data", "adsb", "table"]:
                if key in obj and isinstance(obj[key], pd.DataFrame):
                    return obj[key]
            for value in obj.values():
                # nested conversions (e.g., dict holding a Traffic object)
                try:
                    return _as_dataframe(value)
                except Exception:
                    continue
            # dict of equal-length lists/arrays
            try:
                df_try = pd.DataFrame(obj)
                if not df_try.empty and df_try.columns.size > 1:
                    return df_try
            except Exception:
                logger.debug("Dict payload could not be converted directly", exc_info=True)

        # 4) list/tuple containers
        if isinstance(obj, (list, tuple)):
            # list of traffic Flight objects or any objects with .data
            try:
                dfs = []
                for item in obj:
                    if hasattr(item, "data") and isinstance(getattr(item, "data"), pd.DataFrame):
                        dfs.append(getattr(item, "data"))
                if dfs:
                    return pd.concat(dfs, ignore_index=True)
            except Exception:
                logger.debug("Failed to concatenate list of traffic objects", exc_info=True)
            # list of DFs
            if all(isinstance(item, pd.DataFrame) for item in obj):
                return pd.concat(list(obj), ignore_index=True)
            # first element DF
            if obj and isinstance(obj[0], pd.DataFrame):
                return obj[0]
            # list of dicts
            if obj and isinstance(obj[0], dict):
                return pd.DataFrame(list(obj))
            # list/tuple -> try DataFrame
            try:
                return pd.DataFrame(obj)
            except Exception:
                logger.debug("Generic list/tuple payload could not be converted", exc_info=True)

        # 5) numpy structured array / recarray
        try:
            import numpy as _np
            if isinstance(obj, _np.ndarray) and obj.dtype.names:
                return pd.DataFrame(obj)
        except Exception:
            logger.debug("NumPy structured array conversion failed", exc_info=True)

        # 6) path-like payloads -> read parquet/csv
        if isinstance(obj, (str, Path)):
            path_obj = Path(obj)
            if path_obj.suffix.lower() in {".parquet", ".pq"} and path_obj.exists():
                return pd.read_parquet(path_obj)
            if path_obj.suffix.lower() in {".csv"} and path_obj.exists():
                return pd.read_csv(path_obj)

        raise TypeError("Could not coerce joblib payload to DataFrame: " + _inspect(obj))

    try:
        df_ads = _as_dataframe(ads_obj).copy()
    except Exception as exc:
        logger.error("Unsupported joblib payload: %s", _inspect(ads_obj))
        raise TypeError(
            "ADS-B joblib should contain a table-like object. Supported: pandas/pyarrow/polars/dask; "
            "dicts/lists/tuples holding those; numpy structured arrays; or a path to parquet/csv."
            f" Details: {exc}"
        ) from exc

    # Validate & normalize timestamp/geo columns
    required_cols = {"timestamp", "latitude", "longitude", "icao24", "callsign"}
    missing = [c for c in required_cols if c not in df_ads.columns]
    if missing:
        lower_map = {str(c).strip().lower(): c for c in df_ads.columns}
        for want in list(missing):
            if want in lower_map:
                df_ads.rename(columns={lower_map[want]: want}, inplace=True)
                missing.remove(want)
        if missing:
            raise ValueError(f"ADS-B DataFrame missing required columns: {missing}; columns present={list(df_ads.columns)}")

    df_ads["timestamp"] = pd.to_datetime(df_ads["timestamp"], utc=True, errors="coerce")
    df_ads = (
        df_ads.dropna(subset=["timestamp", "latitude", "longitude"])
        .dropna(subset=["icao24", "callsign"], how="any")  # ensure identifiers are present
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    logger.info(
        "ADS-B dataset contains %d rows and columns %s",
        len(df_ads),
        list(df_ads.columns),
    )

    # ---------------------------------------
    # MP receiver coordinates (HAJ region)
    # ---------------------------------------
    def parse_dms(dms_str: str) -> float:
        """Convert a DMS coordinate string into decimal degrees."""

        cleaned = dms_str.replace("°", " ").replace("'", " ").replace('"', " ").strip()
        deg, minu, sec, hemi = cleaned.split()
        decimal_degrees = float(deg) + float(minu) / 60.0 + float(sec) / 3600.0
        return -decimal_degrees if hemi in ("S", "W") else decimal_degrees

    mp_coords: Dict[str, Tuple[float, float]] = {
        "MP1": (parse_dms('52°27\'13"N'), parse_dms('9°45\'37"E')),
        "MP2": (parse_dms('52°27\'57"N'), parse_dms('9°45\'02"E')),
        "MP3": (parse_dms('52°27\'60"N'), parse_dms('9°47\'25"E')),
        "MP4": (parse_dms('52°27\'19"N'), parse_dms('9°48\'48"E')),
        "MP5": (parse_dms('52°28\'05"N'), parse_dms('9°49\'57"E')),
        "MP6": (parse_dms('52°27\'14"N'), parse_dms('9°37\'31"E')),
        "MP7": (parse_dms('52°27\'40"N'), parse_dms('9°34\'55"E')),
        "MP8": (parse_dms('52°28\'07"N'), parse_dms('9°32\'48"E')),
        "MP9": (parse_dms('52°28\'06"N'), parse_dms('9°37\'09"E')),
    }

    # Reference point at the HAJ aerodrome centre used for 5 km filtering.
    airport_center: Tuple[float, float] = (
        parse_dms('52°27\'36.77"N'),
        parse_dms('9°41\'00.68"E'),
    )
    transformer = Transformer.from_crs("epsg:4326", "epsg:32632", always_xy=True)
    ax, ay = transformer.transform(airport_center[1], airport_center[0])

    def distance_to_airport(lat: pd.Series, lon: pd.Series) -> pd.Series:
        """Return Euclidean distance (metres) from series of coordinates to the airport centre."""

        x, y = transformer.transform(lon.to_numpy(), lat.to_numpy())
        dx = x - ax
        dy = y - ay
        return np.sqrt(dx * dx + dy * dy)

    def get_bbox(lat0: float, lon0: float, dist_m: float, buf_frac: float) -> Tuple[float, float, float, float]:
        """Return latitude/longitude bounds for a circular search region."""

        distance_metres = 1000.0 if pd.isna(dist_m) or dist_m <= 0 else dist_m
        radius = distance_metres * (1.0 + buf_frac)
        deg_lat = radius / 111_195.0
        deg_lon = deg_lat / np.cos(np.deg2rad(lat0))
        return lat0 - deg_lat, lat0 + deg_lat, lon0 - deg_lon, lon0 + deg_lon

    # ------------------------------------------
    # Row-wise search + extract trajectory window
    # ------------------------------------------
    tol = pd.Timedelta(seconds=tol_sec)
    win = pd.Timedelta(minutes=window_min)
    traj_slices: list[pd.DataFrame] = []
    matches_found: int = 0  # track filled rows to support early exit in test mode

    for i, row in df_noise.iterrows():
        t0 = row.get("t_ref", pd.NaT)
        if pd.isna(t0):
            continue
        mp = str(row.get("MP", "")).upper().replace(" ", "")
        lat0, lon0 = mp_coords.get(mp, (np.nan, np.nan))
        if pd.isna(lat0) or pd.isna(lon0):
            continue

        lat_min, lat_max, lon_min, lon_max = get_bbox(
            lat0, lon0, float(row.get("Abstand [m]", np.nan)), buffer_frac
        )

        # First time window, then spatial filter
        m_time = (df_ads["timestamp"] >= (t0 - tol)) & (df_ads["timestamp"] <= (t0 + tol))
        cand = df_ads.loc[m_time]
        if cand.empty:
            logger.debug("No ADS-B records within ±%ss for MP %s", tol_sec, mp)
            continue

        m_space = (
            (cand["latitude"] >= lat_min) & (cand["latitude"] <= lat_max) &
            (cand["longitude"] >= lon_min) & (cand["longitude"] <= lon_max)
        )
        hits = cand.loc[m_space]
        if hits.empty:
            logger.debug("No spatial match for MP %s at %s", mp, t0)
            continue

        # Choose the closest-in-time hit
        best_idx = (hits["timestamp"] - t0).abs().idxmin()
        icao24 = hits.at[best_idx, "icao24"]
        callsign = hits.at[best_idx, "callsign"]

        df_noise.at[i, "icao24"] = icao24
        df_noise.at[i, "callsign"] = callsign

        icao24_clean = str(icao24).strip().lower() if pd.notna(icao24) else ""
        callsign_clean = str(callsign).strip().upper() if pd.notna(callsign) else ""
        aircraft_type_adsb: Optional[str] = None

        if aircraft_db is not None:
            ac = None
            if icao24_clean:
                ac = aircraft_db.get(icao24_clean)
            if ac is None and callsign_clean:
                ac = aircraft_db.get(callsign_clean)

            if ac is not None:
                aircraft_type_adsb = getattr(ac, "typecode", None) or getattr(ac, "model", None)

        if aircraft_type_adsb:
            df_noise.at[i, "aircraft_type_adsb"] = str(aircraft_type_adsb)

        noise_type = df_noise.at[i, "aircraft_type_noise"]
        if pd.notna(noise_type) and aircraft_type_adsb:
            df_noise.at[i, "aircraft_type_match"] = (
                str(noise_type).strip().upper() == str(aircraft_type_adsb).strip().upper()
            )
        matches_found += 1

        # Extract ±window slice for that aircraft/callsign
        m2 = (
            (df_ads["icao24"] == icao24) &
            (df_ads["callsign"] == callsign) &
            (df_ads["timestamp"] >= (t0 - win)) &
            (df_ads["timestamp"] <= (t0 + win))
        )
        sl = df_ads.loc[m2].copy()
        if sl.empty:
            logger.debug("No trajectory slice for ICAO24=%s, callsign=%s", icao24, callsign)
            continue

        if not sl.empty and {"latitude", "longitude"}.issubset(sl.columns):
            sl["dist_to_airport_m"] = distance_to_airport(sl["latitude"], sl["longitude"])
            sl = sl[sl["dist_to_airport_m"] <= 5000.0].copy()

        if sl.empty:
            continue

        sl["MP"] = mp
        sl["t_ref"] = t0
        sl["aircraft_type_noise"] = df_noise.at[i, "aircraft_type_noise"]
        sl["aircraft_type_adsb"] = df_noise.at[i, "aircraft_type_adsb"]
        sl["aircraft_type_match"] = df_noise.at[i, "aircraft_type_match"]
        traj_slices.append(sl)

        if test_mode and matches_found >= TEST_MODE_MATCH_LIMIT:
            logger.info(
                "Test mode active: stopping after %d matches to emit early output",
                matches_found,
            )
            break

    df_noise["aircraft_type_adsb"] = df_noise["aircraft_type_adsb"].astype("string")
    df_noise["aircraft_type_match"] = df_noise["aircraft_type_match"].astype("boolean")

    df_traj = pd.concat(traj_slices, ignore_index=True) if traj_slices else pd.DataFrame()

    if dedupe_traj and not df_traj.empty:
        # Remove overlapping ADS-B samples that share the same aircraft/timestamp pair.
        key_cols = [c for c in ["icao24", "timestamp"] if c in df_traj.columns]
        if key_cols:
            df_traj = (
                df_traj.sort_values(key_cols)
                .drop_duplicates(subset=key_cols, keep="first")
            )
        df_traj = df_traj.reset_index(drop=True)

    if not df_traj.empty:
        # Guarantee downstream consumers see the canonical column set even when
        # the underlying ADS-B export lacks certain optional signals.
        desired_cols = [
            "timestamp",
            "latitude",
            "longitude",
            "altitude",
            "geoaltitude",
            "baro_altitude",
            "groundspeed",
            "vertical_rate",
            "track",
            "icao24",
            "callsign",
            "MP",
            "t_ref",
            "aircraft_type_noise",
            "aircraft_type_adsb",
            "aircraft_type_match",
            "dist_to_airport_m",
        ]
        for col in desired_cols:
            if col not in df_traj.columns:
                df_traj[col] = pd.NA

    # -----------------
    # Finalize / save
    # -----------------
    if out_traj_parquet and not df_traj.empty:
        Path(out_traj_parquet).parent.mkdir(parents=True, exist_ok=True)
        df_traj.to_parquet(out_traj_parquet, index=False)
        logger.info("Saved trajectory slices to %s", out_traj_parquet)
    elif out_traj_parquet:
        logger.info("No trajectory slices produced; skipping parquet export to %s", out_traj_parquet)

    return df_noise, df_traj


def main() -> None:
    """Parse CLI arguments, configure logging, and run the matching workflow."""

    parser = argparse.ArgumentParser(description="Match noise measurements against ADS-B trajectories.")
    parser.add_argument("--noise-xlsx", default="noise_data.xlsx", help="Path to the Excel workbook containing noise measurements.")
    parser.add_argument("--adsb-joblib", default="adsb/data_2022_april.joblib", help="Path to the joblib file containing ADS-B data.")
    parser.add_argument("--out-traj-parquet", default="matched_trajs.parquet", help="Output parquet file with matched trajectory slices.")
    parser.add_argument("--tol-sec", type=int, default=10, help="Temporal tolerance in seconds for matching.")
    parser.add_argument("--buffer-frac", type=float, default=1.5, help="Spatial buffer fraction applied to the distance.")
    parser.add_argument("--window-min", type=int, default=3, help="Half-width of the trajectory extraction window in minutes.")
    parser.add_argument("--log-file", default=None, help="Optional path to the log file. Defaults to logs/python/<timestamp>.log")
    parser.add_argument(
        "--no-dedupe-traj",
        action="store_false",
        dest="dedupe_traj",
        help="Disable trajectory deduplication (not recommended for clustering).",
    )
    parser.set_defaults(dedupe_traj=True)
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help=(
            "Enable early-exit test mode that writes outputs once the first "
            f"{TEST_MODE_MATCH_LIMIT} matches are available."
        ),
    )

    args = parser.parse_args()

    log_path = setup_logging(args.log_file)
    logger.info("Writing detailed execution log to %s", log_path)

    df_noise, df_traj = match_noise_to_adsb(
        noise_xlsx=args.noise_xlsx,
        adsb_joblib=args.adsb_joblib,
        out_traj_parquet=args.out_traj_parquet,
        tol_sec=args.tol_sec,
        buffer_frac=args.buffer_frac,
        window_min=args.window_min,
        test_mode=args.test_mode,
        dedupe_traj=args.dedupe_traj if hasattr(args, "dedupe_traj") else True,
    )

    matched_count = int(df_noise["icao24"].notna().sum())
    logger.info("Matching completed successfully.")
    logger.info("Matched noise events: %d / %d", matched_count, len(df_noise))
    logger.info("Trajectory slice rows: %d", len(df_traj))
    if args.out_traj_parquet:
        logger.info("Trajectory parquet target: %s", args.out_traj_parquet)


if __name__ == "__main__":
    main()
