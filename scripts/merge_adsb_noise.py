from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import pytz
from pyproj import Transformer
from pytz.exceptions import AmbiguousTimeError

try:
    from traffic.data import aircraft as aircraft_db
except ImportError:  # pragma: no cover - optional dependency
    aircraft_db = None


# Configure a module-level logger that can be reused by the helper functions.
logger: logging.Logger = logging.getLogger(__name__)


def setup_logging(
    log_file: Optional[Union[str, Path]] = None,
    console_level: int = logging.INFO,
) -> Path:
    """Set up application-wide logging and return the selected log file path.

    Parameters
    ----------
    log_file:
        Optional explicit location for the log file. When ``None`` the helper
        creates ``logs/python`` and generates a UTC timestamped file name.
    console_level:
        Logging level applied to the stream handler that mirrors key messages to
        stdout (defaults to :data:`logging.INFO`).

    Returns
    -------
    Path
        The fully qualified path to the log file capturing detailed execution
        information.
    """

    # Determine the log destination while creating the directory structure as needed.
    if log_file is not None:
        log_path = Path(log_file)
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        log_dir = Path("logs") / "python"
        log_path = log_dir / f"{timestamp}.log"

    log_path.parent.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # Remove any leftover handlers from previous runs so we do not duplicate output.
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
    console_handler.setLevel(console_level)
    console_handler.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))
    root_logger.addHandler(console_handler)

    root_logger.debug("Initialised logging. Writing details to %s", log_path)
    return log_path


def _detect_header_row(
    rows: pd.DataFrame, expected_columns: Iterable[str]
) -> Optional[int]:
    """Return the row index that most closely matches the expected column names.

    Notes
    -----
    The helper inspects each row until the expected columns are fully present.
    Logging is used to highlight which sheet rows were evaluated.
    """

    # Normalise expected column names to a comparable representation.
    expected = {col.strip().lower() for col in expected_columns}
    for idx, row in rows.iterrows():
        # Convert the row values to string for a robust comparison.
        row_values = {str(value).strip().lower() for value in row.tolist() if pd.notna(value)}
        if expected.issubset(row_values):
            logger.debug("Detected header row %s with expected columns", idx)
            return idx
    logger.debug("Failed to detect header row containing %s", expected)
    return None


def _load_noise_excel(noise_excel: Union[str, Path]) -> pd.DataFrame:
    """Load and tidy noise measurements from a workbook with multiple sheets.

    Parameters
    ----------
    noise_excel:
        Path to the formatted Excel workbook that may contain several measurement
        point sheets (e.g., ``MP1`` through ``MP9``) and possibly decorative pages.

    Returns
    -------
    pd.DataFrame
        A concatenated DataFrame containing the cleaned noise records from every
        recognised measurement-point sheet.
    """

    path = Path(noise_excel)

    logger.info("Loading noise workbook from %s", path)

    # Read every sheet without headers so we can locate the proper column row per sheet.
    raw_sheets: Dict[str, pd.DataFrame] = pd.read_excel(path, sheet_name=None, header=None)

    expected_columns: List[str] = [
        "MP",
        "TLASmax",
        "Abstand [m]",
    ]

    cleaned_frames: List[pd.DataFrame] = []

    for sheet_name, sheet_raw in raw_sheets.items():
        # Skip sheets that do not correspond to measurement points (e.g., summary pages).
        if not sheet_name.upper().startswith("MP"):
            logger.debug("Skipping non-measurement sheet %s", sheet_name)
            continue

        header_row = _detect_header_row(sheet_raw, expected_columns)
        if header_row is None:
            # No valid table was found on this sheet, so skip quietly.
            logger.warning("Sheet %s does not contain a recognised header row", sheet_name)
            continue

        df_sheet = pd.read_excel(path, sheet_name=sheet_name, header=header_row)

        # Drop fully empty rows that frequently appear in formatted Excel sheets.
        df_sheet = df_sheet.dropna(how="all").reset_index(drop=True)

        # Ensure the measurement-point column exists even if the sheet omits it explicitly.
        if "MP" not in df_sheet.columns:
            df_sheet["MP"] = sheet_name

        # Remove rows lacking essential identifiers while preserving measurement data.
        if "MP" in df_sheet.columns:
            df_sheet = df_sheet[df_sheet["MP"].notna()].copy()

        if not df_sheet.empty:
            logger.debug("Loaded %d rows from sheet %s", len(df_sheet), sheet_name)
            cleaned_frames.append(df_sheet)

    if not cleaned_frames:
        raise ValueError(
            "Unable to locate valid measurement data in any MP sheet of the workbook."
        )

    # Concatenate all measurement sheets into a single table for downstream processing.
    df_noise = pd.concat(cleaned_frames, ignore_index=True)

    logger.info("Loaded %d total noise rows", len(df_noise))

    return df_noise


def _localize_berlin(series: pd.Series) -> pd.Series:
    """Convert naive timestamps to Europe/Berlin while resolving DST transitions.

    Parameters
    ----------
    series:
        pandas Series containing timezone-naive ``Timestamp`` objects that are
        expressed in local Berlin time.

    Returns
    -------
    pd.Series
        A timezone-aware Series in the ``Europe/Berlin`` timezone where
        ambiguous timestamps (e.g., the clock rollback hour) are coerced to the
        standard-time occurrence so that downstream UTC conversion succeeds.
    """

    berlin_tz = pytz.timezone("Europe/Berlin")

    try:
        # Fast path where pandas can infer DST transitions automatically.
        return series.dt.tz_localize(
            berlin_tz, ambiguous="infer", nonexistent="shift_forward"
        )
    except AmbiguousTimeError:
        # Fall back to a manual pass if ambiguous timestamps occur (e.g. DST end).
        logger.warning("Encountered DST ambiguity; applying manual localisation fallback")
        localized = series.dt.tz_localize(
            berlin_tz, ambiguous="NaT", nonexistent="shift_forward"
        )
        ambiguous_mask = localized.isna() & series.notna()

        if ambiguous_mask.any():
            # For repeated times, assume the later (standard-time) occurrence.
            logger.debug(
                "Resolving %d ambiguous timestamps by assuming standard time",
                ambiguous_mask.sum(),
            )
            localized.loc[ambiguous_mask] = series.loc[ambiguous_mask].apply(
                lambda value: pd.Timestamp(
                    berlin_tz.localize(value.to_pydatetime(), is_dst=False)
                )
            )

        return localized


def _load_adsb_joblib(adsb_joblib: Union[str, Path]) -> pd.DataFrame:
    """Load ADS-B data from a Joblib file and normalise it to a DataFrame.

    Parameters
    ----------
    adsb_joblib:
        Path to the Joblib artefact. The payload may originate from the
        ``traffic`` library (e.g., :class:`traffic.core.Traffic` or
        :class:`traffic.core.Flight`) or be stored as a raw pandas structure.

    Returns
    -------
    pd.DataFrame
        A copy of the ADS-B samples with a mandatory ``timestamp`` column in
        UTC. Additional columns are carried over untouched for downstream
        matching.

    Raises
    ------
    TypeError
        If the Joblib payload cannot be converted into a pandas DataFrame.
    KeyError
        If the resulting DataFrame lacks the required ``timestamp`` column.
    """

    path = Path(adsb_joblib)
    logger.info("Loading ADS-B joblib payload from %s", path)

    adsb_obj: Any = joblib.load(path)
    logger.debug("ADS-B joblib payload type: %s", type(adsb_obj))

    df_ads: Optional[pd.DataFrame] = None

    if isinstance(adsb_obj, pd.DataFrame):
        # Direct pandas serialisation can be used as-is.
        df_ads = adsb_obj.copy()
    elif hasattr(adsb_obj, "data") and isinstance(getattr(adsb_obj, "data"), pd.DataFrame):
        # Traffic library objects (Traffic/Flight) expose their samples via ``.data``.
        df_ads = getattr(adsb_obj, "data").copy()
        logger.info("Extracted ADS-B records from traffic object via .data attribute")
    elif hasattr(adsb_obj, "to_dataframe"):
        # Some helpers (e.g., LazyTraffic) provide a conversion method.
        candidate = adsb_obj.to_dataframe()  # type: ignore[call-arg]
        if isinstance(candidate, pd.DataFrame):
            df_ads = candidate.copy()
            logger.info("Extracted ADS-B records via to_dataframe() helper")
    elif isinstance(adsb_obj, dict):
        df_ads = pd.DataFrame.from_dict(adsb_obj)
    elif isinstance(adsb_obj, (list, tuple)):
        df_ads = pd.DataFrame(adsb_obj)
    elif isinstance(adsb_obj, Iterable) and not isinstance(adsb_obj, (str, bytes)):
        # Fallback for generic iterables of records.
        df_ads = pd.DataFrame(list(adsb_obj))

    if df_ads is None:
        raise TypeError(
            "Unsupported ADS-B Joblib payload. Expected a pandas DataFrame, a "
            "traffic.core object exposing a .data DataFrame, or an iterable of records."
        )

    if "timestamp" not in df_ads.columns:
        raise KeyError("ADS-B dataset does not contain the required 'timestamp' column.")

    # Ensure timestamps are timezone-aware UTC for matching operations.
    df_ads["timestamp"] = pd.to_datetime(df_ads["timestamp"], utc=True)

    logger.info("Loaded %d ADS-B records", len(df_ads))

    return df_ads


def _normalise_identifier(value: Any) -> Optional[str]:
    """Convert raw ADS-B identifiers to clean uppercase strings or ``None``.

    The helper copes with mixed input types that frequently appear in the
    Joblib payloads. ``None``-like values and placeholder strings (e.g.,
    ``"nan"``) are treated as missing. Valid identifiers are stripped of
    surrounding whitespace and uppercased to provide a consistent format for
    subsequent CSV/Parquet serialisation.

    Parameters
    ----------
    value:
        Arbitrary identifier pulled from the ADS-B frame. This may be a float,
        integer, string, or even a pandas scalar object.

    Returns
    -------
    Optional[str]
        ``None`` when the identifier should be treated as missing, otherwise
        the cleaned string representation.
    """

    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None

    # Convert pandas scalars (e.g., pd.NA) to native Python values first.
    try:
        if pd.isna(value):
            return None
    except TypeError:
        # ``pd.isna`` may raise when handed non-scalar objects; treat them as valid
        # and fall through to the generic string conversion.
        pass

    normalised: str = str(value).strip().upper()

    if normalised in {"", "NA", "NAN", "NONE"}:
        return None

    return normalised


def _downsample_by_interval(
    df: pd.DataFrame,
    time_column: str,
    interval_seconds: int,
) -> pd.DataFrame:
    """Return the DataFrame limited to one sample per ``interval_seconds`` bin.

    The helper preserves the first observation in each interval, providing a
    simple knob to trade temporal resolution for reduced output size.
    """

    if interval_seconds <= 0 or df.empty or time_column not in df.columns:
        return df

    df_sorted = df.sort_values(time_column)
    binned = df_sorted[time_column].dt.floor(f"{interval_seconds}S")
    keep_mask = ~binned.duplicated()
    return df_sorted.loc[keep_mask].copy()


def parse_bool(value: Union[str, bool, int]) -> bool:
    """Return a Python ``bool`` from flexible user-provided representations.

    Parameters
    ----------
    value:
        Input value that should be interpreted as ``True`` or ``False``. Strings
        such as ``"true"``, ``"1"``, and ``"yes"`` are recognised in a
        case-insensitive manner. Integers behave like Python truthiness rules.

    Returns
    -------
    bool
        The normalised boolean result suitable for configuring CLI flags or
        environment-driven toggles.
    """

    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value != 0
    normalized = str(value).strip().lower()
    truthy = {"1", "true", "t", "yes", "y", "on"}
    falsy = {"0", "false", "f", "no", "n", "off", ""}
    if normalized in truthy:
        return True
    if normalized in falsy:
        return False
    raise ValueError(f"Cannot interpret value '{value}' as boolean")


def match_noise_to_adsb(
    df_noise: Union[str, Path],
    adsb_joblib: Union[str, Path],
    out_traj_parquet: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    tol_sec: int = 10,
    buffer_frac: float = 0.5,
    window_min: int = 3,
    sample_interval_sec: int = 2,
    max_airport_distance_m: float = 25_000.0,
    dedupe_traj: bool = True,
    test_mode: bool = False,
    test_mode_match_limit: int = 5,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Match noise measurements from Excel to ADS-B trajectories.

    Parameters
    ----------
    df_noise:
        Path to the Excel file containing noise events. The file may include empty
        or decorative rows, which will be stripped automatically.
    adsb_joblib:
        Path to the Joblib file containing ADS-B samples. The file must provide a
        serialised pandas DataFrame or a structure convertible to one.
    out_traj_parquet:
        Optional file name (with optional subdirectories) that determines how the
        extracted ADS-B trajectory snippets are written. By default outputs are
        stored under ``data/merged`` using the same relative subdirectory and base name.
    output_dir:
        Optional directory to write both parquet and CSV outputs. When provided,
        it overrides the ``data/merged`` layout.
    tol_sec:
        Time tolerance in seconds used when searching for an ADS-B hit around the
        noise measurement timestamp.
    buffer_frac:
        Fractional buffer applied to the lateral distance filter. This widens the
        spatial bounding box by ``1 + buffer_frac``.
    window_min:
        Size of the time window in minutes to extract the matched trajectory
        segment around the reference timestamp.
    sample_interval_sec:
        Minimum spacing between retained ADS-B samples (in seconds) inside the
        extracted trajectory slices.
    max_airport_distance_m:
        Maximum distance (in metres) from the airport centre to retain ADS-B
        samples in each extracted trajectory slice.
    dedupe_traj:
        When ``True`` (default) overlapping ADS-B samples per measurement
        context (``MP``, ``t_ref``, ``icao24``, ``timestamp``) are deduplicated to
        keep clustering inputs tidy without discarding distinct microphone views.

    test_mode:
        When ``True`` the loop stops after ``test_mode_match_limit`` matches and
        immediately writes output files so the workflow can be smoke-tested
        quickly. Set to ``False`` (default) for full-batch processing.
    test_mode_match_limit:
        Maximum number of matched noise rows to process before exiting in test
        mode. Ignored when ``test_mode`` is disabled.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        A tuple containing the enriched noise DataFrame and the concatenated
        trajectory slices limited to ``max_airport_distance_m`` around the
        airport center.
    """

    logger.info("Starting noise to ADS-B matching workflow")

    # 1) Load and tidy the noise measurements from Excel.
    df_noise = _load_noise_excel(df_noise)
    logger.debug("Noise dataframe columns: %s", df_noise.columns.tolist())

    if "A/D" not in df_noise.columns:
        df_noise["A/D"] = pd.NA
    if "Runway" not in df_noise.columns:
        df_noise["Runway"] = pd.NA

    if "Flugzeugtyp" in df_noise.columns:
        # Preserve the aircraft type specified in the Excel workbook for later validation.
        df_noise["aircraft_type_noise"] = df_noise["Flugzeugtyp"].astype("string")
    else:
        df_noise["aircraft_type_noise"] = pd.NA

    if "TLASmax_UTC" in df_noise.columns:
        # The Excel sheet already provides UTC timestamps.
        df_noise["t_ref"] = pd.to_datetime(df_noise["TLASmax_UTC"], utc=True, errors="coerce")
        logger.info("Detected pre-computed UTC timestamps in TLASmax_UTC column")
    else:
        # Fallback: convert the local TLASmax timestamps (Berlin time) to UTC.
        df_noise["TLASmax"] = pd.to_datetime(
            df_noise["TLASmax"], dayfirst=True, errors="coerce"
        )
        localized = _localize_berlin(df_noise["TLASmax"])
        df_noise["t_ref"] = localized.dt.tz_convert(pytz.UTC)
        logger.info("Converted TLASmax values from local Berlin time to UTC")

    # Drop rows where the timestamp could not be parsed.
    df_noise = df_noise[df_noise["t_ref"].notna()].copy()
    logger.info("Retained %d noise rows after timestamp parsing", len(df_noise))

    if "Abstand [m]" not in df_noise.columns:
        raise KeyError("Noise dataset does not contain the required 'Abstand [m]' column.")

    if "MP" not in df_noise.columns:
        raise KeyError("Noise dataset does not contain the required 'MP' column.")

    # Ensure the distance column can be used numerically even if commas are used.
    df_noise["Abstand [m]"] = pd.to_numeric(
        df_noise["Abstand [m]"].astype(str).str.replace(",", ".", regex=False), errors="coerce"
    )
    df_noise = df_noise[df_noise["Abstand [m]"].notna()].copy()

    df_noise["MP"] = (
        df_noise["MP"].astype(str).str.strip().str.upper().str.replace(" ", "")
    )

    # Map potential short MP identifiers (e.g., MP1) to the dictionary keys (M01).
    df_noise["MP"] = df_noise["MP"].str.replace("MP", "M", regex=False)
    df_noise["MP"] = df_noise["MP"].str.replace(r"^(M)(\d)$", r"\g<1>0\g<2>", regex=True)

    # Prepare output columns that will hold the matched flight identifiers.
    df_noise["icao24"] = pd.NA
    df_noise["callsign"] = pd.NA
    df_noise["aircraft_type_adsb"] = pd.NA
    df_noise["aircraft_type_match"] = pd.NA

    # 2) Load ADS-B data from Joblib and ensure timestamps are in UTC.
    df_ads = _load_adsb_joblib(adsb_joblib)
    logger.debug("ADS-B dataframe columns: %s", df_ads.columns.tolist())

    # 3) MP coordinates and bounding-box helper utilities.
    def parse_dms(dms_str: str) -> float:
        """Convert a DMS coordinate string (e.g. ``52°27'13"N``) into decimal degrees."""

        # Replace compass markers with space separators and split into components.
        sanitized = (
            dms_str.replace("°", " ").replace("'", " ").replace("\"", " ").strip()
        )
        deg_str, min_str, sec_str, hemisphere = sanitized.split()
        degrees = float(deg_str) + float(min_str) / 60 + float(sec_str) / 3600
        return -degrees if hemisphere in ("S", "W") else degrees

    mp_coords: Dict[str, Tuple[float, float]] = {
        "M01": (parse_dms("52°27'13\"N"), parse_dms("9°45'37\"E")),
        "M02": (parse_dms("52°27'57\"N"), parse_dms("9°45'02\"E")),
        "M03": (parse_dms("52°27'60\"N"), parse_dms("9°47'25\"E")),
        "M04": (parse_dms("52°27'19\"N"), parse_dms("9°48'48\"E")),
        "M05": (parse_dms("52°28'05\"N"), parse_dms("9°49'57\"E")),
        "M06": (parse_dms("52°27'14\"N"), parse_dms("9°37'31\"E")),
        "M07": (parse_dms("52°27'40\"N"), parse_dms("9°34'55\"E")),
        "M08": (parse_dms("52°28'07\"N"), parse_dms("9°32'48\"E")),
        "M09": (parse_dms("52°28'06\"N"), parse_dms("9°37'09\"E")),
    }

    airport_center: Tuple[float, float] = (
        parse_dms("52°27'36.77\"N"),
        parse_dms("9°41'00.68\"E"),
    )
    transformer: Transformer = Transformer.from_crs(
        "epsg:4326", "epsg:32632", always_xy=True
    )
    ax, ay = transformer.transform(airport_center[1], airport_center[0])

    def distance_to_airport(lat: pd.Series, lon: pd.Series) -> pd.Series:
        """Return Euclidean distance in metres from the airport center for each point."""

        x, y = transformer.transform(lon.to_numpy(), lat.to_numpy())
        dx = x - ax
        dy = y - ay
        return np.sqrt(dx * dx + dy * dy)

    def get_bbox(
        lat0: float,
        lon0: float,
        dist_m: float,
        buf_frac: float,
    ) -> Tuple[float, float, float, float]:
        """Return a latitude/longitude bounding box centred at the microphone."""

        radius = dist_m * (1 + buf_frac)
        deg_lat = radius / 111_195.0
        deg_lon = deg_lat / np.cos(np.deg2rad(lat0))
        return lat0 - deg_lat, lat0 + deg_lat, lon0 - deg_lon, lon0 + deg_lon

    # 4) Loop over noise rows, find ±tol_sec match, extract ±window_min trajectory
    tol = pd.Timedelta(seconds=tol_sec)
    win = pd.Timedelta(minutes=window_min)
    trajs: List[pd.DataFrame] = []

    # Track successful matches so the optional test-mode limit can terminate early.
    matches_found: int = 0

    for i, row in df_noise.iterrows():
        # Iterate over every noise measurement, attempting to identify a matching flight.
        t0 = row["t_ref"]
        mp = row["MP"]
        dist = row["Abstand [m]"]
        lat0, lon0 = mp_coords.get(mp, (None, None))
        if lat0 is None:
            logger.warning("Measurement point %s missing from coordinate map", mp)
            continue

        # spatial + time mask to identify flight
        lat_min, lat_max, lon_min, lon_max = get_bbox(lat0, lon0, dist, buffer_frac)
        m1 = (
          (df_ads.timestamp >=  t0 - tol) &
          (df_ads.timestamp <=  t0 + tol) &
          (df_ads.latitude  >= lat_min) &
          (df_ads.latitude  <= lat_max) &
          (df_ads.longitude >= lon_min) &
          (df_ads.longitude <= lon_max)
        )
        hits = df_ads.loc[m1]
        if hits.empty:
            logger.debug("No ADS-B hits found for MP %s at %s", mp, t0)
            continue

        # Stamp the noise row with the first matching aircraft identifiers.
        raw_icao24, raw_callsign = hits.iloc[0][["icao24", "callsign"]]
        icao24_clean = _normalise_identifier(raw_icao24)
        callsign_clean = _normalise_identifier(raw_callsign)

        if icao24_clean is None and callsign_clean is None:
            logger.debug(
                "Skipping assignment for MP %s at %s due to missing identifiers",
                mp,
                t0,
            )
            continue

        df_noise.at[i, "icao24"] = icao24_clean if icao24_clean is not None else pd.NA
        df_noise.at[i, "callsign"] = (
            callsign_clean if callsign_clean is not None else pd.NA
        )

        aircraft_type_adsb: Optional[str] = None
        if aircraft_db is not None:
            # Probe the optional aircraft DB via ICAO24 first and fall back to callsign.
            ac = None
            if icao24_clean:
                ac = aircraft_db.get(icao24_clean)
            if ac is None and callsign_clean:
                ac = aircraft_db.get(callsign_clean)
            if ac is not None:
                aircraft_type_adsb = getattr(ac, "typecode", None) or getattr(
                    ac, "model", None
                )

        if aircraft_type_adsb:
            df_noise.at[i, "aircraft_type_adsb"] = str(aircraft_type_adsb)

        noise_type = df_noise.at[i, "aircraft_type_noise"]
        if pd.notna(noise_type) and aircraft_type_adsb:
            df_noise.at[i, "aircraft_type_match"] = (
                str(noise_type).strip().upper()
                == str(aircraft_type_adsb).strip().upper()
            )

        matches_found += 1
        logger.info(
            "Matched MP %s at %s to aircraft %s (%s)",
            mp,
            t0,
            callsign_clean,
            icao24_clean,
        )

        # extract the full ±window slice
        m2 = (
          (df_ads.timestamp >= t0 - win) &
          (df_ads.timestamp <= t0 + win)
        )
        if icao24_clean is not None:
            m2 &= df_ads.icao24 == raw_icao24
        if callsign_clean is not None:
            m2 &= df_ads.callsign == raw_callsign
        slice6 = df_ads.loc[m2].copy()
        slice6["MP"] = mp
        slice6["t_ref"] = t0
        slice6["icao24"] = icao24_clean if icao24_clean is not None else pd.NA
        slice6["callsign"] = (
            callsign_clean if callsign_clean is not None else pd.NA
        )
        slice6["aircraft_type_noise"] = df_noise.at[i, "aircraft_type_noise"]
        slice6["aircraft_type_adsb"] = df_noise.at[i, "aircraft_type_adsb"]
        slice6["aircraft_type_match"] = df_noise.at[i, "aircraft_type_match"]
        slice6["A/D"] = df_noise.at[i, "A/D"] if "A/D" in df_noise.columns else pd.NA
        slice6["Runway"] = df_noise.at[i, "Runway"] if "Runway" in df_noise.columns else pd.NA

        if not slice6.empty and {"latitude", "longitude"}.issubset(slice6.columns):
            slice6["dist_to_airport_m"] = distance_to_airport(
                slice6["latitude"], slice6["longitude"]
            )
            slice6 = slice6[slice6["dist_to_airport_m"] <= float(max_airport_distance_m)].copy()

        if not slice6.empty:
            slice6 = _downsample_by_interval(slice6, "timestamp", sample_interval_sec)

        if slice6.empty:
            continue

        if "dist_to_airport_m" not in slice6.columns:
            # Guarantee the downstream parquet schema even if the ADS-B slice lacked lat/lon.
            slice6["dist_to_airport_m"] = np.nan
        trajs.append(slice6)

        if test_mode and matches_found >= test_mode_match_limit:
            # Exit the processing loop once the desired preview count is reached.
            logger.info(
                "Test mode enabled: collected %d matches; stopping early.",
                matches_found,
            )
            break

    # 5) finalize
    for column_name in (
        "icao24",
        "callsign",
        "aircraft_type_noise",
        "aircraft_type_adsb",
        "A/D",
        "Runway",
    ):
        if column_name in df_noise.columns:
            # Enforce pandas' native string dtype so parquet serialises mixed values reliably.
            df_noise[column_name] = df_noise[column_name].astype("string")
    if "aircraft_type_match" in df_noise.columns:
        df_noise["aircraft_type_match"] = df_noise["aircraft_type_match"].astype(
            "boolean"
        )

    df_traj = pd.concat(trajs, ignore_index=True) if trajs else pd.DataFrame()
    if not df_traj.empty:
        if dedupe_traj:
            # Include the measurement context (MP/t_ref) in the dedupe key so distinct
            # microphones observing the same aircraft at the same instant retain their
            # individual slices while still removing true duplicates within a slice.
            key_cols = [
                c
                for c in ["MP", "t_ref", "icao24", "timestamp"]
                if c in df_traj.columns
            ]
            if key_cols:
                df_traj = (
                    df_traj.sort_values(key_cols).drop_duplicates(subset=key_cols, keep="first")
                )
        for column_name in (
            "icao24",
            "callsign",
            "aircraft_type_noise",
            "aircraft_type_adsb",
            "A/D",
            "Runway",
        ):
            if column_name in df_traj.columns:
                df_traj[column_name] = df_traj[column_name].astype("string")
        if "aircraft_type_match" in df_traj.columns:
            df_traj["aircraft_type_match"] = df_traj["aircraft_type_match"].astype(
                "boolean"
            )
        required_columns: List[str] = [
            "timestamp",
            "latitude",
            "longitude",
            "altitude",
            "geoaltitude",
            "groundspeed",
            "vertical_rate",
            "track",
            "icao24",
            "callsign",
            "MP",
            "t_ref",
            "A/D",
            "Runway",
            "aircraft_type_noise",
            "aircraft_type_adsb",
            "aircraft_type_match",
            "dist_to_airport_m",
        ]
        for column_name in required_columns:
            if column_name not in df_traj.columns:
                df_traj[column_name] = pd.NA
        df_traj = df_traj[required_columns + [col for col in df_traj.columns if col not in required_columns]]
    logger.info("Constructed %d trajectory slices", len(trajs))

    matched_rows = int(df_noise["icao24"].notna().sum())
    logger.info("Matched %d noise rows out of %d", matched_rows, len(df_noise))

    if out_traj_parquet and not df_traj.empty:
        requested_path = Path(out_traj_parquet)
        base_stem = requested_path.stem or requested_path.name
        if not base_stem:
            base_stem = "matched_trajectories"

        relative_dir = requested_path.parent
        if str(relative_dir) in (".", ""):
            relative_dir = Path()

        if output_dir is None:
            output_root = Path("data") / "merged"
        else:
            output_root = Path(output_dir)
        parquet_path = output_root / relative_dir / f"{base_stem}.parquet"
        csv_path = output_root / relative_dir / f"{base_stem}.csv"

        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Writing matched trajectory data to %s", parquet_path)
        df_traj.to_parquet(parquet_path, index=False)

        logger.info("Writing matched trajectory data to %s", csv_path)
        df_traj.to_csv(csv_path, index=False)
    elif out_traj_parquet:
        logger.info("No trajectory data to write for %s", out_traj_parquet)

    logger.info("Noise to ADS-B matching completed")

    return df_noise, df_traj


def main() -> None:
    """Parse command-line arguments and execute the ADS-B/noise matching workflow."""

    import argparse

    parser = argparse.ArgumentParser(
        description="Match noise measurements from Excel with ADS-B Joblib data."
    )
    parser.add_argument("noise_excel", type=Path, help="Path to the noise Excel file.")
    parser.add_argument("adsb_joblib", type=Path, help="Path to the ADS-B Joblib file.")
    parser.add_argument(
        "--traj-output",
        type=Path,
        default=None,
        help=(
            "Base file name (with optional subdirectories) for trajectory outputs. "
            "Files are written to data/merged unless --output-dir is set."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/merged"),
        help=(
            "Directory to write both parquet and CSV outputs. "
            "Defaults to data/merged."
        ),
    )
    parser.add_argument(
        "--tol-sec",
        type=int,
        default=10,
        help="Time tolerance in seconds for matching noise to ADS-B records.",
    )
    parser.add_argument(
        "--buffer-frac",
        type=float,
        default=0.5,
        help="Fractional buffer applied to the spatial bounding box.",
    )
    parser.add_argument(
        "--window-min",
        type=int,
        default=3,
        help="Time window in minutes to extract around the reference timestamp.",
    )
    parser.add_argument(
        "--sample-interval-sec",
        type=int,
        default=2,
        help="Retain at most one ADS-B sample every N seconds inside each trajectory slice.",
    )
    parser.add_argument(
        "--max-airport-distance-m",
        type=float,
        default=25_000.0,
        help="Maximum distance in metres from the airport centre to retain in each trajectory slice.",
    )
    parser.add_argument(
        "--max-airport-distance-km",
        type=float,
        default=None,
        help="Maximum distance in kilometres from the airport centre (overrides --max-airport-distance-m).",
    )
    parser.add_argument(
        "--no-dedupe-traj",
        action="store_true",
        help="Disable deduplication of overlapping ADS-B samples in the parquet output.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Console logging level (e.g. DEBUG, INFO, WARNING).",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help=(
            "Optional explicit log file path. Defaults to logs/python/<timestamp>.log"
        ),
    )
    parser.add_argument(
        "--test-mode",
        type=parse_bool,
        nargs="?",
        const=True,
        default=False,
        help=(
            "Enable early exit after a limited number of matches. Accepts true/false "
            "values; omitting the value defaults to true."
        ),
    )
    parser.add_argument(
        "--test-mode-match-limit",
        type=int,
        default=5,
        help="Maximum matches to collect before stopping when test mode is active.",
    )

    args = parser.parse_args()

    console_level = getattr(logging, args.log_level.upper(), logging.INFO)
    log_path = setup_logging(args.log_file, console_level=console_level)
    logger.info("Writing detailed log to %s", log_path)
    logger.info("Console log level set to %s", logging.getLevelName(console_level))

    max_airport_distance_m = float(args.max_airport_distance_m)
    if args.max_airport_distance_km is not None:
        max_airport_distance_m = float(args.max_airport_distance_km) * 1000.0

    match_noise_to_adsb(
        df_noise=args.noise_excel,
        adsb_joblib=args.adsb_joblib,
        out_traj_parquet=args.traj_output,
        output_dir=args.output_dir,
        tol_sec=args.tol_sec,
        buffer_frac=args.buffer_frac,
        window_min=args.window_min,
        sample_interval_sec=args.sample_interval_sec,
        max_airport_distance_m=max_airport_distance_m,
        dedupe_traj=not args.no_dedupe_traj,
        test_mode=args.test_mode,
        test_mode_match_limit=args.test_mode_match_limit,
    )

    logger.info("Merge process finished successfully")


if __name__ == "__main__":
    main()
