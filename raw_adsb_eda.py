"""RAW ADS-B exploratory data analysis (EDA) for monthly Joblib datasets.

This script characterizes surveillance data quality and coverage before any
noise matching, runway assignment, A/D labeling, airport-radius filtering,
trajectory extraction, or flight segmentation.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MONTH_ORDER = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}

MISSING_FIELDS = [
    "callsign",
    "origin",
    "destination",
    "alt_or_geoalt",
    "gs",
    "track",
    "vr",
    "onground",
    "lat",
    "lon",
]

MAX_SPATIAL_POINTS_PER_MONTH = 300_000
MAX_SPATIAL_POINTS_OVERALL = 2_500_000
MAX_DELTA_POINTS_PER_MONTH = 250_000
MAX_ALT_POINTS_PER_MONTH = 250_000
MAX_GS_POINTS_PER_MONTH = 200_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Perform RAW ADS-B EDA across monthly joblib files. "
            "No matching, runway mapping, A/D labeling, or segmentation is applied."
        )
    )
    parser.add_argument(
        "--input_dir",
        default="data/adsb",
        help="Folder containing monthly ADS-B joblib files (default: data/adsb).",
    )
    parser.add_argument(
        "--output_dir",
        default="output/eda",
        help="Base output folder. A timestamped subfolder is created inside it.",
    )
    return parser.parse_args()


def _configure_logging(out_dir: Path) -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("raw_adsb_eda")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(out_dir / "logs.txt", encoding="utf-8")
    file_handler.setFormatter(fmt)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def _month_from_filename(path: Path) -> Optional[str]:
    m = re.search(r"data_2022_([a-zA-Z]+)\.joblib$", path.name)
    if not m:
        return None
    token = m.group(1).strip().lower()
    if token not in MONTH_ORDER:
        return None
    return token


def discover_monthly_files(input_dir: Path) -> List[Tuple[str, Path]]:
    files: List[Tuple[str, Path]] = []
    for p in sorted(input_dir.glob("*.joblib")):
        month = _month_from_filename(p)
        if month is not None:
            files.append((month, p))
    files.sort(key=lambda item: MONTH_ORDER[item[0]])
    return files


def _load_joblib_as_dataframe(path: Path) -> pd.DataFrame:
    obj = joblib.load(path)
    if isinstance(obj, pd.DataFrame):
        return obj
    if hasattr(obj, "data") and isinstance(getattr(obj, "data"), pd.DataFrame):
        return getattr(obj, "data")
    if hasattr(obj, "to_dataframe"):
        candidate = obj.to_dataframe()  # type: ignore[call-arg]
        if isinstance(candidate, pd.DataFrame):
            return candidate
    if isinstance(obj, dict):
        return pd.DataFrame.from_dict(obj)
    if isinstance(obj, (list, tuple)):
        return pd.DataFrame(obj)
    return pd.DataFrame(list(obj))


def _norm_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def _build_column_map(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    norm_to_col = {_norm_name(col): col for col in df.columns}

    def pick(candidates: Iterable[str]) -> Optional[str]:
        for c in candidates:
            n = _norm_name(c)
            if n in norm_to_col:
                return norm_to_col[n]
        return None

    return {
        "timestamp": pick(["timestamp", "time", "datetime", "ts"]),
        "icao24": pick(["icao24", "icao", "icao_24"]),
        "callsign": pick(["callsign", "call_sign"]),
        "lat": pick(["lat", "latitude"]),
        "lon": pick(["lon", "longitude", "long"]),
        "alt": pick(["alt", "altitude"]),
        "geoalt": pick(["geoalt", "geoaltitude", "geo_alt", "geo_altitude"]),
        "gs": pick(["gs", "groundspeed", "ground_speed", "speed"]),
        "track": pick(["track", "heading"]),
        "vr": pick(["vr", "vertical_rate", "verticalspeed", "vert_rate"]),
        "onground": pick(["onground", "on_ground"]),
        "origin": pick(["origin", "departure", "originairport"]),
        "destination": pick(["destination", "dest", "arrival", "destinationairport"]),
    }


def _as_datetime_utc(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce")


def _as_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _normalized_text(series: pd.Series) -> pd.Series:
    s = series.astype("string").str.strip().str.upper()
    return s.mask(s.isin({"", "NA", "NAN", "NONE", "<NA>"}))


def _day_summary(day_counts: pd.Series) -> Dict[str, float]:
    if day_counts.empty:
        return {
            "n_days": 0,
            "rows_per_day_mean": np.nan,
            "rows_per_day_median": np.nan,
            "rows_per_day_p5": np.nan,
            "rows_per_day_p95": np.nan,
        }
    values = day_counts.to_numpy(dtype=float)
    return {
        "n_days": int(len(values)),
        "rows_per_day_mean": float(np.mean(values)),
        "rows_per_day_median": float(np.median(values)),
        "rows_per_day_p5": float(np.percentile(values, 5)),
        "rows_per_day_p95": float(np.percentile(values, 95)),
    }


def _sample_array(values: np.ndarray, max_size: int, rng: np.random.RandomState) -> np.ndarray:
    if values.size <= max_size:
        return values
    idx = rng.choice(values.size, size=max_size, replace=False)
    return values[idx]


def _infer_gs_units(gs_samples: np.ndarray) -> Tuple[str, float, float]:
    if gs_samples.size == 0:
        return "unknown", 350.0, np.nan
    q99 = float(np.percentile(gs_samples, 99))
    if q99 > 350.0:
        return "knots", 700.0, q99
    return "m/s", 350.0, q99


def _infer_alt_units(alt_samples: np.ndarray) -> Tuple[str, float, float]:
    if alt_samples.size == 0:
        return "unknown", 60_000.0, np.nan
    q95 = float(np.percentile(alt_samples, 95))
    if q95 > 20_000.0:
        return "feet", 60_000.0, q95
    return "meters", 18_288.0, q95


def _plot_sampling_hist(delta_values: np.ndarray, out_path: Path) -> None:
    if delta_values.size == 0:
        return
    vals = delta_values[np.isfinite(delta_values) & (delta_values >= 0)]
    if vals.size == 0:
        return
    positives = vals[vals > 0]
    if positives.size == 0:
        return

    vmin = max(float(np.percentile(positives, 1)), 1e-3)
    vmax = max(float(np.percentile(positives, 99.9)), vmin * 1.2)
    bins = np.logspace(np.log10(vmin), np.log10(vmax), 80)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(positives, bins=bins, color="#4E79A7", alpha=0.9)
    ax.set_xscale("log")
    ax.set_title("Sampling Delta Histogram (RAW ADS-B)")
    ax.set_xlabel("Delta t [s] (log scale)")
    ax.set_ylabel("Count")
    ax.grid(True, which="both", linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_sampling_ecdf(delta_values: np.ndarray, out_path: Path) -> None:
    if delta_values.size == 0:
        return
    vals = delta_values[np.isfinite(delta_values) & (delta_values >= 0)]
    if vals.size == 0:
        return
    vals = np.sort(vals)
    y = np.arange(1, len(vals) + 1) / len(vals)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(vals, y, color="#E15759", linewidth=1.5)
    ax.set_xscale("log")
    ax.set_title("Sampling Delta ECDF (RAW ADS-B)")
    ax.set_xlabel("Delta t [s] (log scale)")
    ax.set_ylabel("ECDF")
    ax.grid(True, which="both", linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_spatial_overall(lon: np.ndarray, lat: np.ndarray, out_path: Path) -> None:
    if lon.size == 0 or lat.size == 0:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    hb = ax.hexbin(lon, lat, gridsize=120, mincnt=1, cmap="viridis")
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label("Point density")
    ax.set_title("RAW ADS-B Spatial Footprint (Overall)")
    ax.set_xlabel("Longitude [deg]")
    ax.set_ylabel("Latitude [deg]")
    ax.grid(True, linestyle=":", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_spatial_by_month(
    month_samples: Dict[str, Tuple[np.ndarray, np.ndarray]],
    out_path: Path,
) -> None:
    month_names = [m for m, _ in sorted(MONTH_ORDER.items(), key=lambda x: x[1])]
    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    axes = axes.ravel()

    for idx, month in enumerate(month_names):
        ax = axes[idx]
        if month in month_samples and month_samples[month][0].size > 0:
            lon, lat = month_samples[month]
            ax.hexbin(lon, lat, gridsize=55, mincnt=1, cmap="viridis")
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=9)
        ax.set_title(month.capitalize())
        ax.set_xlabel("Lon")
        ax.set_ylabel("Lat")
        ax.grid(True, linestyle=":", alpha=0.2)

    fig.suptitle("RAW ADS-B Spatial Footprint by Month", fontsize=14)
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_alt_hist(alt_values: np.ndarray, alt_units: str, out_path: Path) -> None:
    if alt_values.size == 0:
        return
    vals = alt_values[np.isfinite(alt_values)]
    if vals.size == 0:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(vals, bins=80, color="#59A14F", alpha=0.9)
    ax.set_title("Altitude Distribution (Overall RAW ADS-B)")
    unit_label = "ft" if alt_units == "feet" else ("m" if alt_units == "meters" else "unknown")
    ax.set_xlabel(f"Altitude [{unit_label}]")
    ax.set_ylabel("Count")
    ax.grid(True, linestyle=":", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_missingness_heatmap(df_missing: pd.DataFrame, out_path: Path) -> None:
    if df_missing.empty:
        return
    plot_df = df_missing.copy()
    plot_df = plot_df[plot_df["month"] != "overall"].copy()
    if plot_df.empty:
        return
    month_names = [m for m, _ in sorted(MONTH_ORDER.items(), key=lambda x: x[1])]
    plot_df["month"] = pd.Categorical(plot_df["month"], categories=month_names, ordered=True)
    plot_df = plot_df.sort_values("month")

    fields = MISSING_FIELDS + ["callsign_and_icao24_present"]
    data = plot_df[fields].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(data, aspect="auto", cmap="magma", vmin=0, vmax=100)
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("Percent [%]")
    ax.set_yticks(np.arange(len(plot_df)))
    ax.set_yticklabels([str(m).capitalize() for m in plot_df["month"]])
    ax.set_xticks(np.arange(len(fields)))
    ax.set_xticklabels(fields, rotation=35, ha="right")
    ax.set_title("Missingness Rates by Month (RAW ADS-B)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_rows_per_day_summary(df_day: pd.DataFrame, out_path: Path) -> None:
    """Plot rows/day summary per month using median and p5-p95 interval."""

    if df_day.empty:
        return
    plot_df = df_day[df_day["month"] != "overall"].copy()
    if plot_df.empty:
        return

    month_names = [m for m, _ in sorted(MONTH_ORDER.items(), key=lambda x: x[1])]
    plot_df["month"] = pd.Categorical(plot_df["month"], categories=month_names, ordered=True)
    plot_df = plot_df.sort_values("month")

    x = np.arange(len(plot_df))
    med = plot_df["rows_per_day_median"].to_numpy(dtype=float)
    p5 = plot_df["rows_per_day_p5"].to_numpy(dtype=float)
    p95 = plot_df["rows_per_day_p95"].to_numpy(dtype=float)
    mean = plot_df["rows_per_day_mean"].to_numpy(dtype=float)

    # Asymmetric error around median.
    yerr = np.vstack([med - p5, p95 - med])

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.errorbar(
        x,
        med,
        yerr=yerr,
        fmt="o",
        color="#4E79A7",
        ecolor="#4E79A7",
        capsize=4,
        label="Median with p5-p95",
    )
    ax.plot(x, mean, color="#E15759", marker="s", linewidth=1.5, label="Mean")
    ax.set_xticks(x)
    ax.set_xticklabels([str(m).capitalize() for m in plot_df["month"]], rotation=35, ha="right")
    ax.set_ylabel("Rows per day")
    ax.set_title("Rows-per-day Summary by Month (RAW ADS-B)")
    ax.grid(True, linestyle=":", alpha=0.35)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _write_markdown_report(
    out_dir: Path,
    volume_by_month: pd.DataFrame,
    volume_by_day_summary: pd.DataFrame,
    sampling_delta_summary: pd.DataFrame,
    bbox_by_month: pd.DataFrame,
    altitude_summary: pd.DataFrame,
    missingness_by_month: pd.DataFrame,
    sanity_checks: pd.DataFrame,
    gs_units: str,
    gs_threshold: float,
    alt_units: str,
    alt_threshold: float,
) -> None:
    overall_volume = volume_by_month.loc[volume_by_month["month"] == "overall"].iloc[0]
    overall_day = volume_by_day_summary.loc[volume_by_day_summary["month"] == "overall"].iloc[0]
    overall_sampling = sampling_delta_summary.loc[sampling_delta_summary["month"] == "overall"].iloc[0]
    overall_bbox = bbox_by_month.loc[bbox_by_month["month"] == "overall"].iloc[0]
    overall_alt = altitude_summary.loc[altitude_summary["month"] == "overall"].iloc[0]
    overall_missing = missingness_by_month.loc[missingness_by_month["month"] == "overall"].iloc[0]
    overall_sanity = sanity_checks.loc[sanity_checks["month"] == "overall"].iloc[0]

    top_missing_field = max(MISSING_FIELDS, key=lambda f: float(overall_missing[f]))
    top_missing_value = float(overall_missing[top_missing_field])

    lines: List[str] = []
    lines.append("# RAW ADS-B EDA Report")
    lines.append("")
    lines.append(
        "This report characterizes the raw ADS-B surveillance dataset before any matching, "
        "runway/A-D labeling, airport-radius filtering, trajectory extraction, or segmentation."
    )
    lines.append("")
    lines.append("## A. Volume and Temporal Coverage")
    lines.append(
        f"- Total rows: **{int(overall_volume['n_rows']):,}**; "
        f"unique icao24: **{int(overall_volume['n_unique_icao24']):,}**; "
        f"unique callsign: **{int(overall_volume['n_unique_callsign']):,}**."
    )
    lines.append(
        f"- Timestamp span: **{overall_volume['timestamp_min']}** to **{overall_volume['timestamp_max']}**."
    )
    lines.append(
        f"- Rows/day (overall): mean={overall_day['rows_per_day_mean']:.1f}, "
        f"median={overall_day['rows_per_day_median']:.1f}, "
        f"p5={overall_day['rows_per_day_p5']:.1f}, p95={overall_day['rows_per_day_p95']:.1f}."
    )
    lines.append("- Tables: `volume_by_month.csv`, `volume_by_day_summary.csv`.")
    lines.append("")
    lines.append("## B. Sampling Characteristics (Irregular Sampling)")
    lines.append(
        f"- Delta summary (overall): median={overall_sampling['delta_median_s']:.3f}s, "
        f"mean={overall_sampling['delta_mean_s']:.3f}s, "
        f"p90={overall_sampling['delta_p90_s']:.3f}s, p99={overall_sampling['delta_p99_s']:.3f}s."
    )
    lines.append(
        f"- Gap fractions: >5s={overall_sampling['frac_delta_gt_5s_pct']:.2f}%, "
        f">30s={overall_sampling['frac_delta_gt_30s_pct']:.2f}%, "
        f">120s={overall_sampling['frac_delta_gt_120s_pct']:.2f}%."
    )
    lines.append("- Figures: `figures/sampling_delta_hist.png`, `figures/sampling_delta_ecdf.png`.")
    lines.append("- Table: `sampling_delta_summary.csv`.")
    lines.append("")
    lines.append("## C. Spatial Coverage / Footprint")
    lines.append(
        f"- Overall bbox: lat[{overall_bbox['lat_min']:.5f}, {overall_bbox['lat_max']:.5f}], "
        f"lon[{overall_bbox['lon_min']:.5f}, {overall_bbox['lon_max']:.5f}]."
    )
    lines.append("- Figures: `figures/spatial_hexbin_overall.png`, `figures/spatial_hexbin_by_month.png`.")
    lines.append("- Table: `bounding_box_by_month.csv`.")
    lines.append("")
    lines.append("## D. Altitude Characteristics")
    lines.append(
        f"- Inferred altitude unit: **{alt_units}** "
        f"(high-value threshold used in sanity checks: {alt_threshold:.1f})."
    )
    lines.append(
        f"- Overall altitude stats: mean={overall_alt['alt_mean']:.2f}, median={overall_alt['alt_median']:.2f}, "
        f"p5={overall_alt['alt_p5']:.2f}, p95={overall_alt['alt_p95']:.2f}, max={overall_alt['alt_max']:.2f}."
    )
    lines.append(f"- Missingness (alt/geoalt): {overall_alt['alt_missing_pct']:.2f}%.")
    lines.append("- Figure: `figures/altitude_hist_overall.png`; Table: `altitude_summary.csv`.")
    lines.append("")
    lines.append("## E. Field Availability / Missingness")
    lines.append(
        f"- Highest overall missingness among core fields: **{top_missing_field} = {top_missing_value:.2f}%**."
    )
    lines.append(
        f"- Rows with both callsign and icao24 present: {overall_missing['callsign_and_icao24_present']:.2f}%."
    )
    lines.append("- Figure: `figures/missingness_heatmap.png`; Table: `missingness_by_month.csv`.")
    lines.append("")
    lines.append("## F. Basic Sanity Checks (Diagnostics Only)")
    lines.append(
        f"- Invalid lat/lon rows: {int(overall_sanity['invalid_lat_lon_rows']):,}; "
        f"track outside [0,360]: {int(overall_sanity['track_out_of_range_rows']):,}."
    )
    lines.append(
        f"- Groundspeed unit inference: **{gs_units}**, high-speed threshold={gs_threshold:.1f}; "
        f"rows above threshold: {int(overall_sanity['gs_unrealistically_high_rows']):,}; "
        f"rows with gs<0: {int(overall_sanity['gs_negative_rows']):,}."
    )
    lines.append(
        f"- Altitude below -1000: {int(overall_sanity['alt_below_minus_1000_rows']):,}; "
        f"altitude above threshold ({alt_threshold:.1f}): {int(overall_sanity['alt_unrealistically_high_rows']):,}."
    )
    lines.append("- Table: `sanity_checks.csv`.")
    lines.append("")

    (out_dir / "raw_adsb_eda.md").write_text("\n".join(lines), encoding="utf-8")


def run_eda(input_dir: Path, output_base: Path) -> Path:
    run_tag = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = output_base / f"raw_adsb_eda_{run_tag}"
    figs_dir = out_dir / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)
    logger = _configure_logging(out_dir)

    month_files = discover_monthly_files(input_dir)
    if not month_files:
        raise FileNotFoundError(
            f"No monthly joblib files found in {input_dir}. Expected names like data_2022_may.joblib."
        )

    logger.info("Discovered %d monthly files in %s", len(month_files), input_dir)
    rng = np.random.RandomState(42)

    volume_rows: List[dict] = []
    day_rows: List[dict] = []
    sampling_rows: List[dict] = []
    bbox_rows: List[dict] = []
    altitude_rows: List[dict] = []
    missing_rows: List[dict] = []
    sanity_rows: List[dict] = []

    overall_rows = 0
    overall_icao_set: set[str] = set()
    overall_callsign_set: set[str] = set()
    overall_min_ts: Optional[pd.Timestamp] = None
    overall_max_ts: Optional[pd.Timestamp] = None
    overall_day_counts: Dict[pd.Timestamp, int] = {}
    overall_bbox = {"lat_min": np.inf, "lat_max": -np.inf, "lon_min": np.inf, "lon_max": -np.inf}

    spatial_month_samples: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    overall_spatial_lon_parts: List[np.ndarray] = []
    overall_spatial_lat_parts: List[np.ndarray] = []
    delta_plot_parts: List[np.ndarray] = []
    alt_plot_parts: List[np.ndarray] = []
    alt_for_unit_parts: List[np.ndarray] = []
    gs_for_unit_parts: List[np.ndarray] = []

    delta_agg = {"count": 0, "sum": 0.0, "gt5": 0, "gt30": 0, "gt120": 0, "max": np.nan}
    alt_agg = {"count": 0, "sum": 0.0, "max": np.nan}

    overall_missing_counts = {field: 0 for field in MISSING_FIELDS}
    overall_callsign_icao_present = 0
    overall_sanity = {
        "invalid_lat_lon_rows": 0,
        "gs_negative_rows": 0,
        "gs_gt_350_rows": 0,
        "gs_gt_700_rows": 0,
        "track_out_of_range_rows": 0,
        "alt_below_minus_1000_rows": 0,
        "alt_gt_60000_rows": 0,
        "alt_gt_18288_rows": 0,
    }

    for month, path in month_files:
        logger.info("Processing month=%s file=%s", month, path.name)
        df = _load_joblib_as_dataframe(path)
        n_rows = int(len(df))
        if n_rows == 0:
            logger.warning("Skipping %s: empty DataFrame", path.name)
            continue
        overall_rows += n_rows

        colmap = _build_column_map(df)
        ts_col = colmap["timestamp"]
        ts = _as_datetime_utc(df[ts_col]) if ts_col is not None else pd.Series(pd.NaT, index=df.index)
        if ts_col is None:
            logger.warning("%s: missing timestamp column; time-based metrics disabled.", path.name)

        icao_col = colmap["icao24"]
        callsign_col = colmap["callsign"]
        if icao_col is None:
            logger.warning("%s: missing icao24 column.", path.name)
            icao_norm = pd.Series(pd.NA, index=df.index, dtype="string")
            unique_icao = 0
        else:
            icao_norm = _normalized_text(df[icao_col])
            unique_icao = int(icao_norm.dropna().nunique())
            overall_icao_set.update(icao_norm.dropna().astype(str).unique().tolist())

        if callsign_col is None:
            logger.warning("%s: missing callsign column.", path.name)
            callsign_norm = pd.Series(pd.NA, index=df.index, dtype="string")
            unique_callsign = 0
        else:
            callsign_norm = _normalized_text(df[callsign_col])
            unique_callsign = int(callsign_norm.dropna().nunique())
            overall_callsign_set.update(callsign_norm.dropna().astype(str).unique().tolist())

        ts_valid = ts.dropna()
        ts_min = ts_valid.min() if not ts_valid.empty else pd.NaT
        ts_max = ts_valid.max() if not ts_valid.empty else pd.NaT
        if pd.notna(ts_min):
            overall_min_ts = ts_min if overall_min_ts is None else min(overall_min_ts, ts_min)
        if pd.notna(ts_max):
            overall_max_ts = ts_max if overall_max_ts is None else max(overall_max_ts, ts_max)

        volume_rows.append(
            {
                "month": month,
                "n_rows": n_rows,
                "n_unique_icao24": unique_icao,
                "n_unique_callsign": unique_callsign,
                "timestamp_min": str(ts_min) if pd.notna(ts_min) else pd.NA,
                "timestamp_max": str(ts_max) if pd.notna(ts_max) else pd.NA,
            }
        )

        day_counts = ts_valid.dt.floor("D").value_counts().sort_index() if not ts_valid.empty else pd.Series(dtype=int)
        for day, cnt in day_counts.items():
            overall_day_counts[day] = overall_day_counts.get(day, 0) + int(cnt)
        row = {"month": month}
        row.update(_day_summary(day_counts))
        day_rows.append(row)

        lat_col = colmap["lat"]
        lon_col = colmap["lon"]
        lat_num = _as_numeric(df[lat_col]) if lat_col is not None else pd.Series(np.nan, index=df.index)
        lon_num = _as_numeric(df[lon_col]) if lon_col is not None else pd.Series(np.nan, index=df.index)
        if lat_col is None or lon_col is None:
            logger.warning("%s: missing lat/lon column(s).", path.name)
            bbox_rows.append({"month": month, "lat_min": np.nan, "lat_max": np.nan, "lon_min": np.nan, "lon_max": np.nan, "n_valid_points": 0})
        else:
            valid_spatial = lat_num.notna() & lon_num.notna()
            lat_valid = lat_num[valid_spatial].to_numpy(dtype=float)
            lon_valid = lon_num[valid_spatial].to_numpy(dtype=float)
            if lat_valid.size == 0:
                bbox_rows.append({"month": month, "lat_min": np.nan, "lat_max": np.nan, "lon_min": np.nan, "lon_max": np.nan, "n_valid_points": 0})
            else:
                lat_min, lat_max = float(np.min(lat_valid)), float(np.max(lat_valid))
                lon_min, lon_max = float(np.min(lon_valid)), float(np.max(lon_valid))
                bbox_rows.append({"month": month, "lat_min": lat_min, "lat_max": lat_max, "lon_min": lon_min, "lon_max": lon_max, "n_valid_points": int(lat_valid.size)})
                overall_bbox["lat_min"] = min(overall_bbox["lat_min"], lat_min)
                overall_bbox["lat_max"] = max(overall_bbox["lat_max"], lat_max)
                overall_bbox["lon_min"] = min(overall_bbox["lon_min"], lon_min)
                overall_bbox["lon_max"] = max(overall_bbox["lon_max"], lon_max)
                idx = _sample_array(np.arange(lat_valid.size), MAX_SPATIAL_POINTS_PER_MONTH, rng).astype(int)
                spatial_month_samples[month] = (lon_valid[idx], lat_valid[idx])
                overall_spatial_lon_parts.append(lon_valid[idx])
                overall_spatial_lat_parts.append(lat_valid[idx])

        alt_col = colmap["alt"]
        geoalt_col = colmap["geoalt"]
        alt_used = alt_col if alt_col is not None else geoalt_col
        if alt_used is None:
            logger.warning("%s: missing altitude and geoaltitude columns.", path.name)
            alt_num = pd.Series(np.nan, index=df.index)
            alt_source = "missing"
        else:
            alt_num = _as_numeric(df[alt_used])
            alt_source = alt_used
        alt_valid = alt_num.dropna().to_numpy(dtype=float)
        alt_missing_pct = float(100.0 * alt_num.isna().mean())
        altitude_rows.append(
            {
                "month": month,
                "alt_source": alt_source,
                "alt_mean": float(np.mean(alt_valid)) if alt_valid.size else np.nan,
                "alt_median": float(np.median(alt_valid)) if alt_valid.size else np.nan,
                "alt_p5": float(np.percentile(alt_valid, 5)) if alt_valid.size else np.nan,
                "alt_p95": float(np.percentile(alt_valid, 95)) if alt_valid.size else np.nan,
                "alt_max": float(np.max(alt_valid)) if alt_valid.size else np.nan,
                "alt_missing_pct": alt_missing_pct,
                "n_valid_alt": int(alt_valid.size),
            }
        )
        if alt_valid.size:
            alt_agg["count"] += int(alt_valid.size)
            alt_agg["sum"] += float(np.sum(alt_valid))
            alt_agg["max"] = float(np.max(alt_valid)) if not np.isfinite(alt_agg["max"]) else max(float(alt_agg["max"]), float(np.max(alt_valid)))
            alt_sample = _sample_array(alt_valid, MAX_ALT_POINTS_PER_MONTH, rng)
            alt_plot_parts.append(alt_sample)
            alt_for_unit_parts.append(alt_sample)

        # Missingness
        def missing_count(col_name: Optional[str]) -> int:
            if col_name is None:
                return n_rows
            return int(df[col_name].isna().sum())

        month_missing_counts = {
            "callsign": missing_count(callsign_col),
            "origin": missing_count(colmap["origin"]),
            "destination": missing_count(colmap["destination"]),
            "gs": missing_count(colmap["gs"]),
            "track": missing_count(colmap["track"]),
            "vr": missing_count(colmap["vr"]),
            "onground": missing_count(colmap["onground"]),
            "lat": missing_count(lat_col),
            "lon": missing_count(lon_col),
        }
        if alt_col is None and geoalt_col is None:
            alt_geo_missing = n_rows
        else:
            alt_a = df[alt_col].isna() if alt_col is not None else pd.Series(True, index=df.index)
            alt_b = df[geoalt_col].isna() if geoalt_col is not None else pd.Series(True, index=df.index)
            alt_geo_missing = int((alt_a & alt_b).sum())
        month_missing_counts["alt_or_geoalt"] = alt_geo_missing

        if icao_col is not None and callsign_col is not None:
            present_count = int((_normalized_text(df[icao_col]).notna() & _normalized_text(df[callsign_col]).notna()).sum())
        else:
            present_count = 0
        miss_row = {"month": month}
        for field in MISSING_FIELDS:
            miss_row[field] = 100.0 * month_missing_counts[field] / n_rows
            overall_missing_counts[field] += month_missing_counts[field]
        miss_row["callsign_and_icao24_present"] = 100.0 * present_count / n_rows
        overall_callsign_icao_present += present_count
        missing_rows.append(miss_row)

        # Sampling deltas
        if ts_col is None or icao_col is None:
            sampling_rows.append({"month": month, "n_deltas": 0, "delta_mean_s": np.nan, "delta_median_s": np.nan, "delta_p90_s": np.nan, "delta_p99_s": np.nan, "delta_max_s": np.nan, "frac_delta_gt_5s_pct": np.nan, "frac_delta_gt_30s_pct": np.nan, "frac_delta_gt_120s_pct": np.nan})
        else:
            sample_df = pd.DataFrame({"timestamp": ts, "icao24": icao_norm}).dropna(subset=["timestamp", "icao24"])
            keys = ["icao24"]
            if callsign_col is not None:
                sample_df["callsign"] = callsign_norm.fillna("<NA>")
                keys.append("callsign")
            if sample_df.empty:
                deltas = np.array([], dtype=float)
            else:
                sample_df = sample_df.sort_values(keys + ["timestamp"], kind="mergesort")
                dsec = sample_df.groupby(keys, sort=False)["timestamp"].diff().dt.total_seconds()
                deltas = dsec[dsec.notna() & (dsec >= 0)].to_numpy(dtype=float)
            if deltas.size:
                n_d = int(deltas.size)
                gt5, gt30, gt120 = int(np.sum(deltas > 5)), int(np.sum(deltas > 30)), int(np.sum(deltas > 120))
                sampling_rows.append({"month": month, "n_deltas": n_d, "delta_mean_s": float(np.mean(deltas)), "delta_median_s": float(np.median(deltas)), "delta_p90_s": float(np.percentile(deltas, 90)), "delta_p99_s": float(np.percentile(deltas, 99)), "delta_max_s": float(np.max(deltas)), "frac_delta_gt_5s_pct": 100.0 * gt5 / n_d, "frac_delta_gt_30s_pct": 100.0 * gt30 / n_d, "frac_delta_gt_120s_pct": 100.0 * gt120 / n_d})
                delta_agg["count"] += n_d
                delta_agg["sum"] += float(np.sum(deltas))
                delta_agg["gt5"] += gt5
                delta_agg["gt30"] += gt30
                delta_agg["gt120"] += gt120
                delta_agg["max"] = float(np.max(deltas)) if not np.isfinite(delta_agg["max"]) else max(float(delta_agg["max"]), float(np.max(deltas)))
                delta_plot_parts.append(_sample_array(deltas, MAX_DELTA_POINTS_PER_MONTH, rng))
            else:
                sampling_rows.append({"month": month, "n_deltas": 0, "delta_mean_s": np.nan, "delta_median_s": np.nan, "delta_p90_s": np.nan, "delta_p99_s": np.nan, "delta_max_s": np.nan, "frac_delta_gt_5s_pct": np.nan, "frac_delta_gt_30s_pct": np.nan, "frac_delta_gt_120s_pct": np.nan})

        # Sanity checks
        invalid_lat_lon = int(((lat_num < -90) | (lat_num > 90) | (lon_num < -180) | (lon_num > 180)).fillna(False).sum()) if (lat_col is not None and lon_col is not None) else 0
        gs_col = colmap["gs"]
        if gs_col is None:
            gs_negative = gs_gt_350 = gs_gt_700 = 0
        else:
            gs_num = _as_numeric(df[gs_col])
            gs_negative = int((gs_num < 0).fillna(False).sum())
            gs_gt_350 = int((gs_num > 350).fillna(False).sum())
            gs_gt_700 = int((gs_num > 700).fillna(False).sum())
            gs_positive = gs_num[(gs_num > 0).fillna(False)].to_numpy(dtype=float)
            if gs_positive.size:
                gs_for_unit_parts.append(_sample_array(gs_positive, MAX_GS_POINTS_PER_MONTH, rng))
        track_col = colmap["track"]
        track_out = int(((_as_numeric(df[track_col]) < 0) | (_as_numeric(df[track_col]) > 360)).fillna(False).sum()) if track_col is not None else 0
        alt_below = int((alt_num < -1000).fillna(False).sum())
        alt_gt_60000 = int((alt_num > 60000).fillna(False).sum())
        alt_gt_18288 = int((alt_num > 18288).fillna(False).sum())

        sanity_rows.append({"month": month, "n_rows": n_rows, "invalid_lat_lon_rows": invalid_lat_lon, "gs_negative_rows": gs_negative, "gs_gt_350_rows": gs_gt_350, "gs_gt_700_rows": gs_gt_700, "track_out_of_range_rows": track_out, "alt_below_minus_1000_rows": alt_below, "alt_gt_60000_rows": alt_gt_60000, "alt_gt_18288_rows": alt_gt_18288})
        overall_sanity["invalid_lat_lon_rows"] += invalid_lat_lon
        overall_sanity["gs_negative_rows"] += gs_negative
        overall_sanity["gs_gt_350_rows"] += gs_gt_350
        overall_sanity["gs_gt_700_rows"] += gs_gt_700
        overall_sanity["track_out_of_range_rows"] += track_out
        overall_sanity["alt_below_minus_1000_rows"] += alt_below
        overall_sanity["alt_gt_60000_rows"] += alt_gt_60000
        overall_sanity["alt_gt_18288_rows"] += alt_gt_18288

        logger.info("Finished %s: rows=%d unique_icao24=%d valid_ts=%d", month, n_rows, unique_icao, int(ts_valid.size))

    if overall_rows == 0:
        raise RuntimeError("No rows processed from input files.")

    volume_rows.append(
        {
            "month": "overall",
            "n_rows": int(overall_rows),
            "n_unique_icao24": int(len(overall_icao_set)),
            "n_unique_callsign": int(len(overall_callsign_set)),
            "timestamp_min": str(overall_min_ts) if overall_min_ts is not None else pd.NA,
            "timestamp_max": str(overall_max_ts) if overall_max_ts is not None else pd.NA,
        }
    )
    overall_day_series = pd.Series(overall_day_counts).sort_index()
    day_overall = {"month": "overall"}
    day_overall.update(_day_summary(overall_day_series))
    day_rows.append(day_overall)
    bbox_rows.append(
        {
            "month": "overall",
            "lat_min": float(overall_bbox["lat_min"]) if np.isfinite(overall_bbox["lat_min"]) else np.nan,
            "lat_max": float(overall_bbox["lat_max"]) if np.isfinite(overall_bbox["lat_max"]) else np.nan,
            "lon_min": float(overall_bbox["lon_min"]) if np.isfinite(overall_bbox["lon_min"]) else np.nan,
            "lon_max": float(overall_bbox["lon_max"]) if np.isfinite(overall_bbox["lon_max"]) else np.nan,
            "n_valid_points": int(np.sum([arr.size for arr in overall_spatial_lat_parts])) if overall_spatial_lat_parts else 0,
        }
    )

    delta_plot = np.concatenate(delta_plot_parts) if delta_plot_parts else np.array([], dtype=float)
    if delta_agg["count"] > 0:
        sampling_rows.append(
            {
                "month": "overall",
                "n_deltas": int(delta_agg["count"]),
                "delta_mean_s": float(delta_agg["sum"] / delta_agg["count"]),
                "delta_median_s": float(np.percentile(delta_plot, 50)) if delta_plot.size else np.nan,
                "delta_p90_s": float(np.percentile(delta_plot, 90)) if delta_plot.size else np.nan,
                "delta_p99_s": float(np.percentile(delta_plot, 99)) if delta_plot.size else np.nan,
                "delta_max_s": float(delta_agg["max"]),
                "frac_delta_gt_5s_pct": 100.0 * delta_agg["gt5"] / delta_agg["count"],
                "frac_delta_gt_30s_pct": 100.0 * delta_agg["gt30"] / delta_agg["count"],
                "frac_delta_gt_120s_pct": 100.0 * delta_agg["gt120"] / delta_agg["count"],
            }
        )
    else:
        sampling_rows.append({"month": "overall", "n_deltas": 0, "delta_mean_s": np.nan, "delta_median_s": np.nan, "delta_p90_s": np.nan, "delta_p99_s": np.nan, "delta_max_s": np.nan, "frac_delta_gt_5s_pct": np.nan, "frac_delta_gt_30s_pct": np.nan, "frac_delta_gt_120s_pct": np.nan})

    alt_plot = np.concatenate(alt_plot_parts) if alt_plot_parts else np.array([], dtype=float)
    alt_unit_sample = np.concatenate(alt_for_unit_parts) if alt_for_unit_parts else np.array([], dtype=float)
    gs_unit_sample = np.concatenate(gs_for_unit_parts) if gs_for_unit_parts else np.array([], dtype=float)
    gs_units, gs_threshold, gs_q99 = _infer_gs_units(gs_unit_sample)
    alt_units, alt_threshold, alt_q95 = _infer_alt_units(alt_unit_sample)

    if alt_agg["count"] > 0:
        altitude_rows.append(
            {
                "month": "overall",
                "alt_source": "alt_if_present_else_geoalt",
                "alt_mean": float(alt_agg["sum"] / alt_agg["count"]),
                "alt_median": float(np.percentile(alt_plot, 50)) if alt_plot.size else np.nan,
                "alt_p5": float(np.percentile(alt_plot, 5)) if alt_plot.size else np.nan,
                "alt_p95": float(np.percentile(alt_plot, 95)) if alt_plot.size else np.nan,
                "alt_max": float(alt_agg["max"]),
                "alt_missing_pct": 100.0 * overall_missing_counts["alt_or_geoalt"] / overall_rows,
                "n_valid_alt": int(alt_agg["count"]),
            }
        )
    else:
        altitude_rows.append({"month": "overall", "alt_source": "alt_if_present_else_geoalt", "alt_mean": np.nan, "alt_median": np.nan, "alt_p5": np.nan, "alt_p95": np.nan, "alt_max": np.nan, "alt_missing_pct": 100.0, "n_valid_alt": 0})

    miss_overall = {"month": "overall"}
    for field in MISSING_FIELDS:
        miss_overall[field] = 100.0 * overall_missing_counts[field] / overall_rows
    miss_overall["callsign_and_icao24_present"] = 100.0 * overall_callsign_icao_present / overall_rows
    missing_rows.append(miss_overall)

    sanity_rows.append(
        {
            "month": "overall",
            "n_rows": int(overall_rows),
            "invalid_lat_lon_rows": int(overall_sanity["invalid_lat_lon_rows"]),
            "gs_negative_rows": int(overall_sanity["gs_negative_rows"]),
            "gs_gt_350_rows": int(overall_sanity["gs_gt_350_rows"]),
            "gs_gt_700_rows": int(overall_sanity["gs_gt_700_rows"]),
            "track_out_of_range_rows": int(overall_sanity["track_out_of_range_rows"]),
            "alt_below_minus_1000_rows": int(overall_sanity["alt_below_minus_1000_rows"]),
            "alt_gt_60000_rows": int(overall_sanity["alt_gt_60000_rows"]),
            "alt_gt_18288_rows": int(overall_sanity["alt_gt_18288_rows"]),
        }
    )

    volume_df = pd.DataFrame(volume_rows)
    day_df = pd.DataFrame(day_rows)
    sampling_df = pd.DataFrame(sampling_rows)
    bbox_df = pd.DataFrame(bbox_rows)
    altitude_df = pd.DataFrame(altitude_rows)
    missing_df = pd.DataFrame(missing_rows)
    sanity_df = pd.DataFrame(sanity_rows)
    sanity_df["gs_unrealistically_high_rows"] = sanity_df["gs_gt_700_rows"] if gs_units == "knots" else sanity_df["gs_gt_350_rows"]
    sanity_df["alt_unrealistically_high_rows"] = sanity_df["alt_gt_60000_rows"] if alt_units == "feet" else sanity_df["alt_gt_18288_rows"]
    sanity_df["gs_unit_inferred"] = gs_units
    sanity_df["gs_high_threshold_used"] = gs_threshold
    sanity_df["alt_unit_inferred"] = alt_units
    sanity_df["alt_high_threshold_used"] = alt_threshold

    volume_df.to_csv(out_dir / "volume_by_month.csv", index=False)
    day_df.to_csv(out_dir / "volume_by_day_summary.csv", index=False)
    sampling_df.to_csv(out_dir / "sampling_delta_summary.csv", index=False)
    bbox_df.to_csv(out_dir / "bounding_box_by_month.csv", index=False)
    altitude_df.to_csv(out_dir / "altitude_summary.csv", index=False)
    missing_df.to_csv(out_dir / "missingness_by_month.csv", index=False)
    sanity_df.to_csv(out_dir / "sanity_checks.csv", index=False)

    _plot_sampling_hist(delta_plot, figs_dir / "sampling_delta_hist.png")
    _plot_sampling_ecdf(delta_plot, figs_dir / "sampling_delta_ecdf.png")
    _plot_rows_per_day_summary(day_df, figs_dir / "volume_by_day_summary.png")
    if overall_spatial_lon_parts and overall_spatial_lat_parts:
        overall_lon = np.concatenate(overall_spatial_lon_parts)
        overall_lat = np.concatenate(overall_spatial_lat_parts)
        if overall_lon.size > MAX_SPATIAL_POINTS_OVERALL:
            idx = _sample_array(np.arange(overall_lon.size), MAX_SPATIAL_POINTS_OVERALL, rng).astype(int)
            overall_lon = overall_lon[idx]
            overall_lat = overall_lat[idx]
        _plot_spatial_overall(overall_lon, overall_lat, figs_dir / "spatial_hexbin_overall.png")
    _plot_spatial_by_month(spatial_month_samples, figs_dir / "spatial_hexbin_by_month.png")
    _plot_alt_hist(alt_plot, alt_units, figs_dir / "altitude_hist_overall.png")
    _plot_missingness_heatmap(missing_df, figs_dir / "missingness_heatmap.png")

    _write_markdown_report(
        out_dir=out_dir,
        volume_by_month=volume_df,
        volume_by_day_summary=day_df,
        sampling_delta_summary=sampling_df,
        bbox_by_month=bbox_df,
        altitude_summary=altitude_df,
        missingness_by_month=missing_df,
        sanity_checks=sanity_df,
        gs_units=gs_units,
        gs_threshold=gs_threshold,
        alt_units=alt_units,
        alt_threshold=alt_threshold,
    )

    logger.info("Inferred units: gs=%s (q99=%.3f, threshold=%.1f)", gs_units, gs_q99, gs_threshold)
    logger.info("Inferred units: altitude=%s (q95=%.3f, threshold=%.1f)", alt_units, alt_q95, alt_threshold)
    logger.info("RAW ADS-B EDA completed. Output folder: %s", out_dir)
    return out_dir


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    run_eda(input_dir=input_dir, output_base=output_dir)


if __name__ == "__main__":
    main()
