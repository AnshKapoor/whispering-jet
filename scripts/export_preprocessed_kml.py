"""Export sampled trajectories from a preprocessed CSV to a KML file.

The script samples up to N flights per flow (A/D, Runway) and writes trajectory
polylines in WGS84 lon/lat coordinates.

Input columns required:
  - flight_id
  - A/D
  - Runway
  - latitude
  - longitude

Optional:
  - step (preferred ordering inside each flight)
  - altitude (written as KML altitude; defaults to 0 if absent)

Example:
  python scripts/export_preprocessed_kml.py ^
    --preprocessed output/preprocessed/preprocessed_1.csv ^
    --max-flights-per-flow 200
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import pandas as pd
from pyproj import Transformer


FLOW_COLORS = [
    "#E15759",
    "#59A14F",
    "#4E79A7",
    "#F28E2B",
    "#B07AA1",
    "#76B7B2",
    "#EDC948",
    "#FF9DA7",
]

DEFAULT_STARTPOINTS: Dict[Tuple[str, str], Tuple[float, float]] = {
    # Source: noise_simulation/generate_doc29_inputs.py::STARTPOINTS
    ("27L", "landing"): (546026.9, 5811859.4),
    ("27L", "take-off"): (548229.4, 5811780.4),
    ("09R", "landing"): (548229.4, 5811780.4),
    ("09R", "take-off"): (546026.9, 5811859.4),
    ("27R", "landing"): (544424.6, 5813317.4),
    ("27R", "take-off"): (547474.3, 5813208.8),
    ("09L", "landing"): (547474.3, 5813208.8),
    ("09L", "take-off"): (544424.6, 5813317.4),
}

try:
    from noise_simulation.generate_doc29_inputs import STARTPOINTS as STARTPOINTS_FROM_DOC29
except Exception:
    STARTPOINTS_FROM_DOC29 = {}

STARTPOINTS: Dict[Tuple[str, str], Tuple[float, float]] = (
    STARTPOINTS_FROM_DOC29 or DEFAULT_STARTPOINTS
)

MEASURING_POINT_COORDS: Dict[str, Tuple[float, float]] = {
    # Source: merge_adsb_noise.py (mp_coords -> parse_dms)
    "M01": (52.453611111111115, 9.760277777777778),
    "M02": (52.465833333333336, 9.750555555555556),
    "M03": (52.46666666666667, 9.790277777777778),
    "M04": (52.45527777777778, 9.813333333333334),
    "M05": (52.46805555555556, 9.8325),
    "M06": (52.45388888888889, 9.625277777777779),
    "M07": (52.461111111111116, 9.581944444444444),
    "M08": (52.468611111111116, 9.546666666666667),
    "M09": (52.468333333333334, 9.619166666666667),
}

MODE_TO_FLOW_PREFIX = {
    "landing": "Landung",
    "take-off": "Start",
}


def _kml_header(name: str) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>{name}</name>
"""


def _kml_footer() -> str:
    return """  </Document>
</kml>
"""


def _kml_color(hex_rgb: str, alpha: str = "ff") -> str:
    """Convert #RRGGBB to KML AABBGGRR."""

    rgb = hex_rgb.lstrip("#")
    if len(rgb) != 6:
        return "ff0000ff"
    r, g, b = rgb[0:2], rgb[2:4], rgb[4:6]
    return f"{alpha}{b}{g}{r}"


def _style_block(style_id: str, color: str) -> str:
    return f"""    <Style id="{style_id}">
      <LineStyle>
        <color>{color}</color>
        <width>2</width>
      </LineStyle>
    </Style>
"""


def _placemark(name: str, description: str, coords: str, style: str | None = None) -> str:
    style_line = f"<styleUrl>#{style}</styleUrl>" if style else ""
    return f"""    <Placemark>
      <name>{name}</name>
      {style_line}
      <description><![CDATA[{description}]]></description>
      <LineString>
        <tessellate>1</tessellate>
        <coordinates>{coords}</coordinates>
      </LineString>
</Placemark>
"""


def _point_placemark(name: str, description: str, lon: float, lat: float) -> str:
    return f"""    <Placemark>
      <name>{name}</name>
      <description><![CDATA[{description}]]></description>
      <Point>
        <coordinates>{lon},{lat},0</coordinates>
      </Point>
    </Placemark>
"""


def _format_flow_label(runway: str, mode: str) -> str:
    """Return a human-readable flow label (e.g., 'Landung 27L')."""

    flow_prefix = MODE_TO_FLOW_PREFIX.get(mode, mode)
    return f"{flow_prefix} {runway}"


def _sample_flights_per_flow(
    preprocessed_path: Path,
    max_flights_per_flow: int,
    seed: int,
) -> pd.DataFrame:
    """Return one row per sampled flight with flow metadata."""

    meta_cols = ["flight_id", "A/D", "Runway"]
    meta = pd.read_csv(preprocessed_path, usecols=meta_cols)
    meta = meta.dropna(subset=["flight_id", "A/D", "Runway"])
    meta["flight_id"] = meta["flight_id"].astype(int)
    meta["A/D"] = meta["A/D"].astype(str)
    meta["Runway"] = meta["Runway"].astype(str)
    meta["flow_label"] = meta["A/D"] + "_" + meta["Runway"]
    meta = meta.drop_duplicates(subset=["flight_id"], keep="first")

    sampled = []
    for _, group in meta.groupby("flow_label", sort=True):
        n = min(max_flights_per_flow, len(group))
        sampled.append(group.sample(n=n, random_state=seed))
    if not sampled:
        return pd.DataFrame(columns=["flight_id", "A/D", "Runway", "flow_label"])

    return pd.concat(sampled, ignore_index=True)


def _load_sampled_trajectories(
    preprocessed_path: Path,
    sampled_meta: pd.DataFrame,
) -> pd.DataFrame:
    """Load trajectory rows only for sampled flight IDs using chunked IO."""

    if sampled_meta.empty:
        return pd.DataFrame()

    sampled_ids = set(sampled_meta["flight_id"].astype(int).tolist())
    base_cols = ["flight_id", "latitude", "longitude", "step", "altitude"]
    available_cols = pd.read_csv(preprocessed_path, nrows=0).columns.tolist()
    usecols = [c for c in base_cols if c in available_cols]
    if not {"flight_id", "latitude", "longitude"}.issubset(usecols):
        raise ValueError(
            "Preprocessed CSV must include at least: flight_id, latitude, longitude."
        )

    chunks = []
    for chunk in pd.read_csv(preprocessed_path, usecols=usecols, chunksize=200_000):
        chunk["flight_id"] = chunk["flight_id"].astype(int)
        chunk = chunk[chunk["flight_id"].isin(sampled_ids)]
        if not chunk.empty:
            chunks.append(chunk)
    if not chunks:
        return pd.DataFrame()

    traj = pd.concat(chunks, ignore_index=True)
    traj = traj.merge(
        sampled_meta[["flight_id", "A/D", "Runway", "flow_label"]],
        on="flight_id",
        how="inner",
    )
    if "step" not in traj.columns:
        traj["step"] = traj.groupby("flight_id").cumcount()
    if "altitude" not in traj.columns:
        traj["altitude"] = 0.0
    return traj


def _build_flow_styles(flow_labels: Iterable[str]) -> Dict[str, str]:
    styles: Dict[str, str] = {}
    for idx, flow in enumerate(sorted(flow_labels)):
        styles[flow] = f"flow_{flow.replace(' ', '_')}"
    return styles


def main() -> None:
    parser = argparse.ArgumentParser(description="Export sampled preprocessed flights to KML.")
    parser.add_argument(
        "--preprocessed",
        default="output/preprocessed/preprocessed_1.csv",
        help="Path to preprocessed CSV.",
    )
    parser.add_argument(
        "--max-flights-per-flow",
        type=int,
        default=200,
        help="Maximum flights sampled per flow label (A/D_Runway).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for per-flow flight sampling.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output KML path. Default: output/eda/preprocessed_kml/<stem>_sampled_<N>_per_flow.kml",
    )
    parser.add_argument(
        "--include-runway-labels",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include combined runway-flow labels (e.g., 'Landung 27L/Start 09R').",
    )
    parser.add_argument(
        "--include-measuring-points",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include measuring-point placemarks (M01-M09).",
    )
    args = parser.parse_args()

    preprocessed_path = Path(args.preprocessed)
    if not preprocessed_path.exists():
        raise FileNotFoundError(f"Preprocessed CSV not found: {preprocessed_path}")

    sampled_meta = _sample_flights_per_flow(
        preprocessed_path=preprocessed_path,
        max_flights_per_flow=args.max_flights_per_flow,
        seed=args.seed,
    )
    if sampled_meta.empty:
        raise ValueError("No flights found to export.")

    traj = _load_sampled_trajectories(preprocessed_path, sampled_meta)
    if traj.empty:
        raise ValueError("No trajectory rows found for sampled flight IDs.")

    if args.out:
        out_path = Path(args.out)
    else:
        out_dir = Path("output") / "eda" / "preprocessed_kml"
        out_path = out_dir / f"{preprocessed_path.stem}_sampled_{args.max_flights_per_flow}_per_flow.kml"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    flow_counts = sampled_meta.groupby("flow_label").size().to_dict()
    styles = _build_flow_styles(flow_counts.keys())

    parts = [_kml_header(preprocessed_path.stem)]
    for idx, flow in enumerate(sorted(styles)):
        color = _kml_color(FLOW_COLORS[idx % len(FLOW_COLORS)])
        parts.append(_style_block(styles[flow], color))

    if args.include_runway_labels and STARTPOINTS:
        transformer = Transformer.from_crs("epsg:32632", "epsg:4326", always_xy=True)
        runway_points = {}
        for (runway, mode), (x, y) in STARTPOINTS.items():
            lon, lat = transformer.transform(x, y)
            runway_points[(runway, mode)] = (lon, lat, x, y)

        grouped_flow_points: Dict[Tuple[float, float], Dict[str, object]] = {}
        for (runway, mode), (lon, lat, x, y) in runway_points.items():
            # Group reciprocal operations that share the same threshold coordinate.
            key = (float(x), float(y))
            flow_label = _format_flow_label(runway, mode)
            if key not in grouped_flow_points:
                grouped_flow_points[key] = {
                    "flow_labels": [],
                    "runways": [],
                    "modes": [],
                    "x_utm": x,
                    "y_utm": y,
                    "lon": lon,
                    "lat": lat,
                }
            grouped_flow_points[key]["flow_labels"].append(flow_label)
            grouped_flow_points[key]["runways"].append(runway)
            grouped_flow_points[key]["modes"].append(mode)

        for point_meta in sorted(
            grouped_flow_points.values(),
            key=lambda item: "/".join(sorted(item["flow_labels"])),
        ):
            flow_labels = sorted(point_meta["flow_labels"])
            combined_label = "/".join(flow_labels)
            parts.append(
                _point_placemark(
                    name=f"Flow {combined_label}",
                    description=json.dumps(
                        {
                            "flow_labels": flow_labels,
                            "runways": sorted(point_meta["runways"]),
                            "modes": sorted(point_meta["modes"]),
                            "x_utm": point_meta["x_utm"],
                            "y_utm": point_meta["y_utm"],
                        },
                        indent=2,
                    ),
                    lon=float(point_meta["lon"]),
                    lat=float(point_meta["lat"]),
                )
            )

    if args.include_measuring_points:
        for mp, (lat, lon) in sorted(MEASURING_POINT_COORDS.items()):
            parts.append(
                _point_placemark(
                    name=f"MP {mp}",
                    description=json.dumps(
                        {"measurement_point": mp, "latitude": lat, "longitude": lon},
                        indent=2,
                    ),
                    lon=lon,
                    lat=lat,
                )
            )

    for (flow_label, flight_id), flight in traj.groupby(["flow_label", "flight_id"], sort=True):
        flight = flight.sort_values("step")
        coords = " ".join(
            f"{lon},{lat},{float(alt) if pd.notna(alt) else 0.0}"
            for lon, lat, alt in flight[["longitude", "latitude", "altitude"]].itertuples(index=False, name=None)
        )
        meta = flight.iloc[0]
        desc = {
            "flow_label": flow_label,
            "A/D": meta.get("A/D"),
            "Runway": meta.get("Runway"),
            "flight_id": int(flight_id),
            "source": str(preprocessed_path),
        }
        parts.append(
            _placemark(
                name=f"{flow_label}_flight_{int(flight_id)}",
                description=json.dumps(desc, indent=2),
                coords=coords,
                style=styles.get(flow_label),
            )
        )

    parts.append(_kml_footer())
    out_path.write_text("".join(parts), encoding="utf-8")

    print(f"Wrote KML to {out_path}")
    print("Flights sampled per flow:")
    for flow in sorted(flow_counts):
        print(f"  {flow}: {int(flow_counts[flow])}")


if __name__ == "__main__":
    main()
