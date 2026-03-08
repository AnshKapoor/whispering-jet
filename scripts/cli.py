"""CLI entry point for the backbone tracks pipeline.

Orchestrates loading, segmentation, preprocessing, clustering, backbone
computation, and optional plotting, honoring test-mode caps from the config.
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

from backbone_tracks.backbone import compute_backbones
from backbone_tracks.clustering import cluster_flow
from backbone_tracks.config import get_nested, load_config
from backbone_tracks.io import add_utm_coordinates, ensure_required_columns, load_monthly_csvs, save_dataframe
from backbone_tracks.preprocessing import preprocess_flights
from backbone_tracks.segmentation import limit_flights_per_flow, segment_flights


def configure_logging(log_cfg: Dict[str, object]) -> None:
    """Configure root logger with both file and console handlers."""

    log_dir = Path(log_cfg.get("dir", "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    filename = log_cfg.get("filename", "backbone.log")
    log_path = log_dir / filename
    level_name = str(log_cfg.get("level", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)

    fmt = "%(asctime)s - %(levelname)s - %(message)s"
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(fmt))
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter(fmt))

    root.addHandler(file_handler)
    root.addHandler(stream_handler)
    root.info("Logging to %s (level=%s)", log_path, level_name)


def _apply_flow_filters(df: pd.DataFrame, flow_keys: Iterable[str], include: List[List[str]] | None) -> pd.DataFrame:
    """Filter dataframe to only the allowed flows, if provided."""

    if not include:
        return df
    include_set = {tuple(item) for item in include}
    mask = df[list(flow_keys)].apply(tuple, axis=1).isin(include_set)
    filtered = df[mask].reset_index(drop=True)
    logging.info("Filtered flows to include list: %d rows retained", len(filtered))
    return filtered


def main(config_path: str = "config/backbone.yaml") -> None:
    cfg = load_config(config_path)

    configure_logging(cfg.get("logging", {}) or {})
    testing_cfg = cfg.get("testing", {}) or {}
    test_mode = bool(testing_cfg.get("enabled", False))
    logging.info("Test mode: %s", test_mode)

    input_cfg = cfg.get("input", {}) or {}
    csv_glob = input_cfg.get("csv_glob", "Enhanced/matched_*.csv")
    parse_dates = input_cfg.get("parse_dates", ["timestamp", "firstseen", "lastseen"])
    max_rows = testing_cfg.get("max_rows_total") if test_mode else None
    coord_cfg = cfg.get("coordinates", {}) or {}
    requested_use_utm = bool(coord_cfg.get("use_utm", True))
    utm_crs = coord_cfg.get("utm_crs", "epsg:32632")
    # Always convert coordinates to UTM so downstream steps operate in a planar system.
    use_utm = True
    if not requested_use_utm:
        logging.info("Forcing UTM conversion despite use_utm=%s in config.", requested_use_utm)

    df = load_monthly_csvs(csv_glob=csv_glob, parse_dates=parse_dates, max_rows_total=max_rows)
    df = ensure_required_columns(df)
    df = add_utm_coordinates(df, utm_crs=utm_crs)
    logging.info("Using UTM coordinates (CRS=%s) for clustering.", utm_crs)

    flow_keys = cfg.get("flows", {}).get("flow_keys", ["A/D", "Runway"])
    include_flows = cfg.get("flows", {}).get("include", []) or []
    if include_flows:
        df = _apply_flow_filters(df, flow_keys, include_flows)

    seg_cfg = cfg.get("segmentation", {}) or {}
    df = segment_flights(
        df,
        time_gap_sec=float(seg_cfg.get("time_gap_sec", 60)),
        distance_jump_m=float(seg_cfg.get("distance_jump_m", 600)),
        min_points_per_flight=int(seg_cfg.get("min_points_per_flight", 10)),
    )
    logging.info("After segmentation: %d rows", len(df))

    if test_mode and testing_cfg.get("max_flights_per_flow"):
        df = limit_flights_per_flow(
            df,
            max_flights_per_flow=int(testing_cfg["max_flights_per_flow"]),
            flow_keys=flow_keys,
        )
    if test_mode and testing_cfg.get("sample_flows_only") and include_flows:
        df = _apply_flow_filters(df, flow_keys, include_flows)

    preprocessing_cfg = cfg.get("preprocessing", {}) or {}
    smoothing_cfg = preprocessing_cfg.get("smoothing", {})
    resampling_cfg = preprocessing_cfg.get("resampling", {})
    filter_cfg = preprocessing_cfg.get("filter", {}) or {}
    clustering_cfg: Dict[str, object] = cfg.get("clustering", {}) or {}
    method = clustering_cfg.get("method", "optics")

    output_cfg = cfg.get("output", {}) or {}
    output_dir = Path(output_cfg.get("dir", "output"))
    run_dir = output_dir / "run"
    csv_dir = run_dir / "csv"
    plots_dir = run_dir / "figures"
    csv_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    exp_name = str(output_cfg.get("experiment_name", f"{str(method).upper()}_exp_1"))
    logging.info("Using run directory %s (experiment=%s)", run_dir, exp_name)
    logging.info("Rows loaded before segmentation/filtering: %d", len(df))
    if "A/D" in df.columns and "Runway" in df.columns:
        logging.info("Unique flows before segmentation: %d", df[["A/D", "Runway"]].drop_duplicates().shape[0])

    preprocessed = preprocess_flights(
        df,
        smoothing_cfg=smoothing_cfg,
        resampling_cfg=resampling_cfg,
        filter_cfg=filter_cfg,
        use_utm=use_utm,
        flow_keys=flow_keys,
    )
    if preprocessed.empty:
        logging.warning("No preprocessed trajectories available after filtering/segmentation; exiting.")
        return
    if "flight_id" in preprocessed.columns:
        logging.info(
            "After preprocessing: %d rows, %d flights, n_points per flight target=%d",
            len(preprocessed),
            preprocessed["flight_id"].nunique(),
            int(resampling_cfg.get("n_points", 40)),
        )

    if output_cfg.get("save_preprocessed", True) and not preprocessed.empty:
        save_dataframe(preprocessed, csv_dir / f"preprocessed_{exp_name}.csv")

    # Clustering per flow
    n_points = int(resampling_cfg.get("n_points", 40))
    clustered_parts: List[pd.DataFrame] = []

    for flow_vals, flow_df in preprocessed.groupby(flow_keys):
        # flow_vals is scalar if one key, tuple otherwise
        if not isinstance(flow_vals, tuple):
            flow_vals = (flow_vals,)
        flow_label = ", ".join(str(v) for v in flow_vals)
        logging.info("Clustering flow %s with %d flights", flow_label, flow_df["flight_id"].nunique())
        clustered = cluster_flow(
            flow_df,
            method=method,
            cfg=clustering_cfg,
            n_points=n_points,
            max_clusters_per_flow=testing_cfg.get("max_clusters_per_flow") if test_mode else None,
            use_utm=use_utm,
            flow_keys=tuple(flow_keys),
        )
        clustered_parts.append(clustered)

    clustered_df = pd.concat(clustered_parts, ignore_index=True) if clustered_parts else pd.DataFrame()
    if output_cfg.get("save_flight_metadata", True) and not clustered_df.empty:
        save_dataframe(clustered_df, csv_dir / f"clustered_flights_{exp_name}.csv")

    backbone_cfg = cfg.get("backbone", {}) or {}
    backbones = compute_backbones(
        clustered_df,
        percentiles=backbone_cfg.get("percentiles", [10, 50, 90]),
        min_flights_per_cluster=int(backbone_cfg.get("min_flights_per_cluster", 5)),
        use_utm=use_utm,
    )

    if output_cfg.get("save_backbones", True) and not backbones.empty:
        save_dataframe(backbones, csv_dir / f"backbone_tracks_{exp_name}.csv")

    if output_cfg.get("save_plots", False):
        from backbone_tracks.plots import plot_backbone

        for (ad, runway, cluster_id), subset in backbones.groupby(["A/D", "Runway", "cluster_id"]):
                plot_backbone(
                    subset,
                    ad,
                    runway,
                    cluster_id,
                    plots_dir / f"backbone_{ad}_{runway}_{cluster_id}_{exp_name}.png",
                    use_utm=use_utm,
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backbone tracks clustering pipeline.")
    parser.add_argument(
        "-c",
        "--config",
        default="config/backbone.yaml",
        help="Path to YAML config file.",
    )
    args = parser.parse_args()
    main(args.config)
