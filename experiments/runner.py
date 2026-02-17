"""Experiment runner for extended clustering and evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clustering.distances import build_feature_matrix, pairwise_distance_matrix
from clustering.evaluation import compute_internal_metrics
from clustering.registry import get_clusterer


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def filter_flows(df: pd.DataFrame, flow_keys: List[str], include: List[List[str]] | None) -> pd.DataFrame:
    if not include or not flow_keys:
        return df
    include_set = {tuple(item) for item in include}
    mask = df[flow_keys].apply(tuple, axis=1).isin(include_set)
    return df[mask].reset_index(drop=True)


def weighted_mean(values: List[float], weights: List[int]) -> float:
    vals = np.array(values, dtype=float)
    w = np.array(weights, dtype=float)
    return float(np.average(vals, weights=w))


def _latest_preprocessed() -> Path | None:
    for csv_dir in (Path("data") / "preprocessed", Path("output") / "preprocessed"):
        if not csv_dir.exists():
            continue
        candidates = sorted(csv_dir.glob("preprocessed_*.csv"))
        if candidates:
            return candidates[-1]
    return None


def _flow_label(flow_keys: List[str], flow_vals: tuple) -> str:
    if flow_keys:
        return "_".join(str(v) for v in flow_vals)
    return "GLOBAL"


def _ordered_flight_ids(flow_df: pd.DataFrame) -> List[int]:
    """Return deterministic flight_id ordering used by feature generation."""
    return [int(fid) for fid in flow_df.groupby("flight_id", sort=True).size().index.tolist()]


def _infer_effective_n_points(df: pd.DataFrame) -> tuple[int | None, tuple[int, int, int] | None]:
    """
    Infer effective points-per-flight from loaded data.

    Returns:
    - constant_n_points if every flight has the same row count, else None
    - (min, median, max) counts for visibility when variable
    """
    if df.empty or "flight_id" not in df.columns:
        return None, None
    counts = df.groupby("flight_id").size()
    if counts.empty:
        return None, None
    min_pts = int(counts.min())
    med_pts = int(counts.median())
    max_pts = int(counts.max())
    if min_pts == max_pts:
        return min_pts, (min_pts, med_pts, max_pts)
    return None, (min_pts, med_pts, max_pts)


def _build_labeled_flights(
    flow_df: pd.DataFrame,
    flow_keys: List[str],
    flow_label: str,
    ordered_flight_ids: List[int],
    labels: np.ndarray,
) -> pd.DataFrame:
    """
    Build one-row-per-flight labels table aligned to ordered_flight_ids.
    Raises ValueError if label-to-flight alignment checks fail.
    """
    if len(labels) != len(ordered_flight_ids):
        raise ValueError(
            f"Label count mismatch: labels={len(labels)} flights={len(ordered_flight_ids)}"
        )

    meta_cols = ["flight_id"]
    for col in (*flow_keys, "A/D", "Runway", "icao24", "callsign"):
        if col in flow_df.columns and col not in meta_cols:
            meta_cols.append(col)
    for col in flow_df.columns:
        if "aircraft_type" in col and col not in meta_cols:
            meta_cols.append(col)

    flight_meta = (
        flow_df[meta_cols]
        .sort_values(["flight_id"])
        .drop_duplicates(subset=["flight_id"], keep="first")
        .set_index("flight_id")
    )
    labeled = flight_meta.reindex(ordered_flight_ids).reset_index()
    if labeled["flight_id"].isna().any():
        raise ValueError("Missing metadata rows after reindexing labeled flights.")
    if len(labeled) != len(ordered_flight_ids):
        raise ValueError(
            f"Labeled row mismatch: labeled={len(labeled)} flights={len(ordered_flight_ids)}"
        )

    labeled["cluster_id"] = labels
    labeled["flow_label"] = flow_label
    return labeled


def _cluster_counts_by_flow(df_lab_all: pd.DataFrame) -> pd.DataFrame:
    """Return per-flow cluster totals with a stable output schema."""
    df_work = df_lab_all.copy()
    for col in ("A/D", "Runway"):
        if col not in df_work.columns:
            df_work[col] = pd.NA

    counts = (
        df_work.groupby(["flow_label", "A/D", "Runway", "cluster_id"], dropna=False)
        .size()
        .reset_index(name="n_flights")
        .sort_values(["flow_label", "cluster_id", "n_flights"], ascending=[True, True, False])
    )
    counts["is_noise_cluster"] = counts["cluster_id"] == -1
    return counts


def run_experiment(cfg_path: Path, preprocessed_override: Path | None = None) -> None:
    cfg = load_config(cfg_path)
    clustering_cfg: Dict[str, object] = cfg.get("clustering", {}) or {}
    eval_cfg: Dict[str, object] = clustering_cfg.get("evaluation", {}) or {}
    method = clustering_cfg.get("method", "optics")
    distance_metric = clustering_cfg.get("distance_metric", "euclidean")
    distance_params = clustering_cfg.get("distance_params", {}) or {}
    experiment_name = cfg.get("output", {}).get("experiment_name", "experiment")

    flows_cfg = cfg.get("flows", {}) or {}
    flow_keys = list(flows_cfg.get("flow_keys") or [])
    include_flows = flows_cfg.get("include", []) or []

    input_cfg = cfg.get("input", {}) or {}
    preprocessed_csv = preprocessed_override or input_cfg.get("preprocessed_csv")
    if preprocessed_csv is None:
        latest = _latest_preprocessed()
        if latest is None:
            raise FileNotFoundError(
                "No preprocessed CSV specified and none found under data/preprocessed. "
                "Set input.preprocessed_csv in the config or pass --preprocessed."
            )
        preprocessed_csv = latest
    df = pd.read_csv(preprocessed_csv)
    df = filter_flows(df, flow_keys, include_flows)

    output_dir = Path(cfg.get("output", {}).get("dir", "output")) / "experiments" / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare outputs
    metrics_rows: List[dict] = []
    label_paths: List[Path] = []
    log_lines: List[str] = []

    # Build experiment log header (human-readable).
    run_start = datetime.now(timezone.utc)
    log_lines.append(f"Experiment: {experiment_name}")
    log_lines.append(f"Config: {cfg_path}")
    log_lines.append(f"Run started (UTC): {run_start.isoformat()}")
    log_lines.append(f"Method: {method}")
    log_lines.append(f"Distance metric: {distance_metric}")
    log_lines.append(f"Preprocessed CSV: {preprocessed_csv}")
    vector_cols = cfg.get("features", {}).get("vector_cols", ["x_utm", "y_utm"])
    if any(col in {"latitude", "longitude"} for col in vector_cols):
        raise ValueError("Vector columns must use UTM coordinates only (x_utm, y_utm).")
    log_lines.append(f"Vector columns: {vector_cols}")
    if distance_metric in {"dtw", "frechet"} and distance_params:
        log_lines.append(f"Distance params: {distance_params}")
    configured_n_points = cfg.get("preprocessing", {}).get("resampling", {}).get("n_points")
    effective_n_points, effective_stats = _infer_effective_n_points(df)
    if effective_n_points is not None:
        log_lines.append(f"Resampling n_points: {effective_n_points}")
    elif effective_stats is not None:
        min_pts, med_pts, max_pts = effective_stats
        log_lines.append(
            f"Resampling n_points: variable (per_flight min/med/max={min_pts}/{med_pts}/{max_pts})"
        )
    elif configured_n_points is not None:
        log_lines.append(f"Resampling n_points: {configured_n_points}")
    if configured_n_points is not None and effective_n_points is not None and configured_n_points != effective_n_points:
        log_lines.append(
            f"Resampling n_points note: config={configured_n_points} but effective={effective_n_points} from input data"
        )
    log_lines.append(f"Flow keys: {flow_keys if flow_keys else ['GLOBAL']}")

    method_params = clustering_cfg.get(method, {}) or {}
    if method == "two_stage":
        stage1 = method_params.get("stage1_method")
        stage2 = method_params.get("stage2_method")
        stage1_params = method_params.get("stage1_params", {})
        stage2_params = method_params.get("stage2_params", {})
        log_lines.append(f"Stage1: {stage1} params={stage1_params}")
        log_lines.append(f"Stage2: {stage2} params={stage2_params}")
    else:
        log_lines.append(f"Params: {method_params}")
    log_lines.append("")

    # Per-flow clustering (or global if no flow_keys provided)
    if flow_keys:
        flow_iter = df.groupby(flow_keys)
    else:
        flow_iter = [(("ALL",), df)]

    for flow_vals, flow_df in flow_iter:
        if not isinstance(flow_vals, tuple):
            flow_vals = (flow_vals,)
        flow_name = "_".join(str(v) for v in flow_vals) if flow_keys else "ALL"
        flow_label = _flow_label(flow_keys, flow_vals)

        # Build features/trajectories and lock deterministic flight order.
        vector_cols = cfg.get("features", {}).get("vector_cols", ["x_utm", "y_utm"])
        ordered_flight_ids = _ordered_flight_ids(flow_df)
        print(
            f"[flow] {experiment_name} {flow_label} start flights={len(ordered_flight_ids)}",
            flush=True,
        )
        try:
            X, trajs = build_feature_matrix(flow_df, vector_cols=vector_cols)
        except Exception as exc:  # pragma: no cover - defensive re-raise
            raise RuntimeError(f"Flow {flow_label} failed during feature construction: {exc}") from exc
        if X.shape[0] != len(ordered_flight_ids):
            raise ValueError(
                f"Feature/flight mismatch in {flow_label}: features={X.shape[0]} flights={len(ordered_flight_ids)}"
            )
        if X.size:
            sample_idx = 0
            sample_flight_id = ordered_flight_ids[sample_idx] if ordered_flight_ids else None
            sample_vec = X[sample_idx]
            sample_vec_str = np.array2string(
                sample_vec[: min(12, sample_vec.shape[0])],
                precision=3,
                separator=", ",
                threshold=12,
            )
            log_lines.append(
                f"  Input vector sample (flight_id={sample_flight_id}, len={sample_vec.shape[0]}): {sample_vec_str}"
            )
            if distance_metric in {"dtw", "frechet"} and trajs:
                sample_traj = trajs[sample_idx]
                log_lines.append(f"  Trajectory sample shape (for {distance_metric}): {sample_traj.shape}")

        clusterer = get_clusterer(method)
        precomputed_needed = distance_metric in {"dtw", "frechet"}
        if precomputed_needed and not clusterer.supports_precomputed:
            raise ValueError(f"{method} does not support precomputed distances ({distance_metric}).")

        if precomputed_needed:
            try:
                params = {
                    "distance_metric": distance_metric,
                    "n_points": cfg.get("preprocessing", {}).get("resampling", {}).get("n_points"),
                }
                params.update(distance_params)
                D = pairwise_distance_matrix(
                    trajs if distance_metric in {"dtw", "frechet"} else X,
                    metric=distance_metric,
                    cache_dir=output_dir / "cache",
                    flow_name=flow_name,
                    params=params,
                )
                if distance_metric in {"dtw", "frechet"} and method == "optics" and hasattr(D, "nnz"):
                    log_lines.append(
                        "  Note: sparse precomputed distances used; OPTICS may treat missing edges as 0. "
                        "Consider HDBSCAN for sparse DTW/Frechet."
                    )
                labels = clusterer.fit_predict(D, metric="precomputed")
                metrics = compute_internal_metrics(
                    D,
                    labels,
                    metric_mode="precomputed",
                    include_noise=eval_cfg.get("include_noise", False),
                )
            except Exception as exc:
                raise RuntimeError(f"Flow {flow_label} failed during clustering: {exc}") from exc
        else:
            try:
                labels = clusterer.fit_predict(
                    X,
                    **(clustering_cfg.get(method, {}) or {}),
                )
                metrics = compute_internal_metrics(
                    X,
                    labels,
                    metric_mode="features",
                    include_noise=eval_cfg.get("include_noise", False),
                )
                if method == "gmm" and getattr(clusterer, "last_model", None) is not None:
                    model = clusterer.last_model
                    metrics["bic"] = float(model.bic(X))
                    metrics["aic"] = float(model.aic(X))
            except Exception as exc:
                raise RuntimeError(f"Flow {flow_label} failed during clustering: {exc}") from exc

        labels = np.asarray(labels)
        if len(labels) != len(ordered_flight_ids):
            raise ValueError(
                f"Label count mismatch in {flow_label}: labels={len(labels)} flights={len(ordered_flight_ids)}"
            )

        # Log per-flow details.
        flight_counts = flow_df.groupby("flight_id").size()
        n_points_min = int(flight_counts.min()) if not flight_counts.empty else 0
        n_points_med = int(flight_counts.median()) if not flight_counts.empty else 0
        n_points_max = int(flight_counts.max()) if not flight_counts.empty else 0

        log_lines.append(f"Flow: {flow_label}")
        log_lines.append(f"  Flights: {len(labels)}")
        log_lines.append(
            f"  Points: total={len(flow_df)} per_flight[min/med/max]={n_points_min}/{n_points_med}/{n_points_max}"
        )
        log_lines.append(
            "  Metrics: "
            f"n_clusters={metrics.get('n_clusters')} "
            f"noise_frac={metrics.get('noise_frac')} "
            f"n_noise_flights={metrics.get('n_noise_flights')} "
            f"n_clustered_flights={metrics.get('n_clustered_flights')}"
        )
        log_lines.append(
            f"  Metrics: silhouette={metrics.get('silhouette')} "
            f"davies_bouldin={metrics.get('davies_bouldin')} "
            f"calinski_harabasz={metrics.get('calinski_harabasz')}"
        )
        if metrics.get("reason"):
            log_lines.append(f"  Metrics: reason={metrics.get('reason')}")

        counts = pd.Series(labels).value_counts().sort_index()
        counts_str = ", ".join(f"{int(k)}={int(v)}" for k, v in counts.items())
        cluster_ids_str = ", ".join(str(int(k)) for k in counts.index.tolist())
        log_lines.append(f"  Cluster IDs: [{cluster_ids_str}]")
        log_lines.append(f"  Cluster counts: {counts_str}")

        if flow_keys:
            metrics.update({key: val for key, val in zip(flow_keys, flow_vals)})
        else:
            metrics["flow"] = "ALL"
        metrics["n_flights"] = len(labels)
        metrics_rows.append(metrics)

        try:
            labeled = _build_labeled_flights(
                flow_df=flow_df,
                flow_keys=flow_keys,
                flow_label=flow_label,
                ordered_flight_ids=ordered_flight_ids,
                labels=labels,
            )
        except Exception as exc:
            raise RuntimeError(f"Flow {flow_label} failed during label assembly: {exc}") from exc
        label_check_ok = len(labeled) == len(ordered_flight_ids)
        log_lines.append(
            f"  Label rows check: expected={len(ordered_flight_ids)} actual={len(labeled)} ok={label_check_ok}"
        )
        label_path = output_dir / f"labels_{flow_name}.csv"
        labeled.to_csv(label_path, index=False)
        label_paths.append(label_path)
        log_lines.append(f"  Labels: {label_path}")
        log_lines.append("")
        print(
            f"[flow] {experiment_name} {flow_label} done "
            f"n_clusters={metrics.get('n_clusters')} noise_frac={metrics.get('noise_frac')}",
            flush=True,
        )

    # Cluster/runway breakdown (helpful when clustering across all flows)
    labels_all = []
    for p in label_paths:
        df_lab = pd.read_csv(p)
        labels_all.append(df_lab)
    if labels_all:
        df_lab_all = pd.concat(labels_all, ignore_index=True)
        df_lab_all.to_csv(output_dir / "labels_ALL.csv", index=False)
        counts_by_flow = _cluster_counts_by_flow(df_lab_all)
        counts_by_flow.to_csv(output_dir / "cluster_counts_by_flow.csv", index=False)

        # Integrity check: per-flow totals in counts file must match labels rows.
        flow_totals_from_labels = df_lab_all.groupby("flow_label").size().sort_index()
        flow_totals_from_counts = counts_by_flow.groupby("flow_label")["n_flights"].sum().sort_index()
        if not flow_totals_from_labels.equals(flow_totals_from_counts):
            raise ValueError(
                "Cluster totals mismatch between labels_ALL.csv and cluster_counts_by_flow.csv."
            )

        if "Runway" in df_lab_all.columns:
            counts = (
                df_lab_all.groupby(["flow_label", "cluster_id", "Runway"])
                .size()
                .reset_index(name="n_flights")
                .sort_values(["flow_label", "cluster_id", "n_flights"], ascending=[True, True, False])
            )
            counts.to_csv(output_dir / "cluster_runway_counts.csv", index=False)
        if "aircraft_type_match" in df_lab_all.columns:
            counts = (
                df_lab_all.groupby(["flow_label", "cluster_id", "aircraft_type_match"])
                .size()
                .reset_index(name="n_flights")
                .sort_values(["flow_label", "cluster_id", "n_flights"], ascending=[True, True, False])
            )
            counts.to_csv(output_dir / "cluster_aircraft_type_counts.csv", index=False)

    # Aggregate metrics globally (weighted by flights)
    if metrics_rows:
        df_metrics = pd.DataFrame(metrics_rows)
        df_metrics.to_csv(output_dir / "metrics_by_flow.csv", index=False)
        weight = df_metrics["n_flights"]
        agg = {"total_flights": int(weight.sum())}
        if "n_noise_flights" in df_metrics.columns:
            agg["total_noise_flights"] = int(df_metrics["n_noise_flights"].fillna(0).sum())
            agg["total_clustered_flights"] = int(df_metrics["n_clustered_flights"].fillna(0).sum())
            agg["noise_frac"] = (
                float(agg["total_noise_flights"] / agg["total_flights"]) if agg["total_flights"] else 0.0
            )
        for metric_name in ("davies_bouldin", "silhouette", "calinski_harabasz"):
            if metric_name not in df_metrics:
                agg[metric_name] = np.nan
                continue
            valid = df_metrics[metric_name].notna()
            if not valid.any():
                agg[metric_name] = np.nan
                continue
            agg[metric_name] = weighted_mean(
                df_metrics.loc[valid, metric_name].tolist(),
                df_metrics.loc[valid, "n_flights"].tolist(),
            )
        pd.DataFrame([agg]).to_csv(output_dir / "metrics_global.csv", index=False)

    # Append run completion metadata
    run_end = datetime.now(timezone.utc)
    elapsed = (run_end - run_start).total_seconds()
    log_lines.append(f"Run finished (UTC): {run_end.isoformat()}")
    log_lines.append(f"Elapsed seconds: {elapsed:.1f}")

    # Save resolved config
    (output_dir / "config_resolved.yaml").write_text(yaml.dump(cfg), encoding="utf-8")
    log_path = output_dir / "experiment_log.txt"
    log_path.write_text("\n".join(log_lines), encoding="utf-8")
    logs_dir = Path("logs") / "experiments"
    logs_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / f"{experiment_name}.log").write_text("\n".join(log_lines), encoding="utf-8")
    (output_dir / "runtime_log.txt").write_text(
        json.dumps({"labels": [str(p) for p in label_paths], "experiment_log": str(log_path)}, indent=2),
        encoding="utf-8",
    )
    print(f"[run] {experiment_name} finished output={output_dir}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run extended clustering experiment.")
    parser.add_argument("-c", "--config", type=Path, required=True, help="Path to YAML config.")
    parser.add_argument(
        "--preprocessed",
        type=Path,
        default=None,
        help="Override path to preprocessed CSV. If omitted, uses config input.preprocessed_csv or latest preprocessed_*.csv in data/preprocessed.",
    )
    args = parser.parse_args()
    run_experiment(args.config, preprocessed_override=args.preprocessed)
