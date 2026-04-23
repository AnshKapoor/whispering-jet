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
from scipy.sparse import issparse
from scipy.sparse.csgraph import connected_components
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

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


def _format_vector_preview(vec: np.ndarray | None, max_len: int = 12) -> str | None:
    """Return a compact 1D preview string for log output."""
    if vec is None:
        return None
    arr = np.asarray(vec, dtype=float).reshape(-1)
    if arr.size == 0:
        return "[]"
    return np.array2string(
        arr[: min(max_len, arr.size)],
        precision=3,
        separator=", ",
        threshold=max_len,
    )


def _format_grouped_vector_preview(
    vec: np.ndarray | None,
    dim_labels: List[str],
    max_steps: int = 4,
) -> str | None:
    """Return a compact grouped preview like [(x=..., y=..., z=...), ...]."""
    if vec is None:
        return None
    n_dims = len(dim_labels)
    if n_dims <= 0:
        return None
    arr = np.asarray(vec, dtype=float).reshape(-1)
    if arr.size == 0 or (arr.size % n_dims) != 0:
        return None
    steps = arr.reshape(-1, n_dims)
    parts: List[str] = []
    for row in steps[: max_steps]:
        row_text = ", ".join(f"{label}={value:.3f}" for label, value in zip(dim_labels, row))
        parts.append(f"({row_text})")
    suffix = " ..." if steps.shape[0] > max_steps else ""
    return "[" + ", ".join(parts) + suffix + "]"


def _classical_mds_embedding(D: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Return deterministic classical-MDS embedding from a dense distance matrix."""

    D = np.asarray(D, dtype=float)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("Classical MDS expects a square dense distance matrix.")
    n = D.shape[0]
    n_components = max(1, int(n_components))
    if n == 0:
        return np.zeros((0, n_components), dtype=float)
    if n == 1:
        return np.zeros((1, n_components), dtype=float)

    np.fill_diagonal(D, 0.0)
    J = np.eye(n) - np.ones((n, n), dtype=float) / n
    B = -0.5 * J @ (D ** 2) @ J
    eigvals, eigvecs = np.linalg.eigh(B)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    positive = eigvals > 1e-12
    keep = min(int(np.sum(positive)), n_components)
    emb = np.zeros((n, n_components), dtype=float)
    if keep > 0:
        vals = np.sqrt(eigvals[:keep])
        emb[:, :keep] = eigvecs[:, :keep] * vals
    return emb


def _infer_effective_n_points(
    df: pd.DataFrame,
    flow_keys: List[str] | None = None,
) -> tuple[int | None, tuple[int, int, int] | None]:
    """
    Infer effective points-per-flight from loaded data.

    Returns:
    - constant_n_points if every flight has the same row count, else None
    - (min, median, max) counts for visibility when variable
    """
    if df.empty or "flight_id" not in df.columns:
        return None, None
    group_cols: List[str] = []
    for col in (flow_keys or []):
        if col in df.columns:
            group_cols.append(col)
    group_cols.append("flight_id")
    counts = df.groupby(group_cols).size()
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


def _fit_hdbscan_by_connected_components(clusterer, D, cluster_params: dict | None = None) -> tuple[np.ndarray, int]:
    """
    Run HDBSCAN per connected component for sparse precomputed distance graphs.
    Returns merged labels and number of connected components.
    """
    n_components, comp_labels = connected_components(D, directed=False, connection="weak")
    labels_merged = np.full(D.shape[0], -1, dtype=int)
    next_cluster_id = 0

    for comp_id in range(n_components):
        idx = np.where(comp_labels == comp_id)[0]
        if idx.size <= 1:
            continue
        D_sub = D[idx][:, idx]
        params = dict(cluster_params or {})
        labels_sub = np.asarray(clusterer.fit_predict(D_sub, metric="precomputed", **params), dtype=int)
        unique_sub = [int(c) for c in np.unique(labels_sub) if c != -1]
        remap = {c: (next_cluster_id + i) for i, c in enumerate(unique_sub)}
        for local_pos, lbl in enumerate(labels_sub):
            if lbl == -1:
                continue
            labels_merged[idx[local_pos]] = remap[int(lbl)]
        next_cluster_id += len(unique_sub)
    return labels_merged, n_components


def _sanitize_precomputed_dense(D: np.ndarray) -> np.ndarray:
    """
    Enforce dense precomputed-matrix invariants expected by sklearn/hdbscan:
    - finite values
    - symmetric matrix
    - strictly zero diagonal
    """

    D = np.asarray(D, dtype=float)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("Precomputed distance matrix must be square.")
    if not np.isfinite(D).all():
        n_bad = int((~np.isfinite(D)).sum())
        raise ValueError(f"Precomputed distance matrix contains non-finite values (count={n_bad}).")
    D = 0.5 * (D + D.T)
    np.fill_diagonal(D, 0.0)
    return D


def _resolve_kmeans_n_clusters_by_ad(
    cluster_params: dict,
    flow_df_fit: pd.DataFrame,
) -> tuple[dict, str | None]:
    """
    Resolve optional KMeans n_clusters override by A/D value.

    Expected config shape:
      kmeans:
        n_clusters: 3
        n_clusters_by_ad:
          Landung: 2
          Start: 3
    """
    params = dict(cluster_params or {})
    mapping = params.pop("n_clusters_by_ad", None)
    if not isinstance(mapping, dict):
        return params, None

    if "A/D" not in flow_df_fit.columns or flow_df_fit.empty:
        return params, None
    ad_val = str(flow_df_fit["A/D"].iloc[0]).strip()

    chosen = None
    if ad_val in mapping:
        chosen = mapping[ad_val]
    else:
        # Case-insensitive fallback for robustness.
        lower_map = {str(k).strip().lower(): v for k, v in mapping.items()}
        chosen = lower_map.get(ad_val.lower())

    if chosen is None:
        return params, None
    params["n_clusters"] = int(chosen)
    return params, ad_val


def _apply_feature_transform(
    X: np.ndarray,
    features_cfg: Dict[str, object] | None,
    log_lines: List[str],
) -> tuple[np.ndarray, dict]:
    """
    Optionally standardize and project feature vectors before clustering.

    Expected config:
      features:
        vector_cols: ["x_utm", "y_utm", "altitude"]
        transform:
          standardize: true
          pca_components: 5
          impute_strategy: interpolate
    """
    features_cfg = features_cfg or {}
    transform_cfg = features_cfg.get("transform", {}) or {}
    if not transform_cfg:
        return X, {}

    if isinstance(X, np.ndarray) and X.dtype == object:
        raise ValueError("Feature transforms require fixed-length dense feature vectors, not ragged arrays.")

    X_work = np.asarray(X, dtype=float)
    n_samples, n_features = X_work.shape if X_work.ndim == 2 else (0, 0)
    meta: dict = {"feature_dim_in": int(n_features)}

    n_nan = int(np.isnan(X_work).sum()) if X_work.size else 0
    meta["n_missing_features"] = n_nan
    if n_nan > 0:
        impute_strategy = str(transform_cfg.get("impute_strategy", "interpolate")).strip().lower()
        if impute_strategy not in {"interpolate", "linear", "median", "mean"}:
            raise ValueError(f"Unsupported impute_strategy: {impute_strategy}")
        vector_cols = list(features_cfg.get("vector_cols", []) or [])
        fallback_missing = 0
        if impute_strategy in {"interpolate", "linear"} and vector_cols:
            n_dims = len(vector_cols)
            if n_dims <= 0 or (n_features % n_dims) != 0:
                raise ValueError(
                    "Interpolation imputation requires feature length divisible by len(vector_cols)."
                )
            n_steps = int(n_features // n_dims)
            X_seq = X_work.reshape(n_samples, n_steps, n_dims).copy()
            step_idx = np.arange(n_steps, dtype=float)
            for i in range(n_samples):
                for d in range(n_dims):
                    vals = X_seq[i, :, d]
                    mask = np.isfinite(vals)
                    if mask.all():
                        continue
                    valid_count = int(mask.sum())
                    if valid_count == 0:
                        fallback_missing += n_steps
                        continue
                    if valid_count == 1:
                        vals[:] = vals[mask][0]
                    else:
                        vals[:] = np.interp(step_idx, step_idx[mask], vals[mask])
                    X_seq[i, :, d] = vals
            X_work = X_seq.reshape(n_samples, n_features)
            remaining_nan = int(np.isnan(X_work).sum())
            if remaining_nan > 0:
                fallback_strategy = str(transform_cfg.get("fallback_impute_strategy", "median")).strip().lower()
                if fallback_strategy not in {"median", "mean"}:
                    raise ValueError(f"Unsupported fallback_impute_strategy: {fallback_strategy}")
                imputer = SimpleImputer(strategy=fallback_strategy)
                X_work = imputer.fit_transform(X_work)
                log_lines.append(
                    "  Feature transform: missing values interpolated along step index "
                    f"(count={n_nan}); fallback imputation applied to remaining={remaining_nan} "
                    f"using {fallback_strategy}."
                )
            else:
                log_lines.append(
                    "  Feature transform: missing values interpolated along step index "
                    f"(count={n_nan})."
                )
        else:
            imputer = SimpleImputer(strategy=impute_strategy)
            X_work = imputer.fit_transform(X_work)
            log_lines.append(
                "  Feature transform: missing values imputed "
                f"(count={n_nan}, strategy={impute_strategy})."
            )
        meta["n_missing_features_fallback"] = int(fallback_missing)

    if n_samples > 0:
        meta["sample_after_impute"] = np.asarray(X_work[0], dtype=float).copy()

    if bool(transform_cfg.get("standardize", False)):
        scaler = StandardScaler()
        X_work = scaler.fit_transform(X_work)
        meta["standardized"] = True
        if n_samples > 0:
            meta["sample_after_standardize"] = np.asarray(X_work[0], dtype=float).copy()
        log_lines.append("  Feature transform: StandardScaler applied.")
    else:
        meta["standardized"] = False

    pca_components = transform_cfg.get("pca_components")
    if pca_components is not None:
        n_comp = max(1, int(pca_components))
        n_comp_eff = min(n_comp, n_samples, n_features)
        pca = PCA(n_components=n_comp_eff, svd_solver="full")
        X_work = pca.fit_transform(X_work)
        meta["pca_components"] = int(n_comp_eff)
        meta["pca_explained_variance_sum"] = float(np.sum(pca.explained_variance_ratio_))
        if n_samples > 0:
            meta["sample_after_pca"] = np.asarray(X_work[0], dtype=float).copy()
        log_lines.append(
            "  Feature transform: PCA applied "
            f"(n_components={n_comp_eff}, explained_variance_sum={meta['pca_explained_variance_sum']:.4f})."
        )
    else:
        meta["pca_components"] = 0
        meta["pca_explained_variance_sum"] = np.nan

    meta["feature_dim_out"] = int(X_work.shape[1]) if X_work.ndim == 2 else 0
    return X_work, meta


def run_experiment(cfg_path: Path, preprocessed_override: Path | None = None) -> None:
    cfg = load_config(cfg_path)
    clustering_cfg: Dict[str, object] = cfg.get("clustering", {}) or {}
    eval_cfg: Dict[str, object] = clustering_cfg.get("evaluation", {}) or {}
    method = clustering_cfg.get("method", "optics")
    distance_metric = clustering_cfg.get("distance_metric", "euclidean")
    distance_params = clustering_cfg.get("distance_params", {}) or {}
    sample_cfg: Dict[str, object] = clustering_cfg.get("sample_for_fit", {}) or {}
    sample_enabled = bool(sample_cfg.get("enabled", False))
    sample_max_flights = int(sample_cfg.get("max_flights_per_flow", 1200))
    sample_random_state = int(sample_cfg.get("random_state", 11))
    sample_mode = str(sample_cfg.get("mode", "sample_only")).strip().lower()
    if distance_metric == "lcss" and not sample_enabled:
        # Keep LCSS runs feasible by default even when config omits sampling.
        sample_enabled = True
        sample_max_flights = 1200
        sample_random_state = 11
        sample_mode = "sample_only"
    if sample_enabled:
        if sample_mode != "sample_only":
            raise ValueError("clustering.sample_for_fit.mode currently supports only 'sample_only'.")
        if sample_max_flights <= 0:
            raise ValueError("clustering.sample_for_fit.max_flights_per_flow must be > 0.")
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
    df = pd.read_csv(preprocessed_csv, low_memory=False)
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
    if distance_metric in {"dtw", "frechet", "lcss", "euclidean_weighted"} and distance_params:
        log_lines.append(f"Distance params: {distance_params}")
    if sample_enabled:
        log_lines.append(
            "Sample-for-fit: "
            f"enabled mode={sample_mode} max_flights_per_flow={sample_max_flights} random_state={sample_random_state}"
        )
    else:
        log_lines.append("Sample-for-fit: disabled")
    configured_n_points = cfg.get("preprocessing", {}).get("resampling", {}).get("n_points")
    effective_n_points, effective_stats = _infer_effective_n_points(df, flow_keys=flow_keys)
    if effective_n_points is not None:
        log_lines.append(f"Resampling n_points (effective from input rows): {effective_n_points}")
    elif effective_stats is not None:
        min_pts, med_pts, max_pts = effective_stats
        log_lines.append(
            f"Resampling n_points (effective from input rows): variable "
            f"(per_flight min/med/max={min_pts}/{med_pts}/{max_pts})"
        )
    elif configured_n_points is not None:
        log_lines.append(f"Resampling n_points (from config fallback): {configured_n_points}")
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

        # Optionally sample flights for model fit (sample-only semantics).
        total_flight_ids = _ordered_flight_ids(flow_df)
        flow_df_fit = flow_df
        ordered_flight_ids = total_flight_ids
        if sample_enabled and len(total_flight_ids) > sample_max_flights:
            rng = np.random.default_rng(sample_random_state)
            sampled_ids = rng.choice(
                np.array(total_flight_ids, dtype=int),
                size=sample_max_flights,
                replace=False,
            )
            sampled_set = set(int(fid) for fid in sampled_ids.tolist())
            flow_df_fit = flow_df[flow_df["flight_id"].isin(sampled_set)].copy()
            ordered_flight_ids = _ordered_flight_ids(flow_df_fit)
        n_flights_total_flow = len(total_flight_ids)
        n_flights_used_for_fit = len(ordered_flight_ids)
        fit_sampling_mode = sample_mode if sample_enabled else "none"

        # Build features/trajectories and lock deterministic flight order.
        features_cfg = cfg.get("features", {}) or {}
        vector_cols = features_cfg.get("vector_cols", ["x_utm", "y_utm"])
        print(
            f"[flow] {experiment_name} {flow_label} start flights={n_flights_used_for_fit}"
            f" total={n_flights_total_flow} mode={fit_sampling_mode}",
            flush=True,
        )
        try:
            X, trajs = build_feature_matrix(
                flow_df_fit,
                vector_cols=vector_cols,
                allow_ragged=distance_metric in {"dtw", "frechet", "lcss"},
            )
        except Exception as exc:  # pragma: no cover - defensive re-raise
            raise RuntimeError(f"Flow {flow_label} failed during feature construction: {exc}") from exc
        if X.shape[0] != len(ordered_flight_ids):
            raise ValueError(
                f"Feature/flight mismatch in {flow_label}: features={X.shape[0]} flights={len(ordered_flight_ids)}"
            )
        raw_sample_vec = None
        raw_sample_len = None
        raw_sample_points_from_vec = None
        if X.size:
            raw_sample_vec = np.asarray(X[0], dtype=float)
            raw_sample_len = int(raw_sample_vec.shape[0]) if raw_sample_vec.ndim > 0 else 1
            if len(vector_cols) > 0 and raw_sample_len % len(vector_cols) == 0:
                raw_sample_points_from_vec = int(raw_sample_len // len(vector_cols))
        precomputed_needed = distance_metric in {"dtw", "frechet", "lcss", "euclidean_weighted"}
        feature_transform_meta: dict = {}
        if not precomputed_needed:
            try:
                X, feature_transform_meta = _apply_feature_transform(X, features_cfg, log_lines)
            except Exception as exc:
                raise RuntimeError(f"Flow {flow_label} failed during feature transform: {exc}") from exc
        if X.size:
            sample_idx = 0
            sample_flight_id = ordered_flight_ids[sample_idx] if ordered_flight_ids else None
            sample_vec = np.asarray(X[sample_idx], dtype=float)
            sample_len = int(sample_vec.shape[0]) if sample_vec.ndim > 0 else 1
            raw_vec_str = _format_vector_preview(raw_sample_vec)
            if raw_vec_str is not None:
                log_lines.append(
                    f"  Raw flattened sample (flight_id={sample_flight_id}, len={raw_sample_len}): {raw_vec_str}"
                )
            raw_grouped_str = _format_grouped_vector_preview(raw_sample_vec, vector_cols)
            if raw_grouped_str is not None:
                log_lines.append(f"  Raw grouped sample by step: {raw_grouped_str}")
            if raw_sample_len is not None and raw_sample_points_from_vec is not None:
                log_lines.append(
                    "  Input vector interpretation: "
                    f"raw_len={raw_sample_len} = {raw_sample_points_from_vec} points x {len(vector_cols)} dims"
                )
            elif raw_sample_len is not None:
                log_lines.append(
                    f"  Input vector interpretation: raw_len={raw_sample_len} (not divisible by dims={len(vector_cols)})"
                )
            sample_after_impute = feature_transform_meta.get("sample_after_impute")
            if sample_after_impute is not None:
                imputed_vec_str = _format_vector_preview(sample_after_impute)
                if imputed_vec_str is not None:
                    log_lines.append(f"  Post-imputation sample: {imputed_vec_str}")
                imputed_grouped_str = _format_grouped_vector_preview(sample_after_impute, vector_cols)
                if imputed_grouped_str is not None and feature_transform_meta.get("n_missing_features", 0) > 0:
                    log_lines.append(f"  Post-imputation grouped sample: {imputed_grouped_str}")
            sample_after_standardize = feature_transform_meta.get("sample_after_standardize")
            if sample_after_standardize is not None:
                standardized_vec_str = _format_vector_preview(sample_after_standardize)
                if standardized_vec_str is not None:
                    log_lines.append(f"  Standardized sample: {standardized_vec_str}")
                standardized_grouped_str = _format_grouped_vector_preview(sample_after_standardize, vector_cols)
                if standardized_grouped_str is not None:
                    log_lines.append(f"  Standardized grouped sample: {standardized_grouped_str}")
            sample_after_pca = feature_transform_meta.get("sample_after_pca")
            if sample_after_pca is not None:
                pca_vec_str = _format_vector_preview(sample_after_pca)
                if pca_vec_str is not None:
                    log_lines.append(f"  PCA sample (first components): {pca_vec_str}")
            final_vec_str = _format_vector_preview(sample_vec)
            if final_vec_str is not None:
                log_lines.append(
                    f"  Clustering vector sample (flight_id={sample_flight_id}, len={sample_len}): {final_vec_str}"
                )
            if distance_metric in {"dtw", "frechet", "lcss"} and trajs:
                sample_traj = trajs[sample_idx]
                log_lines.append(f"  Trajectory sample shape (for {distance_metric}): {sample_traj.shape}")

        clusterer = get_clusterer(method)
        precomputed_kmeans_mode = bool(precomputed_needed and method == "kmeans")
        cluster_params = dict(clustering_cfg.get(method, {}) or {})
        if method == "kmeans":
            cluster_params, ad_override = _resolve_kmeans_n_clusters_by_ad(cluster_params, flow_df_fit)
            if ad_override is not None:
                log_lines.append(
                    f"  KMeans n_clusters override: A/D={ad_override} -> n_clusters={cluster_params.get('n_clusters')}"
                )
        if precomputed_needed and not clusterer.supports_precomputed and not precomputed_kmeans_mode:
            raise ValueError(f"{method} does not support precomputed distances ({distance_metric}).")

        if precomputed_needed:
            try:
                params = {
                    "distance_metric": distance_metric,
                    "n_points": cfg.get("preprocessing", {}).get("resampling", {}).get("n_points"),
                }
                params.update(distance_params)
                if method == "optics":
                    min_req = int((clustering_cfg.get("optics", {}) or {}).get("min_samples", 5))
                    params["min_required_neighbors"] = max(1, min_req)
                D = pairwise_distance_matrix(
                    trajs if distance_metric in {"dtw", "frechet", "lcss"} else X,
                    metric=distance_metric,
                    cache_dir=output_dir / "cache",
                    flow_name=flow_name,
                    params=params,
                    cache_ids=ordered_flight_ids,
                )
                if not issparse(D):
                    D = _sanitize_precomputed_dense(D)
                if distance_metric in {"dtw", "frechet"} and method == "optics" and hasattr(D, "nnz"):
                    log_lines.append(
                        "  Note: sparse precomputed distances used; OPTICS may treat missing edges as 0. "
                        "Consider HDBSCAN for sparse DTW/Frechet."
                    )
                if precomputed_needed and method == "kmeans":
                    mds_n_components = int(distance_params.get("mds_n_components", 3))
                    if issparse(D):
                        if distance_metric in {"dtw", "frechet"}:
                            n = int(D.shape[0])
                            candidate_k = int(params.get("candidate_k", params.get("knn_k", 30)))
                            if n > 1 and candidate_k < (n - 1):
                                raise ValueError(
                                    "KMeans with precomputed DTW/Frechet requires full pairwise distances. "
                                    "Set candidate_k >= n_flights-1 (or a very large value)."
                                )
                        D_dense = D.toarray()
                    else:
                        D_dense = np.asarray(D, dtype=float)
                    D_dense = _sanitize_precomputed_dense(D_dense)
                    X_embed = _classical_mds_embedding(D_dense, n_components=mds_n_components)
                    labels = clusterer.fit_predict(
                        X_embed,
                        **cluster_params,
                    )
                    metrics = compute_internal_metrics(
                        X_embed,
                        labels,
                        metric_mode="features",
                        include_noise=eval_cfg.get("include_noise", False),
                        sparse_precomputed_policy=str(
                            eval_cfg.get("sparse_precomputed_policy", "dense_if_small")
                        ),
                        sparse_precomputed_max_n=int(eval_cfg.get("sparse_precomputed_max_n", 1500)),
                        precomputed_embed_for_dbch=bool(
                            eval_cfg.get("precomputed_embed_for_dbch", False)
                        ),
                        precomputed_embed_components=int(
                            eval_cfg.get("precomputed_embed_components", 3)
                        ),
                    )
                    log_lines.append(
                        f"  {distance_metric.upper()}+KMeans mode: classical MDS embedding used "
                        f"(n_components={mds_n_components})."
                    )
                else:
                    try:
                        labels = clusterer.fit_predict(D, metric="precomputed", **cluster_params)
                    except Exception as exc:
                        msg = str(exc).lower()
                        if (
                            method == "hdbscan"
                            and issparse(D)
                            and "multiple connected components" in msg
                        ):
                            labels, n_comp = _fit_hdbscan_by_connected_components(
                                clusterer,
                                D,
                                cluster_params=cluster_params,
                            )
                            log_lines.append(
                                f"  HDBSCAN fallback: clustered per connected component (components={n_comp})."
                            )
                        elif (
                            method == "optics"
                            and issparse(D)
                            and "neighbors per samples are required" in msg
                        ):
                            relaxed = dict(params)
                            relaxed["candidate_k"] = max(
                                int(relaxed.get("candidate_k", 30)),
                                int(relaxed.get("min_required_neighbors", 5)) * 4,
                                80,
                            )
                            relaxed.pop("tau", None)
                            relaxed.pop("tau_quantile", None)
                            if distance_metric == "dtw":
                                relaxed["use_lb_keogh"] = False
                            D_relaxed = pairwise_distance_matrix(
                                trajs if distance_metric in {"dtw", "frechet", "lcss"} else X,
                                metric=distance_metric,
                                cache_dir=output_dir / "cache",
                                flow_name=f"{flow_name}_relaxed",
                                params=relaxed,
                                cache_ids=ordered_flight_ids,
                            )
                            if not issparse(D_relaxed):
                                D_relaxed = _sanitize_precomputed_dense(D_relaxed)
                            labels = clusterer.fit_predict(D_relaxed, metric="precomputed", **cluster_params)
                            D = D_relaxed
                            log_lines.append(
                                "  OPTICS fallback: rebuilt sparse graph with relaxed distance params "
                                f"(candidate_k={relaxed['candidate_k']}, tau disabled)."
                            )
                        else:
                            raise
                    metrics = compute_internal_metrics(
                        D,
                        labels,
                        metric_mode="precomputed",
                        include_noise=eval_cfg.get("include_noise", False),
                        sparse_precomputed_policy=str(
                            eval_cfg.get("sparse_precomputed_policy", "dense_if_small")
                        ),
                        sparse_precomputed_max_n=int(eval_cfg.get("sparse_precomputed_max_n", 1500)),
                        precomputed_embed_for_dbch=bool(
                            eval_cfg.get("precomputed_embed_for_dbch", False)
                        ),
                        precomputed_embed_components=int(
                            eval_cfg.get("precomputed_embed_components", 3)
                        ),
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
                    sparse_precomputed_policy=str(
                        eval_cfg.get("sparse_precomputed_policy", "dense_if_small")
                    ),
                    sparse_precomputed_max_n=int(eval_cfg.get("sparse_precomputed_max_n", 1500)),
                    precomputed_embed_for_dbch=bool(
                        eval_cfg.get("precomputed_embed_for_dbch", False)
                    ),
                    precomputed_embed_components=int(
                        eval_cfg.get("precomputed_embed_components", 3)
                    ),
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
        flight_counts = flow_df_fit.groupby("flight_id").size()
        n_points_min = int(flight_counts.min()) if not flight_counts.empty else 0
        n_points_med = int(flight_counts.median()) if not flight_counts.empty else 0
        n_points_max = int(flight_counts.max()) if not flight_counts.empty else 0

        metrics["n_flights_total_flow"] = n_flights_total_flow
        metrics["n_flights_used_for_fit"] = n_flights_used_for_fit
        metrics["fit_sampling_mode"] = fit_sampling_mode
        if feature_transform_meta:
            metrics.update(feature_transform_meta)
        if distance_metric == "lcss":
            metrics["lcss_epsilon_m"] = float(distance_params.get("lcss_epsilon_m", 300.0))
            metrics["lcss_delta_alpha"] = float(distance_params.get("lcss_delta_alpha", 0.10))
            metrics["lcss_normalization"] = str(distance_params.get("lcss_normalization", "min_len"))

        log_lines.append(f"Flow: {flow_label}")
        log_lines.append(
            f"  Flights: used={len(labels)} total_flow={n_flights_total_flow} fit_sampling_mode={fit_sampling_mode}"
        )
        log_lines.append(
            f"  Points: used={len(flow_df_fit)} total_flow={len(flow_df)} "
            f"per_flight[min/med/max]={n_points_min}/{n_points_med}/{n_points_max}"
        )
        if feature_transform_meta:
            log_lines.append(
                "  Feature dims: "
                f"in={feature_transform_meta.get('feature_dim_in')} "
                f"out={feature_transform_meta.get('feature_dim_out')} "
                f"pca_components={feature_transform_meta.get('pca_components')}"
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
                flow_df=flow_df_fit,
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
