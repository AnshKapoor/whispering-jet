"""Feature and distance helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import logging
import time
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from scipy.spatial.distance import cdist
from scipy.sparse import coo_matrix, load_npz, save_npz
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import sort_graph_by_row_values

from distance_metrics import dtw_banded, frechet_discrete, lb_keogh, lcss_trajectory_distance, rdp

logger = logging.getLogger(__name__)


def build_feature_matrix(
    flights_df,
    vector_cols: Sequence[str],
    allow_ragged: bool = False,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Returns (X, trajectories) where X is (n_flights, n_features).
    Expects flights_df to have columns: step, flight_id, and vector_cols per step.
    """

    feature_rows: List[List[float]] = []
    trajs: List[np.ndarray] = []
    # Use deterministic ordering of flights to keep labels aligned with metadata.
    for _, flight in flights_df.groupby("flight_id", sort=True):
        flight_sorted = flight.sort_values("step")
        vec: List[float] = []
        traj_coords: List[Tuple[float, float]] = []
        for _, row in flight_sorted.iterrows():
            coords = [float(row[col]) for col in vector_cols if col in row]
            vec.extend(coords) # Alternating x and y coordinates
            if len(coords) >= 2:
                traj_coords.append((coords[0], coords[1]))
        feature_rows.append(vec)
        trajs.append(np.array(traj_coords, dtype=float))
    if allow_ragged:
        try:
            X = np.array(feature_rows, dtype=float)
        except ValueError:
            # Variable-length flights (e.g. non-resampled inputs) are valid for
            # DTW/Frechet/LCSS precomputed distances; keep rows aligned as object arrays.
            X = np.array([np.asarray(v, dtype=float) for v in feature_rows], dtype=object)
    else:
        X = np.array(feature_rows, dtype=float)
    return X, trajs


def _build_knn_edges(
    Z: np.ndarray,
    k: int,
    n_jobs: int | None = None,
) -> tuple[list[tuple[int, int]], np.ndarray]:
    """Return unique undirected kNN edges and the raw kNN distances."""

    if Z.ndim != 2:
        raise ValueError("kNN embedding must be a 2D array.")
    n = Z.shape[0]
    if n == 0:
        return [], np.array([], dtype=float)

    k = max(1, min(k, n - 1))
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean", n_jobs=n_jobs)
    nn.fit(Z)
    dists, idx = nn.kneighbors(Z, return_distance=True)

    edges: set[tuple[int, int]] = set()
    dist_values: list[float] = []
    for i in range(n):
        for pos in range(1, idx.shape[1]):
            j = int(idx[i, pos])
            if i == j:
                continue
            a, b = (i, j) if i < j else (j, i)
            edges.add((a, b))
            dist_values.append(float(dists[i, pos]))

    return sorted(edges), np.array(dist_values, dtype=float)


def _estimate_tau(dist_values: np.ndarray, params: dict) -> float | None:
    """Estimate pruning threshold tau from kNN distances when requested."""

    tau = params.get("tau")
    if tau is None:
        tau_q = params.get("tau_quantile")
        if tau_q is not None and dist_values.size:
            tau = float(np.quantile(dist_values, float(tau_q)))
    tau_scale = params.get("tau_scale")
    if tau is not None and tau_scale is not None:
        tau = float(tau) * float(tau_scale)
    return float(tau) if tau is not None else None


def _hash_config(flow_name: str, metric: str, params: dict, flight_ids: Iterable) -> str:
    payload = json.dumps(
        {"flow": flow_name, "metric": metric, "params": params, "flight_ids": list(flight_ids)}, sort_keys=True
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _resample_traj_for_embedding(traj: np.ndarray, n_points: int) -> np.ndarray:
    """
    Build a fixed-length 2D representation for kNN candidate selection.
    This keeps sparse-graph construction working even when trajectories
    have variable lengths (e.g. non-resampled preprocessing variants).
    """
    arr = np.asarray(traj, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2 or arr.shape[0] == 0:
        return np.zeros((n_points, 2), dtype=float)
    xy = arr[:, :2]
    if xy.shape[0] == 1:
        return np.repeat(xy, n_points, axis=0)
    src = np.linspace(0.0, 1.0, num=xy.shape[0])
    dst = np.linspace(0.0, 1.0, num=n_points)
    x = np.interp(dst, src, xy[:, 0])
    y = np.interp(dst, src, xy[:, 1])
    return np.column_stack([x, y])


def _trajectory_knn_embedding(
    trajectories: Sequence[np.ndarray],
    params: dict,
) -> np.ndarray:
    """
    Return a fixed-width embedding used only to build candidate kNN edges.
    """
    mode = str(params.get("knn_embedding", "auto")).lower()
    if mode == "flatten":
        return np.stack([np.asarray(t, dtype=float).ravel() for t in trajectories], axis=0)
    if mode == "resample":
        m = int(params.get("knn_embed_points", 16))
        return np.stack(
            [_resample_traj_for_embedding(np.asarray(t, dtype=float), m).ravel() for t in trajectories],
            axis=0,
        )
    # auto: try flatten (fast when equal-length), fallback to resampled embedding.
    try:
        return np.stack([np.asarray(t, dtype=float).ravel() for t in trajectories], axis=0)
    except ValueError:
        m = int(params.get("knn_embed_points", 16))
        return np.stack(
            [_resample_traj_for_embedding(np.asarray(t, dtype=float), m).ravel() for t in trajectories],
            axis=0,
        )


def pairwise_distance_matrix(
    trajectories: Sequence[np.ndarray] | np.ndarray,
    metric: str = "euclidean",
    cache_dir: Path | None = None,
    flow_name: str | None = None,
    params: dict | None = None,
    cache_ids: Iterable | None = None,
):
    """
    Compute symmetric pairwise distance matrix (dense or sparse).
    Supported metrics: euclidean, dtw, frechet, lcss.
    """

    metric = metric.lower()
    params = params or {}
    if cache_dir and flow_name:
        cache_dir.mkdir(parents=True, exist_ok=True)
        hash_ids = cache_ids if cache_ids is not None else range(len(trajectories))
        key = _hash_config(flow_name, metric, params, hash_ids)
        if metric in {"dtw", "frechet"}:
            cache_path = cache_dir / f"dist_{key}.npz"
            if cache_path.exists():
                return load_npz(cache_path)
        else:
            cache_path = cache_dir / f"dist_{key}.npy"
            if cache_path.exists():
                D = np.load(cache_path)
                # Silhouette with precomputed distances requires zero diagonal.
                np.fill_diagonal(D, 0.0)
                return D

    if metric == "euclidean":
        # Accept either a 2D feature matrix, or a sequence of arrays (flattened).
        if isinstance(trajectories, np.ndarray):
            if trajectories.ndim != 2:
                raise ValueError("Euclidean distance requires a 2D feature matrix.")
            flat = trajectories.astype(float, copy=False)
        else:
            flat = np.stack([np.asarray(t, dtype=float).ravel() for t in trajectories], axis=0)
        D = cdist(flat, flat, metric="euclidean")
    elif metric in {"dtw", "frechet"}:
        if isinstance(trajectories, np.ndarray):
            raise ValueError(f"{metric} distance requires a sequence of (T,D) trajectories.")
        n = len(trajectories)
        if n <= 1:
            return coo_matrix((n, n)).tocsr()

        k = int(params.get("candidate_k", params.get("knn_k", 30)))
        min_required_neighbors = int(params.get("min_required_neighbors", 1))
        if min_required_neighbors > 1:
            # Keep precomputed sparse graph compatible with neighbor-based methods
            # (e.g. OPTICS min_samples).
            k = max(k, min_required_neighbors)
        k = max(1, min(k, n - 1))
        n_jobs = params.get("n_jobs")
        Z = _trajectory_knn_embedding(trajectories, params)
        edges, knn_dists = _build_knn_edges(
            Z,
            k=k,
            n_jobs=n_jobs,
        )
        tau = _estimate_tau(knn_dists, params)
        batch_size = int(params.get("batch_size", 10000))

        if metric == "dtw":
            w = params.get("dtw_window_size", params.get("window", 8))
            w = int(w) if w is not None else 8
            use_lb = bool(params.get("use_lb_keogh", True))
            lb_radius = params.get("lb_keogh_radius")
            if lb_radius is None and use_lb:
                lb_radius = w
        else:
            rdp_epsilon = float(params.get("rdp_epsilon", 50.0))
            simplified = [rdp(np.asarray(t, dtype=float), rdp_epsilon) for t in trajectories]

        logger.info(
            "Computing %s distances on %d candidate edges (k=%d, n=%d).",
            metric,
            len(edges),
            k,
            n,
        )
        if metric == "dtw":
            logger.info("DTW band w=%s, tau=%s, lb_keogh=%s", w, tau, use_lb)
        else:
            logger.info("Frechet RDP epsilon=%.2f, tau=%s", rdp_epsilon, tau)

        def _compute_batch(batch_edges: list[tuple[int, int]]) -> list[tuple[int, int, float]]:
            results: list[tuple[int, int, float]] = []
            for i, j in batch_edges:
                if metric == "dtw":
                    if use_lb and tau is not None and lb_radius is not None:
                        try:
                            if lb_keogh(trajectories[i], trajectories[j], int(lb_radius)) > (tau * tau):
                                continue
                        except Exception:
                            # Variable-length trajectories can break LB-Keogh envelope
                            # construction at sequence boundaries; skip pruning in that case.
                            pass
                    d = dtw_banded(trajectories[i], trajectories[j], w=int(w), tau=tau)
                else:
                    d = frechet_discrete(simplified[i], simplified[j], tau=tau)
                if np.isfinite(d):
                    results.append((i, j, float(d)))
            return results

        start = time.perf_counter()
        if n_jobs and int(n_jobs) != 1:
            from joblib import Parallel, delayed

            batches = [edges[i : i + batch_size] for i in range(0, len(edges), batch_size)]
            results_nested = Parallel(n_jobs=int(n_jobs), prefer="processes")(
                delayed(_compute_batch)(batch) for batch in batches
            )
            results = [item for batch in results_nested for item in batch]
        else:
            results = _compute_batch(edges)

        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        for i, j, d in results:
            rows.extend([i, j])
            cols.extend([j, i])
            data.extend([d, d])
        if n:
            rows.extend(range(n))
            cols.extend(range(n))
            data.extend([0.0] * n)
        D = coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
        try:
            D = sort_graph_by_row_values(D, warn_when_not_sorted=False)
        except Exception:
            # Older sklearn builds may not expose this helper.
            pass
        logger.info(
            "Computed %s sparse distances in %.1fs (edges kept=%d).",
            metric,
            time.perf_counter() - start,
            len(results),
        )
    elif metric == "lcss":
        if isinstance(trajectories, np.ndarray):
            raise ValueError("lcss distance requires a sequence of (T,D) trajectories.")
        n = len(trajectories)
        D = np.zeros((n, n), dtype=float)
        if n <= 1:
            return D

        epsilon_m = float(params.get("lcss_epsilon_m", 300.0))
        delta_alpha = float(params.get("lcss_delta_alpha", 0.10))
        normalization = str(params.get("lcss_normalization", "min_len"))
        log_every = int(params.get("log_every", 0))

        logger.info(
            "Computing lcss dense distances (n=%d, epsilon=%.3f, delta_alpha=%.3f, normalization=%s).",
            n,
            epsilon_m,
            delta_alpha,
            normalization,
        )
        total_pairs = n * (n - 1) // 2
        pair_idx = 0
        start = time.perf_counter()
        for i in range(n):
            D[i, i] = 0.0
            for j in range(i + 1, n):
                d = lcss_trajectory_distance(
                    np.asarray(trajectories[i], dtype=float),
                    np.asarray(trajectories[j], dtype=float),
                    epsilon_m=epsilon_m,
                    delta_alpha=delta_alpha,
                    normalization=normalization,
                )
                D[i, j] = D[j, i] = float(d)
                pair_idx += 1
                if log_every > 0 and pair_idx % log_every == 0:
                    elapsed = time.perf_counter() - start
                    rate = pair_idx / elapsed if elapsed > 0 else 0.0
                    logger.info(
                        "LCSS progress: %d/%d pairs (%.1f%%), %.2f pairs/s",
                        pair_idx,
                        total_pairs,
                        (100.0 * pair_idx / total_pairs) if total_pairs else 100.0,
                        rate,
                    )
        np.fill_diagonal(D, 0.0)
        logger.info("Computed lcss dense matrix in %.1fs.", time.perf_counter() - start)
    else:
        raise ValueError(f"Unsupported distance metric: {metric}")

    if cache_dir and flow_name:
        if metric in {"dtw", "frechet"}:
            save_npz(cache_dir / f"dist_{key}.npz", D)
        else:
            np.save(cache_dir / f"dist_{key}.npy", D)
    return D
