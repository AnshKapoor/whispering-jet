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

from distance_metrics import dtw_banded, frechet_discrete, lb_keogh, rdp

logger = logging.getLogger(__name__)


def build_feature_matrix(
    flights_df,
    vector_cols: Sequence[str],
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


def pairwise_distance_matrix(
    trajectories: Sequence[np.ndarray] | np.ndarray,
    metric: str = "euclidean",
    cache_dir: Path | None = None,
    flow_name: str | None = None,
    params: dict | None = None,
):
    """
    Compute symmetric pairwise distance matrix (dense or sparse).
    Supported metrics: euclidean, dtw, frechet.
    """

    metric = metric.lower()
    params = params or {}
    if cache_dir and flow_name:
        cache_dir.mkdir(parents=True, exist_ok=True)
        key = _hash_config(flow_name, metric, params, range(len(trajectories)))
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
        k = max(1, min(k, n - 1))
        n_jobs = params.get("n_jobs")
        edges, knn_dists = _build_knn_edges(
            np.stack([np.asarray(t, dtype=float).ravel() for t in trajectories], axis=0),
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
                        if lb_keogh(trajectories[i], trajectories[j], int(lb_radius)) > (tau * tau):
                            continue
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
        logger.info(
            "Computed %s sparse distances in %.1fs (edges kept=%d).",
            metric,
            time.perf_counter() - start,
            len(results),
        )
    else:
        raise ValueError(f"Unsupported distance metric: {metric}")

    if cache_dir and flow_name:
        if metric in {"dtw", "frechet"}:
            save_npz(cache_dir / f"dist_{key}.npz", D)
        else:
            np.save(cache_dir / f"dist_{key}.npy", D)
    return D
