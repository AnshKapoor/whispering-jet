"""Interchangeable trajectory distance metrics.

Primary DTW/Frechet implementations use external libraries:
- dtw-python (`dtw` package)
- frechetdist (`frechetdist` package)

NumPy fallbacks are kept for robustness when optional libs are unavailable.
Trajectories are expected as numpy arrays shaped (T, D) with D in {2, 3}.
"""

from __future__ import annotations

from typing import Callable, Sequence

import logging
import time

import numpy as np

try:
    from dtw import dtw as _dtw_python
except Exception:  # pragma: no cover - depends on environment
    _dtw_python = None

try:
    from frechetdist import frdist as _frechetdist
except Exception:  # pragma: no cover - depends on environment
    _frechetdist = None

Trajectory = np.ndarray  # shape (T, D), D = 2 or 3
DistanceFn = Callable[[Trajectory, Trajectory], float]

logger = logging.getLogger(__name__)


def _validate_trajectories(traj1: Trajectory, traj2: Trajectory) -> tuple[np.ndarray, np.ndarray]:
    """Ensure both trajectories are 2-D NumPy arrays with matching dimensions."""

    if not isinstance(traj1, np.ndarray) or not isinstance(traj2, np.ndarray):
        raise ValueError("Trajectories must be numpy.ndarray instances.")
    if traj1.ndim != 2 or traj2.ndim != 2:
        raise ValueError("Trajectories must be 2-D arrays of shape (T, D).")
    if traj1.shape[1] != traj2.shape[1]:
        raise ValueError(f"Dimensionality mismatch: {traj1.shape[1]} vs {traj2.shape[1]}.")
    if np.isnan(traj1).any() or np.isnan(traj2).any():
        raise ValueError("Trajectories contain NaN values.")
    return traj1.astype(float, copy=False), traj2.astype(float, copy=False)


def euclidean_distance(traj1: Trajectory, traj2: Trajectory) -> float:
    """
    Compute simple Euclidean distance between two trajectories of equal length.

    Distance: || vec(traj1) - vec(traj2) ||_2
    """

    traj1, traj2 = _validate_trajectories(traj1, traj2)
    if traj1.shape != traj2.shape:
        raise ValueError(f"Shape mismatch: {traj1.shape} vs {traj2.shape}.")
    diff = traj1.ravel() - traj2.ravel()
    return float(np.linalg.norm(diff))


def dtw_distance(traj1: Trajectory, traj2: Trajectory, window_size: int | None = None) -> float:
    """Compute DTW distance using dtw-python, with NumPy fallback."""

    traj1, traj2 = _validate_trajectories(traj1, traj2)
    t1, t2 = traj1.shape[0], traj2.shape[0]
    if window_size is not None and window_size < 0:
        raise ValueError("window_size must be non-negative or None.")

    if _dtw_python is not None:
        kwargs = {
            "dist_method": "euclidean",
            "step_pattern": "symmetric2",
            "distance_only": True,
            "keep_internals": False,
        }
        if window_size is not None:
            kwargs["window_type"] = "sakoechiba"
            kwargs["window_args"] = {"window_size": int(window_size)}
        try:
            alignment = _dtw_python(traj1, traj2, **kwargs)
            return float(alignment.distance)
        except Exception as exc:
            logger.debug("dtw-python failed, falling back to NumPy DTW: %s", exc)

    dp = np.full((t1 + 1, t2 + 1), np.inf, dtype=float)
    dp[0, 0] = 0.0

    for i in range(1, t1 + 1):
        i_idx = i - 1
        j_start = 1
        j_end = t2 + 1
        if window_size is not None:
            j_start = max(1, i - window_size)
            j_end = min(t2 + 1, i + window_size + 1)
        for j in range(j_start, j_end):
            j_idx = j - 1
            cost = float(np.linalg.norm(traj1[i_idx] - traj2[j_idx]))
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])

    return float(dp[t1, t2])


def discrete_frechet_distance(traj1: Trajectory, traj2: Trajectory) -> float:
    """Compute discrete Frechet distance between two trajectories."""

    traj1, traj2 = _validate_trajectories(traj1, traj2)
    if _frechetdist is not None:
        try:
            return float(_frechetdist(traj1.tolist(), traj2.tolist()))
        except Exception as exc:
            logger.debug("frechetdist failed, falling back to NumPy Frechet: %s", exc)
    t1, t2 = traj1.shape[0], traj2.shape[0]
    ca = np.full((t1, t2), np.inf, dtype=float)

    def d(i: int, j: int) -> float:
        return float(np.linalg.norm(traj1[i] - traj2[j]))

    ca[0, 0] = d(0, 0)
    for i in range(1, t1):
        ca[i, 0] = max(ca[i - 1, 0], d(i, 0))
    for j in range(1, t2):
        ca[0, j] = max(ca[0, j - 1], d(0, j))

    for i in range(1, t1):
        for j in range(1, t2):
            ca[i, j] = max(min(ca[i - 1, j], ca[i - 1, j - 1], ca[i, j - 1]), d(i, j))

    return float(ca[t1 - 1, t2 - 1])


def get_trajectory_distance_fn(name: str) -> DistanceFn:
    """Return the distance function for a given metric name."""

    normalized = name.lower()
    if normalized == "euclidean":
        return euclidean_distance
    if normalized == "dtw":
        return dtw_distance
    if normalized == "frechet":
        return discrete_frechet_distance
    raise ValueError(f"Unsupported distance metric: {name}")


def pairwise_distance_matrix(
    trajectories: Sequence[Trajectory],
    metric: str = "euclidean",
    dtw_window_size: int | None = None,
    log_every: int | None = None,
) -> np.ndarray:
    """Compute a symmetric pairwise distance matrix for the provided trajectories."""

    if not trajectories:
        raise ValueError("No trajectories provided for distance computation.")

    metric = metric.lower()
    n = len(trajectories)
    mat = np.zeros((n, n), dtype=float)
    total_pairs = n * (n - 1) // 2
    log_every = int(log_every) if log_every else None

    if metric in {"dtw", "frechet"}:
        logger.info(
            "Computing %s distance matrix for %d trajectories (%d pairs).",
            metric,
            n,
            total_pairs,
        )
        if metric == "dtw" and dtw_window_size is not None:
            logger.info("DTW window size: %s", dtw_window_size)
        start = time.perf_counter()
        pairs_done = 0

    for i in range(n):
        mat[i, i] = 0.0
        for j in range(i + 1, n):
            if metric == "dtw":
                d = dtw_distance(trajectories[i], trajectories[j], window_size=dtw_window_size)
            elif metric == "frechet":
                d = discrete_frechet_distance(trajectories[i], trajectories[j])
            elif metric == "euclidean":
                d = euclidean_distance(trajectories[i], trajectories[j])
            else:
                raise ValueError(f"Unsupported distance metric: {metric}")
            mat[i, j] = mat[j, i] = d
            if metric in {"dtw", "frechet"}:
                pairs_done += 1
                if log_every and pairs_done % log_every == 0:
                    elapsed = time.perf_counter() - start
                    rate = pairs_done / elapsed if elapsed > 0 else 0.0
                    remaining = total_pairs - pairs_done
                    eta = remaining / rate if rate > 0 else float("inf")
                    logger.info(
                        "Distance progress: %d/%d pairs (%.1f%%), %.2f pairs/s, ETA %.1fs",
                        pairs_done,
                        total_pairs,
                        100.0 * pairs_done / total_pairs if total_pairs else 100.0,
                        rate,
                        eta,
                    )

    if metric in {"dtw", "frechet"}:
        elapsed = time.perf_counter() - start
        logger.info("Computed %s distance matrix in %.1fs.", metric, elapsed)
    return mat


def dtw_banded(
    traj1: Trajectory,
    traj2: Trajectory,
    w: int,
    tau: float | None = None,
) -> float:
    """Compute banded DTW (dtw-python with fallback DP)."""

    traj1, traj2 = _validate_trajectories(traj1, traj2)
    n, m = traj1.shape[0], traj2.shape[0]
    if w < 0:
        raise ValueError("w must be non-negative.")
    if n == 0 or m == 0:
        return float("inf")

    if w < abs(n - m):
        w = abs(n - m)

    # Prefer dtw-python when available; apply tau threshold as post-check.
    if _dtw_python is not None:
        d = dtw_distance(traj1, traj2, window_size=w)
        if tau is not None and d > tau:
            return float("inf")
        return d

    tau_sq = tau * tau if tau is not None else None
    prev = np.full(m + 1, np.inf, dtype=float)
    curr = np.full(m + 1, np.inf, dtype=float)
    prev[0] = 0.0

    for i in range(1, n + 1):
        curr.fill(np.inf)
        j_start = max(1, i - w)
        j_end = min(m, i + w)
        for j in range(j_start, j_end + 1):
            cost = float(np.sum((traj1[i - 1] - traj2[j - 1]) ** 2))
            curr[j] = cost + min(prev[j], curr[j - 1], prev[j - 1])
        if tau_sq is not None:
            row_min = float(np.min(curr[j_start : j_end + 1]))
            if row_min > tau_sq:
                return float("inf")
        prev, curr = curr, prev

    final = prev[m]
    return float(np.sqrt(final))


def frechet_discrete(
    traj1: Trajectory,
    traj2: Trajectory,
    tau: float | None = None,
) -> float:
    """Compute discrete Frechet distance with rolling DP and early abandoning."""

    traj1, traj2 = _validate_trajectories(traj1, traj2)
    n, m = traj1.shape[0], traj2.shape[0]
    if n == 0 or m == 0:
        return float("inf")

    # Prefer frechetdist when available; apply tau threshold as post-check.
    if _frechetdist is not None:
        d = discrete_frechet_distance(traj1, traj2)
        if tau is not None and d > tau:
            return float("inf")
        return d

    tau_sq = tau * tau if tau is not None else None

    def dist_sq(i: int, j: int) -> float:
        diff = traj1[i] - traj2[j]
        return float(np.dot(diff, diff))

    prev = np.full(m, np.inf, dtype=float)
    curr = np.full(m, np.inf, dtype=float)

    prev[0] = dist_sq(0, 0)
    for j in range(1, m):
        prev[j] = max(prev[j - 1], dist_sq(0, j))

    for i in range(1, n):
        curr[0] = max(prev[0], dist_sq(i, 0))
        for j in range(1, m):
            curr[j] = max(min(prev[j], prev[j - 1], curr[j - 1]), dist_sq(i, j))
        if tau_sq is not None and float(np.min(curr)) > tau_sq:
            return float("inf")
        prev, curr = curr, prev

    return float(np.sqrt(prev[m - 1]))


def lb_keogh(traj1: Trajectory, traj2: Trajectory, radius: int) -> float:
    """Compute LB_Keogh lower bound (squared) for multi-dimensional trajectories."""

    traj1, traj2 = _validate_trajectories(traj1, traj2)
    n = traj1.shape[0]
    if radius < 0:
        raise ValueError("radius must be non-negative.")

    lb_sum = 0.0
    for i in range(n):
        start = max(0, i - radius)
        end = min(n, i + radius + 1)
        window = traj2[start:end]
        upper = np.max(window, axis=0)
        lower = np.min(window, axis=0)
        diff = np.where(traj1[i] > upper, traj1[i] - upper, 0.0)
        diff = np.where(traj1[i] < lower, lower - traj1[i], diff)
        lb_sum += float(np.sum(diff ** 2))
    return lb_sum


def rdp(points: np.ndarray, epsilon: float) -> np.ndarray:
    """Ramer-Douglas-Peucker simplification for 2D/3D polylines."""

    if points.shape[0] <= 2 or epsilon <= 0:
        return points

    def _perp_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        if np.allclose(a, b):
            return float(np.linalg.norm(p - a))
        ab = b - a
        t = float(np.dot(p - a, ab) / np.dot(ab, ab))
        t = max(0.0, min(1.0, t))
        proj = a + t * ab
        return float(np.linalg.norm(p - proj))

    stack = [(0, points.shape[0] - 1)]
    keep = np.zeros(points.shape[0], dtype=bool)
    keep[0] = True
    keep[-1] = True

    while stack:
        start, end = stack.pop()
        max_dist = 0.0
        idx = None
        a = points[start]
        b = points[end]
        for i in range(start + 1, end):
            dist = _perp_dist(points[i], a, b)
            if dist > max_dist:
                max_dist = dist
                idx = i
        if idx is not None and max_dist > epsilon:
            keep[idx] = True
            stack.append((start, idx))
            stack.append((idx, end))

    return points[keep]


if __name__ == "__main__":
    traj1 = np.array([[0, 0], [1, 1], [2, 2]], dtype=float)
    traj2 = np.array([[0, 0], [1, 2], [2, 4]], dtype=float)

    print("Euclidean:", euclidean_distance(traj1, traj2))
    print("DTW:", dtw_distance(traj1, traj2))
    print("Frechet:", discrete_frechet_distance(traj1, traj2))


