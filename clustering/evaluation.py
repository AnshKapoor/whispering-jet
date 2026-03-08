"""Cluster quality metrics."""

from __future__ import annotations

from typing import Literal

import numpy as np
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from scipy.sparse import issparse


def _classical_mds_from_distance(D: np.ndarray, n_components: int = 3) -> np.ndarray:
    """Return classical-MDS embedding from a dense distance matrix."""

    D = np.asarray(D, dtype=float)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("Distance matrix must be square for classical MDS.")
    n = D.shape[0]
    if n == 0:
        return np.zeros((0, max(1, n_components)), dtype=float)
    if n == 1:
        return np.zeros((1, max(1, n_components)), dtype=float)

    np.fill_diagonal(D, 0.0)
    J = np.eye(n) - np.ones((n, n), dtype=float) / n
    B = -0.5 * J @ (D ** 2) @ J
    eigvals, eigvecs = np.linalg.eigh(B)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    n_components = max(1, int(n_components))
    positive = eigvals > 1e-12
    keep = min(int(np.sum(positive)), n_components)
    emb = np.zeros((n, n_components), dtype=float)
    if keep > 0:
        emb[:, :keep] = eigvecs[:, :keep] * np.sqrt(eigvals[:keep])
    return emb


def compute_internal_metrics(
    X_or_D,
    labels,
    metric_mode: Literal["features", "precomputed"],
    include_noise: bool = False,
    sparse_precomputed_policy: Literal["skip", "dense_if_small"] = "dense_if_small",
    sparse_precomputed_max_n: int = 1500,
    precomputed_embed_for_dbch: bool = False,
    precomputed_embed_components: int = 3,
) -> dict:
    labels = np.asarray(labels)
    precomputed_mode = metric_mode == "precomputed"
    total_flights = int(len(labels))
    n_noise_flights = int(np.sum(labels == -1)) if total_flights else 0
    n_clustered_flights = int(total_flights - n_noise_flights)
    noise_frac = float(n_noise_flights / total_flights) if total_flights else 0.0

    # Work on a copy for internal metric computation; keep raw-label stats intact.
    labels_for_metrics = labels.copy()
    X_for_metrics = X_or_D
    if not include_noise:
        if precomputed_mode:
            idx = np.where(labels_for_metrics != -1)[0]
            # For precomputed distances (dense or sparse), filter both axes.
            X_for_metrics = X_for_metrics[idx][:, idx]
            labels_for_metrics = labels_for_metrics[idx]
        else:
            mask = labels_for_metrics != -1
            X_for_metrics = X_for_metrics[mask]
            labels_for_metrics = labels_for_metrics[mask]

    unique = [c for c in np.unique(labels_for_metrics) if c != -1]
    if len(unique) < 2:
        return {
            "davies_bouldin": float("nan"),
            "silhouette": float("nan"),
            "calinski_harabasz": float("nan"),
            "n_clusters": len(unique),
            "noise_frac": noise_frac,
            "n_noise_flights": n_noise_flights,
            "n_clustered_flights": n_clustered_flights,
            "reason": "<2 clusters",
        }

    metrics = {
        "n_clusters": len(unique),
        "noise_frac": noise_frac,
        "n_noise_flights": n_noise_flights,
        "n_clustered_flights": n_clustered_flights,
    }

    if precomputed_mode:
        if issparse(X_for_metrics):
            n = int(X_for_metrics.shape[0])
            if sparse_precomputed_policy == "dense_if_small" and n <= int(sparse_precomputed_max_n):
                # For moderate n, densify sparse precomputed distances to enable silhouette.
                X_for_metrics = X_for_metrics.toarray()
                X_for_metrics = np.asarray(X_for_metrics, dtype=float)
                np.fill_diagonal(X_for_metrics, 0.0)
                metrics["silhouette"] = float(
                    silhouette_score(X_for_metrics, labels_for_metrics, metric="precomputed")
                )
                if precomputed_embed_for_dbch:
                    X_embed = _classical_mds_from_distance(
                        X_for_metrics,
                        n_components=int(precomputed_embed_components),
                    )
                    metrics["davies_bouldin"] = float(davies_bouldin_score(X_embed, labels_for_metrics))
                    metrics["calinski_harabasz"] = float(calinski_harabasz_score(X_embed, labels_for_metrics))
                    metrics["reason"] = "dense_from_sparse_precomputed; db_ch_from_cmdscale_embedding"
                else:
                    metrics["davies_bouldin"] = float("nan")
                    metrics["calinski_harabasz"] = float("nan")
                    metrics["reason"] = "dense_from_sparse_precomputed; db_ch_not_defined"
                return metrics
            metrics["silhouette"] = float("nan")
            metrics["davies_bouldin"] = float("nan")
            metrics["calinski_harabasz"] = float("nan")
            metrics["reason"] = "sparse_precomputed_distances"
            return metrics
        # Dense precomputed matrices must be square with zero diagonal.
        X_for_metrics = np.asarray(X_for_metrics, dtype=float)
        np.fill_diagonal(X_for_metrics, 0.0)
        metrics["silhouette"] = float(silhouette_score(X_for_metrics, labels_for_metrics, metric="precomputed"))
        if precomputed_embed_for_dbch:
            X_embed = _classical_mds_from_distance(
                X_for_metrics,
                n_components=int(precomputed_embed_components),
            )
            metrics["davies_bouldin"] = float(davies_bouldin_score(X_embed, labels_for_metrics))
            metrics["calinski_harabasz"] = float(calinski_harabasz_score(X_embed, labels_for_metrics))
            metrics["reason"] = "db_ch_from_cmdscale_embedding"
    else:
        metrics["silhouette"] = float(silhouette_score(X_for_metrics, labels_for_metrics))
        metrics["davies_bouldin"] = float(davies_bouldin_score(X_for_metrics, labels_for_metrics))
        metrics["calinski_harabasz"] = float(calinski_harabasz_score(X_for_metrics, labels_for_metrics))

    if precomputed_mode:
        metrics.setdefault("davies_bouldin", float("nan"))
        metrics.setdefault("calinski_harabasz", float("nan"))
    return metrics
