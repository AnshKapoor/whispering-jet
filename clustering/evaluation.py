"""Cluster quality metrics."""

from __future__ import annotations

from typing import Literal

import numpy as np
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from scipy.sparse import issparse


def compute_internal_metrics(
    X_or_D,
    labels,
    metric_mode: Literal["features", "precomputed"],
    include_noise: bool = False,
) -> dict:
    labels = np.asarray(labels)
    total_flights = int(len(labels))
    n_noise_flights = int(np.sum(labels == -1)) if total_flights else 0
    n_clustered_flights = int(total_flights - n_noise_flights)
    noise_frac = float(n_noise_flights / total_flights) if total_flights else 0.0

    # Work on a copy for internal metric computation; keep raw-label stats intact.
    labels_for_metrics = labels.copy()
    X_for_metrics = X_or_D
    if not include_noise:
        if metric_mode == "precomputed" and issparse(X_for_metrics):
            idx = np.where(labels_for_metrics != -1)[0]
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

    if metric_mode == "precomputed":
        if issparse(X_for_metrics):
            metrics["silhouette"] = float("nan")
            metrics["davies_bouldin"] = float("nan")
            metrics["calinski_harabasz"] = float("nan")
            metrics["reason"] = "sparse_precomputed_distances"
            return metrics
        metrics["silhouette"] = float(silhouette_score(X_for_metrics, labels_for_metrics, metric="precomputed"))
    else:
        metrics["silhouette"] = float(silhouette_score(X_for_metrics, labels_for_metrics))
        metrics["davies_bouldin"] = float(davies_bouldin_score(X_for_metrics, labels_for_metrics))
        metrics["calinski_harabasz"] = float(calinski_harabasz_score(X_for_metrics, labels_for_metrics))

    if metric_mode == "precomputed":
        metrics["davies_bouldin"] = float("nan")
        metrics["calinski_harabasz"] = float("nan")
    return metrics
