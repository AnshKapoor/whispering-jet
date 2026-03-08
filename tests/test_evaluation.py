import numpy as np

from clustering.evaluation import compute_internal_metrics


def test_all_noise_returns_nan():
    X = np.random.rand(5, 2)
    labels = np.array([-1, -1, -1, -1, -1])
    metrics = compute_internal_metrics(X, labels, metric_mode="features", include_noise=False)
    assert np.isnan(metrics["silhouette"])
    assert metrics["reason"] == "<2 clusters"
    assert metrics["noise_frac"] == 1.0
    assert metrics["n_noise_flights"] == 5
    assert metrics["n_clustered_flights"] == 0


def test_single_cluster_returns_nan():
    X = np.random.rand(5, 2)
    labels = np.array([0, 0, 0, 0, 0])
    metrics = compute_internal_metrics(X, labels, metric_mode="features", include_noise=False)
    assert np.isnan(metrics["silhouette"])
    assert metrics["noise_frac"] == 0.0
    assert metrics["n_noise_flights"] == 0
    assert metrics["n_clustered_flights"] == 5


def test_noise_stats_preserved_when_excluding_noise():
    X = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.1],
            [5.0, 5.0],
            [5.1, 5.1],
            [10.0, 10.0],
            [11.0, 11.0],
        ]
    )
    labels = np.array([0, 0, 1, 1, -1, -1])
    metrics = compute_internal_metrics(X, labels, metric_mode="features", include_noise=False)
    assert metrics["n_clusters"] == 2
    assert metrics["n_noise_flights"] == 2
    assert metrics["n_clustered_flights"] == 4
    assert metrics["noise_frac"] == 2 / 6


def test_dense_precomputed_with_noise_subsets_both_axes():
    # 4 samples, last one is noise; two valid clusters among first three labels.
    D = np.array(
        [
            [0.0, 1.0, 5.0, 9.0],
            [1.0, 0.0, 4.0, 8.0],
            [5.0, 4.0, 0.0, 7.0],
            [9.0, 8.0, 7.0, 0.0],
        ],
        dtype=float,
    )
    labels = np.array([0, 0, 1, -1], dtype=int)
    metrics = compute_internal_metrics(D, labels, metric_mode="precomputed", include_noise=False)
    assert metrics["n_clusters"] == 2
    assert metrics["n_noise_flights"] == 1
    assert metrics["n_clustered_flights"] == 3
    assert np.isfinite(metrics["silhouette"])
