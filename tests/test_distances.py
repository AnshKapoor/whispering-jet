import numpy as np
import pytest

from clustering.distances import pairwise_distance_matrix
from distance_metrics import lcss_trajectory_distance


def test_distance_matrix_symmetry():
    X = np.array([[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 2.0, 2.0]])
    D = pairwise_distance_matrix(X, metric="euclidean")
    assert D.shape == (2, 2)
    assert np.allclose(D, D.T)


def test_lcss_distance_identical_is_zero():
    pytest.importorskip("lcsspy")
    t = np.array([[0.0, 0.0], [100.0, 100.0], [200.0, 200.0]], dtype=float)
    d = lcss_trajectory_distance(t, t, epsilon_m=1.0, delta_alpha=0.1, normalization="min_len")
    assert d == 0.0


def test_lcss_distance_monotonic_with_epsilon():
    pytest.importorskip("lcsspy")
    t1 = np.array([[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]], dtype=float)
    t2 = np.array([[0.0, 0.0], [20.0, 20.0], [40.0, 40.0]], dtype=float)
    d_small = lcss_trajectory_distance(t1, t2, epsilon_m=1.0, delta_alpha=0.1, normalization="min_len")
    d_large = lcss_trajectory_distance(t1, t2, epsilon_m=30.0, delta_alpha=0.1, normalization="min_len")
    assert d_large <= d_small


def test_lcss_dense_matrix_basic_properties():
    pytest.importorskip("lcsspy")
    trajs = [
        np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=float),
        np.array([[0.0, 0.0], [1.0, 1.2], [2.0, 2.2]], dtype=float),
        np.array([[10.0, 10.0], [11.0, 11.0], [12.0, 12.0]], dtype=float),
    ]
    D = pairwise_distance_matrix(
        trajs,
        metric="lcss",
        params={"lcss_epsilon_m": 0.5, "lcss_delta_alpha": 0.2, "lcss_normalization": "min_len"},
    )
    assert D.shape == (3, 3)
    assert np.allclose(D, D.T)
    assert np.allclose(np.diag(D), 0.0)
