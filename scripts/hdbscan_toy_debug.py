"""Toy HDBSCAN run for debugger stepping.

Use this to attach a debugger into hdbscan_.py and step through fit().
"""

from __future__ import annotations

import numpy as np


def make_toy_data(seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    c1 = rng.normal(loc=(-5.0, -5.0), scale=0.6, size=(60, 2))
    c2 = rng.normal(loc=(5.5, 5.0), scale=0.7, size=(70, 2))
    c3 = rng.normal(loc=(0.0, 7.0), scale=0.5, size=(50, 2))
    noise = rng.uniform(low=-9.0, high=9.0, size=(30, 2))
    return np.vstack([c1, c2, c3, noise])


def main() -> None:
    import hdbscan

    X = make_toy_data()
    # Adjust parameters to get non-trivial structure but still some noise.
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=15,
        min_samples=5,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(X)

    # Keep output small for debugging runs.
    unique, counts = np.unique(labels, return_counts=True)
    print("labels:", dict(zip(unique.tolist(), counts.tolist())))
    print("outliers:", int(np.sum(labels == -1)))


if __name__ == "__main__":
    main()
