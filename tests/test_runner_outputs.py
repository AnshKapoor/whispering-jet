import numpy as np
import pandas as pd
import pytest
import yaml

from experiments.runner import run_experiment


def _write_preprocessed_csv(path, rows):
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def _write_config(path, preprocessed_csv, output_dir, experiment_name):
    cfg = {
        "clustering": {
            "method": "kmeans",
            "distance_metric": "euclidean",
            "kmeans": {"n_clusters": 2, "random_state": 42, "n_init": 1},
            "evaluation": {"include_noise": False},
        },
        "flows": {"flow_keys": ["A/D", "Runway"], "include": []},
        "features": {"vector_cols": ["x_utm", "y_utm"]},
        "input": {"preprocessed_csv": str(preprocessed_csv)},
        "preprocessing": {"resampling": {"n_points": 2}},
        "output": {"dir": str(output_dir), "experiment_name": experiment_name},
    }
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")


def test_runner_aligns_labels_and_writes_cluster_counts(tmp_path, monkeypatch):
    csv_path = tmp_path / "preprocessed.csv"
    _write_preprocessed_csv(
        csv_path,
        [
            {"flight_id": 20, "step": 0, "x_utm": 0.0, "y_utm": 0.0, "A/D": "Start", "Runway": "09L"},
            {"flight_id": 20, "step": 1, "x_utm": 1.0, "y_utm": 1.0, "A/D": "Start", "Runway": "09L"},
            {"flight_id": 10, "step": 0, "x_utm": 3.0, "y_utm": 3.0, "A/D": "Start", "Runway": "09L"},
            {"flight_id": 10, "step": 1, "x_utm": 4.0, "y_utm": 4.0, "A/D": "Start", "Runway": "09L"},
        ],
    )

    class FakeClusterer:
        supports_precomputed = False
        last_model = None

        def fit_predict(self, X, **kwargs):
            return np.array([7, -1], dtype=int)

    monkeypatch.setattr("experiments.runner.get_clusterer", lambda method: FakeClusterer())

    cfg_path = tmp_path / "cfg.yaml"
    out_dir = tmp_path / "out"
    _write_config(cfg_path, preprocessed_csv=csv_path, output_dir=out_dir, experiment_name="EXPTEST")

    run_experiment(cfg_path)

    exp_dir = out_dir / "experiments" / "EXPTEST"
    labels = pd.read_csv(exp_dir / "labels_Start_09L.csv")
    mapping = dict(zip(labels["flight_id"].tolist(), labels["cluster_id"].tolist()))
    assert mapping[10] == 7
    assert mapping[20] == -1

    metrics = pd.read_csv(exp_dir / "metrics_by_flow.csv")
    assert metrics.loc[0, "noise_frac"] == 0.5
    assert metrics.loc[0, "n_noise_flights"] == 1
    assert metrics.loc[0, "n_clustered_flights"] == 1

    counts = pd.read_csv(exp_dir / "cluster_counts_by_flow.csv")
    assert counts.columns.tolist() == [
        "flow_label",
        "A/D",
        "Runway",
        "cluster_id",
        "n_flights",
        "is_noise_cluster",
    ]
    assert counts["n_flights"].sum() == 2
    assert set(counts["cluster_id"].tolist()) == {-1, 7}

    log_text = (exp_dir / "experiment_log.txt").read_text(encoding="utf-8")
    assert "Cluster IDs:" in log_text
    assert "Label rows check:" in log_text


def test_runner_fails_immediately_on_flow_error(tmp_path, monkeypatch):
    csv_path = tmp_path / "preprocessed.csv"
    _write_preprocessed_csv(
        csv_path,
        [
            {"flight_id": 20, "step": 0, "x_utm": 0.0, "y_utm": 0.0, "A/D": "Landung", "Runway": "09L"},
            {"flight_id": 20, "step": 1, "x_utm": 1.0, "y_utm": 1.0, "A/D": "Landung", "Runway": "09L"},
            {"flight_id": 10, "step": 0, "x_utm": 3.0, "y_utm": 3.0, "A/D": "Landung", "Runway": "09L"},
            {"flight_id": 10, "step": 1, "x_utm": 4.0, "y_utm": 4.0, "A/D": "Landung", "Runway": "09L"},
            {"flight_id": 30, "step": 0, "x_utm": 5.0, "y_utm": 5.0, "A/D": "Start", "Runway": "27R"},
            {"flight_id": 30, "step": 1, "x_utm": 6.0, "y_utm": 6.0, "A/D": "Start", "Runway": "27R"},
            {"flight_id": 40, "step": 0, "x_utm": 7.0, "y_utm": 7.0, "A/D": "Start", "Runway": "27R"},
            {"flight_id": 40, "step": 1, "x_utm": 8.0, "y_utm": 8.0, "A/D": "Start", "Runway": "27R"},
        ],
    )

    calls = {"count": 0}

    class FailingClusterer:
        supports_precomputed = False
        last_model = None

        def fit_predict(self, X, **kwargs):
            calls["count"] += 1
            if calls["count"] == 2:
                raise ValueError("boom")
            return np.zeros(X.shape[0], dtype=int)

    monkeypatch.setattr("experiments.runner.get_clusterer", lambda method: FailingClusterer())

    cfg_path = tmp_path / "cfg.yaml"
    out_dir = tmp_path / "out"
    _write_config(cfg_path, preprocessed_csv=csv_path, output_dir=out_dir, experiment_name="EXPFAIL")

    with pytest.raises(RuntimeError, match=r"Flow .* failed during clustering: boom"):
        run_experiment(cfg_path)

    exp_dir = out_dir / "experiments" / "EXPFAIL"
    assert not (exp_dir / "metrics_by_flow.csv").exists()


def test_runner_logs_effective_resampling_points(tmp_path, monkeypatch):
    csv_path = tmp_path / "preprocessed.csv"
    _write_preprocessed_csv(
        csv_path,
        [
            {"flight_id": 1, "step": 0, "x_utm": 0.0, "y_utm": 0.0, "A/D": "Start", "Runway": "09L"},
            {"flight_id": 1, "step": 1, "x_utm": 1.0, "y_utm": 1.0, "A/D": "Start", "Runway": "09L"},
            {"flight_id": 2, "step": 0, "x_utm": 3.0, "y_utm": 3.0, "A/D": "Start", "Runway": "09L"},
            {"flight_id": 2, "step": 1, "x_utm": 4.0, "y_utm": 4.0, "A/D": "Start", "Runway": "09L"},
        ],
    )

    class SingleClusterer:
        supports_precomputed = False
        last_model = None

        def fit_predict(self, X, **kwargs):
            return np.zeros(X.shape[0], dtype=int)

    cfg = {
        "clustering": {
            "method": "dbscan",
            "distance_metric": "euclidean",
            "dbscan": {"eps": 10, "min_samples": 2},
            "evaluation": {"include_noise": False},
        },
        "flows": {"flow_keys": ["A/D", "Runway"], "include": []},
        "features": {"vector_cols": ["x_utm", "y_utm"]},
        "input": {"preprocessed_csv": str(csv_path)},
        # Intentionally mismatched to verify effective-data logging.
        "preprocessing": {"resampling": {"n_points": 50}},
        "output": {"dir": str(tmp_path / "out"), "experiment_name": "EXPLOGPTS"},
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    monkeypatch.setattr("experiments.runner.get_clusterer", lambda method: SingleClusterer())
    run_experiment(cfg_path)

    log_text = (tmp_path / "out" / "experiments" / "EXPLOGPTS" / "experiment_log.txt").read_text(encoding="utf-8")
    assert "Resampling n_points: 2" in log_text
    assert "config=50 but effective=2" in log_text
