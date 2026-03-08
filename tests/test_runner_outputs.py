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


def test_runner_lcss_sample_for_fit_metadata(tmp_path, monkeypatch):
    csv_path = tmp_path / "preprocessed.csv"
    rows = []
    for fid in [1, 2, 3, 4, 5]:
        rows.append(
            {
                "flight_id": fid,
                "step": 0,
                "x_utm": float(fid),
                "y_utm": float(fid),
                "A/D": "Landung",
                "Runway": "09L",
            }
        )
        rows.append(
            {
                "flight_id": fid,
                "step": 1,
                "x_utm": float(fid) + 1.0,
                "y_utm": float(fid) + 1.0,
                "A/D": "Landung",
                "Runway": "09L",
            }
        )
    _write_preprocessed_csv(csv_path, rows)

    class FakeKMeans:
        supports_precomputed = False
        last_model = None

        def fit_predict(self, X, **kwargs):
            n = X.shape[0]
            base = np.array([0, 1, 0], dtype=int)
            if n <= 3:
                return base[:n]
            return np.resize(base, n)

    def fake_pairwise_distance_matrix(*args, **kwargs):
        trajs = args[0]
        n = len(trajs)
        D = np.full((n, n), 2.0, dtype=float)
        np.fill_diagonal(D, 0.0)
        return D

    monkeypatch.setattr("experiments.runner.get_clusterer", lambda method: FakeKMeans())
    monkeypatch.setattr("experiments.runner.pairwise_distance_matrix", fake_pairwise_distance_matrix)

    cfg = {
        "clustering": {
            "method": "kmeans",
            "distance_metric": "lcss",
            "kmeans": {"n_clusters": 2, "random_state": 42, "n_init": 1},
            "distance_params": {
                "lcss_epsilon_m": 300.0,
                "lcss_delta_alpha": 0.10,
                "lcss_normalization": "min_len",
                "mds_n_components": 2,
            },
            "sample_for_fit": {
                "enabled": True,
                "max_flights_per_flow": 3,
                "random_state": 11,
                "mode": "sample_only",
            },
            "evaluation": {"include_noise": False},
        },
        "flows": {"flow_keys": ["A/D", "Runway"], "include": []},
        "features": {"vector_cols": ["x_utm", "y_utm"]},
        "input": {"preprocessed_csv": str(csv_path)},
        "preprocessing": {"resampling": {"n_points": 2}},
        "output": {"dir": str(tmp_path / "out"), "experiment_name": "EXPLCSSSAMPLE"},
    }
    cfg_path = tmp_path / "cfg_lcss.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    run_experiment(cfg_path)

    exp_dir = tmp_path / "out" / "experiments" / "EXPLCSSSAMPLE"
    labels = pd.read_csv(exp_dir / "labels_Landung_09L.csv")
    assert len(labels) == 3

    metrics = pd.read_csv(exp_dir / "metrics_by_flow.csv")
    assert int(metrics.loc[0, "n_flights_total_flow"]) == 5
    assert int(metrics.loc[0, "n_flights_used_for_fit"]) == 3
    assert metrics.loc[0, "fit_sampling_mode"] == "sample_only"
