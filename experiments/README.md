# Experiment Grid Files

This folder contains the YAML files used to launch clustering experiments and parameter sweeps.

The main execution entry points are:

- [`runner.py`](runner.py) for a single resolved experiment configuration,
- [`../scripts/run_experiment_grid.py`](../scripts/run_experiment_grid.py) for a YAML-defined experiment grid.

## Main Files

- [`experiment_grid.yaml`](experiment_grid.yaml)  
  Main experiment grid used for the thesis experiment series.

- [`examples/experiment_grid.example.yaml`](examples/experiment_grid.example.yaml)  
  Minimal, annotated example for new users.

The other `experiment_grid_*.yaml` files in this folder are specialized sweeps created for specific study phases.

## Available Experiment Grid Files

- [`experiment_grid.yaml`](experiment_grid.yaml)
- [`experiment_grid.before_exp032_036_only.yaml`](experiment_grid.before_exp032_036_only.yaml)
- [`experiment_grid_021_024_dtw_dbscan.yaml`](experiment_grid_021_024_dtw_dbscan.yaml)
- [`experiment_grid_045_dtw_dense.yaml`](experiment_grid_045_dtw_dense.yaml)
- [`experiment_grid_051_070_dtw_frechet.yaml`](experiment_grid_051_070_dtw_frechet.yaml)
- [`experiment_grid_061_070_frechet.yaml`](experiment_grid_061_070_frechet.yaml)
- [`experiment_grid_089_100_dtw_dense.yaml`](experiment_grid_089_100_dtw_dense.yaml)
- [`experiment_grid_101_110_optics_tuning.yaml`](experiment_grid_101_110_optics_tuning.yaml)
- [`experiment_grid_111_120_hdbscan_dtw_tuning.yaml`](experiment_grid_111_120_hdbscan_dtw_tuning.yaml)
- [`experiment_grid_121_126_hdbscan_dtw_refine.yaml`](experiment_grid_121_126_hdbscan_dtw_refine.yaml)
- [`experiment_grid_127_131_weighted_euclidean_dbscan.yaml`](experiment_grid_127_131_weighted_euclidean_dbscan.yaml)
- [`experiment_grid_132_pre11_dtw_dbscan.yaml`](experiment_grid_132_pre11_dtw_dbscan.yaml)
- [`experiment_grid_133_137_weighted_euclidean_dbscan_reversed.yaml`](experiment_grid_133_137_weighted_euclidean_dbscan_reversed.yaml)
- [`experiment_grid_138_142_weighted_euclidean_hdbscan.yaml`](experiment_grid_138_142_weighted_euclidean_hdbscan.yaml)
- [`experiment_grid_151_dtw_dbscan_minpts160.yaml`](experiment_grid_151_dtw_dbscan_minpts160.yaml)
- [`examples/experiment_grid.example.yaml`](examples/experiment_grid.example.yaml)

## Grid Structure

```yaml
experiments:
  - name: "EXP001"
    experiment_name: "EXP001"
    method: "optics"
    distance_metric: "euclidean"
    optics:
      min_samples: 12
      xi: 0.04
      min_cluster_size: 0.04
    evaluation:
      include_noise: false
    input:
      preprocessed_csv: "output/preprocessed/preprocessed_1.csv"

flows:
  flow_keys: ["A/D", "Runway"]
  include: []

input:
  preprocessed_csv: "output/preprocessed/preprocessed_1.csv"

output:
  dir: "output"
  skip_completed: true

features:
  vector_cols: ["x_utm", "y_utm"]
```

## Root-Level Grid Fields

- `experiments`  
  List of experiment definitions. This field is required.

- `flows`  
  Optional shared flow settings applied to all experiments.

- `input.preprocessed_csv`  
  Optional shared preprocessed input path used unless an experiment overrides it.

- `output.dir`  
  Base output directory.

- `output.skip_completed`  
  If `true`, already completed experiment folders are skipped.

- `features`  
  Optional shared feature settings applied across experiments.

## Per-Experiment Fields

Each item inside `experiments:` may contain:

- `name`  
  Human-readable identifier for the grid entry.

- `experiment_name`  
  Final output experiment folder name, for example `EXP021`. Use this when numbering must stay fixed.

- `method`  
  Clustering method name.

- `distance_metric`  
  Distance metric name.

- `distance_params`  
  Additional metric-specific settings.

- `evaluation`  
  Evaluation options merged into the clustering config.

- `sample_for_fit`  
  Optional sampling configuration used before fitting.

- `features`  
  Optional experiment-specific feature overrides.

- `input`  
  Optional experiment-specific input overrides, especially `preprocessed_csv`.

- one method-specific parameter block matching `method`, such as `optics`, `dbscan`, or `hdbscan`

## Supported Clustering Methods

The current clustering registry supports these method names:

- `optics`
- `dbscan`
- `hdbscan`
- `kmeans`
- `minibatch_kmeans`
- `agglomerative`
- `birch`
- `spectral`
- `gmm`
- `meanshift`
- `affinity_propagation`
- `two_stage`

These are defined in [`../clustering/registry.py`](../clustering/registry.py).

## Supported Distance Metrics

The current distance layer supports:

- `euclidean`
- `euclidean_weighted`
- `dtw`
- `frechet`
- `lcss`

These are implemented in [`../clustering/distances.py`](../clustering/distances.py).

## Metric-Specific Notes

### Weighted Euclidean

Typical extra fields inside `distance_params`:

- `weighted_n_dims`
- `weighted_groups_by_operation`

### DTW

Typical extra fields inside `distance_params`:

- `dtw_window_size` or `window`
- `use_lb_keogh`
- `lb_keogh_radius`
- dense/sparse precomputed-distance options when used by the experiment setup

### Frechet

Typical extra fields inside `distance_params`:

- `rdp_epsilon`

### LCSS

Typical extra fields inside `distance_params`:

- `lcss_epsilon_m`
- `lcss_delta_alpha`
- `lcss_normalization`

## Important Practical Note

Method-specific parameter blocks are passed through to the underlying clustering implementation. That means:

- `optics:` fields should match what the OPTICS wrapper expects,
- `dbscan:` fields should match DBSCAN parameters,
- `hdbscan:` fields should match HDBSCAN parameters,
- and so on.

For a new user, the safest approach is:

1. copy [`examples/experiment_grid.example.yaml`](examples/experiment_grid.example.yaml),
2. change only `experiment_name`, `method`, `distance_metric`, the relevant method block, and `input.preprocessed_csv`,
3. keep the other structure unchanged until the run works.
