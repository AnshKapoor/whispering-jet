# Configuration Files

This folder contains the main YAML configuration files used by the thesis workflow.

The recommended way to understand and reuse them is:

1. start from one of the example YAMLs,
2. compare it with the canonical production config, and
3. adjust only the fields relevant to your run.

## Canonical Files

- [`merge_adsb_noise.yaml`](merge_adsb_noise.yaml)  
  Batch configuration for linking noise events to ADS-B data and writing matched trajectories.

- [`backbone_full.yaml`](backbone_full.yaml)  
  Main end-to-end preprocessing and clustering configuration used as the base for standardized runs.

- [`backbone.yaml`](backbone.yaml)  
  Smaller or test-oriented backbone configuration.

- [`backbone_full_corrected.yaml`](backbone_full_corrected.yaml)  
  Corrected full backbone configuration variant kept for reference.

- [`preprocess_grid.yaml`](preprocess_grid.yaml)  
  Grid of preprocessing variants built on top of a base config.

- [`preprocess_grid_corrected_1_10.yaml`](preprocess_grid_corrected_1_10.yaml)  
  Variant grid focused on the corrected preprocessing set.

- [`velocity_leakage_audit.yaml`](velocity_leakage_audit.yaml)  
  Audit configuration for Euclidean/velocity-leakage diagnostics.

- [`params.yaml`](params.yaml)  
  Auxiliary parameter file retained from earlier workflow stages.

- [`paths.yaml`](paths.yaml)  
  Auxiliary path settings retained for compatibility/reference.

## Example Templates

- [`examples/merge_adsb_noise.example.yaml`](examples/merge_adsb_noise.example.yaml)
- [`examples/backbone_full.example.yaml`](examples/backbone_full.example.yaml)
- [`examples/preprocess_grid.example.yaml`](examples/preprocess_grid.example.yaml)

## 1. Matching Config: `merge_adsb_noise.yaml`

This YAML is consumed by [`scripts/run_merge_adsb_noise_batch.py`](../scripts/run_merge_adsb_noise_batch.py), which calls [`scripts/merge_adsb_noise.py`](../scripts/merge_adsb_noise.py).

### Structure

```yaml
noise_excel: ...

adsb:
  joblib_glob: ...
  joblibs: ...

output:
  traj_output_template: ...
  output_dir: ...

matching:
  tol_sec: ...
  buffer_frac: ...
  window_min: ...
  sample_interval_sec: ...
  max_airport_distance_km: ...
  dedupe_traj: ...

testing:
  enabled: ...
  match_limit: ...

logging:
  level: ...
  log_file: ...
```

### Field notes

- `noise_excel`  
  Path to the noise-event workbook.

- `adsb.joblib_glob`  
  Glob used to find monthly ADS-B joblib files.

- `adsb.joblibs`  
  Optional explicit list of joblib files. Use this if you do not want a glob.

- `output.traj_output_template`  
  Output file naming pattern for matched trajectory exports.

- `output.output_dir`  
  Folder where matched CSV and Parquet files are written.

- `matching.tol_sec`  
  Temporal tolerance for identifying ADS-B hits near a noise event.

- `matching.buffer_frac`  
  Fractional expansion of the microphone-centred spatial search radius.

- `matching.window_min`  
  Half-window in minutes for trajectory extraction around the reference time.

- `matching.sample_interval_sec`  
  Minimum kept spacing between ADS-B samples inside the extracted slice.

- `matching.max_airport_distance_km`  
  Airport-distance cutoff used to keep only terminal-area points.

- `matching.dedupe_traj`  
  Whether repeated ADS-B rows in the extracted trajectory slices are deduplicated.

## 2. Backbone / Preprocessing Config: `backbone_full.yaml`

This YAML is used by:

- [`scripts/save_preprocessed.py`](../scripts/save_preprocessed.py)
- [`scripts/cli.py`](../scripts/cli.py)
- indirectly by [`scripts/run_preprocess_grid.py`](../scripts/run_preprocess_grid.py)
- indirectly by [`scripts/run_experiment_grid.py`](../scripts/run_experiment_grid.py), which uses it as the base config

### Main sections

```yaml
input:
coordinates:
segmentation:
flows:
preprocessing:
clustering:
backbone:
testing:
output:
logging:
```

### Input fields

- `input.csv_glob`  
  Matched trajectory CSV input pattern.

- `input.parse_dates`  
  Columns parsed as datetimes during CSV loading.

- `input.timezone`  
  Informational timezone setting used in the workflow configuration.

- `input.preprocessed_csv`  
  Preprocessed input path used later by experiment runs.

### Coordinate fields

- `coordinates.use_utm`  
  If `true`, add `x_utm` and `y_utm` and use them in downstream clustering logic.

- `coordinates.utm_crs`  
  UTM CRS string, for example `epsg:32632`.

### Segmentation fields

- `segmentation.time_gap_sec`
- `segmentation.distance_jump_m`
- `segmentation.min_points_per_flight`
- `segmentation.split_on_identity`

These fields control when the matched ADS-B rows are split into separate `flight_id` trajectories.

### Flow fields

- `flows.flow_keys`  
  Columns used to define flow partitions. The standard thesis setup is `["A/D", "Runway"]`.

- `flows.include`  
  Optional whitelist of flows to keep.

### Preprocessing fields

#### `preprocessing.filter`

- `max_airport_distance_km`
- `min_length_km`
- `allowed_runways`
- `simplify.rdp_enabled`
- `simplify.rdp_epsilon_m`
- `simplify.rdp_min_points`

#### `preprocessing.repetition_check`

- `enabled`
- `window_minutes`
- `timezone`
- `require_same_date`
- `identity`
- `action`
- `keep_policy`
- `output_dir`

#### `preprocessing.smoothing`

- `enabled`
- `method`
- `window_length`
- `polyorder`
- `columns`

Supported smoothing methods in the current code are:

- `auto`
- `savgol`
- `savitzky_golay`
- `moving_average`
- `median`
- `ewm`
- `none`

#### `preprocessing.resampling`

- `enabled`
- `n_points`
- `method`

Supported resampling modes in the current code are:

- `time`
- `index`
- `none` / disabled mode through `enabled: false`

### Clustering fields

- `clustering.method`
- `clustering.distance_metric`
- `clustering.distance_params`
- `clustering.sample_for_fit`
- `clustering.evaluation`
- method-specific blocks such as `optics`, `dbscan`, `hdbscan`, `kmeans`
- `clustering.random_state`

Supported distance metrics in the current code are:

- `euclidean`
- `euclidean_weighted`
- `dtw`
- `frechet`
- `lcss`

For clustering methods, see [`../experiments/README.md`](../experiments/README.md), because the same method names and parameter blocks are reused there.

## 3. Preprocess Grid Config: `preprocess_grid.yaml`

This YAML is consumed by [`scripts/run_preprocess_grid.py`](../scripts/run_preprocess_grid.py).

### Structure

```yaml
base_config: config/backbone_full.yaml

variants:
  - preprocessed_id: 1
    preprocessing:
      ...
  - preprocessed_id: 2
    preprocessing:
      ...
```

### How it works

- `base_config` points to a full backbone-style YAML.
- each item in `variants` overrides part of that base config,
- and the script writes `preprocessed_<id>.csv` outputs.

The merge logic is recursive for nested dictionaries, so a variant can override only the fields it needs.

## Recommended Usage

For a new user:

1. start with [`examples/backbone_full.example.yaml`](examples/backbone_full.example.yaml),
2. if multiple preprocessing variants are needed, copy [`examples/preprocess_grid.example.yaml`](examples/preprocess_grid.example.yaml),
3. if matched trajectories still need to be created, start from [`examples/merge_adsb_noise.example.yaml`](examples/merge_adsb_noise.example.yaml),
4. for experiment sweeps, use [`../experiments/examples/experiment_grid.example.yaml`](../experiments/examples/experiment_grid.example.yaml).
