# Flight Trajectory Clustering and Noise Simulation

Thesis codebase to match ADS-B trajectories with noise measurements, cluster flights, generate backbone tracks, and build ECAC Doc29 noise simulation inputs.

## Setup
- Python 3.8+
- Install deps: `pip install -r requirements.txt`
- Optional for Parquet output: `pip install pyarrow` (or `fastparquet`)

## Pipeline at a glance
ADS-B joblib -> (optional) Parquet -> match noise to ADS-B -> matched_trajectories CSV
-> preprocessing and clustering -> backbone tracks -> Doc29 groundtracks -> Doc29 inputs -> noise simulation -> plots

## Inputs and outputs
- ADS-B joblib: `adsb/*.joblib`
- Noise Excel: `noise_data.xlsx`
- Matched trajectories: `matched_trajectories/*.csv` (or `data/merged` if `--output-dir` is set)
- Preprocessed output: `data/preprocessed/preprocessed_*.csv`
- Experiment outputs: `output/experiments/<experiment>/`
- Doc29 inputs: `noise_simulation/doc-29-implementation/Groundtracks/<EXP>` and `Flight_<EXP>.csv`
- Doc29 results: `noise_simulation/doc-29-implementation/Results_Python/*.csv`

## Scripts and how to run them
Run commands from the repo root. VS Code launch names refer to `.vscode/launch.json`.

### Dataset generation (ADS-B + noise)

#### `scripts/convert_adsb_joblib_to_csv.py`
Purpose: Convert ADS-B joblib files to Parquet.

CLI:
```bash
python scripts/convert_adsb_joblib_to_csv.py --input-dir adsb --glob "*.joblib" --output-dir data/adsb_parquet --chunksize 200000 --compression snappy
```
Single file:
```bash
python scripts/convert_adsb_joblib_to_csv.py --joblib adsb/data_2022_april.joblib --output-dir data/adsb_parquet
```
VS Code launch: `Convert ADSB Joblib to Parquet (April)`

#### `scripts/merge_adsb_noise.py`
Purpose: Match noise measurements (noise_data.xlsx) to ADS-B joblib and write matched trajectories to Parquet + CSV.

CLI:
```bash
python scripts/merge_adsb_noise.py noise_data.xlsx adsb/data_2022_april.joblib --traj-output matched_trajs_april_2022.parquet --output-dir data/merged --tol-sec 10 --buffer-frac 0.5 --window-min 3 --max-airport-distance-km 25
```
Matching logic (defaults):
- Time match window: keep ADS-B samples with `|t - t_ref| <= 10 s`.
- Spatial match window: let `d = Abstand [m]`, `r = d * (1 + 0.5) = 1.5 d`. Convert to degrees:
  - `dlat = r / 111195`
  - `dlon = dlat / cos(lat0)`
  Keep samples with `lat in [lat0 - dlat, lat0 + dlat]` and `lon in [lon0 - dlon, lon0 + dlon]`.
- If any hits exist, the first hit provides `icao24`/`callsign` for that noise row.
- Trajectory window: keep samples with `|t - t_ref| <= 3 min`, filtered to the matched `icao24`/`callsign` when available.
- Airport filter: compute UTM distance `d_airport = sqrt((x-ax)^2 + (y-ay)^2)` and keep `d_airport <= 25000 m`.
- Downsample: keep at most one sample every `2 s` by 2-second time bins.
- Deduplicate: drop duplicates by `(MP, t_ref, icao24, timestamp)`.
VS Code launch: `Python Debugger: Current File with Arguments` (open `scripts/merge_adsb_noise.py` first)

#### `scripts/run_merge_adsb_noise_batch.py`
Purpose: Batch matching for multiple joblibs via config.

CLI:
```bash
python scripts/run_merge_adsb_noise_batch.py -c config/merge_adsb_noise.yaml
```
VS Code launch: none

### Preprocessing and backbone clustering

#### `scripts/save_preprocessed.py`
Purpose: Run segmentation + preprocessing only and save `data/preprocessed/preprocessed_*.csv`.

CLI:
```bash
python scripts/save_preprocessed.py -c config/backbone_full.yaml
```
VS Code launch: `Save Preprocessed Data`

#### `scripts/run_preprocess_grid.py`
Purpose: Run a standardized preprocessing grid (n_points, interpolation, smoothing).

CLI:
```bash
python scripts/run_preprocess_grid.py --grid config/preprocess_grid.yaml
```
VS Code launch: `Run Preprocess Grid`

#### `scripts/cli.py`
Purpose: Full backbone pipeline (load, segment, preprocess, cluster, backbone export).

CLI:
```bash
python scripts/cli.py -c config/backbone_full.yaml
```
Test mode config:
```bash
python scripts/cli.py -c config/backbone.yaml
```
VS Code launch: `Backbone Clustering (Full Mode)` and `Backbone Clustering (Test Mode)`

#### `experiments/runner.py`
Purpose: Run extended clustering + evaluation on a preprocessed CSV.

CLI:
```bash
python experiments/runner.py -c config/experiments/global_optics.yaml --preprocessed data/preprocessed/preprocessed_1.csv
```
VS Code launch: `Experiment Runner - OPTICS`, `Experiment Runner - DBSCAN`, `Experiment Runner - HDBSCAN`, `Experiment Runner - KMeans`, `Experiment Runner - Agglomerative`, and `Experiment Runner - GLOBAL OPTICS/DBSCAN/HDBSCAN/KMeans/Agglomerative`

#### `scripts/run_experiment_grid.py`
Purpose: Run parameter sweeps defined in a grid YAML.

CLI:
```bash
python scripts/run_experiment_grid.py --grid experiments/experiment_grid.yaml
```
VS Code launch: `Run Experiment Grid`

#### `scripts/eda_raw_lengths.py`
Purpose: Pre-assess raw flight length distribution before preprocessing filters.

CLI:
```bash
python scripts/eda_raw_lengths.py -c config/backbone_full.yaml --outdir output/eda/raw_lengths
```
VS Code launch: `EDA Raw Flight Lengths`

### Visualization and reporting

#### `scripts/plot_experiment_results.py`
Purpose: Plot experiment metrics, trajectories, and derived backbones under `output/experiments/<experiment>/plots`.

CLI:
```bash
python scripts/plot_experiment_results.py EXP77 --max-flights-per-cluster 30
```
VS Code launch: `Plot Experiment Results` and `Generate Backbone Tracks`

Note: the legacy script `scripts/plot_exp_latlon.py` was archived to `legacy/scripts/`.

#### `scripts/plot_backbone_tracks.py`
Purpose: Generate backbone tracks per flow/cluster (lat/lon) and Doc29-style arrival overlays.

CLI:
```bash
python scripts/plot_backbone_tracks.py --experiment EXP77 --arrival-scheme seven
```
VS Code launch: `Plot Backbone Tracks (Lat/Lon)`

### Noise simulation (Doc29)

#### `scripts/doc29_tracks.py`
Purpose: Create Doc29 7-track layouts (center + offsets) from preprocessed data and optionally export groundtrack CSVs.

CLI:
```bash
python scripts/doc29_tracks.py --preprocessed data/preprocessed/preprocessed_1.csv --all-flows --export-dir output/eda/doc29_tracks --out output/eda/doc29_tracks.png
```
With cluster labels:
```bash
python scripts/doc29_tracks.py --preprocessed data/preprocessed/preprocessed_1.csv --labels output/experiments/EXP77 --all-flows --export-dir output/eda/doc29_tracks
```
Note: preprocessed CSV must include `x_utm`, `y_utm`, `step`, `flight_id`, `A/D`, and `Runway` (generated when `coordinates.use_utm: true`).
VS Code launch: none

#### `noise_simulation/generate_doc29_inputs.py`
Purpose: Convert exported groundtracks into Doc29 inputs (Groundtracks + Flight_EXP*.csv).

CLI:
```bash
python noise_simulation/generate_doc29_inputs.py --tracks-root output/eda/doc29_tracks --preprocessed data/preprocessed/preprocessed_1.csv --experiment-name EXP46 --output-root noise_simulation/doc-29-implementation --combine
```
VS Code launch: none

#### `noise_simulation/doc-29-implementation/main.py`
Purpose: Run the Doc29 noise calculation.

How to run:
- Edit `noise_simulation/doc-29-implementation/main.py` to set `input_file_flights` and output file name.
- Then run:
```bash
python noise_simulation/doc-29-implementation/main.py
```
See `noise_simulation/doc-29-implementation/README.md` for model details and required assets.
VS Code launch: none

#### `noise_simulation/compare_experiment_totals.py`
Purpose: Fair per-experiment totals comparison between clustered prediction and individual-flight ground truth.

CLI:
```bash
python noise_simulation/compare_experiment_totals.py --summary noise_simulation/results/EXP001/summary_mse.csv --out noise_simulation/results/EXP001/aggregate_totals --subtracks-weighting weighted
```
Key outputs:
- `overall_aligned_9points.csv` (9 receiver points side-by-side)
- `overall_summary.json` (MAE/MSE/RMSE and average cumulative level delta)

#### `scripts/plot_noise_results.py`
Purpose: Plot noise simulation results from `Results_Python/*.csv`.

CLI:
```bash
python scripts/plot_noise_results.py noise_simulation/doc-29-implementation/Results_Python/EXP46.csv --column cumulative_res --mode contour
```
VS Code launch: none

#### `scripts/plot_metrics_doc29.py`
Purpose: Plot Doc29 experiment metrics.

CLI:
```bash
python scripts/plot_metrics_doc29.py --stage2 output/eda/metrics_stage2_by_flow.csv --quality output/eda/metrics_quality_global.csv --out-dir output/eda/figures
```
VS Code launch: none

## Key config files
- `config/merge_adsb_noise.yaml`: batch matching settings
- `config/backbone.yaml`: test-mode clustering pipeline
- `config/backbone_full.yaml`: full clustering pipeline
- `config/experiments/global_*.yaml`: global experiment configs
- `experiments/named_experiments.yaml`: grid sweep definitions
- `config/backbone_legacy.yaml`: legacy backbone tracks config

## Config tips
- If you store matched trajectories in a different folder, update `input.csv_glob` in `config/backbone_full.yaml` and `config/backbone.yaml`.

## Slurm helpers
- `jobs/run_adsb_joblib_to_csv.job`: convert all joblib files in a directory (usage: `sbatch jobs/run_adsb_joblib_to_csv.job /path/to/joblib_dir`)
- `jobs/run_merge_adsb_noise.job`: single-month noise matching
- `jobs/run_preprocessing_grid.job`: preprocessing grid
- `jobs/run_experiment_grid.job`: experiment grid
- `jobs/run_adsb_monthly_eda.job`: monthly ADS-B EDA
- `jobs/run_doc29_ground_truth.job`: batched Doc29 ground truth
- `jobs/run_doc29_experiment.job`: Doc29 experiment runner
- `jobs/run_doc29_compare_totals.job`: totals comparison (cluster vs ground truth)

Archived jobs/scripts are under `legacy/`.
