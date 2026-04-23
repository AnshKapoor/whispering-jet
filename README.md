# Thesis: Development of a clustering Framework for Aircarft Noise Simulations

This repository contains the thesis workflow used to:

1. match airport noise events to ADS-B trajectories,
2. preprocess and standardize the matched trajectories,
3. cluster flights and derive representative backbone tracks, and
4. prepare inputs for downstream ECAC Doc.29 noise simulation.

The goal of this README is not to document every utility in the repository. It is to help a thesis reader quickly understand where the main code lives, how the folders relate to each other, and which scripts correspond to each stage of the workflow.
<img width="1024" height="339" alt="image" src="https://github.com/user-attachments/assets/67ea6222-66d4-45d7-bce3-a7c325523996" />


## What This Repository Covers

The implemented workflow is:

`raw ADS-B + noise events -> matched trajectories -> preprocessed flights -> clustering experiments -> backbone tracks -> Doc.29 preparation`

The actual noise simulation model can depend on external resources and local assets that may not be distributed with the thesis-facing version of this repository. For that reason:

- [`output/`](output/README.md) should be treated as generated content,
- [`noise_simulation/`](noise_simulation/README.md) should be treated as the interface layer to the noise-simulation stage, and
- some subfolders may be empty in a shared or cleaned thesis package.

## Quick Start for a New User

Commands below are run from the repository root.

### 1. Create a Python environment

The workflow should be run inside a dedicated virtual environment. This avoids
mixing thesis dependencies with system-wide Python packages and makes the
package versions easier to reproduce.

Reference environment used for the current repository:

- Python `3.13.5`
- isolated virtual environment in `.venv/`

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

After activation, you can verify the interpreter with:

```bash
python --version
```

If you want to ensure the commands run against the virtual environment, use:

```bash
python -m pip install -r requirements.txt
```

instead of a bare `pip install`.

On Linux or macOS:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Place the raw input data

The default thesis configuration expects:

- ADS-B monthly joblib files in [`data/adsb/`](data/) with names matching `data_2022_*.joblib`
- the noise-event workbook at the repository root as `noise_data.xlsx`

These defaults come from:

- [`config/merge_adsb_noise.yaml`](config/merge_adsb_noise.yaml)

If your files are in a different location, edit that YAML first.

### 2a. Reference package versions used

The repository currently records unpinned dependencies in
[`requirements.txt`](requirements.txt), but the environment used during the
current thesis-facing setup had the following installed versions:

| Package | Version used |
|---|---|
| `joblib` | `1.5.2` |
| `matplotlib` | `3.10.7` |
| `numpy` | `2.3.4` |
| `openpyxl` | `3.1.5` |
| `pandas` | `2.3.3` |
| `pyproj` | `3.7.2` |
| `PyYAML` | `6.0.2` |
| `pytz` | `2025.2` |
| `scipy` | `1.16.2` |
| `shapely` | `2.1.2` |
| `traffic` | `2.13` |
| `scikit-learn` | `1.7.2` |
| `hdbscan` | `0.8.41` |
| `dtw-python` | `1.7.4` |
| `frechetdist` | `0.6` |
| `lcsspy` | `1.0.1` |

These versions are informative rather than enforced. A fresh install from
[`requirements.txt`](requirements.txt) may resolve to newer compatible package
versions unless the file is later pinned.

### 3. Edit the starting config for your first run

For a new machine, the first file to check is usually:

- [`config/merge_adsb_noise.yaml`](config/merge_adsb_noise.yaml)

The most important fields for a first run are:

- `noise_excel`
- `adsb.joblib_glob` or `adsb.joblibs`
- `output.output_dir`

After matching works, the next file to check is:

- [`config/backbone_full.yaml`](config/backbone_full.yaml)

The most important fields there are:

- `input.csv_glob`
- `coordinates.use_utm`
- `segmentation.*`
- `preprocessing.smoothing.*`
- `preprocessing.resampling.*`
- `output.dir`

If you want to generate multiple standardized preprocessing variants, edit:

- [`config/preprocess_grid.yaml`](config/preprocess_grid.yaml)

If you want to run clustering experiments, edit:

- [`experiments/experiment_grid.yaml`](experiments/experiment_grid.yaml)

### 4. Run the main stages

```bash
python scripts/run_merge_adsb_noise_batch.py -c config/merge_adsb_noise.yaml
python scripts/run_preprocess_grid.py --grid config/preprocess_grid.yaml
python scripts/run_experiment_grid.py --grid experiments/experiment_grid.yaml
```

If you only want one preprocessing run instead of the full grid:

```bash
python scripts/save_preprocessed.py -c config/backbone_full.yaml
```

## What to Expect After Each Step

With the current canonical configs, the usual generated locations are:

| Stage | Main script | Primary config | Typical output location |
|---|---|---|---|
| ADS-B / noise matching | [`scripts/run_merge_adsb_noise_batch.py`](scripts/run_merge_adsb_noise_batch.py) | [`config/merge_adsb_noise.yaml`](config/merge_adsb_noise.yaml) | `data/merged/` and/or [`matched_trajectories/`](matched_trajectories/) |
| Single preprocessing run | [`scripts/save_preprocessed.py`](scripts/save_preprocessed.py) | [`config/backbone_full.yaml`](config/backbone_full.yaml) | `output/preprocessed/preprocessed_<id>.csv` |
| Preprocessing grid | [`scripts/run_preprocess_grid.py`](scripts/run_preprocess_grid.py) | [`config/preprocess_grid.yaml`](config/preprocess_grid.yaml) | `output/preprocessed/` |
| Clustering experiment grid | [`scripts/run_experiment_grid.py`](scripts/run_experiment_grid.py) | [`experiments/experiment_grid.yaml`](experiments/experiment_grid.yaml) | `output/experiments/EXP###/` |
| EDA and summary plots | various scripts in [`scripts/`](scripts/) | script-specific | `output/eda/` |
| Logs | batch and workflow scripts | script-specific | `logs/experiments/` and `logs/eda/` |

For the exact output folder used by a run, check the relevant YAML:

- `output.output_dir` in [`config/merge_adsb_noise.yaml`](config/merge_adsb_noise.yaml)
- `output.dir` in [`config/backbone_full.yaml`](config/backbone_full.yaml)
- experiment-specific output settings in [`experiments/experiment_grid.yaml`](experiments/experiment_grid.yaml)

## Repository Structure

The most important folders are:

```text
Flight__Clustering/
|- README.md
|- AGENTS.md
|- requirements.txt
|- config/
|- scripts/
|- backbone_tracks/
|- clustering/
|- experiments/
|- matched_trajectories/
|- data/
|- noise_simulation/
|- output/
|- logs/
|- thesis/
`- tests/
```

### Folder guide

- [`config/`](config/)  
  Main YAML configuration files for matching, preprocessing, and clustering.

- [`scripts/`](scripts/)  
  Entry-point scripts for matching, preprocessing, EDA, plotting, and experiment-grid execution.

- [`backbone_tracks/`](backbone_tracks/)  
  Core preprocessing pipeline modules: I/O, segmentation, smoothing, resampling, clustering wrapper, backbone extraction, and plotting helpers.

- [`clustering/`](clustering/)  
  Distance-matrix construction, clustering registry, and evaluation utilities.
  For a user-facing explanation of clustering outputs, label files, and cluster numbering,
  see [`clustering/README.md`](clustering/README.md).

- [`experiments/`](experiments/)  
  Experiment runner plus experiment-grid YAML files for parameter sweeps.

- [`matched_trajectories/`](matched_trajectories/)  
  Matched ADS-B trajectory snippets created after linking noise events to ADS-B data.

- [`data/`](data/)  
  Input data area. In the standard setup, place raw ADS-B files under `data/adsb/`. Some generated intermediate files may also be written under `data/merged/`.

- [`noise_simulation/`](noise_simulation/README.md)  
  Bridge from clustering outputs to the external Doc.29 noise-simulation stage.

- [`output/`](output/README.md)  
  Generated EDA, experiments, plots, temporary files, and exported artifacts.

- [`logs/`](logs/)  
  Run logs for preprocessing, experiments, and diagnostics.

- [`thesis/docs/`](thesis/docs/)  
  Supporting documentation and registries, especially the preprocessing and experiment registries used in the thesis write-up.

## Main Workflow

### 1. Match noise events to ADS-B trajectories

Primary script:

- [`scripts/merge_adsb_noise.py`](scripts/merge_adsb_noise.py)

Batch launcher:

- [`scripts/run_merge_adsb_noise_batch.py`](scripts/run_merge_adsb_noise_batch.py)

Main config:

- [`config/merge_adsb_noise.yaml`](config/merge_adsb_noise.yaml)

Output:

- matched trajectory CSVs in [`matched_trajectories/`](matched_trajectories/) or `data/merged/`, depending on the run setup.

### 2. Preprocess matched trajectories

Primary script:

- [`scripts/save_preprocessed.py`](scripts/save_preprocessed.py)

Grid runner:

- [`scripts/run_preprocess_grid.py`](scripts/run_preprocess_grid.py)

Core implementation:

- [`backbone_tracks/io.py`](backbone_tracks/io.py)
- [`backbone_tracks/segmentation.py`](backbone_tracks/segmentation.py)
- [`backbone_tracks/preprocessing.py`](backbone_tracks/preprocessing.py)

Main configs:

- [`config/backbone_full.yaml`](config/backbone_full.yaml)
- [`config/preprocess_grid.yaml`](config/preprocess_grid.yaml)

Typical output:

- `output/preprocessed/preprocessed_<id>.csv` with the current canonical configs

### 3. Run clustering experiments

Primary script:

- [`experiments/runner.py`](experiments/runner.py)

Grid runner:

- [`scripts/run_experiment_grid.py`](scripts/run_experiment_grid.py)

Core implementation:

- [`clustering/distances.py`](clustering/distances.py)
- [`clustering/registry.py`](clustering/registry.py)
- [`clustering/evaluation.py`](clustering/evaluation.py)

Main experiment grid:

- [`experiments/experiment_grid.yaml`](experiments/experiment_grid.yaml)

Typical output:

- `output/experiments/EXP###/`

### 4. Build backbone tracks and visualize clustered flows

Main scripts:

- [`scripts/cli.py`](scripts/cli.py)
- [`scripts/plot_backbone_tracks.py`](scripts/plot_backbone_tracks.py)
- [`scripts/plot_exp_latlon_flows.py`](scripts/plot_exp_latlon_flows.py)

Core implementation:

- [`backbone_tracks/backbone.py`](backbone_tracks/backbone.py)
- [`backbone_tracks/plots.py`](backbone_tracks/plots.py)

### 5. Prepare for noise simulation

Main scripts:

- [`noise_simulation/generate_doc29_inputs.py`](noise_simulation/generate_doc29_inputs.py)
- [`noise_simulation/run_doc29_experiment.py`](noise_simulation/run_doc29_experiment.py)
- [`noise_simulation/run_doc29_ground_truth.py`](noise_simulation/run_doc29_ground_truth.py)
- [`noise_simulation/compare_experiment_totals.py`](noise_simulation/compare_experiment_totals.py)

Important note:

- the actual simulation backend and some required assets may be external to this repository; see [`noise_simulation/README.md`](noise_simulation/README.md).

## Thesis-Oriented Script Guide

If a reader wants to understand the thesis pipeline quickly, these are the first files to open:

- Matching logic: [`scripts/merge_adsb_noise.py`](scripts/merge_adsb_noise.py)
- Preprocessing entry point: [`scripts/save_preprocessed.py`](scripts/save_preprocessed.py)
- Segmentation rules: [`backbone_tracks/segmentation.py`](backbone_tracks/segmentation.py)
- Smoothing and resampling: [`backbone_tracks/preprocessing.py`](backbone_tracks/preprocessing.py)
- Experiment execution: [`experiments/runner.py`](experiments/runner.py)
- Distance metrics: [`clustering/distances.py`](clustering/distances.py)
- Metric evaluation: [`clustering/evaluation.py`](clustering/evaluation.py)
- Clustering outputs and labels guide: [`clustering/README.md`](clustering/README.md)
- Backbone extraction: [`backbone_tracks/backbone.py`](backbone_tracks/backbone.py)
- Noise-simulation interface: [`noise_simulation/generate_doc29_inputs.py`](noise_simulation/generate_doc29_inputs.py)

## Important Config Files

- [`config/merge_adsb_noise.yaml`](config/merge_adsb_noise.yaml)  
  Matching parameters such as temporal tolerance, bounding-box buffer, trajectory extraction window, and airport-distance cutoff.

- [`config/backbone_full.yaml`](config/backbone_full.yaml)  
  Main preprocessing and clustering configuration used for the full standardized workflow.

- [`config/backbone.yaml`](config/backbone.yaml)  
  Smaller or test-oriented backbone pipeline configuration.

- [`config/preprocess_grid.yaml`](config/preprocess_grid.yaml)  
  Grid of preprocessing variants.

- [`experiments/experiment_grid.yaml`](experiments/experiment_grid.yaml)  
  Main experiment grid for clustering studies.

For a structured YAML reference, see:

- [`config/README.md`](config/README.md)
- [`experiments/README.md`](experiments/README.md)
- [`config/examples/`](config/examples/)
- [`experiments/examples/`](experiments/examples/)

## YAML Templates and How to Reuse Them

If a reader wants to run the workflow on another machine, the recommended approach is to copy one of the example YAMLs instead of editing the production files directly.

### Matching template

- [`config/examples/merge_adsb_noise.example.yaml`](config/examples/merge_adsb_noise.example.yaml)

Used with:

```bash
python scripts/run_merge_adsb_noise_batch.py -c path/to/your_merge_adsb_noise.yaml
```

### Preprocessing template

- [`config/examples/backbone_full.example.yaml`](config/examples/backbone_full.example.yaml)

Used with:

```bash
python scripts/save_preprocessed.py -c path/to/your_backbone_full.yaml
```

### Preprocess-grid template

- [`config/examples/preprocess_grid.example.yaml`](config/examples/preprocess_grid.example.yaml)

Used with:

```bash
python scripts/run_preprocess_grid.py --grid path/to/your_preprocess_grid.yaml
```

### Experiment-grid template

- [`experiments/examples/experiment_grid.example.yaml`](experiments/examples/experiment_grid.example.yaml)

Used with:

```bash
python scripts/run_experiment_grid.py --grid path/to/your_experiment_grid.yaml
```

### What fields can be changed?

The field-level reference is documented here:

- matching and preprocessing configs: [`config/README.md`](config/README.md)
- experiment-grid configs: [`experiments/README.md`](experiments/README.md)

These two documents explain:

- which top-level keys are expected,
- which options are supported for smoothing, resampling, clustering method, and distance metric,
- which YAMLs are canonical thesis configs,
- and which example YAMLs are the safest starting point for a new user.

## Generated and External Folders

Some folders are generated during runs and may be empty in a clean copy:

- [`output/`](output/README.md)
- [`logs/`](logs/)

Some folders may require local or external resources that are not bundled with the thesis package:

- [`noise_simulation/`](noise_simulation/README.md)
- [`noise_simulation/doc-29-implementation/`](noise_simulation/doc-29-implementation/README.md)

## Supporting Documentation

Useful thesis-facing documents:

- [`thesis/docs/preprocessed_registry.md`](thesis/docs/preprocessed_registry.md)
- [`thesis/docs/CODEBASE_OVERVIEW.md`](thesis/docs/CODEBASE_OVERVIEW.md)
- [`CHANGELOG_STANDARDIZATION.md`](CHANGELOG_STANDARDIZATION.md)

## Running the Main Steps

Commands are run from the repository root.

### Match ADS-B and noise data

```bash
python scripts/run_merge_adsb_noise_batch.py -c config/merge_adsb_noise.yaml
```

### Save one preprocessed dataset

```bash
python scripts/save_preprocessed.py -c config/backbone_full.yaml
```

### Run a preprocessing grid

```bash
python scripts/run_preprocess_grid.py --grid config/preprocess_grid.yaml
```

### Run an experiment grid

```bash
python scripts/run_experiment_grid.py --grid experiments/experiment_grid.yaml
```

### Plot experiment-level clustered flows

```bash
python scripts/plot_exp_latlon_flows.py EXP001
```

## Development Notes

- The standardized experiment outputs are expected under `output/experiments/<EXP>/`.
- The standardized EDA outputs are expected under `output/eda/`.
- The thesis workflow uses UTM coordinates for clustering-distance calculations.
- Only the four operational runways `09L`, `09R`, `27L`, and `27R` are treated as the standard runway set unless a script explicitly states otherwise.

## VS Code Launch Configurations

The repository already contains launch configurations in [`.vscode/launch.json`](.vscode/launch.json). These are useful if the reader wants to reproduce the main stages without rebuilding command lines manually.
