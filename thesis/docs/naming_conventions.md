#  Naming & File Organization Conventions followed

## General Rules
- Use **lowercase only**, no spaces. Use `_` for file stems and `-` for folder names.
- Allowed characters: `a–z`, `0–9`, `_`, `-`.
- Prefer **ISO dates**: `YYYYMMDD` or `YYYYMMDD_HHMM`.
- Keep names **short but descriptive** (≤ 50 characters recommended).
- Use **single extensions** (`.py`, `.csv`, `.ipynb`, etc.).

---

## Directory Names
Use **kebab-case**:
data/raw
data/interim
data/processed
reports/figures
reports/tables
reports/logs
reports/notebooks 

---

## Python Code
| Element | Convention | Example |
|----------|-------------|----------|
| File/module | `snake_case.py` | `trajectory_preprocessing.py` |
| Class | `PascalCase` | `TrajectoryCleaner` |
| Function / variable | `snake_case` | `preprocess_trajectory` |
| Constant | `UPPER_SNAKE` | `DEFAULT_TARGET_LENGTH = 100` |

---

## Notebooks
Prefix with a step number and short action:
00_colab_setup.ipynb
01_prepare_dataset.ipynb
02_extract_trajectories.ipynb
03_clustering_optics.ipynb
04_results_and_figures.ipynb

---

## Config & Experiment Files
- Global configs:  
  `paths.yaml`, `params.yaml`
- Per-experiment overrides:  
  `exp-YYYYMMDD-<tag>.yaml` → e.g. `exp-20251027-optics-scan.yaml`

---

## Data Files
| Type | Pattern | Example |
|------|----------|----------|
| Raw | `<domain>_<source>_<granularity>_<date>.<ext>` | `noise_bestpractice_by_mp_201905.xlsx` |
| Interim | `<artifact>_interim_<date>.<ext>` | `flights_noise_bp_weather_interim_201905.parquet` |
| Processed | `<artifact>_processed_<date>.<ext>` | `trajectories_processed_201905.parquet` |

---

## Figures & Tables
- Figures: `fig-<topic>_<metric>_<date>.<ext>`  
  e.g. `fig-clustering_dbindex_20251027.png`
- Tables: `tbl-<topic>_<summary>_<date>.csv`  
  e.g. `tbl-optics_hyperparams_20251027.csv`

---

## Logs & Runs
- Metrics/logs: `run-YYYYMMDD_HHMM-<tag>.json`  
  → `run-20251027_1422-optics-baseline.json`
- Checkpoints: `ckpt-<model>-<step>.pkl`

---

## CLI Scripts
Action-oriented 
`snake_case.py`:
`prepare_dataset.py`
`extract_trajectories.py`
`run_clustering.py`
`make_figures.py`

---

## Versioning Suffix
Add `-vNN` before extension for major snapshots:

`trajectories_processed_201905-v02.parquet`

---

## Temporary or Local Files
Prefix with `_tmp-` and exclude via `.gitignore`:
`_tmp-scratch.ipynb`
`_tmp-sample.csv`

---

## Quick Examples
* `fig-noise_contours_day_20251027.pdf`  
* `run-20251027_1422-optics-baseline.json`  
* `Final Figure (latest).pdf`  
* `TrajectoryProcessing2 (copy).ipynb`
