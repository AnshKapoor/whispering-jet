# Standardization Changelog

This log captures structural changes made to standardize experiments, outputs, and logs.

## 2026-03-17
- Added `experiments/experiment_grid_111_120_hdbscan_dtw_tuning.yaml` with DTW+HDBSCAN pilot experiments (`EXP111-EXP120`) for low-noise parameter tuning on `preprocessed_1` with deterministic per-flow sampling (`500/flow`).
- Sweep includes `min_cluster_size`, `min_samples`, `cluster_selection_method` (`eom` vs `leaf`), `allow_single_cluster`, and `cluster_selection_epsilon` under dense DTW settings.
- Added VS Code launch entry `Grid: Run EXP111-120 HDBSCAN DTW Tuning` in `.vscode/launch.json`.
- Noted manual config updates to `dtw_window_size` for selected initial DTW experiment definitions in experiment-grid YAMLs (kept as user-driven tuning changes).
- Added `experiments/experiment_grid_121_126_hdbscan_dtw_refine.yaml` as a narrower DTW+HDBSCAN follow-up around `EXP111`, targeting arrival recovery and lower noise with `eom`, `allow_single_cluster=true`, reduced `min_samples`, and small `cluster_selection_epsilon` / `dtw_window_size` sweeps.
- Added VS Code launch entry `Grid: Run EXP121-126 HDBSCAN DTW Refine` in `.vscode/launch.json`.
- Updated `experiments/experiment_grid.yaml` so only `EXP037-EXP045` remain active for the final rerun block: preserved `EXP037-EXP040` GMM baselines and repurposed `EXP041-EXP045` to full-data DTW (`HDBSCAN`/`OPTICS`) with dense precomputed evaluation enabled (`sparse_precomputed_max_n: 9000`) to emit silhouette / Davies-Bouldin / Calinski-Harabasz metrics.

## 2026-03-14
- Added dense-DTW pilot grid `experiments/experiment_grid_089_100_dtw_dense.yaml` for `EXP089-EXP100` across `kmeans`, `dbscan`, `hdbscan`, and `optics` on `preprocessed_1` with `sample_for_fit.max_flights_per_flow=900`.
- Updated `scripts/run_experiment_grid.py` to support per-experiment `evaluation` overrides merged into `clustering.evaluation`.
- Added `scripts/probe_dtw_dbscan_eps.py` for deterministic DBSCAN `eps` probing from DTW k-distance knees (per-flow + median `eps_base` summary).
- Added `scripts/rank_dtw_pilot_and_promote.py` to rank DTW pilot runs with a composite score and optionally rewrite `EXP021-EXP024` in `experiments/experiment_grid.yaml`.
- Added VS Code launch entries for running the dense-DTW pilot grid, probing DTW DBSCAN `eps`, and ranking/promoting pilot winners.

## 2026-03-15
- Added `scripts/attach_global_ground_truth_to_experiment_totals.py` to append all-flights ground truth columns to `aggregate_totals/overall_aligned_9points.csv` and update `overall_summary.json`.
- Documented cloud-to-local Doc29 workflow: create `_summary_mse_local_paths.csv` from `summary_mse.csv`, then run `noise_simulation/compare_experiment_totals.py` to produce `aggregate_totals/`.

## 2026-02-02
- Standardized experiment outputs under `output/experiments/<EXP>/`.
- Standardized EDA outputs under `output/eda/` with logs in `logs/eda/`.
- Standardized preprocessed outputs under `data/preprocessed/preprocessed_<id>.csv`.
- Updated `scripts/save_preprocessed.py` to auto-assign IDs and update `thesis/docs/preprocessed_registry.md`.
- Added `config/preprocess_grid.yaml` + `scripts/run_preprocess_grid.py` for standardized preprocessing variants.
- Updated clustering runner to write labels as CSV and enforce UTM-only vector columns.
- Added `scripts/plot_backbone_tracks.py` for per-flow/cluster backbone plots in lat/lon.
- Removed redundant visualization script `visualize_outputs.py` and its job.

## 2026-02-03 13:50:19
- Ran raw-length EDA (`scripts/eda_raw_lengths.py`) and selected p25 length threshold.
- Updated preprocessing minimum length filter to 9.35 km (p25) in `config/backbone_full.yaml` and `config/backbone.yaml`.
- Simplified `.vscode/launch.json` to core workflows and prompt-based inputs.
The results were at the following path: Wrote output\eda\raw_lengths\flight_lengths.csv
Wrote output\eda\raw_lengths\length_summary.json

## 2026-02-04 03:29:54
- Added identity-based flight segmentation (split on icao24/callsign).
- Added EDA flight count script with aircraft type lookup via traffic.
- Added EDA combo runner for raw lengths + flight counts.
- Added launch configs for EDA combo, ADS-B monthly EDA, and prompt-based experiment grid runs.
- Updated preprocess grid to skip existing outputs and continue on errors.

## 2026-02-04 04:45:12
- Preprocess grid run: all preprocessed files created successfully except variants 4 and 9.

## 2026-02-05 00:10:53
- Completed full experiment grid (EXP001–EXP050); all preprocessed files available.

## 2026-02-05 00:20:00
- Added optional RDP (Ramer–Douglas–Peucker) simplification in preprocessing (disabled by default).

## 2026-02-05 04:21:00
- Fixed experiment grid override so per-experiment preprocessed files are respected.
- Added flow-aware cluster reporting (`flow_label`) to avoid cross-flow cluster ID confusion.
- Added sample input vector logging and run duration metadata to experiment logs.
- Added KML export (with optional runway placemarks and noise in gray).
- Added aircraft-specific height profiles + spectral class mapping (with fallbacks + missing-profile warnings).
- Added aggregation script to compare cluster subtracks vs ground truth in energy domain.

## 2026-02-06 04:21:00
- Updated Doc29 main experiment runner (`noise_simulation/run_doc29_experiment.py`) to avoid type-folder collisions by using `ICAO__NPD` folder keys instead of NPD-only keys.
- Extended experiment summaries (`summary_mse.csv` + `mse.json`) with `npd_id`, `type_folder`, `npd_table`, `height_profile`, and `spectral_class` for reproducibility.
- Updated Doc29 ground-truth runner (`noise_simulation/run_doc29_ground_truth.py`) with the same `ICAO__NPD` folder strategy to prevent output overwrites when multiple ICAO types share one NPD ID.
- Updated ground-truth runner to generate track groups keyed by ICAO type (while still selecting NPD tables via mapped NPD ID), so profile/spectral mapping remains type-correct.
- Added robust preprocessed input path fallback in ground-truth runner (`data/preprocessed/...` to `output/preprocessed/...`) to prevent FileNotFound errors in standardized runs.

## 2026-02-06 22:42:02
- Added `noise_simulation/compare_experiment_totals.py` for fair post-Doc29 comparison of clustered prediction vs per-category ground truth using energy-domain aggregation.
- New outputs per experiment:
  - `aggregate_totals/category_summary.csv` (category-level MAE/MSE/RMSE + log-energy average cumulative level)
  - `aggregate_totals/category_aligned_receivers.csv` (receiver-level aligned prediction/ground truth)
  - `aggregate_totals/overall_summary.json` (overall MAE/MSE/RMSE + average cumulative level delta)
- Supports both subtrack weighting modes:
  - `weighted`: uses `Nr.day` values already encoded in `Flight_subtracks.csv`
  - `unweighted`: scales by `n_flights / tracks_per_cluster`

## 2026-02-06 23:28:10
- Archived redundant scripts to reduce active surface area:
  - `legacy/scripts/run_clustering.py`
  - `legacy/scripts/generate_backbone_tracks.py`
  - `legacy/scripts/preprocess_dtw_frechet.py`
  - `legacy/scripts/plot_exp_latlon.py`
  - `legacy/scripts/eda_utm.py`
- Archived redundant noise scripts:
  - `legacy/noise_simulation/aggregate_cluster_cumulative.py`
  - `legacy/noise_simulation/interpret_results.py`
- Archived redundant Slurm jobs:
  - `legacy/jobs/run_eda.job`
  - `legacy/jobs/run_experiment_optics.job`
  - `legacy/jobs/run_save_preprocessed.job`
  - `legacy/jobs/run_backbone_clustering_full.job`
- Updated launch config to replace legacy Doc29 aggregate runner with:
  - `Doc29 Compare Totals (Fair)` using `noise_simulation/compare_experiment_totals.py`
- Updated docs (`README.md`, `CODEBASE_OVERVIEW.md`) to reflect active scripts and archived paths.

## 2026-03-10
- Added `scripts/eda_noise_clustering_summary.py` to generate one mixed thesis-evaluation view for `EXP001..EXP062` linking clustering outputs and Doc29 noise results.
- Standardized new summary outputs under `output/eda/`:
  - `exp001_062_noise_clustering_experiment_summary.csv`
  - `exp001_062_noise_clustering_flow_summary.csv`
  - `exp001_062_noise_clustering_block_summary.csv`
  - `exp001_062_noise_clustering_method_summary.csv`
  - `exp001_062_noise_clustering_correlations.csv`
- Added thesis note `thesis/docs/noise_clustering_results_exp001_062.md` to explain grouped vs experiment-wise interpretation and the use of all-flights ground truth as the primary overall ranking metric.

2026-02-07 18:26:42
- Added Doc29 Slurm jobs:
  - `jobs/run_doc29_ground_truth.job`
  - `jobs/run_doc29_experiment.job`
  - `jobs/run_doc29_compare_totals.job`

2026-02-09 14:19:46
[
L_{\mathrm{eq},w} = 10 \log_{10}!\left(\frac{t_0}{T_0}\sum_{i=1}^{N} g_i \cdot 10^{L_{E,i}/10}\right) + C
]

Where (L_{E,i}) is the single‑event exposure level, (g_i) is the time‑of‑day weight, (T_0) is the reference period, and (t_0) is the time basis used for normalization

## 2026-02-18
- Updated `jobs/run_experiment_grid_061_070_frechet.job` to be self-contained (no external parameters required at submit time).
- Fixed Slurm logging behavior for the 061-070 Frechet job by removing runtime log-folder creation that caused permission errors in spool paths.
- Standardized the Frechet-grid job flow to run from the cloned project root under `~/Clustering/psychic-broccoli`.
- Updated `scripts/plot_exp_latlon_flows.py` with `--exclude-noise` to generate cluster maps without `cluster_id = -1` flights.
- Added no-noise output naming in flow plots: `clusters_<flow>_latlon_no_noise.png`.

## 2026-03-21
- Added operation-aware `euclidean_weighted` distance support in:
  - `clustering/distances.py`
  - `experiments/runner.py`
- `euclidean_weighted` computes dense precomputed weighted-Euclidean distances on flattened fixed-length trajectories using point-index group weights, with operation-specific group selection from the flow label (`Landung` -> `Landing`, `Start` -> `Departure`).
- Added weighted-Euclidean DBSCAN pilot grid:
  - `experiments/experiment_grid_127_131_weighted_euclidean_dbscan.yaml`
- Pilot design:
  - `preprocessed_1`
  - current sample pilot uses `1000 flights/flow`
  - initially DBSCAN, then HDBSCAN `leaf`, then OPTICS, then rewritten back to weighted-Euclidean DBSCAN as the current variant
  - current sweep:
    - `eps = 9500, 10500, 11500, 12500, 13500`
    - `min_samples = 8`
  - operation-aware weighted Euclidean with:
    - landing early emphasis
    - departure late emphasis
- Promoted the reverted weighted-Euclidean DBSCAN configuration into the main experiment grid as the final full-data sweep:
  - `EXP046`-`EXP050`
  - `distance_metric: euclidean_weighted`
  - `eps = 8000, 9000, 10050, 11050, 12050`
  - `min_samples = 8`
  - operation-specific weights based on inverse `(p50 * spread)` step-group profiles

## 2026-02-19
- Added `scripts/eda_unique_matched_trajectories.py` to compute repetition-aware unique trajectory counts directly from `matched_trajectories/matched_trajs_*.csv`.
- Script outputs summary and metadata breakdowns for unique flights after MP repetition deduplication:
  - operation (`A/D`), runway, flow (`A/D + Runway`), and aircraft type percentages.
- Expanded matched-trajectory EDA outputs with complete CSV breakdowns:
  - `unique_flights_overview.csv`
  - `unique_flights_by_operation_runway.csv`
  - `unique_flights_by_operation_aircraft_type.csv`
  - `unique_flights_by_runway_aircraft_type.csv`
  - `unique_flights_output_index.csv`

## 2026-02-20
- Integrated library-based LCSS distance using `lcsspy` in:
  - `distance_metrics.py` (1D wrapper + 2D dual-channel trajectory mapping)
  - `clustering/distances.py` (dense precomputed LCSS matrix path + cache compatibility)
- Updated experiment runner for LCSS workflows:
  - New `distance_metric: lcss` handling.
  - Deterministic sample-for-fit support (`clustering.sample_for_fit`) with sample-only mode.
  - Added flow metadata fields to metrics/logs:
    - `n_flights_total_flow`
    - `n_flights_used_for_fit`
    - `fit_sampling_mode`
    - LCSS params used.
  - Added LCSS+KMeans path via deterministic classical MDS embedding.
- Updated `scripts/run_experiment_grid.py` to pass `sample_for_fit` overrides from experiment grid entries.
- Repurposed `EXP058`-`EXP070` in `experiments/experiment_grid_051_070_dtw_frechet.yaml`:
  - `EXP058`-`EXP068`: LCSS block (HDBSCAN/OPTICS/KMeans-MDS).
  - `EXP069`-`EXP070`: RDP dataset checks (`Euclidean`, `DTW`) using `preprocessed_12`.
- Updated `config/preprocess_grid.yaml`:
  - `preprocessed_12` now configured as RDP variant (`rdp_enabled=true`, `rdp_epsilon_m=50`, `rdp_min_points=10`).
- Updated dependencies:
  - Added `lcsspy` to `requirements.txt`.
- Updated registry:
  - Refreshed `thesis/docs/experiments_registry.md` with new EXP058-EXP070 definitions.
