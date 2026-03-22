# Noise and Clustering Results: EXP001-EXP062

## Scope
This note links clustering quality and Doc29 noise-simulation performance for experiments `EXP001..EXP062`.

Primary evaluation axis:
- `noise_mse_all_flights_gt`: thesis-level metric; compares experiment prediction against the shared all-flights ground truth and therefore penalizes both fit error and coverage loss.
- `noise_mse_clustered_gt`: conditional fit metric; compares clustered prediction against clustered-category ground truth only.

Generated from:
- `output/eda/exp001_062_noise_clustering_experiment_summary.csv`
- `output/eda/exp001_062_noise_clustering_block_summary.csv`
- `output/eda/exp001_062_noise_clustering_method_summary.csv`
- `output/eda/exp001_062_noise_clustering_correlations.csv`

## Headline Findings
- Best overall experiment by all-flights noise MSE: **EXP010** (optics, euclidean, preprocessed_10; `MSE_all=0.831`, `noise_frac=0.078`, `silhouette_valid=0.779`).
- Worst overall experiment by all-flights noise MSE: **EXP061** (optics, lcss, preprocessed_2; `MSE_all=125.232`, `noise_frac=0.688`).
- Best clustered-only Doc29 fit: **EXP024** with `MSE_clustered=0.069`.
- Best block by mean all-flights noise MSE: **EXP016-020 HDBSCAN baselines** (`mean=1.073`).
- Weakest block by mean all-flights noise MSE: **EXP058-062 LCSS extension** (`mean=80.281`).
- Strongest global monotonic relationship with `noise_mse_all_flights_gt`: `n_clusters_total` with Spearman `rho=0.457` (p=`0.000`).

## Block Summary
| Block | Label | n | mean MSE all-flights | mean MSE clustered | mean noise frac | mean silhouette_valid |
|---|---|---:|---:|---:|---:|---:|
| EXP001-010 | Preprocessing variants | 10 | 1.303 | 5.225 | 0.120 | 0.786 |
| EXP011-015 | KMeans baselines | 5 | 1.358 | 3.200 | 0.000 | 0.456 |
| EXP016-020 | HDBSCAN baselines | 5 | 1.073 | 3.694 | 0.187 | 0.344 |
| EXP021-024 | Non-Euclidean pilots | 4 | 26.575 | 1.054 | 0.540 | NA |
| EXP025-031 | OPTICS sensitivity | 7 | 4.080 | 2.966 | 0.204 | 0.798 |
| EXP032-036 | DBSCAN sensitivity | 5 | NA | NA | 1.000 | NA |
| EXP037-041 | GMM selection | 5 | 1.771 | 1.200 | 0.000 | 0.276 |
| EXP042-046 | Agglomerative and Birch | 5 | 1.296 | 3.107 | 0.000 | 0.470 |
| EXP047-050 | 3D altitude features | 4 | 1.514 | 9.695 | 0.108 | 0.491 |
| EXP051-057 | DTW extension | 7 | 35.440 | 8.243 | 0.306 | NA |
| EXP058-062 | LCSS extension | 5 | 80.281 | 5.187 | 0.430 | NA |

## Method and Distance Summary
| Method | Distance | n | mean MSE all-flights | mean noise frac | mean clustered frac |
|---|---|---:|---:|---:|---:|
| minibatch_kmeans | euclidean | 1 | 0.938 | 0.000 | 1.000 |
| hdbscan | euclidean | 6 | 1.054 | 0.190 | 0.810 |
| agglomerative | euclidean | 4 | 1.289 | 0.000 | 1.000 |
| birch | euclidean | 1 | 1.326 | 0.000 | 1.000 |
| kmeans | euclidean | 6 | 1.431 | 0.000 | 1.000 |

## Interpretation
- Use the experiment-level CSV when you need exact case comparisons in the thesis text.
- Use the block summary when discussing research-question trends or method families.
- Prefer `noise_mse_all_flights_gt` for the final overall ranking, because it exposes cases where apparent cluster quality comes from labeling large parts of the dataset as noise or from evaluating only a reduced subset.

## Missing Noise Outputs
- Missing or incomplete Doc29 aggregate outputs were detected for: EXP032, EXP033, EXP034, EXP035, EXP036.
