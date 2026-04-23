# Clustering Module and Experiment Outputs

This folder contains the clustering-side implementation used by the thesis
workflow:

- [`distances.py`](distances.py) builds dense or sparse pairwise distances and
  dense feature matrices
- [`registry.py`](registry.py) resolves the configured clustering method and
  runs the corresponding estimator
- [`evaluation.py`](evaluation.py) computes internal clustering metrics such as
  silhouette, Davies--Bouldin, noise fraction, and cluster counts

The actual experiment execution is performed by
[`experiments/runner.py`](../experiments/runner.py), which calls these modules
and writes the output files under `output/experiments/<EXP>/`.

## What a User Gets After Clustering

For a standard experiment such as `EXP001`, the output folder typically looks
like this:

```text
output/experiments/EXP001/
|- config_resolved.yaml
|- experiment_log.txt
|- runtime_log.txt
|- metrics_by_flow.csv
|- metrics_global.csv
|- labels_ALL.csv
|- labels_<flow>.csv
|- cluster_counts_by_flow.csv
|- cluster_runway_counts.csv
|- cluster_aircraft_type_counts.csv
|- backbone_tracks.csv
|- graphs/
`- plots/
```

Typical examples of flow-specific label files are:

- `labels_Start_09L.csv`
- `labels_Start_27R.csv`
- `labels_Landung_09L.csv`
- `labels_Landung_27R.csv`

Each `labels_<flow>.csv` file contains one row per flight for that flow, not
one row per trajectory point.

## Meaning of the Label Files

The label files are the main output of the clustering stage.

### `labels_<flow>.csv`

This file contains the cluster assignment for one operational flow
(for example `Start_09L` or `Landung_27R`).

Important columns usually include:

- `flight_id`
- `cluster_id`
- `A/D`
- `Runway`
- `icao24`
- `callsign`
- `flow_label`
- possibly aircraft-type columns if available in the input

### `labels_ALL.csv`

This file is the concatenation of all flow-specific label files into one table.
It is useful for:

- global counting across all flows
- plotting summaries
- downstream noise-simulation preparation
- EDA scripts that compare experiments

## How Cluster Numbering Works

The important rule is:

- `cluster_id = -1` means the flight was treated as noise or unassigned

For non-noise flights:

- cluster IDs such as `0`, `1`, `2`, `3`, ... identify clusters within that
  flow
- the numbering is an internal identifier, not a physical semantic label
- cluster `0` in one flow is not comparable to cluster `0` in another flow
- cluster `0` in `EXP001` is not guaranteed to mean the same thing as cluster
  `0` in `EXP002`

In other words, the cluster numbers are useful as stable identifiers inside one
experiment and one flow, but they should not be interpreted as universal labels
across experiments.

## How the Labels Are Assigned

The runner builds one row per flight and then stores the cluster label returned
by the chosen clustering method. The output table is created in
[`experiments/runner.py`](../experiments/runner.py).

Relevant code points:

- label column assignment:
  [`experiments/runner.py`](../experiments/runner.py) sets
  `labeled["cluster_id"] = labels`
- per-flow label export:
  [`experiments/runner.py`](../experiments/runner.py) writes
  `labels_<flow>.csv`
- global label export:
  [`experiments/runner.py`](../experiments/runner.py) writes `labels_ALL.csv`

The active workflow treats `-1` as noise explicitly in both the runner and the
metric code.

## Other Important Output Files

### `metrics_by_flow.csv`

One row per flow with the main internal clustering metrics. Typical fields
include:

- number of clusters
- number of clustered flights
- number of noise flights
- noise fraction
- silhouette score
- Davies--Bouldin score

This is the first file to inspect when comparing clustering quality between
flows within one experiment.

### `metrics_global.csv`

Experiment-level aggregate metrics across all flows. This is useful for ranking
or comparing experiments at a high level.

### `cluster_counts_by_flow.csv`

Counts the number of flights in each `(flow, cluster_id)` pair. This is the
main summary file for understanding cluster sizes.

Important detail:

- rows with `cluster_id = -1` are the noise flights
- the column `is_noise_cluster` is included for convenience

### `cluster_runway_counts.csv`

Counts flights by:

- `flow_label`
- `cluster_id`
- `Runway`

This is mainly a diagnostic check and is especially useful if a user wants to
verify that clusters do not mix runway definitions unexpectedly.

### `cluster_aircraft_type_counts.csv`

Counts flights by:

- `flow_label`
- `cluster_id`
- `aircraft_type_match`

This is useful for checking whether a cluster is dominated by particular
aircraft categories.

### `backbone_tracks.csv`

This file is not the direct output of clustering alone, but it is the next
stage derived from the cluster labels. It contains the representative backbone
tracks for the retained clusters.

Noise flights (`cluster_id = -1`) are not used to create backbone tracks.

## Visual Outputs

### `graphs/`

This folder usually contains flow-level cluster visualizations, for example:

- `clusters_Start_09L_latlon.png`
- `clusters_Landung_27R_latlon.png`

These plots show the clustered flight geometries for one flow.

### `plots/`

This folder usually contains per-cluster backbone visualizations and related
derived plots.

## Practical Interpretation Notes

When reading clustering outputs, keep the following in mind:

- the clustering is performed flow by flow, not as one global pool unless a
  specific configuration changes that behavior
- `cluster_id = -1` is expected for density-based methods when flights do not
  belong to a retained cluster
- cluster numbering is primarily bookkeeping; cluster size, geometry, and
  metrics matter more than the literal numeric ID
- the labels are one row per flight, while the preprocessed input files are one
  row per resampled trajectory point

## Recommended Files to Open First

If a new user wants to inspect one finished experiment quickly, open these in
order:

1. `config_resolved.yaml`
2. `metrics_global.csv`
3. `metrics_by_flow.csv`
4. `cluster_counts_by_flow.csv`
5. `labels_ALL.csv`
6. one or two `labels_<flow>.csv` files
7. the corresponding figures in `graphs/`
