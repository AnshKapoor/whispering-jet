# Noise Simulation Results: EXP011-EXP020

## Scope
This note summarizes Doc.29 noise-simulation outcomes for experiments **EXP011-EXP020**.

- Clustering outputs source: `output/experiments/EXP011...EXP020`
- Noise simulation source: `noise_simulation/results/EXP011...EXP020`
- Aggregated summary used: `output/eda/exp011_020_noise_overall_summary.csv`

## Experiment Block (for RQ2)
This block corresponds to **RQ2 (KMeans/HDBSCAN baselines and parameter effects)**.

- `EXP011-EXP015`: KMeans-family (Euclidean)
- `EXP016-EXP020`: HDBSCAN (Euclidean)

## Overall Results (9 receivers, 80 categories each)

| Experiment | Method | Preprocessed | MAE | MSE | RMSE | Noise frac |
|---|---|---|---:|---:|---:|---:|
| EXP011 | kmeans (k=4) | preprocessed_1 | 1.168 | 2.347 | 1.532 | 0.000 |
| EXP012 | kmeans (k=6) | preprocessed_1 | 0.928 | 1.561 | 1.250 | 0.000 |
| EXP013 | kmeans (k=8) | preprocessed_1 | **0.858** | **1.461** | **1.209** | 0.000 |
| EXP014 | minibatch_kmeans (k=6) | preprocessed_1 | 1.205 | 2.081 | 1.443 | 0.000 |
| EXP015 | kmeans (k=6) | preprocessed_2 | 2.204 | 8.550 | 2.924 | 0.000 |
| EXP016 | hdbscan (20,10) | preprocessed_1 | 1.208 | 2.450 | 1.565 | 0.202 |
| EXP017 | hdbscan (30,10) | preprocessed_1 | 1.242 | 2.554 | 1.598 | 0.136 |
| EXP018 | hdbscan (20,15) | preprocessed_1 | 1.220 | 2.471 | 1.572 | 0.200 |
| EXP019 | hdbscan (20,10) | preprocessed_3 | 2.295 | 9.201 | 3.033 | 0.201 |
| EXP020 | hdbscan (20,10) | preprocessed_5 | 1.018 | 1.795 | 1.340 | 0.197 |

## Brief Discussion

### 1) RQ2: Baselines and parameter effects
- On `preprocessed_1`, KMeans improves from `k=4 -> 6 -> 8`, with **EXP013 best** in this block.
- `minibatch_kmeans` (EXP014) is weaker than full KMeans at similar `k`.
- Changing preprocessing can dominate algorithm effect: EXP015 (`preprocessed_2`) degrades strongly versus EXP012 despite same KMeans family setup.
- For HDBSCAN, hyperparameter changes (`min_cluster_size`, `min_samples`) produce modest shifts on `preprocessed_1` (EXP016-018), while preprocessing variant changes can be much larger (EXP019 vs EXP020).

### 2) Flow-level error concentration
- The worst flow is almost always **Start_27R** (EXP011-014, 016-020), often dominating total MSE.
- This indicates aggregate error is driven by a few difficult departure-flow conditions rather than uniform performance loss across all flows.

### 3) Noise impact (cluster_id = -1 share)
- `EXP011-EXP015` have zero noise labels; `EXP016-EXP020` have notable noise share (~13.6% to ~20.2%).
- Correlation of noise fraction with overall error across EXP011-020 is weak:
  - Pearson(noise_frac, MSE) = `+0.118` (p=0.745)
  - Spearman(noise_frac, MSE) = `+0.459` (p=0.182)
- Interpretation: in this block, higher noise fraction **does not consistently imply** worse simulation error.
  - Example: EXP020 has ~19.7% noise but low MSE (1.795), while EXP019 has similar noise (~20.1%) but high MSE (9.201).

## RQ-Oriented Conclusion for EXP011-EXP020
For **RQ2**, the main takeaway is that both algorithm choice and parameterization matter, but **preprocessing variant choice can have even larger downstream impact** on noise-simulation quality. Noise-label rate alone is not a reliable predictor of simulation error in this block.

## Supporting Files
- `output/eda/exp011_020_noise_overall_summary.csv`
- `output/eda/exp011_020_noise_flow_summary.csv`
- `output/eda/exp011_020_noise_correlation.csv`
- `output/eda/exp011_020_noise_worst_flows.csv`
