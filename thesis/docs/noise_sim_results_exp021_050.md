# Noise Simulation Results: EXP021-EXP050

## Scope
This note extends the noise-simulation interpretation beyond EXP011-020 and covers three result blocks:

- `EXP021-EXP030`
- `EXP031-EXP040` (with `EXP032-EXP036` intentionally excluded from benchmarking due all-noise DBSCAN behavior)
- `EXP041-EXP050`

Data basis:
- `output/eda/exp021_050_noise_overall_summary.csv`
- `output/eda/exp021_050_noise_range_summary.csv`
- `output/eda/exp021_050_noise_worst_flow_per_exp.csv`

---

## Results for EXP021-EXP030

### Setup context
- `EXP021-EXP024`: non-Euclidean pilot block (DTW/Fr\'echet; HDBSCAN/OPTICS)
- `EXP025-EXP030`: OPTICS Euclidean sensitivity block (same preprocessed input family)

### Benchmark summary
- Best run by MSE: **EXP024** (`mse=0.069`, `mae=0.159`) 
- Worst run by MSE: **EXP029** (`mse=4.362`, `mae=1.749`)
- Block mean: `mean_mse=2.127`, `mean_mae=1.038`

### Noise-impact comment
- Mean noise fraction is relatively high (`0.349`), with very high maxima (`0.896`).
- Despite that, this block contains the strongest overall fit in this document segment (EXP024), so high noise fraction does not automatically imply weak simulation agreement.
- Correlation in this block is strongly negative:
  - Pearson(noise\_frac, MSE) = `-0.852` (p=0.0018)
  - Spearman(noise\_frac, MSE) = `-0.818` (p=0.0038)
- Interpretation: in this block, higher noise-label share tends to coincide with lower MSE, likely because the retained clustered subset is more coherent for Doc.29 abstraction.

### Dominant error flow
- Worst-flow frequency in this block:
  - `Start_27R`: 7 experiments
  - `Landung_09L`: 2 experiments
  - `Start_27L`: 1 experiment

---

## Results for EXP031-EXP040

### Scope note
- For this block, **EXP032-EXP036 are intentionally excluded** from validity benchmarking (all-noise DBSCAN regime).
- Effective compared set: `EXP031`, `EXP037`, `EXP038`, `EXP039`, `EXP040`.

### Benchmark summary
- Best run by MSE: **EXP039** (`mse=1.078`, `mae=0.654`)
- Worst run by MSE: **EXP031** (`mse=3.709`, `mae=1.595`)
- Block mean (available runs): `mean_mse=1.744`, `mean_mae=0.949`

### Noise-impact comment
- Noise fraction is low overall (`mean=0.0198`, max `0.0988`).
- Pearson correlation with MSE is high positive in this small sample (`r=0.992`), but this should be treated carefully due only 5 points and one highly influential run.
- Operationally, this block is comparatively stable versus EXP041-050.

### Dominant error flow
- `Start_27R` is worst flow in **all** included runs (5/5).

---

## Results for EXP041-EXP050

### Setup context
- Mixed algorithm-family benchmarking (GMM/Agglomerative/Birch and later runs on `preprocessed_3`).

### Benchmark summary
- Best run by MSE: **EXP041** (`mse=0.992`, `mae=0.644`)
- Worst run by MSE: **EXP047** (`mse=10.386`, `mae=2.402`)
- Block mean: `mean_mse=5.531`, `mean_mae=1.652` (highest of the three blocks)

### Noise-impact comment
- Noise fraction is moderate (`mean=0.043`, max `0.232`), lower than EXP021-030 but higher than EXP031-040.
- Correlation with MSE is positive but not strongly significant in this block:
  - Pearson(noise\_frac, MSE) = `+0.598` (p=0.068)
  - Spearman(noise\_frac, MSE) = `+0.548` (p=0.101)
- Interpretation: noise fraction may contribute, but error variation is also driven by method/representation choices and difficult flow geometry.

### Dominant error flow
- `Start_27R` is worst flow in **all 10** runs.

---

## Cross-Block Interpretation
- **Best-performing block:** `EXP031-040` (on available valid runs) and strong outliers in `EXP021-030` (EXP024).
- **Most volatile block:** `EXP041-050` (largest spread and highest mean MSE).
- **Persistent system bottleneck:** `Start_27R` is the dominant high-error flow across all three blocks.

Practical implication: for Doc.29-oriented deployment, flow-specific refinement for `Start_27R` likely yields larger gains than global algorithm switching alone.
