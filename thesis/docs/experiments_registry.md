# Experiments Registry

This registry records standardized clustering experiments and their purpose.

Columns:
- `id`: experiment number (e.g., `EXP001`)
- `question`: research question / ablation axis
- `preprocessed_id`: link to preprocessed registry (e.g., `preprocessed_1`)
- `distance`: `euclidean` / `dtw` / `frechet` / `lcss`
- `algorithm`: `kmeans` / `optics` / `hdbscan` / etc.
- `params`: key parameter changes
- `notes`: optional remarks

| id | uid | question | preprocessed_id | distance | algorithm | params | notes |
|---|---|---|---|---|---|---|---|
| EXP001-EXP050 | see plan | Baseline + core ablations | preprocessed_1..7 | euclidean/dtw/frechet | mixed | see `exp01_50_plan.md` | Detailed matrix in plan |
| EXP051-EXP057 | see plan | Post-correction reruns / DTW carryover | preprocessed_1..2 | dtw | hdbscan | dense DTW setup | Retained from prior block |
| EXP058-EXP065 | see plan | LCSS (library-based) sensitivity across preprocess variants | preprocessed_1,2,3,11 | lcss | hdbscan/optics | `lcss_epsilon_m=300`, `lcss_delta_alpha=0.10`, sample-only cap `1200` | Uses `lcsspy`; no custom LCSS |
| EXP066-EXP068 | see plan | KMeans on LCSS via classical MDS embedding | preprocessed_1,3,11 | lcss | kmeans | same LCSS params + `mds_n_components=3`, sample-only cap `1200` | Embedding + vector-space clustering |
| EXP069-EXP070 | see plan | RDP variant checks | preprocessed_12 | euclidean/dtw | hdbscan | preprocessed_12 uses RDP simplify (`epsilon=50m`, `min_points=10`) | `EXP069`: Euclidean, `EXP070`: DTW |
