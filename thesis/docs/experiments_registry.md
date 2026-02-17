# Experiments Registry

This registry records standardized clustering experiments and their purpose.

Columns:
- id: experiment number (e.g., EXP001)
- question: research question / ablation axis
- preprocessed_id: link to preprocessed registry (e.g., preprocessed_1)
- distance: euclidean / dtw / frechet
- algorithm: kmeans / optics / hdbscan / etc.
- params: key parameter changes
- notes: optional remarks

| id | uid | question | preprocessed_id | distance | algorithm | params | notes |
|---|---|---|---|---|---|---|---|
| EXP001–EXP050 | see plan | See plan | preprocessed_1..7 | euclidean/dtw/frechet | mixed | see `exp01_50_plan.md` | Detailed matrix in plan |
| EXP051-EXP070 | see plan | Post-50 distance extension | corrected_1..3 plus 11..13 | dtw/frechet | hdbscan/optics | dense exact + sparse-edge variants | Includes variable-length no-resample checks |
