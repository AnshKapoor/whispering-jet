# Preprocessed And Ground-Truth Audit

Audit scope:
- `output/preprocessed/preprocessed_1.csv` to `preprocessed_10.csv`
- `noise_simulation/results_ground_truth/preprocessed_*_final/ground_truth_cumulative.csv`

Main findings:
- All 10 preprocessed files have the same 14-column schema and the same `36,578` flights.
- The main structural difference is resampling density: files carry exactly 40, 50, 60, 70, or 80 points per flight.
- Flight metadata used by the ground-truth pipeline (`A/D`, `Runway`, `icao24`) is identical across all 10 files relative to `preprocessed_1.csv`.
- No malformed row-length mismatches were found in the first 5,000 rows of any preprocessed file.
- Ground-truth receiver variation is small at most points; the largest receiver range is `4.055` dB at `(x=555267.09, y=5811988.499, z=64.325)`.

Preprocessed file summary:

| file                | rows    | unique_flights | avg_points_per_flight | min_points_per_flight | max_points_per_flight | expected_cols_first_5000 | bad_rows_first_5000 |
| ------------------- | ------- | -------------- | --------------------- | --------------------- | --------------------- | ------------------------ | ------------------- |
| preprocessed_1.csv  | 1463120 | 36578          | 40.0                  | 40                    | 40                    | 14                       | 0                   |
| preprocessed_2.csv  | 2194680 | 36578          | 60.0                  | 60                    | 60                    | 14                       | 0                   |
| preprocessed_3.csv  | 2926240 | 36578          | 80.0                  | 80                    | 80                    | 14                       | 0                   |
| preprocessed_4.csv  | 1828900 | 36578          | 50.0                  | 50                    | 50                    | 14                       | 0                   |
| preprocessed_5.csv  | 1828900 | 36578          | 50.0                  | 50                    | 50                    | 14                       | 0                   |
| preprocessed_6.csv  | 1828900 | 36578          | 50.0                  | 50                    | 50                    | 14                       | 0                   |
| preprocessed_7.csv  | 1828900 | 36578          | 50.0                  | 50                    | 50                    | 14                       | 0                   |
| preprocessed_8.csv  | 2560460 | 36578          | 70.0                  | 70                    | 70                    | 14                       | 0                   |
| preprocessed_9.csv  | 2560460 | 36578          | 70.0                  | 70                    | 70                    | 14                       | 0                   |
| preprocessed_10.csv | 1828900 | 36578          | 50.0                  | 50                    | 50                    | 14                       | 0                   |

Flight-level metadata diffs versus `preprocessed_1.csv`:

| file                | flights_only_in_preprocessed_1 | flights_only_in_other | A/D_diff_flights | Runway_diff_flights | icao24_diff_flights |
| ------------------- | ------------------------------ | --------------------- | ---------------- | ------------------- | ------------------- |
| preprocessed_2.csv  | 0                              | 0                     | 0                | 0                   | 0                   |
| preprocessed_3.csv  | 0                              | 0                     | 0                | 0                   | 0                   |
| preprocessed_4.csv  | 0                              | 0                     | 0                | 0                   | 0                   |
| preprocessed_5.csv  | 0                              | 0                     | 0                | 0                   | 0                   |
| preprocessed_6.csv  | 0                              | 0                     | 0                | 0                   | 0                   |
| preprocessed_7.csv  | 0                              | 0                     | 0                | 0                   | 0                   |
| preprocessed_8.csv  | 0                              | 0                     | 0                | 0                   | 0                   |
| preprocessed_9.csv  | 0                              | 0                     | 0                | 0                   | 0                   |
| preprocessed_10.csv | 0                              | 0                     | 0                | 0                   | 0                   |

Endpoint deltas versus `preprocessed_1.csv` (Euclidean distance in metres):

| file                | first_mean_m       | first_p95_m       | first_max_m        | last_mean_m        | last_p95_m        | last_max_m         |
| ------------------- | ------------------ | ----------------- | ------------------ | ------------------ | ----------------- | ------------------ |
| preprocessed_2.csv  | 0.0                | 0.0               | 0.0                | 0.0                | 0.0               | 0.0                |
| preprocessed_3.csv  | 0.0                | 0.0               | 0.0                | 0.0                | 0.0               | 0.0                |
| preprocessed_4.csv  | 10.742794317139701 | 28.80711101320117 | 576.7665207907162  | 24.78753025495423  | 77.43638024984486 | 411.45163360735455 |
| preprocessed_5.csv  | 24.32671575879101  | 64.6150646712749  | 1523.4354952731817 | 48.1438881297443   | 140.9048055150978 | 1059.4614092791858 |
| preprocessed_6.csv  | 267.15387422938477 | 388.8912616069139 | 2007.974750263902  | 253.46104673308292 | 514.6355934857588 | 1405.954317285648  |
| preprocessed_7.csv  | 259.2387126974011  | 390.6449609251156 | 3120.706695310264  | 248.49294102966854 | 548.6877248262668 | 2062.1659573547986 |
| preprocessed_8.csv  | 0.0                | 0.0               | 0.0                | 0.0                | 0.0               | 0.0                |
| preprocessed_9.csv  | 10.742794317139701 | 28.80711101320117 | 576.7665207907162  | 24.78753025495423  | 77.43638024984486 | 411.45163360735455 |
| preprocessed_10.csv | 24.32671575879101  | 64.6150646712749  | 1523.4354952731817 | 536.2650264031391  | 960.8715051184639 | 1715.0431795771663 |

Ground-truth receiver variation across all 10 preprocessed variants:

| x         | y           | z                 | min_cumulative_res | max_cumulative_res | mean_cumulative_res | std_cumulative_res   | range_cumulative_res |
| --------- | ----------- | ----------------- | ------------------ | ------------------ | ------------------- | -------------------- | -------------------- |
| 537135.69 | 5813301.049 | 45.64636708512502 | 48.27711288119923  | 48.44217986221303  | 48.39289838625776   | 0.04507179623309498  | 0.16506698101380124  |
| 539538.85 | 5812485.541 | 50.80489064176144 | 50.576926030592674 | 50.59461052708559  | 50.58479477649173   | 0.007035599925075389 | 0.01768449649291881  |
| 542060.92 | 5813309.888 | 46.62666894079927 | 55.32476263747933  | 55.54692760464579  | 55.4196971529258    | 0.06346641271393717  | 0.2221649671664565   |
| 542489.95 | 5811706.809 | 48.30735841919886 | 52.39126633883032  | 52.4108207646928   | 52.39455849407485   | 0.005900995614915393 | 0.019554425862480684 |
| 550989.13 | 5813116.422 | 48.48394465532417 | 51.091036962918864 | 51.28879073945846  | 51.177792962713774  | 0.06960675122592519  | 0.19775377653959936  |
| 551663.9  | 5811763.862 | 49.44723665024979 | 52.18448368811358  | 52.404995477768594 | 52.24427224005052   | 0.062041105793370944 | 0.2205117896550135   |
| 553686.61 | 5813237.888 | 48.38476056255978 | 47.505717764954014 | 47.75753072353142  | 47.64174800306896   | 0.06666879606486073  | 0.2518129585774034   |
| 555267.09 | 5811988.499 | 64.32536221148862 | 48.24352519797507  | 52.29853366636817  | 48.712589009062924  | 1.2641241888420753   | 4.055008468393105    |
| 556553.09 | 5813424.585 | 81.0447772120765  | 44.652544539864536 | 44.78911687806741  | 44.73744427449347   | 0.039743442426651614 | 0.13657233820287473  |

Interpretation:
- The ground-truth pipeline reads `flight_id`, `A/D`, `Runway`, `icao24`, `step`, `x_utm`, and `y_utm` from the preprocessed CSVs.
- Because the same flights and flight metadata are preserved across all 10 files, most ground-truth differences come only from the resampled track geometry.
- Doc29 then re-interpolates tracks again at a fixed spacing, which dampens many small preprocessing differences before they reach the receivers.
