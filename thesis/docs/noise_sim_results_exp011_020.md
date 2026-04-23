# Noise Simulation Results: EXP011-EXP020

Brief block note for the finalized 9-point `L_eq` outputs. Each experiment table below preserves the exact receiver-level values from `aggregate_totals/overall_aligned_9points.csv`.

## Group Summary

| Experiment | MAE vs GT | MAE vs Cluster | Best MP | Best |err| GT | Worst MP | Worst |err| GT | Higher than GT | Lower than GT |
| ---------- | --------- | -------------- | ------- | ------------- | -------- | -------------- | -------------- | ------------- |
| EXP011     | 3.372     | 3.374          | MP5     | 0.392         | MP3      | 10.580         | 1              | 8             |
| EXP012     | 1.440     | 1.442          | MP4     | 0.252         | MP2      | 5.532          | 4              | 5             |
| EXP013     | 1.898     | 1.899          | MP4     | 0.202         | MP2      | 6.616          | 4              | 5             |
| EXP014     | 2.290     | 2.291          | MP6     | 0.807         | MP2      | 10.378         | 1              | 8             |
| EXP015     | 2.019     | 2.019          | MP4     | 0.196         | MP3      | 5.823          | 5              | 4             |
| EXP016     | 1.248     | 1.092          | MP3     | 0.027         | MP6      | 4.927          | 2              | 7             |
| EXP017     | 3.690     | 2.770          | MP4     | 1.732         | MP7      | 5.116          | 0              | 9             |
| EXP018     | 2.463     | 1.757          | MP1     | 1.208         | MP3      | 3.330          | 0              | 9             |
| EXP019     | 3.075     | 2.886          | MP5     | 0.014         | MP7      | 7.918          | 0              | 9             |
| EXP020     | 2.642     | 2.475          | MP5     | 0.025         | MP7      | 7.949          | 2              | 7             |

## Short Interpretation

- Best overall against all-flights ground truth: `EXP016` with `MAE = 1.248` dB.
- Worst overall against all-flights ground truth: `EXP017` with `MAE = 3.690` dB.
- Most experiments underestimate the all-flights ground truth at most receiver points; only a few experiments show several overestimates.
- The most consistently difficult receivers in this block are `MP7`, `MP2`, and `MP3` by mean absolute error against all-flights ground truth.
- The most consistently stable receiver is `MP5` by mean absolute error against all-flights ground truth.

Receiver-level pattern across the block:

| measuring_point | mean_abs_err_ground_truth | max_abs_err_ground_truth | higher_than_gt | lower_than_gt |
| --------------- | ------------------------- | ------------------------ | -------------- | ------------- |
| MP7             | 3.663                     | 7.949                    | 0              | 10            |
| MP2             | 3.437                     | 10.378                   | 6              | 4             |
| MP3             | 2.847                     | 10.580                   | 4              | 6             |
| MP9             | 2.814                     | 5.346                    | 0              | 10            |
| MP6             | 2.580                     | 4.927                    | 0              | 10            |
| MP8             | 2.562                     | 4.636                    | 0              | 10            |
| MP4             | 1.517                     | 4.758                    | 2              | 8             |
| MP1             | 1.289                     | 2.024                    | 2              | 8             |
| MP5             | 1.013                     | 3.459                    | 5              | 5             |

## EXP011

- `MAE vs all-flights ground truth = 3.372` dB; `MAE vs clustered ground truth = 3.374` dB.
- Best agreement vs all-flights ground truth: `MP5` with `|error| = 0.392` dB.
- Worst agreement vs all-flights ground truth: `MP3` with `|error| = 10.580` dB.
- Against all-flights ground truth, prediction is higher at `1` point(s) and lower at `8` point(s).

| MP  | L_eq_pred | L_eq_ground_truth | delta_ground_truth | abs_err_ground_truth | L_eq_cluster | delta_cluster | abs_err_cluster |
| --- | --------- | ----------------- | ------------------ | -------------------- | ------------ | ------------- | --------------- |
| MP1 | 50.241    | 52.265            | -2.024             | 2.024                | 52.265       | -2.024        | 2.024           |
| MP2 | 49.982    | 51.289            | -1.307             | 1.307                | 51.289       | -1.307        | 1.307           |
| MP3 | 58.338    | 47.758            | 10.580             | 10.580               | 47.758       | 10.580        | 10.580          |
| MP4 | 45.896    | 48.292            | -2.396             | 2.396                | 48.292       | -2.396        | 2.396           |
| MP5 | 44.397    | 44.789            | -0.392             | 0.392                | 44.789       | -0.392        | 0.392           |
| MP6 | 50.147    | 52.396            | -2.249             | 2.249                | 52.396       | -2.249        | 2.249           |
| MP7 | 46.733    | 50.587            | -3.854             | 3.854                | 50.587       | -3.855        | 3.855           |
| MP8 | 45.213    | 48.442            | -3.230             | 3.230                | 48.445       | -3.232        | 3.232           |
| MP9 | 51.230    | 55.547            | -4.317             | 4.317                | 55.556       | -4.326        | 4.326           |

## EXP012

- `MAE vs all-flights ground truth = 1.440` dB; `MAE vs clustered ground truth = 1.442` dB.
- Best agreement vs all-flights ground truth: `MP4` with `|error| = 0.252` dB.
- Worst agreement vs all-flights ground truth: `MP2` with `|error| = 5.532` dB.
- Against all-flights ground truth, prediction is higher at `4` point(s) and lower at `5` point(s).

| MP  | L_eq_pred | L_eq_ground_truth | delta_ground_truth | abs_err_ground_truth | L_eq_cluster | delta_cluster | abs_err_cluster |
| --- | --------- | ----------------- | ------------------ | -------------------- | ------------ | ------------- | --------------- |
| MP1 | 51.798    | 52.265            | -0.467             | 0.467                | 52.265       | -0.467        | 0.467           |
| MP2 | 56.821    | 51.289            | 5.532              | 5.532                | 51.289       | 5.532         | 5.532           |
| MP3 | 48.190    | 47.758            | 0.433              | 0.433                | 47.758       | 0.433         | 0.433           |
| MP4 | 48.544    | 48.292            | 0.252              | 0.252                | 48.292       | 0.252         | 0.252           |
| MP5 | 45.318    | 44.789            | 0.529              | 0.529                | 44.789       | 0.529         | 0.529           |
| MP6 | 50.203    | 52.396            | -2.193             | 2.193                | 52.396       | -2.193        | 2.193           |
| MP7 | 48.933    | 50.587            | -1.654             | 1.654                | 50.587       | -1.654        | 1.654           |
| MP8 | 47.035    | 48.442            | -1.407             | 1.407                | 48.445       | -1.410        | 1.410           |
| MP9 | 55.049    | 55.547            | -0.498             | 0.498                | 55.556       | -0.507        | 0.507           |

## EXP013

- `MAE vs all-flights ground truth = 1.898` dB; `MAE vs clustered ground truth = 1.899` dB.
- Best agreement vs all-flights ground truth: `MP4` with `|error| = 0.202` dB.
- Worst agreement vs all-flights ground truth: `MP2` with `|error| = 6.616` dB.
- Against all-flights ground truth, prediction is higher at `4` point(s) and lower at `5` point(s).

| MP  | L_eq_pred | L_eq_ground_truth | delta_ground_truth | abs_err_ground_truth | L_eq_cluster | delta_cluster | abs_err_cluster |
| --- | --------- | ----------------- | ------------------ | -------------------- | ------------ | ------------- | --------------- |
| MP1 | 53.447    | 52.265            | 1.181              | 1.181                | 52.265       | 1.181         | 1.181           |
| MP2 | 57.905    | 51.289            | 6.616              | 6.616                | 51.289       | 6.616         | 6.616           |
| MP3 | 50.437    | 47.758            | 2.680              | 2.680                | 47.758       | 2.680         | 2.680           |
| MP4 | 48.090    | 48.292            | -0.202             | 0.202                | 48.292       | -0.202        | 0.202           |
| MP5 | 45.825    | 44.789            | 1.036              | 1.036                | 44.789       | 1.036         | 1.036           |
| MP6 | 51.053    | 52.396            | -1.344             | 1.344                | 52.396       | -1.344        | 1.344           |
| MP7 | 49.452    | 50.587            | -1.135             | 1.135                | 50.587       | -1.135        | 1.135           |
| MP8 | 46.822    | 48.442            | -1.620             | 1.620                | 48.445       | -1.623        | 1.623           |
| MP9 | 54.281    | 55.547            | -1.265             | 1.265                | 55.556       | -1.275        | 1.275           |

## EXP014

- `MAE vs all-flights ground truth = 2.290` dB; `MAE vs clustered ground truth = 2.291` dB.
- Best agreement vs all-flights ground truth: `MP6` with `|error| = 0.807` dB.
- Worst agreement vs all-flights ground truth: `MP2` with `|error| = 10.378` dB.
- Against all-flights ground truth, prediction is higher at `1` point(s) and lower at `8` point(s).

| MP  | L_eq_pred | L_eq_ground_truth | delta_ground_truth | abs_err_ground_truth | L_eq_cluster | delta_cluster | abs_err_cluster |
| --- | --------- | ----------------- | ------------------ | -------------------- | ------------ | ------------- | --------------- |
| MP1 | 50.436    | 52.265            | -1.829             | 1.829                | 52.265       | -1.829        | 1.829           |
| MP2 | 61.667    | 51.289            | 10.378             | 10.378               | 51.289       | 10.378        | 10.378          |
| MP3 | 46.566    | 47.758            | -1.191             | 1.191                | 47.758       | -1.191        | 1.191           |
| MP4 | 46.682    | 48.292            | -1.610             | 1.610                | 48.292       | -1.610        | 1.610           |
| MP5 | 43.725    | 44.789            | -1.064             | 1.064                | 44.789       | -1.064        | 1.064           |
| MP6 | 51.589    | 52.396            | -0.807             | 0.807                | 52.396       | -0.807        | 0.807           |
| MP7 | 49.263    | 50.587            | -1.324             | 1.324                | 50.587       | -1.324        | 1.324           |
| MP8 | 47.191    | 48.442            | -1.251             | 1.251                | 48.445       | -1.254        | 1.254           |
| MP9 | 54.393    | 55.547            | -1.154             | 1.154                | 55.556       | -1.163        | 1.163           |

## EXP015

- `MAE vs all-flights ground truth = 2.019` dB; `MAE vs clustered ground truth = 2.019` dB.
- Best agreement vs all-flights ground truth: `MP4` with `|error| = 0.196` dB.
- Worst agreement vs all-flights ground truth: `MP3` with `|error| = 5.823` dB.
- Against all-flights ground truth, prediction is higher at `5` point(s) and lower at `4` point(s).

| MP  | L_eq_pred | L_eq_ground_truth | delta_ground_truth | abs_err_ground_truth | L_eq_cluster | delta_cluster | abs_err_cluster |
| --- | --------- | ----------------- | ------------------ | -------------------- | ------------ | ------------- | --------------- |
| MP1 | 53.470    | 52.215            | 1.254              | 1.254                | 52.215       | 1.254         | 1.254           |
| MP2 | 54.015    | 51.150            | 2.865              | 2.865                | 51.150       | 2.865         | 2.865           |
| MP3 | 53.457    | 47.634            | 5.823              | 5.823                | 47.634       | 5.823         | 5.823           |
| MP4 | 48.468    | 48.272            | 0.196              | 0.196                | 48.272       | 0.196         | 0.196           |
| MP5 | 45.811    | 44.737            | 1.074              | 1.074                | 44.737       | 1.074         | 1.074           |
| MP6 | 50.915    | 52.392            | -1.477             | 1.477                | 52.392       | -1.477        | 1.477           |
| MP7 | 48.607    | 50.579            | -1.973             | 1.973                | 50.579       | -1.973        | 1.973           |
| MP8 | 46.670    | 48.401            | -1.731             | 1.731                | 48.401       | -1.731        | 1.731           |
| MP9 | 53.619    | 55.396            | -1.776             | 1.776                | 55.396       | -1.776        | 1.776           |

## EXP016

- `MAE vs all-flights ground truth = 1.248` dB; `MAE vs clustered ground truth = 1.092` dB.
- Best agreement vs all-flights ground truth: `MP3` with `|error| = 0.027` dB.
- Worst agreement vs all-flights ground truth: `MP6` with `|error| = 4.927` dB.
- Against all-flights ground truth, prediction is higher at `2` point(s) and lower at `7` point(s).

| MP  | L_eq_pred | L_eq_ground_truth | delta_ground_truth | abs_err_ground_truth | L_eq_cluster | delta_cluster | abs_err_cluster |
| --- | --------- | ----------------- | ------------------ | -------------------- | ------------ | ------------- | --------------- |
| MP1 | 51.221    | 52.265            | -1.044             | 1.044                | 51.239       | -0.017        | 0.017           |
| MP2 | 51.405    | 51.289            | 0.116              | 0.116                | 50.480       | 0.925         | 0.925           |
| MP3 | 47.731    | 47.758            | -0.027             | 0.027                | 46.874       | 0.857         | 0.857           |
| MP4 | 47.469    | 48.292            | -0.823             | 0.823                | 47.426       | 0.043         | 0.043           |
| MP5 | 44.849    | 44.789            | 0.060              | 0.060                | 43.934       | 0.916         | 0.916           |
| MP6 | 47.469    | 52.396            | -4.927             | 4.927                | 51.702       | -4.233        | 4.233           |
| MP7 | 48.173    | 50.587            | -2.414             | 2.414                | 50.067       | -1.894        | 1.894           |
| MP8 | 47.357    | 48.442            | -1.085             | 1.085                | 47.966       | -0.609        | 0.609           |
| MP9 | 54.815    | 55.547            | -0.732             | 0.732                | 55.151       | -0.336        | 0.336           |

## EXP017

- `MAE vs all-flights ground truth = 3.690` dB; `MAE vs clustered ground truth = 2.770` dB.
- Best agreement vs all-flights ground truth: `MP4` with `|error| = 1.732` dB.
- Worst agreement vs all-flights ground truth: `MP7` with `|error| = 5.116` dB.
- Against all-flights ground truth, prediction is higher at `0` point(s) and lower at `9` point(s).

| MP  | L_eq_pred | L_eq_ground_truth | delta_ground_truth | abs_err_ground_truth | L_eq_cluster | delta_cluster | abs_err_cluster |
| --- | --------- | ----------------- | ------------------ | -------------------- | ------------ | ------------- | --------------- |
| MP1 | 50.356    | 52.265            | -1.909             | 1.909                | 51.178       | -0.822        | 0.822           |
| MP2 | 46.930    | 51.289            | -4.359             | 4.359                | 50.443       | -3.512        | 3.512           |
| MP3 | 43.590    | 47.758            | -4.167             | 4.167                | 46.831       | -3.240        | 3.240           |
| MP4 | 46.560    | 48.292            | -1.732             | 1.732                | 47.367       | -0.807        | 0.807           |
| MP5 | 41.331    | 44.789            | -3.459             | 3.459                | 43.875       | -2.545        | 2.545           |
| MP6 | 49.365    | 52.396            | -3.032             | 3.032                | 51.629       | -2.264        | 2.264           |
| MP7 | 45.472    | 50.587            | -5.116             | 5.116                | 49.656       | -4.184        | 4.184           |
| MP8 | 43.870    | 48.442            | -4.572             | 4.572                | 47.501       | -3.631        | 3.631           |
| MP9 | 50.683    | 55.547            | -4.864             | 4.864                | 54.604       | -3.921        | 3.921           |

## EXP018

- `MAE vs all-flights ground truth = 2.463` dB; `MAE vs clustered ground truth = 1.757` dB.
- Best agreement vs all-flights ground truth: `MP1` with `|error| = 1.208` dB.
- Worst agreement vs all-flights ground truth: `MP3` with `|error| = 3.330` dB.
- Against all-flights ground truth, prediction is higher at `0` point(s) and lower at `9` point(s).

| MP  | L_eq_pred | L_eq_ground_truth | delta_ground_truth | abs_err_ground_truth | L_eq_cluster | delta_cluster | abs_err_cluster |
| --- | --------- | ----------------- | ------------------ | -------------------- | ------------ | ------------- | --------------- |
| MP1 | 51.057    | 52.265            | -1.208             | 1.208                | 51.257       | -0.199        | 0.199           |
| MP2 | 48.374    | 51.289            | -2.915             | 2.915                | 50.510       | -2.136        | 2.136           |
| MP3 | 44.427    | 47.758            | -3.330             | 3.330                | 46.909       | -2.482        | 2.482           |
| MP4 | 45.924    | 48.292            | -2.367             | 2.367                | 47.460       | -1.535        | 1.535           |
| MP5 | 42.315    | 44.789            | -2.474             | 2.474                | 43.974       | -1.659        | 1.659           |
| MP6 | 50.264    | 52.396            | -2.132             | 2.132                | 51.703       | -1.439        | 1.439           |
| MP7 | 47.291    | 50.587            | -3.296             | 3.296                | 50.074       | -2.783        | 2.783           |
| MP8 | 46.891    | 48.442            | -1.551             | 1.551                | 47.968       | -1.077        | 1.077           |
| MP9 | 52.654    | 55.547            | -2.893             | 2.893                | 55.152       | -2.498        | 2.498           |

## EXP019

- `MAE vs all-flights ground truth = 3.075` dB; `MAE vs clustered ground truth = 2.886` dB.
- Best agreement vs all-flights ground truth: `MP5` with `|error| = 0.014` dB.
- Worst agreement vs all-flights ground truth: `MP7` with `|error| = 7.918` dB.
- Against all-flights ground truth, prediction is higher at `0` point(s) and lower at `9` point(s).

| MP  | L_eq_pred | L_eq_ground_truth | delta_ground_truth | abs_err_ground_truth | L_eq_cluster | delta_cluster | abs_err_cluster |
| --- | --------- | ----------------- | ------------------ | -------------------- | ------------ | ------------- | --------------- |
| MP1 | 51.231    | 52.184            | -0.953             | 0.953                | 51.190       | 0.041         | 0.041           |
| MP2 | 51.045    | 51.258            | -0.214             | 0.214                | 50.564       | 0.480         | 0.480           |
| MP3 | 47.450    | 47.648            | -0.198             | 0.198                | 46.775       | 0.675         | 0.675           |
| MP4 | 47.541    | 52.299            | -4.758             | 4.758                | 51.992       | -4.451        | 4.451           |
| MP5 | 44.708    | 44.722            | -0.014             | 0.014                | 43.862       | 0.846         | 0.846           |
| MP6 | 48.606    | 52.394            | -3.788             | 3.788                | 51.725       | -3.120        | 3.120           |
| MP7 | 42.659    | 50.577            | -7.918             | 7.918                | 50.074       | -7.415        | 7.415           |
| MP8 | 43.846    | 48.384            | -4.539             | 4.539                | 47.908       | -4.062        | 4.062           |
| MP9 | 50.034    | 55.325            | -5.290             | 5.290                | 54.919       | -4.885        | 4.885           |

## EXP020

- `MAE vs all-flights ground truth = 2.642` dB; `MAE vs clustered ground truth = 2.475` dB.
- Best agreement vs all-flights ground truth: `MP5` with `|error| = 0.025` dB.
- Worst agreement vs all-flights ground truth: `MP7` with `|error| = 7.949` dB.
- Against all-flights ground truth, prediction is higher at `2` point(s) and lower at `7` point(s).

| MP  | L_eq_pred | L_eq_ground_truth | delta_ground_truth | abs_err_ground_truth | L_eq_cluster | delta_cluster | abs_err_cluster |
| --- | --------- | ----------------- | ------------------ | -------------------- | ------------ | ------------- | --------------- |
| MP1 | 51.221    | 52.241            | -1.020             | 1.020                | 51.228       | -0.007        | 0.007           |
| MP2 | 51.261    | 51.196            | 0.065              | 0.065                | 50.405       | 0.856         | 0.856           |
| MP3 | 47.611    | 47.652            | -0.041             | 0.041                | 46.824       | 0.787         | 0.787           |
| MP4 | 47.487    | 48.326            | -0.839             | 0.839                | 47.428       | 0.059         | 0.059           |
| MP5 | 44.777    | 44.751            | 0.025              | 0.025                | 43.898       | 0.879         | 0.879           |
| MP6 | 48.537    | 52.391            | -3.854             | 3.854                | 51.702       | -3.165        | 3.165           |
| MP7 | 42.633    | 50.583            | -7.949             | 7.949                | 50.067       | -7.434        | 7.434           |
| MP8 | 43.789    | 48.425            | -4.636             | 4.636                | 47.942       | -4.153        | 4.153           |
| MP9 | 50.080    | 55.427            | -5.346             | 5.346                | 55.018       | -4.938        | 4.938           |

