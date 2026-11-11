# Physical shape experiment findings

The exact joint initial-tilt/body-bias model improves position tails under tested calibration mismatch, but does not improve moderate-error heading over Profile K2. Its benefit depends substantially on assumed nuisance priors. No independent real-data generalization or InEKF tracking improvement has been established.

## Matched paper panel

`results_paper200/` contains 5,000 fits: 200 paper-RNG seeds, five error levels, five models. Paper K1/K2 use one lifted yaw and two translation starts, and reproduce Table III to its reported precision. All 5,000 fits converged and all saved numeric outputs are finite. Entries below are position / physical endpoint-heading p95, in metres / degrees.

| Level | Profile K1 | Profile K2 | Bias + prior | Bias/tilt + prior | Free bias/tilt |
|---|---:|---:|---:|---:|---:|
| Ideal | 0.546 / 3.578 | 0.732 / 4.747 | 0.559 / 4.611 | 0.565 / 4.679 | 1.020 / 12.159 |
| Mild | 1.031 / 16.902 | 0.731 / 5.525 | 0.563 / 5.316 | 0.553 / 5.339 | 1.009 / 12.411 |
| Moderate | 4.535 / 54.854 | 0.761 / 12.393 | 0.809 / 14.241 | 0.624 / 12.487 | 1.003 / 14.219 |
| Bias only | 1.147 / 20.608 | 0.738 / 4.756 | 0.589 / 4.546 | 0.618 / 4.897 | 1.015 / 12.078 |
| Tilt only | 4.364 / 44.146 | 0.732 / 4.669 | 0.640 / 5.938 | 0.556 / 4.518 | 1.015 / 11.088 |

Compared with K2, joint bias/tilt with prior reduces moderate position p95 by 18.0%, while heading p95 is slightly worse. Under ideal input, K1 remains best. Under isolated bias, the bias-only model is preferable to the more flexible joint model. Removing priors substantially worsens both errors. A universal replacement of K1/K2 is therefore unsupported.

The prior is assumed at 0.10 m/s2 per body-bias component and 2 degrees per tilt component; it is not empirically validated. Static gravity alignment correlates tilt and acceleration-bias uncertainty, whereas the experiment assumes independent Gaussian nuisance priors. The projected information is local yaw-coordinate information, not a physical-heading covariance or validated coverage statement.

## Raw same-date development panel

`results/raw_metrics.csv` contains 330 fits on 66 overlapping windows from six original later Delft CSV recordings, with separate earlier-recording range calibration. Every time window has the paper's endpoint and duration, all eight anchors, and one inertial frame. These are existing development recordings, not independent acquisitions. No historical cache or result was read.

| Model | Position median (m) | Position p95 (m) | Position RMSE (m) | Nonconverged fits |
|---|---:|---:|---:|---:|
| Profile K1 | 0.3967 | 3.3154 | 1.7940 | 1 |
| Profile K2 | 0.2599 | 0.5650 | 0.3203 | 0 |
| Bias + prior | 0.2470 | 0.4797 | 0.2787 | 0 |
| Bias/tilt + prior | 0.2389 | 0.4712 | 0.2782 | 0 |
| Free bias/tilt | 0.2499 | 0.5289 | 0.3007 | 4 |

All outcomes, including evaluation-cap failures, remain in the summaries. The joint prior model beats K2 on position in 49/66 paired windows. Its raw RMSE differs little from bias-only, so these logs do not establish that the additional tilt states are necessary. Log-level p95 improves relative to K2 in all six recordings, but this is descriptive, with overlapping windows and prior development exposure.

## Verification and boundary

- Exact cancellation of known body bias plus initial tilt, with correct gyro increments: maximum shape discrepancy 1.59e-13 m.
- Analytic raw-range Jacobian against central differences: maximum absolute difference 6.13e-8.
- Constant attitude/constant force with free body bias: data-only yaw information 8.77e-27, exposing the expected ambiguity.
- One fixed gyro-mismatch diagnostic retains 1.613 m shape RMS even after oracle body-bias and initial-tilt correction; the model cannot correct time-varying gyro attitude error.
- Three predetermined paper seeds were rerun with current source: 75 fits, maximum difference across all saved numeric quantities except runtime 4.55e-13.
- Direct comparison against existing polynomial solver: K1 exact match; K2 objective difference 9.66e-13 and maximum parameter difference 1.46e-8 on a preset control seed.

`analysis/comparison.png` and `.svg` show errors and prior/data information. `analysis/verification.json` and `replay_audit.json` retain the checks. Reproduction commands and physical formulas are in `README.md`.
