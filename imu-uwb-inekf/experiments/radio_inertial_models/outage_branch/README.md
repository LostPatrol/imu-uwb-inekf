# Matched 10-second UWB outage study

This isolated experiment tests whether retaining an inertial joint prior improves the paper's weakest Delft outage result. It rebuilds radio calibration from the earlier log of each of six motion pairs and evaluates the later recordings. It reads raw sensor data and current source code. Evaluation OptiTrack positions are stored separately and loaded only after estimation for scoring; the existing loader uses truth availability to select the valid interval. The common 0.20 s sensor delay and static 1.5 s IMU prefix are inherited assumptions. These are same-day development recordings, not an independent blind test.

## Reproduction

Run from `E:\0syncthing\Code\InEKF` with project Python:

```powershell
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
./.venv/Scripts/python.exe imu-uwb-inekf/experiments/radio_inertial_models/outage_branch/study.py --prepare --workers 6
./.venv/Scripts/python.exe imu-uwb-inekf/experiments/radio_inertial_models/outage_branch/analyze.py
```

The clock starts after the 1.5 s static prefix. All methods receive the same changed, positive asynchronous ranges, with transactions removed for `15 <= t < 25` in masked cases. Startup is the original Profile-1 two-second initialization into the common 15-state InEKF. The post-start radio gate is NIS 12.25 and Huber threshold 2 for every method. Reacquisition triggers only when an actual range arrives after more than one second of radio silence. The mask boundary is not a runtime reset signal.

## Compared methods

| Method | During silence | Upon resumed ranging |
|---|---|---|
| `profile_reset` | IMU prediction | Original 2 s Profile-1 acquisition, blind predicted output until handoff |
| `profile_reset_live` | Identical IMU prediction | Original Profile-1 window; independently continue online InEKF output during acquisition |
| `continuous` | Identical IMU prediction | Immediately continue the same 15-state InEKF and gate |
| `joint_map` | Identical IMU prediction | 2 s window with the predicted full 15D prior, blind output until handoff |
| `diagonal_map` | Identical IMU prediction | Same window with all prior cross-covariances removed |
| `joint_map_live` | Identical IMU prediction | Same joint window; independently continue online InEKF output during acquisition |
| `diagonal_map_live` | Identical IMU prediction | Same live-output policy with diagonal prior |
| `damped_gap` | A simple damped-velocity motion prior after 0.25 s without actual ranges | Immediately resume the same InEKF and gate |

For the window MAP, let the predicted state at the first resumed event be the prior. Its physical error is

$$\delta=[\delta\theta_b,\delta v_w,\delta p_w,\delta b_g,\delta b_a]^\top,\qquad P_0=LL^\top.$$

All 15 variables, including tilt and biases, participate in

$$\min_\delta \|L^{-1}\delta\|^2+\sum_i\rho_H\!\left(\frac{\|p(t_i;\delta)-a_i\|-(z_i-\beta_i)}{\sigma_i}\right).$$

Here `p(t_i;delta)` follows the same held-sample IMU step as the tracker. The initial rotation is right-injected by `Exp(delta_theta)`. The exact sampled physical transition propagates the analytic Jacobian. In the live-output cases, the independent MAP prior is saved **before** the new window ranges enter the online tracker. The online posterior is never fed back into that MAP, and window transactions are not replayed after handoff. Thus the final branch consumes each window range once.

This MAP uses deterministic IMU integration within its 2 s window and omits the temporal correlation induced by process noise. It adds the accumulated process covariance at the endpoint after the local curvature update. Its covariance is therefore an approximation, not a calibrated smoothing posterior. Recorded computation time is not added to replay timestamps; this is an offline causal experiment, not a demonstrated real-time implementation.

The simple motion comparator uses

$$\dot v=-v/\tau,\quad\dot p=v,\quad\tau=2\ \mathrm{s},$$

only after the observable 0.25 s radio-silence threshold. Its covariance follows that kinematic prior with uncertain acceleration. Attitude still propagates with gyro data. This is an indoor-hover assumption, not a new inertial observation or an airframe/drag model. In particular, there is no repeated use of accelerometer samples as both independent inputs and pseudo-observations.

## Interpretation and outputs

`results/metrics.csv` contains all 96 method/log/mask cases. `summary.csv` and `paired_metrics.csv` separate the 15-25 s no-range interval, the first two seconds after radio resumes, and the period following window handoff. Continuous-ranging cases quantify whether gap handling changes the ordinary trajectory. `validation.json` checks the paired prefixes, equal inertial trajectories inside the mask, sensor/truth separation, and finite outputs. Full estimated trajectories and scoring truth are retained in the result histories.

The original triangle replay reproduces the paper's 27.715 m error peak. At the end of the ten-second mask its error is already 17.584 m; blind propagation during the next two-second acquisition raises it to 27.715 m. Changing a future reacquisition window cannot remove the preceding no-range error.

Across six recordings, the original Profile-1 reset has median common RMSE 2.164 m, median post-resumption RMSE 2.197 m, and five terminal stable recoveries. Same-gate continuous tracking reduces these RMSEs to 1.947/0.699 m, but has only four terminal stable recoveries. A live-output joint MAP gives 1.785/0.573 m and five recoveries. The diagonal-prior ablation gives essentially the same result. There is consequently no evidence here that retaining the full joint prior is the source of the improvement. Most of the transient improvement comes from immediately resuming ordinary range corrections.

The decisive output-policy ablation keeps the original Profile-1 initializer and only supplies ordinary InEKF output during its waiting window. It achieves 1.785/0.576 m and the same five recoveries, closely matching the much more expensive joint MAP. Keeping live output and preserving the original recapture is therefore the simpler supported engineering change; the joint-prior experiment does not establish a substantial independent contribution.

The damped motion prior is included as a cautionary baseline: it bounds drift for several indoor flights but can worsen the no-range error. It must be judged by the paired segment errors, not only its aggregate or eventual recovery count. No result establishes a general solution to ten seconds of unaided inertial drift.

Its final median common/post-resumption RMSEs are 1.442/0.337 m, maximum error 6.435 m, and six eventual stable recoveries. Nevertheless, no-range RMSE increases from 2.631 to 3.472 m on hourglass and from 1.286 to 3.082 m on star. Hourglass post-resumption RMSE remains 3.363 m despite eventual recovery. Continuous tracking fails terminal recovery on hourglass and square; Profile-1 and both MAP families fail it on square. The success criterion requires the final error to stay below 0.5 m for at least five seconds, so it does not summarize all earlier error.

Without the artificial mask, all InEKF/MAP families reproduce identical trajectories: median common RMSE 0.285241 m and five of six stable recoveries. The damped prior also reacts to occasional natural 0.25 s radio gaps: median common RMSE is 0.285185 m; the largest per-log RMSE increase is about 0.000011 m on star, while octagon improves by about 0.000113 m. These observed small costs do not validate the prior for other motion distributions.

`paired_errors.png` visualizes all six errors over replay time. `paired_segments.png` compares no-range and post-resumption errors. The analytical window Jacobian was independently checked against central differences on the triangle case: maximum absolute difference `3.16e-8`, relative Frobenius difference `1.81e-10` (temporary check evidence under `agent/codex/ri_outage`).
