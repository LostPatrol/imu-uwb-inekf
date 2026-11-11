The experiment uses the twelve genuine Delft TDoA CSV recordings, independently
of Toronto UTIL. It never synthesizes differences from TWR. The first six files
in chronological order calibrate hardware; the final six all remain evaluation
recordings, including `hourglass_BAD`. The exact list and all fixed times are in
`protocol.json`. These six recordings cover four named motion classes and one
surveyed anchor geometry, not six independent geometries.

The official dataset README specifies `timeTick` in milliseconds, acceleration
in g, gyro in degrees/s, and surveyed Cyberzoo anchor coordinates in metres.
The publisher's `SUPfeiffer/uwb-simulator`, `UWBsim/airframe/drone.py`, method
`_get_tdoa`, reads `tdoaij` and negates it when constructing its reversed `(j,i)`
measurement. Its EKF predicts distance B minus distance A. Consequently this
adapter directly supplies `(i,j,tdoaij)` to Oursland: `y = rho_j - rho_i`.
Only a changed channel value identifies a new transaction; held values are not
fresh measurements. The timestamp cannot recover transactions hidden between
CSV rows or consecutive exactly equal measurements. The publisher's parser has
the same limitation. The official IMU adapter uses the axes without permutation;
the stationary prefix has positive body Z gravity. Logged Optitrack is already
in the Crazyflie control/Cyberzoo coordinates, rather than raw Optitrack axes.

The raw data contains only `otX/otY/otZ`; there is no quaternion or heading truth.
Only absolute XY radial RMSE and height RMSE are reported. There is no test
position alignment, test heading initialization, fitted drift or truth tuning.
The source does not publish a Delft antenna lever arm. We configure zero lever
and describe this uncalibrated physical limitation; the Toronto surveyed lever
is not transferred to a different airframe experiment.

Early-record calibration freezes a common radio/Optitrack clock shift and a
separate directed-pair median offset. With known surveyed anchors, the training
residual is

`e_ij(t,d) = y_ij(t) - (||q_GT(t+d)-a_j|| - ||q_GT(t+d)-a_i||)`.

The clock grid is `d=-.30,-.29,...,.30` s. Its objective is the median over the
six recordings of the median absolute residual after centering each pair by
its within-record median. After selecting `d`, pooled pair medians become
fixed measurement offsets. Pair MAD scales are saved for diagnostics, clipped
to `.08–.50 m`; the original filter and common candidates use the original
main's fixed `.13 m^2` observation variance. The frozen shift is `-.23 s`:
median training objective `.14544 m`, versus `.16641 m` at zero. Channel
`tdoa16` has a large training offset, approximately `-2.229 m`; it is retained.
This is external training calibration, not evidence that each offset is an
identifiable single-anchor hardware parameter. Lever and residual clock
uncertainty remain possible sources of systematic error.

Sensor origin is the first finite IMU sample with acceleration norm above
`.5 g`, without reading truth. Initialization waits `.6 s`; it fits a radio
constant-velocity antenna path from the surveyed anchor centroid and estimates
gravity tilt, accelerometer norm offset, and gyro mean from the same prefix.
Yaw is nominal zero, never motion-capture initialized. The original author's
main covariance, process settings, ground clamp and rejection rule remain
unchanged. `run.py` imports the untouched backup class and actually calls its
`predict/correct`; source SHA is saved. No research filter replaces those calls.

Two explicitly separate timestamp adaptations are retained. `first_per_stamp`
is primary and offers only the lexicographically first changed pair at each
CSV timestamp, following the original main's first-per-timestamp radio budget.
This restriction applies before initialization or any candidate receives data.
`all_packets_zero_predict` is an additional adaptation: all new packets are
offered, with native `predict(dt=0)` before extra same-timestamp corrections.
The original class leaves `X_hat/P_hat` stale after correction; the explicit
zero-time native call refreshes them without changing its functions. This
secondary wrapper is not the literal original-main event schedule. Each
adaptation has its own matching initialization and packet budget, so results
are compared within adaptation, never pooled as extra recordings.

The frozen recovery time is `T=25 s`. Raw radio in `[25-L,25)` is withheld for
`L=2,5,10,20 s`; a continuous-radio `L=0` reference is retained. Replay runs to
45 s. Metrics use a fixed 100 Hz grid on `[25,27)`, `[27,30)`, and `[31,36)`;
the horizontal metric is `sqrt(mean(dx^2+dy^2))`. Estimator states use causal
zero-order hold (`searchsorted(right)-1`); only truth is interpolated. This
repairs the first scoring version's future-state interpolation at display
discontinuities. Its tables remain as `metrics_superseded_linear_interpolation.csv`.
All policies and native intervals were rescored together; no filter, candidate,
clock calibration or gate changed. At and after 27 s the output function
queries K directly, including between native samples. Every test trace covers all
three periods. The four outages per recording are repeated counterfactuals,
not independent trials. All sixty native trajectories completed without a
nonfinite/1000 m failure. Native baseline XY medians across the six recordings:

| Primary first-per-stamp | 0–2 s | 2–5 s | 6–11 s |
|---|---:|---:|---:|
| Continuous | .191 m | .213 m | .218 m |
| 2 s outage | .205 m | .213 m | .218 m |
| 5 s outage | .461 m | .212 m | .217 m |
| 10 s outage | 17.02 m | 34.61 m | 36.11 m |
| 20 s outage | 91.88 m | 93.33 m | 70.95 m |

The shared translation candidates receive the same offered calibrated raw
packets, with outage packets physically removed, plus the native trajectory.
They change display XY/velocity only during the first two seconds after
resumption and return exactly to the unmodified native track afterward. They
do not change filter state, heading, covariance, later trajectory, or radio
updates. This is a translation output overlay, not a permanently recovered
InEKF state. `guarded_C` uses the shared past-block radio prediction evidence;
`always_C` retains every valid CV fit without its selector. The independently
registered `bound_xy` uses the same raw budget and a zero Delft lever radius.

In the primary queue's 24 recording/outage pairs, first-two-second guarded-C
improves 12, ties 2, and worsens 10; its worst regression is `.04331 m` and
median paired change `-.00095 m`. Always-C improves 11 and worsens 13, with
worst regression `.29102 m`. Bound-XY improves 5, ties 17, and worsens 2;
worst regression is `.00078 m`, median paired change zero. For 10/20 s outages,
guarded-C reduces the six-record median `17.02 -> 9.82 m` and
`91.88 -> 69.53 m`; the errors remain large. Short-outage marginal medians
slightly worsen. Both later intervals and all Z/R outputs remain exactly K.

Bound-XY is active on mean `3.44%` of the primary queue's first-two-second
100 Hz samples (`6.21%` secondary). Measured XYZ radio residuals satisfy its
fixed 3-sigma envelope in `97.90%` of primary post-resumption transactions
(`98.31%` secondary). Among active native output samples, the true horizontal
position lies in all applied nonlinear bounds in `104/108 = 96.30%`
(`187/200 = 93.50%` secondary). Thus conditional nonexpansion is not an
unconditional empirical guarantee. Physical packet percentages repeat the
same post-resumption packets across outage counterfactuals; they are not
independent observations for confidence estimates.

Return-to-K jumps are a serious limitation. Guarded-C's maximum added offset
jump is `168.26 m` primary and `362.82 m` secondary at 27 s. Bound-XY adds no
return offset in this primary matrix, but up to `50.85 m` in the secondary
matrix. A temporary display improvement must not be called lasting filter
recovery or a smooth trajectory improvement. Per-case raw physical checks,
activation rates and sampled boundary jumps are in `paired/bound_containment.csv`
and `paired/return_jumps.csv`.

`verification.json` checks the original class against the independent shared
mirror on the first calibration recording's actual `.6–5 s` transactions,
including the zero-lever and zero-time-refresh cases. State differences are
below `1.3e-15`, covariance below `5.0e-16`, and all gate decisions agree. The
full baseline nonetheless uses direct original-class calls.

`causality.json` poisons every actual L=10 blackout packet and separately all
radio plus K samples at/after 26 or 27 s: native hidden-packet replay and both
past candidate outputs remain exactly unchanged. A separate synthetic scoring
sentinel verifies that a future 100 m state cannot interpolate backward.

Run from `E:/0syncthing/Code/InEKF` in PowerShell:

```powershell
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
./.venv/Scripts/python.exe imu-uwb-inekf/experiments/strong_recovery/delft_tdoa/run.py --queues first_per_stamp all_packets_zero_predict
./.venv/Scripts/python.exe imu-uwb-inekf/experiments/strong_recovery/delft_tdoa/verify.py
./.venv/Scripts/python.exe imu-uwb-inekf/experiments/strong_recovery/delft_tdoa/paired.py
./.venv/Scripts/python.exe imu-uwb-inekf/experiments/strong_recovery/delft_tdoa/verify_causality.py
./.venv/Scripts/python.exe imu-uwb-inekf/experiments/strong_recovery/delft_tdoa/diagnose.py
./.venv/Scripts/python.exe imu-uwb-inekf/experiments/strong_recovery/delft_tdoa/analyze.py
```

`baseline/metrics.csv` contains every native case and scoring interval;
`baseline/records.json` stores packet counts, initialization, runtime and
failures. Every `.npz` retains the full native pose/velocity trajectory.
`calibration.json` stores the training-only lag curve, pair offsets, counts
and hashes. `paired/metrics.csv` and `paired/paired_xy.csv` retain matched
candidate outcomes; no BAD recording or adverse result is removed.
