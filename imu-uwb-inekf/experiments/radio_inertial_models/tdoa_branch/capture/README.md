# Dynamic TDoA capture: matched negative result

This second isolated study tests a real TDoA/IMU heading acquisition hypothesis.
It did not overcome const3's heading error in three model attempts. Lower raw
TDoA objectives and successful multi-start convergence did not imply correct
physical heading, nor better subsequent tracking.

## Fixed protocol and information budget

The acquisition interval was fixed to recording times **10-16 seconds**, followed
by the identical white-variance InEKF on **16-45 seconds**. Endpoints use the first
100 Hz IMU tick at or after these nominal times. No truth-dependent window
selection or temporal alignment was used. The earliest possible fresh output is
after the six-second capture wait; there is no claimed output during acquisition.
This is a fixed-time recapture study, not an entire system startup benchmark.

All methods receive exactly the same IMU, surveyed anchors, real ordered TDoA
transactions, physical lever arm, initial two-second IMU gravity alignment,
common radio CV seed, Huber threshold 2, solver cap 120 evaluations per start,
and post-handoff invariant tracker. TDoA2 uses the documented eight directed ring
pairs. TDoA3 retains its native ordered observations. Capture radio packets are
used only in the batch fit and **never replayed into the subsequent filter**;
that filter starts strictly after the capture endpoint.

The projected candidate reuses capture radio to construct a starting point for
the same objective; it does not constitute a second likelihood term. The strong
`uniform8` control optimizes the same objective from eight yaw starts spaced by
45 degrees. Its best raw objective is selected without truth. `raw_local` uses
a single zero yaw seed. `zero_yaw` fixes yaw at zero but fits all remaining
enabled nuisance variables using the same window. Zero is just the inertial
frame convention; no true heading is supplied to this policy.

All policies use the same conservative physical handoff covariance, including
pi-radian yaw standard deviation, so this isolates changes of fitted nominal
state. It does not test covariance extraction or a capture supervisor. Subsequent
tracking uses native hyperbolas, the actual lever arm, 0.14 m^2 white variance
(0.22 m^2 for const4), Huber threshold 2, and NIS gate 25, with the previous
study's 100 Hz scheduler. No OU/pair-error states appear in this study.

Truth is loaded by the shared loader but is accessed only by endpoint and
post-handoff scoring. No true pose, bias, motion label, or heading participates
in fitting, calibration, candidate selection, or trajectory alignment.

## Inertial shape and candidate

Gyroscope integration supplies a rotation curve Q(t) in an arbitrary yaw frame.
Reanchored double integration supplies d(T)=d_dot(T)=0 and bias sensitivity
B(T)=B_dot(T)=0. The actual antenna model is

\[
q(t)=p_T+v_T\tau+R_z(\psi)[d(t)-B(t)b_a+Q(t)l],\qquad\tau=t-T,
\]

\[
z_{ab}(t)=\|q(t)-a_b\|-\|q(t)-a_a\|+e_{ab}(t).
\]

The rotating-body sensitivity B is the reanchored double integral of Q, not
a scalar quadratic approximation. The bias-prior model appends residual
\(b_a/(0.2\;m/s^2)\). All policies use the same prior.

The candidate first independently localizes antenna positions from native raw
TDoA in fixed 200 ms bins. It then removes affine time components from these
positions and the inertial antenna shape. For the resulting horizontal curves
P and D, the 2D Procrustes yaw is

\[
\psi_c=\operatorname{atan2}\left(\sum_i(D_{ix}P_{iy}-D_{iy}P_{ix}),
                               \sum_i(D_{ix}P_{ix}+D_{iy}P_{iy})\right).
\]

This candidate seeds one common nonlinear raw-TDoA refinement. Its independent
radio-bin fits add computation, recorded separately as `candidate_evaluations`;
do not compare only raw-refinement evaluations and claim a total speedup.

The third model additionally estimates a two-axis initial tilt correction
\(E=R_z(\psi)\operatorname{Exp}([\delta\theta_x,\delta\theta_y,0])\)
with 0.08 rad component priors. Because gravity must remain in the world frame,
the corrected antenna shape is

\[
q(t)=p_T+v_T\tau+E[d(t)-\tfrac12g\tau^2-B(t)b_a+Q(t)l]
                        +\tfrac12g\tau^2.
\]

The g correction is essential when E includes tilt; a simple rotation of the
whole gravity-containing inertial displacement would be physically incorrect.

## Three attempts and physical results

1. `first_attempt`: const3-trial2-tdoa2, zero gyro bias, fixed accelerometer bias
   and then the common accelerometer-bias prior. The fixed-bias raw objective
   decreases from 2614 to 1577 while endpoint yaw worsens from about -2.49 to
   +89.05 degrees. Local, projected, and uniform-eight fits agree on that wrong
   solution. Bias freedom improves endpoint position from about 1.03 to 0.13 m,
   but subsequent heading remains worse than the fixed-zero control.
2. `gyro_calibrated`: common causal calibration of gyro bias from the initial
   two-second IMU prefix, only when every prefix acceleration norm is within
   0.05 g of 1 g and every angular-rate norm is below 2 deg/s. Otherwise the
   common zero gyro bias is retained. Four recordings and both radio protocols
   were tested, with the accelerometer-bias prior. All solved-yaw policies
   remained worse than zero yaw in subsequent physical heading.
3. `full_attitude`: the same four recordings, calibration policy and window,
   adding the two tilt variables and their common prior. The conclusion persists.

The final attempt's values are shown below. Position RMSE and yaw RMSE refer
to the common **post-capture** 16-45 s interval in the unaligned surveyed frame.

| Recording | Zero-yaw endpoint / tracking yaw | Projected endpoint / tracking yaw | Zero / projected tracking position |
|---|---:|---:|---:|
| const3 trial2 TDoA2 | -2.30 / 28.14 deg | -19.30 / 56.18 deg | 0.2368 / 0.2377 m |
| const3 trial3 TDoA2 | -2.30 / 11.62 deg | -8.56 / 18.03 deg | 0.1940 / 0.1944 m |
| const3 trial3 TDoA3 | -6.34 / 30.45 deg | +21.48 / 86.16 deg | 0.1387 / 0.1461 m |
| const1 trial3 TDoA2 | -4.05 / 7.41 deg | +37.43 / 24.40 deg | 0.1036 / 0.1033 m |

Raw-local, projected-single and uniform-eight agree to numerical optimization
tolerance in objective, with minor physical differences near flat minima.
The uniform control rules out the tested single-start policy as the cause of
the physical failure; it does not prove a global minimum for every possible
continuous state. These windows happen to place the arbitrary zero yaw frame
near the true heading. This is not evidence that zero yaw would solve arbitrary
initial heading; it makes the failure of the fitted yaw particularly clear.

The horizontal radio-only affine-profiled curve energies are 1.21, 1.44, 7.47,
and 1.07 m^2 across the four logs: the selected windows contain nonzero radio
motion, and were not chosen to be static. Data-only profiled yaw information
after removing p/v/bias/tilt nuisance columns is roughly 21-61 rad^-2 at the
solved final models. These are local Gaussian quantities, not trustworthy
heading-confidence certificates under correlated radio errors, bin-localization
noise, IMU calibration mismatch, and limited motion excitation. In particular,
constant-acceleration curvature can be absorbed by bias/tilt nuisance freedom.

No window, prior, seed count, or heading gate was adjusted using the scored
physical errors. The third negative attempt concludes this bounded study.
Remaining candidates include independently learned pair-error calibration and
better excitation/covariance modeling, but neither was silently introduced here.

## Reproduce from the project root (PowerShell)

```powershell
$captureScript = 'imu-uwb-inekf/experiments/radio_inertial_models/tdoa_branch/capture/capture_study.py'
& .venv/Scripts/python.exe $captureScript --names const3-trial2-tdoa2 --output first_attempt
& .venv/Scripts/python.exe $captureScript --gyro-calibration --names const3-trial2-tdoa2 const3-trial3-tdoa2 const3-trial3-tdoa3 const1-trial3-tdoa2 --bias-models prior --output gyro_calibrated
& .venv/Scripts/python.exe $captureScript --gyro-calibration --tilt-model --names const3-trial2-tdoa2 const3-trial3-tdoa2 const3-trial3-tdoa3 const1-trial3-tdoa2 --bias-models prior --output full_attitude
& .venv/Scripts/python.exe $captureScript --gyro-calibration --tilt-model --names const3-trial2-tdoa2 const3-trial3-tdoa2 const3-trial3-tdoa3 const1-trial3-tdoa2 --bias-models prior --check-only --output verification
& .venv/Scripts/python.exe imu-uwb-inekf/experiments/radio_inertial_models/tdoa_branch/capture/plot_capture.py
```

The complete 12-parameter raw-TDoA/p/v/yaw/accelerometer-bias/tilt Jacobian was
checked against central differences on all four real windows; maximum element
error was 3.3e-7. `--check-only` also prints motion energies without replaying
the tracker. CSV files retain every policy, fit cost/evaluation count, endpoint
physical error, information quantity and tracking metric. NPZ files retain
capture states, biases and subsequent timestamped body trajectories. Timing
columns named `capture_runtime` measure raw refinement only; newer runs also
record common CV/candidate preprocessing separately. File loading, shape
integration, and scoring are excluded. No total-runtime advantage is claimed.
