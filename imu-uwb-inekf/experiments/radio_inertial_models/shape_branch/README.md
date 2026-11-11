# Physical IMU shape branch

This isolated experiment compares the paper's matched Profile K1/K2 models with a rotating body-bias model and an exact initial-tilt shape correction. It changes neither the initializer nor the InEKF tracker used elsewhere.

For endpoint time T, tau=t-T and nominal attitude Q, define the held-sample equivalent of

\[
B_a(t)=\int_t^T(s-t)Q(s)\,ds.
\]

If E is a constant left correction of nominal attitude and b is a constant body accelerometer-bias correction, the corrected endpoint-anchored displacement is

\[
d_c(t)=E\left[d(t)-B_a(t)b-\tfrac12\tau^2g\right]+\tfrac12\tau^2g,
\qquad p(t)=p_T+v_T\tau+R_z(\psi)d_c(t).
\]

We parameterize E=Exp([epsilon_x,epsilon_y,0]x). The model is exact for a constant initial attitude error and constant body accelerometer bias when gyro increments are correct, using the same held-sample integration. It explicitly rotates the nonpolynomial force shape as well as modeling gravity leakage. A free quadratic world coefficient only models the latter exactly. Time-varying attitude error from gyro bias is outside this model.

The existing `recovery_bootstrap/bias_real.py` already contains rotating accelerometer-bias sensitivity. The new part here is its joint exact initial-tilt shape correction and paired controls, not the bias integral itself.

`physical_shape.py` contains integration, exact raw-range derivatives, nuisance priors, and paired simulation. K1/K2 use their respective single polynomial lifted yaw and two translation starts (lift-recovered translation and common CV translation). Physical models use the same K1 lift and its two translation starts with zero added nuisances. All use 120 evaluations per refinement, Huber threshold 2, and 1e-8 tolerances. They do not have equal parameter counts or equal nonlinear work. Runtime includes CV and the candidate but excludes IMU sensitivity construction; `nfev` excludes common CV solves.

The assumed prior standard deviations are 0.10 m/s2 per body-bias component and 2 degrees per tilt component, fixed across all perturbation levels. These are modeling assumptions, not a validated calibration distribution. The free bias/tilt ablation removes both priors. `information` eliminates all motion-model nuisances from the raw range IRLS Jacobian; `prior_information` adds the assumed priors. Both are local yaw-coordinate diagnostics, not calibrated confidence or physical-heading coverage. Objectives across different model orders or priors must not be ranked as identical optimization objectives.

Run from `E:\0syncthing\Code\InEKF` in PowerShell:

```powershell
.\.venv\Scripts\python.exe imu-uwb-inekf/experiments/radio_inertial_models/shape_branch/physical_shape.py --seeds 200 --paper-seeds --output imu-uwb-inekf/experiments/radio_inertial_models/shape_branch/results_paper200
.\.venv\Scripts\python.exe imu-uwb-inekf/experiments/radio_inertial_models/shape_branch/raw_windows.py --logs 6
.\.venv\Scripts\python.exe imu-uwb-inekf/experiments/radio_inertial_models/shape_branch/analyze.py
```

`--paper-seeds` reconstructs the exact master-seed RNG sequence from the existing physical stress source, without reading saved historical protocols. `results/` additionally contains a 40-seed diagnostic using consecutive seed values, which is a separate simulation sample and is not the paper's 200-seed panel. Both share the same physical generator. There is no IMU white noise. Five levels include the paper's ideal/mild/moderate combined errors and isolated bias-only/tilt-only controls. Every model shares the exact same truth, IMU, transaction times, anchors, and noisy ranges for a seed and level.

`raw_windows.py` reloads the original Delft CSVs and estimates range bias/scales on the earlier recording of each motion, then evaluates the later recording. It reconstructs all eleven paper time windows per log, all eight anchors, one inertial yaw frame: 66 overlapping physical windows from six recordings. Static alignment uses the initial 1.5 seconds and the existing common sensor/OptiTrack delay is 0.20 seconds. Truth is used for training calibration, valid-interval cropping, common time alignment, and offline scoring, never passed to a window solve. Source CSV hashes and calibration values are retained in `raw_protocol.json`.

These raw recordings are the same date and already exposed development recordings. Reloading raw data does not make them an independent test. No independently acquired Delft TWR recording was available in this branch's initial dataset panel. Synthetic samples, equivalent yaw frames, anchor subsets, and overlapping windows cannot be described as independent real-data generalization.

`analyze.py` writes a paired comparison against K2, descriptive pooled/raw per-log errors, a figure, and direct analytic verification. Important negative outcomes must remain visible: ideal-input heading can worsen after adding nuisance freedom; improved position need not improve heading; a prior can supply much of the retained information; constant attitude and constant force make yaw unidentifiable with unrestricted body bias; gyro drift remains unmodeled. No end-to-end InEKF tracking improvement is established by these window-only experiments.
