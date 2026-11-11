# Synthetic evidence checks and mathematical audit

These challenges use no real recordings or labels. Fixed settings: 6 seconds, 30 Hz radio, sigma=0.10 m, RNG seed 81126; raw TWR and directed TDoA are both covered. No acceptance threshold is fitted here. Run from the project root using `.venv/Scripts/python.exe`; optional `--module yaw_evidence` runs the current public `decide(**inputs)` alongside the independent diagnostic. Artifacts are saved under `agent/codex/split_checks`. `plot_checks.py` renders all profiles.

The input path is an ideal independent synthetic radio trajectory, deliberately giving evidence the best possible linearization point. It is not an empirical claim that a real radio-only estimator achieves that path accuracy. The systematic-alias case deliberately supplies a wrong path consistent with a coherent radio error. No synthetic physical yaw is included in the public API arguments.

## Profile model

At a fixed radio path pbar, let g_i be the raw radio gradient and h the raw TWR/TDoA model. The local observation is d_i=y_i-h(pbar_i)+g_i^T pbar_i. For each full-circle yaw delta, C=Rz(delta)Cref, s_i=g_i^T C force_i and the nuisance rows are

`A_i = [g_i^T, tau_i*g_i^T, .5*tau_i^2*g_i^T, -g_i^T*C*B_i]`.

The diagnostic profile is `||(I-AA^dagger)(d-s)/sigma||^2`. The bias nuisance columns and their projector must generally be recomputed at every yaw. A fixed projector based on only Cref can incorrectly label body-bias effects as heading evidence. With heterogeneous or correlated radio noise, the complete system must first be whitened.

World-constant acceleration absorbs only a polynomial displacement. Initial tilt error also rotates the time-varying force shape; it is not generally eliminated by a quadratic world nuisance. Similarly, replacing a rotating B_i by `.5*tau_i^2*I` changes the model and can manufacture nominal information. `wrong_bias_model` exposes this distinction without claiming a universal rejection test.

## Exact checks

1. With world polynomial translation through quadratic order, constant-velocity and constant-world-acceleration displacement give a flat yaw profile for every observation, including noisy ones. The script asserts full-circle span below numerical tolerance, rather than asserting that an arbitrary numerical minimizer equals the physical yaw.
2. Adding any world-rotated quadratic contamination to force changes only nuisance coefficients. The strong case asserts that the entire noisy full-circle profile remains equal, not merely its argmin.
3. Excited sinusoidal horizontal displacement survives polynomial elimination; its ideal profile is strongly nonflat. Near 179 degrees the full circle must join continuously at -180/180; angular errors must wrap. Weak-amplitude motion has virtually flat ideal evidence and a noise-selected wrong peak.
4. Coplanar near-zero-height geometry can have poor z information while retaining horizontal yaw evidence. It is incorrect to reject all horizontal inference solely because a full XYZ condition number is bad. Conversely, fixing z while reporting XY covariance creates artificial precision.
5. For any alternative physical yaw trajectory pwrong, choosing radio systematic error `e_i=h(pwrong_i)-h(ptrue_i)` makes observations exactly indistinguishable from that wrong trajectory. No same-window detector can guarantee to reject all such errors. Strong profile curvature, time-block agreement and link agreement do not prove physical correctness against unrestricted coherent errors.
6. `bias_only_excitation` uses rotating Q but `force=-.5*tau^2*g+B*b`, with true bias b. Its actual position is constant velocity. Correct yaw-dependent bias elimination gives a flat ideal profile (span about 5e-21), whereas an incorrectly fixed bias projector creates nonzero ideal evidence (span 0.451 at the prescribed bias). Even the correct noisy profile can become nonflat (span 8.4 in this fixed realization), because different nuisance subspaces fit noise differently. A noisy profile peak is therefore not a proof of physical yaw identifiability.

## XY Schur and position-only handoff

For information matrix H partitioned into XY and z, marginal XY information is `Hxx-Hxz*Hzz^dagger*Hzx`, provided the marginalized problem is identifiable. `Hxx` alone assumes z known. The numerical check gives conditional x variance 0.25 versus actual marginal x variance 2.564. A pseudoinverse of an unidentifiable full problem does not turn null directions into known zero-variance states; such directions must be represented as unconstrained or given an explicit prior.

An ordinary position Kalman update can alter yaw through P_yaw,p. The included example gives yaw gain 0.727 even though H has no direct yaw component. To promise position-first handoff without heading overwrite, explicitly freeze yaw gain/mean and use a covariance update consistent with that restricted gain, such as Joseph/Schmidt. The example preserves yaw variance 1 and PSD. Merely omitting an attitude field from the externally supplied endpoint does not freeze the internal filter yaw.

## Independence and information reuse

A radio-only path fit using the same window is algorithmically independent of IMU, but not statistically independent of the radio residuals subsequently evaluated at that path. Estimating path on all rows and then deleting one block only from the final evidence objective is not genuine held-out validation. Refit the path without the held-out block/link, or describe the test as a sensitivity diagnostic. Time blocks also need to reflect correlation duration; link folds can remain correlated through common anchors, clock errors, NLOS environment or shared path estimation.

No method may assimilate the same capture measurements again as independent online updates. The scalar unit-prior/unit-noise example gives posterior variance 1/2 after one observation; reusing the identical observation as independent incorrectly yields 1/3. Likewise, combining radio position and yaw summaries from the same rows requires their joint uncertainty or an explicit conservative approximation. Dropping their covariance does not establish independence.

Local curvature measures nominal conditional identifiability. It is not calibrated coverage or a proof that the assumed force, B, Cref, timing and radio error model are correct. Acceptance rates and sd_deg from the optional public API are recorded without changing its settings to match these cases.
