# Moving-window full-attitude checks

This branch tests the parent-provided `rotation_window.py` API with no static prefix and no known initial yaw or tilt. It does not implement a second solver. The physical generator supplies raw specific force and angular rate, integrates local attitude from identity, and retains truth separately for scoring.

For initial-body-to-world C, locally integrated gyro attitude Q(t), endpoint-anchored specific-force displacement D(t), and a known lever arm ell,

\[
p_{tag}(t)=p_T+\tau v_T+\tfrac12g\tau^2+C H(t),\qquad H(t)=D(t)+Q(t)\ell.
\]

The default lever arm is zero. The physical generator uses 200 Hz held IMU integration, four-second windows, 240 asynchronous noisy TWR transactions, 0.10 m range standard deviation, eight anchors at 30 m stand-off, and height spans of 3 m or 0.1 m. Sensor biases and IMU white noise are zero in this first acquisition test. All body intervals rotate; there is no hidden static segment. True initial tilt takes 0, 60, 120, or 179 degrees about a random horizontal axis, with independent random yaw. For a fixed case seed, changing C changes the physically measured body IMU stream but retains the same world trajectory and same noisy zero-lever ranges. Different tilt cases are therefore paired coordinate/IMU conditions, not independent range acquisitions.

The four motions are the paper-family rich world acceleration, 3% of that excitation plus a constant world acceleration, constant world acceleration, and constant velocity. Body rotation alone does not guarantee full initial-attitude observability: with zero lever arm and constant world acceleration, D has one temporal/vector direction after affine translation is removed. The final two regimes deliberately expose this ambiguity despite having no static IMU intervals.

Given a radio-only path seed and positive weights W, remove the same affine trajectory span from both sides:

\[
A=W^{1/2}[1,\tau],\quad
X=(I-AA^\dagger)W^{1/2}H,\quad
Y=(I-AA^\dagger)W^{1/2}(p_{radio}-\tfrac12g\tau^2).
\]

For Y'X=U Sigma V', a proper Wahba seed is C=U diag(1,1,det(UV')) V'. With an exact known path, rank(X)>=2 identifies the proper rotation. Rank one leaves a continuous rotational ambiguity. A noisy, smoothed radio path is used only to seed the final fit to original ranges; it is not an added position observation and its truth is unknown to every evaluated method.

All evaluated policies use the same raw observations, common radio-only path, raw-range model, and 120-evaluation refinement cap per solve:

- `single_identity`: one identity attitude seed.
- `moving_gravity`: align the whole moving-window mean body specific force with world up, with its conventional minimum-rotation yaw choice; it assumes no actual static segment.
- `local_gravity`: first undo body rotation with the gyro-integrated Q, then align the mean Q f with world up. This stronger gravity baseline still assumes mean world acceleration is negligible; it uses neither a static segment nor truth.
- `radio_wahba`: proper Wahba attitude from the estimated radio-only path.
- `cube24`: all 24 proper cube rotations, retaining the lowest original-range objective. It spends 24 refinements and is a finite strong reference, not a global certificate.

`cases.py` additionally computes the range rotation-information spectrum after eliminating free endpoint position/velocity. This truth-point spectrum is an offline observability diagnostic and is never used to select a seed or accept a result. The geodesic SO(3) error is primary; a 10-degree threshold is a descriptive acquisition threshold, not posterior coverage. Exact-position Wahba appears only in verification; no evaluated final range solve receives an oracle path or attitude.

From project root in PowerShell:

```powershell
.\.venv\Scripts\python.exe imu-uwb-inekf/experiments/radio_inertial_models/shape_branch/cold_rotation_checks/run_checks.py --seeds 10
.\.venv\Scripts\python.exe imu-uwb-inekf/experiments/radio_inertial_models/shape_branch/cold_rotation_checks/supplement_local_gravity.py
.\.venv\Scripts\python.exe imu-uwb-inekf/experiments/radio_inertial_models/shape_branch/cold_rotation_checks/analyze_checks.py
```

`results/metrics.csv` retains every outcome, including capped solves. `summary.csv`, `aggregate.csv`, `paired_vs_cube24.csv`, `protocol.json`, and `cold_rotation.png/.svg` support review. The separate `pilot/` panel has only two seed groups and normal geometry. These are physical simulations; they establish neither independent real-data generalization nor complete InEKF tracking performance. Results with nonzero sensor biases, clock errors, or unmodeled lever arms are not claimed.
