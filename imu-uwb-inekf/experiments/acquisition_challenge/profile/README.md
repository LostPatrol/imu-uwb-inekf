<!-- Reproducible sensor-only acquisition and explicit wrong-prior experiment semantics. -->
# Profile acquisition challenge

`solver.py` reuses the original two-fit robust CV translation and exact circular
lifted yaw. Each model then receives one raw-range refinement, capped at 120
evaluations. `profile1` is the original seven-state model. `body_bias_prior` is
the current ten-state physical model with independent 0.10 m/s² component priors.
Its output is checked numerically against the unchanged published
`published_window_comparison/profile_main.py` entry point.

`run.py` consumes the common `imu.csv`, `radio.csv`, `anchors.csv`, `meta.json`
interface. IMU units are SI. Common radio calibration has already been applied.
R0 and IMU bias estimates come from the common sensor-only prefix. The body-to-tag
lever is included in predicted ranges; the state position remains the body origin.
The script neither changes sensor vectors nor reads position truth.

Native acquisition runs at 2, 2.5, …, 15 seconds. It uses `[0,t]` through 8 seconds
and the latest 8 seconds thereafter, always rebuilding its candidate from sensors.
There is no previous position, yaw or covariance seed. Before 2 seconds it produces
no estimate. `arrival_s` describes the measurement deadline, `state_s` the state
epoch, and `publish_s` includes measured computation and serial per-method queues.
Timing includes held-IMU integration, window construction, CV/lift and refinement;
file I/O and common prefix calibration are excluded. Sensor integration is cached
offline but each window uses only samples at or before its measurement deadline.

The wrong-prior experiment is a separate pair of `[0,2]` and `[0,10]` snapshots.
`priors.csv` contains absolute position centers and absolute ZYX heading at t=0.
The prior preserves the common R0 tilt. Position residuals use the physical t=0
position predicted by the window model; attitude residuals use the principal SO(3)
logarithm of the relative full attitude. These Gaussian penalties are applied after
robust radio residuals. Neither model receives a prior-derived initial candidate.
Only heading is optimized; the common R0 tilt is fixed. Because prior and predicted
attitude share that tilt, the isotropic SO(3) penalty norm equals the absolute
wrapped heading difference. This representation does not introduce tilt freedom.
The wide 20 m/120° and tight 0.10 m/5° assumptions are supplied by the common input.
The 0.10 m/s² internal body-bias prior is distinct from these external state priors.

Three semantics are preserved in separate records:

- `native.csv`: original sensor-only acquisition, independent of erroneous priors.
- `stress.csv`: original sensor-only snapshots (`case_id=native`) and explicit
  common soft-prior MAP extensions for each wrong-prior case.
- `native_reset_references.csv`: unconditional native resets which ignore the
  external prior and reference the corresponding existing sensor-only snapshot.
  These references are not separately executed or independent experimental trials.

For example, from the project root in PowerShell:

```powershell
./.venv/Scripts/python.exe imu-uwb-inekf/experiments/acquisition_challenge/profile/run.py --streams <common-episodes> --output agent/codex/profile_challenge/results --workers 5
./.venv/Scripts/python.exe imu-uwb-inekf/experiments/acquisition_challenge/profile/checks.py --output agent/codex/profile_challenge/checks.json
```

`checks.py` verifies the physical-start prior Jacobian, full-turn circularity,
original and current published solver agreement, analytic body-bias derivatives,
and asynchronous tag/IMU/bias integration against a separate closed form.
Raw radio packet hashes and common-file hashes document pairing. All finite solver
limits and numerical exceptions remain in the output. Data-only and assumed-prior
yaw information are separate local curvature diagnostics, not calibrated certainty.

`trajectory.py` is the separate continuous MAP protocol. It grows `[0,t]`
independently through 15 seconds at the integer arrivals 2, 3, …, 15, preserving
the t=0 prior origin. It runs only the current `body_bias_prior` main method and
includes matching `case_id=native` growing-window trajectories. Each prior case
has its own simulated serial computation queue:
`compute_start_s=max(arrival_s, previous_publish_s)` and
`publish_s=compute_start_s+wall_s`. `state_s=arrival_s` remains the epoch of the
estimated state, so a delayed output is not relabeled as a current-state estimate.
The initial interval ends at the first actual `publish_s`, not at 2 seconds.
Shared sensor candidates are physically computed once per window and confidence
batch; full candidate cost is charged to each case's nominal latency. The merged
file is `agent/codex/profile_challenge/results/trajectory.csv`, with the separate
`trajectory_protocol.json` manifest. `trajectory_reset_references.csv` explicitly
maps every prior case/node to its existing sensor-only growing snapshot. It must
be scored with its matching growing
native rows, not with native.csv's rolling-eight-second rows after t=8.
