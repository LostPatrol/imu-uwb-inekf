# Delft split position and attitude handoffs

This isolated experiment compares attitude decisions after a common radio-only translation handoff. It uses the six later Delft TWR recordings, with radio bias and scale calibrated only on the six earlier motion-matched recordings. Runtime estimators never access evaluation positions. Delft provides no trusted yaw truth for this experiment: the reported rotation offsets are controlled perturbations, and all performance comparisons concern subsequent position.

## Fixed protocol

The replay clock starts after a 1.5 s static IMU prefix, with the inherited common 0.20 s sensor delay. A sensor-only Profile-1 initializer hands off at 2 s using all eight anchors. Thereafter the geometry is either all eight anchors or the four surveyed anchors with the lowest heights. The latter has a 0.49 m height span. At 15 s the existing mean attitude receives a left world-vertical rotation of 0, +90, -90, or 180 degrees. Position, velocity, biases, and physical covariance are retained. These are injected changes to an estimate, not errors measured against true heading. Every yaw condition within a recording/geometry has an identical pre-15 s prefix.

| Interval | Shared processing |
|---|---|
| 15-25 s | Remove every radio transaction; propagate the same IMU model |
| 25-27 s | Acquire one radio-only CV window; no scalar InEKF reuse of those ranges |
| 27 s | All strategies receive the same endpoint position/velocity; retain attitude and bias means |
| 27-31 s | Continue scalar InEKF updates on odd per-anchor arrival counts; reserve even counts for heading evidence |
| 31 s | Fork only the attitude decision, with common translation, biases, and physical uncertainty |
| After 31 s | Resume all available radio transactions in every branch |

The odd/even counters advance only on received positive transactions; packets removed by the mask do not advance them. Heading factors start strictly after the 27 s position-window endpoint, so neither the early translation factors nor the live heading-window factors are reused in the attitude solve. The live branch is not held blind for the four-second heading wait.

The primary score is the radial horizontal RMSE on the same future interval in every recording,

$$e_{xy}=\sqrt{\operatorname{mean}_{31\leq t<36}\!\left[(\hat x-x)^2+(\hat y-y)^2\right]}.$$

Vertical RMSE is secondary. The shortest recording ends at 38.42 s, so a 31-41 s interval is unavailable. Outputs also preserve the no-range interval, common 25-27 s blind acquisition, 27-31 s live waiting interval, 25 s through the recording end, and the terminal five seconds. The short future score does not imply that the preceding outage has been solved.

## Matched attitude decisions

Let `Qend` be gyro propagation from the heading-window start and `Rlive` the actual terminal live InEKF attitude. The reference supplied to the shared gate is `Cref = Rlive @ Qend.T`. Thus its returned world-z correction is applied directly to the current terminal live attitude.

| Method | Attitude at 31 s |
|---|---|
| `position_only` | Retain `Rlive` |
| `full_attitude` | Replace it with the raw-distance full-SO(3) fit `C @ Qend` |
| `forced_yaw` | Apply the shared gate's candidate yaw regardless of acceptance |
| `evidence_yaw` | Apply exactly that candidate only if the gate accepts |
| `projected_full_yaw` | Project the full-SO(3) competitor onto a world-z rotation of `Rlive` |

The full-attitude competitor uses the existing `rotation_window.solve_attitude`, not an unrefined Procrustes seed. Two raw-distance refinements start from Procrustes and `Cref`; the smaller raw objective is retained. A constant body acceleration-bias error is a fitted nuisance with the existing 0.10 m/s² prior. Its estimated bias and translation do not overwrite the common live state. Optimizer non-success flags and degenerate geometry remain in the outputs.

The gate is the frozen shared `split_handoff/yaw_evidence.py`. Its `candidate`, `delta`, `accepted`, `sd_deg`, fold gains and rejection reasons are recorded without using evaluation errors. `accepted` in each method's metric row describes the same shared gate decision; it does not mean that the full-attitude or position-only method applies that decision.

At the position handoff, translation covariance comes from robust radio-only curvature plus 0.50 m position and 1.00 m/s velocity floors, with unsupported translation cross terms removed. At 31 s every strategy removes attitude cross terms and receives exactly the same **world-coordinate** attitude covariance: the live covariance plus 10-degree tilt and 45-degree yaw variance floors. This common covariance is mapped to each new body's physical error chart before invariant conversion. All other physical covariance blocks and all position/velocity/bias means are identical at the fork. This is a conservative comparison policy, not a calibrated joint posterior.

Disjoint radio groups prevent direct same-window radio reuse. They do not make the branches statistically independent: both share prior sensor history, the IMU, calibration, and state-dependent references. The gyro bias is frozen within the four-second shape integration; acceleration bias and tilt uncertainty enter the shared evidence model. The antenna lever arm is zero in this Delft experiment.

## Results

There are 48 fixed conditions and 240 complete trajectories. Median 31-36 s radial XY RMSE is:

| Geometry | Retain attitude | Full attitude | Forced yaw | Evidence yaw | Project full yaw |
|---|---:|---:|---:|---:|---:|
| all8 | 0.096320 m | 0.095291 m | 0.095146 m | 0.096107 m | 0.095305 m |
| lower4 | 0.148245 m | 0.139875 m | 0.139657 m | 0.148245 m | 0.139688 m |

These aggregate medians should not be confused with paired improvement. Full attitude and forced yaw each worsen 13 of 24 paired cases in each geometry. Their median paired changes are close to zero and slightly adverse. The gate accepts 8 of 48 conditions, producing four improvements and four regressions. Its worst accepted regression is +0.005879 m; its largest improvement is -0.016026 m. It leaves both geometry-wide worst errors unchanged. This study does not demonstrate a substantial independent advantage for the evidence gate under this live-tracking protocol.

At zero injected offset the gate rejects every case. Forced yaw slightly worsens 11 of those 12 cases. This is useful rejection behavior, but it is not evidence that all accepted large-offset corrections help. The accepted-case table includes the adverse cases explicitly.

The full-SO(3) competitor has eight optimizer non-success flags out of 48, and its largest attitude replacement changes the tilt by 33.39 degrees. These cases are retained. There is no measured yaw-accuracy score, and the position results do not establish that a particular heading estimate is more accurate.

Common outage errors remain large: the maximum radial XY error during the shared 25-27 s blind translation acquisition is 27.70 m with all8 and 30.85 m with lower4. Changing a 31 s attitude decision cannot repair that earlier history. Lower4 vertical error also remains substantial: median fixed-window Z RMSE is about 0.983 m for retaining attitude and 0.981 m for the evidence gate.

## Reproduction and artifacts

From `E:\0syncthing\Code\InEKF`:

```powershell
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
./.venv/Scripts/python.exe imu-uwb-inekf/experiments/split_handoff/delft/study.py --prepare --workers 6
./.venv/Scripts/python.exe imu-uwb-inekf/experiments/split_handoff/delft/analyze.py
```

`results/metrics.csv` retains all interval scores; `paired_metrics.csv` adds each method's change from its matched attitude-retaining branch. `summary.csv` groups by geometry and injected offset, while `aggregate.csv` groups by geometry. `decisions.csv`, `accepted_cases.csv`, and `gate_reasons.csv` expose candidate quality and rejection behavior. Full sensor-only caches, separate scoring positions, estimated positions/velocities/rotations/biases, common physical covariances, and raw heading-window inputs are retained under `results/cache` and `results/histories`.

`validation.json` verifies identical branch prefixes, position/velocity/bias means at the attitude fork, common world attitude covariance, exact gravity-direction preservation for yaw-only changes, causal disjoint radio groups, and finite output in every case. Same-day recordings, inherited truth-assisted training calibration/timing and valid-interval selection, conservative covariance choices, and the single fixed radio split limit generalization. Computational timing is offline and does not establish real-time handoff latency.
