<!-- Research entry point: raw UWB geometry, compliant inertial trajectories and moving acquisition. -->
# UWB–inertial window models

This research module compares two ways to improve position acquisition. With a previous gravity alignment, a rotating accelerometer-bias model and a direction-dependent radio mean outperform the affine initializer on Delft TWR. Without a previous attitude, a local full-rotation model works with either TWR or native signed TDoA. The ordinary InEKF entry point is unchanged.

The [Chinese mathematical note](../../../tex/range_inertial_models_zh.pdf) contains the main equations and comparisons. [Compact evidence](evidence) retains tables for the principal claims; [paired window scores](results_restart/windows.csv) include capped solves.

| Delft endpoint RMSE (m) | All eight TWR | Four lowest TWR |
|---|---:|---:|
| Affine IMU shape, static radio offset | 1.799 | 2.029 |
| Free quadratic shape, same radio model | 0.320 | 0.924 |
| Rotating body bias with prior, same radio model | 0.279 | 0.874 |
| Rotating body bias and geometric radio mean | 0.239 | 0.560 |

All rows share four yaw starts. Each layout uses the same 66 overlapping windows in six evaluation flights. With the target motion entirely excluded from calibration, the geometric mean still reduces rotating-bias RMSE from 0.289 to 0.250 m (all eight) and from 0.864 to 0.616 m (four lowest). This is transfer across motions within one room and date.

The moving-window method integrates local angular increments from identity, estimates the full initial rotation, and never needs a preceding static interval. On Delft, its fixed-bias endpoint RMSE is 0.263 m on all-eight TWR and 0.417 m on all-eight TDoA, compared with 0.340 and 0.602 m for the radio-only motion smoother on exactly the same events. The six TDoA test recordings are distinct from the six TWR test recordings. The Delft logs provide no attitude truth.

On the second site, UTIL, the separate full-rotation experiment **does not establish reliable attitude acquisition**: 16 preselected moving windows from eight trial3 recordings yield a 37.0° median rotation error and only 6/16 below 15°. Each competing seed uses the same 0.10 m/s² body-bias prior and the known lever arm. The antenna-only and body position targets differ here.

## Models and code

| Question | Implementation / evidence |
|---|---|
| How much is due to finite IMU trajectory stiffness? | [window_dynamics.py](window_dynamics.py) implements a Gaussian correction per acceleration cell; [delft_windows.py](delft_windows.py) compares rigid, constant-defect and radio-only controls. The fixed one-second cell model does not consistently beat a single constant correction. |
| Can position be acquired without prior tilt or yaw? | [rotation_window.py](rotation_window.py) integrates force and optional body-bias sensitivity, eliminates affine position/velocity for a proper-rotation seed, then fits the raw radio factors. [moving_restart.py](moving_restart.py) uses only window IMU samples. |
| Are bias/tilt and radio calibration separate effects? | [Shape controls](shape_branch/README.md), [joint TWR factors](joint_branch/FORMULAS.md) and [held-motion calibration](joint_branch/novel_motion/study.py). The learned line-of-sight term is an empirical mean, not an identified NLOS condition or corrected survey. |
| Does nominal anchor displacement transfer between radio protocols? | [effective_anchors.py](effective_anchors.py) fits TWR-only, TDoA-only and joint apparent anchor positions; the [transfer matrix](results_geometry_transfer/summary.csv) includes the no-target-calibration failure. |
| Are attitude gains supported by ground truth? | [Large-tilt and weak-excitation simulation](shape_branch/cold_rotation_checks/FINDINGS.md), [UTIL real-window audit](transfer_rotation/README.md) and [TDoA tracking controls](tdoa_branch/README.md). |
| Does joint covariance fix radio outages? | [Matched 10-second outage](outage_branch/README.md). Immediately using resumed readings explains most of the recovery improvement. During the gap the ordinary inertial trajectories are exactly identical. |

## Reproduction

Run from the repository root in PowerShell using the project environment. These commands rebuild the primary continuous-shape and moving-window panels from CSV. Inputs default to the local Delft data in the project configuration; the scripts accept a different dataset root. No new dependencies are required.

~~~powershell
$study = 'imu-uwb-inekf/experiments/radio_inertial_models'
$env:OPENBLAS_NUM_THREADS = '1'
& .\.venv\Scripts\python.exe "$study/delft_windows.py" --sensitivity
& .\.venv\Scripts\python.exe "$study/moving_restart.py"
& .\.venv\Scripts\python.exe "$study/effective_anchors.py"
~~~

For Table 1, run the scripts in this order: 'radio_branch/study.py', 'radio_branch/analyze.py', 'radio_branch/ablate_ar.py', 'radio_branch/consolidate.py', 'joint_branch/study.py', 'joint_branch/analyze.py', and 'joint_branch/novel_motion/study.py'. The branch READMEs list the simulation, UTIL and outage commands. After regenerating those panels, 'synthesize.py' checks the physical-window keys and training splits, compares two independently executed control paths, exports the compact tables, and recreates the figure. Run XeLaTeX twice in 'tex' on 'range_inertial_models_zh.tex' to rebuild the note.

The evaluation clock follows the baseline: Delft OptiTrack selects the valid interval and provides the common 0.20 s alignment for scoring, not a 0.20 s relative UWB–IMU correction. The 1.5 s prefix is consumed for the aligned controls; moving restart only uses the same time origin to identify comparable endpoints. Training uses separate logs with truth to calibrate radio statistics. No test position or orientation is passed to a solver. Holding an old measurement constant does not count as another observation.
