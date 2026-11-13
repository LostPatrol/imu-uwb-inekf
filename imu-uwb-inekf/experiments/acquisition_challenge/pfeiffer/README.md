# Pfeiffer author kernels: acquisition replay

This directory evaluates the unmodified Python EKF/MHE classes released by
[Pfeiffer et al.](https://github.com/SUPfeiffer/uwb-simulator), using the frozen
Delft acquisition streams and the existing publication-script parameters.
It does not reproduce arXiv:2006.02688v2, which is a different paper by
Theodosis, Berkane, and Dimarogonas.

`run.py` reads only `imu.csv`, `radio.csv`, `anchors.csv`, and, for the sealed
recovery stress, `priors.csv`. The replay never opens scoring files. `analyze.py`
opens scoring separately and holds the latest published state at each deadline.
No truth is extrapolated before the first available OptiTrack sample.

```powershell
.venv/Scripts/python.exe imu-uwb-inekf/experiments/acquisition_challenge/pfeiffer/run.py --workers 2
.venv/Scripts/python.exe imu-uwb-inekf/experiments/acquisition_challenge/pfeiffer/run.py --stress --method EKF --horizon 10 --workers 2 --output imu-uwb-inekf/experiments/acquisition_challenge/pfeiffer/stress_ekf
.venv/Scripts/python.exe imu-uwb-inekf/experiments/acquisition_challenge/pfeiffer/run.py --stress --method MHE --horizon 10 --workers 2 --output imu-uwb-inekf/experiments/acquisition_challenge/pfeiffer/stress_mhe
```

The native cold treatment supplies original zero position, velocity, and yaw
seeds. Both classes consume the full gravity prefix with a local positive
clock. The radio distances are already training-calibrated; they are passed
once without another correction. The supplied sigma is a standard deviation:
EKF squares it internally, whereas the author MHE gradient/RANSAC ignores it.
No additional square root or test-set calibration is applied.

The original drone adapter derives `thrust = acc_z * mass`; this replay uses
the same definition, not another measured input. Nevertheless both cores use
the original quadrotor thrust/drag dynamics. EKF discards the accelerometer
vector in `step`; MHE uses it in its separate Mahony attitude estimate.
Neither is presented as a generic full-vector inertial mechanization. The
OptiTrack-derived altitude and truth-dependent online range variance of the
original simulator are not used.

The recovery stress uses the sealed supplied center at release, zero velocity,
and the author original reset conventions. EKF accepts a pure-yaw initial
quaternion and its existing covariance array receives the requested diagonal
prior; these are explicit state injections, not a new optimizer or changed
filter equations. Its existing `MAX_COV=100` may clip a requested 400 m²
diagonal during a measurement update or finalization. EKF reset copies only
quaternion w/z and starts its rotation matrix at identity, so no full tilt
initialization is claimed. After the reset the clock starts one nominal update
before release to retain the original integrator's positive `dt` requirement.

MHE reset uses initial position only: it ignores initial quaternion and has
no covariance input. Its 6-state optimization never estimates yaw; Mahony gets
no UWB feedback. Therefore only the 16 distinct XY position centers per
recovery episode are replayed for MHE. The corresponding `_0d_wide` case IDs
identify rows in the sealed prior file; neither heading nor width is an
effective MHE parameter. Running 96 combinations would count repeated identical
treatments. Delft yaw offsets are relative to the sensor convention, not known
physical yaw errors. No accelerometer vector is rotated to manufacture them.

On replay, files in `native`, `stress_ekf`, and `stress_mhe` contain every author update;
`wall_s` is cumulative replay compute including prefix, not a queue simulation.
`arrival_s` is the sensor-time publication opportunity and `state_s` is the
returned author state epoch. Fixed 2/10 s stress accuracy and first usable
position are different outcomes.

Completed per-update CSVs are archived under
`agent/codex/pfeiffer_acquisition/traces/{native,stress_ekf,stress_mhe}` to keep
large trajectories out of the code tree. The code-side directories retain
`execution.csv` and per-case `metrics.csv`; `trace_manifest.json` records counts
and frozen input hashes. The final run contains 24 native replays, 576 EKF
recovery cases, and 96 distinct MHE position seeds. All 24 native and 96 MHE
runs completed; 34 EKF stress cases hit original numerical state assertions.
The remaining 542 EKF cases completed. A recorded failed filter is not held
at its last finite state for later endpoint scoring.
