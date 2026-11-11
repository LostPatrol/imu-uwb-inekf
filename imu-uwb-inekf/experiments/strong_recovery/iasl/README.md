# i-ASL source audit: excluded from absolute localization validation

All three official scenarios were selected before any algorithm output, with
radio recovery fixed at 25 s and outage lengths 2/5/10/20 s. All twelve slots
remain in the audit as `excluded_metadata`. Neither the reference K nor the
shared recovery overlay was replayed, and no localization score was computed.

Official source: <https://github.com/i-ASL/uwb_imu_dataset>, pinned revision
`41e585b47d84e810d7b448bdb21e3d98a8a9423e`. All 14 files, including the three
approximately 2.2 MB ROS bags, were downloaded unmodified to
`E:/datasets/iasl-uwb-imu`. Every byte count and Git blob hash was verified.

| Scenario | Raw IMU messages | IMU median dt | TWR frames | TWR median dt | GT records |
|---|---:|---:|---:|---:|---:|
| 1 | 1927 | 0.051996 s | 4991 | 0.020000 s | 1000 |
| 2 | 1975 | 0.051956 s | 5090 | 0.020000 s | 1000 |
| 3 | 1928 | 0.051950 s | 4974 | 0.020000 s | 1000 |

The bags contain `/imu/data` with three accelerometer and three gyro channels,
high precision ROS header timestamps, and a common host receive clock. The
message definition states m/s² and rad/s. This is usable six-axis inertial data,
at approximately 19.2 Hz rather than the high-rate Delft input. The official
README calls UWB 60 Hz, but the recorded frame rate is approximately 50 Hz.
Each UWB frame contains eight ranges in `dis_arr`; the supplied YAML gives the
eight anchor coordinates. The CSV IMU timestamps discard fractional seconds.
Scenarios 1/2 CSV values match the bag; scenario 3 includes small gyro value
differences (maximum 0.000373 rad/s), so the bag is the appropriate raw source.
The sensor-only NPZ exports contain no motion-capture labels or device position.

The blocking issue is geometric and temporal metadata. GT CSV time is exactly
0.1…100.0 s for every recording, while the IMU durations are 99.770, 101.763 and
99.428 s. The bags contain neither Qualisys messages nor TF. No official file
specifies a GT clock trigger, a fixed Qualisys-to-anchor transform, a surveyed
tag lever arm, or an IMU-to-Qualisys body rotation. The initial supplied device
UWB position and GT position are numerically different coordinate origins.

All three first GT rotations are close to identity (Frobenius distance 0,
0.000150, 0.000619). This suggests an initialization convention but **does not
prove** that each recording was reset: a common physical starting pose is also
compatible. The official README, its only revision, and the repository issue
provide no statement that the GT frames and clock relation are shared across
recordings. Consequently, fitting one transform/time offset on scenario 1 and
transferring it to scenarios 2/3 lacks the required shared-frame evidence. That
secondary protocol was also excluded. No test GT registration, scale change,
coordinate guessing, timestamp stretching, or per-record offset was applied.

Reproduce using this project's Python environment from the repository root:

```powershell
uv pip install --python ./.venv/Scripts/python.exe rosbags==0.11.9
./.venv/Scripts/python.exe imu-uwb-inekf/experiments/strong_recovery/iasl/download.py
./.venv/Scripts/python.exe imu-uwb-inekf/experiments/strong_recovery/iasl/audit.py
```

Artifacts are in `agent/codex/strong_iasl/`: `download_manifest.json`,
`source_audit.json`, and the three truth-free sensor NPZ files. Source inspection
is lightweight; it performs no nonlinear estimation, training or parameter
selection. If the authors supply a common-frame calibration and trigger
definition, the twelve preselected slots can be reinstated with the frozen K
adapter and shared recovery rules, explicitly reporting the low-rate migration.
