# SFUISE author-core acquisition comparison

This adapter builds [KIT-ISAS/SFUISE](https://github.com/KIT-ISAS/SFUISE) at
`75bf5a32f1a8e5c3046a5bd1a1ddf659fa996f7d`. It retains the author's spline
representation, analytic residuals/Jacobians, custom LM, integration, window growth,
and sliding-window logic. It removes ROS transport, configuration and visualization.
The author code is GPL-3.0; its original repository and license remain in the isolated clone.

The method is described in [Continuous-Time Ultra-Wideband-Inertial Fusion,
RA-L 2023, especially IV-B](https://arxiv.org/html/2301.09033v2#S4.SS2).
The native algorithm jointly estimates gravity and the navigation-to-anchor transform.
It initializes the local pose at identity/zero, and supports no external initial-pose
covariance. Surveyed anchor coordinates are accepted. Ground-truth pose values and
timestamps are never read by this runner.

## Two explicitly different runs

- `native.csv`: sensor restart, with the author's identity initial frame transform;
  supplied recovery priors are ignored.
- `stress.csv`: an **initial-iterate adapter** sets the initial navigation-to-anchor
  transform to the supplied body position and attitude center. The transform remains
  freely optimized. No prior factor or covariance is added. Wide/tight rows with
  the same center reuse one trajectory and are marked in `audit.json`.

The initial local body pose is identity/zero, so an initial transform `(R,p)` gives
exactly the supplied physical body attitude `R` and IMU-origin position `p` in the
surveyed anchor frame. `R0` supplies common sensor-only tilt; `yaw_rad` replaces its
arbitrary yaw by left multiplication about anchor-frame Z. IMU vectors, sensor
coordinates and scoring truth are unchanged. Only common fixed sensor bias is
subtracted before the author estimates incremental biases starting at zero.

## Input and output

Each common stream folder contains:

- `imu.csv`: `t,ax,ay,az,gx,gy,gz`, SI units, including the common negative-time prefix.
- `radio.csv`: `t,anchor_id,range_m,sigma_m`, already commonly range-calibrated.
- `anchors.csv`: `anchor_id,x,y,z`, in meters.
- `meta.json`: `R0,gyro_bias,accel_bias,lever_m` and dataset provenance.
- Optional `priors.csv`: recovery centers and case labels.

Only packets with `0 <= t <= duration` enter the estimator. Bias estimates in metadata
are fixed common preprocessing. The numerical kernel receives zero extra per-anchor
offsets. In particular, the author's Walk1 range offsets are **not** additionally
applied. `sigma_m` is retained in common input but the author uses its frozen
configuration weights instead.

Outputs contain `episode,case_id,arrival_s,state_s,x,y,z,wall_s,yaw_rad,status`, plus
tag position, body quaternion and the full estimated navigation-to-anchor transform.
`x,y,z` refer to the IMU origin in anchor coordinates. `arrival_s` is the causal 1 ms
stream polling clock; `state_s` is the latest supported spline state timestamp.
`wall_s` records elapsed computation time separately. No forward extrapolation or
ground-truth-time resampling is used. The first output follows an actual optimization
with two active knots, typically at approximately 0.1 s. `optimized` means that the
author optimization ran, and does not assert that LM converged before its iteration cap.

These are **survey-known fusion-core** times. The original ISAS ROS interface first
averages 20 `/anchor_list` messages before handing anchors to the core. The common
protocol supplies a fixed survey at time zero, so this transport warm-up is omitted.
It is not valid to call the measured 0.1 s fusion-core publication a reproduction of
complete ROS startup latency. A sensor calibration prefix may overlap that anchor
warm-up; the two delays must not be blindly added. Original ROS visualization also
used ground-truth timestamps, which this runner removes entirely.

`audit_anchor_warmup.py` measures actual raw-bag arrivals; `Anchorlist` has no Header.
With the common 1.5 s prefix, the twentieth anchor message is already available at
the following times relative to protocol zero. The last column instead assumes a
new subscription beginning at zero, with no prefix history, and is only an I/O
availability lower bound on complete ROS publication.

| Episode | Ready with prefix (s) | Ready without prefix (s) |
| --- | ---: | ---: |
| Walk2 start | -0.274496 | 1.255430 |
| Walk2 recovery | -0.276859 | 1.228828 |
| Walk3 start | -0.316651 | 1.131998 |
| Walk3 recovery | -0.394452 | 1.061009 |

The first-to-twentieth message wait at the beginning of the full Walk2/3 bags is
1.223776/1.181598 s. Sensor buffering and computation constrain actual publication
separately; this table is not an additional delay to add to fusion-core timing.

## Build and run on Windows

Build artifacts and isolated dependencies live under `agent/codex/sfuise_challenge`.
Use the project's `.venv/Scripts/python.exe`. Dependencies are Eigen 3.4,
SuiteSparse 7.10.1 and Boost headers 1.84, obtained from conda-forge using an isolated
micromamba prefix. Set `MAMBA_ROOT_PREFIX` inside that build directory to avoid long
Windows package-cache paths. No ROS, WSL or system settings are required.

```powershell
& imu-uwb-inekf/experiments/acquisition_challenge/sfuise/build.ps1
& .venv/Scripts/python.exe imu-uwb-inekf/experiments/acquisition_challenge/sfuise/run_batch.py `
  --streams agent/codex/acquisition_challenge_20261113/streams `
  --output agent/codex/sfuise_challenge/results --episodes delft_random_start `
  --native-only --duration 15
```

Use `--stress-only --workers 2` with recovery episodes to run their distinct centers.
Every center starts a fresh executable, resetting author function-local static state.
BLAS/OpenMP threads are fixed to one per process. A 90 s wall timeout terminates an
unexpectedly slow case and records the failure. Run records and aliases are in
`audit.json`; `iterate_###.csv` files retain the individual results.
A zero process return code means finite publications, not successful localization.
Large errors and divergent trajectories remain in the result files for scoring.

## Limited portability changes

`prepare_core.py` verifies the fixed commit and writes `source_manifest.json` with
source hashes and every transformation. Most numeric headers remain byte-equivalent
apart from UTF-8 newline normalization. GCC variable-length arrays are replaced by
bounded arrays/vector storage with identical indexing for MSVC.

One startup boundary is explicitly corrected. At one active knot, author
`initialization()` calls `integration()` before adding the second knot. Interpolation
at the first timestamp requests `knots[1]`, although only `knots[0]` exists. On this
Windows/MSVC build the original unchecked access terminated with `0xC0000005` before
any optimized publication. This is a C++ bounds issue; it does not establish what a
particular original Linux execution happened to return. The port repeats the sole
endpoint for that one request, with checked indexing elsewhere. This preserves the
constant initial pose/bias needed for the first integration and leaves all later
residuals, Jacobians and optimizations unchanged.

`validate_boundary.cpp` compares this one-knot boundary evaluation with a valid spline
containing two duplicate initial endpoints. At the first integration timestamp, the
combined difference of position, velocity, acceleration, quaternion, angular rate
and bias is exactly zero. The original failure is retained under
`agent/codex/sfuise_challenge/smoke/isas_walk1/audit.json`. Therefore the delivered
implementation is an **author-core port with a disclosed startup boundary correction**,
not an unmodified binary reproduction or a new implementation of the method.

## Frozen configuration

The Walk1 author settings are used across datasets without test tuning: 100 knots,
10 Hz knots, five LM iterations, 50 full-window calibration steps, UWB weight 7,
range rejection threshold 0.3 m enabled once the window reaches 50 knots, and the
published accelerometer/gyroscope/bias weights. The calibration transform remains
free during startup, as required by the author's IV-B design. Its early orientation
can consequently fluctuate considerably; outputs and failures are retained as measured.
