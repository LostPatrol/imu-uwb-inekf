# ISAS-Walk held-out acquisition episodes

Exports the three official [SFUISE ISAS-Walk recordings](https://github.com/ASIG-X/SFUISE/tree/f29744d81e41266a10df72d22ffd39b841004b22/dataset), including original IMU vectors, absolute UWB ranges, fixed anchors, and HTC VIVE tracker poses. These are ToA absolute ranges, not UTIL TDoA differences. The published source does not specify the physical two-way ranging transaction protocol, so the defensible label is **absolute-range ToA**.

Walk1 supplies independent training calibration only. Walk2/3 each supply a 15 s start episode and a 15 s midpoint recovery episode. Sensor availability alone selects restart times. Original IMU vectors remain in the published body frame. Official per-anchor additive range offsets are applied identically to all recordings; `sigma_m=0.15` is a shared nominal model. Stale `ra=0` ranges are excluded.

## Fixed reference calibration

Native VIVE and RTLS coordinates differ, and the authors publish no tracker-to-IMU rigid transform. `calibrate_training.py` estimates these from **Walk1 only**:

1. Centered angular-rate alignment estimates tracker-from-IMU rotation, gyro bias, and an effective gyro/VIVE delay.
2. Robust direct range residuals estimate VIVE-to-RTLS rotation/translation and the tracker-to-tag lever.
3. The published IMU-to-tag lever `[0.1,-0.025,0] m` converts tracker poses to the IMU origin.
4. One fixed Walk1 opening gravity direction defines a z-up world. Only anchors and scoring positions/orientations change world coordinates; IMU/range vectors remain unchanged.

Walk2/3 do not fit any transformation, lever, bias, time offset, or trajectory alignment. Their scoring yaw is a **training-calibrated physical heading reference**, with calibration error. It is not an author-published metrology-grade orientation. Training half splits show approximately 0.4–1.7° variation in the fitted world rotation and 0.5–0.7° in tracker rotation; this is a stability probe, not a statistical confidence interval. Running range variance is not used as a sensor noise variance.

The gyro/VIVE effective delay is about 36 ms. Estimator timestamps retain their original sensor headers; score timestamps add that fixed training delay. There is no independently published hardware-clock calibration, and radio/VIVE effective delay remains uncertain. This limits exact synchronization claims.

## Shared episode contract

`streams/isas_walk{2,3}_{start,recovery}/` contains `imu.csv`, `radio.csv`, `anchors.csv`, and `meta.json`. IMU includes the adjacent `[-1.5,0)` prefix and `[0,15]` interval. An explicit t=0 zero-order-held boundary copies the latest original vector, marked in metadata. `prepare.prefix_estimate` supplies the common R0 and bias estimates using only that local prefix. It receives no earlier gyro history. Recovery episodes use `prepare.write_priors` for the common 96 interventions, with interpolated t=0 training-calibrated XY/yaw and anchor-survey mean Z.

Scoring references live separately in `scoring/<episode>.csv`, cover slightly beyond `[0,15]`, and represent the **IMU origin**. Full native recordings are extraction/calibration material; they are not additional test episodes.

## Reproduce

Run from the project root using its environment:

```powershell
$py = '.\.venv\Scripts\python.exe'
$src = '.\imu-uwb-inekf\experiments\acquisition_challenge\isas_data'
$out = '.\agent\codex\acquisition_challenge_20261113'
& $py "$src\export_isas.py" --raw-dir 'E:\datasets\sfuise-isas-walk' --output-dir "$out\streams"
& $py "$src\calibrate_training.py" --streams "$out\streams" --output "$out\scoring\isas_walk1_calibration.json"
& $py "$src\build_episodes.py" --root $out
& $py "$src\validate_reference.py" --root $out
```

The validator checks exact preservation of physical samples, t=0 held boundaries, prefix/scoring support, and prior offsets. It also writes residual diagnostics and a trajectory/heading plot. Dependencies are the project environment's NumPy, SciPy, pandas, Matplotlib, PyYAML, and rosbags.

NTU VIRAL was screened but not downloaded: its official ground truth is tracked Leica-prism position, without independent full attitude, and its multi-gigabyte recordings do not resolve this experiment's heading-reference requirement.
