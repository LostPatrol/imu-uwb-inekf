# UTIL dynamic-window rotation transfer

This experiment checks the current full-SO(3) window acquisition on UTIL trial3, independently of the Delft TWR evaluation. It is a cross-site/modality check, not a fully blind test: some trial3 trajectories had previous project exposure.

Settings were fixed before scores: const1--4, TDoA2 and TDoA3, const4 traj1, windows [10,16] and [25,31] seconds from the common sensor origin. All eight requested files were present. No trajectory or window was selected by its result.

Raw accelerometer and gyro streams are cropped to the requested window before interpolation. A 100 Hz grid is merged with selected radio transaction times; the first radio event in each 50 ms bin is kept. Thus integration is at 100 Hz plus actual radio times, with no rounded radio timestamp. Raw acceleration g and gyro deg/s are converted to SI. No static prefix, prior attitude, prior-window bias calibration or truth measurement enters an estimator. The published body-to-radio lever arm is retained; plus is idB and minus is idA. Noise sigma=0.30 m and body accelerometer-bias prior sigma=0.10 m/s2 are fixed; gyro bias is not estimated. This isolates the current method and is not a claim to handle all calibration errors.

Every inertial method shares the same radio-only compliant path seed and raw events, and solves with the same bias prior. Identity and window-mean gravity provide single initial rotations; projected Wahba supplies the proposed seed; uniform4 tests four fixed world-yaw rotations of the same gravity seed and picks the lowest objective. Its total cost includes all four solves. The in-window force mean used for the gravity seed is a heuristic during dynamic motion.

Truth is read only after fitting for endpoint position, SO(3) angle and wrapped yaw; no rigid alignment is fitted. Inertial endpoint positions target the body. Radio-only has no orientation, so its endpoint targets the antenna; truth rotation is used only to construct the correct antenna scoring target. Its position numbers are therefore not precisely the same body metric.

## Result

Across 16 windows, projected Wahba median body error is 0.240 m, full SO(3) error 37.01 degrees and yaw error 36.91 degrees. Identity, window gravity and uniform4 have essentially identical median position and about 33.99 degrees SO(3) error. Each has only 6/16 windows below 15 degrees SO(3) error; this descriptive threshold is not an acquisition gate. Wahba does not show an attitude-accuracy advantage. One const2 TDoA2 window reaches a different higher-cost solution (cost difference about 0.343 versus identity); the other 15 objectives agree to numerical tolerance.

Median complete common-seed plus method solve time is about 8 ms for Wahba, 12 ms for identity/gravity and 41 ms for uniform4 on this machine. These tiny-window desktop timings are not embedded latency guarantees. Radio-only antenna median error is 0.221 m. The inertial body 95th percentile is about 1.166 m; radio-only antenna is about 1.317 m.

The prescribed identity seed happens already to have only 1.73 degrees median physical endpoint attitude error on these recordings, whereas its fitted result has about 33.99 degrees median error. This is a particularly clear warning: objective convergence does not establish attitude correctness, and these recordings do not test arbitrarily large true initial rotations. Projected-Wahba initial error is 17.43 degrees median. Weak horizontal excitation and bias/attitude ambiguity remain material limitations.

Saved rotation information removes translation and free constant accelerometer bias, and separately adds the actual bias prior. The minimum robust local curvature eigenvalue without the prior spans approximately 0.0030--0.265 rad^-2. These are conditioning diagnostics, not calibrated confidence probabilities or proofs of global unobservability. Force cross-covariance singular values are also saved; dominant gravity does not establish yaw excitation.

Run from the project root with its `.venv/Scripts/python.exe`:

```powershell
& ./.venv/Scripts/python.exe imu-uwb-inekf/experiments/radio_inertial_models/transfer_rotation/study.py
& ./.venv/Scripts/python.exe imu-uwb-inekf/experiments/radio_inertial_models/transfer_rotation/plot_results.py
```

`metrics.json` and per-window JSON retain every method and window; `metrics.csv` is a flat view; `transfer.png` displays all outcomes. No settings were adjusted to improve these scores.
