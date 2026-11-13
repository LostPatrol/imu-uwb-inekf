# Toronto SFUISE / Oursland native TDoA comparison

This independent adapter runs the pinned KIT-ISAS/SFUISE numerical core
(`75bf5a32f1a8e5c3046a5bd1a1ddf659fa996f7d`) on the same four UTIL constX
trial4 single-tag TDoA2 streams as `toronto_heading`. It does not replace the
separate absolute-ranging comparison.

`run.py` reads exported accelerometer, interpolated gyro, native eight directed
TDoA pairs and surveyed anchors. It never supplies Vicon position or quaternion
to the native executable. TDoA is in meters, `rhoB-rhoA`; acceleration and gyro
are converted from g and degrees/s exactly once. Official UTIL settings are
preserved, including 100 Hz IMU decimation, every radio packet, 10 Hz knots,
100-knot window, five iterations, noise weights, and the published rounded
lever arm `[-.012,.001,.091]` m. Oursland retains its original calibrated lever
`[-.01245,.00127,.0908]` m. No dataset tuning is performed.

The generated numerical headers are an unchanged independent copy of the sibling
SFUISE portability extraction. Its explicit startup correction repeats the sole
first spline endpoint where upstream reads an absent second knot. This is a
ported author core, not an unmodified original executable. The rest of the
generated changes and hashes are recorded in the copied `source_manifest.json`.

Both methods are evaluated from the first raw accelerometer timestamp to 15 s.
Oursland starts publishing after its existing 1.5 s arrived IMU + TDoA-CV sensor
prefix. SFUISE starts from native zero local pose and identity world transform,
may publish earlier, and continuously estimates its transform. These startup
procedures use the same available raw stream, but do not use identical bootstrap
algorithms. The official SFUISE wrapper decimates IMU; Oursland uses every sample.

Availability includes a conservative fixed maximum future-gyro interpolation
wait, then a serial computational service queue. Native SFUISE records each
optimization's cumulative wall time; successive differences are queued at the
logical arrival times. Oursland retains its recorded event computation queue,
shifted to the common raw origin plus the same gyro delay. Earlier native outputs
remain available; no state is backfilled. Every 0.5 s, the latest available pose
is held and compared with current-time Vicon without spatial/yaw alignment.
Three consecutive good samples confirm one second capture. The table also
records fraction of the entire grid within each threshold and whether accuracy
persists after first capture, as well as 10/15 s physical XY and yaw errors.

Build with `build.ps1`, then run the project's `.venv/Scripts/python.exe run.py`.
The read-only dependencies remain in `agent/codex/sfuise_challenge/deps/Library`;
the executable and converted input/output files are isolated under
`agent/codex/toronto_sfuise`. `metrics.csv`, grid CSVs, `execution.json`,
`protocol.json`, `hashes.json`, and `overview.png` are the deliverables here.
This branch tests four native starts only. Recovery and explicit probabilistic
priors are not tested by this adapter; SFUISE has no covariance prior input.

The recorded CPU times were collected on the shared workstation while other
experiment jobs were active. The availability curves faithfully include those
observed latencies, but are not an isolated hardware benchmark. `audit.py`
validates finite outputs and causal monotonic clocks and records shared sensor
hashes plus the unchanged Oursland unbuffered array digest. The first IMU packet
is retained at the common zero stream origin; later packets use the official
strict `dt > .0099 s` decimation rule.
