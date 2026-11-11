# Predictive circular heading bank

`bank.py` exports `decide(tau, force, B, plus, minus, y, sigma, path, Cref)` for the shared UTIL replay. Angles are radians; rejected corrections return `delta=0`. `candidate` remains the winning grid angle. Diagnostics contain only JSON-serializable values.

The eight particles hold world-vertical yaw offsets at 45-degree spacing. They estimate separate translation, velocity and constant body accelerometer bias from a past radio prefix:

\[
 z_i=h_i\{p+\tau_i v+\tfrac12g\tau_i^2+
 R_z(\delta_j)C_{ref}[f_i-B_i b_j]\}+\epsilon_i,
 \qquad b_j\sim N(0,0.1^2I).
\]

The first prefix spans one second. Each particle predicts the next nonoverlapping block before those radio observations enter any particle fit. A tempered robust predictive score updates circular weights; the expanded prefix is refitted for the next block. The retained hypothesis receives prior odds 4:1 against each alternative. A switch needs at least three scored blocks, the same winner for two blocks, mass at least 0.85 and odds at least 9:1 against retention. These numerical rules are frozen in `frozen_rules.json` before UTIL scoring. They are not calibrated frequentist confidence statements.

This differs from selecting and gating an offline optimum: no full-window radio trajectory or full-window fitted yaw supplies the candidates or their p/v seeds. The `path` argument is deliberately ignored. Endpoint-reanchoring of the IMU shape adds affine functions of time. Projecting the force, gravity and bias bases against `[1,tau]` using prefix rows removes that gauge without restricting p/v, including when the endpoint lies after the prefix. The current API still conditions all hypotheses on the supplied `Cref`; if this reference comes from a terminal radio-updated attitude, the complete pipeline must not be called a strictly online causal filter. Only the internal prefix-versus-next-block fit/score separation has been verified.

The coarse grid separates large heading basins. It does not resolve arbitrary subgrid heading errors, supply new tilt estimates, or claim accurate heading uncertainty: `sd_deg` includes a 12.99-degree grid quantization floor. The shared InEKF provides subsequent local estimation. Cold two-second windows have fewer than three scored blocks and deliberately retain. Weak motion may retain a wrong initial heading indefinitely.

Run the synthetic implementation checks from the project root:

```powershell
.\.venv\Scripts\python.exe imu-uwb-inekf/experiments/split_handoff/bank/verify_bank.py
```

The shared `split_handoff/util/run.py` owns real raw-packet generation, no-truth decisions, matched reset covariance and subsequent InEKF integration, followed by a separate truth-scoring phase. Bank does not rebuild those components. Its fixed split uses trial3 development and trial4/5 holdout, all four constellations and TDoA2/3, with nominal and injected +90/-90/180-degree cases. Refer to the shared protocol and generated `metrics.csv` for the exact schedule and common baselines.

The corrected shared replay and bank summaries can be reproduced with:

```powershell
.\.venv\Scripts\python.exe -u imu-uwb-inekf/experiments/split_handoff/util/run.py --split development --output development_final
.\.venv\Scripts\python.exe -u imu-uwb-inekf/experiments/split_handoff/util/run.py --split holdout --output holdout_final
.\.venv\Scripts\python.exe imu-uwb-inekf/experiments/split_handoff/bank/analyze_util.py --inputs imu-uwb-inekf/experiments/split_handoff/util/development_final imu-uwb-inekf/experiments/split_handoff/util/holdout_final
```

The final command only scores existing matched output tables. It writes bank-local paired case tables, scenario/geometry/injection summaries and the post-handoff XY comparison figure; it does not fit or change any heading rules.

Optional exploratory accepted-case short-horizon scoring reuses the shared truth scorer and saved trajectories, with no filter replay:

```powershell
.\.venv\Scripts\python.exe imu-uwb-inekf/experiments/split_handoff/bank/score_accepted.py --inputs imu-uwb-inekf/experiments/split_handoff/util/development_final imu-uwb-inekf/experiments/split_handoff/util/holdout_final
```

`FINDINGS.md` reports the complete negative result: only five switches among 288 matched cases; all three held-out switches worsen immediate heading, and the bank has no general subsequent XY gain. The rules are retained as frozen rather than retuned to those outcomes.

The synthetic checks are implementation evidence only, not independent real-data generalization. They supply oracle tilt while perturbing yaw; real replay uses its estimated common reference. `verification/implementation_checks.json` records source hashes, fixed physical smoke cases, unused-path invariance, affine-gauge invariance, unchanged early traces after late-radio corruption, and the two-second defer check.
