# Early horizontal position handoff

This independent experiment does not overwrite the frozen split-heading matrix.
It uses its sensor caches, original six later Delft TWR recordings, six earlier
calibration recordings, all8/lower4 layouts, Profile-1 startup, and controlled
world-z mean offsets 0/+90/-90/180 degrees at 15 s. These offsets are not measured
yaw errors: evaluation yaw truth is unavailable.

## Frozen availability rule

After the 15–25 s artificial radio silence, check received packets at intervals
of at least 0.05 s. Fit radio-only constant-velocity position/velocity using the
received prefix (at most the latest two seconds if no trigger has occurred).
Use per-range standard deviation at least 0.30 m, with an additional
`0.5 * 3 m/s² * tau²` acceleration-model envelope in quadrature. These are fixed
conservative working scales, not noise estimates fitted to evaluation errors.

Project all z/vx/vy/vz Jacobian columns out of the two horizontal columns using
rank-aware SVD. Do not invert a singular nuisance block or turn null information
into a zero variance. Require at least four distinct anchors, horizontal rank
two, normalized residual RMS at most 2, nuisance speed at most 8 m/s, and no
comparable reflected-z solution separated by more than 1 m in XY. Add a 0.50 m
horizontal covariance floor. Require the 95% horizontal ellipse major radius to
be at most 2 m and this covariance to be smaller than the predicted InEKF XY
covariance in every direction. The 2 m limit is a horizontal usage requirement
at one reference operating point, not a universal physical constant.

The local information calculation remains conditional on the fitted model;
absolute scales, floors and the reflected seed make it more conservative but
do not prove global uncertainty coverage. Lower4 has about 0.49 m anchor height
span, not exactly coplanar geometry. Rank-deficient horizontal information is
rejected. A z-only null direction is projected by its actual numerical rank;
vertical position is never handed over. Reflected-z alternatives are checked for
horizontal consistency rather than choosing a vertical branch using truth.

## Branches and information accounting

- `fixed_2s_pv`: predict without radio from 25–27 s; at 27 s use the original
  robust radio CV handoff of p and v, retaining the current attitude and biases.
- `native_gate`: from the first packet at 25 s, apply the existing scalar InEKF
  radio gate continuously. No window reset.
- `early_only_live`: reserve packets until the availability rule triggers;
  replace **only pXY**, preserve v/z/R/bias and their retained physical covariance
  block, clear XY cross-covariances, then update on **later packets only**. There
  is no 27 s replacement. The triggering sample is not scalar-updated again.
- `early_common27`: diagnostic attribution only; shares the early branch until
  27 s, then copies the fixed branch's complete state and all future outputs.
  Its future equality is imposed and is not evidence of independent recovery.
- `continuous`: same 15 s controlled attitude perturbation, but radio remains
  available throughout 15–25 s. This is a no-outage reference.

All branches share the exact pre-15 s prefix. None performs an explicit heading
fit or replacement. After 27 s every tracking branch uses all newly received
active-anchor packets, so future scores belong to this independent protocol,
not to the frozen heading experiment's half-radio split. Native range updates
may subsequently affect attitude and bias through the filter. Any early gain
is from timely horizontal position information, not an explicit yaw estimate.

The branch estimators share IMU, prior, calibration and radio history; their
posteriors are not independent. Radio window packets are used only once in the
early branch, and no evaluation positions enter its fit, gate or initialization.
Training-position calibration, inherited data-valid intervals and the fixed
sensor timing correction are shared with the frozen matrix. Offline processing
time is not added to sensor timestamps; this is not a measured latency claim.

## Outputs and reproduction

`metrics.csv` includes every branch/case's fixed 25–27 s XY radial RMSE and peak,
31–36 s future RMSE/peak, post25 score and secondary Z score. `paired_metrics.csv`
contains paired changes versus fixed waiting. `triggers.csv` records trigger
delays relative to 25 s and all non-trigger cases. `attempts.csv` retains every
availability check, including rejected rank/geometry/model cases. Histories
contain complete p/v/R/bias outputs, with separate JSON handoff-preservation
checks and packet counts. `validation.json` audits invariants and finite output.

From the project root in PowerShell:

```powershell
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
./.venv/Scripts/python.exe imu-uwb-inekf/experiments/split_handoff/delft/early_position.py --workers 6
```

`protocol.json` was written before the first result. No threshold is selected
using the new results. All conditions and failures are retained.

## Completed results

All 48 conditions completed with finite outputs (240 frozen branch trajectories,
plus 48 separately derived display trajectories). All yaw offsets have exactly
the same pre-15 s prefix. At every XY handoff, the maximum change in R, bias,
velocity and z was exactly zero; the retained physical covariance block changed
by at most `7.11e-15` numerically. Synthetic horizontal coplanar geometry accepted
XY; coincident and collinear anchors rejected as `horizontal_rank_deficient`
(`degeneracy_sanity.json`). The real matrix's rejected checks were insufficient
distinct anchors (44), implausible velocity nuisance (20), or the horizontal
usage requirement (8); all 48 ultimately triggered. The first four anchors at
one IMU timestamp can already provide a zero-delay trigger in a bundled batch.

The availability rule triggered at 0–0.12 s after 25 s for all8 (median 0.070 s),
and 0.06–0.18 s for lower4 (median 0.125 s). Median delays after the first received
packet were 0.065 and 0.115 s respectively (`packet_trigger_timing.csv`). These
are sensor-time evidence delays; offline computation latency is excluded.

`early_live_display` is a separate deterministic, causal output selector:
before the stored trigger it outputs the already-running native branch; from
the trigger onward it outputs the independent early branch. If never triggered
it would remain native. It does not merge posteriors or change either estimate.
See `display_analysis.py`, `display_histories/` and `display_validation.json`.

| Layout | Method | 25–27 XY median RMSE (m) | Worst early XY peak (m) | 31–36 XY median / worst RMSE (m) |
|---|---|---:|---:|---:|
| all8 | Fixed 2 s p/v | 12.370 | 27.698 | 0.0945 / 0.189 |
| all8 | Native gate | 3.001 | 63.871 | 4.153 / 170.876 |
| all8 | Early XY only | 1.979 | 18.863 | 0.0944 / 2.211 |
| all8 | Causal live display | 0.453 | 42.831 | 0.0944 / 2.211 |
| all8 | No silence | 0.098 | 0.244 | 0.0936 / 0.184 |
| lower4 | Fixed 2 s p/v | 14.878 | 30.850 | 0.1451 / 0.328 |
| lower4 | Native gate | 3.369 | 71.009 | 6.175 / 212.892 |
| lower4 | Early XY only | 3.476 | 22.992 | 0.1542 / 64.730 |
| lower4 | Causal live display | 1.701 | 46.508 | 0.1542 / 64.730 |
| lower4 | No silence | 0.140 | 0.482 | 0.1348 / 0.274 |

Early XY reduced the early RMSE against waiting in **48/48** paired conditions.
It was worse than native in 8/24 all8 and 10/24 lower4 conditions; display was
worse than native in 4/24 and 2/24. Thus native-first output reduces typical
waiting error, but the native transient before switching can exceed the fixed
waiting peak. The display does not solve the backend's remaining failures.

Using 1 m only as a descriptive future-error reference, early XY had 1/24 all8
and 6/24 lower4 cases above it, versus native's 14/24 and 16/24. Fixed 2 s p/v and
no-silence had none. All early-XY cases above that reference are retained below:

| Motion | Layout | Injected offset | 31–36 XY RMSE (m) |
|---|---|---:|---:|
| octagon | all8 | 180° | 2.211 |
| octagon | lower4 | 180° | 1.016 |
| hourglass | lower4 | 0° | 58.490 |
| hourglass | lower4 | +90° | 64.730 |
| square | lower4 | 0° | 9.069 |
| square | lower4 | +90° | 10.616 |
| square | lower4 | 180° | 22.001 |

The hourglass/lower4/+90 condition's complete post25 XY RMSE reaches 202.666 m;
fixed-future scores alone do not represent its later divergence. XY-only handoff
deliberately leaves drifted velocity, height, attitude and biases unchanged.
The retained state plus native scalar gate can still fail to recover. These
results establish an early-output benefit at the frozen operating point, not
general recovery from a ten-second outage, and not a yaw-estimation benefit.

`early_common27` has the same early errors as early XY and **exactly** the same
future states as fixed waiting by construction. It isolates the cost of
withholding already available XY; it cannot establish independent future
recovery. The all-new continuous-radio future protocol also differs from the
frozen heading matrix's reserved-radio protocol, so their future medians must
not be compared as though only the handoff timing changed.

`combined_paired_metrics.csv` contains all 288 rows, every paired early/future
change versus fixed waiting and native, with secondary Z scores.
`combined_aggregate.csv` and `early_display_comparison.png` summarize the full
distribution; `paired_position.png` deliberately retains the large failures.

## Independent staged XY then p/v supplement

`../staged_position.py` adds 48 further conditions without modifying either
frozen table or availability threshold. It replays to the stored trigger to
recover exact physical covariance; the reconstructed p/v/R/bias at that instant
matches the saved early branch exactly. The early XY batch ends at the trigger.
For strictly later packets through the 27 s endpoint, restart each anchor's
received-packet count at zero: odd packets update the live InEKF and even packets
are reserved for a single radio-only CV p/v fit. At 27 s, perform the original
conservative p/v physical-covariance handoff from the reserved packets only,
**retaining the actual live R and bias**, then process all later packets normally.
No old branch's attitude or complete state is restored.

The XY batch, live batch and reserved p/v batch do not overlap. The branches
still share IMU, prior and calibration, so the batch estimates are correlated;
the handoff clears unsupported translation cross terms and uses the same
0.50 m position / 1 m/s velocity covariance floors as fixed waiting. It is not
a claim of independent posterior fusion. Offline solve time remains excluded.

`staged_xy_pv` outputs inertial propagation before the trigger and actual staged
tracking thereafter. `staged_live_display` instead displays the independently
running native filter until the trigger, then the same staged state. This
separates early-display choice from the actual backend p/v recovery.

| Layout | Output | 25–27 XY median RMSE (m) | Worst early XY peak (m) | 31–36 XY median / worst RMSE (m) | Complete post25 worst RMSE (m) |
|---|---|---:|---:|---:|---:|
| all8 | Staged, inertial prefix | 1.979 | 18.863 | 0.0942 / 0.188 | 1.289 |
| all8 | Staged, native display prefix | 0.448 | 42.831 | 0.0942 / 0.188 | 2.918 |
| lower4 | Staged, inertial prefix | 3.347 | 22.992 | 0.1450 / 1.113 | 2.894 |
| lower4 | Staged, native display prefix | 1.414 | 46.508 | 0.1450 / 1.113 | 2.887 |

Every staged case improved early RMSE against two-second waiting. The slower
disjoint p/v handoff reduced the early-only worst future errors from 2.211 to
0.188 m in all8 and from 64.730 to 1.113 m in lower4. It reduced lower4's worst
complete-post RMSE from 202.666 to 2.894 m. At the descriptive 1 m future
reference, the staged cases are 0/24 all8 and 1/24 lower4. The remaining lower4
octagon/180° case is **1.113 m**, versus fixed waiting's smaller worst error
0.328 m; the staged solution is not uniformly better or failure-free. Its
all8/lower4 median paired future change versus early-only is approximately
`-0.000020 / +0.000008 m`; the main improvement is preventing several large
failures, not improving every already-recovered case.

At the p/v handoff, R and bias changes are exactly zero, with retained physical
covariance difference at most `6.01e-16`. `staged_packet_audit.json` independently
recomputes received-packet parity, verifies complete and disjoint live/reserved
coverage, and checks every pre-trigger state against the original early branch.
All 96 staged output trajectories are finite. No explicit yaw estimate or yaw
truth is introduced, and the large native-prefix display peaks remain visible.

Outputs: `staged_metrics.csv`, `staged_combined_paired.csv` (384 rows),
`staged_aggregate.csv`, `staged_counts.csv`, `staged_validation.json`,
`staged_histories/`, `staged_comparison.png`, and the full troublesome condition
`staged_hourglass_failure.png`. Run the supplement and audit with:

```powershell
./.venv/Scripts/python.exe imu-uwb-inekf/experiments/split_handoff/delft/staged_position.py --workers 6
./.venv/Scripts/python.exe imu-uwb-inekf/experiments/split_handoff/delft/early_results/staged_visualize.py
```
