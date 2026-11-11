<!-- Reproducible author-class mechanism audit and separate recovery cohorts. -->

# Author-class mechanism audit and strong recovery comparison

The author source is always read-only. Every primary or supplementary recovery K trajectory below uses the actual author class, not the numerical mirror. No thresholds were fitted to these test scores. The two trial-4 recordings are mechanism development data; trial-2 results are reported in two separate cohorts.

## What explains the heading difference

The production class `scripts/estimation/inekf.py` with `FilterConfig.proposed()` and the standalone research kernel `split_handoff/util/kernel.py` are different algorithms. Production retains the author's transformed TDoA Jacobian. The research kernel uses the complete world right-invariant observation Jacobian, robust weighting and a finite covariance reset.

On the same native queue, original numeric invariant P, sensor-only X/bias and 180-degree yaw injection, the following are 5–45 s wrapped Euler yaw RMSE in degrees. These are controlled development comparisons, not independent test results.

| Method | const1/trial4/tdoa2 | const2/trial4/tdoa3 |
|---|---:|---:|
| Unchanged author class | 6.369 | 17.160 |
| Actual production proposed class | 6.150 | 17.684 |
| Production with only complete observation Jacobian enabled | 173.328 | 170.479 |
| Standalone research kernel | 175.740 | 172.757 |
| Author mirror, only complete update Jacobian substituted | 173.326 | 170.577 |
| Author mirror, only physical covariance propagation substituted | 6.367 | 17.158 |

Single-step propagation means agree to floating-point precision. The discrepancy starts at correction. In the explicitly tested world error chart, the complete Jacobian matches central finite differences below 7.81e-10. This local derivative agreement does not establish better finite-error convergence: replacing this Jacobian alone reproduces the large-heading failure in these two cases. Other one-factor ablations—prior refresh, gate consistency, updated bias, ground clamp, covariance propagation—do not account for its roughly 160-degree effect. Changing radio variance from the main script's .13 to the class constructor's .05 is not uniformly helpful: one development condition reaches 359.93 m XY RMSE. It remains in the results.

`audit/`, `diagnostic/`, `ablation/`, and `mechanism/` contain raw differences, exact variant settings and all per-case results. `mechanism/heading_mechanism.png` plots the development trajectories.

## Metrics, initialization and native queue

The source paper-style orientation calculation uses differences of SO(3)-log components, with smoothed ground-truth components. It is not physical wrapped Euler yaw, nor a relative-rotation geodesic. `mechanism/paper_style_metric_comparison.csv` reports all three on the same diagnostic grid. This is a metric comparison, not a reproduction of the paper's full-data numerical table.

Recovery initialization is sensor-only: first .2 s IMU gravity alignment and zero yaw; .6 s native radio constant-velocity fit for antenna position/velocity, translated by the known Toronto lever arm. A stationary gyro bias estimate is used only when the frozen IMU stationary check passes. Initialization does not use GT position, yaw or alignment. The original numeric invariant covariance is used without remapping: standard deviations `[.01,.01,1,.01,.01,.01,.1,.1,.1,.0001,.0001,.0001,.000001,.000001,.000001]`. Author nominal bias ordering and covariance/noise quirks are preserved.

The original main-style queue uses the union of raw IMU/radio times and only the first radio row per duplicate timestamp. The original prediction/correction order, stale correction prior, ground clamp and sigma gate are retained. Outages mask radio corrections, including their use by candidates, while retaining the same event clocks. Wrapper changes are sensor-only initialization, controlled radio masking, shared full-object branch snapshots and output logging. The source author's GT-initialized main is not called a sensor-only experiment. Variance is the original main's .13 m²; candidate standard deviation is its square root, .360555 m. Toronto's original lever arm is retained.

All scoring is causal zero-order hold at 20 Hz, with no truth alignment, over half-open windows `[25,27)`, `[27,30)`, `[31,36)`. Coverage and wrapped yaw are included. Failure is retained for exceptions/nonfinite states or position above 1000 m; errors below that threshold are not silently excluded.

## Balanced independent trial-2 cohort

Eight prelisted recordings cover const1–4 and native TDoA2/3; const4 uses traj1. All four outage lengths 2/5/10/20 s end at 25 s, with nominal heading. Actual source K generation took 89.8 s wall, with at most six workers and one BLAS thread per worker. CPU scheduling prefers separate physical performance cores. All 32 source trajectories completed with full scoring coverage.

The frozen candidates are output-only: `always_C`, prediction-guarded `guarded`, and `radio_bounds`. They may replace XY only in the first two seconds and return exactly to the continuing K recursion at 27 s. Their yaw and height are exactly K. They are not state-feedback recovery algorithms.

| Outage (s) | K early XY median (m) | always_C | guarded | radio_bounds |
|---|---:|---:|---:|---:|
| 2 | .0683 | .0837 | .0662 | .0725 |
| 5 | .2107 | .2099 | .2116 | .2146 |
| 10 | 4.4492 | 2.8234 | 4.4490 | 4.4659 |
| 20 | 51.7228 | 18.0337 | 35.9891 | 44.8372 |

Guarded improves/worsens 2/4, 2/4, 4/2, 6/0 of eight cases for these four outage lengths; remaining cases tie. Bounds worsens two short-outage cases, by up to .5993 m early RMSE. The largest added output step is 107.50 m for guarded/always_C and 117.38 m for bounds. All middle/later windows are exactly K by construction; the apparent early benefit is not lasting filter recovery. `independent/paired.csv`, `overview.json`, `integrity.json` and the two PNGs preserve all cases and damage.

The scorer emits a common explicit 27 s K-held sample for every method. This fixes a reporting-only ZOH boundary leakage from the last pre-27 candidate sample; raw K/candidate caches and candidate algorithms remain unchanged. `score_provenance.json` records it. `score_independent.py`, direct `independent.score()` and `independent.py --stage score` all use this corrected scorer.

## Supplementary four const4 recordings

The audit found four more trial-2 recordings: TDoA2/3 × traj2/3. They were named and timestamp-checked before their scores were read; no file substitution was made. Their output-only results are separate in `feedback_extension/paired.csv`. At 20 s, early XY medians are K 70.576 m, always_C 23.780 m, guarded 48.601 m and bounds 51.137 m. Later outputs are exactly K.

A separately registered, prospective supplementary arm uses a single actual state translation. The first arriving packet in `[25,27)` whose frozen single-packet bound is violated at the author's current post-correction mean supplies an XY displacement. The central `feedback.single_offset` and `translate_original` helpers apply it once. Position is translated; instantaneous R/v/Z/bias and physical additive covariance remain unchanged. The invariant covariance is rebased by `A[p,theta]=skew(d)` without shrinking it, and X/Xhat/P/Phat are synchronized only for a nonzero shift. Subsequent original recursion continues with no 27 s forced return.

All 16 supplementary conditions were generated before scoring, with four parallel recording jobs (BLAS1), in 90.3 s. Three trigger; thirteen nontriggers are bitwise identical to K. Maximum instantaneous physical covariance difference is 1.42e-14; instantaneous R/v/Z/bias differences are zero. No numerical failures occurred; all three score windows have 100% coverage.

| Triggered condition | Three-window XY K → feedback (m) | Later yaw K → feedback (deg) |
|---|---|---:|
| tdoa2/traj2, 20 s | 84.295/104.746/128.411 → 4.960/.078/.105 | 11.422 → 1.958 |
| tdoa3/traj2, 10 s | 11.112/31.082/72.565 → 1.288/.075/.087 | 12.293 → .938 |
| tdoa3/traj3, 20 s | 93.231/112.032/135.533 → 64.462/84.125/109.269 | 3.871 → 7.223 |

The largest initial translation is 75.34 m and maximum added XY step is 77.18 m. One case damages later yaw by 3.35 degrees. These data support a limited position recovery effect in this supplementary cohort, not smooth recovery or a universal no-regret claim. All individual rows, preservation checks and every triggered trajectory are in `feedback_extension/feedback_scores/`. No feedback arm has been run on the already-scored original eight flights.

## Exact acceleration and reproduction

`fast_original.py` is an explicitly separate numerical mirror, never called the original class. Its finite nilpotent transition replaces dense `expm`. Full 45 s comparisons against the unchanged source on all four development conditions agree in rows below 3.72e-15, P below 1.88e-16, bias below 1.70e-19, with identical gates/failures. All 32 primary and 16 supplementary actual-source recovery K trajectories independently validate against this mirror. Primary worst row/P differences are 1.65e-10/6.63e-11. Warm mirror throughput is about .5–.6 s per 45 s flight. See `audit/fast_full_validation.json`; scientific test K still uses the source class.

Run from `E:/0syncthing/Code/InEKF` using this project's Python. Stages are intentionally separate:

```powershell
$py = './.venv/Scripts/python.exe'
$dir = 'imu-uwb-inekf/experiments/strong_recovery/oursland'
& $py "$dir/audit.py"
& $py "$dir/diagnose.py"
& $py "$dir/validate_fast.py" --full
& $py "$dir/ablate.py"
& $py "$dir/analyze_dev.py"
& $py "$dir/inventory.py"
& $py "$dir/independent.py" --stage k
& $py "$dir/independent.py" --stage candidates
& $py "$dir/independent.py" --stage score
& $py "$dir/analyze_independent.py"
& $py "$dir/feedback_runner.py" --stage freeze
& $py "$dir/feedback_runner.py" --stage run
& $py "$dir/extension_candidates.py" --stage candidates
& $py "$dir/extension_candidates.py" --stage score
& $py "$dir/feedback_runner.py" --stage score
& $py "$dir/analyze_extension.py"
```

Existing complete K caches are reused. `replay_recordings.ps1` contains one runnable command per recording; `replay_one.py NAME [--supplementary]` runs any one row without altering aggregate protocols or scores. Each row below generates the four fixed gap lengths; these are the exact recording identifiers:

| Balanced primary | Supplementary, separately reported |
|---|---|
| const1-trial2-tdoa2 | const4-trial2-tdoa2-traj2 |
| const1-trial2-tdoa3 | const4-trial2-tdoa2-traj3 |
| const2-trial2-tdoa2 | const4-trial2-tdoa3-traj2 |
| const2-trial2-tdoa3 | const4-trial2-tdoa3-traj3 |
| const3-trial2-tdoa2 | |
| const3-trial2-tdoa3 | |
| const4-trial2-tdoa2-traj1 | |
| const4-trial2-tdoa3-traj1 | |

Cached NPZ files include raw native observations, truth for later scoring, original full X/P/Xhat/Phat/bias split snapshots and final states. Source hashes and worker timing are recorded beside each stage. Shared candidate/helper modules and the backup author class are never modified here. Scoring-only dispatch and supplementary scoring-folder changes after registration do not change any sensor-generation code or helper thresholds.
