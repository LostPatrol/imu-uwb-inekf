# Real UTIL split handoff

`protocol.json` freezes trial3 for development and previously unscored trials4/5 for holdout, four surveyed geometries, TDoA2/3, and initial world-yaw offsets 0/90/-90/180 degrees. Every recording runs for 45 s. Fixed handoffs occur at 2 s (cold), 16 s (dynamic window 10–16 s), and 26 s (actual radio outage 10–20 s and recovery window 20–26 s). Actual IMU ticks determine the final endpoints. No truth initializes, aligns, selects, fits, or decides; truth enters `score_split` after sensor-only generation for the complete requested split.

## Observation and information accounting

The antenna position model is

\[
s_i=p_e+\tau_i v_e+C\{f_i-B_i\delta b_a\}+\tfrac12g\tau_i^2,
\qquad y_i=\|s_i-a_i^+\|-\|s_i-a_i^-\|+\epsilon_i.
\]

Here \(C\) maps the window-start body frame to world, \(f_i\) is endpoint-reanchored double-integrated local specific force **including the rotating calibrated antenna lever arm**, and \(B_i\) is the matching double integral of local rotation. World gravity is added separately. IMU biases used for this integration are frozen at the window start. The common conditional reference is \(C_{ref}=R_{normal,e}Q_e^T\); it contains normal tracking corrections through the handoff. Consequently prefix/prediction diagnostics are conditional on this common terminal reference; this experiment does not establish that the complete heading estimator is an online causal decision.

Outage packets are physically deleted before any radio-only fit. Candidate paths use no packet later than the actual handoff. Independent radio-only antenna position/CV velocity outputs use 0.2 s bins with a trailing 0.6 s fit, at least ten available packets and three new packets in the current bin. First output is normally 0.2 s; initial filter state waits until 0.6 s. Recovery first output is normally 20.2 s. These are distinct from terminal reconstruction and from the first saved 20 Hz filter trajectory row. The terminal shared p/v is fitted from the last 0.6 s. All actual times are retained per case.

Candidate observations are thinned by directed-pair time spacing of 0.1 s and capped at 600 using time-only subsampling. Each candidate sees the same packet budget. Full attitude uses eight fixed yaw starts with the same raw hyperbolic model and bias/tilt priors. Evidence and bank functions are separately frozen public source modules. `forced_evidence_yaw` applies the evidence candidate regardless of its gate; `forced_yaw_only` applies only the heading of the full-pose candidate.

The main panel is explicitly a **p/v reset** ablation. Every strict policy receives identical body position, velocity, biases and physical covariance; only orientation changes. The position lever correction uses the same initial nominal tilt for every counterfactual. Normal shadow, full pose, evidence and bank are alternative estimates conditioned on the same radio window. Their radio likelihoods are never multiplied together, and no window packet is replayed as an independent update after handoff. A common broad physical covariance is remapped to each invariant coordinate frame; it is a controlled reconstruction prior, not a claim of exact posterior independence. `normal_unmodified` is the strong normally updating no-reset trajectory; its original p/v/covariance distinguish it from the matched strict ablation.

`position_only.py` additionally reuses frozen holdout candidates for 0/180 degrees and retains the normal shadow terminal velocity. Only common p is reconstructed; the three matched keep/full/evidence policies share p/v/bias/physical covariance. This panel requires no new fit or heading decision.

## Reproduce

From `E:/0syncthing/Code/InEKF` in PowerShell:

```powershell
$env:OPENBLAS_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
.venv/Scripts/python.exe imu-uwb-inekf/experiments/split_handoff/util/verify_kernel.py
.venv/Scripts/python.exe imu-uwb-inekf/experiments/split_handoff/util/verify_causality.py
.venv/Scripts/python.exe imu-uwb-inekf/experiments/split_handoff/util/run.py --split development --output development_final
.venv/Scripts/python.exe imu-uwb-inekf/experiments/split_handoff/util/run.py --split holdout --output holdout_final
.venv/Scripts/python.exe imu-uwb-inekf/experiments/split_handoff/util/analyze.py development_final
.venv/Scripts/python.exe imu-uwb-inekf/experiments/split_handoff/util/analyze.py holdout_final
.venv/Scripts/python.exe imu-uwb-inekf/experiments/split_handoff/util/position_only.py
```

Use `--resume` to resume only complete per-case checkpoints. BLAS thread limits prevent small repeated least-squares problems from oversubscribing the CPU; they do not change data, estimator rules, or packet budgets.

The primary metric is unaligned surveyed-world post-handoff XY trajectory RMSE on the identical future segment for each policy. Z/yaw/SO(3), whole trajectory on a common 20 Hz clock, actual arrival/retained/handed-off heading errors, failures, and first-output availability are retained. Injection at initialization is not assumed to survive normal tracking; the actual errors at the window boundaries are scored explicitly. Four yaw counterfactuals from one log are not independent recordings; `recording_units.csv` averages them before recording-level comparison.

`metrics.csv` preserves available-segment scores even when a trajectory fails. Nonfinite states or positions beyond 1000 m terminate replay and are explicitly counted as failures. **Use `complete_period_metrics.csv` and `complete_period_summary.csv` for full-period RMSE**: incomplete scores are missing there, never silently compared with complete trajectories. `complete_period_paired.csv` compares matching complete clocks and states the requested, complete-pair and failure counts. `failure_examples.csv` preserves the failed partial results. Success-only RMSE must always be accompanied by the failure rate.

`verification/causality.json` poisons all masked and post-handoff real packets by +1e6, requiring exact equality of shared state/P/bias, path, all candidate inputs, full fitted rotation and both decision diagnostics (wall-clock profiling excluded). The compiled kernel is independently checked against existing invariant source by `verify_kernel.py`. `audit.json` checks shared physical handoffs, unchanged tilt for yaw-only policies, and exact retain trajectories on rejection.

The older `smoke/`, `development/`, and `holdout/` directories are **invalid pre-causality-fix partial/interface runs**, retained for provenance only. Both partial main runs were stopped before truth scoring. Final evaluation uses only `development_final/`, `holdout_final/`, and `holdout_position_only/`. No synthetic result is used as real-flight evidence.

## Frozen real-flight results

All 672 development and 1344 holdout trajectories completed generation, plus 288 cached-candidate position-only counterfactuals. There are 8 development and 16 holdout recordings, not 2304 independent flights. The following holdout values are median full-period XY RMSE in metres; failure counts are retained separately.

| Policy | Cold | Dynamic | Outage recovery | Recovery failures |
|---|---:|---:|---:|---:|
| Normal, no reset | 0.07129 | 0.06714 | 13.36400 | 8/64 |
| Common p/v, retain normal R | 0.07372 | 0.06759 | 0.06647 | 0/64 |
| Common p/v, full attitude | 0.07262 | 0.06752 | 0.06632 | 0/64 |
| Common p/v, evidence yaw | 0.07372 | 0.06759 | 0.06647 | 0/64 |
| Common p/v, heading bank | 0.07372 | 0.06759 | 0.06647 | 0/64 |

The normal outage median uses its 56 complete trajectories; on the same 56 cases the p/v-retain median is 0.06011 m. The remaining eight normal trajectories failed and do not receive fabricated full-period scores. The normal filter is slightly better in ordinary cold/dynamic operation; reconstruction is not a universal improvement.

The evidence gate accepts 0/192 holdout windows, hence exactly reproduces retain. Its unconditional candidate improves XY in 89 counterfactuals and worsens it in 103; the paired median difference is +0.214 mm. Full attitude improves 88 and worsens 104, paired median +0.151 mm. These are repeated counterfactuals from 16 recordings; recording-level paired means are also retained in `heading_recording_paired.csv`. Bank accepts 3/192, with one XY improvement and two regressions (worst +11.74 mm). This is no demonstrated general heading benefit. The gate's 2/96 development accepts also both worsen future XY.

Literal position-only recovery is weaker: for the predetermined 0/180-degree panel, retain/evidence have 31/32 complete recoveries, median 0.12856 m, and one failure; full attitude has four failures. Its paired median XY change on 28 common complete cases is +0.143 mm. The source p/v-retain panel completes all 32 with median 0.06780 m. Recovery gains therefore cannot be attributed to position alone or to solving heading; correcting velocity drift matters materially.

This is a position-mean-only ablation under the same fixed physical reconstruction covariance as the p/v panel. It does not rule out another recovery design that preserves a larger, correctly correlated velocity covariance from the shadow filter.

The independent first 0.2 s positions have holdout median XY error 0.08044 m and maximum 0.58337 m across the 16 recordings, compared with median 0.08282 m / maximum 0.65194 m at the native 0.6 s initialization. This is an availability result with explicit real errors, not a claim that every short prefix determines heading. Actual retained endpoint yaw errors have cold/dynamic/recovery medians 77.79/17.96/32.81 degrees, despite the four nominal initialization perturbations. Full Z/yaw metrics, paired examples, failures and coverage remain in the CSV artifacts.
