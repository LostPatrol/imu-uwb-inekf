This is a **post-hoc exploratory** follow-up after the six-flight Delft output
overlay scores were already known. It neither amends their frozen policies nor
provides an independent confirmatory test. All six late recordings and all
four fixed outage durations are retained: 24 new primary-queue trajectories,
paired with the existing native K cache. No new heading injection, calibration,
threshold, recording exclusion or truth alignment is introduced.

The native original class predicts and corrects each offered packet. For the
first newly arrived packet in `[25,27)` whose age-zero single-sided geometric
bound excludes the current native post-update horizontal mean, shared
`strong_recovery/feedback.py` supplies one horizontal displacement `d`. Its
`single_offset` uses exactly the frozen `radio_bounds.py`, original `.13 m^2`
radio variance, and zero uncalibrated Delft lever. No additional packet average,
validation window or future observation enters this operation.

The single reset is

`p' = p+d, R'=R, z'=z, v'=v, b'=b`,

`P' = A P A^T, A=I, A[p,theta]=[d]_x`.

The author's `X_hat/P_hat` snapshots synchronize only on this actual reset.
The physical error map has blocks `J[v,theta]=-[v]_x` and
`J[p,theta]=-[p]_x`, so `J(p+d) A = J(p)` and full physical covariance is
preserved. This is a change of invariant chart, not covariance shrinkage or an
independent Gaussian pseudo-measurement. The packet has already been offered
to native correction; the bound is an additional nonprobabilistic constraint
on that same information budget. Every subsequent prediction/correction is
the original recurrence, without another projection or a return-to-K policy.
Later attitude, height and velocity may differ through that recurrence.

All trajectories completed with full scoring coverage. Seven cases triggered
and seventeen remained exactly K; the two flights with no trigger at any
duration were `14_06_55_hourglass` and `14_15_41_random`. Before every actual
trigger, states also match K exactly. Trigger shifts span `.0931–75.6883 m`.
Instantaneous attitude/height/velocity/bias changes are exactly zero; physical
covariance's maximum absolute/relative discrepancies are `9.09e-13` and
`2.09e-16`. Prediction snapshots synchronize exactly. These audits are in
`results/audit.json` and `results/triggers.csv`.

With the same causal 100 Hz ZOH scoring, wins/ties/losses over all 24 matched
cases are:

| After resumption | Wins | Exact ties | Losses | Worst XY RMSE regression |
|---|---:|---:|---:|---:|
| 0–2 s | 5 | 17 | 2 | .00632 m |
| 2–5 s | 5 | 17 | 2 | .02414 m |
| 6–11 s | 2 | 17 | 5 | 19.18338 m |

The octagon 20 s outage demonstrates a real sustained repair in this case:
native K XY RMSE `73.993 / 62.388 / 62.780 m` becomes
`7.074 / .222 / .171 m` across the three periods. Conversely, star 20 s outage
initially improves but later worsens `79.129 -> 98.312 m`; BAD hourglass 10 s
outage worsens late `43.084 -> 54.914 m`; random 10 s outage worsens late
`79.068 -> 80.078 m`. The short-outage BAD cases worsen slightly throughout.
Thus a single bound can enable native recapture, but one constraint does not
identify full position, velocity, attitude or biases, and early error reduction
does not establish general persistent state repair. There is no valid Delft
heading truth, so no heading accuracy is reported.

Every outcome, including nontriggers and all late regressions, remains in
`results/metrics.csv`, `paired_xy.csv`, `outcomes_by_duration.csv`, and
`paired_differences.csv`. `trigger_truth_diagnostics.csv` is scoring-only and
does not affect triggering. Illustrations deliberately show one sustained
success and two late failures, selected after scoring for explanation; they
are not additional validation cohorts. Four outages within a flight are
repeated counterfactuals, not 24 independent flights.

Run from `E:/0syncthing/Code/InEKF` with the project environment:

```powershell
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
./.venv/Scripts/python.exe imu-uwb-inekf/experiments/strong_recovery/delft_tdoa/posthoc_state_projection/run.py
./.venv/Scripts/python.exe imu-uwb-inekf/experiments/strong_recovery/delft_tdoa/posthoc_state_projection/analyze.py
./.venv/Scripts/python.exe imu-uwb-inekf/experiments/strong_recovery/delft_tdoa/posthoc_state_projection/examples.py
```

The runner resumes existing per-case caches; `--score-only` only rescans truth
and saved trajectories. Source hashes are retained in `results/provenance.json`.
