# Single InEKF with conditional window replacement

This is an exploratory closed-loop implementation, with a frozen two-second
acquisition rule. It tests whether the manuscript's window candidate, extended
with the current body-frame accelerometer-bias seed, helps a full-state
conditional update. It does not certify stable superiority or calibrated
covariance. See `results/paired_changes.csv` for failures to improve as well as
benefits, and `results/solver_diagnostics.csv` for convergence and computation.

## Posterior and packet accounting

The online state, IMU step, sample-noise variances, measurement geometry and
frozen anchor calibration are the existing `roadmap_stage1/Invariant15` model.
The checkpoint is the last completed radio posterior before a sensor-only gap
exceeding one second. At startup it is the original centroid/static-prefix
prior, before the first range. The complete checkpoint Gaussian is retained.

The replacement represents the same interval distribution

\[
p(x_{c:T}\mid\mathcal D_{\le c},u_{c:T},z_{c:T})\propto
p(x_c\mid\mathcal D_{\le c})
\prod_{k=c}^{T-1}p(x_{k+1}\mid x_k,u_k)
\prod_{i:c<t_i\le T}p(z_i\mid x_{t_i}).
\]

The startup product also includes packets at its initial timestamp because its
checkpoint precedes those packets. A recovered checkpoint already includes all
its own packets, so those packets are excluded. Packet identities are checked
for uniqueness. No fitted endpoint is fused as an additional measurement.

`smoother.py` implements four IEKS passes. Each pass starts again from the same
original checkpoint mean and full covariance. Physical right-attitude/additive
translation error coordinates reparameterize the same InEKF state; the exact
native discrete dynamics Jacobian and noise map are imported directly. A
forward linear conditional filter and RTS backward smoother jointly update all
15 components, including both IMU biases and every covariance cross block.
The final endpoint covariance is reset and mapped into invariant coordinates.
It is a local linearization/IRLS covariance, not an exact nonlinear posterior.

The batch uses Huber raw-range residuals at the original calibrated scale,
with threshold 2. The online native baseline uses its existing NIS gate and
Huber policy. Thus native versus IEKS tests the whole conditional-update policy;
IEKS Local versus IEKS Profile isolates reference initialization under the
identical conditional solver. A finite four-pass result can remain in a wrong
basin; terminal corrections are reported rather than called convergence.

## Reference trajectory and real handoff

The Profile arm first calls the manuscript's `initializer.solve_window`, then
the current `joint_branch.study.solve` with `BiasPrior` and the static radio
model. The existing 0.10 m/s² bias prior only helps construct this numerical
seed; it is not appended to the conditional posterior. The actual conditional
prior remains the unmodified checkpoint Gaussian, including its learned bias
uncertainty and cross terms. Profile yaw is a world-vertical **increment to the
checkpoint-propagated attitude**, not a separately aligned absolute yaw.
The seed changes p/v/yaw and accelerometer bias; gyro bias retains its checkpoint
seed. IEKS subsequently optimizes all 15 components.

At startup and radio return the filter continues its own ordinary native
emissions while collecting two seconds of past packets. At least sixteen
positive packets from at least four anchors are required; otherwise that
acquisition is recorded as unsupported. The seed uses only those post-return
packets, while the conditional solver also includes the checkpoint-to-return
IMU interval. At handoff the old interval posterior is discarded and replaced
by the new endpoint X, complete P and both biases. All subsequent native
propagations and updates start from this endpoint. There is no output splice,
forced return to the baseline, or covariance from a separate window fit.

`profile_startup_native_recovery` repeats exactly the same Profile startup and
native prehistory, then disables only the recovery replacement. Per-case audits
require bitwise equality of X, bias and P diagonals before T=27 s, including
the whole first-two-second recovery phase. Its later paired difference is the
causal recovery-handoff effect. Differences against `native` before T=27 can
instead come from the earlier startup replacement.

This is a causal **sensor-time offline replay**. The two-second waiting interval
is charged to output scoring; actual per-solve wall times are saved separately.
New packets arriving during computation are not simulated as an asynchronous
queue. The results therefore establish neither real-time deadlines nor a total
latency of exactly two seconds. No retrospectively smoothed past state replaces
an already emitted/scored sample.

## Data, comparisons and limits

The matrix has six existing Delft development flights and all ten previously
verified sensor-eligible QDrone recordings, each with 2/5/10/20 s outages ending
at T=25 s. Existing exclusions and missing truth remain visible. Delft's clock
starts after the original 1.5 s static prefix and preserves the existing common
0.2 s sensor delay correction. QDrone retains its released recording clock and
the original event-timeline IMU subdivisions; replay begins after its prefix.
No new truth alignment, lever calibration, radio training or range correction
is fitted. QDrone still inherits the existing zero-lever/raw-axis approximation
and provider-interpolated prism truth limitations.

The historical `old_lifted` arm loads the intact original native capture/tracker
trajectories. Delft uses the existing `strong_twr/length_cache`; QDrone uses
the verified-cohort saved K arrays. All new arms share its initial centroid,
static tilt/bias and physical covariance for the corresponding dataset. No old
trajectory is modified. The present score is XY RMSE on a uniform 20 Hz grid,
using the last already emitted estimate. A summary median is the median of
**per-recording window RMSEs**, not a pooled RMS or a mean of instantaneous
errors. This differs from manuscript all-output 3D maxima/full-run RMSE and
from prior event-row-weighted tables; such historical numbers are not directly
interchangeable. Full sample trajectories are retained to audit the difference.

Phase windows relative to radio return are [0,2), [2,5), and [6,11). Each phase
requires 95% supported finite samples for a primary paired score. Unsupported
records and numerical failures are retained explicitly and counted separately;
no support-qualified summary should be presented as all ten QDrone flights.
These caches provide position truth only. Physical yaw error cannot be measured
on these replays; saved attitude trajectories and handoff-angle changes are
estimator diagnostics, not yaw accuracy evidence.

## Run and inspect

Use the project's `.venv` Python from the workspace root, with BLAS thread counts
set to one. All generated files stay in this directory.

For a fresh replay, prepare Delft caches with `split_handoff/delft/study.py --prepare-only`,
then `strong_recovery/twr/replay.py --workers 6` to generate the historical lifted states.
For QDrone use `strong_recovery/external_twr/qdrone_run.py --prepare-only` and then
`--run --workers 6` on the verified public release (default primary clock).
The replay also needs the established roadmap training/holdout protocol; do not
replace it with another calibration when reproducing the paired numbers.

```powershell
./.venv/Scripts/python.exe imu-uwb-inekf/experiments/unified_window_recovery/verify.py
./.venv/Scripts/python.exe imu-uwb-inekf/experiments/unified_window_recovery/run.py --workers 6
./.venv/Scripts/python.exe imu-uwb-inekf/experiments/unified_window_recovery/run.py --ablation --workers 4
./.venv/Scripts/python.exe imu-uwb-inekf/experiments/unified_window_recovery/analyze.py
./.venv/Scripts/python.exe imu-uwb-inekf/experiments/unified_window_recovery/phase_sensitivity.py
```

`verification.json` checks native F by finite differences and a scalar Gaussian
case: four relinearizations of one measurement still produce variance 0.5,
not the variance 0.2 obtained by treating that measurement as four independent
observations. A preliminary representative run before correcting the prior
chart Jacobian is segregated in `pilot_before_prior_jacobian_fix/` and excluded
from every final summary. There was no parameter search after physical scores.
The same-history ablation and sampling-phase diagnostics were added after
preliminary/full results at the parent task's request; the original primary
arms, numerical policy and primary scoring grid were retained.
