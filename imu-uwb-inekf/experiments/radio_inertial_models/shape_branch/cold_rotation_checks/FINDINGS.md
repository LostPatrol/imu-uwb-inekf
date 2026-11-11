# Cold full-attitude findings

The radio-path/Wahba seed supports full initial-attitude acquisition in the tested richly excited moving windows, including initially upside-down bodies. It matches the 24-rotation reference on the descriptive acquisition count while using one final raw-range refinement. Weak excitation and weak-height position ambiguity remain important limits.

The frozen panel has ten base random seeds, four motions, two anchor geometries, and four true initial tilts (0/60/120/179 degrees): 320 paired attitude conditions and 1,600 method estimates after adding the compensated-gravity baseline. Tilt variants share a world/radio trajectory and range noise; these are not 320 independent acquisitions. All windows lack a static prefix; every IMU interval has angular-rate norm at least 0.4457 rad/s. Bias and IMU white noise are zero. This is physical simulation, not real-data generalization.

Every final fit uses the same original 240 asynchronous ranges, free endpoint position/velocity and full SO(3). All seed policies use the same common radio-only path; that path is never added as a position factor. `cube24` spends 24 refinements and is not a global certificate. `local_gravity` averages gyro-compensated Q f, whereas `moving_gravity` is the weaker naive mean of moving body-frame force.

## Rich excitation

| Policy | SO(3) error <=10 degrees / 80 | Nonconverged |
|---|---:|---:|
| Single identity | 38 | 9 |
| Body-mean gravity | 55 | 0 |
| Gyro-compensated gravity | 42 | 0 |
| Radio path + Wahba | 80 | 0 |
| 24 cube rotations | 80 | 0 |

Radio/Wahba initial seed geodesic error is median 2.197 degrees / p95 5.584 degrees. Final full-SO(3) p95 is 2.1815 degrees, essentially the same as the cube reference. The rich-motion success count is complete at every tested tilt in both geometries.

Normal height-span geometry gives radio/Wahba position p95 0.312 m and rotation p95 2.389 degrees. Weak-height geometry still gives rotation p95 2.171 degrees, but position p95 is 4.518 m. Accurate attitude therefore does not establish accurate position. Against cube24, radio/Wahba has the same objective in all 40 normal-height conditions; in weak height it is worse once and better four times under the fixed objective tolerance. Neither method is a certified global optimizer.

The median number of final function evaluations across rich conditions is 6 for radio/Wahba versus 1,574 across all cube starts. This excludes shared CV/radio-path work; wall time under concurrent desktop load is not an embedded latency claim.

## Weak and degenerate motion

At 3% of the rich acceleration excitation, both radio/Wahba and cube24 acquire only 8/80 conditions. Their rotation p95 reaches approximately 171-172 degrees despite near-optimal range objectives. Better seeding cannot supply missing motion information. With constant world acceleration or constant velocity and zero lever arm, one full-attitude direction is exactly unobservable; chance closeness to truth is not evidence of identification. Normal-height position medians remain around 0.20 m while full rotations are largely undetermined.

The oracle-position Wahba check is confined to verification. It recovers arbitrary initial attitude in rich/weak cases with exact positions, while degenerate cases have a vanishing smallest rotation-information eigenvalue after p/v elimination. The analytic original-range right-tangent Jacobian differs from central differences by at most 7.11e-8.

An additional single-seed lever diagnostic shows that known lever [0.12,-0.05,0.08] m breaks the constant-velocity zero-lever ambiguity: smallest information eigenvalue changes from 0 to 0.2699, and exact-position Wahba recovers rotation. That remaining information is weak at this far-field noise level. It is a theoretical oracle-path diagnostic, not a measured acquisition result.

## Direct lift: theory and negative seed diagnostic only

The separate `theory_lift.py` removes a radio-only trajectory prerequisite from candidate generation. Let G=0.5 g tau^2. Squared-range expansion is linear in vec(C) after relaxing sixteen nuisance columns:

\[
N=[1,\tau,\tau^2,\tau^3,H,\tau H,-a,-\tau a],\qquad
M_i=(G_i-a_i)^\top\otimes H_i^\top.
\]

If the nuisance-projected nine-column M has rank nine, noiseless C is recovered by linear least squares, followed by a proper SVD rotation projection. A previously completed single-seed 179-degree-tilt rich example recovers within 5.1e-12 degrees without noise, but its seed error becomes 95.43 degrees with 0.10 m range noise. The weak example is similarly noise-sensitive; constant velocity has projected rank only three. This relaxation is not established as a practical candidate. No direct-lift benchmark panel or nonlinear refinement was completed or is claimed; the parent requested retaining only this theoretical/single-seed diagnostic.

## Deliverables and audit

`results/metrics.csv`, `summary.csv`, `aggregate.csv`, `paired_vs_cube24.csv`, `protocol.json`, and `local_gravity_supplement.json` retain all outcomes. `base_metrics.csv` preserves the original four-policy panel before the fifth baseline was added. `cold_rotation.png/.svg` show acquisition and observability. Two preset conditions were replayed with current shared solver source: ten method estimates, maximum non-timing numeric difference 1.82e-12. All 1,600 saved method estimates have finite numeric fields. Reproduction commands are in `README.md`.
