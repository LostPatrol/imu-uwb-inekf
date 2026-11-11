# Quadratic-output observer comparison

Reproduces the navigation example in the local `2006.02688v2.pdf` (Theodosis,
Berkane and Dimarogonas, Section 6), then compares conditional position/velocity
modules on the six fixed late Delft TWR recordings. The source example is a
simulation with known inertial acceleration, not a flight experiment.

## Model and source fidelity

The actual immersion is nine dimensional:

$$
Z=[q_0,q_1,q_2,p^T,v^T]^T,
\quad q_0=\tfrac12p^Tp,\ q_1=p^Tv,\ q_2=v^Tv,
$$
$$
\dot q_0=q_1,\quad\dot q_1=q_2+u^Tp,\quad
\dot q_2=2u^Tv,\quad\dot p=v,\quad\dot v=u.
$$

`observer.py` integrates the affine state exactly for each held world input.
`run.py` also integrates the original continuous Kalman-type observer using the
paper's analytic trajectory and input. No attitude/yaw state exists in this
observer. No evaluation MoCap position or attitude enters an estimator.

Section 6 prints `S0=100 I6`, `V=.0001 I6`, `W=I6`, which is inconsistent with
its nine-state scalar-output system. This reproduction explicitly uses
`M0=100 I9`, `V=.0001 I9`, `W=1`, and zero initial estimates because the paper
does not specify them. Therefore it reproduces the equations and trajectory,
not the exact unpublished initial conditions of the plotted curves.

The 20 s original interval is preserved, including the large initial transient
and nonzero endpoint error. `checks.py` additionally continues the same state
and covariance to 100 s to check asymptotic convergence; that extension is not
presented as the original paper's numerical result.

## Real-data panels and adaptations

Both modules start from the same sensor/radio-only all-eight-anchor Profile-1
at 2 s, using the exact same p/v mean and covariance. The source lift's initial
moments are the exact quadratic moments of that Gaussian p/v prior. World
acceleration comes from gyro-only integration, fixed sensor-derived bias and
the startup heading; it receives no further radio attitude corrections.

The fixed input-heading errors `0,+90,-90,180` start at 15 s. They describe
controlled errors in the *common input*, not yaw estimates. Both modules use
all actual changed positive transactions of the selected anchors, skip
`[15,25)` s, perform no later resets, and report identical fixed `31–36` s and
whole post-25 s windows. There is no NIS gate or Huber weighting in either
module. Training-recording truth calibrates radio bias/noise; evaluation truth
is read only after each replay for scoring and for the shared raw valid crop.

* `single0`: original single-absolute-range structure, translated from the
  origin to the preselected surveyed anchor ID 0; six-state range EKF uses
  exactly the same one anchor and world input.
* `all8`, `lower4`: explicitly **multi-anchor extensions** of the source method,
  using each asynchronous range as a separate quadratic observation. These are
  not results reported in the source paper.

For a calibrated Gaussian range with variance $\sigma_j^2$, the adaptation uses

$$
\tilde y_j=\tfrac12(\rho_j^2-\sigma_j^2)-\tfrac12a_j^Ta_j,
\quad H_j=[1,0,0,-a_j^T,0],
$$
$$
R_j=\|\hat p-a_j\|^2\sigma_j^2+\tfrac12\sigma_j^4.
$$

This removes the square's Gaussian bias and uses range-dependent variance;
the unknown true distance in the variance is approximated by the current
position estimate. Both modules share held acceleration sample variance
`4 I` and `1e-4 I6` per-second p/v Riccati forcing; the lifted process uses the
exact input Jacobian and additional `1e-4 I3` latent forcing. These discrete,
noisy-data choices are stated adaptations, not source-paper parameters proven
appropriate for Delft. There is no projection onto the quadratic manifold.

The six-state comparison controls known-input and early-prior differences. It
does not establish equivalence to a complete fifteen-state attitude/bias InEKF.
The latter can change its own input through radio-aided attitude updates.

## Running and outputs

From the project root, using its own environment:

```powershell
& '.venv/Scripts/python.exe' 'imu-uwb-inekf/experiments/split_handoff/quadratic_observer/run.py'
& '.venv/Scripts/python.exe' 'imu-uwb-inekf/experiments/split_handoff/quadratic_observer/checks.py'
```

The current Delft study must first have prepared its `results/cache`.
`results/protocol.json` fixes the complete comparison. `metrics.csv` retains
all 144 outputs (72 paired cases), `summary.csv` separates the four input
errors, and `aggregate.csv` aggregates across the 24 cases per geometry.
`histories/*.npz` retain p/v trajectories, latent states, common inputs,
scoring truth and starting priors. No finite performance failure is dropped.

`paper_example.png`, `paper_continuation.png`, `delft_panels.png` and
`delft_example.png` provide the comparison figures. `paper_checks.json` and
`validation.json` report both successful checks and the original 20 s
convergence threshold that was not attained.

## Current observed result

Fixed 31–36 s median RMSE across six late recordings and four controlled
input errors, in metres:

| Geometry | Quadratic XY | Range EKF XY | Quadratic Z | Range EKF Z |
|---|---:|---:|---:|---:|
| Single anchor ID0 | 28.139 | 9.866 | 18.392 | 2.308 |
| All8 extension | 0.114 | 0.111 | 0.428 | 0.280 |
| Lower4 extension | 0.988 | 0.152 | 25.046 | 0.815 |

All 144 fixed-window outputs are finite. Finite does not mean accurate: the
single-anchor and lower4 lifted estimates exhibit severe physical errors.
These outcomes show sensitivity under this predeclared real-data adaptation
and imperfect known-input approximation. They do not disprove the source
paper's convergence theorem for an exact known input and sufficient excitation.
All8 XY is similar; it gives no evidence of a general improvement over the
matched range EKF. Original 20 s simulation endpoint p/v errors are
`0.09570 m` / `0.01177 m/s`; the extra 100 s continuation reaches
`6.04e-6 m` / `7.96e-7 m/s` with the same parameters.
