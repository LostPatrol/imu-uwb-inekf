# Real-flight TDoA nuisance-state study

This isolated research module compares native hyperbolic TDoA observations with
pair-specific temporally correlated radio errors on raw UTIAS/Bitcraze flight
CSV files. It does not modify the production estimator. The present result does
**not** establish a consistent accuracy improvement over a strong native baseline.

## Model

For the surveyed ordered anchor pair `(a,b)` and antenna lever arm `l`,

\[
z_k=h_k(X_k)+b_{ab,k}+e_k,\quad
h_k=\|p_k+R_kl-a_b\|-\|p_k+R_kl-a_a\|.
\]

`pair_ou` uses independent pair Gauss-Markov states:

\[
b_{ab,k+1}=\phi b_{ab,k}+w_k,\quad \phi=e^{-\Delta t/\tau},\quad
Q_b=\sigma_b^2(1-\phi^2),\quad P_{b,0}=\sigma_b^2.
\]

Its raw residual covariance is
\(K_{ij}=\sigma_b^2 e^{-|t_i-t_j|/\tau}+r\delta_{ij}\) within a pair.
Thus augmentation performs correlated-likelihood filtering without feeding
the same radio observation into both a raw and a differenced update.
A temporal difference would produce
\(z_k-\phi z_{k-1}=h_k-\phi h_{k-1}+w_k+e_k-\phi e_{k-1}\);
its white-noise part has variance \(r(1+\phi^2)\) and consecutive covariance
\(-\phi r\). Ignoring that covariance would double-count information.

All modes retain a right-invariant navigation error and full lever-arm Jacobian.
For `tag=p+lag*v+R*l`, the observation row is
\([-g^T[tag]_\times,\;lag\,g^T,\;g^T,\;0,\;0]\),
where \(g=(tag-a_b)/\rho_b-(tag-a_a)/\rho_a\).
The navigation propagation reuses the repository's exact discrete physical
Jacobian, transformed into invariant coordinates; injection and reset use SE_2(3).

## Matched information budget

- All methods use identical surveyed anchors, raw IMU/TDoA, lever arm, first
  two seconds of radio constant-velocity initialization, initial covariance,
  zero initial IMU bias, Huber threshold 2 and NIS rejection threshold 25.
- Initial tilt comes from mean accelerometer specific force; yaw starts at zero
  with standard deviation pi. This is **not** a general nonstatic attitude
  initializer: motion during the prefix can bias gravity alignment, and global
  yaw acquisition is not guaranteed. No truth initialization or trajectory
  alignment is applied.
- IMU is processed at a fixed 100 Hz with midpoint input. Radio is processed
  in arrival order with up to 10 ms latency and first-order position time
  compensation. OU covariance transitions occur at these processing ticks.
  Results characterize this common research scheduler, not the original 1 kHz
  production replay. IMU interpolation uses neighboring raw samples offline.
- Native variance is 0.05 m^2, or 0.13 m^2 for const4, matching repository
  defaults. `white_inflated` adds 0.09 m^2, matching the OU model's stationary
  marginal radio variance. `pair_ou` uses tau=2 s and sigma=0.3 m throughout.
- These exploratory parameters were fixed before the trial2 runs. Truth is
  used only for residual diagnostics and final scoring. Diagnostic residuals
  are not fed into the initializer or any filter. Trial2 results are a small
  development evaluation, not a preregistered blind benchmark; do not tune
  against them and then call them held out.
- TDoA2 uses the documented eight directed ring pairs, matching the repository
  native loader. TDoA3 allocates all 56 directed pairs before replay. No future
  observations determine pair inventory. Pair-static is an exploratory ablation.

## Real-data findings

`results_validation` contains six distinct raw recordings, four anchor layouts,
and both radio protocols. All methods evaluate the same body trajectory from
2 to 60 seconds. RMSE uses the surveyed world frame without an alignment fit;
attitude is also scored by SO(3) geodesic angle. `tail_*` metrics separately
score times after 10 s; they are not the principal metrics.

| Recording (trial2) | Native pos/yaw | White pos/yaw | Pair OU pos/yaw |
|---|---:|---:|---:|
| const1 TDoA2 | 0.1169 m / 29.43 deg | 0.1178 / 12.06 | 0.1181 / 10.03 |
| const2 TDoA2 | 0.1050 / 26.21 | 0.1043 / 12.25 | 0.1065 / 13.37 |
| const3 TDoA2 | 0.2277 / 34.46 | 0.2104 / 26.69 | 0.2174 / 43.64 |
| const4 traj1 TDoA2 | 0.2207 / 36.96 | 0.2190 / 21.59 | 0.2200 / 23.77 |
| const1 TDoA3 | 0.1137 / 8.14 | 0.1130 / 13.24 | 0.1027 / 14.61 |
| const3 TDoA3 | 0.1762 / 32.73 | 0.1674 / 23.60 | 0.1725 / 40.46 |

Across recordings, median position/yaw RMSE is native 0.1465 m / 31.08 deg,
white 0.1426 m / 17.42 deg, and pair OU 0.1453 m / 19.19 deg. Pair OU does
not beat the marginal-white control overall, and substantially worsens const3
yaw. A large part of the apparent gain over native is explained by observation
weighting. Static pair offsets can absorb position error and performed worse
in the exploratory runs. No general precision breakthrough is demonstrated.

Truth-only residual diagnostics show pair median offsets up to about 0.175 m,
and median lag-1 clipped-residual correlation 0.276-0.571 for TDoA2 (typical pair
spacing 16.6 ms). TDoA3 has both positive and negative >1 m tails; a one-sided
NLOS model on the **differenced** measurement would therefore be inappropriate.
The fixed two-second OU time constant is not an estimate of these correlations.
A subsequent study should estimate correlation/offset priors on separate
training recordings and give every native control exactly the same calibration.

Runtime is about 2-4 s per 60 s native/white/TDoA2-OU replay and 7-9 s for
56-state-pair TDoA3-OU on this machine. File loading and diagnostics are extra.
`results_first` is an earlier 30 s exploratory run before matching the ring-pair
inventory; use `results_validation` for the reproducible final comparison.

## Reproduce (PowerShell, project root)

```powershell
& .venv/Scripts/python.exe imu-uwb-inekf/experiments/radio_inertial_models/tdoa_branch/study.py --duration 60 --modes native white_inflated pair_ou --output results_validation --names const1-trial2-tdoa2 const2-trial2-tdoa2 const3-trial2-tdoa2 const4-trial2-tdoa2-traj1 const1-trial2-tdoa3 const3-trial2-tdoa3
& .venv/Scripts/python.exe imu-uwb-inekf/experiments/radio_inertial_models/tdoa_branch/check_and_plot.py
```

Core checks: complete asynchronous lever-arm invariant Jacobian agrees with
central differences to 5.6e-10; sequential OU likelihood agrees with the dense
correlated Gaussian likelihood to 1.8e-15. Saved NPZ files contain timestamp,
body position and XYZW quaternion; `metrics.csv` and `diagnostics.csv` contain
unaggregated metrics and pair statistics. `comparison.png` shows all six logs.
