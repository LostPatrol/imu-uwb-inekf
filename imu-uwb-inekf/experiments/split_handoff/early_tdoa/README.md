# Frozen early-XY requirements on native TDoA

The Delft `delft/early_position.py` engineering requirements are transferred to raw UTIL TDoA without adjusting thresholds on UTIL results. Trial 1, geometries 1–4, protocols 2/3 and yaw stresses 0/180 degrees are fixed before generation. Geometry 4 uses `traj1`. All eight files exist: 16 cases and 64 output traces are retained, with no missing cases or numerical failures.

All policies use sensor-only initialization at 0.6 s and the same native 15-state right-invariant filter to 10 s. The world-Z yaw stress is injected at 10 s, preserving physical covariance; this corresponds to Delft's stress at its 15 s blackout onset. Radio is then withheld during [10,20) s. Zero stress means no added rotation, not true heading initialization. Sensor-only generation finishes before truth scoring.

The three estimator policies are ordinary InEKF from the first returning packet, a fixed 2 s radio-only PV handoff at 22 s, and first-qualified XY handoff. A fourth output uses ordinary InEKF before the XY trigger and then switches to the independent reserved-window branch; it only switches outputs. No raw recovery packet enters a reserved branch before its handoff. Only strictly later packets are fused afterwards. The early branch receives no additional reset at 22 s.

## Native frontend and unchanged requirements

The CV antenna trajectory and native measurement are

\[
q_i=p_e+\tau_i v_e,\quad
h_i=\|q_i-a_{b_i}\|-\|q_i-a_{a_i}\|,\quad
g_i=\frac{q_i-a_{b_i}}{\|q_i-a_{b_i}\|}-\frac{q_i-a_{a_i}}{\|q_i-a_{a_i}\|},
\quad J_i=[g_i^T,\tau_i g_i^T]/\sigma_i.
\]

The frontend uses native nominal standard deviation `sqrt(.14)` m, or `sqrt(.22)` m in geometry 4. The unchanged Delft scale rule is

\[
\sigma_i^2=\max(\sigma_{native},0.30)^2+
\left(\tfrac12\,3.0\,\tau_i^2\right)^2.
\]

Two CV starts have 120 evaluations each and the reflected fit has 80, with Huber scale 2. A squared-hyperbola algebraic seed and anchor centroid replace the TWR algebraic seed and centroid; every accepted estimate minimizes original TDoA residuals, with no inertial position or truth seed. Antenna XY is translated to IMU XY using the native calibrated lever and retained live rotation.

Partition the robust weighted Jacobian into target `J_xy` and nuisance `J_nu=[J_z,J_vx,J_vy,J_vz]`. With the rank-aware left singular vectors `U_r` of the nuisance,

\[
I_{xy}=J_{xy}^T(I-U_rU_r^T)J_{xy},\qquad
C_{xy}=I_{xy}^{-1}+0.50^2I_2.
\]

The frozen tests require four distinct anchors, full XY rank at relative tolerance `1e-8`, normalized RMS at most 2, fitted speed at most 8 m/s,

\[
\sqrt{5.991464547\,\lambda_{max}(C_{xy})}\le2\;\mathrm m,
\qquad P^-_{xy}-C_{xy}\succ0.
\]

A comparable reflected solution is rejected only when its XY separation exceeds 1 m; comparable means its twice-Huber cost is within 2 of the primary solution. Checks use existing 100 Hz IMU ticks every five samples and the original rolling maximum 2 s buffer. These are unchanged engineering operating requirements, not universal observability constants.

At handoff, only XY mean and its covariance block are replaced. XY cross blocks are cleared. The complete physical covariance block for the other 13 components, together with velocity, Z, rotation and biases, is preserved. Maximum retained-covariance discrepancy is `1.51e-14`; mean preservation discrepancies are zero. The native analytic Jacobian agrees with central differences to `2.55e-9`. Changing future and masked radio values by 10000 m leaves the audited decision identical.

## Complete fixed-window results

Primary windows include every available 20 Hz sample in [20,22) and [26,31), for all sixteen cases. All methods have identical output clocks and 100% coverage. No interval is selected from a trigger time.

| Method | 20–22 RMSE median / worst | 26–31 RMSE median / worst | 20–22 P95 median / worst |
| --- | ---: | ---: | ---: |
| Ordinary native InEKF | 16.736 / 97.363 | 12.517 / 334.214 | 21.863 / 121.170 |
| Fixed 2 s PV | 11.741 / 23.499 | 0.05314 / 0.16350 | 13.792 / 27.267 |
| Early XY | 1.6944 / 12.8869 | 0.05067 / 0.15933 | 0.10443 / 15.094 |
| Native output, then early branch | 1.6944 / 17.4857 | 0.05067 / 0.15933 | 0.10443 / 19.367 |

Units are metres; worst means the maximum across all sixteen cases. Early XY beats fixed PV in 14/16 early cases and ties in two. Against ordinary native InEKF it wins 13/16 early cases and 14/16 future cases. Against fixed PV, future errors win 9/16 and lose 7/16: this does not establish a consistent future advantage over the strong fixed-window solution.

Every-log median across its two yaw stresses is retained in `per_log_medians.csv`. Early XY beats ordinary native InEKF on 7/8 log medians in the early window and 8/8 future log medians. The early exception is geometry 3 / protocol 2: 0.69992 m versus native 0.50075 m. Both individual early cases lose there. Geometry 4 / protocol 3 / 180 degrees is the third early loss. Two geometry-2 / 180-degree future cases also lose to native and remain in the paired table.

Fourteen cases trigger at 20.05 s or 20.10 s. Geometry 4 / protocol 3 waits until 23.40 s for both yaw stresses. At 22 s its normalized RMS is 14.425, above 2, so the frozen rule correctly reports that the radio model fails its operating requirement despite a 1.233 m horizontal ellipse. Its complete early-window outcomes are:

| Yaw stress | Ordinary native | Fixed PV / early XY | Native output then early branch |
| --- | ---: | ---: | ---: |
| 0 degrees | 17.4857 | 12.8869 | 17.4857 |
| 180 degrees | 7.84915 | 8.00432 | 7.84915 |

The primary early RMSE includes the first 20 Hz output at 20 s, before the first useful batch. Across all sixteen early-XY cases, that single sample accounts for a median 99.7824% of total squared XY error in [20,22). Thus the complete-window RMSE and post-arrival tracking accuracy describe different aspects of availability.

At the parent's request, a secondary *common absolute* [20.2,22) diagnostic is saved separately, without changing primary results, thresholds or cases. Early XY median/worst RMSE is 0.05874/13.1143 m, versus native 17.2575/100.550 m and fixed PV 11.9481/23.8903 m. Its worst remains the late geometry-4/protocol-3 case. This diagnostic is not a trigger-relative crop.

## Reproduction and artifacts

```powershell
.\.venv\Scripts\python.exe imu-uwb-inekf/experiments/split_handoff/early_tdoa/run.py --workers 2
.\.venv\Scripts\python.exe imu-uwb-inekf/experiments/split_handoff/early_tdoa/analyze.py
.\.venv\Scripts\python.exe imu-uwb-inekf/experiments/split_handoff/early_tdoa/audit.py
.\.venv\Scripts\python.exe imu-uwb-inekf/experiments/split_handoff/early_tdoa/additional_scores.py
```

`protocol.json` contains prewritten thresholds and source hashes. `native_pairs.csv` has each case's comparisons against ordinary InEKF. `paired_metrics.csv` retains both native and fixed baselines. `geometry_summary.csv`, `per_log_medians.csv`, `triggers.csv`, `attempts.csv`, `validation.json` and `frontend_audit.json` expose strata and checks. `paired_position.png` displays every case. This is evidence of cross-source transfer on one fixed independent UTIL trial, not a blind deployment or a guarantee for every geometry.
