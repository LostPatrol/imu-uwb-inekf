# Deferred heading handoff

This experiment runs the normal 15-state right-invariant IMU/TDoA filter continuously after a sensor-only position handoff. It reuses the frozen shared `yaw_evidence.decide` screen and changes heading at the first accepted scheduled attempt. Truth is discarded during trajectory generation and loaded only for scoring.

Trial 6, geometries 1–4 and TDoA protocols 2/3 are fixed in advance. Geometry 4 always uses `traj1`. Initial yaw stresses are 0 and 180 degrees relative to the same gravity-based initial estimate; zero stress does not imply zero true heading error. There are 32 matched cases and 128 policy runs. Trial 6 is distinct from the other UTIL group's trial 3/4/5 experiment. An independent recording date has not been established.

The cold schedule is 2/6/10/14 s from sensor origin, using a growing window from origin. The outage schedule masks radio during [10,20) s, then attempts at 22/26/30/34 s using a growing window from 20 s. IMU and normal live corrections continue between attempts. No position, velocity or bias reset is applied after the initial 0.6 s radio-only CV estimate. The outage experiment therefore also retains failure to recover position after a long gap.

After the shared initialization prefix, alternating transactions within each directed anchor pair are permanently assigned to either live correction or heading evidence. Initial packets are excluded from both later streams. Masked candidate packets are physically removed before radio-only path fitting. Candidate paths and observations are clipped at each actual decision time. The accepted yaw mean consumes reserved evidence, while the later filter uses only live packets strictly after that decision.

At attempt endpoint `e`, with gyro-relative integration `Q_e` from the window start, the actual current live attitude supplies

\[
C_{ref,e}=R^{live}_e Q_e^T.
\]

The unchanged screen profiles native TDoA residuals with affine motion, bias/tilt and channel nuisances. On first acceptance, only world-gravity yaw changes:

\[
R^+_e=R_z(\delta_e)R^-_e,\qquad
p^+_e=p^-_e,\quad v^+_e=v^-_e,\quad b^+_e=b^-_e.
\]

If `M(X)` maps physical local errors to the invariant chart, the coordinate transport is

\[
P_{phys}=M(X^-)^{-1}P^-M(X^-)^{-T},\qquad
P^+=M(X^+)P_{phys}M(X^+)^T.
\]

Thus acceptance does not add a covariance reduction. The four policies are retain throughout, first accepted yaw, forced candidate only at the first clock, and forced candidate at every clock. Forced candidates include rejected ones. Every policy uses the same initial state, radio partition, live filter and fixed clock schedule.

Primary future XY errors use all cases after the same absolute last clock: 14 s for cold and 34 s for outage, ending at 45 s. Trigger time never chooses a baseline or scoring interval. Incomplete coverage and failed traces are retained explicitly; their partial errors must not be interpreted as full-duration success. No thresholds or schedules are changed after looking at scores.

Run from the repository root:

```powershell
.\.venv\Scripts\python.exe imu-uwb-inekf/experiments/split_handoff/deferred/run.py --workers 2
.\.venv\Scripts\python.exe imu-uwb-inekf/experiments/split_handoff/deferred/audit.py
.\.venv\Scripts\python.exe imu-uwb-inekf/experiments/split_handoff/deferred/analyze.py
```

`results/protocol.json` records the fixed settings and source hashes. `metrics.csv`, `paired_future.csv`, `accepted_cases.csv` and the two figures retain matched outcomes. `causality_audit.json` checks large perturbations of future sensors and masked radio. `mechanism_audit.json` checks timing, yaw-only changes, physical covariance preservation, coverage and source identity.

The final run accepted 4/32 cases, all after earlier rejections: two cold handoffs at 14 s and two outage handoffs at 30 s. Future XY improved in two cases, worsened in two, and was unchanged in 28. The all-case future medians are:

| Scenario / yaw stress | Retain | First accepted | Forced first | Forced each |
| --- | ---: | ---: | ---: | ---: |
| Cold / 0 degrees | 0.07127 | 0.07137 | 0.07146 | 0.07340 |
| Cold / 180 degrees | 0.07394 | 0.07394 | 0.07298 | 0.07322 |
| Outage / 0 degrees | 25.970 | 25.970 | 34.577 | 27.285* |
| Outage / 180 degrees | 21.588 | 21.588 | 29.493 | 20.758 |

Units are metres. *Forced-each includes one failed trace with only 45.2% future coverage; its partial RMSE remains visible and is marked in the figure. All nonfailed policies have identical output clocks. The small cold errors show the live filter already recovers, so these data do not establish a positioning gain from deferred yaw. The long-outage experiment retains severe position failure: even the two improved accepted cases remain at 76.32 m and 264.99 m future XY RMSE. It does not establish successful recovery after a new accurate position handoff.

One accepted cold case, `const4-trial6-tdoa3-traj1`, worsens immediate true heading error from 19.77 to 48.84 degrees and future XY from 0.16244 to 0.17470 m. The screen is therefore not a reliable correctness certificate in this experiment. Three of the four accepted handoffs reduce immediate heading error, which is insufficient to establish benefit after normal filtering continues.

The candidate/live intersection is empty. Future sensor and masked radio perturbations produce exactly zero change in the audited candidate inputs. Handoff position and velocity changes are zero; body vertical changes stay below 2.3e-16 and relative physical covariance change below 1.8e-14. No candidate packet is later than its decision time.

After the final launch, the other UTIL group added `--resume` only to its unused command entrypoint. `results/imported_util_run.py.txt` preserves the exact launched bytes, verified against the prewritten SHA-256; `source_provenance.json` verifies every imported function's AST is unchanged. Consequently the current whole-file hash differs while the imported numerical functions remain the frozen ones. No rerun or parameter change was made for that unrelated entrypoint edit.
