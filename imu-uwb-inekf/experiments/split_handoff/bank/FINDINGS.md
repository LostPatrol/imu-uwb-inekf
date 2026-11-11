# Frozen predictive heading bank

The implementation and numerical rules were frozen before any UTIL truth score was inspected. The source/rule hashes are in `verification/implementation_checks.json`. No bank parameter has been changed after inspecting the first shared replay outputs.

## Real-data result

The bank does not show a general improvement over retaining the matched normal-track heading. In corrected `util/development_final`, it switches in 2/96 cases: one improves and one worsens subsequent XY. In independent trial4/5 `util/holdout_final`, it switches in 3/192 cases: two worsen and one improves subsequent XY. Every switch is in outage recovery. All cold and dynamic cases retain. The other 283 trajectories equal matched retention exactly; every scenario/injection-group median paired XY change is zero. All 288 cases have finite scores, no replay-failure flag and complete trajectories to approximately 44.95 s.

The held-out decisions are especially unfavorable as heading recovery: all three increase immediate heading error despite pseudo-masses above 0.93. A small eventual XY benefit in one case does not make its initial heading correction correct.

| Split / recording | Initial yaw injection | Applied yaw | Heading error before -> after | Post XY change vs matched retain |
|---|---:|---:|---:|---:|
| dev / const3-trial3-tdoa2 | +90 deg | +45 deg | 49.44 -> 94.44 deg | -1.215 mm |
| dev / const4-trial3-tdoa3-traj1 | -90 deg | +135 deg | 147.68 -> 12.68 deg | +1.285 mm |
| holdout / const3-trial4-tdoa2 | 0 deg | -135 deg | 20.03 -> 114.97 deg | +10.168 mm |
| holdout / const3-trial4-tdoa2 | -90 deg | -135 deg | 30.41 -> 165.41 deg | +11.740 mm |
| holdout / const4-trial5-tdoa2-traj1 | -90 deg | +45 deg | 22.48 -> 67.48 deg | -5.455 mm |

Positive XY change means worse RMSE. Initial injection is applied at initialization, not at handoff; it must not be mistaken for the actual heading error that survives to handoff.

The held-out post-handoff XY RMSE medians are descriptive paired-schedule summaries, in metres:

| Scenario / injection | Cases / switches | Bank | Matched retain | Forced evidence yaw | Full attitude |
|---|---:|---:|---:|---:|---:|
| cold / nominal | 16 / 0 | .071325 | .071325 | .073222 | .071273 |
| cold / large | 48 / 0 | .073721 | .073721 | .073231 | .072621 |
| dynamic / nominal | 16 / 0 | .067538 | .067538 | .067767 | .067737 |
| dynamic / large | 48 / 0 | .067720 | .067720 | .067456 | .067518 |
| outage / nominal | 16 / 1 | .067135 | .067135 | .066220 | .066395 |
| outage / large | 48 / 2 | .066468 | .066468 | .066171 | .066317 |

`results/paired_cases.csv` preserves every paired case, including initial/arrival/retained/handoff heading errors, and comparisons to forced full-fit yaw, full attitude, forced evidence yaw and unmodified normal tracking. Scenario, geometry/protocol and per-injection tables preserve the fixed configurations rather than selecting successful cases. `results/paired_post_xy.png` marks all five switches. The numerical rules remain unchanged after both splits.

Exploratory first-2/5-second scores reuse saved trajectories and the shared truth scorer only for all five accepted cases. Both development switches worsen XY at both horizons. In holdout, the const3 -90 case worsens by 6.49/13.68 mm; const3 nominal improves by 11.16 mm at 2 s but worsens by 1.85 mm at 5 s; const4 -90 improves by 3.37/8.35 mm. These selected accepted-case diagnostics in `results/accepted_short_horizon.csv` do not replace the primary complete post-handoff comparison and do not support a general benefit.

## Implementation checks

Physical asynchronous held-IMU/TWR smoke cases use seed 20261111 and a 20-degree initial tilt. These implementation checks supply oracle tilt and perturb only world yaw; real UTIL uses the common estimated reference. Rich motion accepts corrections for injected +90 and 180 degrees while retaining the correct reference. Weak motion and constant velocity defer for all three tested reference headings. The rotating-body, constant-world-acceleration cases also select the injected correction under the specified bias prior; this does not establish general yaw or full SO(3) observability. These twelve cases are implementation checks, not real-data or independent generalization evidence.

Replacing the unused full-window radio path with NaNs leaves every non-timing output identical. Corrupting the last predicted radio block leaves all earlier traces identical. Arbitrary affine additions to the IMU force and bias bases leave the decision unchanged, with maximum block prediction-loss difference 7.3e-7. A two-second crop provides only two prediction blocks and retains by the frozen rule.

A fixed constructed coherent-radio-fault check keeps the IMU and retained heading correct, but replaces ranges by those of a +90-degree alternative trajectory with the same original measurement noise. The fault RMS is 4.52 m. The bank incorrectly accepts +90 degrees with pseudo-mass 1 and sd 12.99 degrees. This is a strong adversarial identifiability counterexample, not typical-radio-noise or real-data evidence. It confirms that future-block prediction cannot expose a persistent observation fault that consistently imitates another physical trajectory.

## Comparison protocol and cost

Only the corrected final directories are primary evidence. Earlier smoke/partial outputs used superseded common interpolation and are excluded. The shared replay owner removed outage radio before path construction, truncated radio-only paths at actual capture endpoints, and verified cold/dynamic/outage post-end poison sentinels. Array and decision differences were zero after excluding the nondeterministic `elapsed_s` timing field. Cold arrival scoring uses the actual 0.6 s initializer time.

Matched reset covariance/p/v/bias and native subsequent InEKF integration belong to the common replay. `normal_unmodified` also preserves its own p/v/covariance, so differences against it include the common handoff reset, not only heading. Matched retention is the primary heading comparison. A separate literal-position-only shared panel does not include bank and is not claimed as bank evidence.

Bank median measured decision time is 0.151 s in development and 0.131 s in holdout; maximums are 1.541/2.162 s under the shared machine workload. This is host timing, not an embedded or real-time guarantee. Three holdout prefix particle fits hit the iteration limit, all in rejected cases; all five accepted cases have zero unconverged prefix fits. No failed fit or trial is dropped.

## Boundaries

The eight-point yaw grid has a 22.5-degree worst nearest-grid error; its reported circular spread has a 12.99-degree quantization floor and is not a calibrated posterior uncertainty. Weak excitation intentionally leaves some large heading errors unresolved. A coarse bank cannot establish local precision, and this implementation does not estimate tilt.

The prefix fit excludes the next radio block, but the shared API supplies a terminal reference attitude. If that reference already used window radio updates, only the bank's internal fit/score separation is causal; the complete shared experiment is a conditional-reference window method. This distinction must survive any paper or benchmark claim.

An independent read-only theory audit confirmed the TDoA sign, body-bias Jacobian, robust-loss derivatives and affine gauge algebra. Its additional limits matter: window duration and block plan must be known in advance for online use; constant attitude/acceleration can leave a yaw-bias ambiguity that the bias prior resolves by preference rather than data identification; coherent radio errors can favor a wrong particle in every prediction block; tilt or timing errors can partly appear as yaw. The tempered clipped score does not integrate p/v/bias uncertainty and yields pseudo-weights rather than calibrated probabilities. Unconverged prefix fits are counted and remain in the evidence; they are not silently discarded.

For example, with constant local attitude, specific force `s` and world acceleration `a`, every heading can reproduce the same trajectory by choosing

\[
 b(\delta)=s-C_{ref}^{T}R_z(-\delta)(a-g).
\]

Raw radio then cannot separate heading from an unrestricted constant body bias. A small-bias prior expresses an extra physical assumption; prediction on later samples of the same constant-motion regime does not remove that ambiguity. The bank favors retaining a heading unless another particle has stable predictive evidence under this assumption; it is not a general yaw observability certificate.

Development uses trial3, all four geometries and TDoA2/3 (8 recordings, 96 related scenario/injection cases). Holdout uses trial4/5 with the same fixed geometries/protocols (16 distinct recordings, 192 related cases). Variants within a recording are related observations, not independent trials. The synthetic checks and superseded smoke do not count as independent real-data generalization. Reproduction commands are in `README.md`.
