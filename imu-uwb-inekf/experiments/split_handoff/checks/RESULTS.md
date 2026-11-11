# Frozen public yaw-evidence audit

Implementation SHA256: `62e88bfa3ab41160698ee149c2efda8b7d8a4296713b0857f29dc7010050e33b`. Settings and inputs are those of `synthetic_evidence.py`; no threshold, noise seed or challenge parameter was adjusted after running `decide`.

Angles below are degrees. Physical yaw is 40.107 except near_180, where it is 179. Candidate is reported even when rejected; actual delta remains zero on rejection. sd is the implementation's diagnostic estimate, not established confidence coverage.

| Synthetic case | Candidate | Accept | sd | CV lower | Decision reason |
|---|---:|:---:|---:|---:|---|
| strong | 36.17 | yes | 1.59 | 2.40 | accepted |
| weak | -51.49 | no | 383.56 | — | ambiguous_profile |
| constant_velocity | -165.00 | no | 5.73e13 | — | ambiguous_profile |
| constant_acceleration | 133.30 | no | 5.73e13 | — | ambiguous_profile |
| near_180 | -179.12 | yes | 2.04 | 11.68 | accepted |
| coplanar_weak_z | 36.74 | no | 2.31 | -12.07 | no_predictive_advantage |
| rotating_bias | 36.08 | yes | 3.60 | 0.32 | accepted |
| wrong_bias_model | 35.80 | yes | 1.54 | 2.32 | accepted |
| tdoa_strong | 36.56 | no | 2.20 | -6.12 | no_predictive_advantage |
| systematic_yaw_alias | 108.25 | yes | 1.50 | 8.84 | accepted |
| link_offsets | 36.69 | yes | 1.93 | 2.25 | accepted |
| tdoa_link_offsets | 37.26 | no | 2.79 | -2.90 | no_predictive_advantage |
| bias_only_excitation | -164.25 | no | 163.20 | — | ambiguous_profile |

The implementation rejects exact CV/constant-acceleration ambiguity, weak motion and the bias-only counterexample. It handles the circular seam. It also rejects excited TDoA and coplanar examples because held-block path prediction is unfavorable; this is evidence of conservative recovery behavior, not a proof those geometries have no horizontal information.

The coherent-alias case is accepted with a 68.14-degree physical error and a reported sd of 1.50 degrees. This is the expected indistinguishability counterexample: its radio observations and fitted path are consistent with the wrong trajectory. It must remain visible in reporting. The wrong-B case also passes, so acceptance does not certify the nominal bias model. In this single fixed noise realization, strong and wrong-B physical errors exceed 1.96 times their reported sd; one realization cannot estimate coverage, but neither result supports a calibrated 95% claim.

## Implementation audit

`audit_implementation.py` checks the frozen module independently:

- The sine/cosine decomposition, bias sign, Cref order and local tilt nuisance terms agree with direct vector rotation to maximum error 2.27e-13.
- Perturbing every held-out radio measurement by +1000 m changes neither the training-only radio path nor the training profile cost: both differences are exactly zero. In this module there is no observed held-radio value leakage into a fold's fit or weights. Full-window path construction outside this module still needs its own review.
- A separate 1441-angle full-circle scan found no lower basin than the public refined result in nine nontrivial selected challenges. This finite comparison is not a global guarantee for all inputs. The public search brackets only its best 15-degree grid basin; narrow competing minima and profile-arc coverage can in principle be missed between mesh points.
- The base affine/world-acceleration/link space is removed without penalties. Rotating bias and tilt columns are recombined at each yaw, then regularized for fitting. Data-only curvature projects those columns without their priors, as intended. Numerical rank decisions are column-normalized.
- Fold paths genuinely refit radio-only on training measurements. Using all time stamps, survey geometry, the common IMU window and the known directed-link inventory is structural input, not a held-out radio-value leak.

No algebra/sign error or held-radio value leakage was identified in the audited module. There are material statistical limits: the Huber weights are fixed from radio-path residuals rather than independently known; the four folds share training data, are not four independent experiments, and can share correlated radio errors. A Student-t margin with three degrees of freedom is therefore a screening heuristic, not a proved 95% lower confidence bound. Jackknife spread and local curvature similarly do not calibrate physical heading coverage under wrong B, timing, Cref or coherent radio errors. Per-link offsets remove constant means, not arbitrary shared-anchor or time-correlated noise.

## Reproduction

From the project root:

```powershell
& ./.venv/Scripts/python.exe imu-uwb-inekf/experiments/split_handoff/checks/synthetic_evidence.py --module yaw_evidence
& ./.venv/Scripts/python.exe imu-uwb-inekf/experiments/split_handoff/checks/audit_implementation.py
& ./.venv/Scripts/python.exe imu-uwb-inekf/experiments/split_handoff/checks/plot_checks.py
```

Complete per-case outputs, fold decisions, algebra checks and NPZ inputs remain under `agent/codex/split_checks`. The public algorithm and its thresholds were not modified.
