# Controlled decision-time coordinate loss

This experiment asks whether heading correction improves future XY when heading is deliberately changed at the handoff instant, rather than several seconds earlier when normal filtering may already have corrected it. It is a controlled restart/coordinate-loss experiment, not evidence that the same jumps naturally occur during outages.

Rules are fixed in `rules.json`. Trial7 has only three usable geometry/protocol slots: const3 TDoA2 manual1, const3 TDoA3 manual3, const4 TDoA2 manual1. These are the first manual filenames by availability, chosen before reading sensor values. Const1 and const2 both protocols, plus const4 TDoA3, are unavailable. The eight-slot manifest retains all five missing slots; no different trial is substituted.

The fixed dynamic window is [10,16] s. The recovery window is [20,26] s after physically deleting radio in [10,20) s. One normal shadow runs to each decision endpoint. At that instant its R is left-multiplied by 0, +90, -90 or 180 degrees, while its body position, original velocity, body bias and physical covariance are shared exactly. Four policies change only R: retain, frozen public yaw-evidence gate, the same candidate without its gate, and full-attitude fitting. No case is selected by actual physical error, improvement, gate acceptance or injected label.

The public rule is recomputed using each jumped `R_now @ Qend.T`. Eight-start full-attitude fitting is reused across the 90-degree world-yaw reference reparameterizations, since its yaw grid and isotropic horizontal tilt prior are invariant under these rotations. The first available +90 case is also directly refitted and the discrepancy is saved, rather than assumed zero. The full fit's internal translation/bias outputs are not injected.

All groups retain the same original shadow physical covariance, including any prior overconfidence after a coordinate loss. This controls uncertainty and isolates mean-attitude changes; it is not an uncertainty-reset design. The shared native numba InEKF is used without source changes. To prevent endpoint interpolation from using a raw sample after the decision, this experiment's loader uses causal previous-sample hold on a 100 Hz grid, instead of the shared loader's linear interpolation. All policies receive exactly this same resampled sensor stream.

The generator reads sensor columns only; pose columns are first opened after every decision and trajectory is saved. Candidate path and radio are clipped to the endpoint. No capture radio is replayed as a new future measurement. Future XY, Z, yaw and SO(3) are scored for the fixed next 5 and 15 seconds. Original shadow, artificially jumped and post-correction endpoint yaw errors are separately retained in `metrics.csv`; an injected 90-degree label is never substituted for an actual 90-degree error.

Run from the project root:

```powershell
$env:OPENBLAS_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
& ./.venv/Scripts/python.exe imu-uwb-inekf/experiments/split_handoff/reference_jump/run.py
& ./.venv/Scripts/python.exe imu-uwb-inekf/experiments/split_handoff/reference_jump/analyze.py
```

All generated artifacts, including the manifest, source hashes, complete trajectories, decisions, paired future scores and figures, remain in this directory. Missing coverage and unsuccessful policies must remain explicit in the final interpretation.

## Observed result

All 96 future trajectories completed without a numerical abort: 3 recordings, 6 physical windows, 24 correlated nominal/perturbed reference cases and 4 policies. This is not 8 recordings or 96 independent acquisitions. Five geometry/protocol slots were unavailable, so the requested four-geometry coverage was not achieved.

The gate accepted 4/24 cases, all during continuous-ranging dynamic windows; all 12 recovery cases were rejected. The following numbers pool the fixed reference interventions within each scenario and are descriptive, not independent-sample statistics.

| Scenario | Policy | Next 5 s XY median / max (m) | Next 5 s yaw median (deg) | Next 15 s XY median / max (m) |
|---|---|---:|---:|---:|
| Dynamic | retain | 0.121 / 0.231 | 84.87 | 0.149 / 0.172 |
| Dynamic | gated | 0.116 / 0.212 | 11.61 | 0.147 / 0.154 |
| Dynamic | ungated same candidate | 0.117 / 0.205 | 5.14 | 0.148 / 0.150 |
| Dynamic | full attitude | 0.118 / 0.201 | 5.16 | 0.148 / 0.151 |
| Recovery | retain | 1.424 / 29.185 | 87.99 | 0.836 / 24.671 |
| Recovery | gated | 1.424 / 29.185 | 87.99 | 0.836 / 24.671 |
| Recovery | ungated same candidate | 1.410 / 27.737 | 14.67 | 0.824 / 24.118 |
| Recovery | full attitude | 1.412 / 28.572 | 11.81 | 0.825 / 23.798 |

Actual shadow yaw errors before the artificial intervention were 10.63, 163.61 and 23.82 degrees for the three dynamic windows, and 17.51, 100.85 and 54.65 degrees for recovery. An artificial 180-degree jump can therefore improve a shadow already near 180 degrees wrong. All such nominal and perturbed cases remain included and individually labeled.

Across the 24 paired reference cases, gated next-5-second XY improvement has median zero and mean 2.51 mm. Three accepted cases improve XY and one worsens it; the largest improvement is 2.60 cm and largest worsening is 0.91 cm. Large heading correction was demonstrated in some cases, but a substantial XY rescue was not demonstrated. In particular, const3 TDoA3 recovery already has 29.072 m XY error at the shared shadow endpoint; changing only heading does not repair that position loss. Numerical non-abort must not be interpreted as localization success.

The reused full-attitude orientation and a direct +90-degree-reference refit differ by 0.001075 degrees SO(3), with objectives 249.826436214 and 249.826436249. Full-attitude future XY scores are identical across reference jumps, confirming that its common physical orientation and common p/v/bias/covariance were preserved.
