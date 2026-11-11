# Frozen closed-loop outcome

The implementation achieves a real full-state conditional handoff with unique
packet accounting. The frozen policy does **not** meet a stable cross-domain
improvement target. Its current research route is stopped without retuning.

All 64 recording/outage cases completed; no numerical exception was removed.
Large finite physical failures remain in the tables. Strict same-history
ablation passed bitwise X/bias/P-diagonal equality before T=27 in all 64 cases.
Every first-two-second score is therefore exactly unchanged by the recovery
handoff itself. Earlier differences against the original `native` arm originate
from the startup replacement, not the later recovery intervention.

These are medians of per-flight XY RMSE, in metres, on the frozen 20 Hz grid.
Each pair is **same Profile startup followed by native recovery / Profile IEKS
recovery**. Delft has 6/6 supported flights in each phase. QDrone has 7/10 in
[0,2), 7/10 in [2,5), and 8/10 in [6,11), for every outage length; unsupported
truth is not extrapolated. Cross-length cases are correlated, not independent
new recordings.

| Domain | Gap | First 2 s, both arms | 2–5 s: native / replacement | 6–11 s: native / replacement |
|---|---:|---:|---:|---:|
| Delft | 2 | 0.105606 | 0.089717 / 0.088999 | 0.093628 / 0.093384 |
| Delft | 5 | 0.178711 | 0.087704 / 0.087133 | 0.093339 / 0.092973 |
| Delft | 10 | 0.362347 | 0.115790 / 0.086670 | 0.094547 / 0.092328 |
| Delft | 20 | 15.832200 | 16.450939 / 0.109417 | 39.429273 / 0.092308 |
| QDrone | 2 | 0.180746 | 0.118914 / 0.622182 | 0.551152 / 0.551154 |
| QDrone | 5 | 0.278686 | 0.117452 / 0.622084 | 0.545335 / 0.545347 |
| QDrone | 10 | 0.860001 | 0.622547 / 0.622551 | 0.634863 / 0.547243 |
| QDrone | 20 | 1.391077 | 0.620363 / 0.619642 | 0.577241 / 0.532639 |

The important negative result is larger than these medians suggest. On QDrone
building1 with a 2 s outage, the same-history ablation has 2–5 s RMSE 0.095214 m
and 6–11 s RMSE 12.464105 m. The actual Profile conditional handoff changes these
to 49.552409 m and 115.316161 m: paired deterioration **49.457195 m and
102.852056 m**. QDrone 2 s and 5 s future phases each have 1 win, 1 tie, 6 losses
against identical-history native recovery. No uncertainty guarantee or reliable
acceptance rule was established. The four-pass endpoint corrections can remain
large: QDrone Profile maximum last-pass attitude correction is 173.3 degrees.
Finite covariance and a solved seed do not establish successful acquisition.

Delft 20 s recovery does show a useful local mechanism: all six flights improve
in both later phases over identical-history native recovery. However, the old
lifted strong baseline remains materially better in early tails. Its 20 s
first-phase median is 3.015371 m versus Profile's 15.832200 m on the frozen
20 Hz phase; its 2–5 s worst flight is 0.114038 m versus Profile's 3.480500 m.
`results/delft_gap20_xy.png` and `results/qdrone_counterexample_xy.png` show the
actual persistent filtered trajectories, including the harmful handoff.

## Sampling phase and historical reconciliation

Additional saved-output analyses do not change or replace the frozen primary
protocol. Delft 20 s first-two-second medians are:

| Arm | Frozen 20 Hz at T=25 | Original 100 Hz emissions | 20 Hz offset +25 ms |
|---|---:|---:|---:|
| Old lifted | 3.015371 | **0.978615** | 0.159093 |
| Profile replacement | 15.832200 | 15.414753 | 15.402942 |
| Identical-history native recovery | 15.832200 | 15.414753 | 15.402942 |

The 0.978615 m entry exactly reproduces the historical original-density print.
The old lifted branch moves quickly after radio arrival, so weighting its first
transient sample by 1/40 instead of 1/200 matters. Triangle and hourglass also
have a recorded source timestamp 24.999999999999993 at the T=25 past-held score;
that sample still satisfies the literal outage mask `t < 25`. The phase boundary
audit preserves exact source representations. This is a large sampling-phase
sensitivity, not evidence that a different trajectory was substituted. The
Profile branch remains worse under all three declared views.

## Practical scope

The filter waits two seconds (QDrone median 2.006 s on its event timeline), then
computes. Reported solve wall times were measured during concurrent runs, not an
isolated real-time benchmark: Profile medians are approximately 0.60 s Delft
and 2.09 s QDrone; maxima 3.90 s and 24.44 s. Packet queuing during computation
is not simulated, and these results do not meet a real-time latency claim.
Those medians pool startup and recovery. For supported post-outage attempts
alone the medians are 1.58 s on Delft and 6.34 s on QDrone; these are the
recovery-only figures given in the manuscript.
QDrone has 4/40 unsupported recovery acquisitions because fewer than four anchors
were observed in that window; they retain genuine native continuation and are
counted. No physical heading truth is supplied in these caches, so there are no
measured yaw accuracy results.

The compact evidence is in `supported_summary.csv`, `paired_changes.csv`,
`solver_diagnostics.csv`, `phase_sensitivity_summary.csv` and
`phase_boundary_audit.csv`. `all_support_summary.csv` deliberately labels
partially observed medians as available-only; it is not the primary summary.
Full local trajectories are retained as ignored `.npz` caches for inspection.
