<!-- Map manuscript figures and tables to their independent experimental inputs. -->
# Revised manuscript evidence

The current manuscript is `tex/recovery_window_revision.tex` in the workspace root.
The two vector figures are generated from recorded numerical outputs. With the project
Python, run `plot.py` to rebuild them and `verify_tables.py` to check the 194 numeric
cells printed in the six tables. Compile the TeX twice with pdfLaTeX from its own
directory. The figures and bibliography are included in the separate manuscript source ZIP.

| Claim | Evidence relative to the experiments directory |
|---|---|
| Affine vs bias with one yaw seed, endpoint, and model/seed controls | `radio_inertial_models/evidence/paired_acquisition/` |
| Shared Delft windows: two published Pfeiffer estimators vs single-start bias; native-author conditions separately | `published_window_comparison/evidence/profile1_author_overall.csv` and the directory README |
| Same Delft source transactions, published Pacholska relaxation without IMU | `published_range_only/results/overall_summary.csv`, per-window rows and author-function validation |
| Physical heading on UTIL, unchanged Oursland class | `split_handoff/oursland/early_overlay/results/summary.csv` (exact copy `toronto_author_summary.csv` here) |
| Theodosis v2 example, 20 s and extended numerical validation | `split_handoff/quadratic_observer/` |
| Same-history single-filter replacement vs native continuation and the old lifted baseline | `unified_window_recovery/results/metrics.csv`, `paired_changes.csv`, `phase_sensitivity_summary.csv` |

The stored plot samples contain causal outputs on the common evaluation grid for
the first Delft flight and the largest late QDrone loss following a 2 s gap.
The latter is a counterexample, not the typical flight. There is no test-truth
alignment and no claim that the single-filter replacement is ready for flight.
