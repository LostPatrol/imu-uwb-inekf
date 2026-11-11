<!-- Reproduction guide and claim boundaries for the November 2026 research study. -->
# Range/IMU recovery research

This experimental directory preserves the mature TDoA entry point. It requires the existing project environment and the sibling TU Delft/deep-filter research sources already used by `experiments/roadmap_stage1`.

Activate the environment defined by `conda/conda-env.yaml` from Windows PowerShell. On another computer set `INEKF_TWR_DATASET` to its TU Delft dataset directory before prepare or real replay. Set `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, and `OMP_NUM_THREADS` to `1` when using process workers. The study used up to 20 workers in total on the 20-core Intel Core Ultra 7 265KF. No new dependencies were installed.

From the `imu-uwb-inekf` directory:

```powershell
conda activate imu-uwb-inekf
$env:INEKF_TWR_DATASET = 'E:/datasets/tu-delft-uwb-mhe/data_v1/UWB dataset'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$study = 'results/recovery_bootstrap_reproduction'
python -m unittest discover -s experiments/recovery_bootstrap -p 'test_*.py' -v
python experiments/recovery_bootstrap/run_research.py --suite prepare --output $study
python experiments/recovery_bootstrap/run_research.py --suite real --output $study --workers 20
python experiments/recovery_bootstrap/run_research.py --suite synthetic --output $study --workers 20 --seeds 200
python experiments/recovery_bootstrap/polynomial_bias.py --output "$study/bias_metrics.csv" --workers 20 --seeds 1000
python experiments/recovery_bootstrap/analyze_research.py --results $study --figures ../tex/recovery_bootstrap_figures
python experiments/recovery_bootstrap/diagnose_startup.py --cache "$study/cache" --output "$study/startup_diagnosis" --workers 20
```

Additional paired experiments use the same prepared cache. Run the timing panel alone, after all parallel workloads finish:

```powershell
python experiments/recovery_bootstrap/bias_real.py --cache "$study/cache" --output results/recovery_bias_reproduction --workers 20
python experiments/recovery_bootstrap/analyze_bias_real.py --result results/recovery_bias_reproduction
python experiments/recovery_bootstrap/run_geometry_stress.py --output results/recovery_geometry_reproduction --workers 20 --seeds 200
python experiments/recovery_bootstrap/window_solver_study.py --cache "$study/cache" --output results/recovery_window_solver_reproduction --workers 20
python experiments/recovery_bootstrap/window_candidate_study.py --cache "$study/cache" --base results/recovery_window_solver_reproduction --output results/recovery_window_candidates_reproduction --workers 20
python experiments/recovery_bootstrap/window_candidate_study.py --cache "$study/cache" --output results/recovery_window_candidates_reproduction --timing-only
python experiments/recovery_bootstrap/run_imu_mismatch_stress.py --output results/recovery_imu_mismatch_reproduction --workers 20 --seeds 200
python experiments/recovery_bootstrap/run_imu_mismatch_stress.py --output results/recovery_imu_mismatch_reproduction --audit
python experiments/recovery_bootstrap/run_imu_polynomial_ablation.py --input results/recovery_imu_mismatch_reproduction --output results/recovery_imu_polynomial_reproduction --workers 20
python experiments/recovery_bootstrap/run_imu_polynomial_ablation.py --input results/recovery_imu_mismatch_reproduction --output results/recovery_imu_polynomial_reproduction --audit
python experiments/recovery_bootstrap/run_imu_affine_control.py --input results/recovery_imu_mismatch_reproduction --output results/recovery_imu_affine_control_reproduction --workers 20
python experiments/recovery_bootstrap/run_imu_affine_control.py --input results/recovery_imu_mismatch_reproduction --output results/recovery_imu_affine_control_reproduction --audit
```

Check `analyze_bias_real.py --help` if supplying a different cache or output directory. Formal evidence is preserved in these directories:

| Result directory | Recorded evidence |
|---|---|
| `results/recovery_bootstrap_20261107_final` | 96 complete replays, 4,800 basic synthetic fits, 12,000 bias fits; independent trajectory audit |
| `results/recovery_bias_20261107_final` | 60 rotating-bias complete replays and independent verification |
| `results/recovery_geometry_20261107` | 6,400 geometry fits, 1,600 paired conditions, independent summaries and 80 exact repeat fits |
| `results/recovery_window_solver_20261107` | Initial three-method recorded-window study |
| `results/recovery_window_candidates_20261107` | Combined six-method study: 528 configurations, 3,168 fits, 720 sequential timings |
| `results/recovery_imu_mismatch_20261107` | 1,800 physical IMU calibration-mismatch fits; 180 independent repeat fits and a separate frame/integration audit |
| `results/recovery_imu_polynomial_20261107` | 600 quadratic-model fits on the identical physical cases; 60 repeat fits |
| `results/recovery_imu_affine_control_20261107` | 600 affine-model fits with the same two-translation-seed strategy as the quadratic model; 60 repeat fits |
| `results/recovery_candidate_budget_20261111` | 400 paired windows, 2,000 fits, 400 sequential timings, protocols, summaries and validation |
| `results/recovery_real_windows_20261111` | 3,168 saved endpoint fits, per-log/layout and frame-zero summaries, historical timing summary and 36-fit verification |

The 528 window configurations come from 66 physical trajectory windows, two anchor subsets and four yaw-frame parameterizations; they are not independent recordings. The candidate count and objective tolerance are fixed in the saved protocols. A miss denotes an objective above the best value found by the compared solvers, not failure against a certified global optimum. All CV initialization solves are included in the recorded nonlinear evaluation totals; SVD and polynomial costs are included in wall time, not in the nonlinear evaluation count.

The physical IMU stress study couples accelerometer bias, gyroscope bias and initial tilt at two fixed scales. It uses rotating attitudes and held-sample integration with shared range noise, but no IMU white noise. Its mild case preserves the profile candidate's best-found objective reliability; the moderate case exposes substantial model-mismatch errors. It is a controlled simulation, not a new real recording or an isolated single-factor causal test.

The two polynomial-order experiments call the same frozen `solve_polynomial` with degree 1 or 2. Both use a lifted translation seed and a robust CV seed, with the lifted yaw for that model order. Thus they control the seed policy and optimizer limits while changing the model order and induced lift. The quadratic model trades some ideal-data precision for materially better mismatch tolerance. Raw objectives from different model dimensions are not ranked as objective misses against each other.

Unsuffixed bootstrap/bias directories are preliminary local runs and are omitted from the Git snapshot. The prior `roadmap_stage1_20261104_final/protocol.json` supplies the frozen calibration and original dataset fingerprints. All twelve real logs have prior development exposure; the split is not a blind test.

Additional figure builders use the saved formal directories: `plot_window_study.py` produces the complete six-method window plot; `experiments/recovery_bootstrap/geometry_stress_audit.py` independently rechecks the geometry summaries, repeats 80 fits, and renders the geometry plot. The latter is an audit script rather than an application entry point. `analyze_research.py` generates the other main and supplementary figures.

The algebraic solver globally minimizes a relaxed two-dimensional unit-circle problem. This is not a global convergence guarantee for raw-range refinement or the 15-state invariant tracker. Polynomial bias invariance is exact only for the stated additive polynomial model; arbitrary rotating body-frame biases require the separate matrix-preintegration model. IRLS curvature and heuristic handoff covariance floors have not been validated as calibrated posterior uncertainty.

Real window methods deliberately report startup unavailability. Subsequent outage predictions remain in the metrics, including large finite errors. `startup_diagnosis` deliberately uses truth initialization only to isolate update-policy effects; it is not cold-start performance evidence.

Paper sources and PDFs are in the project-root `tex` directory. From the project root, run `& ./tex/build_recovery_paper.ps1` with the installed TeX Live 2026. The script builds the English paper and supplement with IEEEtran/pdfLaTeX/BibTeX, and the Chinese mathematical companion with XeLaTeX. Compiler intermediates are kept under `agent/codex/recovery_paper_build`; final PDFs are copied to `tex`. Compact frozen tables, protocols and audit summaries are retained; full trajectory/cache arrays and local delivery inventories are not part of Git. See the workspace-root `docs/evidence.md` for retained evidence scope.

## Conference revision: candidate cost and real endpoint error

`budget_candidate_study.py` compares Local, Profile-1 and Uniform-2/4/12 on the same two existing 30 m excited-motion cells, each with 200 original seeds. Two CV translation fits select one shared translation; each yaw then receives one raw-range refinement. Per-solve settings match, while total evaluation/time budgets differ and are explicitly measured. Misses use the five-method joint best; a second column retains the twelve-start reference. Wall timing includes CV, profile generation and final curvature, from a separate sequential subset.

`real_window_metrics.py` resummarizes saved endpoint position errors and computation by log/layout, with frame-zero summaries to expose repeated parameterizations. It uses 66 overlapping physical windows, 132 layout windows and 528 correlated configurations. Optional cached replay checks endpoint coordinates against raw OptiTrack interpolation. These data have no usable heading truth and are not blind testing.

From the workspace root in Windows PowerShell, using the local project environment:

```powershell
$env:INEKF_TWR_DATASET = 'E:/datasets/tu-delft-uwb-mhe/data_v1/UWB dataset'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:MPLBACKEND = 'Agg'
& ./.venv/Scripts/python.exe imu-uwb-inekf/experiments/recovery_bootstrap/budget_candidate_study.py --seeds 200 --workers 4 --main-only --output imu-uwb-inekf/results/candidate_cost_reproduction
# Run sequential timing after other numerical experiments are idle.
& ./.venv/Scripts/python.exe imu-uwb-inekf/experiments/recovery_bootstrap/budget_candidate_study.py --seeds 200 --timing-only --output imu-uwb-inekf/results/candidate_cost_reproduction
& ./.venv/Scripts/python.exe imu-uwb-inekf/experiments/recovery_bootstrap/real_window_metrics.py --output imu-uwb-inekf/results/endpoint_reproduction
```

For optional endpoint verification, first prepare a fresh cache with `run_research.py --suite prepare --output <study>`, then add `--cache <study>/cache` to the endpoint command. All endpoint times use the post-prefix retained clock, not raw recording start; a first 2 s capture follows the approximately 1.5 s static prefix. The November 11 timing uses an i5-12600KF, while saved November 7 real-window timings use the original Core Ultra 7 265KF. See `docs/conference-revision.md` at workspace root for result paths and interpretation.

## Manuscript visuals

`plot_paper_visuals.py` draws a vector method overview and a two-panel real-window plot from the saved per-log CSV. The trajectory and projection panels are model schematics; the real points use the recorded objective misses and endpoint position percentiles. Profile-1 is teal, Local orange, and uniform policies use cool hues with distinct markers.

From the workspace root:

```powershell
$env:MPLBACKEND = 'Agg'
# Optional: rebuild endpoint summaries from the retained historical CSVs, without raw data or fits.
& ./.venv/Scripts/python.exe imu-uwb-inekf/experiments/recovery_bootstrap/real_window_metrics.py --output imu-uwb-inekf/results/endpoint_reproduction
& ./.venv/Scripts/python.exe imu-uwb-inekf/experiments/recovery_bootstrap/plot_paper_visuals.py
& ./.venv/Scripts/python.exe imu-uwb-inekf/experiments/recovery_bootstrap/budget_candidate_study.py --plot-only
Copy-Item -LiteralPath imu-uwb-inekf/results/recovery_candidate_budget_20261111/candidate_budget_compact.pdf, imu-uwb-inekf/results/recovery_candidate_budget_20261111/candidate_budget_compact.png -Destination tex/recovery_bootstrap_figures
```

These commands use retained CSVs without rerunning fits. The endpoint command reads the saved `recovery_window_candidates_20261107` and `recovery_window_solver_20261107` tables; adding `--cache` enables separate replay verification. The budget plot needs only its saved `summary.csv` and `protocol.json`, and updates the plot fingerprint in that protocol. Its `--summarize-only` option also runs six fit checks, so use `--plot-only` for figure-only work.

Use `--real-summary` and `--output` on `plot_paper_visuals.py` to select regenerated summaries or a different destination. The TeX build uses the copied figures directly. The November 11 evidence snapshot retains protocols, per-fit metrics, summaries, timing and verification records (20 files, about 1.67 MiB); raw data, caches, NPZ and duplicate result-directory images are excluded. See the workspace-root `docs/evidence.md` for the retained scope. Submission access and anonymity must be arranged for the selected venue.
