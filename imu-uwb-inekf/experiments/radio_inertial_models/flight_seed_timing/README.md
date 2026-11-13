<!-- Actual serial wall-clock measurement of fixed real Delft windows. -->
# Delft real-window yaw-policy timing

This one-off panel measures the existing BODY-bias plus directional-radio objective on six Delft flights. Each flight uses the prespecified 10 s endpoint and 2/4/8 s windows, eight surveyed anchors, and the default yaw coordinates. All 18 windows were available; all contain observations from all eight anchors (461–1,986 changed positive ranges per window). No window was substituted or selected using fitted outcomes.

`run.py` rebuilds record-level physics from the raw CSVs using the existing loader, 0.20 s sensor delay, 1.5 s initial prefix, and unchanged preintegration. All resulting arrays exactly match the original cached arrays. The frozen training calibration and existing `joint_branch/study.py` BODY-bias/geometry refinement are reused without modification.

Every timed invocation starts from fresh window preprocessing, runs both common robust CV translation fits, constructs its own yaw candidates and zero-bias seeds, and runs every required raw refinement. Local uses yaw zero, Profile-1 calls the original static-corrected `lifted_candidate`, and Uniform-2/4 use `linspace(-pi, pi, K, endpoint=False)`. Uniform-2 is the two-angle restriction of the existing uniform formula, newly measured on this real BODY target. Profile-1 makes one raw fit, with no preliminary seven-state fit. All methods use the same final directional-radio target and fixed 0.10 m/s² body-bias prior, Huber range loss, analytic Jacobian, and 120-evaluation cap per raw fit.

The timer includes event extraction, endpoint displacement reanchoring, body-bias sensitivity integration, both CV fits, candidate/seed construction, all raw refinements, and best-fit selection. It excludes imports, CSV loading, prefix attitude/bias estimation and record-level preintegration, training calibration, truth scoring, fingerprints, and file output. Those record-level integrals are common input state; no CV, Profile, window sensitivity, or refined fit is cached. The existing BODY solver computes no covariance/curvature, so none is added to this measurement. This is window-solver latency from common integrated input, not raw-file end-to-end startup or filter latency.

One process with one BLAS thread runs four unrecorded warmups, followed by three serial rounds over the fixed windows with cyclically mixed method order. Each table entry is first the median of three calls on that window/method; aggregate milliseconds are the median of those 18 window medians. Paired ratios are computed within windows before taking their median. The six reused flights and overlapping windows are descriptive evidence, not independent trials.

| Policy | Median ms | p95 of window medians, ms | Paired ratio / Local | Paired ratio / Profile-1 | Objective miss | XY RMSE, m |
|---|---:|---:|---:|---:|---:|---:|
| Local | 30.620 | 65.273 | 1.000 | 0.965 | 0/18 | 0.104125 |
| Profile-1 | 30.745 | 79.797 | 1.036 | 1.000 | 0/18 | 0.104125 |
| Uniform-2 | 76.338 | 164.286 | 2.341 | 2.280 | 0/18 | 0.104125 |
| Uniform-4 | 135.372 | 312.569 | 4.318 | 4.170 | 0/18 | 0.104125 |

Miss uses the existing threshold `J - Jbest > 1e-4 * (1 + Jbest)`, with best found across these four policies. It does not certify a global minimum. This panel shows a measured cost reduction relative to uniform multistart, but no Profile-1 reliability advantage over Local: all four methods achieve the same observed objective basins here. Median Profile-1 candidate/seed construction costs 0.710 ms; its median paired total excess over Local is 0.554 ms.

| Flight | Window s | Local ms | Profile-1 ms | Uniform-2 ms | Uniform-4 ms |
|---|---:|---:|---:|---:|---:|
| triangle | 2 | 15.861 | 17.332 | 37.115 | 75.669 |
| triangle | 4 | 67.501 | 78.694 | 108.147 | 148.920 |
| triangle | 8 | 58.272 | 86.046 | 140.426 | 311.252 |
| octagon | 2 | 12.708 | 12.724 | 42.079 | 66.322 |
| octagon | 4 | 34.501 | 30.196 | 77.206 | 130.769 |
| octagon | 8 | 64.880 | 50.058 | 123.142 | 246.410 |
| hourglass | 2 | 24.872 | 28.191 | 37.822 | 87.944 |
| hourglass | 4 | 28.109 | 34.131 | 65.425 | 123.987 |
| hourglass | 8 | 49.104 | 46.832 | 139.710 | 239.791 |
| random | 2 | 15.648 | 16.386 | 36.658 | 61.401 |
| random | 4 | 27.747 | 22.347 | 104.224 | 176.808 |
| random | 8 | 48.634 | 59.186 | 246.443 | 320.032 |
| star | 2 | 18.684 | 18.868 | 34.280 | 57.989 |
| star | 4 | 33.131 | 31.293 | 75.469 | 139.975 |
| star | 8 | 57.488 | 50.719 | 138.881 | 283.839 |
| square | 2 | 14.435 | 14.804 | 32.039 | 59.348 |
| square | 4 | 23.688 | 26.294 | 59.188 | 97.520 |
| square | 8 | 52.747 | 64.853 | 149.788 | 236.532 |

The 54 matching Local/Profile-1/Uniform-4 rows reproduce `evidence/paired_acquisition/delft_new_objective_seeds.csv`: maximum objective difference is 4.17e-8 (the archived CSV is rounded), maximum XY-error difference is 4.92e-12 m, and all total evaluation counts match exactly. A separate check against the full-precision source `agent/codex/decision_matrix_20261112/seed_check/frames_metrics.csv` pairs all three repeats (162 rows): maximum endpoint coordinate difference is 2.22e-16 m, with identical actual timestamps and event counts. Within this run all repeated objectives are identical, all 216 timed outputs are finite and report successful selected solves, and packet fingerprints/endpoints agree across all methods and repeats within every window.

`summary.csv` and `paired_window_ms.csv` retain the original frozen small tables here. Full precision timing calls, window medians, coverage, checks, parity rows, environment, hashes, and protocol for that measurement are under `agent/codex/flight_seed_timing_20261113/` at the workspace root. The exact originally measured source is retained as the audit-only text snapshot `measured_run_20261113.py.txt` (SHA256 `884c0867348665be24f4757da7adabb44142881866b57c30cdf2150fda240699`). Subsequent reproduction runs never update these frozen tables or the original results.

Reproduction requires the complete repository checkout (including its tracked sibling `imu-uwb-state-augment-deep-coupled-two-filters` loader sources), public Delft data, and Python packages NumPy, pandas, SciPy, Matplotlib, PyYAML, and threadpoolctl. The measurement used Python 3.11.15, NumPy 1.26.4, and SciPy 1.17.1. From `imu-uwb-inekf/`, run with the project environment:

```powershell
& ../.venv/Scripts/python.exe -B experiments/radio_inertial_models/flight_seed_timing/run.py --dataset 'E:/datasets/tu-delft-uwb-mhe/data_v1/UWB dataset' --output ../agent/codex/flight_seed_timing_runs/my_rerun
```

`--dataset` points to the directory containing `anchor_positions.yaml` and `twr/`. Omitting `--output` creates a fresh timestamped subdirectory of `agent/codex/flight_seed_timing_runs/`. An explicit output directory must be empty; every generated table and diagnostic is written only there. No runtime read or write of `agent` history is required. `--help` lists the options.

`calibration.json` is an exact copy of the original fixed calibration artifact, SHA256 `f53b37ffb56687317a19350730471ee062549254213e638ded3fb62ca6cb939c`. `fixed_inputs.json` contains the unchanged six training and six evaluation recording names and the original calibration/protocol provenance hashes. The runner reads these local artifacts instead of untracked historical `results/` protocols or calibrations. Calibration is frozen rather than refitted on the evaluation flights.

The optional `--archive-cache PATH` only compares freshly integrated arrays with old NPZ files if available. Missing files, shape differences, and numerical differences are recorded in `input_checks.json`; cached arrays never supply estimator inputs and missing archives do not prevent fitting. By default no NPZ archive is consulted. `checks.json` separately reports objective/endpoint-error parity coverage against the tracked `evidence/paired_acquisition/delft_new_objective_seeds.csv`: the expected coverage is 54 Local/Profile-1/Uniform-4 rows, with Uniform-2 explicitly unreferenced. If that CSV is absent, fitting and summaries still run and archive parity is marked missing. Full-precision endpoint-coordinate parity against the historical `frames_metrics.csv` described above belongs to the original measurement audit and is not required by the portable runner.
