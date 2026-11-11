"""Evaluate the published Relax and Recover range-only linear reconstruction.

Each estimator receives only arrived calibrated ranges, surveyed anchors and times.
The common Delft loader supplies the evaluation clock and separate scoring truth.
"""

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

# A small sequential calculation needs no nested BLAS parallelism.
for _name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_name] = "1"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy

# These paths keep all new evidence inside this independent experiment.
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
RADIO = HERE.parent / "radio_inertial_models"
sys.path.insert(0, str(HERE.parent / "recovery_bootstrap"))
from run_research import load_real_log, load_anchors, DATASET

# K counts coefficients; K=2 is the author's published linear-polynomial setting.
SETTINGS = (("relax_recover_K2", 2, False),
            ("relax_recover_weighted_K2", 2, True),
            ("relax_recover_weighted_K3_selected_here", 3, True))
# This is exactly the author's get_constraints weighted distance regularizer.
WEIGHT_REGULARIZER_M2 = 0.1
# These eleven windows are the existing physical-window protocol, without yaw copies.
WINDOWS = tuple((e, w) for e in (5., 10., 20., 30.) for w in (2., 4., 8.) if w <= e)
# Acquisition-to-log correction and crop match joint_branch/study.py.
SENSOR_DELAY_S = 0.20
STATIC_PREFIX_S = 1.5
# Freeze the upstream revision, rather than following future source changes.
AUTHOR_COMMIT = "b2a2b58205a3c33d79c10615e22a3252a55ba090"


def sha256(path):
    """Identify the precise source or data bytes used in this execution."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reconstruct(tau, anchors, ranges, complexity, weighted):
    """Solve paper equations (5)-(6), using SVD least squares for stability."""
    scale = max(float(np.ptp(tau)), 1.0)
    basis = (tau[:, None] / scale) ** np.arange(complexity)[None, :]
    left = (anchors[:, :, None] * basis[:, None, :]).reshape(len(tau), -1)
    right = (basis[:, :, None] * basis[:, None, :]).reshape(len(tau), -1)
    rhs = (np.sum(anchors * anchors, axis=1) - ranges**2) / 2
    weights = 1 / np.sqrt(ranges**2 + WEIGHT_REGULARIZER_M2) if weighted else np.ones(len(tau))
    left, right, rhs = left * weights[:, None], right * weights[:, None], rhs * weights
    u, singular, _ = np.linalg.svd(right, full_matrices=False)
    nuisance_rank = 2 * complexity - 1
    # The polynomial Gram term has only 2K-1 independent sampled monomials.
    design = np.column_stack((left, -.5 * u[:, :nuisance_rank] * singular[:nuisance_rank]))
    solution, _, rank, design_singular = np.linalg.lstsq(design, rhs, rcond=None)
    coeff = solution[:3 * complexity].reshape(3, complexity)
    diagnostic = dict(rank=int(rank), expected_rank=design.shape[1],
                      condition=float(design_singular[0] / design_singular[-1]),
                      gram_rank=int(np.linalg.matrix_rank(right)),
                      linear_residual_norm=float(np.linalg.norm(design @ solution - rhs)),
                      basis_time_scale_s=scale)
    return coeff, basis, diagnostic


def upstream_solver():
    """Load only original numerical functions without optional CVXPY dependencies."""
    namespace = {"np": np}
    names = {"verify_dimensions", "get_left_submatrix", "get_right_submatrix", "get_constraints",
             "trajectory_recovery"}
    for name in ("source__constraints.py", "source__solvers.py"):
        tree = ast.parse((HERE / "sources" / name).read_text(encoding="utf-8-sig"))
        for function in tree.body:
            if isinstance(function, ast.FunctionDef) and function.name in names:
                exec(compile(ast.Module(body=[function], type_ignores=[]), name, "exec"), namespace)
    return namespace["trajectory_recovery"]


def check_formula():
    """Check noiseless 3D recovery and literal author-function agreement."""
    rng = np.random.default_rng(20261112)
    anchors = rng.normal(size=(8, 3)) * 4
    tau = np.linspace(-2., 0., 200)
    ids = rng.integers(0, 8, len(tau))
    original = upstream_solver()
    checks = []
    for method, complexity, weighted in SETTINGS:
        truth_coeff = rng.normal(size=(3, complexity))
        phi = (tau[:, None] / np.ptp(tau)) ** np.arange(complexity)
        distances = np.linalg.norm(phi @ truth_coeff.T - anchors[ids], axis=1)
        coeff, basis, diagnostic = reconstruct(tau, anchors[ids], distances, complexity, weighted)
        squared = np.zeros((len(tau), len(anchors)))
        squared[np.arange(len(tau)), ids] = distances**2
        author_coeff = original(squared, anchors.T, basis.T, weighted=weighted)
        noiseless_error = float(np.max(np.abs(coeff - truth_coeff)))
        source_difference = float(np.max(np.abs(coeff - author_coeff)))
        assert noiseless_error < 1e-10 and source_difference < 1e-9
        checks.append(dict(method=method, noiseless_max_coefficient_error=noiseless_error,
                           upstream_max_coefficient_difference=source_difference, **diagnostic))
    return checks


def evaluate_record(name, anchors, cal):
    """Rebuild changed positive raw events and score each endpoint after fitting."""
    log = load_real_log(DATASET, name, sensor_delay_s=SENSOR_DELAY_S)
    selected = log.time >= STATIC_PREFIX_S
    times = log.time[selected] - log.time[selected][0]
    ranges, changed = log.ranges[selected], log.changed[selected]
    truth = log.ground_truth_position[selected]
    rows, checks = [], []
    author = upstream_solver()
    for endpoint, duration in WINDOWS:
        end = int(np.searchsorted(times, endpoint, side="right") - 1)
        start = int(np.searchsorted(times, times[end] - duration, side="left"))
        for layout in ("all8", "lower4"):
            mask = changed[start:end+1] & (ranges[start:end+1] > 0)
            if layout == "lower4":
                mask[:, 4:] = False
            indices, ids = np.nonzero(mask)
            indices += start
            tau = times[indices] - times[end]
            measured = ranges[indices, ids] - cal["static"][ids]
            sigma = cal["sigma"][ids]
            payload = np.column_stack((times[indices], ids, ranges[indices, ids], measured))
            event_hash = hashlib.sha256(payload.astype("<f8").tobytes()).hexdigest()
            for method, complexity, weighted in SETTINGS:
                started = time.perf_counter()
                coeff, phi, diagnostic = reconstruct(tau, anchors[ids], measured, complexity, weighted)
                elapsed = time.perf_counter() - started
                valid = diagnostic["rank"] == diagnostic["expected_rank"] and np.isfinite(coeff).all()
                position = coeff[:, 0]
                error = position - truth[end]
                predicted = np.linalg.norm(phi @ coeff.T - anchors[ids], axis=1)
                residual = (predicted - measured) / sigma
                # Shared Huber-2 diagnostic; it is never optimized by this method.
                abs_res = np.abs(residual)
                huber = float(np.sum(np.where(abs_res <= 2, residual**2, 4 * abs_res - 4)))
                rows.append(dict(sequence=name, endpoint_s=endpoint, actual_endpoint_s=times[end],
                    duration_s=duration, layout=layout, frame_deg=0, method=method,
                    complexity=complexity, polynomial_degree=complexity-1, weighted=weighted,
                    events=len(tau), unique_times=len(np.unique(tau)), event_sha256=event_hash,
                    solver_success=bool(valid), finite=bool(np.isfinite(coeff).all()),
                    position_error_m=float(np.linalg.norm(error)) if valid else np.inf,
                    xy_error_m=float(np.linalg.norm(error[:2])) if valid else np.inf,
                    abs_z_error_m=float(abs(error[2])) if valid else np.inf,
                    px_m=position[0], py_m=position[1], pz_m=position[2],
                    truth_x_m=truth[end, 0], truth_y_m=truth[end, 1], truth_z_m=truth[end, 2],
                    diagnostic_huber2_cost=huber, raw_range_rmse_m=float(np.sqrt(np.mean((predicted-measured)**2))),
                    elapsed_s=elapsed, **diagnostic))
                if endpoint == 10. and duration == 4.:
                    matrix = np.zeros((len(tau), 8))
                    matrix[np.arange(len(tau)), ids] = measured**2
                    literal = author(matrix, anchors.T, phi.T, weighted=weighted)
                    difference = float(np.max(np.abs(coeff - literal)))
                    relative = difference / max(1., float(np.max(np.abs(coeff))))
                    assert relative < 1e-6
                    checks.append(dict(sequence=name, layout=layout, method=method,
                        max_coefficient_difference=difference, relative_difference=relative))
    return rows, checks


def summarize(frame, group):
    """Keep all windows and expose failures rather than dropping nonfinite errors."""
    rows = []
    for key, part in frame.groupby(group, sort=False):
        keys = key if isinstance(key, tuple) else (key,)
        row = dict(zip(group, keys), windows=len(part), failures=int((~part.solver_success).sum()),
                   median_solve_ms=float(np.median(part.elapsed_s) * 1000),
                   p95_solve_ms=float(np.quantile(part.elapsed_s, .95) * 1000),
                   max_condition=float(part.condition.max()))
        for axis in ("position_error_m", "xy_error_m", "abs_z_error_m"):
            values = part[axis].to_numpy()
            row.update({axis + "_median": float(np.median(values)),
                        axis + "_p95": float(np.quantile(values, .95)),
                        axis + "_rmse": float(np.sqrt(np.mean(values**2))),
                        axis + "_max": float(np.max(values))})
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    """Run only the prespecified three settings on the 132 paired configurations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "results")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    calibration_file = RADIO / "radio_branch/results/calibration.json"
    protocol_file = RADIO / "radio_branch/results/protocol.json"
    source = json.loads(protocol_file.read_text(encoding="utf-8"))
    raw_cal = json.loads(calibration_file.read_text(encoding="utf-8"))
    cal = {key: np.asarray(raw_cal[key]) for key in ("static", "sigma")}
    anchors = load_anchors(DATASET)
    validation = dict(formula=check_formula(), literal_author_panel=[])
    rows = []
    for name in source["evaluation"]:
        result, checks = evaluate_record(name, anchors, cal)
        rows.extend(result)
        validation["literal_author_panel"].extend(checks)
        print(f"DONE {name}: {len(rows)} fits", flush=True)
    frame = pd.DataFrame(rows)
    assert len(frame) == 396
    frame.to_csv(args.output / "windows.csv", index=False)
    for keys, name in ((["method", "layout"], "overall"),
                       (["method", "layout", "sequence"], "per_log"),
                       (["method", "layout", "duration_s"], "duration")):
        summarize(frame, keys).to_csv(args.output / f"{name}_summary.csv", index=False)
    paired = pd.read_csv(RADIO / "evidence/paired_acquisition/delft_same_profile.csv")
    frozen_keys = ["sequence", "endpoint_s", "duration_s", "layout"]
    shared = frame.drop_duplicates(frozen_keys).merge(paired[paired.frame_deg == 0], on=frozen_keys,
                                                     validate="one_to_one", suffixes=("", "_reference"))
    assert len(shared) == 132 and (shared.events == shared.events_reference).all()
    validation["matched_reference_windows"] = len(shared)
    validation["all_event_counts_identical"] = True
    validation["all_fits_full_rank"] = bool(frame.solver_success.all())
    (args.output / "validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    metadata = dict(doi="10.1109/LRA.2020.2970952", author_commit=AUTHOR_COMMIT,
        author_repo="https://github.com/LCAV/continuous-localization", settings=SETTINGS,
        training=source["training"], evaluation=source["evaluation"], calibration=calibration_file.as_posix(),
        calibration_static=cal["static"].tolist(), calibration_sigma=cal["sigma"].tolist(),
        no_imu_in_solver=True, no_truth_initialization=True, no_hyperparameter_search=True,
        true_position_used_for_offline_scoring_only=True, common_truth_clock_crop_in_loader=True,
        model="r(tau)=C*[1,tau/scale,...]; K=2 main, K=3 selected here, weighted as author code",
        numerical_change="SVD least squares replaces normal-equation inverse; same full-rank linear objective",
        weighting="Author weighted option: row multiplier 1/sqrt(calibrated_range^2+0.1); sigma used only for Huber diagnostics",
        independent_units="6 same-day logs, 66 overlapping time windows x 2 anchor layouts; not iid",
        blas_threads=1, workers=1, python=sys.version, numpy=np.__version__, scipy=scipy.__version__,
        pandas=pd.__version__, platform=platform.platform(),
        source_sha256={str(p): sha256(p) for p in (Path(__file__), calibration_file, protocol_file)},
        upstream_source_sha256={p.name: sha256(p) for p in (HERE / "sources").iterdir() if p.is_file()},
        dataset_sha256={name: sha256(DATASET / "twr" / name) for name in source["evaluation"]})
    (args.output / "protocol.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, layout in zip(axes, ("all8", "lower4")):
        for method, _, _ in SETTINGS:
            errors = np.sort(frame[(frame.layout == layout) & (frame.method == method)].position_error_m)
            ax.plot(errors, np.arange(1, len(errors)+1)/len(errors), label=method.replace("relax_recover", "R&R"))
        ax.set_title(layout)
        ax.set_xlabel("Endpoint 3D error (m)")
        ax.set_xscale("log")
        ax.grid(alpha=.25)
    axes[0].set_ylabel("Empirical fraction of windows")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(args.output / "endpoint_error_cdf.png", dpi=160)
    plt.close(fig)
    print(summarize(frame, ["method", "layout"]).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
