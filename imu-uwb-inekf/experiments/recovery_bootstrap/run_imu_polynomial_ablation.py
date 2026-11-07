"""Evaluate frozen quadratic nuisance fitting on the frozen physical-IMU cases.

This changes model order and uses two translation seeds; it is not an isolated
bias-variable or equal-seed comparison with the primary one-profile experiment.
"""

from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import os

# Keep process-level concurrency separate from BLAS concurrency.
for _variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_variable] = "1"

import numpy as np
import pandas as pd
import initializer
import polynomial_bias
from run_imu_mismatch_stress import (physical_case, measured_shape, LEVELS,
                                     DURATION, TRANSACTIONS, RANGE_SIGMA, HERE)

# This inherited operational threshold is not a calibrated posterior confidence.
YAW_ERROR_THRESHOLD_RAD = 0.35
INFORMATION_THRESHOLD = 1/YAW_ERROR_THRESHOLD_RAD**2


def evaluate_case(specification):
    """Fit the unchanged K=2 model on all three mismatch levels of one case."""
    case_id, seed = specification
    data = physical_case(seed)
    tau = data["event_times"]-DURATION
    true_rotation = initializer.yaw_rotation(data["yaw"])@data["q_true"][-1]
    true_heading = np.arctan2(true_rotation[1, 0], true_rotation[0, 0])
    rows = []
    for level, accel_std, gyro_std, tilt_std in LEVELS:
        displacement, q_end = measured_shape(data, accel_std, gyro_std, tilt_std)
        original, calls = initializer.least_squares, []
        def counted(*args, **kwargs):
            """Record both shared CV calls and the two polynomial refinements."""
            result = original(*args, **kwargs)
            calls.append(result)
            return result
        initializer.least_squares = polynomial_bias.least_squares = counted
        try:
            parameters, information, objective, elapsed, diagnostic = polynomial_bias.solve_polynomial(
                tau, displacement, data["anchors"], data["ranges"],
                np.full(TRANSACTIONS, RANGE_SIGMA), degree=2)
            chosen = min(calls[-2:], key=lambda fit: initializer.robust_cost(fit.fun))
            np.testing.assert_array_equal(parameters, chosen.x)
            estimate_rotation = initializer.yaw_rotation(parameters[-1])@q_end
            heading = np.arctan2(estimate_rotation[1, 0], estimate_rotation[0, 0])
            error = float(np.arctan2(np.sin(heading-true_heading), np.cos(heading-true_heading)))
            row = dict(case_id=case_id, seed=seed, level=level, method="quadratic_two_translation_seeds",
                position_error_m=float(np.linalg.norm(parameters[:3]-data["position"])),
                velocity_error_mps=float(np.linalg.norm(parameters[3:6]-data["velocity"])),
                yaw_error_deg=float(abs(np.rad2deg(error))), yaw_information=information,
                diagnostic_accepted=bool(information >= INFORMATION_THRESHOLD),
                accepted_but_inaccurate=bool(information >= INFORMATION_THRESHOLD and
                                              abs(error)>YAW_ERROR_THRESHOLD_RAD),
                robust_objective=objective, elapsed_s=elapsed,
                total_nfev=sum(fit.nfev for fit in calls), cv_nfev=sum(fit.nfev for fit in calls[:2]),
                least_squares_calls=len(calls), solver_success=bool(chosen.success),
                finite=bool(np.isfinite(parameters).all()), error="")
        except Exception as error:
            row = dict(case_id=case_id, seed=seed, level=level, method="quadratic_two_translation_seeds",
                position_error_m=float("inf"), velocity_error_mps=float("inf"),
                yaw_error_deg=float("inf"), yaw_information=0.0, diagnostic_accepted=False,
                accepted_but_inaccurate=False, robust_objective=float("inf"), elapsed_s=float("nan"),
                total_nfev=sum(fit.nfev for fit in calls), cv_nfev=sum(fit.nfev for fit in calls[:2]),
                least_squares_calls=len(calls), solver_success=False, finite=False, error=repr(error))
        finally:
            initializer.least_squares = polynomial_bias.least_squares = original
        rows.append(row)
    return rows


def summarize(frame):
    """Report unconditional errors and information rejection alongside every failure."""
    rows = []
    for level, part in frame.groupby("level", sort=False):
        row = dict(level=level, cases=len(part), accepted=int(part.diagnostic_accepted.sum()),
            accepted_but_inaccurate=int(part.accepted_but_inaccurate.sum()),
            nonfinite=int((~part.finite).sum()), nonconverged=int((~part.solver_success).sum()),
            total_nfev=int(part.total_nfev.sum()))
        for name in ("position_error_m", "velocity_error_mps", "yaw_error_deg", "yaw_information",
                     "elapsed_s", "total_nfev"):
            row[name+"_median"] = float(part[name].median())
            row[name+"_p95"] = float(part[name].quantile(.95))
            row[name+"_max"] = float(part[name].max())
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    """Freeze the additional model comparison and optionally rerun twenty cases."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--input", type=Path, default=HERE.parents[1]/"results/recovery_imu_mismatch_20261107")
    parser.add_argument("--output", type=Path, default=HERE.parents[1]/"results/recovery_imu_polynomial_20261107")
    args = parser.parse_args()
    original_protocol = json.loads((args.input/"protocol.json").read_text(encoding="utf-8"))
    cases = original_protocol["cases"]
    args.output.mkdir(parents=True, exist_ok=True)
    if args.audit:
        cases = [cases[index] for index in original_protocol["audit_case_ids"]]
    else:
        if (args.output/"metrics.csv").exists():
            raise FileExistsError("Completed results will not be overwritten")
        protocol = dict(created_utc=datetime.now(timezone.utc).isoformat(), input=str(args.input),
            parent_protocol_sha256=hashlib.sha256((args.input/"protocol.json").read_bytes()).hexdigest(),
            cases=cases, workers=args.workers, blas_threads=1, degree=2,
            methods="Frozen solve_polynomial: lift translation and CV-plus-zero-quadratic translation; both use lifted yaw",
            scope="Model and seed-policy ablation, not identical translation seeds or equal optimization budgets",
            information="Data-only projected Huber IRLS yaw information; threshold 1/.35^2 is diagnostic, not calibrated coverage",
            input_generation="Identical physical_case and measured_shape functions as the frozen 1800-fit experiment",
            source_sha256={path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in
                (Path(__file__), HERE/"run_imu_mismatch_stress.py", HERE/"polynomial_bias.py", HERE/"initializer.py")})
        (args.output/"protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        frame = pd.DataFrame([row for group in pool.map(evaluate_case, cases) for row in group])
    frame = frame.sort_values(["case_id", "level"]).reset_index(drop=True)
    if args.audit:
        saved = pd.read_csv(args.output/"metrics.csv")
        subset = saved[saved.case_id.isin(original_protocol["audit_case_ids"])].sort_values(
            ["case_id", "level"]).reset_index(drop=True)
        columns = [name for name in frame.select_dtypes(include=np.number) if name != "elapsed_s"]
        np.testing.assert_allclose(frame[columns], subset[columns], rtol=1e-12, atol=1e-11)
        for name in ("solver_success", "finite", "diagnostic_accepted", "accepted_but_inaccurate"):
            np.testing.assert_array_equal(frame[name], subset[name])
        result = dict(cases=len(cases), rerun_fits=len(frame),
            maximum_numeric_difference=float(np.max(np.abs(frame[columns].to_numpy()-subset[columns].to_numpy()))),
            all_outputs_finite=bool(saved.finite.all()), completed_utc=datetime.now(timezone.utc).isoformat())
        (args.output/"audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result), flush=True)
    else:
        frame.to_csv(args.output/"metrics.csv", index=False)
        summary = summarize(frame)
        summary.to_csv(args.output/"summary.csv", index=False)
        print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
