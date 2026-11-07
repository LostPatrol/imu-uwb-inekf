"""Paired outside-hull and weak-height stress test of frozen recovery solvers.

This experiment uses exact calibrated inertial displacement, Gaussian ranges,
and uniform unknown yaw. Multistart is an empirical optimization reference,
not ground truth or a certified global solver. CV has a different range model.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

# Prevent each process from creating its own numerical-library thread pool.
for _thread_variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_thread_variable] = "1"

import numpy as np

# Import the existing, frozen implementation without copying or modifying it.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from initializer import solve_window, yaw_rotation

# The four alternatives receive exactly the same transactions in each case.
METHODS = ("cv", "local", "profile", "multistart")
# This master seed determines case seeds independently of worker scheduling.
MASTER_SEED = 202611070659
# Each window has four seconds of asynchronous transactions from eight anchors.
WINDOW_SECONDS, TRANSACTIONS, RANGE_SIGMA = 4.0, 240, 0.1
# Errors are diagnostic thresholds rather than declared system requirements.
POSITION_THRESHOLD_M, YAW_THRESHOLD_RAD = 0.5, 0.35
# A gap must exceed both numerical scales to count as a worse fitted objective.
OBJECTIVE_ABS_TOL, OBJECTIVE_REL_TOL = 1e-3, 1e-5


def evaluate_case(specification):
    """Generate one physical case and evaluate all four methods on shared data."""
    case_id, seed, distance, height, motion = specification
    rng = np.random.default_rng(seed)
    tau = np.sort(rng.uniform(-WINDOW_SECONDS, 0.0, TRANSACTIONS))
    anchor_set = np.array([[x, y, z] for x in (-4.0, 4.0)
                           for y in (-4.0, 4.0) for z in (0.0, height)])
    anchors = anchor_set[rng.integers(0, len(anchor_set), TRANSACTIONS)]
    bearing = rng.uniform(-np.pi, np.pi)
    position = np.array([distance*np.cos(bearing), distance*np.sin(bearing), 1.2])
    velocity = rng.uniform(-0.5, 0.5, 3)
    yaw = rng.uniform(-np.pi, np.pi)
    if motion == "excited":
        displacement = np.column_stack((0.8*np.sin(1.7*tau),
            0.6*(np.cos(1.1*tau)-1.0), 0.15*np.sin(0.7*tau)))
        # Remove the initial polynomial component so both endpoint derivatives vanish.
        displacement -= tau[:, None]*np.array([1.36, 0.0, 0.105])
    else:
        displacement = 0.5*tau[:, None]**2*np.array([0.4, -0.3, 0.1])
    positions = position + tau[:, None]*velocity + displacement@yaw_rotation(yaw).T
    true_ranges = np.linalg.norm(positions-anchors, axis=1)
    ranges = true_ranges + rng.normal(0.0, RANGE_SIGMA, TRANSACTIONS)
    sigmas = np.full(TRANSACTIONS, RANGE_SIGMA)
    rows = []
    for method in METHODS:
        estimate = solve_window(tau, displacement, anchors, ranges, sigmas, method)
        delta = estimate.parameters[:3]-position
        angle_error = np.arctan2(np.sin(estimate.parameters[6]-yaw),
                                np.cos(estimate.parameters[6]-yaw))
        rows.append(dict(case_id=case_id, seed=seed, distance_m=distance,
            anchor_height_m=height, motion=motion, method=method,
            true_px=position[0], true_py=position[1], true_pz=position[2],
            true_yaw_rad=yaw, estimated_px=estimate.parameters[0],
            estimated_py=estimate.parameters[1], estimated_pz=estimate.parameters[2],
            estimated_yaw_rad=estimate.parameters[6],
            position_error_m=float(np.linalg.norm(delta)),
            horizontal_error_m=float(np.linalg.norm(delta[:2])),
            height_error_m=float(abs(delta[2])), yaw_error_deg=float(abs(np.rad2deg(angle_error))),
            position_failure=int(np.linalg.norm(delta)>POSITION_THRESHOLD_M),
            yaw_failure=int(abs(angle_error)>YAW_THRESHOLD_RAD),
            robust_objective=estimate.robust_objective,
            yaw_information=estimate.yaw_information, elapsed_s=estimate.elapsed_s,
            evaluations=estimate.evaluations,
            solver_success=estimate.diagnostics.get("solver_success", "not_reported")))
    reference = rows[-1]["robust_objective"]
    tolerance = max(OBJECTIVE_ABS_TOL, OBJECTIVE_REL_TOL*abs(reference))
    for row in rows:
        # CV lacks inertial displacement, so its objective is not a basin comparison.
        gap = row["robust_objective"]-reference
        row["objective_gap_vs_multistart"] = gap if row["method"] != "cv" else ""
        row["worse_objective_vs_multistart"] = int(gap>tolerance) if row["method"] != "cv" else ""
        row["better_objective_vs_multistart"] = int(gap < -tolerance) if row["method"] != "cv" else ""
    return rows


def summarize(rows):
    """Aggregate truth errors, objective gaps, and measured total solver time."""
    summaries = []
    for distance in (10.0, 30.0):
        for height in (3.0, 0.1):
            for motion in ("excited", "constant_acceleration"):
                for method in METHODS:
                    subset = [r for r in rows if (r["distance_m"], r["anchor_height_m"],
                        r["motion"], r["method"]) == (distance, height, motion, method)]
                    if not subset:
                        continue
                    result = dict(distance_m=distance, anchor_height_m=height,
                                  motion=motion, method=method, cases=len(subset))
                    for name in ("position_error_m", "horizontal_error_m", "height_error_m",
                                 "yaw_error_deg", "robust_objective", "elapsed_s"):
                        values = np.array([r[name] for r in subset])
                        result[name+"_median"] = float(np.median(values))
                        result[name+"_p95"] = float(np.quantile(values, 0.95))
                        result[name+"_max"] = float(np.max(values))
                    for name in ("position_failure", "yaw_failure"):
                        result[name+"_count"] = sum(r[name] for r in subset)
                    for name in ("worse_objective_vs_multistart", "better_objective_vs_multistart"):
                        result[name+"_count"] = sum(r[name] for r in subset) if method != "cv" else ""
                    result["solver_not_success_count"] = sum(r["solver_success"] is False for r in subset)
                    summaries.append(result)
    return summaries


def write_csv(path, rows):
    """Write a stable UTF-8 comma-separated table with descriptive column names."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    """Run the preregistered grid with reproducible per-case seeds and paired data."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=200)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--output", type=Path, default=HERE.parents[1]/"results"/"recovery_geometry_20261107")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/"metrics.csv").exists():
        raise FileExistsError("Refusing to replace an existing completed metrics table")
    rng = np.random.default_rng(MASTER_SEED)
    cases = []
    for distance in (10.0, 30.0):
        for height in (3.0, 0.1):
            for motion in ("excited", "constant_acceleration"):
                for _ in range(args.seeds):
                    cases.append((len(cases), int(rng.integers(0, 2**32)), distance, height, motion))
    protocol = dict(created_utc=datetime.now(timezone.utc).isoformat(), master_seed=MASTER_SEED,
        seeds_per_grid=args.seeds, workers=args.workers, methods=METHODS, cases=cases,
        distance_definition="Endpoint horizontal distance from anchor centroid; azimuth uniform.",
        anchor_definition="Eight anchors at x,y in {-4,+4}, z in {0,height} metres.",
        true_position_z_m=1.2, true_yaw="uniform [-pi,pi]", bias="fixed zero calibration error",
        displacement="Exact known preintegrated displacement; gravity alignment exact.",
        excited_displacement="[.8 sin(1.7t), .6(cos(1.1t)-1), .15 sin(.7t)] - t[1.36,0,.105]",
        constant_acceleration_displacement="0.5*t^2*[.4,-.3,.1]",
        window_seconds=WINDOW_SECONDS, transactions=TRANSACTIONS, sigma_m=RANGE_SIGMA,
        noise="Independent zero-mean Gaussian, no outliers", blas_threads=1,
        position_failure_threshold_m=POSITION_THRESHOLD_M, yaw_failure_threshold_rad=YAW_THRESHOLD_RAD,
        objective_gap_tolerance=dict(absolute=OBJECTIVE_ABS_TOL, relative=OBJECTIVE_REL_TOL),
        objective_reference="12-yaw multistart; empirical reference, not certified optimum",
        cv_caveat="CV omits displacement and reports yaw=0 fallback, not an estimated heading.",
        source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in (Path(__file__), HERE/"initializer.py")})
    (args.output/"protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    started, rows = time.perf_counter(), []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(evaluate_case, case) for case in cases]
        for completed, future in enumerate(as_completed(futures), 1):
            rows.extend(future.result())
            if completed % 50 == 0 or completed == len(cases):
                print(f"Completed {completed}/{len(cases)} paired cases in {time.perf_counter()-started:.1f}s", flush=True)
    rows.sort(key=lambda row: (row["case_id"], METHODS.index(row["method"])))
    write_csv(args.output/"metrics.csv", rows)
    summary = summarize(rows)
    write_csv(args.output/"summary.csv", summary)
    (args.output/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    protocol["elapsed_s"] = time.perf_counter()-started
    protocol["completed_utc"] = datetime.now(timezone.utc).isoformat()
    (args.output/"protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(f"Saved {len(rows)} method rows to {args.output}", flush=True)


if __name__ == "__main__":
    main()
