"""Freeze paired physical-IMU perturbations for the existing far-field solvers.

This is a development simulation, not independent real-data validation. True
and measured IMU streams share the same held-sample model and endpoint anchor.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

# Each worker owns a single numerical-library thread.
for _variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_variable] = "1"

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

# Frozen solvers are imported without editing their implementation.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import initializer

# One seed generates one physical experiment shared across all error levels.
MASTER_SEED = 202611070714
METHODS = ("local", "profile", "multistart")
# Per-axis constant accelerometer/gyro bias and initial body tilt standard deviations.
LEVELS = (("ideal", 0.0, 0.0, 0.0), ("mild", 0.02, 0.002, 0.5),
          ("moderate", 0.10, 0.01, 2.0))
# The geometry and range noise match the previous 30 m / 3 m excited stress cell.
DURATION, IMU_DT, TRANSACTIONS, RANGE_SIGMA = 4.0, 0.005, 240, 0.1
GRAVITY = np.array([0.0, 0.0, -9.81])
OBJECTIVE_ABS_TOL, OBJECTIVE_REL_TOL = 1e-3, 1e-5


def integrate(acceleration, event_times):
    """Integrate held acceleration and reanchor displacement at the endpoint."""
    velocity = np.vstack((np.zeros(3), np.cumsum(acceleration * IMU_DT, axis=0)))
    position = np.vstack((np.zeros(3), np.cumsum(
        velocity[:-1] * IMU_DT + 0.5 * acceleration * IMU_DT**2, axis=0)))
    index = np.minimum((event_times / IMU_DT).astype(int), len(acceleration)-1)
    offset = event_times-index*IMU_DT
    event_position = (position[index] + offset[:, None]*velocity[index]
                      + 0.5*offset[:, None]**2*acceleration[index])
    return event_position-position[-1]-(event_times-DURATION)[:, None]*velocity[-1]


def physical_case(seed):
    """Generate true rotating IMU data, asynchronous ranges, and shared perturbations."""
    rng = np.random.default_rng(seed)
    event_times = np.sort(rng.uniform(0.0, DURATION, TRANSACTIONS))
    anchor_set = np.array([[x, y, z] for x in (-4.0, 4.0)
                          for y in (-4.0, 4.0) for z in (0.0, 3.0)])
    anchors = anchor_set[rng.integers(0, len(anchor_set), TRANSACTIONS)]
    bearing = rng.uniform(-np.pi, np.pi)
    position = np.array([30*np.cos(bearing), 30*np.sin(bearing), 1.2])
    velocity = rng.uniform(-0.5, 0.5, 3)
    yaw = rng.uniform(-np.pi, np.pi)
    grid = np.arange(round(DURATION/IMU_DT)+1)*IMU_DT
    tau = grid[:-1]-DURATION
    # Nonquadratic world acceleration uses the original excited trajectory derivatives.
    acceleration = np.column_stack((-0.8*1.7**2*np.sin(1.7*tau),
        -0.6*1.1**2*np.cos(1.1*tau), -0.15*0.7**2*np.sin(0.7*tau)))
    # A physical, time-varying roll/pitch attitude; no supplied yaw information.
    angles = np.column_stack((np.deg2rad(10)*np.sin(0.9*grid),
                             np.deg2rad(8)*np.sin(1.3*grid), np.zeros(len(grid))))
    q_true = Rotation.from_euler("xyz", angles).as_matrix()
    increments = np.einsum("nji,njk->nik", q_true[:-1], q_true[1:])
    gyro = Rotation.from_matrix(increments).as_rotvec()/IMU_DT
    specific_force = np.einsum("nji,nj->ni", q_true[:-1], acceleration-GRAVITY)
    true_displacement = integrate(acceleration, event_times)
    positions = (position + (event_times-DURATION)[:, None]*velocity
                 + true_displacement@initializer.yaw_rotation(yaw).T)
    ranges = np.linalg.norm(positions-anchors, axis=1)+rng.normal(0, RANGE_SIGMA, TRANSACTIONS)
    return dict(event_times=event_times, anchors=anchors, ranges=ranges,
        position=position, velocity=velocity, yaw=yaw, q_true=q_true,
        gyro=gyro, specific_force=specific_force, true_displacement=true_displacement,
        accel_bias_standard=rng.normal(size=3), gyro_bias_standard=rng.normal(size=3),
        tilt_standard=rng.normal(size=2))


def measured_shape(data, accel_std, gyro_std, tilt_std_deg):
    """Propagate biased gyro/force with an initial roll/pitch error and no correction."""
    accel_bias = accel_std*data["accel_bias_standard"]
    gyro_bias = gyro_std*data["gyro_bias_standard"]
    tilt = np.r_[np.deg2rad(tilt_std_deg)*data["tilt_standard"], 0.0]
    q_est = data["q_true"][0]@Rotation.from_rotvec(tilt).as_matrix()
    rotations = Rotation.from_rotvec((data["gyro"]+gyro_bias)*IMU_DT).as_matrix()
    acceleration = np.empty_like(data["specific_force"])
    for index, rotation in enumerate(rotations):
        acceleration[index] = q_est@(data["specific_force"][index]+accel_bias)+GRAVITY
        q_est = q_est@rotation
    return integrate(acceleration, data["event_times"]), q_est


def counted_solve(values, method):
    """Count both common CV fits and every raw-range refinement without source edits."""
    original, counts = initializer.least_squares, []
    def counted(*args, **kwargs):
        """Record one completed SciPy solve, including unsuccessful termination."""
        result = original(*args, **kwargs)
        counts.append(int(result.nfev))
        return result
    initializer.least_squares = counted
    try:
        fit = initializer.solve_window(*values, method=method)
    finally:
        initializer.least_squares = original
    return fit, sum(counts), sum(counts[:2])


def evaluate_case(specification):
    """Run all nine fits on shared truth/ranges and keep all failure outcomes."""
    case_id, seed = specification
    data = physical_case(seed)
    tau = data["event_times"]-DURATION
    true_rotation = initializer.yaw_rotation(data["yaw"])@data["q_true"][-1]
    true_heading = np.arctan2(true_rotation[1, 0], true_rotation[0, 0])
    rows = []
    for level, accel_std, gyro_std, tilt_std in LEVELS:
        displacement, q_end = measured_shape(data, accel_std, gyro_std, tilt_std)
        shape_error = np.linalg.norm(displacement-data["true_displacement"], axis=1)
        if level == "ideal":
            np.testing.assert_allclose(displacement, data["true_displacement"], atol=1e-10, rtol=0)
            np.testing.assert_allclose(q_end, data["q_true"][-1], atol=1e-12, rtol=0)
        values = (tau, displacement, data["anchors"], data["ranges"],
                  np.full(TRANSACTIONS, RANGE_SIGMA))
        group = []
        # Balanced deterministic order avoids always timing one method first.
        for method in np.roll(np.array(METHODS), case_id % len(METHODS)):
            row = dict(case_id=case_id, seed=seed, level=level, method=str(method),
                displacement_error_max_m=float(shape_error.max()),
                displacement_error_rms_m=float(np.sqrt(np.mean(shape_error**2))),
                accel_bias_norm=float(accel_std*np.linalg.norm(data["accel_bias_standard"])),
                gyro_bias_norm=float(gyro_std*np.linalg.norm(data["gyro_bias_standard"])),
                initial_tilt_norm_deg=float(tilt_std*np.linalg.norm(data["tilt_standard"])))
            try:
                fit, total_nfev, cv_nfev = counted_solve(values, str(method))
                estimated_rotation = initializer.yaw_rotation(fit.parameters[6])@q_end
                estimated_heading = np.arctan2(estimated_rotation[1, 0], estimated_rotation[0, 0])
                heading_error = np.arctan2(np.sin(estimated_heading-true_heading),
                                          np.cos(estimated_heading-true_heading))
                parameter_error = np.arctan2(np.sin(fit.parameters[6]-data["yaw"]),
                                            np.cos(fit.parameters[6]-data["yaw"]))
                row.update(position_error_m=float(np.linalg.norm(fit.parameters[:3]-data["position"])),
                    velocity_error_mps=float(np.linalg.norm(fit.parameters[3:6]-data["velocity"])),
                    yaw_error_deg=float(abs(np.rad2deg(heading_error))),
                    yaw_parameter_error_deg=float(abs(np.rad2deg(parameter_error))),
                    attitude_error_deg=float(np.rad2deg(Rotation.from_matrix(
                        estimated_rotation.T@true_rotation).magnitude())),
                    robust_objective=fit.robust_objective, elapsed_s=fit.elapsed_s,
                    total_nfev=total_nfev, cv_nfev=cv_nfev,
                    solver_success=bool(fit.diagnostics.get("solver_success", False)),
                    finite=bool(np.isfinite(fit.parameters).all()), error="")
            except Exception as error:
                row.update({name: float("inf") for name in ("position_error_m", "velocity_error_mps",
                    "yaw_error_deg", "yaw_parameter_error_deg", "attitude_error_deg", "robust_objective")})
                row.update(elapsed_s=float("nan"), total_nfev=0, cv_nfev=0,
                           solver_success=False, finite=False, error=repr(error))
            group.append(row)
        reference = next(r["robust_objective"] for r in group if r["method"] == "multistart")
        tolerance = max(OBJECTIVE_ABS_TOL, OBJECTIVE_REL_TOL*abs(reference))
        for row in group:
            row["reference_finite"] = bool(np.isfinite(reference))
            row["objective_gap_vs_multistart"] = row["robust_objective"]-reference
            row["missed_multistart"] = (not row["finite"] or
                (np.isfinite(reference) and row["objective_gap_vs_multistart"]>tolerance))
            row["better_than_multistart"] = bool(np.isfinite(reference) and
                                                 row["objective_gap_vs_multistart"] < -tolerance)
        rows.extend(group)
    return rows


def summarize(frame):
    """Report every error level, truth-error tails, solver failures, and full cost."""
    results = []
    for (level, method), subset in frame.groupby(["level", "method"], sort=False):
        row = dict(level=level, method=method, cases=len(subset),
            objective_misses=int(subset["missed_multistart"].sum()),
            better_than_multistart=int(subset["better_than_multistart"].sum()),
            nonfinite=int((~subset["finite"]).sum()),
            nonconverged=int((~subset["solver_success"]).sum()),
            unavailable_reference=int((~subset["reference_finite"]).sum()),
            position_over_half_m=int((subset["position_error_m"]>0.5).sum()),
            yaw_over_035rad=int((subset["yaw_error_deg"]>np.rad2deg(0.35)).sum()),
            total_nfev=int(subset["total_nfev"].sum()))
        for name in ("position_error_m", "velocity_error_mps", "yaw_error_deg",
                     "yaw_parameter_error_deg", "attitude_error_deg", "displacement_error_max_m",
                     "elapsed_s", "total_nfev"):
            row[name+"_median"] = float(subset[name].median())
            row[name+"_p95"] = float(subset[name].quantile(0.95))
            row[name+"_max"] = float(subset[name].max())
        results.append(row)
    return pd.DataFrame(results)


def audit(output, workers):
    """Independently rerun twenty predetermined cases in fresh worker processes."""
    protocol = json.loads((output/"protocol.json").read_text(encoding="utf-8"))
    saved = pd.read_csv(output/"metrics.csv").sort_values(["case_id", "level", "method"])
    cases = [protocol["cases"][index] for index in protocol["audit_case_ids"]]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rerun = pd.DataFrame([row for group in pool.map(evaluate_case, cases) for row in group])
    rerun = rerun.sort_values(["case_id", "level", "method"]).reset_index(drop=True)
    subset = saved[saved.case_id.isin(protocol["audit_case_ids"])].reset_index(drop=True)
    columns = [name for name in rerun.select_dtypes(include=np.number).columns if name != "elapsed_s"]
    differences = {}
    for name in columns:
        np.testing.assert_allclose(rerun[name], subset[name], rtol=1e-12, atol=1e-11)
        differences[name] = float(np.max(np.abs(rerun[name]-subset[name])))
    for name in ("finite", "solver_success", "missed_multistart", "better_than_multistart"):
        np.testing.assert_array_equal(rerun[name], subset[name])
    summary = pd.read_csv(output/"summary.csv")
    rebuilt = summarize(saved).set_index(["level", "method"])
    summary = summary.set_index(["level", "method"]).loc[rebuilt.index]
    np.testing.assert_allclose(summary.to_numpy(float), rebuilt.to_numpy(float), rtol=1e-12, atol=1e-11)
    result = dict(cases=len(cases), rerun_fits=len(rerun), summary_groups=len(summary),
        maximum_differences=differences, zero_error_shape_max_m=float(
            saved.loc[saved.level=="ideal", "displacement_error_max_m"].max()),
        all_outputs_finite=bool(saved.finite.all()),
        completed_utc=datetime.now(timezone.utc).isoformat())
    (output/"audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


def main():
    """Write the complete fixed protocol before dispatching any estimator fits."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=200)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--output", type=Path, default=HERE.parents[1]/"results/recovery_imu_mismatch_20261107")
    args = parser.parse_args()
    if args.audit:
        audit(args.output, args.workers)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/"metrics.csv").exists():
        raise FileExistsError("Completed results will not be overwritten")
    rng = np.random.default_rng(MASTER_SEED)
    cases = [(index, int(rng.integers(0, 2**32))) for index in range(args.seeds)]
    protocol = dict(created_utc=datetime.now(timezone.utc).isoformat(), master_seed=MASTER_SEED,
        cases=cases, audit_case_ids=rng.choice(args.seeds, min(20, args.seeds), replace=False).tolist(),
        methods=METHODS, levels=LEVELS, workers=args.workers, blas_threads=1,
        imu_dt_s=IMU_DT, duration_s=DURATION, transactions=TRANSACTIONS, range_sigma_m=RANGE_SIGMA,
        geometry="30 m horizontal stand-off; endpoint height 1.2 m; eight (+/-4,+/-4,0/3) anchors",
        motion="Discrete held acceleration from the original excited shape second derivative",
        attitude="Q=Euler_xyz(10deg*sin(.9t),8deg*sin(1.3t),0); R=Rz(uniform yaw)*Q",
        angular_rate="log(Q[k]^T Q[k+1])/dt, exactly matching the discrete propagator",
        force="Q[k]^T(a[k]-g), then add constant per-axis Gaussian accelerometer bias",
        initial_error="Qhat[0]=Qtrue[0]*Exp([tilt_x,tilt_y,0]); each tilt component independent Gaussian",
        perturbations="Same unit Gaussian bias and tilt draws scaled across levels; no measurement white noise",
        pairing="Each seed shares truth trajectory, timestamps, anchors, and noisy ranges across all nine fits",
        reanchoring="d(t)=P(t)-P(T)-(t-T)V(T) with exact held-acceleration interpolation",
        yaw_metric="Physical endpoint heading atan2(R[1,0],R[0,0]); scalar yaw-parameter error also retained",
        objective_tolerance="J-J12 > max(1e-3,1e-5*abs(J12)); finite reference, not a global certificate",
        interpretation="Combined three-source mismatch levels, not isolated causal ablations; development simulation",
        source_sha256={path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in (Path(__file__), HERE/"initializer.py")})
    (args.output/"protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    started, rows = time.perf_counter(), []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(evaluate_case, case) for case in cases]
        for completed, future in enumerate(as_completed(futures), 1):
            rows.extend(future.result())
            if completed % 25 == 0 or completed == len(cases):
                print(f"Completed {completed}/{len(cases)} paired physical cases in {time.perf_counter()-started:.1f}s", flush=True)
    frame = pd.DataFrame(rows).sort_values(["case_id", "level", "method"])
    frame.to_csv(args.output/"metrics.csv", index=False)
    summary = summarize(frame)
    summary.to_csv(args.output/"summary.csv", index=False)
    protocol["completed_utc"] = datetime.now(timezone.utc).isoformat()
    protocol["elapsed_s"] = time.perf_counter()-started
    (args.output/"protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(summary[["level", "method", "cases", "objective_misses", "position_error_m_p95", "yaw_error_deg_p95"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
