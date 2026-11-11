"""Paired K1/K2 controls and exact held-IMU bias/initial-tilt shape models.

Synthetic calibration stress is development evidence, not independent validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

# Bound numerical-library concurrency for reproducible desktop experiments.
for _name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

# Import frozen source implementations without reading historical result files.
HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1] / "recovery_bootstrap"
sys.path.insert(0, str(BASE))
import initializer as init
import polynomial_bias as poly
import run_imu_mismatch_stress as sim

# Priors are fixed across all truth perturbation levels and are not fitted.
BIAS_STD = 0.10
TILT_STD = np.deg2rad(2.0)
# Gravity and diagnostic heading tolerance follow the paper's convention.
GRAVITY = np.array([0.0, 0.0, -9.81])
YAW_TOL = 0.35


def skew(vector):
    """Return the cross-product matrix of a three-vector."""
    x, y, z = vector
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def right_jacobian(vector):
    """Differentiate an SO(3) exponential in its right tangent chart."""
    theta = np.linalg.norm(vector)
    cross = skew(vector)
    if theta < 1e-5:
        return np.eye(3) - 0.5 * cross + cross @ cross / 6.0
    return (np.eye(3) - (1 - np.cos(theta)) / theta**2 * cross
            + (theta - np.sin(theta)) / theta**3 * cross @ cross)


def reanchor_integral(acceleration, grid, events):
    """Double-integrate held vector or matrix samples and anchor at final time."""
    dt = np.diff(grid)
    expand = (slice(None),) + (None,) * (acceleration.ndim - 1)
    velocity = np.concatenate((np.zeros_like(acceleration[:1]),
                              np.cumsum(acceleration * dt[expand], axis=0)))
    position = np.concatenate((np.zeros_like(acceleration[:1]), np.cumsum(
        velocity[:-1] * dt[expand] + 0.5 * acceleration * dt[expand]**2, axis=0)))
    index = np.minimum(np.searchsorted(grid, events, side="right") - 1, len(dt) - 1)
    offset = (events - grid[index])[expand]
    tau = (events - grid[-1])[expand]
    return position[index] + offset * velocity[index] + 0.5 * offset**2 * acceleration[index] - position[-1] - tau * velocity[-1]


def measured_physics(data, accel_std, gyro_std, tilt_std):
    """Construct measured shape and its exact constant-body-bias sensitivity."""
    bias = accel_std * data["accel_bias_standard"]
    gyro_bias = gyro_std * data["gyro_bias_standard"]
    tilt = np.r_[np.deg2rad(tilt_std) * data["tilt_standard"], 0.0]
    current = data["q_true"][0] @ Rotation.from_rotvec(tilt).as_matrix()
    increments = Rotation.from_rotvec((data["gyro"] + gyro_bias) * sim.IMU_DT).as_matrix()
    rotations, acceleration = [], []
    for index, increment in enumerate(increments):
        rotations.append(current)
        acceleration.append(current @ (data["specific_force"][index] + bias) + GRAVITY)
        current = current @ increment
    grid = np.arange(len(increments) + 1) * sim.IMU_DT
    displacement = reanchor_integral(np.asarray(acceleration), grid, data["event_times"])
    sensitivity = reanchor_integral(np.asarray(rotations), grid, data["event_times"])
    return displacement, sensitivity, current


def raw_model(x, values, sensitivity, tilt_enabled):
    """Evaluate metric ranges and exact bias/tilt/yaw Jacobians."""
    tau, displacement, anchors, ranges, sigma = values
    gravity_shape = 0.5 * tau[:, None]**2 * GRAVITY
    force_shape = displacement - np.einsum("nij,j->ni", sensitivity, x[7:10]) - gravity_shape
    epsilon = np.r_[x[10:12], 0.0] if tilt_enabled else np.zeros(3)
    tilt_rotation = Rotation.from_rotvec(epsilon).as_matrix()
    yaw_rotation = init.yaw_rotation(x[6])
    corrected = force_shape @ tilt_rotation.T + gravity_shape
    rotated = corrected @ yaw_rotation.T
    position = x[:3] + tau[:, None] * x[3:6] + rotated
    delta = position - anchors
    distance = np.maximum(np.linalg.norm(delta, axis=1), 1e-12)
    direction = delta / distance[:, None]
    dyaw = np.column_stack((-rotated[:, 1], rotated[:, 0], np.zeros(len(tau))))
    combined = yaw_rotation @ tilt_rotation
    dbias = -np.einsum("ni,ij,njk->nk", direction, combined, sensitivity)
    columns = [direction, tau[:, None] * direction, np.sum(direction * dyaw, axis=1)[:, None], dbias]
    if tilt_enabled:
        # E exp(Jr(epsilon) delta) gives the exact exponential-coordinate derivative.
        local_direction = direction @ combined
        columns.append(np.cross(force_shape, local_direction) @ right_jacobian(epsilon)[:, :2])
    return (distance - ranges) / sigma, np.column_stack(columns) / sigma[:, None]


def mixed_loss(z, count):
    """Apply Huber to range errors and Gaussian quadratic loss to priors."""
    result = np.vstack((z.copy(), np.ones_like(z), np.zeros_like(z)))
    indices = np.flatnonzero((np.arange(len(z)) < count) & (z > init.HUBER_SCALE**2))
    root = np.sqrt(z[indices])
    result[0, indices] = 2 * init.HUBER_SCALE * root - init.HUBER_SCALE**2
    result[1, indices] = init.HUBER_SCALE / root
    result[2, indices] = -0.5 * init.HUBER_SCALE / root**3
    return result


def solve(values, sensitivity, model, use_prior=True):
    """Use one lifted yaw and two translation starts for every motion model."""
    started = time.perf_counter()
    degree = 2 if model == "K2" else 1
    lifted, _ = poly.polynomial_lift(*values, degree=degree)
    cv, _, _ = init.cv_candidate(values[0], *values[2:])
    tilt_enabled = model == "BiasTilt"
    physical = model in ("Bias", "BiasTilt")
    seeds = [lifted, np.r_[cv, np.zeros(3 * (degree - 1)), lifted[-1]]]
    if physical:
        extra = 5 if tilt_enabled else 3
        seeds = [np.r_[candidate, np.zeros(extra)] for candidate in seeds]
    size = len(seeds[0])
    prior = np.zeros((size - 7 if physical and use_prior else 0, size))
    if len(prior):
        std = np.r_[np.full(3, BIAS_STD), np.full(2 if tilt_enabled else 0, TILT_STD)]
        prior[:, 7:] = np.diag(1 / std)
    def raw(x):
        """Evaluate the selected raw-range motion model."""
        return (raw_model(x, values, sensitivity, tilt_enabled) if physical else
                poly.polynomial_residual_jacobian(x, *values, degree=degree))
    def fun(x):
        """Append independent Gaussian nuisance calibration priors."""
        return np.r_[raw(x)[0], prior @ x]
    def jac(x):
        """Append constant prior derivatives to range Jacobians."""
        return np.vstack((raw(x)[1], prior))
    fits = [least_squares(fun, seed, jac=jac, loss=lambda z: mixed_loss(z, len(values[0])),
                          max_nfev=120, ftol=1e-8, xtol=1e-8, gtol=1e-8) for seed in seeds]
    best = min(fits, key=lambda item: item.cost)
    residual, derivative = raw(best.x)
    weighted = derivative * np.sqrt(np.minimum(1.0, 2.0 / np.maximum(np.abs(residual), 1e-12)))[:, None]
    yaw_index = 6 if physical else size - 1
    nuisance = [index for index in range(size) if index != yaw_index]
    projected, _ = init.project_out(weighted[:, nuisance], weighted[:, yaw_index: yaw_index + 1])
    augmented = np.vstack((weighted, prior))
    conditioned, _ = init.project_out(augmented[:, nuisance], augmented[:, yaw_index: yaw_index + 1])
    return dict(x=best.x, yaw_index=yaw_index, objective=init.robust_cost(residual),
                penalized_objective=float(2 * best.cost), information=float(np.sum(projected**2)),
                prior_information=float(np.sum(conditioned**2)), nfev=sum(item.nfev for item in fits),
                elapsed_s=time.perf_counter() - started, success=bool(best.success),
                nuisance_norm=float(np.linalg.norm(best.x[7:10])) if physical else 0.0)


def evaluate(seed, levels):
    """Pair every model on identical rotating IMU and range samples."""
    data = sim.physical_case(seed)
    truth_rotation = init.yaw_rotation(data["yaw"]) @ data["q_true"][-1]
    truth_heading = np.arctan2(truth_rotation[1, 0], truth_rotation[0, 0])
    rows = []
    for level, accel, gyro, tilt in levels:
        displacement, sensitivity, qend = measured_physics(data, accel, gyro, tilt)
        values = (data["event_times"] - sim.DURATION, displacement, data["anchors"],
                  data["ranges"], np.full(sim.TRANSACTIONS, sim.RANGE_SIGMA))
        for model, prior in (("K1", False), ("K2", False), ("Bias", True),
                             ("BiasTilt", True), ("BiasTilt", False)):
            fit = solve(values, sensitivity, model, prior)
            x = fit.pop("x")
            yaw = x[fit.pop("yaw_index")]
            correction = Rotation.from_rotvec(np.r_[x[10:12], 0.0]).as_matrix() if model == "BiasTilt" else np.eye(3)
            rotation = init.yaw_rotation(yaw) @ correction @ qend
            heading = np.arctan2(rotation[1, 0], rotation[0, 0])
            heading_error = abs(np.arctan2(np.sin(heading - truth_heading), np.cos(heading - truth_heading)))
            rows.append(dict(seed=seed, level=level, method=model + ("_prior" if prior else ""),
                position_error_m=float(np.linalg.norm(x[:3] - data["position"])),
                velocity_error_mps=float(np.linalg.norm(x[3:6] - data["velocity"])),
                yaw_error_deg=float(np.rad2deg(heading_error)),
                attitude_error_deg=float(np.rad2deg(Rotation.from_matrix(rotation.T @ truth_rotation).magnitude())),
                false_confident=bool(fit["information"] >= 1 / YAW_TOL**2 and heading_error > YAW_TOL), **fit))
    return rows


def verify():
    """Check analytic derivatives and exact bias/initial-tilt shape cancellation."""
    data = sim.physical_case(20261111)
    d, b, _ = measured_physics(data, 0.1, 0.0, 2.0)
    tau = data["event_times"] - sim.DURATION
    correction = Rotation.from_rotvec(-np.r_[np.deg2rad(2.0) * data["tilt_standard"], 0.0]).as_matrix()
    corrected = (d - np.einsum("nij,j->ni", b, .1 * data["accel_bias_standard"])
                 - .5 * tau[:, None]**2 * GRAVITY) @ correction.T + .5 * tau[:, None]**2 * GRAVITY
    exact_error = float(np.max(np.abs(corrected - data["true_displacement"])))
    np.testing.assert_allclose(corrected, data["true_displacement"], atol=2e-10, rtol=0)
    values = (tau, d, data["anchors"], data["ranges"], np.full(len(tau), .1))
    x = np.r_[data["position"], data["velocity"], data["yaw"], [.02, -.03, .01], [.02, -.03]]
    analytic = raw_model(x, values, b, True)[1]
    numeric = np.empty_like(analytic)
    for index in range(len(x)):
        step = np.eye(len(x))[index] * 1e-6
        numeric[:, index] = (raw_model(x + step, values, b, True)[0] - raw_model(x - step, values, b, True)[0]) / 2e-6
    error = float(np.max(np.abs(analytic - numeric)))
    np.testing.assert_allclose(analytic, numeric, atol=1e-6, rtol=1e-5)
    bad_d, bad_b, _ = measured_physics(data, .1, .01, 2.)
    bad_corrected = (bad_d - np.einsum("nij,j->ni", bad_b, .1 * data["accel_bias_standard"])
                     - .5 * tau[:, None]**2 * GRAVITY) @ correction.T + .5 * tau[:, None]**2 * GRAVITY
    gyro_rms = float(np.sqrt(np.mean(np.sum((bad_corrected-data["true_displacement"])**2, axis=1))))
    # Constant attitude/force makes yaw exactly absorbable by free body bias.
    constant_d = .5 * tau[:, None]**2 * np.array([.4, -.3, .1])
    constant_b = .5 * tau[:, None, None]**2 * np.eye(3)
    constant_values = (tau, constant_d, *values[2:])
    constant_jac = raw_model(x, constant_values, constant_b, False)[1]
    nuisance = np.r_[0:6, 7:10]
    projected, _ = init.project_out(constant_jac[:, nuisance], constant_jac[:, 6:7])
    constant_info = float(np.sum(projected**2))
    assert constant_info < 1e-15
    return dict(exact_bias_tilt_shape_max_abs_m=exact_error, analytic_jacobian_max_abs=error,
                oracle_bias_initial_tilt_correction_gyro_mismatch_rms_m=gyro_rms,
                constant_attitude_free_bias_yaw_information=constant_info)


def summarize(frame):
    """Summarize unconditional paired errors and data/prior information separately."""
    rows = []
    for (level, method), part in frame.groupby(["level", "method"]):
        row = dict(level=level, method=method, cases=len(part), nonconverged=int((~part.success).sum()),
                   false_confident=int(part.false_confident.sum()))
        for metric in ("position_error_m", "velocity_error_mps", "yaw_error_deg", "attitude_error_deg",
                       "information", "prior_information", "elapsed_s", "nfev"):
            row[metric + "_median"] = float(part[metric].median())
            row[metric + "_p95"] = float(part[metric].quantile(.95))
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    """Run bounded paired synthetic experiments and retain reproducible results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=40)
    parser.add_argument("--paper-seeds", action="store_true")
    parser.add_argument("--output", type=Path, default=HERE / "results")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    checks = verify()
    # Isolated controls expose whether improvement comes from bias, tilt, or both.
    levels = sim.LEVELS + (("bias_only", .1, 0., 0.), ("tilt_only", 0., 0., 2.))
    rows = []
    generator = np.random.default_rng(sim.MASTER_SEED)
    seeds = ([int(generator.integers(0, 2**32)) for _ in range(args.seeds)] if args.paper_seeds
             else [sim.MASTER_SEED + index for index in range(args.seeds)])
    for index, seed in enumerate(seeds):
        rows.extend(evaluate(seed, levels))
        if (index + 1) % 5 == 0:
            print(f"Completed {index + 1}/{args.seeds} paired seeds", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output / "synthetic_metrics.csv", index=False)
    summary = summarize(frame)
    summary.to_csv(args.output / "synthetic_summary.csv", index=False)
    metadata = dict(checks=checks, seeds=args.seeds, case_seeds=seeds, paper_seed_protocol=args.paper_seeds,
        levels=levels, bias_prior_std_mps2=BIAS_STD, tilt_prior_std_deg=float(np.rad2deg(TILT_STD)),
        scope="Development simulation; no IMU white noise; combined gyro mismatch is unmodeled",
        seeding="Each model one lifted yaw, two translation seeds (lift and common CV); physical model uses K1 lift",
        timing="Sequential desktop including candidate and CV; bias sensitivity construction excluded; nfev excludes common CV",
        source_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in
                       (Path(__file__), BASE / "initializer.py", BASE / "polynomial_bias.py", BASE / "run_imu_mismatch_stress.py")})
    (args.output / "protocol.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(summary[["level", "method", "position_error_m_p95", "yaw_error_deg_p95", "nonconverged"]].to_string(index=False))


if __name__ == "__main__":
    main()
