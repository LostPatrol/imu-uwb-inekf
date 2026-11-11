"""Truth-free dynamic TDoA capture with a projected inertial yaw candidate.

Fixed acquisition windows precede a common white-variance invariant tracker.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp

# Parent module supplies the unchanged real-data loader and common navigation math.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from study import load, LEVER, score, coordinate_map, group_exp, skew, Invariant15, rotation_terms

# Fixed robust noise scale matches the previous strong white-variance control.
RADIO_VARIANCE = .14
# Weak accelerometer bias prior is common to all capture policies.
ACCEL_BIAS_STD = .2
# The yaw grid has eight evenly spaced candidates before seeing any truth.
UNIFORM_STARTS = 8


def integrate_shape(data, start, end, gyro_calibration=False):
    """Propagate a gravity-aligned IMU shape and exact rotating-body bias sensitivity."""
    times, imu = data["times"], data["imu"]
    prefix = imu[times <= times[0] + 2]
    vertical = np.mean(prefix[:, :3], axis=0)
    vertical /= np.linalg.norm(vertical)
    stationary = (np.max(np.abs(np.linalg.norm(prefix[:, :3], axis=1) - 1)) < .05
                  and np.max(np.linalg.norm(prefix[:, 3:], axis=1)) < 2.)
    gyro_bias = np.radians(np.mean(prefix[:, 3:], axis=0)) if gyro_calibration and stationary else np.zeros(3)
    rotation = Rotation.align_vectors([[0., 0., 1.]], vertical[None])[0].as_matrix()
    initial_index = int(np.searchsorted(times, start))
    final_index = int(np.searchsorted(times, end))
    for k in range(1, initial_index + 1):
        dt = times[k] - times[k - 1]
        omega = np.radians(.5 * (imu[k - 1, 3:] + imu[k, 3:])) - gyro_bias
        rotation = rotation @ rotation_terms(omega * dt)[0]
    position, velocity = np.zeros(3), np.zeros(3)
    sensitivity, sensitivity_velocity = np.zeros((3, 3)), np.zeros((3, 3))
    rows, rotations, matrices = [], [], []
    for k in range(initial_index, final_index + 1):
        rows.append(np.r_[times[k], position, velocity])
        rotations.append(rotation.copy())
        matrices.append(sensitivity.copy())
        if k == final_index:
            break
        dt = times[k + 1] - times[k]
        sample = .5 * (imu[k] + imu[k + 1])
        acceleration = rotation @ (9.81 * sample[:3]) + [0., 0., -9.81]
        position += velocity * dt + .5 * acceleration * dt**2
        velocity += acceleration * dt
        sensitivity += sensitivity_velocity * dt + .5 * rotation * dt**2
        sensitivity_velocity += rotation * dt
        rotation = rotation @ rotation_terms((np.radians(sample[3:]) - gyro_bias) * dt)[0]
    rows, rotations, matrices = np.array(rows), np.array(rotations), np.array(matrices)
    tau = rows[:, 0] - rows[-1, 0]
    displacement = rows[:, 1:4] - rows[-1, 1:4] - tau[:, None] * rows[-1, 4:7]
    matrices -= matrices[-1].copy() + tau[:, None, None] * sensitivity_velocity
    lever = np.einsum("nij,j->ni", rotations, LEVER)
    return dict(times=rows[:, 0], shape=displacement + lever, sensitivity=matrices,
                endpoint_rotation=rotations[-1], endpoint_lever=lever[-1], endpoint=rows[-1, 0],
                gyro_bias=gyro_bias, stationary_prefix=stationary)


def interpolate_shape(shape, stamps):
    """Interpolate the small sampled trajectory model at actual radio timestamps."""
    values = np.column_stack([np.interp(stamps, shape["times"], shape["shape"][:, j]) for j in range(3)])
    matrices = np.stack([np.interp(stamps, shape["times"], shape["sensitivity"][:, j, k])
                         for j in range(3) for k in range(3)], axis=1).reshape(-1, 3, 3)
    return values, matrices


def radio_cv(observations, anchors, endpoint):
    """Fit a common radio-only constant-velocity antenna seed without inertial yaw."""
    tau = observations[:, 0] - endpoint
    a, b = observations[:, 1:3].astype(int).T
    def residual(parameters):
        """Evaluate antenna hyperbolas under the seed motion model."""
        tag = parameters[:3] + tau[:, None] * parameters[3:]
        return (np.linalg.norm(tag - anchors[b], axis=1) - np.linalg.norm(tag - anchors[a], axis=1) - observations[:, 3]) / np.sqrt(RADIO_VARIANCE)
    fit = least_squares(residual, np.r_[np.mean(anchors, axis=0), np.zeros(3)], loss="huber", f_scale=2, max_nfev=100)
    return fit.x, fit.nfev


def projected_yaw(observations, anchors, shape):
    """Align independently radio-localized antenna positions with affine-profiled IMU shape."""
    stamps, positions, evaluations = [], [], 0
    seed = np.mean(anchors, axis=0)
    # Fixed 200 ms bins supply sufficient distinct anchor transactions.
    bins = np.arange(shape["times"][0], shape["endpoint"], .2)
    for stamp in bins:
        chunk = observations[(observations[:, 0] >= stamp) & (observations[:, 0] < stamp + .2)]
        if len(chunk) < 10:
            continue
        a, b = chunk[:, 1:3].astype(int).T
        def residual(position):
            """Localize a radio-only short-bin antenna center."""
            return (np.linalg.norm(position - anchors[b], axis=1) - np.linalg.norm(position - anchors[a], axis=1) - chunk[:, 3]) / np.sqrt(RADIO_VARIANCE)
        fit = least_squares(residual, seed, loss="huber", f_scale=2, max_nfev=50)
        seed = fit.x
        positions.append(seed.copy())
        stamps.append(np.mean(chunk[:, 0]))
        evaluations += fit.nfev
    stamps, positions = np.array(stamps), np.array(positions)
    inertial, _ = interpolate_shape(shape, stamps)
    nuisance = np.column_stack([np.ones(len(stamps)), stamps - shape["endpoint"]])
    # Project unknown endpoint position and velocity before the 2D Procrustes solve.
    p = positions - nuisance @ np.linalg.lstsq(nuisance, positions, rcond=None)[0]
    d = inertial - nuisance @ np.linalg.lstsq(nuisance, inertial, rcond=None)[0]
    cosine = np.sum(d[:, 0] * p[:, 0] + d[:, 1] * p[:, 1])
    sine = np.sum(d[:, 0] * p[:, 1] - d[:, 1] * p[:, 0])
    return float(np.arctan2(sine, cosine)), float(np.sum(d[:, :2] ** 2)), evaluations, float(np.sum(p[:, :2] ** 2))


def capture(data, start=10., length=6., bias_enabled=False, gyro_calibration=False, tilt_enabled=False, check_only=False):
    """Fit identical raw-TDoA windows from zero, projected, and uniform yaw starts."""
    origin = data["times"][0]
    shape = integrate_shape(data, origin + start, origin + start + length, gyro_calibration)
    observations = data["radio"][(data["radio"][:, 0] >= shape["times"][0]) & (data["radio"][:, 0] <= shape["endpoint"])]
    anchors = data["anchors"]
    tau = observations[:, 0] - shape["endpoint"]
    inertial, sensitivity = interpolate_shape(shape, observations[:, 0])
    a, b = observations[:, 1:3].astype(int).T
    preprocessing_start = time.perf_counter()
    cv, cv_evaluations = radio_cv(observations, anchors, shape["endpoint"])
    cv_runtime = time.perf_counter() - preprocessing_start
    preprocessing_start = time.perf_counter()
    candidate, excitation, candidate_evaluations, radio_excitation = projected_yaw(observations, anchors, shape)
    candidate_runtime = time.perf_counter() - preprocessing_start
    gravity_shape = .5 * tau[:, None]**2 * np.array([0., 0., -9.81])
    def residual_and_jacobian(parameters, fixed_yaw=None):
        """Evaluate raw distance differences and analytic p/v/yaw/bias derivatives."""
        yaw = parameters[6] if fixed_yaw is None else fixed_yaw
        yaw_rotation = Rotation.from_euler("z", yaw).as_matrix()
        tilt = parameters[10:12] if tilt_enabled else np.zeros(2)
        tilt_rotation = Rotation.from_rotvec(np.r_[tilt, 0.]).as_matrix()
        rotation = yaw_rotation @ tilt_rotation
        accel_bias = parameters[7:10] if bias_enabled else np.zeros(3)
        body_shape = inertial - gravity_shape - np.einsum("nij,j->ni", sensitivity, accel_bias)
        rotated = body_shape @ rotation.T
        tag = parameters[:3] + tau[:, None] * parameters[3:6] + rotated + gravity_shape
        da, db = tag - anchors[a], tag - anchors[b]
        ra, rb = np.linalg.norm(da, axis=1), np.linalg.norm(db, axis=1)
        gradient = db / rb[:, None] - da / ra[:, None]
        sigma = np.sqrt(RADIO_VARIANCE)
        residual = (rb - ra - observations[:, 3]) / sigma
        jacobian = np.zeros((len(observations), len(parameters)))
        jacobian[:, :3] = gradient / sigma
        jacobian[:, 3:6] = tau[:, None] * gradient / sigma
        if fixed_yaw is None:
            jacobian[:, 6] = np.sum(gradient * np.cross([0., 0., 1.], rotated), axis=1) / sigma
        if bias_enabled:
            jacobian[:, 7:10] = -np.einsum("ni,ij,njk->nk", gradient, rotation, sensitivity) / sigma
        if tilt_enabled:
            # Two small attitude-prior derivatives use central matrix differences.
            for j in range(2):
                direction = np.zeros(2)
                direction[j] = 1e-6
                plus = yaw_rotation @ Rotation.from_rotvec(np.r_[tilt + direction, 0.]).as_matrix()
                minus = yaw_rotation @ Rotation.from_rotvec(np.r_[tilt - direction, 0.]).as_matrix()
                derivative = body_shape @ ((plus - minus) / 2e-6).T
                jacobian[:, 10 + j] = np.sum(gradient * derivative, axis=1) / sigma
        if bias_enabled:
            prior_jacobian = np.zeros((3, len(parameters)))
            prior_jacobian[:, 7:10] = np.eye(3) / ACCEL_BIAS_STD
            residual = np.r_[residual, accel_bias / ACCEL_BIAS_STD]
            jacobian = np.vstack([jacobian, prior_jacobian])
        if tilt_enabled:
            prior_jacobian = np.zeros((2, len(parameters)))
            prior_jacobian[:, 10:12] = np.eye(2) / .08
            residual = np.r_[residual, tilt / .08]
            jacobian = np.vstack([jacobian, prior_jacobian])
        return residual, jacobian
    if check_only:
        parameters = np.r_[cv, .3, .01, -.02, .03] if bias_enabled else np.r_[cv, .3]
        if tilt_enabled:
            parameters = np.r_[parameters, .005, -.01]
        analytic = residual_and_jacobian(parameters)[1]
        numeric = np.zeros_like(analytic)
        for j in range(len(parameters)):
            direction = np.zeros(len(parameters))
            direction[j] = 1e-6
            numeric[:, j] = (residual_and_jacobian(parameters + direction)[0] - residual_and_jacobian(parameters - direction)[0]) / 2e-6
        error = float(np.max(np.abs(analytic - numeric)))
        assert error < 5e-5, error
        return dict(jacobian_max_error=error, parameters=len(parameters), radio_rows=len(observations),
                    inertial_projected_energy=excitation, radio_projected_energy=radio_excitation,
                    candidate_yaw_deg=float(np.degrees(candidate)))
    results = {}
    policies = dict(zero_yaw=[0.], raw_local=[0.], projected_single=[candidate],
                    uniform8=list(np.arange(UNIFORM_STARTS) * 2 * np.pi / UNIFORM_STARTS))
    for policy, seeds in policies.items():
        started = time.perf_counter()
        fits, evaluations = [], 0
        for yaw in seeds:
            initial = np.r_[cv, yaw, np.zeros(3)] if bias_enabled else np.r_[cv, yaw]
            if tilt_enabled:
                initial = np.r_[initial, np.zeros(2)]
            fixed = 0. if policy == "zero_yaw" else None
            fit = least_squares(lambda x: residual_and_jacobian(x, fixed)[0], initial,
                                jac=lambda x: residual_and_jacobian(x, fixed)[1],
                                loss="huber", f_scale=2, max_nfev=120,
                                ftol=1e-8, xtol=1e-8, gtol=1e-8)
            fits.append(fit)
            evaluations += fit.nfev
        fit = min(fits, key=lambda value: value.cost)
        parameters = fit.x
        yaw = 0. if policy == "zero_yaw" else parameters[6]
        yaw_rotation = Rotation.from_euler("z", yaw).as_matrix()
        if tilt_enabled:
            yaw_rotation = yaw_rotation @ Rotation.from_rotvec(np.r_[parameters[10:12], 0.]).as_matrix()
        state = np.eye(5)
        state[:3, :3] = yaw_rotation @ shape["endpoint_rotation"]
        state[:3, 3:5] = np.column_stack([parameters[3:6], parameters[:3]])
        bias = np.r_[shape["gyro_bias"], parameters[7:10] if bias_enabled else np.zeros(3)]
        _, jacobian = residual_and_jacobian(parameters)
        data_jacobian = jacobian[:len(observations)]
        nuisance_columns = [0, 1, 2, 3, 4, 5] + ([7, 8, 9] if bias_enabled else []) + ([10, 11] if tilt_enabled else [])
        nuisance = data_jacobian[:, nuisance_columns]
        yaw_column = data_jacobian[:, 6]
        profiled = yaw_column - nuisance @ np.linalg.lstsq(nuisance, yaw_column, rcond=None)[0]
        results[policy] = dict(state=state, bias=bias, endpoint=shape["endpoint"],
                               cost=float(fit.cost), capture_runtime=time.perf_counter() - started,
                               fit_evaluations=evaluations, cv_evaluations=cv_evaluations,
                               cv_runtime=cv_runtime,
                               candidate_runtime=candidate_runtime if policy == "projected_single" else 0.,
                               candidate_evaluations=candidate_evaluations if policy == "projected_single" else 0,
                               candidate_yaw_deg=float(np.degrees(candidate)),
                               yaw_information_data=float(profiled @ profiled),
                               excitation=excitation, radio_count=len(observations),
                               radio_excitation=radio_excitation,
                               fit_success=bool(fit.success), bias_norm=float(np.linalg.norm(bias[3:])),
                               gyro_bias_norm=float(np.linalg.norm(bias[:3])),
                               stationary_prefix=bool(shape["stationary_prefix"]))
        results[policy]["tilt_correction_deg"] = float(np.degrees(np.linalg.norm(parameters[10:12]))) if tilt_enabled else 0.
    return results


def replay(data, capture_result):
    """Consume only packets after capture once in the common white-variance InEKF."""
    state, bias, endpoint = (capture_result[k] for k in ("state", "bias", "endpoint"))
    # Shared handoff covariance isolates changes of nominal capture estimates.
    physical = np.diag(np.array([.08, .08, np.pi, .7, .7, .7, .5, .5, .5, .02, .02, .02, .2, .2, .2]) ** 2)
    mapping = coordinate_map(state)
    estimator = Invariant15(state, mapping @ physical @ mapping.T, bias)
    times, imu, radio, anchors = (data[k] for k in ("times", "imu", "radio", "anchors"))
    index = np.searchsorted(radio[:, 0], endpoint, side="right")
    previous, rows = endpoint, []
    accepted, rejected = 0, 0
    started = time.perf_counter()
    for k in range(np.searchsorted(times, endpoint, side="right"), len(times)):
        stamp = times[k]
        estimator.predict(.5 * (imu[k - 1] + imu[k]), stamp - previous)
        while index < len(radio) and radio[index, 0] <= stamp:
            observation = radio[index]
            lag = observation[0] - stamp
            state = estimator.X
            tag = state[:3, 4] + lag * state[:3, 3] + state[:3, :3] @ LEVER
            da, db = tag - anchors[int(observation[1])], tag - anchors[int(observation[2])]
            ra, rb = np.linalg.norm(da), np.linalg.norm(db)
            gradient = db / rb - da / ra
            row = np.zeros(15)
            row[:3], row[3:6], row[6:9] = -gradient @ skew(tag), lag * gradient, gradient
            innovation = observation[3] - (rb - ra)
            projected = estimator.P @ row
            variance = RADIO_VARIANCE + (.08 if "const4" in data["name"] else 0.)
            nis = innovation**2 / (row @ projected + variance)
            if nis > 25:
                rejected += 1
            else:
                weight = min(1., 2. / max(np.sqrt(nis), 1e-12))
                total = row @ projected + variance / weight
                delta = projected * innovation / total
                estimator.P -= np.outer(projected, projected) / total
                estimator.X = group_exp(delta[:9]) @ estimator.X
                estimator.bias += delta[9:15]
                reset = Invariant15.reset_jacobian(delta)
                estimator.P = reset @ estimator.P @ reset.T
                estimator.P = .5 * (estimator.P + estimator.P.T)
                accepted += 1
            index += 1
        if k % 5 == 0:
            rows.append(np.r_[stamp, estimator.X[:3, 4], Rotation.from_matrix(estimator.X[:3, :3]).as_quat()])
        previous = stamp
    return np.array(rows), dict(track_runtime=time.perf_counter() - started, accepted=accepted, rejected=rejected)


def endpoint_metrics(data, result):
    """Use truth only after fitting to score the handoff's physical pose."""
    truth = data["truth"]
    stamp = result["endpoint"]
    position = np.array([np.interp(stamp, truth[:, 0], truth[:, k]) for k in (1, 2, 3)])
    actual = Slerp(truth[:, 0], Rotation.from_quat(truth[:, 4:8]))(stamp)
    predicted = Rotation.from_matrix(result["state"][:3, :3])
    yaw = predicted.as_euler("xyz")[2] - actual.as_euler("xyz")[2]
    return dict(endpoint_position_error=float(np.linalg.norm(result["state"][:3, 4] - position)),
                endpoint_yaw_error_deg=float(np.degrees(np.arctan2(np.sin(yaw), np.cos(yaw)))),
                endpoint_attitude_error_deg=float(np.degrees((actual.inv() * predicted).magnitude())))


def main():
    """Execute fixed windows and store physical capture and subsequent tracking outcomes."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--names", nargs="+", default=["const3-trial2-tdoa2", "const3-trial3-tdoa2", "const3-trial3-tdoa3", "const1-trial3-tdoa2"])
    parser.add_argument("--start", type=float, default=10.)
    parser.add_argument("--length", type=float, default=6.)
    parser.add_argument("--duration", type=float, default=45.)
    parser.add_argument("--bias-models", nargs="+", default=["fixed", "prior"])
    parser.add_argument("--output", default="results")
    parser.add_argument("--gyro-calibration", action="store_true")
    parser.add_argument("--tilt-model", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.tilt_model and args.bias_models != ["prior"]:
        raise ValueError("The full tilt model requires --bias-models prior")
    output = Path(__file__).parent / args.output
    output.mkdir(exist_ok=True)
    rows = []
    for name in args.names:
        data = load(name, duration=args.duration)
        for model in args.bias_models:
            results = capture(data, args.start, args.length, model == "prior", args.gyro_calibration, args.tilt_model, args.check_only)
            if args.check_only:
                record = dict(name=name, model=model, **results)
                rows.append(record)
                (output / "checks.json").write_text(json.dumps(rows, indent=2), encoding="utf8")
                print(json.dumps(record), flush=True)
                continue
            for policy, result in results.items():
                trajectory, counts = replay(data, result)
                row = dict(name=name, model=model, policy=policy,
                           **{k: v for k, v in result.items() if k not in ["state", "bias"]},
                           **endpoint_metrics(data, result), **counts, **score(data, trajectory))
                rows.append(row)
                pd.DataFrame(rows).to_csv(output / "metrics.csv", index=False)
                np.savez_compressed(output / f"{name}_{model}_{policy}.npz", trajectory=trajectory,
                                    capture_state=result["state"], capture_bias=result["bias"])
                print(json.dumps(row), flush=True)
    (output / "arguments.json").write_text(json.dumps(vars(args), indent=2), encoding="utf8")


if __name__ == "__main__":
    main()
