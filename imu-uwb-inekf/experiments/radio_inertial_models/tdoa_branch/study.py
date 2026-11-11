"""Real-flight TDoA diagnostics and matched invariant filtering experiments.

Ground truth is accessed only by diagnostics/scoring, never by initialization.
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

# Repository paths locate existing exact discrete inertial model functions.
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments" / "roadmap_stage1"))
from eskf15 import discrete_error_matrices, nominal_step, rotation_terms, skew
from invariant15 import coordinate_map, inverse_coordinate_map, group_exp, Invariant15

# Dataset root is fixed for this machine and may be overridden on the CLI.
DATA = Path("E:/datasets/dataset/flight-dataset")
# Published antenna lever arm in the vehicle frame, in metres.
LEVER = np.array([-0.01245, 0.00127, 0.0908])
# Predetermined 100 Hz inertial processing keeps the paired study practical.
IMU_STEP = 0.01


def load(name, data_root=DATA, duration=40.0):
    """Read raw streams without borrowing truth for a sensor timestamp or value."""
    geometry = name.split("-")[0]
    frame = pd.read_csv(data_root / "csv-data" / geometry / (name + ".csv"))
    def stream(columns):
        """Read one independently timestamped stream."""
        values = frame[columns].dropna().to_numpy(float)
        return values[np.argsort(values[:, 0], kind="stable")]
    acc = stream(["t_acc", "acc_x", "acc_y", "acc_z"])
    gyr = stream(["t_gyro", "gyro_x", "gyro_y", "gyro_z"])
    radio = stream(["t_tdoa", "idA", "idB", "tdoa_meas"])
    truth = stream(["t_pose", "pose_x", "pose_y", "pose_z", "pose_qx", "pose_qy", "pose_qz", "pose_qw"])
    start = max(acc[0, 0], gyr[0, 0], radio[0, 0])
    end = min(start + duration, acc[-1, 0], gyr[-1, 0], radio[-1, 0])
    times = np.arange(start, end, IMU_STEP)
    imu = np.column_stack([np.interp(times, s[:, 0], s[:, k]) for s in (acc, gyr) for k in (1, 2, 3)])
    radio = radio[(radio[:, 0] >= start) & (radio[:, 0] <= end)]
    if "tdoa2" in name:
        # Use the documented ring protocol, matching the repository native loader.
        ring = {(7, 0), (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)}
        radio = radio[np.array([tuple(row) in ring for row in radio[:, 1:3]])]
    anchors = np.load(data_root / "survey-results" / ("anchor_" + geometry + ".npz"))["an_pos"]
    return dict(name=name, times=times, imu=imu, radio=radio, truth=truth, anchors=anchors)


def diagnose(data):
    """Measure pair residual offsets, tails, and within-pair temporal dependence."""
    radio, truth, anchors = (data[k] for k in ("radio", "truth", "anchors"))
    valid = (radio[:, 0] >= truth[0, 0]) & (radio[:, 0] <= truth[-1, 0])
    radio = radio[valid]
    body = np.column_stack([np.interp(radio[:, 0], truth[:, 0], truth[:, k]) for k in (1, 2, 3)])
    attitude = Slerp(truth[:, 0], Rotation.from_quat(truth[:, 4:8]))(radio[:, 0])
    tag = body + attitude.apply(LEVER)
    a, b = radio[:, 1].astype(int), radio[:, 2].astype(int)
    residual = radio[:, 3] - (np.linalg.norm(tag - anchors[b], axis=1) - np.linalg.norm(tag - anchors[a], axis=1))
    rows = []
    for first, second in np.unique(radio[:, 1:3], axis=0).astype(int):
        mask = (a == first) & (b == second)
        r, t = residual[mask], radio[mask, 0]
        clipped = np.clip(r - np.median(r), -1.0, 1.0)
        if len(r) < 3:
            continue
        rows.append(dict(pair=f"{first}-{second}", count=len(r), median=float(np.median(r)),
                         mad_sigma=float(1.4826 * np.median(np.abs(r - np.median(r)))),
                         lag1=float(np.corrcoef(clipped[:-1], clipped[1:])[0, 1]),
                         dt=float(np.median(np.diff(t))), outlier_1m=float(np.mean(np.abs(r) > 1)),
                         positive_1m=float(np.mean(r > 1)), negative_1m=float(np.mean(r < -1))))
    return rows


def initialize(data, end_time=2.0):
    """Use the same two-second radio CV fit and gravity alignment for all methods."""
    times, imu, radio, anchors = (data[k] for k in ("times", "imu", "radio", "anchors"))
    endpoint = times[0] + end_time
    prefix = imu[times <= endpoint]
    accel = np.mean(prefix[:, :3], axis=0)
    # Only tilt is observable from mean specific force; yaw remains an uncertain zero.
    vertical = accel / np.linalg.norm(accel)
    tilt = Rotation.align_vectors(np.array([[0., 0., 1.]]), vertical[None])[0].as_matrix()
    observations = radio[radio[:, 0] <= endpoint]
    a, b = observations[:, 1].astype(int), observations[:, 2].astype(int)
    tau = observations[:, 0] - endpoint
    def residual(parameters):
        """Evaluate raw hyperbolic observations for a constant velocity antenna path."""
        tag = parameters[:3] + tau[:, None] * parameters[3:]
        return (np.linalg.norm(tag - anchors[b], axis=1) - np.linalg.norm(tag - anchors[a], axis=1) - observations[:, 3]) / 0.3
    seed = np.r_[np.mean(anchors, axis=0), np.zeros(3)]
    fit = least_squares(residual, seed, loss="huber", f_scale=2, max_nfev=100)
    state = np.eye(5)
    state[:3, :3] = tilt
    state[:3, 3] = fit.x[3:]
    state[:3, 4] = fit.x[:3] - tilt @ LEVER
    # Bias is not calibrated using an assumed static interval.
    bias = np.zeros(6)
    covariance = np.diag(np.array([0.08, 0.08, np.pi, 0.7, 0.7, 0.7, 0.5, 0.5, 0.5,
                                   0.02, 0.02, 0.02, 0.2, 0.2, 0.2]) ** 2)
    return endpoint, state, covariance, bias


def run_filter(data, mode, tau=2.0, pair_sigma=0.3):
    """Compare native TDoA and nuisance-augmented right-invariant filtering."""
    endpoint, state, physical, bias = initialize(data)
    radio, anchors, times, imu = (data[k] for k in ("radio", "anchors", "times", "imu"))
    # Protocol pair inventory is specified before replay, never learned from future data.
    pairs = ([(j, (j + 1) % 8) for j in range(8)] if "tdoa2" in data["name"]
             else [(a, b) for a in range(8) for b in range(8) if a != b])
    pair_map = {pair: index for index, pair in enumerate(pairs)}
    extra = len(pairs) if mode.startswith("pair") else 0
    n = 15 + extra
    covariance = np.zeros((n, n))
    mapping = coordinate_map(state)
    covariance[:15, :15] = mapping @ physical @ mapping.T
    covariance[15:, 15:] = np.eye(extra) * pair_sigma**2
    nuisance = np.zeros(extra)
    rows, accepted, rejected = [], 0, 0
    radio_index = np.searchsorted(radio[:, 0], endpoint, side="right")
    previous = endpoint
    started = time.perf_counter()
    for k in range(np.searchsorted(times, endpoint, side="right"), len(times)):
        stamp = times[k]
        sample = 0.5 * (imu[k - 1] + imu[k])
        dt = stamp - previous
        inverse = inverse_coordinate_map(state)
        transition, noise = discrete_error_matrices(state, bias, sample, dt)
        state = nominal_step(state, bias, sample, dt)
        mapping = coordinate_map(state)
        navigation = mapping @ transition @ inverse
        augmented = np.eye(n)
        augmented[:15, :15] = navigation
        # A stationary Gauss-Markov pair error models colored radio error.
        decay = np.exp(-dt / tau) if mode == "pair_ou" else 1.0
        augmented[15:, 15:] *= decay
        covariance = augmented @ covariance @ augmented.T
        variances = np.array([0.1**2] * 3 + [2.0**2] * 3 + [1e-6**2] * 3 + [1e-4**2] * 3)
        mapped_noise = mapping @ noise
        covariance[:15, :15] += (mapped_noise * variances) @ mapped_noise.T
        covariance[15:, 15:] += np.eye(extra) * pair_sigma**2 * (1 - decay**2)
        nuisance *= decay
        while radio_index < len(radio) and radio[radio_index, 0] <= stamp:
            obs = radio[radio_index]
            first, second = int(obs[1]), int(obs[2])
            # A sub-IMU-period time correction avoids pretending UWB was synchronous.
            lag = obs[0] - stamp
            tag = state[:3, 4] + lag * state[:3, 3] + state[:3, :3] @ LEVER
            da, db = tag - anchors[first], tag - anchors[second]
            ra, rb = np.linalg.norm(da), np.linalg.norm(db)
            gradient = db / rb - da / ra
            row = np.zeros(n)
            row[:3] = -gradient @ skew(tag)
            row[3:6] = gradient * lag
            row[6:9] = gradient
            predicted = rb - ra
            if extra:
                index = pair_map[(first, second)]
                predicted += nuisance[index]
                row[15 + index] = 1
            innovation = obs[3] - predicted
            variance = 0.05 if "const4" not in data["name"] else 0.13
            if mode == "white_inflated":
                # Match the OU model's stationary marginal measurement variance.
                variance += pair_sigma**2
            projected = covariance @ row
            total = row @ projected + variance
            nis = innovation**2 / total
            if nis > 25:
                rejected += 1
            else:
                # All modes share the same Huber threshold and hard outlier gate.
                weight = min(1.0, 2.0 / max(np.sqrt(nis), 1e-12))
                total = row @ projected + variance / weight
                delta = projected * (innovation / total)
                covariance -= np.outer(projected, projected) / total
                state = group_exp(delta[:9]) @ state
                bias += delta[9:15]
                if extra:
                    nuisance += delta[15:]
                reset = np.eye(n)
                reset[:15, :15] = Invariant15.reset_jacobian(delta[:15])
                covariance = reset @ covariance @ reset.T
                accepted += 1
            radio_index += 1
        covariance = 0.5 * (covariance + covariance.T)
        if k % 5 == 0:
            rows.append(np.r_[stamp, state[:3, 4], Rotation.from_matrix(state[:3, :3]).as_quat()])
        previous = stamp
        if np.linalg.norm(state[:3, 4]) > 100:
            break
    return np.array(rows), dict(accepted=accepted, rejected=rejected, runtime=time.perf_counter() - started,
                               pair_count=extra, tau=tau, pair_sigma=pair_sigma)


def score(data, trajectory):
    """Score absolute body position and SO(3) attitude with no alignment fit."""
    truth = data["truth"]
    keep = (trajectory[:, 0] >= truth[0, 0]) & (trajectory[:, 0] <= truth[-1, 0])
    trajectory = trajectory[keep]
    position = np.column_stack([np.interp(trajectory[:, 0], truth[:, 0], truth[:, k]) for k in (1, 2, 3)])
    rotation = Slerp(truth[:, 0], Rotation.from_quat(truth[:, 4:8]))(trajectory[:, 0])
    estimated = Rotation.from_quat(trajectory[:, 4:8])
    angle = np.degrees((rotation.inv() * estimated).magnitude())
    error = np.linalg.norm(trajectory[:, 1:4] - position, axis=1)
    yaw = np.degrees(np.arctan2(np.sin(estimated.as_euler("xyz")[:, 2] - rotation.as_euler("xyz")[:, 2]),
                               np.cos(estimated.as_euler("xyz")[:, 2] - rotation.as_euler("xyz")[:, 2])))
    tail = trajectory[:, 0] >= data["times"][0] + 10
    return dict(position_rmse=float(np.sqrt(np.mean(error**2))), position_p95=float(np.quantile(error, .95)),
                attitude_rmse_deg=float(np.sqrt(np.mean(angle**2))), yaw_rmse_deg=float(np.sqrt(np.mean(yaw**2))),
                tail_position_rmse=float(np.sqrt(np.mean(error[tail]**2))),
                tail_yaw_rmse_deg=float(np.sqrt(np.mean(yaw[tail]**2))),
                samples=len(error), final_time=float(trajectory[-1, 0]))


def main():
    """Run explicit named recordings and save every real diagnostic and metric."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--names", nargs="+", default=["const1-trial1-tdoa2", "const2-trial1-tdoa2", "const3-trial1-tdoa2", "const4-trial1-tdoa2-traj1"])
    parser.add_argument("--duration", type=float, default=40)
    parser.add_argument("--modes", nargs="+", default=["native", "pair_static", "pair_ou"])
    parser.add_argument("--output", default="results_first")
    parser.add_argument("--tau", type=float, default=2)
    parser.add_argument("--pair-sigma", type=float, default=.3)
    args = parser.parse_args()
    output = Path(__file__).parent / args.output
    output.mkdir(exist_ok=True)
    metrics, diagnostics = [], []
    for name in args.names:
        data = load(name, duration=args.duration)
        diagnostics.extend(dict(name=name, **row) for row in diagnose(data))
        for mode in args.modes:
            trajectory, counts = run_filter(data, mode, args.tau, args.pair_sigma)
            metric = dict(name=name, mode=mode, **counts, **score(data, trajectory))
            metrics.append(metric)
            np.savez_compressed(output / f"{name}_{mode}.npz", trajectory=trajectory)
            pd.DataFrame(metrics).to_csv(output / "metrics.csv", index=False)
            print(json.dumps(metric), flush=True)
    pd.DataFrame(diagnostics).to_csv(output / "diagnostics.csv", index=False)
    (output / "arguments.json").write_text(json.dumps(vars(args), indent=2), encoding="utf8")


if __name__ == "__main__":
    main()
