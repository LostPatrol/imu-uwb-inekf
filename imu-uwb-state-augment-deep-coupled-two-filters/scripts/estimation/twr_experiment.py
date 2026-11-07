#!/usr/bin/env python3
"""Run task-06 TWR state-augmentation experiments.

The real-data path reads the compact TU Delft Crazyflie dataset.  The primary
comparison uses the same world-frame acceleration from an IMU-InEKF core for:

* a conventional translational TWR EKF with a position-dependent Jacobian;
* the lifted half-squared TWR Kalman observer with constant observation rows.

An ordinary TWR adaptation of the author's InEKF is also reported as a
reference in representative runs.  The synthetic experiment follows the
single-range, persistently exciting setting of Theodosis et al. and is kept
separate from the real-data evidence.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import time
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.stats import spearmanr
import yaml

from inekf import GRAVITY_MAGNITUDE, InEKF, SO3_exp
from lifted_twr import (
    DirectRangeEKF,
    LiftedRangeKF,
    LiftedTWRInEKF,
    SquaredRangeEKF,
    TWRInEKF,
    imu_world_acceleration,
    paper_consistent_initial_covariance,
)


DEFAULT_DATASET = Path(
    r"E:\datasets\tu-delft-uwb-mhe\data_v1\UWB dataset"
)
DEFAULT_SEQUENCE = "2021-02-12_09_56_09_twr_random.csv"
# The TU Delft logger exposes arrival/log time rather than the acquisition
# time of the radio/IMU packet.  A calibration trajectory shows that the
# on-board IMU and TWR streams share an approximately 0.20 s delay relative
# to the OptiTrack time axis.  Keep the generic loader default at zero for
# backward compatibility; the deep-coupled real-data entry point opts in.
TU_DELFT_COMMON_SENSOR_DELAY_S = 0.20
METHOD_COLORS = {
    "direct_twr": "#d95f02",
    "lifted_twr": "#1b9e77",
    "twr_inekf": "#7570b3",
}


@dataclass
class RealLog:
    name: str
    time: np.ndarray
    imu: np.ndarray
    ranges: np.ndarray
    changed: np.ndarray
    ground_truth_position: np.ndarray
    ground_truth_velocity: np.ndarray
    sensor_delay_s: float = 0.0


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _as_vector3_for_experiment(
    value: Iterable[float], name: str
) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain three finite values")
    return vector


def load_anchors(dataset: Path) -> np.ndarray:
    with (dataset / "anchor_positions.yaml").open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    anchors = np.array(
        [[document[index][axis] for axis in ("x", "y", "z")] for index in range(8)],
        dtype=np.float64,
    )
    if anchors.shape != (8, 3) or not np.all(np.isfinite(anchors)):
        raise ValueError("anchor_positions.yaml does not contain eight valid anchors")
    return anchors


def _smooth_velocity(time_values: np.ndarray, position: np.ndarray) -> np.ndarray:
    if time_values.size < 9:
        return np.gradient(position, time_values, axis=0, edge_order=1)
    median_dt = float(np.median(np.diff(time_values)))
    requested = max(9, int(round(0.31 / median_dt)))
    if requested % 2 == 0:
        requested += 1
    window = min(requested, time_values.size - (1 - time_values.size % 2))
    window = max(5, window)
    smoothed = savgol_filter(
        position, window_length=window, polyorder=3, axis=0, mode="interp"
    )
    return np.gradient(smoothed, time_values, axis=0, edge_order=2)


def load_real_log(
    dataset: Path,
    sequence: str,
    duration: float | None = None,
    *,
    sensor_delay_s: float = 0.0,
) -> RealLog:
    """Load one replay and align delayed on-board sensors to OptiTrack.

    ``sensor_delay_s`` is the common acquisition-to-log delay of the IMU and
    TWR streams relative to the OptiTrack/log time axis.  Both on-board
    streams are kept together; only their common time coordinate is shifted.
    This is deliberately not a UWB-only shift, which would destroy the
    already-correct IMU--UWB relative timing.
    """

    delay = float(sensor_delay_s)
    if not np.isfinite(delay) or delay < 0.0:
        raise ValueError("sensor_delay_s must be finite and non-negative")

    path = dataset / "twr" / sequence
    if not path.is_file():
        available = sorted(item.name for item in (dataset / "twr").glob("*.csv"))
        raise FileNotFoundError(
            f"TWR log not found: {path}\nAvailable sequences: {available}"
        )

    frame = pd.read_csv(path, skipinitialspace=True)
    required = (
        ["timeTick", "gyroX", "gyroY", "gyroZ", "accX", "accY", "accZ"]
        + [f"twr{index}" for index in range(8)]
        + ["otX", "otY", "otZ"]
    )
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{path.name} is missing columns: {missing}")

    sensor_columns = (
        ["gyroX", "gyroY", "gyroZ", "accX", "accY", "accZ"]
        + [f"twr{index}" for index in range(8)]
    )
    sensor_finite = np.all(
        np.isfinite(frame[sensor_columns].to_numpy(dtype=np.float64)),
        axis=1,
    )
    raw_truth = frame[["otX", "otY", "otZ"]].to_numpy(dtype=np.float64)
    truth_finite = np.all(np.isfinite(raw_truth), axis=1)
    nonzero_truth = np.linalg.norm(raw_truth, axis=1) > 1e-9
    truth_valid = truth_finite & nonzero_truth
    start_candidates = np.flatnonzero(sensor_finite & truth_valid)
    if start_candidates.size == 0:
        raise ValueError(f"{path.name} has no finite OptiTrack interval")
    start = int(start_candidates[0])

    # Change detection is performed on the raw log, as the logger holds each
    # anchor value until a new TWR transaction arrives.
    raw_ranges = frame[[f"twr{index}" for index in range(8)]].to_numpy(
        dtype=np.float64
    )
    changed = np.ones_like(raw_ranges, dtype=bool)
    changed[1:] = np.abs(np.diff(raw_ranges, axis=0)) > 1e-12

    raw_time = frame["timeTick"].to_numpy(dtype=np.float64) / 1000.0
    initial_time = raw_time[start]
    measurement_time = raw_time - delay

    # A packet logged at t is treated as having been acquired at t-delay.
    # Discard only packets whose corrected acquisition time precedes the
    # valid OptiTrack interval.  Interpolation permits non-integer delays.
    selected = sensor_finite & (measurement_time >= initial_time - 1e-12)
    selected &= measurement_time <= raw_time[truth_valid][-1] + 1e-12
    if duration is not None:
        selected &= (
            measurement_time
            <= initial_time + float(duration) + 1e-12
        )

    time_values = measurement_time[selected] - initial_time
    if time_values.size < 3 or np.any(np.diff(time_values) <= 0):
        raise ValueError(f"{path.name} has an invalid time base")
    imu = frame[
        ["accX", "accY", "accZ", "gyroX", "gyroY", "gyroZ"]
    ].to_numpy(dtype=np.float64)[selected]
    ranges = raw_ranges[selected]
    changed_selected = changed[selected]
    truth_time = raw_time[truth_valid]
    truth_values = raw_truth[truth_valid]
    ground_truth = np.column_stack(
        [
            np.interp(measurement_time[selected], truth_time, truth_values[:, axis])
            for axis in range(3)
        ]
    )
    velocity = _smooth_velocity(time_values, ground_truth)
    return RealLog(
        name=path.name,
        time=time_values,
        imu=imu,
        ranges=ranges,
        changed=changed_selected,
        ground_truth_position=ground_truth,
        ground_truth_velocity=velocity,
        sensor_delay_s=delay,
    )


def rotation_aligning_vectors(source: Iterable[float], target: Iterable[float]) -> np.ndarray:
    # Copy because the normalization below is in-place; callers may still
    # need the original vector magnitude for bias calibration.
    source_vector = np.array(source, dtype=np.float64, copy=True)
    target_vector = np.array(target, dtype=np.float64, copy=True)
    source_vector /= np.linalg.norm(source_vector)
    target_vector /= np.linalg.norm(target_vector)
    cross = np.cross(source_vector, target_vector)
    sine = float(np.linalg.norm(cross))
    cosine = float(np.clip(source_vector @ target_vector, -1.0, 1.0))
    if sine < 1e-12:
        if cosine > 0:
            return np.eye(3)
        trial = np.array([1.0, 0.0, 0.0])
        if abs(source_vector @ trial) > 0.9:
            trial = np.array([0.0, 1.0, 0.0])
        axis = np.cross(source_vector, trial)
        axis /= np.linalg.norm(axis)
        return SO3_exp(math.pi * axis)
    axis = cross / sine
    angle = math.atan2(sine, cosine)
    return SO3_exp(angle * axis)


def estimate_initial_rotation_and_bias(log: RealLog) -> tuple[np.ndarray, np.ndarray]:
    calibration_end = min(log.time[-1], 1.5)
    candidates = log.time <= calibration_end
    gyro_norm = np.linalg.norm(log.imu[:, 3:], axis=1)
    acceleration_norm = np.linalg.norm(log.imu[:, :3], axis=1)
    stationary = (
        candidates
        & (gyro_norm < 8.0)
        & (acceleration_norm > 0.85)
        & (acceleration_norm < 1.15)
    )
    if np.count_nonzero(stationary) < 20:
        stationary = candidates

    mean_acceleration = (
        np.mean(log.imu[stationary, :3], axis=0) * GRAVITY_MAGNITUDE
    )
    rotation = rotation_aligning_vectors(
        mean_acceleration, np.array([0.0, 0.0, GRAVITY_MAGNITUDE])
    )
    gyro_bias = np.deg2rad(np.mean(log.imu[stationary, 3:], axis=0))
    expected_specific_force = rotation.T @ np.array(
        [0.0, 0.0, GRAVITY_MAGNITUDE]
    )
    accelerometer_bias = mean_acceleration - expected_specific_force
    return rotation, np.concatenate((gyro_bias, accelerometer_bias))


def convergence_time(
    time_values: np.ndarray,
    errors: np.ndarray,
    *,
    threshold: float = 0.5,
    dwell_seconds: float = 1.0,
) -> float:
    dt = float(np.median(np.diff(time_values)))
    dwell = max(1, int(math.ceil(dwell_seconds / dt)))
    below = errors <= threshold
    if below.size < dwell:
        return math.nan
    run = np.convolve(below.astype(np.int32), np.ones(dwell, dtype=np.int32), "valid")
    candidates = np.flatnonzero(run == dwell)
    if candidates.size == 0:
        return math.nan
    return float(time_values[int(candidates[0])])


def trajectory_metrics(
    time_values: np.ndarray,
    position: np.ndarray,
    velocity: np.ndarray,
    ground_truth_position: np.ndarray,
    ground_truth_velocity: np.ndarray,
) -> dict[str, float]:
    position_error = np.linalg.norm(position - ground_truth_position, axis=1)
    velocity_error = np.linalg.norm(velocity - ground_truth_velocity, axis=1)
    tail = time_values >= max(0.0, time_values[-1] - 5.0)
    return {
        "position_rmse_m": float(np.sqrt(np.mean(position_error**2))),
        "position_tail5_rmse_m": float(np.sqrt(np.mean(position_error[tail] ** 2))),
        "position_p95_m": float(np.percentile(position_error, 95)),
        "position_max_m": float(np.max(position_error)),
        "velocity_rmse_mps": float(np.sqrt(np.mean(velocity_error**2))),
        "velocity_tail5_rmse_mps": float(
            np.sqrt(np.mean(velocity_error[tail] ** 2))
        ),
        "convergence_0p5m_s": convergence_time(time_values, position_error),
        "finite": bool(
            np.all(np.isfinite(position)) and np.all(np.isfinite(velocity))
        ),
    }


def _initial_state(
    log: RealLog, initial_offset: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rotation, bias = estimate_initial_rotation_and_bias(log)
    state = np.eye(5)
    state[:3, :3] = rotation
    state[:3, 3] = np.zeros(3)
    state[:3, 4] = log.ground_truth_position[0] + initial_offset
    covariance = paper_consistent_initial_covariance(
        attitude_std=(0.05, 0.05, 0.8),
        velocity_std=2.0,
        position_std=8.0,
        gyro_bias_std=0.01,
        accel_bias_std=0.10,
    )
    return state, covariance, bias


def run_real_sequence(
    log: RealLog,
    anchors: np.ndarray,
    *,
    initial_offset: Iterable[float] = (0.0, 0.0, 0.0),
    range_sigma: float = 0.20,
    range_bias: float = -0.18,
    accel_noise_std: float = 10.0,
    include_ordinary_inekf: bool = True,
) -> dict[str, Any]:
    offset = np.asarray(initial_offset, dtype=np.float64)
    if offset.shape != (3,):
        raise ValueError("initial_offset must contain three values")
    initial_state, covariance, bias = _initial_state(log, offset)

    lifted = LiftedTWRInEKF(
        initial_state,
        covariance,
        bias,
        position_std=8.0,
        velocity_std=2.0,
        q_std=80.0,
        s_std=40.0,
        h_std=20.0,
        # The lifted q/s/h dynamics amplify attitude and accelerometer model
        # mismatch.  A 10 m/s^2 effective input uncertainty was selected
        # during development scans of two logs and then frozen for the batch
        # evaluation.  Those logs remain in the descriptive (not held-out)
        # batch, which is disclosed in the task report.
        accel_noise_std=accel_noise_std,
        range_sigma=range_sigma,
        range_bias=range_bias,
        gate_threshold=25.0,
        gate_warmup_updates=64,
    )
    direct = DirectRangeEKF(
        initial_state[:3, 4],
        initial_state[:3, 3],
        position_std=8.0,
        velocity_std=2.0,
        accel_noise_std=accel_noise_std,
        range_sigma=range_sigma,
        range_bias=range_bias,
        gate_threshold=25.0,
        gate_warmup_updates=64,
    )
    ordinary = None
    if include_ordinary_inekf:
        ordinary = TWRInEKF(
            initial_state.copy(),
            covariance.copy(),
            bias.copy(),
            range_sigma=range_sigma,
            range_bias=range_bias,
            gate_threshold=25.0,
            # Preserve the ordinary author's immediate-gating behavior in
            # real-data comparisons.  The synthetic large-error test passes
            # an explicit common 80-update warmup to all three methods.
            gate_warmup_updates=0,
            lever_arm=(0.0, 0.0, 0.0),
        )

    sample_count = log.time.size
    histories = {
        "direct_twr": {
            "position": np.empty((sample_count, 3)),
            "velocity": np.empty((sample_count, 3)),
        },
        "lifted_twr": {
            "position": np.empty((sample_count, 3)),
            "velocity": np.empty((sample_count, 3)),
            "constraint": np.empty((sample_count, 3)),
        },
    }
    if ordinary is not None:
        histories["twr_inekf"] = {
            "position": np.empty((sample_count, 3)),
            "velocity": np.empty((sample_count, 3)),
        }

    start_clock = time.perf_counter()
    for index in range(sample_count):
        if index > 0:
            dt = float(log.time[index] - log.time[index - 1])
            world_acceleration = lifted.predict(log.imu[index - 1], dt)
            direct.predict(world_acceleration, dt)
            if ordinary is not None:
                ordinary.predict(log.imu[index - 1], dt)

        for anchor_index in np.flatnonzero(log.changed[index]):
            measured_range = float(log.ranges[index, anchor_index])
            lifted.correct_twr(anchors[anchor_index], measured_range)
            direct.correct(anchors[anchor_index], measured_range)
            if ordinary is not None:
                ordinary.correct_twr(anchors[anchor_index], measured_range)

        histories["lifted_twr"]["position"][index] = lifted.lifted.position
        histories["lifted_twr"]["velocity"][index] = lifted.lifted.velocity
        histories["lifted_twr"]["constraint"][index] = (
            lifted.lifted.constraint_residual()
        )
        histories["direct_twr"]["position"][index] = direct.position
        histories["direct_twr"]["velocity"][index] = direct.velocity
        if ordinary is not None:
            histories["twr_inekf"]["position"][index] = ordinary.X[:3, 4]
            histories["twr_inekf"]["velocity"][index] = ordinary.X[:3, 3]
    elapsed = time.perf_counter() - start_clock

    metrics: dict[str, Any] = {
        "sequence": log.name,
        "duration_s": float(log.time[-1]),
        "samples": int(sample_count),
        "twr_events": int(np.count_nonzero(log.changed)),
        "initial_offset_m": offset.tolist(),
        "range_sigma_m": float(range_sigma),
        "range_bias_m": float(range_bias),
        "effective_accel_noise_std_mps2": float(accel_noise_std),
        "runtime_s": float(elapsed),
        "methods": {},
    }
    filters = {
        "direct_twr": direct,
        "lifted_twr": lifted.lifted,
    }
    if ordinary is not None:
        filters["twr_inekf"] = ordinary

    for method, history in histories.items():
        method_metrics = trajectory_metrics(
            log.time,
            history["position"],
            history["velocity"],
            log.ground_truth_position,
            log.ground_truth_velocity,
        )
        method_filter = filters[method]
        method_metrics.update(
            {
                "accepted": int(method_filter.stats.accepted),
                "rejected": int(method_filter.stats.rejected),
                "invalid": int(method_filter.stats.invalid),
                "mean_nis": float(method_filter.stats.mean_nis),
                "max_nis": float(method_filter.stats.nis_max),
                "covariance_min_eigenvalue": float(
                    np.min(np.linalg.eigvalsh(method_filter.P))
                ),
            }
        )
        if method == "lifted_twr":
            constraint = history["constraint"]
            method_metrics["constraint_rms"] = [
                float(np.sqrt(np.mean(constraint[:, component] ** 2)))
                for component in range(3)
            ]
            method_metrics["constraint_tail_abs"] = [
                float(abs(constraint[-1, component])) for component in range(3)
            ]
        metrics["methods"][method] = method_metrics

    return {
        "metrics": metrics,
        "time": log.time,
        "ground_truth_position": log.ground_truth_position,
        "ground_truth_velocity": log.ground_truth_velocity,
        "anchors": anchors,
        "histories": histories,
    }


def plot_real_result(result: dict[str, Any], output_path: Path) -> None:
    time_values = result["time"]
    truth = result["ground_truth_position"]
    truth_velocity = result["ground_truth_velocity"]
    anchors = result["anchors"]
    histories = result["histories"]
    metrics = result["metrics"]

    figure, axes = plt.subplots(2, 2, figsize=(13.5, 9.2), constrained_layout=True)
    trajectory_axis, error_axis, component_axis, velocity_axis = axes.flat

    trajectory_axis.plot(truth[:, 0], truth[:, 1], "k-", lw=2.1, label="OptiTrack")
    for method, history in histories.items():
        trajectory_axis.plot(
            history["position"][:, 0],
            history["position"][:, 1],
            color=METHOD_COLORS[method],
            lw=1.25,
            label=method.replace("_", " "),
        )
    trajectory_axis.scatter(
        anchors[:, 0],
        anchors[:, 1],
        marker="^",
        s=55,
        color="#333333",
        label="anchors",
    )
    for index, anchor in enumerate(anchors):
        trajectory_axis.text(anchor[0] + 0.08, anchor[1] + 0.08, str(index), fontsize=8)
    trajectory_axis.set_title("XY trajectory")
    trajectory_axis.set_xlabel("x [m]")
    trajectory_axis.set_ylabel("y [m]")
    trajectory_axis.axis("equal")
    trajectory_axis.grid(alpha=0.25)
    trajectory_axis.legend(fontsize=8, ncol=2)

    for method, history in histories.items():
        error = np.linalg.norm(history["position"] - truth, axis=1)
        error_axis.plot(
            time_values,
            error,
            color=METHOD_COLORS[method],
            lw=1.15,
            label=(
                f"{method.replace('_', ' ')} "
                f"RMSE={metrics['methods'][method]['position_rmse_m']:.3f} m"
            ),
        )
    error_axis.axhline(0.5, color="0.4", ls="--", lw=0.9, label="0.5 m")
    error_axis.set_title("3D position error")
    error_axis.set_xlabel("time [s]")
    error_axis.set_ylabel("error norm [m]")
    error_axis.set_yscale("symlog", linthresh=0.1)
    error_axis.grid(alpha=0.25)
    error_axis.legend(fontsize=8)

    component_axis.plot(time_values, truth[:, 0], "k-", lw=1.8, label="truth x")
    component_axis.plot(time_values, truth[:, 1], "k--", lw=1.5, label="truth y")
    component_axis.plot(time_values, truth[:, 2], "k:", lw=1.8, label="truth z")
    lifted_position = histories["lifted_twr"]["position"]
    for component, label, style in zip(range(3), ("x", "y", "z"), ("-", "--", ":")):
        component_axis.plot(
            time_values,
            lifted_position[:, component],
            color=METHOD_COLORS["lifted_twr"],
            ls=style,
            lw=1.0,
            label=f"lifted {label}",
        )
    component_axis.set_title("Lifted position components")
    component_axis.set_xlabel("time [s]")
    component_axis.set_ylabel("position [m]")
    component_axis.grid(alpha=0.25)
    component_axis.legend(fontsize=8, ncol=3)

    for method, history in histories.items():
        velocity_error = np.linalg.norm(history["velocity"] - truth_velocity, axis=1)
        velocity_axis.plot(
            time_values,
            velocity_error,
            color=METHOD_COLORS[method],
            lw=1.1,
            label=(
                f"{method.replace('_', ' ')} "
                f"RMSE={metrics['methods'][method]['velocity_rmse_mps']:.3f} m/s"
            ),
        )
    velocity_axis.set_title("Velocity error (OptiTrack derivative)")
    velocity_axis.set_xlabel("time [s]")
    velocity_axis.set_ylabel("error norm [m/s]")
    velocity_axis.grid(alpha=0.25)
    velocity_axis.legend(fontsize=8)

    figure.suptitle(
        f"Task 06 real TWR: {metrics['sequence']} | "
        f"offset={metrics['initial_offset_m']} m",
        fontsize=13,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def theodosis_trajectory(time_values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = np.asarray(time_values)
    position = np.column_stack(
        (
            20.0 * np.cos(t) - 20.0,
            10.0 * np.sin(2.0 * t) + 20.0,
            -4.0 * np.cos(4.0 * t),
        )
    )
    velocity = np.column_stack(
        (
            -20.0 * np.sin(t),
            20.0 * np.cos(2.0 * t),
            16.0 * np.sin(4.0 * t),
        )
    )
    acceleration = np.column_stack(
        (
            -20.0 * np.cos(t),
            -40.0 * np.sin(2.0 * t),
            64.0 * np.cos(4.0 * t),
        )
    )
    return position, velocity, acceleration


def run_synthetic_single_range(
    *,
    seed: int,
    initial_position: Iterable[float],
    initial_velocity: Iterable[float] = (0.0, 0.0, 0.0),
    duration: float = 20.0,
    dt: float = 0.01,
    range_rate_hz: float = 20.0,
    range_sigma: float = 0.10,
    tracking_anchor_offsets: Iterable[Iterable[float]] | None = None,
    tracking_anchor_start_s: float = math.inf,
    true_accel_bias_mps2: Iterable[float] = (0.0, 0.0, 0.0),
    inekf_accel_bias_std_mps2: float = 1e-6,
) -> dict[str, Any]:
    """Run the original single-anchor case with an optional tracking phase.

    The default remains the Task06 single-anchor experiment exactly.  When
    ``tracking_anchor_offsets`` is supplied, the origin anchor is retained
    before ``tracking_anchor_start_s`` and the supplied offsets are cycled at
    the same total range-event rate afterwards.  This permits an end-to-end
    capture-then-track experiment without weakening the initial single-anchor
    large-error challenge.
    """

    rng = np.random.default_rng(seed)
    true_accel_bias = _as_vector3_for_experiment(
        true_accel_bias_mps2, "true_accel_bias_mps2"
    )
    accel_bias_std = float(inekf_accel_bias_std_mps2)
    if not np.isfinite(accel_bias_std) or accel_bias_std <= 0.0:
        raise ValueError(
            "inekf_accel_bias_std_mps2 must be finite and positive"
        )
    if tracking_anchor_offsets is None:
        tracking_offsets = np.empty((0, 3), dtype=np.float64)
        tracking_start = math.inf
    else:
        tracking_offsets = np.asarray(
            tuple(tracking_anchor_offsets), dtype=np.float64
        )
        tracking_start = float(tracking_anchor_start_s)
        if (
            tracking_offsets.ndim != 2
            or tracking_offsets.shape[0] == 0
            or tracking_offsets.shape[1] != 3
            or not np.all(np.isfinite(tracking_offsets))
        ):
            raise ValueError(
                "tracking_anchor_offsets must be a finite non-empty Nx3 array"
            )
        if not np.isfinite(tracking_start) or tracking_start < 0.0:
            raise ValueError(
                "tracking_anchor_start_s must be finite and non-negative"
            )
    time_values = np.arange(0.0, duration + 0.5 * dt, dt)
    truth_position, truth_velocity, truth_acceleration = theodosis_trajectory(
        time_values
    )
    # The author's InEKF contains an application-specific z>=0 ground clamp.
    # Translate the complete experiment (trajectory, anchor, and estimates)
    # by the same vector so the unconstrained Theodosis trajectory does not
    # trigger that guard.  Relative ranges, accelerations, initial errors, and
    # direct/lifted metrics are invariant to this coordinate translation.
    coordinate_offset = np.array([0.0, 0.0, 100.0])
    truth_position = truth_position + coordinate_offset
    initial_position_vector = (
        _as_vector3_for_experiment(initial_position, "initial_position")
        + coordinate_offset
    )
    initial_velocity_vector = _as_vector3_for_experiment(
        initial_velocity, "initial_velocity"
    )
    direct = DirectRangeEKF(
        initial_position_vector,
        initial_velocity_vector,
        position_std=60.0,
        velocity_std=30.0,
        accel_noise_std=0.25,
        range_sigma=range_sigma,
        range_bias=0.0,
        gate_threshold=25.0,
        gate_warmup_updates=80,
    )
    lifted = LiftedRangeKF(
        initial_position_vector,
        initial_velocity_vector,
        position_std=60.0,
        velocity_std=30.0,
        q_std=3000.0,
        s_std=3000.0,
        h_std=3000.0,
        accel_noise_std=0.25,
        range_sigma=range_sigma,
        range_bias=0.0,
        gate_threshold=25.0,
        gate_warmup_updates=80,
    )
    ordinary_state = np.eye(5)
    ordinary_state[:3, 3] = initial_velocity_vector
    ordinary_state[:3, 4] = initial_position_vector
    ordinary = TWRInEKF(
        ordinary_state,
        paper_consistent_initial_covariance(
            attitude_std=(1e-3, 1e-3, 1e-3),
            velocity_std=30.0,
            position_std=60.0,
            gyro_bias_std=1e-6,
            accel_bias_std=accel_bias_std,
        ),
        np.zeros(6),
        range_sigma=range_sigma,
        range_bias=0.0,
        gate_threshold=25.0,
        gate_warmup_updates=80,
        lever_arm=(0.0, 0.0, 0.0),
    )
    # Match the reduced translational experiment as closely as the author's
    # covariance model allows: attitude and biases are effectively known,
    # while acceleration uncertainty matches the two translational filters.
    ordinary.w_gyr[:] = 1e-8
    ordinary.w_acc[:] = 0.25**2
    ordinary.w_gyr_b[:] = 1e-12
    ordinary.w_acc_b[:] = 1e-12

    capture_anchor = coordinate_offset.copy()
    update_stride = max(1, int(round(1.0 / (range_rate_hz * dt))))
    tracking_anchor_event = 0
    direct_position = np.empty_like(truth_position)
    direct_velocity = np.empty_like(truth_velocity)
    lifted_position = np.empty_like(truth_position)
    lifted_velocity = np.empty_like(truth_velocity)
    ordinary_position = np.empty_like(truth_position)
    ordinary_velocity = np.empty_like(truth_velocity)
    constraint = np.empty_like(truth_position)

    for index in range(time_values.size):
        if index > 0:
            noisy_acceleration = (
                truth_acceleration[index - 1]
                + rng.normal(0.0, 0.10, size=3)
            )
            measured_acceleration = noisy_acceleration + true_accel_bias
            direct.predict(measured_acceleration, dt)
            lifted.predict(measured_acceleration, dt)
            imu_sample = np.concatenate(
                (
                    (
                        measured_acceleration
                        + np.array([0.0, 0.0, GRAVITY_MAGNITUDE])
                    )
                    / GRAVITY_MAGNITUDE,
                    np.zeros(3),
                )
            )
            ordinary.predict(imu_sample, dt)
        if index % update_stride == 0:
            if time_values[index] + 1e-12 >= tracking_start:
                anchor = (
                    coordinate_offset
                    + tracking_offsets[
                        tracking_anchor_event % tracking_offsets.shape[0]
                    ]
                )
                tracking_anchor_event += 1
            else:
                anchor = capture_anchor
            measured_range = (
                np.linalg.norm(truth_position[index] - anchor)
                + rng.normal(0.0, range_sigma)
            )
            direct.correct(anchor, measured_range)
            lifted.correct(anchor, measured_range)
            ordinary.correct_twr(anchor, measured_range)
        direct_position[index] = direct.position
        direct_velocity[index] = direct.velocity
        lifted_position[index] = lifted.position
        lifted_velocity[index] = lifted.velocity
        ordinary_position[index] = ordinary.X[:3, 4]
        ordinary_velocity[index] = ordinary.X[:3, 3]
        constraint[index] = lifted.constraint_residual()

    histories = {
        "direct_twr": {
            "position": direct_position,
            "velocity": direct_velocity,
        },
        "lifted_twr": {
            "position": lifted_position,
            "velocity": lifted_velocity,
            "constraint": constraint,
        },
        "twr_inekf": {
            "position": ordinary_position,
            "velocity": ordinary_velocity,
        },
    }
    filters = {
        "direct_twr": direct,
        "lifted_twr": lifted,
        "twr_inekf": ordinary,
    }
    methods = {}
    for name, history in histories.items():
        methods[name] = trajectory_metrics(
            time_values,
            history["position"],
            history["velocity"],
            truth_position,
            truth_velocity,
        )
        methods[name]["success_tail5_below_1m"] = (
            methods[name]["position_tail5_rmse_m"] < 1.0
        )
        methods[name]["accepted"] = int(filters[name].stats.accepted)
        methods[name]["rejected"] = int(filters[name].stats.rejected)
        methods[name]["invalid"] = int(filters[name].stats.invalid)
    return {
        "metrics": {
            "seed": int(seed),
            "duration_s": float(duration),
            "range_rate_hz": float(range_rate_hz),
            "range_sigma_m": float(range_sigma),
            "range_geometry": (
                "single anchor"
                if tracking_offsets.shape[0] == 0
                else "single-anchor capture followed by cyclic multi-anchor tracking"
            ),
            "tracking_anchor_start_s": (
                None if not np.isfinite(tracking_start) else tracking_start
            ),
            "tracking_anchor_offsets_m": tracking_offsets.tolist(),
            "true_accel_bias_mps2": true_accel_bias.tolist(),
            "inekf_initial_accel_bias_std_mps2": accel_bias_std,
            "initial_position_m": (
                initial_position_vector - coordinate_offset
            ).tolist(),
            "initial_velocity_mps": initial_velocity_vector.tolist(),
            "coordinate_offset_m": coordinate_offset.tolist(),
            "coordinate_offset_reason": (
                "avoid the author's application-specific z>=0 ground clamp; "
                "all geometry is translated together"
            ),
            "methods": methods,
        },
        "time": time_values,
        "ground_truth_position": truth_position,
        "ground_truth_velocity": truth_velocity,
        "histories": histories,
    }


def plot_synthetic_result(result: dict[str, Any], output_path: Path) -> None:
    time_values = result["time"]
    truth = result["ground_truth_position"]
    histories = result["histories"]
    figure, axes = plt.subplots(1, 3, figsize=(15.0, 4.6), constrained_layout=True)

    axes[0].plot(truth[:, 0], truth[:, 1], "k-", lw=2.0, label="truth")
    for method, history in histories.items():
        axes[0].plot(
            history["position"][:, 0],
            history["position"][:, 1],
            color=METHOD_COLORS[method],
            lw=1.1,
            label=method.replace("_", " "),
        )
    axes[0].scatter([0], [0], marker="^", color="k", s=55, label="single anchor")
    axes[0].set_title("XY trajectory")
    axes[0].set_xlabel("x [m]")
    axes[0].set_ylabel("y [m]")
    axes[0].axis("equal")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)

    for method, history in histories.items():
        error = np.linalg.norm(history["position"] - truth, axis=1)
        axes[1].plot(
            time_values,
            error,
            color=METHOD_COLORS[method],
            lw=1.15,
            label=(
                f"{method.replace('_', ' ')} "
                f"tail={result['metrics']['methods'][method]['position_tail5_rmse_m']:.3f} m"
            ),
        )
    axes[1].axhline(1.0, color="0.4", ls="--", lw=0.9)
    axes[1].set_yscale("log")
    axes[1].set_title("Single-range position convergence")
    axes[1].set_xlabel("time [s]")
    axes[1].set_ylabel("error norm [m]")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)

    residual = histories["lifted_twr"]["constraint"]
    labels = (
        r"$q-\frac{1}{2}p^\top p$",
        r"$s-p^\top v$",
        r"$h-v^\top v$",
    )
    for component, label in enumerate(labels):
        axes[2].plot(time_values, np.abs(residual[:, component]), lw=1.0, label=label)
    axes[2].set_yscale("symlog", linthresh=1e-5)
    axes[2].set_title("Lift consistency residuals (diagnostic)")
    axes[2].set_xlabel("time [s]")
    axes[2].set_ylabel("absolute residual")
    axes[2].grid(alpha=0.25)
    axes[2].legend(fontsize=8)

    figure.suptitle(
        "Theodosis single-range PE experiment | "
        f"initial p={result['metrics']['initial_position_m']}",
        fontsize=12,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def run_synthetic_monte_carlo(
    output_directory: Path, *, trials: int = 20, seed: int = 20260728
) -> dict[str, Any]:
    """Evaluate large-initial-error convergence without selecting one offset."""

    generator = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for trial in range(trials):
        initial_position = generator.uniform(-50.0, 50.0, size=3)
        initial_velocity = generator.uniform(-10.0, 10.0, size=3)
        result = run_synthetic_single_range(
            seed=trial,
            initial_position=initial_position,
            initial_velocity=initial_velocity,
        )
        for method, metrics in result["metrics"]["methods"].items():
            rows.append(
                {
                    "trial": trial,
                    "method": method,
                    "initial_px_m": initial_position[0],
                    "initial_py_m": initial_position[1],
                    "initial_pz_m": initial_position[2],
                    "initial_vx_mps": initial_velocity[0],
                    "initial_vy_mps": initial_velocity[1],
                    "initial_vz_mps": initial_velocity[2],
                    **metrics,
                }
            )

    table = pd.DataFrame(rows)
    output_directory.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_directory / "monte_carlo_metrics.csv", index=False)
    summary: dict[str, Any] = {
        "trials": int(trials),
        "seed": int(seed),
        "initial_position_distribution_m": "independent uniform[-50, 50]",
        "initial_velocity_distribution_mps": "independent uniform[-10, 10]",
        "success_definition": "last-5-s position RMSE < 1 m",
        "methods": {},
    }
    for method, group in table.groupby("method"):
        summary["methods"][method] = {
            "success_count": int(group["success_tail5_below_1m"].sum()),
            "success_rate": float(group["success_tail5_below_1m"].mean()),
            "tail5_rmse_median_m": float(group["position_tail5_rmse_m"].median()),
            "tail5_rmse_p95_m": float(
                np.percentile(group["position_tail5_rmse_m"], 95)
            ),
            "full_rmse_median_m": float(group["position_rmse_m"].median()),
        }
    with (output_directory / "monte_carlo_summary.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(_json_ready(summary), stream, ensure_ascii=False, indent=2)
        stream.write("\n")

    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.4), constrained_layout=True)
    method_order = ["direct_twr", "twr_inekf", "lifted_twr"]
    method_labels = ["direct TWR", "TWR-InEKF", "lifted TWR"]
    success_rates = [
        summary["methods"][method]["success_rate"] for method in method_order
    ]
    axes[0].bar(
        method_labels,
        success_rates,
        color=[METHOD_COLORS[method] for method in method_order],
    )
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_ylabel("success rate")
    axes[0].set_title(f"Large-error convergence ({trials} trials)")
    for index, rate in enumerate(success_rates):
        axes[0].text(index, rate + 0.025, f"{rate:.0%}", ha="center")
    axes[0].grid(axis="y", alpha=0.25)

    box_values = [
        table.loc[table["method"] == method, "position_tail5_rmse_m"].to_numpy()
        for method in method_order
    ]
    boxes = axes[1].boxplot(
        box_values,
        tick_labels=method_labels,
        patch_artist=True,
        showfliers=True,
    )
    for patch, method in zip(boxes["boxes"], method_order):
        patch.set_facecolor(METHOD_COLORS[method])
        patch.set_alpha(0.75)
    axes[1].set_yscale("log")
    axes[1].axhline(1.0, color="0.35", ls="--", lw=0.9, label="success threshold")
    axes[1].set_ylabel("last-5-s position RMSE [m]")
    axes[1].set_title("Terminal accuracy distribution")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(fontsize=8)
    figure.suptitle("Theodosis single-range PE Monte Carlo", fontsize=12)
    figure.savefig(output_directory / "monte_carlo.png", dpi=180)
    plt.close(figure)
    return summary


def save_result(result: dict[str, Any], output_directory: Path) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    with (output_directory / "metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(
            _json_ready(result["metrics"]),
            stream,
            ensure_ascii=False,
            indent=2,
        )
        stream.write("\n")
    arrays: dict[str, np.ndarray] = {
        "time": result["time"],
        "ground_truth_position": result["ground_truth_position"],
        "ground_truth_velocity": result["ground_truth_velocity"],
    }
    for method, history in result["histories"].items():
        for name, values in history.items():
            arrays[f"{method}_{name}"] = values
    np.savez_compressed(output_directory / "timeseries.npz", **arrays)


def real_sequence_files(dataset: Path) -> list[str]:
    return sorted(path.name for path in (dataset / "twr").glob("*.csv"))


def run_full_batch(
    dataset: Path,
    output_directory: Path,
    *,
    duration: float | None,
    range_sigma: float,
    range_bias: float,
) -> dict[str, Any]:
    anchors = load_anchors(dataset)
    offsets = {
        "nominal": np.array([0.0, 0.0, 0.0]),
        "large": np.array([6.0, -6.0, 3.0]),
    }
    rows: list[dict[str, Any]] = []
    for sequence in real_sequence_files(dataset):
        log = load_real_log(dataset, sequence, duration)
        for offset_name, offset in offsets.items():
            result = run_real_sequence(
                log,
                anchors,
                initial_offset=offset,
                range_sigma=range_sigma,
                range_bias=range_bias,
                include_ordinary_inekf=True,
            )
            for method, values in result["metrics"]["methods"].items():
                rows.append(
                    {
                        "sequence": sequence,
                        "offset": offset_name,
                        "method": method,
                        **values,
                    }
                )
    table = pd.DataFrame(rows)
    output_directory.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_directory / "real_batch_metrics.csv", index=False)

    summary: dict[str, Any] = {"rows": len(rows), "groups": {}}
    for (offset_name, method), group in table.groupby(["offset", "method"]):
        key = f"{offset_name}/{method}"
        summary["groups"][key] = {
            "count": int(group.shape[0]),
            "position_rmse_median_m": float(group["position_rmse_m"].median()),
            "position_rmse_mean_m": float(group["position_rmse_m"].mean()),
            "tail5_rmse_median_m": float(group["position_tail5_rmse_m"].median()),
            "converged_0p5m_count": int(group["convergence_0p5m_s"].notna().sum()),
            "finite_count": int(group["finite"].sum()),
            "rejected_total": int(group["rejected"].sum()),
        }
    summary["paired_comparison"] = {}
    for offset_name in offsets:
        subset = table.loc[table["offset"] == offset_name]
        pivot = subset.pivot(
            index="sequence", columns="method", values="position_rmse_m"
        )
        offset_comparisons: dict[str, Any] = {}
        for first, second in (
            ("lifted_twr", "direct_twr"),
            ("twr_inekf", "direct_twr"),
            ("lifted_twr", "twr_inekf"),
        ):
            difference = pivot[first] - pivot[second]
            key = f"{first}_minus_{second}"
            offset_comparisons[key] = {
                "first_lower_rmse_count": int(np.count_nonzero(difference < 0)),
                "second_lower_rmse_count": int(np.count_nonzero(difference > 0)),
                "ties_count": int(np.count_nonzero(difference == 0)),
                "median_m": float(np.median(difference)),
                "mean_m": float(np.mean(difference)),
            }
        summary["paired_comparison"][offset_name] = offset_comparisons
    with (output_directory / "real_batch_summary.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(_json_ready(summary), stream, ensure_ascii=False, indent=2)
        stream.write("\n")

    figure, axes = plt.subplots(1, 2, figsize=(14.0, 5.0), constrained_layout=True)
    for axis, offset_name in zip(axes, offsets):
        subset = table.loc[table["offset"] == offset_name].copy()
        labels = [
            Path(name).stem.replace("2021-02-12_", "").replace("_twr_", "\n")
            for name in sorted(subset["sequence"].unique())
        ]
        x_values = np.arange(len(labels))
        for method in ("direct_twr", "twr_inekf", "lifted_twr"):
            method_rows = (
                subset.loc[subset["method"] == method]
                .sort_values("sequence")
                .reset_index(drop=True)
            )
            axis.plot(
                x_values,
                method_rows["position_rmse_m"],
                marker="o",
                ms=4.5,
                lw=1.2,
                color=METHOD_COLORS[method],
                label=(
                    f"{method.replace('_', ' ')} "
                    f"median={method_rows['position_rmse_m'].median():.3f} m"
                ),
            )
        axis.set_xticks(x_values, labels, rotation=45, ha="right", fontsize=7)
        axis.set_ylabel("full-sequence position RMSE [m]")
        axis.set_title(
            "zero initial offset"
            if offset_name == "nominal"
            else r"initial offset $[6,-6,3]$ m"
        )
        if offset_name == "large":
            # Ordinary TWR-InEKF has two genuine large-error failures.  A log
            # scale keeps those failures visible without flattening the
            # successful direct/lifted runs near 0.3--0.7 m.
            axis.set_yscale("log")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.suptitle("TU Delft real TWR batch (fixed parameters)", fontsize=13)
    figure.savefig(output_directory / "real_batch_comparison.png", dpi=180)
    plt.close(figure)
    return summary


def run_tail_diagnostic_sequence(
    log: RealLog,
    anchors: np.ndarray,
    *,
    range_sigma: float,
    range_bias: float,
    process_noise_values: tuple[float, ...] = (5.0, 10.0, 20.0, 40.0),
) -> list[dict[str, Any]]:
    """Replay one log with ablations that isolate lifted steady-state error."""

    initial_state, covariance, bias = _initial_state(log, np.zeros(3))
    imu_core = InEKF(
        initial_state.copy(), covariance.copy(), bias.copy()
    )
    common = {
        "position_std": 8.0,
        "velocity_std": 2.0,
        "range_sigma": range_sigma,
        "range_bias": range_bias,
        "gate_threshold": 25.0,
        "gate_warmup_updates": 64,
    }
    filters: dict[str, Any] = {
        "direct_twr": DirectRangeEKF(
            initial_state[:3, 4],
            initial_state[:3, 3],
            accel_noise_std=10.0,
            **common,
        ),
        "squared_twr": SquaredRangeEKF(
            initial_state[:3, 4],
            initial_state[:3, 3],
            accel_noise_std=10.0,
            **common,
        ),
        "lifted_projected": LiftedRangeKF(
            initial_state[:3, 4],
            initial_state[:3, 3],
            q_std=80.0,
            s_std=40.0,
            h_std=20.0,
            accel_noise_std=10.0,
            project_constraints=True,
            **common,
        ),
    }
    for process_noise in process_noise_values:
        method = (
            "lifted_twr"
            if math.isclose(process_noise, 10.0)
            else f"lifted_q{process_noise:g}"
        )
        filters[method] = LiftedRangeKF(
            initial_state[:3, 4],
            initial_state[:3, 3],
            q_std=80.0,
            s_std=40.0,
            h_std=20.0,
            accel_noise_std=process_noise,
            **common,
        )

    sample_count = log.time.size
    histories = {
        method: {
            "position": np.empty((sample_count, 3)),
            "velocity": np.empty((sample_count, 3)),
        }
        for method in filters
    }
    constraint_histories = {
        method: np.empty((sample_count, 3))
        for method, filter_instance in filters.items()
        if isinstance(filter_instance, LiftedRangeKF)
    }

    for index in range(sample_count):
        if index > 0:
            dt = float(log.time[index] - log.time[index - 1])
            sample = log.imu[index - 1]
            world_acceleration = imu_world_acceleration(
                imu_core.X[:3, :3], sample, imu_core.bias
            )
            imu_core.predict(sample, dt)
            for filter_instance in filters.values():
                filter_instance.predict(world_acceleration, dt)

        for anchor_index in np.flatnonzero(log.changed[index]):
            measured_range = float(log.ranges[index, anchor_index])
            for filter_instance in filters.values():
                filter_instance.correct(
                    anchors[anchor_index], measured_range
                )

        for method, filter_instance in filters.items():
            histories[method]["position"][index] = filter_instance.position
            histories[method]["velocity"][index] = filter_instance.velocity
            if method in constraint_histories:
                constraint_histories[method][index] = (
                    filter_instance.constraint_residual()
                )

    tail = log.time >= max(0.0, log.time[-1] - 5.0)
    rows: list[dict[str, Any]] = []
    for method, history in histories.items():
        metrics = trajectory_metrics(
            log.time,
            history["position"],
            history["velocity"],
            log.ground_truth_position,
            log.ground_truth_velocity,
        )
        filter_instance = filters[method]
        row: dict[str, Any] = {
            "sequence": log.name,
            "method": method,
            **metrics,
            "accepted": int(filter_instance.stats.accepted),
            "rejected": int(filter_instance.stats.rejected),
        }
        if method in constraint_histories:
            residual = constraint_histories[method]
            row["constraint_tail_rms"] = np.sqrt(
                np.mean(residual[tail] ** 2, axis=0)
            ).tolist()
            row["constraint_final_abs"] = np.abs(residual[-1]).tolist()
        else:
            row["constraint_tail_rms"] = None
            row["constraint_final_abs"] = None
        rows.append(row)
    return rows


def run_tail_diagnostic_batch(
    dataset: Path,
    output_directory: Path,
    *,
    duration: float | None,
    range_sigma: float,
    range_bias: float,
) -> dict[str, Any]:
    """Run all real logs once with causal steady-state-error ablations."""

    anchors = load_anchors(dataset)
    rows: list[dict[str, Any]] = []
    for sequence in real_sequence_files(dataset):
        log = load_real_log(dataset, sequence, duration)
        rows.extend(
            run_tail_diagnostic_sequence(
                log,
                anchors,
                range_sigma=range_sigma,
                range_bias=range_bias,
            )
        )

    table = pd.DataFrame(rows)
    output_directory.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_directory / "tail_ablation_metrics.csv", index=False)

    summary: dict[str, Any] = {
        "sequences": int(table["sequence"].nunique()),
        "rows": int(table.shape[0]),
        "scope": "nominal initial position; final 5 seconds of every real TWR log",
        "methods": {},
    }
    direct_tail = (
        table.loc[table["method"] == "direct_twr"]
        .set_index("sequence")["position_tail5_rmse_m"]
        .sort_index()
    )
    for method, group in table.groupby("method"):
        aligned = group.set_index("sequence").sort_index()
        tail_difference = (
            aligned["position_tail5_rmse_m"] - direct_tail
        )
        summary["methods"][method] = {
            "position_rmse_median_m": float(
                aligned["position_rmse_m"].median()
            ),
            "position_tail5_rmse_median_m": float(
                aligned["position_tail5_rmse_m"].median()
            ),
            "velocity_tail5_rmse_median_mps": float(
                aligned["velocity_tail5_rmse_mps"].median()
            ),
            "tail5_worse_than_direct_count": int(
                np.count_nonzero(tail_difference > 0)
            ),
            "tail5_better_than_direct_count": int(
                np.count_nonzero(tail_difference < 0)
            ),
            "tail5_minus_direct_median_m": float(
                np.median(tail_difference)
            ),
        }

    lifted_rows = (
        table.loc[table["method"] == "lifted_twr"]
        .set_index("sequence")
        .sort_index()
    )
    lifted_penalty = (
        lifted_rows["position_tail5_rmse_m"] - direct_tail
    )
    q_tail_rms = np.array(
        [
            json.loads(value)[0]
            if isinstance(value, str)
            else value[0]
            for value in lifted_rows["constraint_tail_rms"]
        ],
        dtype=np.float64,
    )
    correlation = spearmanr(q_tail_rms, lifted_penalty.to_numpy())
    summary["constraint_diagnosis"] = {
        "quantity": (
            "Spearman correlation between tail RMS of "
            "q-0.5*p^T*p and lifted-minus-direct tail position RMSE"
        ),
        "rho": float(correlation.statistic),
        "p_value_two_sided": float(correlation.pvalue),
        "sample_count": int(q_tail_rms.size),
    }

    squared_rows = (
        table.loc[table["method"] == "squared_twr"]
        .set_index("sequence")
        .sort_index()
    )
    projected_rows = (
        table.loc[table["method"] == "lifted_projected"]
        .set_index("sequence")
        .sort_index()
    )
    summary["projection_equivalence"] = {
        "max_abs_full_rmse_difference_m": float(
            np.max(
                np.abs(
                    squared_rows["position_rmse_m"]
                    - projected_rows["position_rmse_m"]
                )
            )
        ),
        "max_abs_tail5_rmse_difference_m": float(
            np.max(
                np.abs(
                    squared_rows["position_tail5_rmse_m"]
                    - projected_rows["position_tail5_rmse_m"]
                )
            )
        ),
        "interpretation": (
            "projection removes independent q/s/h freedom and numerically "
            "reduces the lifted filter to the six-state squared-range EKF"
        ),
    }

    with (output_directory / "tail_ablation_summary.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(_json_ready(summary), stream, ensure_ascii=False, indent=2)
        stream.write("\n")

    figure, axes = plt.subplots(
        2, 2, figsize=(13.5, 9.0), constrained_layout=True
    )
    comparison_methods = [
        "direct_twr",
        "squared_twr",
        "lifted_twr",
        "lifted_projected",
    ]
    comparison_labels = [
        "direct range",
        "squared 6-state",
        "lifted",
        "lifted + projection",
    ]
    comparison_colors = ["#d95f02", "#e6ab02", "#1b9e77", "#66a61e"]
    values = [
        table.loc[
            table["method"] == method, "position_tail5_rmse_m"
        ].to_numpy()
        for method in comparison_methods
    ]
    boxes = axes[0, 0].boxplot(
        values,
        tick_labels=comparison_labels,
        patch_artist=True,
        showfliers=True,
    )
    for patch, color in zip(boxes["boxes"], comparison_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    axes[0, 0].set_ylabel("last-5-s position RMSE [m]")
    axes[0, 0].set_title("Which mechanism causes the terminal penalty?")
    axes[0, 0].tick_params(axis="x", rotation=15)
    axes[0, 0].grid(axis="y", alpha=0.25)

    axes[0, 1].scatter(
        q_tail_rms,
        lifted_penalty,
        color=METHOD_COLORS["lifted_twr"],
        edgecolor="white",
        linewidth=0.6,
        s=55,
    )
    axes[0, 1].axhline(0.0, color="0.35", ls="--", lw=0.9)
    axes[0, 1].set_xlabel(
        r"tail RMS of $q-\frac{1}{2}p^\top p$ [$m^2$]"
    )
    axes[0, 1].set_ylabel("lifted - direct tail RMSE [m]")
    axes[0, 1].set_title(
        f"Lift inconsistency vs penalty (Spearman "
        f"$\\rho$={correlation.statistic:.3f})"
    )
    axes[0, 1].grid(alpha=0.25)

    process_values = np.array([5.0, 10.0, 20.0, 40.0])
    process_methods = [
        "lifted_q5",
        "lifted_twr",
        "lifted_q20",
        "lifted_q40",
    ]
    position_tail_medians = [
        summary["methods"][method]["position_tail5_rmse_median_m"]
        for method in process_methods
    ]
    full_medians = [
        summary["methods"][method]["position_rmse_median_m"]
        for method in process_methods
    ]
    velocity_tail_medians = [
        summary["methods"][method]["velocity_tail5_rmse_median_mps"]
        for method in process_methods
    ]
    axes[1, 0].plot(
        process_values,
        position_tail_medians,
        "o-",
        color=METHOD_COLORS["lifted_twr"],
        label="last-5-s position",
    )
    axes[1, 0].plot(
        process_values,
        full_medians,
        "s--",
        color="#7570b3",
        label="full-sequence position",
    )
    axes[1, 0].set_xscale("log", base=2)
    axes[1, 0].set_xticks(process_values, [f"{value:g}" for value in process_values])
    axes[1, 0].set_xlabel("effective acceleration process std [m/s²]")
    axes[1, 0].set_ylabel("median position RMSE [m]")
    axes[1, 0].set_title("Process-noise trade-off")
    axes[1, 0].grid(alpha=0.25)
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].plot(
        process_values,
        velocity_tail_medians,
        "o-",
        color="#7570b3",
    )
    axes[1, 1].set_xscale("log", base=2)
    axes[1, 1].set_xticks(process_values, [f"{value:g}" for value in process_values])
    axes[1, 1].set_xlabel("effective acceleration process std [m/s²]")
    axes[1, 1].set_ylabel("median last-5-s velocity RMSE [m/s]")
    axes[1, 1].set_title("Larger Q eventually amplifies velocity noise")
    axes[1, 1].grid(alpha=0.25)

    figure.suptitle(
        "Task 06 terminal-error diagnosis: all 12 TU Delft TWR logs",
        fontsize=13,
    )
    figure.savefig(output_directory / "tail_ablation.png", dpi=180)
    plt.close(figure)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Task 06: compare direct and lifted TWR filtering"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="TU Delft dataset root containing anchor_positions.yaml and twr/",
    )
    parser.add_argument("--sequence", default=DEFAULT_SEQUENCE)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument(
        "--offset",
        type=float,
        nargs=3,
        default=(6.0, -6.0, 3.0),
        metavar=("DX", "DY", "DZ"),
    )
    parser.add_argument("--range-sigma", type=float, default=0.20)
    parser.add_argument("--range-bias", type=float, default=-0.18)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results") / "task06_quick",
    )
    parser.add_argument(
        "--mode",
        choices=("real", "synthetic", "full", "diagnostic"),
        default="real",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if args.mode == "synthetic":
        result = run_synthetic_single_range(
            seed=7,
            initial_position=(35.0, -25.0, 20.0),
            initial_velocity=(0.0, 0.0, 0.0),
        )
        save_result(result, output)
        plot_synthetic_result(result, output / "comparison.png")
        print(json.dumps(_json_ready(result["metrics"]), ensure_ascii=False, indent=2))
        return

    if not args.dataset.is_dir():
        raise FileNotFoundError(
            f"Dataset not found: {args.dataset}. Run download_dataset.ps1 first."
        )
    if args.mode == "diagnostic":
        summary = run_tail_diagnostic_batch(
            args.dataset,
            output,
            duration=args.duration,
            range_sigma=args.range_sigma,
            range_bias=args.range_bias,
        )
        print(json.dumps(_json_ready(summary), ensure_ascii=False, indent=2))
        return
    if args.mode == "full":
        summary = run_full_batch(
            args.dataset,
            output,
            duration=args.duration,
            range_sigma=args.range_sigma,
            range_bias=args.range_bias,
        )
        representative_log = load_real_log(
            args.dataset, DEFAULT_SEQUENCE, args.duration
        )
        representative = run_real_sequence(
            representative_log,
            load_anchors(args.dataset),
            initial_offset=(6.0, -6.0, 3.0),
            range_sigma=args.range_sigma,
            range_bias=args.range_bias,
            include_ordinary_inekf=True,
        )
        save_result(representative, output / "real_representative_large")
        plot_real_result(
            representative,
            output / "real_representative_large" / "comparison.png",
        )
        synthetic = run_synthetic_single_range(
            seed=7,
            initial_position=(35.0, -25.0, 20.0),
            initial_velocity=(0.0, 0.0, 0.0),
        )
        save_result(synthetic, output / "synthetic_representative")
        plot_synthetic_result(
            synthetic, output / "synthetic_representative" / "comparison.png"
        )
        monte_carlo = run_synthetic_monte_carlo(
            output / "synthetic_monte_carlo", trials=20
        )
        combined = {
            "real_batch": summary,
            "synthetic_monte_carlo": monte_carlo,
        }
        print(json.dumps(_json_ready(combined), ensure_ascii=False, indent=2))
        return

    anchors = load_anchors(args.dataset)
    log = load_real_log(args.dataset, args.sequence, args.duration)
    result = run_real_sequence(
        log,
        anchors,
        initial_offset=args.offset,
        range_sigma=args.range_sigma,
        range_bias=args.range_bias,
        include_ordinary_inekf=True,
    )
    save_result(result, output)
    plot_real_result(result, output / "comparison.png")
    print(json.dumps(_json_ready(result["metrics"]), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
