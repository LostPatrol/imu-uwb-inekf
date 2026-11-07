"""Compare the author's pose propagation with the task-04 update.

Only the deterministic velocity and position propagation differs between the
two filters.  Both filters receive the same initialization, IMU samples, UWB
samples, timestamps, covariance logic, and rejection logic.
"""

import argparse
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

# Keep numerical library behavior predictable and avoid oversubscribing this
# small-matrix workload before importing NumPy/SciPy.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.spatial.transform import Rotation, Slerp


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPOSITORY_ROOT.parent
DEFAULT_AUTHOR_REPOSITORY = WORKSPACE_ROOT / "imu-uwb-inekf - backup"
IMPROVED_INEKF = REPOSITORY_ROOT / "scripts" / "estimation" / "inekf.py"
AUTHOR_COMMIT = "f3b664e0e8c86b62c6ed2ea2a112261810228e1c"
TDOA2_PAIRS = (
    (7, 0),
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 5),
    (5, 6),
    (6, 7),
)


@dataclass(frozen=True)
class Dataset:
    anchors: np.ndarray
    t_gt: np.ndarray
    p_gt: np.ndarray
    q_gt_xyzw: np.ndarray
    t_imu: np.ndarray
    imu: np.ndarray
    t_uwb: np.ndarray
    uwb: np.ndarray
    uwb_source_count: int
    uwb_requested_rate_hz: float | None
    uwb_actual_rate_hz: float


@dataclass
class ReplayResult:
    label: str
    times: np.ndarray
    rotations: np.ndarray
    velocities: np.ndarray
    positions: np.ndarray
    biases: np.ndarray
    elapsed_seconds: float
    prediction_count: int
    uwb_count: int
    rejection_count: int
    metrics: dict


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare the untouched author pose propagation against the "
            "J_l/Q_l task-04 update on one UTIL CSV."
        )
    )
    parser.add_argument(
        "-i",
        nargs=2,
        required=True,
        metavar=("ANCHORS", "CSV"),
        help="anchor_const*.npz and one flight-dataset CSV",
    )
    parser.add_argument(
        "--author-repo",
        type=Path,
        default=DEFAULT_AUTHOR_REPOSITORY,
        help="read-only author backup used as the before baseline",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "results" / "pose_update_comparison",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        help="optional replay duration in seconds; omit for the complete file",
    )
    parser.add_argument(
        "--uwb-rate-hz",
        type=float,
        help=(
            "optional aggregate TDoA update rate; selects the source sample "
            "nearest each uniform time-grid point without interpolation"
        ),
    )
    parser.add_argument("--yaw-deg", type=float, default=0.0)
    parser.add_argument(
        "--tdoa-variance",
        type=float,
        default=0.13,
        help="kept at the author's main.py value by default",
    )
    return parser.parse_args()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def clean_rows(frame, columns):
    return frame.loc[:, columns].dropna().to_numpy(dtype=np.float64)


def select_nearest_rate_samples(tdoa, rate_hz):
    """Select original TDoA rows nearest a uniform aggregate-rate grid."""
    if rate_hz is None:
        return tdoa
    if not math.isfinite(rate_hz) or rate_hz <= 0.0:
        raise ValueError("--uwb-rate-hz must be finite and positive")
    if len(tdoa) < 2:
        return tdoa

    times = tdoa[:, 0]
    period = 1.0 / rate_hz
    targets = np.arange(
        times[0], times[-1] + 0.5 * period, period, dtype=np.float64
    )
    right = np.searchsorted(times, targets, side="left")
    right = np.clip(right, 0, len(times) - 1)
    left = np.maximum(right - 1, 0)
    choose_left = np.abs(targets - times[left]) <= np.abs(
        times[right] - targets
    )
    indices = np.where(choose_left, left, right)
    indices = np.unique(indices)
    return tdoa[indices]


def sample_rate_hz(times):
    if len(times) < 2 or times[-1] <= times[0]:
        return math.nan
    return float((len(times) - 1) / (times[-1] - times[0]))


def load_dataset(anchor_path, csv_path, max_duration, uwb_rate_hz=None):
    anchors = np.load(anchor_path)["an_pos"].astype(np.float64)
    frame = pd.read_csv(csv_path)
    ground_truth = clean_rows(
        frame,
        (
            "t_pose",
            "pose_x",
            "pose_y",
            "pose_z",
            "pose_qx",
            "pose_qy",
            "pose_qz",
            "pose_qw",
        ),
    )
    accelerometer = clean_rows(
        frame, ("t_acc", "acc_x", "acc_y", "acc_z")
    )
    gyroscope = clean_rows(
        frame, ("t_gyro", "gyro_x", "gyro_y", "gyro_z")
    )
    tdoa = clean_rows(frame, ("t_tdoa", "idA", "idB", "tdoa_meas"))

    synchronized_gyro = np.column_stack(
        [
            np.interp(accelerometer[:, 0], gyroscope[:, 0], gyroscope[:, axis])
            for axis in range(1, 4)
        ]
    )
    imu = np.column_stack((accelerometer[:, 1:4], synchronized_gyro))

    minimum_time = min(
        float(tdoa[0, 0]),
        float(accelerometer[0, 0]),
        float(ground_truth[0, 0]),
    )
    ground_truth = ground_truth[ground_truth[:, 0] > minimum_time].copy()
    ground_truth[:, 0] -= minimum_time
    t_imu = accelerometer[:, 0] - minimum_time

    tdoa = tdoa.copy()
    tdoa[:, 0] -= minimum_time
    pair_blocks = []
    for anchor_a, anchor_b in TDOA2_PAIRS:
        mask = (tdoa[:, 1] == anchor_a) & (tdoa[:, 2] == anchor_b)
        if np.any(mask):
            pair_blocks.append(tdoa[mask])
    if not pair_blocks:
        raise ValueError("No author-main TDoA2 anchor pairs found in the CSV")
    tdoa = np.concatenate(pair_blocks, axis=0)
    tdoa = tdoa[np.argsort(tdoa[:, 0], kind="stable")]

    if max_duration is not None:
        if max_duration <= 0.0:
            raise ValueError("--max-duration must be positive")
        imu_mask = t_imu <= max_duration
        uwb_mask = tdoa[:, 0] <= max_duration
        t_imu = t_imu[imu_mask]
        imu = imu[imu_mask]
        tdoa = tdoa[uwb_mask]

    uwb_source_count = len(tdoa)
    tdoa = select_nearest_rate_samples(tdoa, uwb_rate_hz)

    if len(t_imu) < 2 or len(tdoa) < 1:
        raise ValueError("Duration leaves too few IMU or UWB measurements")

    return Dataset(
        anchors=anchors,
        t_gt=ground_truth[:, 0],
        p_gt=ground_truth[:, 1:4],
        q_gt_xyzw=ground_truth[:, 4:8],
        t_imu=t_imu,
        imu=imu,
        t_uwb=tdoa[:, 0],
        uwb=tdoa[:, 1:4],
        uwb_source_count=uwb_source_count,
        uwb_requested_rate_hz=uwb_rate_hz,
        uwb_actual_rate_hz=sample_rate_hz(tdoa[:, 0]),
    )


def make_filter(module, dataset, yaw_degrees, tdoa_variance):
    initial_rotation = Rotation.from_quat(
        dataset.q_gt_xyzw[0]
    ).as_matrix()
    initial_orientation = module.SO3_log(initial_rotation)
    initial_orientation = initial_orientation + np.array(
        [0.0, 0.0, math.radians(float(yaw_degrees))]
    )

    state = np.eye(5)
    state[0:3, 0:3] = module.SO3_exp(initial_orientation)
    state[0:3, 4] = dataset.p_gt[0]
    covariance = np.diag(
        [0.01**2, 0.01**2, 1.0**2]
        + [0.01**2] * 3
        + [0.1**2] * 3
        + [0.0001**2] * 3
        + [0.000001**2] * 3
    )
    estimator = module.InEKF(state, covariance, np.zeros(6))
    estimator.w_tdoa = np.array([math.sqrt(float(tdoa_variance))])
    return estimator


def replay(label, module, dataset, yaw_degrees, tdoa_variance):
    estimator = make_filter(
        module, dataset, yaw_degrees, tdoa_variance
    )
    events = np.unique(np.concatenate((dataset.t_imu, dataset.t_uwb)))
    imu_lookup = {
        float(timestamp): index
        for index, timestamp in enumerate(dataset.t_imu)
    }
    # The author's isin() returns only the first matching UWB sample.
    uwb_lookup = {}
    for index, timestamp in enumerate(dataset.t_uwb):
        uwb_lookup.setdefault(float(timestamp), index)

    times = [float(events[0])]
    rotations = [estimator.X[0:3, 0:3].copy()]
    velocities = [estimator.X[0:3, 3].copy()]
    positions = [estimator.X[0:3, 4].copy()]
    biases = [estimator.bias.copy()]
    prediction_count = 0
    uwb_count = 0
    last_imu_propagation_time = 0.0

    start = time.perf_counter()
    for event_index in range(1, len(events)):
        previous_time = float(events[event_index - 1])
        current_time = float(events[event_index])

        imu_index = imu_lookup.get(previous_time)
        if imu_index is not None:
            imu_dt = current_time - last_imu_propagation_time
            last_imu_propagation_time = current_time
            estimator.predict(dataset.imu[imu_index], imu_dt)
            prediction_count += 1

        uwb_index = uwb_lookup.get(previous_time)
        if uwb_index is not None:
            estimator.correct(dataset.uwb[uwb_index], dataset.anchors)
            uwb_count += 1

        times.append(current_time)
        rotations.append(estimator.X[0:3, 0:3].copy())
        velocities.append(estimator.X[0:3, 3].copy())
        positions.append(estimator.X[0:3, 4].copy())
        biases.append(estimator.bias.copy())
    elapsed = time.perf_counter() - start

    return ReplayResult(
        label=label,
        times=np.asarray(times),
        rotations=np.asarray(rotations),
        velocities=np.asarray(velocities),
        positions=np.asarray(positions),
        biases=np.asarray(biases),
        elapsed_seconds=elapsed,
        prediction_count=prediction_count,
        uwb_count=uwb_count,
        rejection_count=int(estimator.rej_cnt),
        metrics={},
    )


def interpolate_truth(dataset, times):
    clipped_times = np.clip(times, dataset.t_gt[0], dataset.t_gt[-1])
    positions = np.column_stack(
        [
            np.interp(clipped_times, dataset.t_gt, dataset.p_gt[:, axis])
            for axis in range(3)
        ]
    )
    velocities = np.column_stack(
        [
            interpolate.splev(
                clipped_times,
                interpolate.splrep(
                    dataset.t_gt, dataset.p_gt[:, axis], s=0.5
                ),
                der=1,
            )
            for axis in range(3)
        ]
    )
    unique_times, unique_indices = np.unique(
        dataset.t_gt, return_index=True
    )
    truth_rotations = Rotation.from_quat(
        dataset.q_gt_xyzw[unique_indices]
    )
    slerp = Slerp(unique_times, truth_rotations)
    rotation_matrices = slerp(
        np.clip(times, unique_times[0], unique_times[-1])
    ).as_matrix()
    return positions, velocities, rotation_matrices


def wrap_angle(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def compute_metrics(result, truth):
    truth_positions, truth_velocities, truth_rotations = truth
    position_errors = result.positions - truth_positions
    velocity_errors = result.velocities - truth_velocities
    relative_rotations = (
        Rotation.from_matrix(truth_rotations).inv()
        * Rotation.from_matrix(result.rotations)
    )
    attitude_errors = relative_rotations.magnitude()
    estimated_yaw = Rotation.from_matrix(result.rotations).as_euler(
        "xyz"
    )[:, 2]
    truth_yaw = Rotation.from_matrix(truth_rotations).as_euler(
        "xyz"
    )[:, 2]
    yaw_errors = wrap_angle(estimated_yaw - truth_yaw)

    metrics = {
        "position_rmse_m": float(
            np.sqrt(np.mean(np.sum(position_errors**2, axis=1)))
        ),
        "velocity_rmse_mps": float(
            np.sqrt(np.mean(np.sum(velocity_errors**2, axis=1)))
        ),
        "attitude_geodesic_rmse_rad": float(
            np.sqrt(np.mean(attitude_errors**2))
        ),
        "yaw_rmse_rad": float(np.sqrt(np.mean(yaw_errors**2))),
        "elapsed_seconds": float(result.elapsed_seconds),
        "prediction_count": int(result.prediction_count),
        "uwb_count": int(result.uwb_count),
        "rejection_count": int(result.rejection_count),
    }
    diagnostics = {
        "position_error_norm": np.linalg.norm(position_errors, axis=1),
        "velocity_error_norm": np.linalg.norm(velocity_errors, axis=1),
        "attitude_error_deg": np.degrees(attitude_errors),
        "yaw_error_deg": np.degrees(np.abs(yaw_errors)),
    }
    result.metrics = metrics
    return diagnostics


def relative_change(before, after):
    if before == 0.0:
        return math.nan
    return 100.0 * (after - before) / before


def source_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def create_figure(
    dataset,
    original,
    improved,
    truth,
    original_diagnostics,
    improved_diagnostics,
    output_path,
):
    truth_positions, _, _ = truth
    times = original.times
    position_delta = np.linalg.norm(
        improved.positions - original.positions, axis=1
    )
    velocity_delta = np.linalg.norm(
        improved.velocities - original.velocities, axis=1
    )

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "figure.dpi": 130,
        }
    )
    figure, axes = plt.subplots(3, 2, figsize=(14, 13))
    rate_label = (
        f"aggregate UWB {dataset.uwb_actual_rate_hz:.3f} Hz"
        if dataset.uwb_requested_rate_hz is not None
        else f"source aggregate UWB {dataset.uwb_actual_rate_hz:.3f} Hz"
    )
    figure.suptitle(
        "Pose propagation: author Euler update vs exact "
        r"$J_l(\phi),Q_l(\phi)$ update"
        f"\n{rate_label}",
        fontsize=15,
    )

    axis = axes[0, 0]
    axis.plot(
        truth_positions[:, 0],
        truth_positions[:, 1],
        color="black",
        linewidth=1.6,
        label="Vicon",
    )
    axis.plot(
        original.positions[:, 0],
        original.positions[:, 1],
        color="#d95f02",
        linewidth=1.0,
        label="Author Euler",
    )
    axis.plot(
        improved.positions[:, 0],
        improved.positions[:, 1],
        color="#1b9e77",
        linewidth=1.0,
        linestyle="--",
        label="Improved J/Q",
    )
    axis.scatter(
        dataset.anchors[:, 0],
        dataset.anchors[:, 1],
        marker="^",
        s=25,
        color="#7570b3",
        label="Anchors",
    )
    axis.set_title("XY trajectory")
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.axis("equal")
    axis.legend(loc="best")

    axis = axes[0, 1]
    axis.plot(
        times,
        original_diagnostics["position_error_norm"],
        color="#d95f02",
        linewidth=0.9,
        label="Author Euler",
    )
    axis.plot(
        times,
        improved_diagnostics["position_error_norm"],
        color="#1b9e77",
        linewidth=0.9,
        linestyle="--",
        label="Improved J/Q",
    )
    axis.set_title("Position error norm")
    axis.set_xlabel("time [s]")
    axis.set_ylabel("error [m]")
    axis.legend(loc="best")

    axis = axes[1, 0]
    axis.plot(
        times,
        original_diagnostics["velocity_error_norm"],
        color="#d95f02",
        linewidth=0.9,
        label="Author Euler",
    )
    axis.plot(
        times,
        improved_diagnostics["velocity_error_norm"],
        color="#1b9e77",
        linewidth=0.9,
        linestyle="--",
        label="Improved J/Q",
    )
    axis.set_title("Velocity error norm")
    axis.set_xlabel("time [s]")
    axis.set_ylabel("error [m/s]")
    axis.legend(loc="best")

    axis = axes[1, 1]
    axis.plot(
        times,
        original_diagnostics["attitude_error_deg"],
        color="#d95f02",
        linewidth=0.9,
        label="Author attitude",
    )
    axis.plot(
        times,
        improved_diagnostics["attitude_error_deg"],
        color="#1b9e77",
        linewidth=0.9,
        linestyle="--",
        label="Improved attitude",
    )
    axis.plot(
        times,
        original_diagnostics["yaw_error_deg"],
        color="#e6ab02",
        linewidth=0.7,
        alpha=0.8,
        label="Author yaw",
    )
    axis.plot(
        times,
        improved_diagnostics["yaw_error_deg"],
        color="#66a61e",
        linewidth=0.7,
        alpha=0.8,
        linestyle="--",
        label="Improved yaw",
    )
    axis.set_title("Attitude and yaw error")
    axis.set_xlabel("time [s]")
    axis.set_ylabel("absolute error [deg]")
    axis.legend(loc="best", ncol=2)

    axis = axes[2, 0]
    floor = 1e-12
    axis.semilogy(
        times,
        np.maximum(position_delta, floor),
        color="#7570b3",
        linewidth=0.9,
        label=r"$\|p_{J/Q}-p_{Euler}\|$ [m]",
    )
    axis.semilogy(
        times,
        np.maximum(velocity_delta, floor),
        color="#e7298a",
        linewidth=0.9,
        label=r"$\|v_{J/Q}-v_{Euler}\|$ [m/s]",
    )
    axis.set_title("Magnified state difference between formulas")
    axis.set_xlabel("time [s]")
    axis.set_ylabel("difference, log scale")
    axis.legend(loc="best")

    axis = axes[2, 1]
    axis.axis("off")
    row_names = (
        "Position RMSE [m]",
        "Velocity RMSE [m/s]",
        "Attitude RMSE [deg]",
        "Yaw RMSE [deg]",
        "Rejected UWB",
        "Replay time [s]",
    )
    before_values = (
        original.metrics["position_rmse_m"],
        original.metrics["velocity_rmse_mps"],
        math.degrees(original.metrics["attitude_geodesic_rmse_rad"]),
        math.degrees(original.metrics["yaw_rmse_rad"]),
        original.metrics["rejection_count"],
        original.metrics["elapsed_seconds"],
    )
    after_values = (
        improved.metrics["position_rmse_m"],
        improved.metrics["velocity_rmse_mps"],
        math.degrees(improved.metrics["attitude_geodesic_rmse_rad"]),
        math.degrees(improved.metrics["yaw_rmse_rad"]),
        improved.metrics["rejection_count"],
        improved.metrics["elapsed_seconds"],
    )
    table_rows = []
    for before, after in zip(before_values, after_values):
        change = relative_change(float(before), float(after))
        table_rows.append(
            (
                f"{before:.8g}",
                f"{after:.8g}",
                "n/a" if not math.isfinite(change) else f"{change:+.4f}%",
            )
        )
    table = axis.table(
        cellText=table_rows,
        rowLabels=row_names,
        colLabels=("Author Euler", "Improved J/Q", "relative change"),
        loc="center",
        cellLoc="right",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.55)
    axis.set_title("Full-sample metrics", pad=18)

    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def serialize_result(result):
    return {
        "label": result.label,
        **result.metrics,
    }


def main():
    args = parse_args()
    anchor_path = Path(args.i[0]).resolve()
    csv_path = Path(args.i[1]).resolve()
    author_inekf = (
        args.author_repo.resolve() / "scripts" / "estimation" / "inekf.py"
    )
    if not author_inekf.exists():
        raise FileNotFoundError(f"Author source not found: {author_inekf}")
    if not IMPROVED_INEKF.exists():
        raise FileNotFoundError(f"Improved source not found: {IMPROVED_INEKF}")

    print(f"[load] {csv_path}")
    dataset = load_dataset(
        anchor_path,
        csv_path,
        args.max_duration,
        args.uwb_rate_hz,
    )
    print(
        "[load] aggregate UWB: "
        f"{dataset.uwb_source_count} -> {len(dataset.t_uwb)} samples, "
        f"actual {dataset.uwb_actual_rate_hz:.6g} Hz"
    )
    author_module = load_module("task04_author_inekf", author_inekf)
    improved_module = load_module("task04_improved_inekf", IMPROVED_INEKF)

    print("[run] author Euler propagation")
    original = replay(
        "author_euler",
        author_module,
        dataset,
        args.yaw_deg,
        args.tdoa_variance,
    )
    print("[run] improved J_l/Q_l propagation")
    improved = replay(
        "improved_jl_ql",
        improved_module,
        dataset,
        args.yaw_deg,
        args.tdoa_variance,
    )

    np.testing.assert_array_equal(original.times, improved.times)
    assert original.prediction_count == improved.prediction_count
    assert original.uwb_count == improved.uwb_count
    truth = interpolate_truth(dataset, original.times)
    original_diagnostics = compute_metrics(original, truth)
    improved_diagnostics = compute_metrics(improved, truth)

    position_delta = np.linalg.norm(
        improved.positions - original.positions, axis=1
    )
    velocity_delta = np.linalg.norm(
        improved.velocities - original.velocities, axis=1
    )
    comparison = {
        "position_rmse_relative_change_percent": relative_change(
            original.metrics["position_rmse_m"],
            improved.metrics["position_rmse_m"],
        ),
        "velocity_rmse_relative_change_percent": relative_change(
            original.metrics["velocity_rmse_mps"],
            improved.metrics["velocity_rmse_mps"],
        ),
        "attitude_rmse_relative_change_percent": relative_change(
            original.metrics["attitude_geodesic_rmse_rad"],
            improved.metrics["attitude_geodesic_rmse_rad"],
        ),
        "yaw_rmse_relative_change_percent": relative_change(
            original.metrics["yaw_rmse_rad"],
            improved.metrics["yaw_rmse_rad"],
        ),
        "max_position_state_difference_m": float(np.max(position_delta)),
        "max_velocity_state_difference_mps": float(np.max(velocity_delta)),
        "final_position_state_difference_m": float(position_delta[-1]),
        "final_velocity_state_difference_mps": float(velocity_delta[-1]),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = args.output_dir / "comparison.png"
    create_figure(
        dataset,
        original,
        improved,
        truth,
        original_diagnostics,
        improved_diagnostics,
        figure_path,
    )

    payload = {
        "task": "04 pose propagation formula improvement",
        "command": [sys.executable, *sys.argv],
        "author_commit": AUTHOR_COMMIT,
        "inputs": {
            "anchors": str(anchor_path),
            "csv": str(csv_path),
            "max_duration_seconds": args.max_duration,
            "uwb_requested_rate_hz": args.uwb_rate_hz,
            "uwb_actual_rate_hz": dataset.uwb_actual_rate_hz,
            "yaw_degrees": args.yaw_deg,
            "tdoa_variance": args.tdoa_variance,
        },
        "source_sha256": {
            "author_inekf": source_sha256(author_inekf),
            "improved_inekf": source_sha256(IMPROVED_INEKF),
        },
        "counts": {
            "event_states": len(original.times),
            "imu_predictions": original.prediction_count,
            "uwb_updates": original.uwb_count,
            "uwb_source_updates_before_rate_limit": dataset.uwb_source_count,
        },
        "results": {
            "author_euler": serialize_result(original),
            "improved_jl_ql": serialize_result(improved),
        },
        "comparison": comparison,
    }
    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    summary_path = args.output_dir / "summary.csv"
    pd.DataFrame(
        [serialize_result(original), serialize_result(improved)]
    ).to_csv(summary_path, index=False)

    print(json.dumps(payload["results"], indent=2))
    print(json.dumps(comparison, indent=2))
    print(f"[saved] {figure_path}")
    print(f"[saved] {metrics_path}")
    print(f"[saved] {summary_path}")


if __name__ == "__main__":
    main()
