"""Efficient event-driven dataset replay and geometry-correct metrics."""

from dataclasses import dataclass
import math
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.spatial.transform import Rotation, Slerp

from inekf import FilterConfig, InEKF, SE3_2_vec, SO3_exp, SO3_log


@dataclass
class Dataset:
    name: str
    anchors: np.ndarray
    t_imu: np.ndarray
    imu: np.ndarray
    t_uwb: np.ndarray
    uwb: np.ndarray
    t_gt: np.ndarray
    p_gt: np.ndarray
    q_gt_xyzw: np.ndarray


@dataclass
class EstimationResult:
    variant: str
    times: np.ndarray
    states: np.ndarray
    rotations: np.ndarray
    covariances: np.ndarray
    biases: np.ndarray
    nis_times: np.ndarray
    nis: np.ndarray
    rejected: np.ndarray
    runtime_seconds: float
    stationary_updates: int
    metrics: dict
    jit_compile_seconds: float = 0.0


def _valid_columns(frame, columns):
    return frame.loc[:, columns].dropna().to_numpy(dtype=float)


def load_dataset(anchor_file, csv_file, max_duration=None):
    """Load the sensor streams without constructing a sparse 130k-row matrix."""
    anchor_file = Path(anchor_file)
    csv_file = Path(csv_file)
    anchors = np.load(anchor_file)["an_pos"]
    frame = pd.read_csv(csv_file)

    acc = _valid_columns(frame, ["t_acc", "acc_x", "acc_y", "acc_z"])
    gyro = _valid_columns(frame, ["t_gyro", "gyro_x", "gyro_y", "gyro_z"])
    tdoa = _valid_columns(frame, ["t_tdoa", "idA", "idB", "tdoa_meas"])
    gt = _valid_columns(
        frame,
        ["t_pose", "pose_x", "pose_y", "pose_z", "pose_qx", "pose_qy", "pose_qz", "pose_qw"],
    )

    gyro_at_acc = np.column_stack(
        [np.interp(acc[:, 0], gyro[:, 0], gyro[:, axis]) for axis in range(1, 4)]
    )
    imu = np.column_stack((acc[:, 1:4], gyro_at_acc))
    t_imu = acc[:, 0]

    if "tdoa2" in csv_file.name.lower():
        allowed_pairs = {(7, 0), (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)}
        keep = np.array([(int(a), int(b)) in allowed_pairs for a, b in tdoa[:, 1:3]])
        tdoa = tdoa[keep]

    order = np.argsort(tdoa[:, 0], kind="stable")
    tdoa = tdoa[order]
    min_time = min(t_imu[0], tdoa[0, 0], gt[0, 0])
    t_imu = t_imu - min_time
    t_uwb = tdoa[:, 0] - min_time
    t_gt = gt[:, 0] - min_time

    gt_keep = t_gt >= 0.0
    t_gt = t_gt[gt_keep]
    gt = gt[gt_keep]

    if max_duration is not None:
        duration = float(max_duration)
        imu_keep = t_imu <= duration
        uwb_keep = t_uwb <= duration
        gt_duration_keep = t_gt <= duration
        t_imu, imu = t_imu[imu_keep], imu[imu_keep]
        t_uwb, tdoa = t_uwb[uwb_keep], tdoa[uwb_keep]
        t_gt, gt = t_gt[gt_duration_keep], gt[gt_duration_keep]

    if min(len(t_imu), len(t_uwb), len(t_gt)) < 2:
        raise ValueError("Dataset slice must contain at least two IMU, UWB, and ground-truth samples")

    return Dataset(
        name=csv_file.stem,
        anchors=anchors,
        t_imu=t_imu,
        imu=imu,
        t_uwb=t_uwb,
        uwb=tdoa[:, 1:4],
        t_gt=t_gt,
        p_gt=gt[:, 1:4],
        q_gt_xyzw=gt[:, 4:8],
    )


def _initial_filter(dataset, yaw_error_degrees, config, initial_std_overrides=None):
    rotation_gt = Rotation.from_quat(dataset.q_gt_xyzw[0]).as_matrix()
    yaw_offset = math.radians(float(yaw_error_degrees))
    if config.name == "legacy":
        # Reproduce the public code's non-compositional rotation-vector offset.
        rotation0 = SO3_exp(SO3_log(rotation_gt) + np.array([0.0, 0.0, yaw_offset]))
    else:
        rotation0 = rotation_gt @ SO3_exp(np.array([0.0, 0.0, yaw_offset]))

    X0 = np.eye(5)
    X0[:3, :3] = rotation0
    X0[:3, 4] = dataset.p_gt[0]

    std_rp = 0.01
    std_yaw = config.initial_yaw_std
    std_velocity = 0.01
    std_position = config.initial_position_std
    std_b_acc = 0.0001
    std_b_gyr = 0.000001
    initial_std_overrides = initial_std_overrides or {}
    std_rp = float(initial_std_overrides.get("roll_pitch", std_rp))
    std_yaw = float(initial_std_overrides.get("yaw", std_yaw))
    std_velocity = float(initial_std_overrides.get("velocity", std_velocity))
    std_position = float(initial_std_overrides.get("position", std_position))
    initial_bias = np.zeros(6)
    if config.consistent_bias_order and config.static_bias_initialization:
        calibration = dataset.t_imu <= dataset.t_imu[0] + 1.0
        calibration_imu = dataset.imu[calibration]
        gyro_si = np.radians(calibration_imu[:, 3:6])
        accel_si = calibration_imu[:, 0:3] * 9.81
        stationary = (
            len(calibration_imu) >= 20
            and np.linalg.norm(np.mean(gyro_si, axis=0)) < math.radians(5.0)
            and abs(np.linalg.norm(np.mean(accel_si, axis=0)) - 9.81) < 0.5
        )
        if stationary:
            # Zero angular rate directly observes b_g even when attitude makes
            # the accelerometer bias ambiguous.  Do not throw away this valid
            # calibration merely because the b_a consistency gate fails.
            initial_bias[:3] = np.mean(gyro_si, axis=0)
            sample_count = math.sqrt(len(calibration_imu))
            std_b_gyr = max(
                1e-3,
                float(np.max(np.std(gyro_si, axis=0))) / sample_count,
            )
            expected_specific_force = rotation0.T @ np.array([0.0, 0.0, 9.81])
            accelerometer_bias = np.mean(accel_si, axis=0) - expected_specific_force
            # Reject a large apparent accelerometer bias: it usually signals
            # imperfect rest/alignment and would otherwise absorb attitude
            # error. Reject only b_a; zero angular rate still observes b_g.
            if np.linalg.norm(accelerometer_bias) <= 0.25:
                initial_bias[3:] = accelerometer_bias
                # Uncertainty of the estimated mean, with conservative floors.
                # The scatter of the mean alone is overconfident for warm-up,
                # temperature drift and imperfect stationarity.  Keep physically
                # meaningful floors so bias and large-yaw errors do not lock each
                # other into an incorrect local solution.
                std_b_acc = max(5e-2, float(np.max(np.std(accel_si, axis=0))) / sample_count)
    if config.consistent_bias_order:
        bias_variances = [std_b_gyr**2] * 3 + [std_b_acc**2] * 3
    else:
        bias_variances = [std_b_acc**2] * 3 + [std_b_gyr**2] * 3
    P0 = np.diag(
        [std_rp**2, std_rp**2, std_yaw**2]
        + [std_velocity**2] * 3
        + [std_position**2] * 3
        + bias_variances
    )
    estimator = InEKF(X0, P0, initial_bias, config=config)
    if config.map_physical_initial_covariance:
        estimator.initialize_right_invariant_covariance(P0)
    estimator.imu_sample_period = float(np.median(np.diff(dataset.t_imu)))
    estimator.w_tdoa = np.array([math.sqrt(0.13 if "const4" in dataset.name else 0.05)])
    return estimator


def _append_state(estimator, timestamp, output):
    output["times"].append(float(timestamp))
    output["states"].append(SE3_2_vec(estimator.X))
    output["rotations"].append(estimator.X[:3, :3].copy())
    output["covariances"].append(estimator.P.copy())
    output["biases"].append(estimator.bias.copy())


def _uwb_groups(times):
    groups = {}
    for index, timestamp in enumerate(times):
        groups.setdefault(float(timestamp), []).append(index)
    return groups


def _record_corrections(estimator, timestamp, indices, dataset, output):
    for index in indices:
        _, _, _, was_rejected = estimator.correct(dataset.uwb[index], dataset.anchors)
        output["nis_times"].append(float(timestamp))
        output["nis"].append(estimator.last_nis)
        output["rejected"].append(bool(was_rejected))


def _legacy_replay(dataset, estimator, output):
    """Public scheduler semantics with O(N) lookup instead of repeated scans."""
    events = np.unique(np.concatenate((dataset.t_imu, dataset.t_uwb)))
    imu_lookup = {float(timestamp): index for index, timestamp in enumerate(dataset.t_imu)}
    uwb_lookup = _uwb_groups(dataset.t_uwb)
    previous_prediction_boundary = 0.0
    _append_state(estimator, events[0], output)

    for k in range(1, len(events)):
        event_time = float(events[k - 1])
        next_time = float(events[k])
        if event_time in imu_lookup:
            imu_dt = next_time - previous_prediction_boundary
            previous_prediction_boundary = next_time
            estimator.predict(dataset.imu[imu_lookup[event_time]], imu_dt)
        if event_time in uwb_lookup:
            _record_corrections(estimator, event_time, uwb_lookup[event_time], dataset, output)
        _append_state(estimator, next_time, output)


def _interpolate_imu(dataset, timestamp):
    return np.array(
        [np.interp(timestamp, dataset.t_imu, dataset.imu[:, axis]) for axis in range(6)]
    )


def _verified_replay(dataset, estimator, output):
    """Propagate exactly to every asynchronous IMU/UWB event timestamp."""
    events = np.unique(np.concatenate((dataset.t_imu, dataset.t_uwb)))
    uwb_lookup = _uwb_groups(dataset.t_uwb)
    current_time = float(events[0])

    if current_time in uwb_lookup:
        _record_corrections(estimator, current_time, uwb_lookup[current_time], dataset, output)
    _append_state(estimator, current_time, output)

    for event in events[1:]:
        event_time = float(event)
        u0 = _interpolate_imu(dataset, current_time)
        u1 = _interpolate_imu(dataset, event_time)
        estimator.predict(u0, event_time - current_time, u_next=u1)
        current_time = event_time
        if event_time in uwb_lookup:
            _record_corrections(estimator, event_time, uwb_lookup[event_time], dataset, output)
        _append_state(estimator, event_time, output)


def _stationary_flags(dataset, window_seconds=0.2):
    """Causal, allocation-light rest detector for the 1 kHz UTIL IMU.

    Every sample in the trailing window must satisfy both conservative tests.
    Requiring persistence prevents isolated low-dynamics flight samples from
    triggering a zero-angular-rate update.
    """
    sample_period = float(np.median(np.diff(dataset.t_imu)))
    window = max(3, int(round(window_seconds / sample_period)))
    acceleration_norm = np.linalg.norm(dataset.imu[:, :3], axis=1)
    angular_rate_norm = np.linalg.norm(dataset.imu[:, 3:], axis=1)
    raw = (np.abs(acceleration_norm - 1.0) < 0.03) & (angular_rate_norm < 2.0)
    cumulative = np.concatenate(([0], np.cumsum(raw, dtype=np.int64)))
    flags = np.zeros(len(raw), dtype=bool)
    flags[window - 1:] = (cumulative[window:] - cumulative[:-window]) == window
    return flags


def _fast_replay(dataset, estimator, output):
    """One propagation per IMU interval; UWB latency is bounded by 1 IMU tick."""
    events = np.unique(np.concatenate((dataset.t_imu, dataset.t_uwb)))
    imu_lookup = {float(timestamp): index for index, timestamp in enumerate(dataset.t_imu)}
    uwb_lookup = _uwb_groups(dataset.t_uwb)
    stationary = (
        _stationary_flags(dataset)
        if estimator.config.stationary_inertial_update
        else np.zeros(len(dataset.t_imu), dtype=bool)
    )
    last_imu_index = 0
    last_stationary_update = -math.inf
    initial_rest_seen = False
    initial_rest_complete = False
    _append_state(estimator, events[0], output)

    for event in events[1:]:
        event_time = float(event)
        if event_time in imu_lookup:
            imu_index = imu_lookup[event_time]
            if imu_index > last_imu_index:
                dt = event_time - float(dataset.t_imu[last_imu_index])
                estimator.predict(dataset.imu[last_imu_index], dt)
                last_imu_index = imu_index
            if (
                stationary[imu_index]
                and not initial_rest_complete
                and event_time - last_stationary_update >= 0.10
            ):
                estimator.correct_stationary(dataset.imu[imu_index, 3:6])
                last_stationary_update = event_time
                initial_rest_seen = True
            elif initial_rest_seen and not stationary[imu_index]:
                # Only use the first confirmed rest interval.  This prevents a
                # later low-dynamics hover from being mistaken for contact.
                initial_rest_complete = True
        if event_time in uwb_lookup:
            _record_corrections(estimator, event_time, uwb_lookup[event_time], dataset, output)
        _append_state(estimator, event_time, output)


def _sustained_convergence_time(times, absolute_error, threshold, window_seconds=2.0):
    for index, timestamp in enumerate(times):
        end = np.searchsorted(times, timestamp + window_seconds, side="right")
        if end > index and np.all(absolute_error[index:end] <= threshold):
            return float(timestamp - times[0])
    return math.nan


def compute_metrics(dataset, result):
    valid = (
        (result.times >= dataset.t_gt[0])
        & (result.times <= dataset.t_gt[-1])
    )
    times = result.times[valid]
    states = result.states[valid]
    rotations = result.rotations[valid]
    if len(times) < 2:
        raise ValueError("No estimator samples overlap the ground-truth interval")

    splines = [
        interpolate.UnivariateSpline(dataset.t_gt, dataset.p_gt[:, axis], s=0.5)
        for axis in range(3)
    ]
    position_gt = np.column_stack([spline(times) for spline in splines])
    velocity_gt = np.column_stack([spline.derivative()(times) for spline in splines])
    position_error = states[:, 6:9] - position_gt
    velocity_error = states[:, 3:6] - velocity_gt

    gt_slerp = Slerp(dataset.t_gt, Rotation.from_quat(dataset.q_gt_xyzw))
    rotation_gt = gt_slerp(times)
    rotation_est = Rotation.from_matrix(rotations)
    relative_rotation = rotation_gt.inv() * rotation_est
    geodesic_error = relative_rotation.magnitude()
    yaw_gt = rotation_gt.as_euler("xyz")[:, 2]
    yaw_est = rotation_est.as_euler("xyz")[:, 2]
    yaw_error = np.arctan2(np.sin(yaw_est - yaw_gt), np.cos(yaw_est - yaw_gt))

    position_norm = np.linalg.norm(position_error, axis=1)
    velocity_norm = np.linalg.norm(velocity_error, axis=1)
    metrics = {
        "position_rmse_m": float(np.sqrt(np.mean(position_norm**2))),
        "velocity_rmse_mps": float(np.sqrt(np.mean(velocity_norm**2))),
        "orientation_geodesic_rmse_rad": float(np.sqrt(np.mean(geodesic_error**2))),
        "yaw_rmse_rad": float(np.sqrt(np.mean(yaw_error**2))),
        "yaw_convergence_10deg_s": _sustained_convergence_time(
            times, np.abs(yaw_error), math.radians(10.0)
        ),
        "yaw_convergence_5deg_s": _sustained_convergence_time(
            times, np.abs(yaw_error), math.radians(5.0)
        ),
        "rejected_measurements": int(np.sum(result.rejected)),
        "total_measurements": int(len(result.rejected)),
        "rejection_rate": float(np.mean(result.rejected)) if len(result.rejected) else math.nan,
        "nis_median": float(np.median(result.nis)) if len(result.nis) else math.nan,
        "runtime_seconds": float(result.runtime_seconds),
        "jit_compile_seconds": float(result.jit_compile_seconds),
        "stationary_updates": int(result.stationary_updates),
        "samples": int(len(times)),
    }
    result.metrics = metrics
    return {
        "times": times,
        "position_gt": position_gt,
        "velocity_gt": velocity_gt,
        "position_error": position_error,
        "velocity_error": velocity_error,
        "rotation_error_rotvec": relative_rotation.as_rotvec(),
        "geodesic_error": geodesic_error,
        "yaw_gt": yaw_gt,
        "yaw_est": yaw_est,
        "yaw_error": yaw_error,
    }


def run_dataset(
    dataset,
    variant="verified",
    yaw_error_degrees=0.0,
    initial_std_overrides=None,
):
    configurations = {
        "legacy": FilterConfig.legacy,
        "aligned_legacy": FilterConfig.aligned_legacy,
        "pre_coordinate_fix": FilterConfig.pre_coordinate_fix,
        "pre_optimization": FilterConfig.pre_optimization,
        "proposed": FilterConfig.proposed,
        "task03_compiled": FilterConfig.task03_compiled,
        "proposed_euler": FilterConfig.proposed_euler,
        "proposed_no_bias_init": FilterConfig.proposed_no_bias_init,
        "proposed_no_stationary": FilterConfig.proposed_no_stationary,
        "verified": FilterConfig.verified,
    }
    if variant not in configurations:
        raise ValueError(f"Unknown estimator variant: {variant}")
    config = configurations[variant]()
    estimator = _initial_filter(
        dataset,
        yaw_error_degrees,
        config,
        initial_std_overrides=initial_std_overrides,
    )
    jit_compile_seconds = 0.0
    if variant == "task03_compiled":
        from compiled_replay import ensure_compiled_kernel, run_compiled_replay

        # Compilation/cache loading is a one-time deployment cost, not steady
        # state estimator latency. It is measured and reported separately.
        jit_compile_seconds = ensure_compiled_kernel()
        stationary = _stationary_flags(dataset)
        start = time.perf_counter()
        times, compiled_output = run_compiled_replay(
            dataset,
            estimator,
            stationary,
        )
        runtime = time.perf_counter() - start
        (
            states,
            rotations,
            covariances,
            biases,
            nis_times,
            nis,
            rejected,
            stationary_updates,
            processed_measurements,
            rejected_measurements,
        ) = compiled_output
        if processed_measurements != len(nis):
            raise AssertionError(
                "Compiled diagnostic count does not match recorded NIS"
            )
        if rejected_measurements != int(np.sum(rejected)):
            raise AssertionError(
                "Compiled rejection count does not match recorded decisions"
            )
        result = EstimationResult(
            variant=variant,
            times=times,
            states=states,
            rotations=rotations,
            covariances=covariances,
            biases=biases,
            nis_times=nis_times,
            nis=nis,
            rejected=rejected,
            runtime_seconds=runtime,
            stationary_updates=stationary_updates,
            metrics={},
            jit_compile_seconds=jit_compile_seconds,
        )
    else:
        output = {
            "times": [],
            "states": [],
            "rotations": [],
            "covariances": [],
            "biases": [],
            "nis_times": [],
            "nis": [],
            "rejected": [],
        }
        start = time.perf_counter()
        if variant == "legacy":
            _legacy_replay(dataset, estimator, output)
        elif variant == "verified":
            _verified_replay(dataset, estimator, output)
        else:
            _fast_replay(dataset, estimator, output)
        runtime = time.perf_counter() - start

        result = EstimationResult(
            variant=variant,
            times=np.asarray(output["times"]),
            states=np.asarray(output["states"]),
            rotations=np.asarray(output["rotations"]),
            covariances=np.asarray(output["covariances"]),
            biases=np.asarray(output["biases"]),
            nis_times=np.asarray(output["nis_times"]),
            nis=np.asarray(output["nis"]),
            rejected=np.asarray(output["rejected"], dtype=bool),
            runtime_seconds=runtime,
            stationary_updates=estimator.stationary_update_count,
            metrics={},
        )
    compute_metrics(dataset, result)
    return result
