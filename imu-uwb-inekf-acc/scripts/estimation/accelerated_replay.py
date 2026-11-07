"""Native fixed-size replay of the untouched paper-author InEKF equations.

This module changes execution only.  It intentionally retains the author's
nominal state, bias layout, initialization, ground clamp, process noise,
left-coordinate pre-gate, right-invariant update, Joseph covariance update,
and all scalar noise/gate settings.  The event plan and output buffers are
prepared before the Numba ``nopython`` loop.
"""

import math
import time

import numpy as np

try:
    from numba import njit
except ImportError as exc:  # pragma: no cover - deployment configuration
    raise ImportError(
        "The accelerated author-equivalent replay requires Numba. "
        "Use D:\\conda-envs\\imu-uwb-inekf or install numba."
    ) from exc


GRAVITY_MAGNITUDE = 9.81
DEG_TO_RAD = math.pi / 180.0
_WARMUP_SECONDS = None


@njit(cache=True)
def _wedge(vector):
    result = np.zeros((3, 3), dtype=np.float64)
    result[0, 1] = -vector[2]
    result[0, 2] = vector[1]
    result[1, 0] = vector[2]
    result[1, 2] = -vector[0]
    result[2, 0] = -vector[1]
    result[2, 1] = vector[0]
    return result


@njit(cache=True)
def _author_so3_exp(phi):
    """Match the author's Rodrigues branch and near-zero first-order branch."""
    angle = math.sqrt(phi @ phi)
    if abs(angle) <= 1e-8:
        return np.eye(3, dtype=np.float64) + _wedge(phi)
    axis = phi / angle
    sine = math.sin(angle)
    cosine = math.cos(angle)
    return (
        cosine * np.eye(3, dtype=np.float64)
        + (1.0 - cosine) * np.outer(axis, axis)
        + sine * _wedge(axis)
    )


@njit(cache=True)
def _author_left_jacobian(phi):
    """Match the author's SE_2(3) injection Jacobian, including its small branch."""
    angle = math.sqrt(phi @ phi)
    if abs(angle) <= 1e-8:
        return np.eye(3, dtype=np.float64) - 0.5 * _wedge(phi)
    axis = phi / angle
    sine = math.sin(angle)
    cosine = math.cos(angle)
    return (
        (sine / angle) * np.eye(3, dtype=np.float64)
        + (1.0 - sine / angle) * np.outer(axis, axis)
        + ((1.0 - cosine) / angle) * _wedge(axis)
    )


@njit(cache=True)
def _author_rotation_log(rotation):
    cosine = 0.5 * (
        rotation[0, 0] + rotation[1, 1] + rotation[2, 2]
    ) - 0.5
    cosine = min(1.0, max(-1.0, cosine))
    angle = math.acos(cosine)
    if abs(angle) <= 1e-8:
        return np.array(
            [rotation[2, 1], rotation[0, 2], rotation[1, 0]],
            dtype=np.float64,
        )
    scale = 0.5 * angle / math.sin(angle)
    return scale * np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ],
        dtype=np.float64,
    )


@njit(cache=True)
def _adjoint15(rotation, velocity, position):
    adjoint = np.eye(15, dtype=np.float64)
    adjoint[0:3, 0:3] = rotation
    adjoint[3:6, 3:6] = rotation
    adjoint[6:9, 6:9] = rotation
    adjoint[3:6, 0:3] = _wedge(velocity) @ rotation
    adjoint[6:9, 0:3] = _wedge(position) @ rotation
    return adjoint


@njit(cache=True)
def _predict_author_equations(
    rotation,
    velocity,
    position,
    covariance,
    bias,
    imu_sample,
    dt,
    noise_variances,
):
    """Execute the public author's predict equations without changing tuning."""
    gravity = np.array([0.0, 0.0, -GRAVITY_MAGNITUDE])

    # Preserve the public nominal bias layout exactly: accelerometer bias first,
    # gyroscope bias second.
    acceleration = (
        imu_sample[0:3] * GRAVITY_MAGNITUDE - bias[0:3]
    )
    angular_rate = imu_sample[3:6] * DEG_TO_RAD - bias[3:6]

    new_rotation = rotation @ _author_so3_exp(angular_rate * dt)
    world_acceleration = rotation @ acceleration + gravity
    new_velocity = velocity + world_acceleration * dt
    new_position = (
        position
        + velocity * dt
        + 0.5 * world_acceleration * dt * dt
    )

    # Preserve the author's ground clamp.
    if new_position[2] < 0.0:
        new_position[2] = 0.0
        new_velocity = np.zeros(3, dtype=np.float64)

    adjoint = _adjoint15(
        new_rotation,
        new_velocity,
        new_position,
    )
    dynamics = np.zeros((15, 15), dtype=np.float64)
    dynamics[0:3, 9:12] = -new_rotation
    dynamics[3:6, 0:3] = _wedge(gravity)
    dynamics[3:6, 9:12] = -_wedge(new_velocity) @ new_rotation
    dynamics[3:6, 12:15] = -new_rotation
    dynamics[6:9, 3:6] = np.eye(3, dtype=np.float64)
    dynamics[6:9, 9:12] = -_wedge(new_position) @ new_rotation

    # For this exact block structure A^4=0, so this is mathematically the same
    # expm(A*dt) used by the public source, without a general-purpose Padé call.
    scaled = dynamics * dt
    squared = scaled @ scaled
    transition = (
        np.eye(15, dtype=np.float64)
        + scaled
        + 0.5 * squared
        + (squared @ scaled) / 6.0
    )

    noise_map = transition @ adjoint
    scaled_noise_map = noise_map * (noise_variances * dt * dt)
    process_covariance = scaled_noise_map @ noise_map.T
    new_covariance = (
        transition @ covariance @ transition.T + process_covariance
    )
    return (
        new_rotation,
        new_velocity,
        new_position,
        new_covariance,
    )


@njit(cache=True)
def _inject_author_error(
    rotation_hat,
    velocity_hat,
    position_hat,
    bias,
    delta,
):
    phi = -delta[0:3]
    correction_rotation = _author_so3_exp(phi)
    correction_jacobian = _author_left_jacobian(phi)
    new_rotation = correction_rotation @ rotation_hat
    new_velocity = (
        correction_rotation @ velocity_hat
        + correction_jacobian @ (-delta[3:6])
    )
    new_position = (
        correction_rotation @ position_hat
        + correction_jacobian @ (-delta[6:9])
    )
    new_bias = bias - delta[9:15]
    return new_rotation, new_velocity, new_position, new_bias


@njit(cache=True)
def _correct_author_equations(
    rotation_hat,
    velocity_hat,
    position_hat,
    covariance_hat,
    bias,
    measurement,
    anchors,
    tag_offset,
    measurement_variance,
):
    """Execute the public author's gate and Joseph-form correction exactly."""
    anchor_a = anchors[int(measurement[0])]
    anchor_b = anchors[int(measurement[1])]
    tag_position = rotation_hat @ tag_offset + position_hat
    anchor_to_tag_a = anchor_a - tag_position
    anchor_to_tag_b = anchor_b - tag_position
    distance_a = math.sqrt(anchor_to_tag_a @ anchor_to_tag_a)
    distance_b = math.sqrt(anchor_to_tag_b @ anchor_to_tag_b)
    predicted = distance_b - distance_a
    residual = measurement[2] - predicted

    # Preserve the author's preliminary left-coordinate gate.
    h_left = np.zeros(15, dtype=np.float64)
    h_left[6:9] = (
        anchor_to_tag_b / distance_b
        - anchor_to_tag_a / distance_a
    )
    gate_variance = (
        h_left @ covariance_hat @ h_left + measurement_variance
    )
    gate_nis = residual * residual / gate_variance
    if gate_nis >= 25.0:
        return (
            rotation_hat,
            velocity_hat,
            position_hat,
            covariance_hat,
            bias,
            gate_nis,
            True,
        )

    # Directly expand H_left @ Ad(X_hat^-1); this is only an execution rewrite.
    transformed_position_row = h_left[6:9] @ rotation_hat.T
    h_right = np.zeros(15, dtype=np.float64)
    h_right[0:3] = (
        -transformed_position_row @ _wedge(position_hat)
    )
    h_right[6:9] = transformed_position_row

    covariance_projection = covariance_hat @ h_right
    innovation_variance = (
        h_right @ covariance_projection + measurement_variance
    )
    gain = covariance_projection / innovation_variance
    delta = gain * residual

    identity = np.eye(15, dtype=np.float64)
    residual_projection = identity - np.outer(gain, h_right)
    new_covariance = (
        residual_projection
        @ covariance_hat
        @ residual_projection.T
        + measurement_variance * np.outer(gain, gain)
    )
    (
        new_rotation,
        new_velocity,
        new_position,
        new_bias,
    ) = _inject_author_error(
        rotation_hat,
        velocity_hat,
        position_hat,
        bias,
        delta,
    )
    return (
        new_rotation,
        new_velocity,
        new_position,
        new_covariance,
        new_bias,
        gate_nis,
        False,
    )


@njit(cache=True)
def _write_state(
    output_index,
    rotation,
    velocity,
    position,
    covariance,
    bias,
    states,
    rotations,
    covariances,
    biases,
    record_covariances,
):
    states[output_index, 0:3] = _author_rotation_log(rotation)
    states[output_index, 3:6] = velocity
    states[output_index, 6:9] = position
    rotations[output_index] = rotation
    if record_covariances:
        covariances[output_index] = covariance
    biases[output_index] = bias


@njit(cache=True)
def _accelerated_replay(
    events,
    event_imu_indices,
    event_uwb_starts,
    event_uwb_ends,
    imu_times,
    imu,
    uwb_times,
    uwb,
    anchors,
    initial_rotation,
    initial_velocity,
    initial_position,
    initial_covariance,
    initial_bias,
    noise_variances,
    tag_offset,
    tdoa_variance,
    record_history,
    record_covariances,
):
    event_count = len(events)
    diagnostic_count = len(uwb) if record_history else 0
    output_count = event_count if record_history else 0
    states = np.empty((output_count, 9), dtype=np.float64)
    rotations = np.empty((output_count, 3, 3), dtype=np.float64)
    covariance_output_count = output_count if record_covariances else 0
    covariances = np.empty(
        (covariance_output_count, 15, 15),
        dtype=np.float64,
    )
    biases = np.empty((output_count, 6), dtype=np.float64)
    nis_times = np.empty(diagnostic_count, dtype=np.float64)
    nis_values = np.empty(diagnostic_count, dtype=np.float64)
    rejected = np.empty(diagnostic_count, dtype=np.bool_)

    # The public class keeps committed X/P separately from X_hat/P_hat.
    rotation = initial_rotation.copy()
    velocity = initial_velocity.copy()
    position = initial_position.copy()
    covariance = initial_covariance.copy()
    bias = initial_bias.copy()
    rotation_hat = rotation.copy()
    velocity_hat = velocity.copy()
    position_hat = position.copy()
    covariance_hat = covariance.copy()

    last_imu_index = 0
    predict_count = 0
    nis_count = 0
    rejected_count = 0

    initial_event_time = events[0]
    for measurement_index in range(
        event_uwb_starts[0],
        event_uwb_ends[0],
    ):
        (
            rotation,
            velocity,
            position,
            covariance,
            bias,
            nis,
            was_rejected,
        ) = _correct_author_equations(
            rotation_hat,
            velocity_hat,
            position_hat,
            covariance_hat,
            bias,
            uwb[measurement_index],
            anchors,
            tag_offset,
            tdoa_variance,
        )
        if record_history:
            nis_times[nis_count] = initial_event_time
            nis_values[nis_count] = nis
            rejected[nis_count] = was_rejected
        if was_rejected:
            rejected_count += 1
        nis_count += 1

    if record_history:
        _write_state(
            0,
            rotation,
            velocity,
            position,
            covariance,
            bias,
            states,
            rotations,
            covariances,
            biases,
            record_covariances,
        )

    for event_index in range(1, event_count):
        event_time = events[event_index]
        imu_index = event_imu_indices[event_index]
        if imu_index >= 0 and imu_index > last_imu_index:
            dt = event_time - imu_times[last_imu_index]
            (
                rotation_hat,
                velocity_hat,
                position_hat,
                covariance_hat,
            ) = _predict_author_equations(
                rotation,
                velocity,
                position,
                covariance,
                bias,
                imu[last_imu_index],
                dt,
                noise_variances,
            )
            rotation = rotation_hat.copy()
            velocity = velocity_hat.copy()
            position = position_hat.copy()
            covariance = covariance_hat.copy()
            last_imu_index = imu_index
            predict_count += 1

        for measurement_index in range(
            event_uwb_starts[event_index],
            event_uwb_ends[event_index],
        ):
            (
                rotation,
                velocity,
                position,
                covariance,
                bias,
                nis,
                was_rejected,
            ) = _correct_author_equations(
                rotation_hat,
                velocity_hat,
                position_hat,
                covariance_hat,
                bias,
                uwb[measurement_index],
                anchors,
                tag_offset,
                tdoa_variance,
            )
            if record_history:
                nis_times[nis_count] = event_time
                nis_values[nis_count] = nis
                rejected[nis_count] = was_rejected
            if was_rejected:
                rejected_count += 1
            nis_count += 1

        if record_history:
            _write_state(
                event_index,
                rotation,
                velocity,
                position,
                covariance,
                bias,
                states,
                rotations,
                covariances,
                biases,
                record_covariances,
            )

    return (
        states,
        rotations,
        covariances,
        biases,
        nis_times[:nis_count],
        nis_values[:nis_count],
        rejected[:nis_count],
        predict_count,
        nis_count,
        rejected_count,
    )


def prepare_event_plan(imu_times, uwb_times):
    """Create the same common event schedule using fixed integer lookups."""
    events = np.unique(np.concatenate((imu_times, uwb_times))).astype(
        np.float64,
        copy=False,
    )
    event_imu_indices = np.full(len(events), -1, dtype=np.int64)
    event_imu_indices[np.searchsorted(events, imu_times)] = np.arange(
        len(imu_times),
        dtype=np.int64,
    )
    event_uwb_starts = np.searchsorted(
        uwb_times,
        events,
        side="left",
    ).astype(np.int64, copy=False)
    event_uwb_ends = np.searchsorted(
        uwb_times,
        events,
        side="right",
    ).astype(np.int64, copy=False)
    return events, event_imu_indices, event_uwb_starts, event_uwb_ends


def prepare_accelerated_replay(dataset, estimator, plan=None):
    """Prepare contiguous inputs outside the measured replay region."""
    if plan is None:
        plan = prepare_event_plan(dataset.t_imu, dataset.t_uwb)
    noise_variances = np.concatenate(
        (
            estimator.w_gyr,
            estimator.w_acc,
            np.zeros(3),
            estimator.w_acc_b,
            estimator.w_gyr_b,
        )
    ).astype(np.float64)
    return (
        *plan,
        np.ascontiguousarray(dataset.t_imu, dtype=np.float64),
        np.ascontiguousarray(dataset.imu, dtype=np.float64),
        np.ascontiguousarray(dataset.t_uwb, dtype=np.float64),
        np.ascontiguousarray(dataset.uwb, dtype=np.float64),
        np.ascontiguousarray(dataset.anchors, dtype=np.float64),
        np.ascontiguousarray(estimator.X[:3, :3], dtype=np.float64),
        np.ascontiguousarray(estimator.X[:3, 3], dtype=np.float64),
        np.ascontiguousarray(estimator.X[:3, 4], dtype=np.float64),
        np.ascontiguousarray(estimator.P, dtype=np.float64),
        np.ascontiguousarray(estimator.bias, dtype=np.float64),
        noise_variances,
        np.ascontiguousarray(estimator.t_uv, dtype=np.float64),
        float(estimator.w_tdoa[0] ** 2),
    )


def execute_accelerated_replay(
    prepared,
    record_history=True,
    record_covariances=True,
):
    """Execute the author-equivalent native loop."""
    if record_covariances and not record_history:
        record_covariances = False
    return _accelerated_replay(
        *prepared,
        bool(record_history),
        bool(record_covariances),
    )


def ensure_accelerated_kernel():
    """Compile or load the fixed signature and report its one-time latency."""
    global _WARMUP_SECONDS
    if _WARMUP_SECONDS is not None:
        return 0.0
    start = time.perf_counter()
    events = np.array([0.0], dtype=np.float64)
    indices = np.array([-1], dtype=np.int64)
    bounds = np.array([0], dtype=np.int64)
    _accelerated_replay(
        events,
        indices,
        bounds,
        bounds,
        events,
        np.zeros((1, 6), dtype=np.float64),
        np.empty(0, dtype=np.float64),
        np.empty((0, 3), dtype=np.float64),
        np.zeros((2, 3), dtype=np.float64),
        np.eye(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        np.eye(15, dtype=np.float64),
        np.zeros(6, dtype=np.float64),
        np.zeros(15, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        0.13,
        False,
        False,
    )
    _WARMUP_SECONDS = time.perf_counter() - start
    return _WARMUP_SECONDS
