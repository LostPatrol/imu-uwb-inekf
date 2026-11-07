"""Numba-compiled fixed-size replay kernel for the production InEKF path.

The kernel intentionally implements the same 15-state equations, process-noise
semantics, UWB gate, scalar rank-one update, and initial stationary gyroscope
update as ``FilterConfig.proposed``.  It changes execution, not the estimator
model: event planning and all output buffers are fixed before entering the
compiled loop.
"""

import math
import time

import numpy as np

try:
    from numba import njit
except ImportError as exc:  # pragma: no cover - exercised by deployment setup
    raise ImportError(
        "The task03_compiled variant requires Numba. Recreate the Conda "
        "environment from conda/conda-env.yaml or install numba."
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
def _motion_terms(phi):
    angle2 = phi[0] * phi[0] + phi[1] * phi[1] + phi[2] * phi[2]
    angle = math.sqrt(angle2)
    phi_wedge = _wedge(phi)
    phi_wedge2 = phi_wedge @ phi_wedge
    if angle < 1e-4:
        angle4 = angle2 * angle2
        exp_1 = 1.0 - angle2 / 6.0 + angle4 / 120.0
        exp_2 = 0.5 - angle2 / 24.0 + angle4 / 720.0
        gamma1_2 = 1.0 / 6.0 - angle2 / 120.0 + angle4 / 5040.0
        gamma2_2 = 1.0 / 24.0 - angle2 / 720.0 + angle4 / 40320.0
    else:
        sine = math.sin(angle)
        cosine = math.cos(angle)
        exp_1 = sine / angle
        exp_2 = (1.0 - cosine) / angle2
        gamma1_2 = (angle - sine) / (angle * angle2)
        gamma2_2 = (
            angle2 + 2.0 * cosine - 2.0
        ) / (2.0 * angle2 * angle2)
    identity = np.eye(3, dtype=np.float64)
    rotation = identity + exp_1 * phi_wedge + exp_2 * phi_wedge2
    gamma1 = identity + exp_2 * phi_wedge + gamma1_2 * phi_wedge2
    gamma2 = 0.5 * identity + gamma1_2 * phi_wedge + gamma2_2 * phi_wedge2
    return rotation, gamma1, gamma2


@njit(cache=True)
def _exp_and_left_jacobian(phi):
    angle2 = phi[0] * phi[0] + phi[1] * phi[1] + phi[2] * phi[2]
    angle = math.sqrt(angle2)
    phi_wedge = _wedge(phi)
    phi_wedge2 = phi_wedge @ phi_wedge
    if angle < 1e-8:
        angle4 = angle2 * angle2
        exp_1 = 1.0 - angle2 / 6.0 + angle4 / 120.0
        exp_2 = 0.5 - angle2 / 24.0 + angle4 / 720.0
        jacobian_2 = 1.0 / 6.0 - angle2 / 120.0 + angle4 / 5040.0
    else:
        sine = math.sin(angle)
        cosine = math.cos(angle)
        exp_1 = sine / angle
        exp_2 = (1.0 - cosine) / angle2
        jacobian_2 = (angle - sine) / (angle * angle2)
    identity = np.eye(3, dtype=np.float64)
    rotation = identity + exp_1 * phi_wedge + exp_2 * phi_wedge2
    jacobian = identity + exp_2 * phi_wedge + jacobian_2 * phi_wedge2
    return rotation, jacobian


@njit(cache=True)
def _rotation_log(rotation):
    cosine = 0.5 * (
        rotation[0, 0] + rotation[1, 1] + rotation[2, 2]
    ) - 0.5
    cosine = min(1.0, max(-1.0, cosine))
    angle = math.acos(cosine)
    antisymmetric = np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ],
        dtype=np.float64,
    )
    if angle < 1e-7:
        return (0.5 + angle * angle / 12.0) * antisymmetric
    if math.pi - angle >= 1e-5:
        return (angle / (2.0 * math.sin(angle))) * antisymmetric

    # Stable matrix-to-quaternion branch for rotations close to pi.
    quaternion = np.zeros(4, dtype=np.float64)
    if rotation[0, 0] >= rotation[1, 1] and rotation[0, 0] >= rotation[2, 2]:
        scale = 2.0 * math.sqrt(max(1e-30, 1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]))
        quaternion[0] = (rotation[2, 1] - rotation[1, 2]) / scale
        quaternion[1] = 0.25 * scale
        quaternion[2] = (rotation[0, 1] + rotation[1, 0]) / scale
        quaternion[3] = (rotation[0, 2] + rotation[2, 0]) / scale
    elif rotation[1, 1] >= rotation[2, 2]:
        scale = 2.0 * math.sqrt(max(1e-30, 1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]))
        quaternion[0] = (rotation[0, 2] - rotation[2, 0]) / scale
        quaternion[1] = (rotation[0, 1] + rotation[1, 0]) / scale
        quaternion[2] = 0.25 * scale
        quaternion[3] = (rotation[1, 2] + rotation[2, 1]) / scale
    else:
        scale = 2.0 * math.sqrt(max(1e-30, 1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]))
        quaternion[0] = (rotation[1, 0] - rotation[0, 1]) / scale
        quaternion[1] = (rotation[0, 2] + rotation[2, 0]) / scale
        quaternion[2] = (rotation[1, 2] + rotation[2, 1]) / scale
        quaternion[3] = 0.25 * scale
    vector_norm = math.sqrt(
        quaternion[1] * quaternion[1]
        + quaternion[2] * quaternion[2]
        + quaternion[3] * quaternion[3]
    )
    if vector_norm < 1e-15:
        return np.zeros(3, dtype=np.float64)
    quaternion_angle = 2.0 * math.atan2(vector_norm, abs(quaternion[0]))
    sign = 1.0 if quaternion[0] >= 0.0 else -1.0
    return sign * quaternion_angle * quaternion[1:4] / vector_norm


@njit(cache=True)
def _inject_error(rotation, velocity, position, bias, delta):
    correction_rotation, correction_jacobian = _exp_and_left_jacobian(
        -delta[0:3]
    )
    corrected_rotation = correction_rotation @ rotation
    corrected_velocity = (
        correction_rotation @ velocity
        + correction_jacobian @ (-delta[3:6])
    )
    corrected_position = (
        correction_rotation @ position
        + correction_jacobian @ (-delta[6:9])
    )
    corrected_bias = bias - delta[9:15]
    return (
        corrected_rotation,
        corrected_velocity,
        corrected_position,
        corrected_bias,
    )


@njit(cache=True)
def _predict(
    rotation,
    velocity,
    position,
    covariance,
    bias,
    imu_sample,
    dt,
    noise_variances,
):
    acceleration = imu_sample[0:3] * GRAVITY_MAGNITUDE - bias[3:6]
    angular_rate = imu_sample[3:6] * DEG_TO_RAD - bias[0:3]
    gravity = np.array([0.0, 0.0, -GRAVITY_MAGNITUDE])
    increment, gamma1, gamma2 = _motion_terms(angular_rate * dt)
    new_rotation = rotation @ increment
    new_velocity = (
        velocity
        + gravity * dt
        + rotation @ (gamma1 @ acceleration) * dt
    )
    new_position = (
        position
        + velocity * dt
        + 0.5 * gravity * dt * dt
        + rotation @ (gamma2 @ acceleration) * dt * dt
    )

    dynamics = np.zeros((15, 15), dtype=np.float64)
    dynamics[0:3, 9:12] = -new_rotation
    dynamics[3:6, 0:3] = _wedge(gravity)
    dynamics[3:6, 9:12] = -_wedge(new_velocity) @ new_rotation
    dynamics[3:6, 12:15] = -new_rotation
    dynamics[6:9, 3:6] = np.eye(3)
    dynamics[6:9, 9:12] = -_wedge(new_position) @ new_rotation

    scaled_dynamics = dynamics * dt
    squared_dynamics = scaled_dynamics @ scaled_dynamics
    transition = (
        np.eye(15)
        + scaled_dynamics
        + 0.5 * squared_dynamics
        + (squared_dynamics @ scaled_dynamics) / 6.0
    )

    adjoint = np.eye(15, dtype=np.float64)
    adjoint[0:3, 0:3] = new_rotation
    adjoint[3:6, 3:6] = new_rotation
    adjoint[6:9, 6:9] = new_rotation
    adjoint[3:6, 0:3] = _wedge(new_velocity) @ new_rotation
    adjoint[6:9, 0:3] = _wedge(new_position) @ new_rotation
    noise_map = transition @ adjoint
    scaled_noise_map = noise_map * (noise_variances * dt * dt)
    process_covariance = scaled_noise_map @ noise_map.T
    new_covariance = (
        transition @ covariance @ transition.T + process_covariance
    )
    new_covariance = 0.5 * (new_covariance + new_covariance.T)
    return new_rotation, new_velocity, new_position, new_covariance


@njit(cache=True)
def _correct_uwb(
    rotation,
    velocity,
    position,
    covariance,
    bias,
    measurement,
    anchors,
    tag_offset,
    measurement_variance,
):
    anchor_a = anchors[int(measurement[0])]
    anchor_b = anchors[int(measurement[1])]
    tag_position = position + rotation @ tag_offset
    delta_a = tag_position - anchor_a
    delta_b = tag_position - anchor_b
    distance_a = math.sqrt(delta_a @ delta_a)
    distance_b = math.sqrt(delta_b @ delta_b)
    if min(distance_a, distance_b) < 1e-9:
        raise ValueError("UWB tag position coincides with an anchor")
    gradient = delta_b / distance_b - delta_a / distance_a
    residual = measurement[2] - (distance_b - distance_a)

    h = np.zeros(15, dtype=np.float64)
    h_position = -gradient
    transformed_position_row = h_position @ rotation.T
    h[0:3] = -transformed_position_row @ _wedge(position)
    h[6:9] = transformed_position_row
    covariance_projection = covariance @ h
    innovation_variance = h @ covariance_projection + measurement_variance
    if innovation_variance <= np.finfo(np.float64).tiny:
        raise FloatingPointError("UWB innovation variance must be positive")
    nis = residual * residual / innovation_variance
    if nis >= 25.0:
        return (
            rotation,
            velocity,
            position,
            covariance,
            bias,
            nis,
            True,
        )

    delta = covariance_projection * (residual / innovation_variance)
    new_covariance = covariance - np.outer(
        covariance_projection,
        covariance_projection,
    ) / innovation_variance
    new_covariance = 0.5 * (new_covariance + new_covariance.T)
    (
        new_rotation,
        new_velocity,
        new_position,
        new_bias,
    ) = _inject_error(rotation, velocity, position, bias, delta)
    return (
        new_rotation,
        new_velocity,
        new_position,
        new_covariance,
        new_bias,
        nis,
        False,
    )


@njit(cache=True)
def _correct_stationary(
    rotation,
    velocity,
    position,
    covariance,
    bias,
    gyroscope_degrees_per_second,
    measurement_variance,
):
    residual = (
        gyroscope_degrees_per_second * DEG_TO_RAD - bias[0:3]
    )
    observation = np.zeros((3, 15), dtype=np.float64)
    observation[:, 9:12] = -np.eye(3)
    measurement_covariance = np.eye(3) * measurement_variance
    innovation = (
        observation @ covariance @ observation.T + measurement_covariance
    )
    gain = np.linalg.solve(innovation, observation @ covariance).T
    delta = gain @ residual
    identity = np.eye(15)
    residual_projection = identity - gain @ observation
    new_covariance = (
        residual_projection @ covariance @ residual_projection.T
        + gain @ measurement_covariance @ gain.T
    )
    new_covariance = 0.5 * (new_covariance + new_covariance.T)
    (
        new_rotation,
        new_velocity,
        new_position,
        new_bias,
    ) = _inject_error(rotation, velocity, position, bias, delta)
    # Per-step SO(3) projection is disabled in proposed. This correction starts
    # from two analytic rotations, whose orthogonality error is machine-scale.
    return (
        new_rotation,
        new_velocity,
        new_position,
        new_covariance,
        new_bias,
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
):
    states[output_index, 0:3] = _rotation_log(rotation)
    states[output_index, 3:6] = velocity
    states[output_index, 6:9] = position
    rotations[output_index] = rotation
    covariances[output_index] = covariance
    biases[output_index] = bias


@njit(cache=True)
def _compiled_replay(
    events,
    event_imu_indices,
    event_uwb_starts,
    event_uwb_ends,
    imu_times,
    imu,
    uwb_times,
    uwb,
    anchors,
    stationary,
    initial_rotation,
    initial_velocity,
    initial_position,
    initial_covariance,
    initial_bias,
    noise_variances,
    tag_offset,
    tdoa_variance,
    zero_angular_rate_variance,
    record_history,
):
    event_count = len(events)
    measurement_count = len(uwb)
    output_count = event_count if record_history else 0
    diagnostic_count = measurement_count if record_history else 0
    states = np.empty((output_count, 9), dtype=np.float64)
    rotations = np.empty((output_count, 3, 3), dtype=np.float64)
    covariances = np.empty((output_count, 15, 15), dtype=np.float64)
    biases = np.empty((output_count, 6), dtype=np.float64)
    nis_times = np.empty(diagnostic_count, dtype=np.float64)
    nis_values = np.empty(diagnostic_count, dtype=np.float64)
    rejected = np.empty(diagnostic_count, dtype=np.bool_)

    rotation = initial_rotation.copy()
    velocity = initial_velocity.copy()
    position = initial_position.copy()
    covariance = initial_covariance.copy()
    bias = initial_bias.copy()
    last_imu_index = 0
    last_stationary_update = -math.inf
    initial_rest_seen = False
    initial_rest_complete = False
    nis_count = 0
    rejected_count = 0
    stationary_update_count = 0
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
        )

    for event_index in range(1, event_count):
        event_time = events[event_index]
        imu_index = event_imu_indices[event_index]
        if imu_index >= 0:
            if imu_index > last_imu_index:
                dt = event_time - imu_times[last_imu_index]
                (
                    rotation,
                    velocity,
                    position,
                    covariance,
                ) = _predict(
                    rotation,
                    velocity,
                    position,
                    covariance,
                    bias,
                    imu[last_imu_index],
                    dt,
                    noise_variances,
                )
                last_imu_index = imu_index
            if (
                stationary[imu_index]
                and not initial_rest_complete
                and event_time - last_stationary_update >= 0.10
            ):
                (
                    rotation,
                    velocity,
                    position,
                    covariance,
                    bias,
                ) = _correct_stationary(
                    rotation,
                    velocity,
                    position,
                    covariance,
                    bias,
                    imu[imu_index, 3:6],
                    zero_angular_rate_variance,
                )
                last_stationary_update = event_time
                initial_rest_seen = True
                stationary_update_count += 1
            elif initial_rest_seen and not stationary[imu_index]:
                initial_rest_complete = True

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
            ) = _correct_uwb(
                rotation,
                velocity,
                position,
                covariance,
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
            )

    return (
        states,
        rotations,
        covariances,
        biases,
        nis_times[:nis_count],
        nis_values[:nis_count],
        rejected[:nis_count],
        stationary_update_count,
        nis_count,
        rejected_count,
    )


def prepare_event_plan(imu_times, uwb_times):
    """Create fixed integer lookup arrays for the compiled event loop."""
    events = np.unique(np.concatenate((imu_times, uwb_times))).astype(
        np.float64,
        copy=False,
    )
    event_imu_indices = np.full(len(events), -1, dtype=np.int64)
    imu_event_positions = np.searchsorted(events, imu_times)
    event_imu_indices[imu_event_positions] = np.arange(
        len(imu_times),
        dtype=np.int64,
    )
    event_uwb_starts = np.searchsorted(uwb_times, events, side="left").astype(
        np.int64,
        copy=False,
    )
    event_uwb_ends = np.searchsorted(uwb_times, events, side="right").astype(
        np.int64,
        copy=False,
    )
    return events, event_imu_indices, event_uwb_starts, event_uwb_ends


def ensure_compiled_kernel():
    """Compile/load the fixed signature once and return one-time latency."""
    global _WARMUP_SECONDS
    if _WARMUP_SECONDS is not None:
        return 0.0
    start = time.perf_counter()
    events = np.array([0.0], dtype=np.float64)
    indices = np.array([-1], dtype=np.int64)
    bounds = np.array([0], dtype=np.int64)
    _compiled_replay(
        events,
        indices,
        bounds,
        bounds,
        events,
        np.zeros((1, 6), dtype=np.float64),
        np.empty(0, dtype=np.float64),
        np.empty((0, 3), dtype=np.float64),
        np.zeros((2, 3), dtype=np.float64),
        np.zeros(1, dtype=np.bool_),
        np.eye(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        np.eye(15, dtype=np.float64),
        np.zeros(6, dtype=np.float64),
        np.zeros(15, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        0.05,
        math.radians(0.5) ** 2,
        False,
    )
    _WARMUP_SECONDS = time.perf_counter() - start
    return _WARMUP_SECONDS


def prepare_compiled_replay(dataset, estimator, stationary, plan=None):
    """Prepare immutable fixed-size inputs outside a measured replay scope."""
    if plan is None:
        plan = prepare_event_plan(dataset.t_imu, dataset.t_uwb)
    noise_variances = np.concatenate(
        (
            estimator.w_gyr,
            estimator.w_acc,
            np.zeros(3),
            estimator.w_gyr_b,
            estimator.w_acc_b,
        )
    ).astype(np.float64)
    return (
        *plan,
        np.ascontiguousarray(dataset.t_imu, dtype=np.float64),
        np.ascontiguousarray(dataset.imu, dtype=np.float64),
        np.ascontiguousarray(dataset.t_uwb, dtype=np.float64),
        np.ascontiguousarray(dataset.uwb, dtype=np.float64),
        np.ascontiguousarray(dataset.anchors, dtype=np.float64),
        np.ascontiguousarray(stationary, dtype=np.bool_),
        np.ascontiguousarray(estimator.X[:3, :3], dtype=np.float64),
        np.ascontiguousarray(estimator.X[:3, 3], dtype=np.float64),
        np.ascontiguousarray(estimator.X[:3, 4], dtype=np.float64),
        np.ascontiguousarray(estimator.P, dtype=np.float64),
        np.ascontiguousarray(estimator.bias, dtype=np.float64),
        noise_variances,
        np.asarray(estimator.t_uv, dtype=np.float64),
        float(estimator.w_tdoa[0] ** 2),
        float(estimator.w_zero_angular_rate),
    )


def execute_compiled_replay(prepared, record_history=True):
    """Execute a prepared replay with or without diagnostic history."""
    return _compiled_replay(*prepared, bool(record_history))


def run_compiled_replay(
    dataset,
    estimator,
    stationary,
    plan=None,
    record_history=True,
):
    """Run the compiled production-equivalent loop and return fixed outputs."""
    if plan is None:
        plan = prepare_event_plan(dataset.t_imu, dataset.t_uwb)
    prepared = prepare_compiled_replay(
        dataset,
        estimator,
        stationary,
        plan=plan,
    )
    return plan[0], execute_compiled_replay(
        prepared,
        record_history=record_history,
    )
