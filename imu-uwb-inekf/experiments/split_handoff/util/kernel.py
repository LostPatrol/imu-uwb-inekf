"""Compiled right-invariant IMU/TDoA replay with the common discrete IMU model."""

import numpy as np
from numba import njit

# Gravity, antenna calibration and per-sample sensor noise match the shared replay.
GRAVITY = 9.81
LEVER = np.array([-.01245, .00127, .0908])
NOISE = np.array([.1**2] * 3 + [2.**2] * 3 + [1e-6**2] * 3 + [1e-4**2] * 3)


@njit
def skew(vector):
    """Return a three-vector's cross-product matrix."""
    x, y, z = vector
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


@njit
def rotation_terms(phi):
    """Return the SO(3) exponential and right Jacobian with stable small angles."""
    angle = np.sqrt(phi @ phi)
    squared = angle * angle
    if angle < 1e-4:
        sinc = 1 - squared / 6 + squared**2 / 120
        cosc = .5 - squared / 24 + squared**2 / 720
        third = 1 / 6 - squared / 120 + squared**2 / 5040
    else:
        sinc = np.sin(angle) / angle
        cosc = (1 - np.cos(angle)) / squared
        third = (angle - np.sin(angle)) / (squared * angle)
    wedge = skew(phi)
    wedge2 = wedge @ wedge
    return np.eye(3) + sinc * wedge + cosc * wedge2, np.eye(3) - cosc * wedge + third * wedge2


@njit
def mapping(state):
    """Map physical body-right errors into right-invariant coordinates."""
    result = np.eye(15)
    rotation = state[:3, :3].copy()
    result[:3, :3] = rotation
    result[3:6, :3] = skew(state[:3, 3]) @ rotation
    result[6:9, :3] = skew(state[:3, 4]) @ rotation
    return result


@njit
def inverse_mapping(state):
    """Invert the physical-to-invariant coordinate transformation analytically."""
    result = np.eye(15)
    result[:3, :3] = state[:3, :3].T
    result[3:6, :3] = -skew(state[:3, 3])
    result[6:9, :3] = -skew(state[:3, 4])
    return result


@njit
def predict(state, covariance, bias, sample, dt):
    """Propagate the exact frozen-R discrete physical model in invariant coordinates."""
    rotation = state[:3, :3].copy()
    acceleration = sample[:3] * GRAVITY - bias[3:]
    omega = sample[3:] * (np.pi / 180) - bias[:3]
    increment, right_jacobian = rotation_terms(omega * dt)
    transition = np.eye(15)
    transition[:3, :3] = increment.T
    transition[:3, 9:12] = -right_jacobian * dt
    coupling = -rotation @ skew(acceleration)
    transition[3:6, :3] = coupling * dt
    transition[3:6, 12:15] = -rotation * dt
    transition[6:9, :3] = coupling * (.5 * dt**2)
    transition[6:9, 3:6] = np.eye(3) * dt
    transition[6:9, 12:15] = -rotation * (.5 * dt**2)
    noise = np.zeros((15, 12))
    noise[:3, :3] = -right_jacobian * dt
    noise[3:6, 3:6] = -rotation * dt
    noise[6:9, 3:6] = -rotation * (.5 * dt**2)
    noise[9:12, 6:9] = np.eye(3) * dt
    noise[12:15, 9:12] = np.eye(3) * dt
    inverse = inverse_mapping(state)
    world_acceleration = rotation @ acceleration + np.array([0., 0., -GRAVITY])
    result = state.copy()
    result[:3, :3] = rotation @ increment
    result[:3, 4] += state[:3, 3] * dt + world_acceleration * (.5 * dt**2)
    result[:3, 3] += world_acceleration * dt
    change = mapping(result)
    invariant_transition = change @ transition @ inverse
    mapped_noise = change @ noise
    posterior = invariant_transition @ covariance @ invariant_transition.T + (mapped_noise * NOISE) @ mapped_noise.T
    return result, .5 * (posterior + posterior.T)


@njit
def inject(state, bias, delta):
    """Apply a left SE_2(3) correction with additive body IMU biases."""
    increment, right_jacobian = rotation_terms(-delta[:3])
    rotation = increment.T
    left_jacobian = right_jacobian
    result = state.copy()
    result[:3, :3] = rotation @ state[:3, :3]
    result[:3, 3] = rotation @ state[:3, 3] + left_jacobian @ delta[3:6]
    result[:3, 4] = rotation @ state[:3, 4] + left_jacobian @ delta[6:9]
    return result, bias + delta[9:15]


@njit
def reset_jacobian(delta):
    """Differentiate a finite left group injection by a converged adjoint series."""
    algebra = np.zeros((9, 9))
    for start in (0, 3, 6):
        algebra[start:start + 3, start:start + 3] = skew(delta[:3])
    algebra[3:6, :3] = skew(delta[3:6])
    algebra[6:9, :3] = skew(delta[6:9])
    jacobian = np.eye(9)
    term = np.eye(9)
    for order in range(1, 50):
        term = term @ algebra / (order + 1)
        jacobian += term
        if np.max(np.abs(term)) < 1e-14:
            break
    result = np.eye(15)
    result[:9, :9] = jacobian
    return result


@njit
def correct(state, covariance, bias, observation, anchors, stamp, variance):
    """Fuse one native TDoA observation with Huber weighting and a common NIS gate."""
    lag = observation[0] - stamp
    tag = state[:3, 4] + lag * state[:3, 3] + state[:3, :3].copy() @ LEVER
    da, db = tag - anchors[int(observation[1])], tag - anchors[int(observation[2])]
    ra, rb = np.sqrt(da @ da), np.sqrt(db @ db)
    gradient = db / rb - da / ra
    row = np.zeros(15)
    row[:3] = -gradient @ skew(tag)
    row[3:6] = gradient * lag
    row[6:9] = gradient
    residual = observation[3] - (rb - ra)
    projected = covariance @ row
    nis = residual**2 / (row @ projected + variance)
    if nis > 25:
        return state, covariance, bias, False
    weight = min(1., 2. / max(np.sqrt(nis), 1e-12))
    total = row @ projected + variance / weight
    delta = projected * (residual / total)
    posterior = covariance - np.outer(projected, projected) / total
    result, new_bias = inject(state, bias, delta)
    reset = reset_jacobian(delta)
    posterior = reset @ posterior @ reset.T
    return result, .5 * (posterior + posterior.T), new_bias, True


@njit
def replay(times, imu, radio, anchors, state, covariance, bias, start, end, variance,
           outage_start=-1., outage_end=-1.):
    """Replay a bounded interval once, preserving output during a real radio outage."""
    state, covariance, bias = state.copy(), covariance.copy(), bias.copy()
    begin = np.searchsorted(times, start, side="right")
    stop = np.searchsorted(times, end, side="right")
    radio_index = np.searchsorted(radio[:, 0], start, side="right")
    output = np.zeros((max(1, stop - begin + 1), 19))
    output_index, accepted, rejected = 0, 0, 0
    previous, failed = start, False
    for k in range(begin, stop):
        stamp = times[k]
        sample = .5 * (imu[k - 1] + imu[k])
        state, covariance = predict(state, covariance, bias, sample, stamp - previous)
        while radio_index < len(radio) and radio[radio_index, 0] <= stamp:
            observation = radio[radio_index]
            if not (outage_start <= observation[0] < outage_end):
                state, covariance, bias, keep = correct(state, covariance, bias, observation, anchors, stamp, variance)
                accepted += int(keep)
                rejected += int(not keep)
            radio_index += 1
        if k % 5 == 0:
            output[output_index, 0] = stamp
            output[output_index, 1:4] = state[:3, 4]
            output[output_index, 4:7] = state[:3, 3]
            output[output_index, 7:16] = state[:3, :3].flatten()
            output[output_index, 16:19] = bias[3:]
            output_index += 1
        previous = stamp
        if not np.all(np.isfinite(state)) or np.max(np.abs(state[:3, 4])) > 1000:
            failed = True
            break
    return state, covariance, bias, output[:output_index].copy(), accepted, rejected, failed
