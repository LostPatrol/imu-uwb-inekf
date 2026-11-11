"""Physical moving-window IMU/TWR cases with completely unknown initial SO(3).

There is no static prefix; local gyro integration always starts from identity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

# Use the same world gravity convention as the existing initialization paper.
GRAVITY = np.array([0.0, 0.0, -9.81])
# Match the paper's held-sample stress-window acquisition budget.
DURATION_S = 4.0
IMU_DT_S = 0.005
RANGE_COUNT = 240
RANGE_SIGMA_M = 0.10


@dataclass
class ColdCase:
    """Raw sensor observations and separately retained truth for offline scoring."""
    seed: int
    motion: str
    tilt_deg: float
    geometry: str
    grid: np.ndarray
    event_times: np.ndarray
    tau: np.ndarray
    gyro: np.ndarray
    force: np.ndarray
    local_rotations: np.ndarray
    event_rotations: np.ndarray
    force_displacement: np.ndarray
    lever_arm: np.ndarray
    shape: np.ndarray
    anchors: np.ndarray
    ranges: np.ndarray
    sigmas: np.ndarray
    truth_rotation: np.ndarray
    truth_position: np.ndarray
    truth_velocity: np.ndarray
    truth_tag_positions: np.ndarray
    truth_imu_positions: np.ndarray
    truth_event_acceleration: np.ndarray


def skew(vector):
    """Return the cross-product matrix of a three-vector."""
    x, y, z = vector
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def integrate_held(acceleration, grid, events):
    """Double-integrate held vector samples with exact within-interval evaluation."""
    dt = np.diff(grid)
    velocity = np.vstack((np.zeros(3), np.cumsum(acceleration * dt[:, None], axis=0)))
    position = np.vstack((np.zeros(3), np.cumsum(
        velocity[:-1] * dt[:, None] + .5 * acceleration * dt[:, None]**2, axis=0)))
    index = np.minimum(np.searchsorted(grid, events, side="right") - 1, len(dt) - 1)
    offset = events - grid[index]
    tau = events - grid[-1]
    values = position[index] + offset[:, None] * velocity[index] + .5 * offset[:, None]**2 * acceleration[index]
    return values - position[-1] - tau[:, None] * velocity[-1]


def make_case(seed, motion="rich", tilt_deg=120.0, geometry="normal", lever_arm=None):
    """Generate asynchronous ranges and exact raw IMU for a nonstatic window."""
    rng = np.random.default_rng(seed)
    grid = np.arange(round(DURATION_S / IMU_DT_S) + 1) * IMU_DT_S
    events = np.sort(rng.uniform(0.0, DURATION_S, RANGE_COUNT))
    tau = events - DURATION_S
    anchor_height = 3.0 if geometry == "normal" else .1
    anchor_set = np.array([[x, y, z] for x in (-4., 4.) for y in (-4., 4.) for z in (0., anchor_height)])
    anchors = anchor_set[rng.integers(0, len(anchor_set), len(events))]
    bearing, yaw, tilt_axis = rng.uniform(-np.pi, np.pi, 3)
    position = np.array([30 * np.cos(bearing), 30 * np.sin(bearing), 1.2])
    velocity = rng.uniform(-.5, .5, 3)
    horizontal_axis = np.array([np.cos(tilt_axis), np.sin(tilt_axis), 0.])
    initial_rotation = (Rotation.from_rotvec(np.array([0., 0., yaw])).as_matrix()
                        @ Rotation.from_rotvec(np.deg2rad(tilt_deg) * horizontal_axis).as_matrix())
    # The strictly increasing yaw component prevents every interval being static.
    angles = np.column_stack((.18 * np.sin(.9 * grid), .14 * np.sin(1.3 * grid), .45 * grid))
    nominal = Rotation.from_euler("xyz", angles).as_matrix()
    increments = np.einsum("nji,njk->nik", nominal[:-1], nominal[1:])
    gyro = Rotation.from_matrix(increments).as_rotvec() / IMU_DT_S
    # Reconstruct Q only from gyro, starting at I; no initial attitude is supplied.
    local = np.empty_like(nominal)
    local[0] = np.eye(3)
    for index, increment in enumerate(Rotation.from_rotvec(gyro * IMU_DT_S).as_matrix()):
        local[index + 1] = local[index] @ increment
    sample_tau = grid[:-1] - DURATION_S
    acceleration = np.column_stack((-.8 * 1.7**2 * np.sin(1.7 * sample_tau),
                                    -.6 * 1.1**2 * np.cos(1.1 * sample_tau),
                                    -.15 * .7**2 * np.sin(.7 * sample_tau)))
    if motion == "weak":
        acceleration = .03 * acceleration + np.array([.15, -.10, .05])
    elif motion == "constant_acceleration":
        acceleration[:] = [.4, -.3, .1]
    elif motion == "constant_velocity":
        acceleration[:] = 0.
    elif motion != "rich":
        raise ValueError(f"Unknown motion: {motion}")
    initial_frame_acceleration = (acceleration - GRAVITY) @ initial_rotation
    force = np.einsum("nji,nj->ni", local[:-1], initial_frame_acceleration)
    integrated_acceleration = np.einsum("nij,nj->ni", local[:-1], force)
    displacement = integrate_held(integrated_acceleration, grid, events)
    world_displacement = integrate_held(acceleration, grid, events)
    event_indices = np.minimum(np.searchsorted(grid, events, side="right") - 1, len(gyro) - 1)
    fractional_increment = Rotation.from_rotvec(gyro[event_indices] * (events-grid[event_indices])[:, None]).as_matrix()
    event_rotation = np.einsum("nij,njk->nik", local[event_indices], fractional_increment)
    lever = np.zeros(3) if lever_arm is None else np.asarray(lever_arm, dtype=float)
    shape = displacement + np.einsum("nij,j->ni", event_rotation, lever)
    imu_position = position + tau[:, None] * velocity + world_displacement
    tag_position = imu_position + np.einsum("ij,njk,k->ni", initial_rotation, event_rotation, lever)
    predicted = position + tau[:, None] * velocity + .5 * tau[:, None]**2 * GRAVITY + shape @ initial_rotation.T
    np.testing.assert_allclose(predicted, tag_position, atol=2e-11, rtol=0)
    ranges = np.linalg.norm(tag_position - anchors, axis=1) + rng.normal(0., RANGE_SIGMA_M, len(events))
    return ColdCase(seed, motion, float(tilt_deg), geometry, grid, events, tau, gyro, force, local,
                    event_rotation, displacement, lever, shape, anchors, ranges,
                    np.full(len(events), RANGE_SIGMA_M), initial_rotation, position, velocity,
                    tag_position, imu_position, acceleration[event_indices])


def residual_jacobian(position, velocity, rotation, case):
    """Differentiate raw ranges using a right local rotation perturbation."""
    predicted = (position + case.tau[:, None] * velocity + .5 * case.tau[:, None]**2 * GRAVITY
                 + case.shape @ rotation.T)
    delta = predicted - case.anchors
    distance = np.maximum(np.linalg.norm(delta, axis=1), 1e-12)
    direction = delta / distance[:, None]
    # n' (-C [h]x) = (h x C'n)' for right tangent coordinates.
    rotation_jac = np.cross(case.shape, direction @ rotation)
    jacobian = np.column_stack((direction, case.tau[:, None] * direction, rotation_jac)) / case.sigmas[:, None]
    return (distance-case.ranges) / case.sigmas, jacobian


def profile_rotation_information(case):
    """Eliminate endpoint p/v and retain the full three-axis range information."""
    _, jacobian = residual_jacobian(case.truth_position, case.truth_velocity, case.truth_rotation, case)
    u, singular, _ = np.linalg.svd(jacobian[:, :6], full_matrices=False)
    rank = int(np.sum(singular > singular[0] * 1e-10))
    projected = jacobian[:, 6:] - u[:, :rank] @ (u[:, :rank].T @ jacobian[:, 6:])
    return np.maximum(np.linalg.eigvalsh(projected.T @ projected), 0.)


def wahba_seed(case, radio_positions, weights=None):
    """Fit C by weighted Wahba after removing arbitrary affine translation."""
    weight = np.ones(len(case.tau)) if weights is None else np.asarray(weights)
    root = np.sqrt(weight)
    affine = np.column_stack((np.ones(len(case.tau)), case.tau)) * root[:, None]
    local = case.shape * root[:, None]
    world = (radio_positions-.5 * case.tau[:, None]**2 * GRAVITY) * root[:, None]
    local -= affine @ np.linalg.lstsq(affine, local, rcond=1e-10)[0]
    world -= affine @ np.linalg.lstsq(affine, world, rcond=1e-10)[0]
    u, singular, vt = np.linalg.svd(world.T @ local)
    rotation = u @ np.diag([1., 1., np.linalg.det(u @ vt)]) @ vt
    translation = np.column_stack((np.ones(len(case.tau)), case.tau))
    remaining = radio_positions - .5 * case.tau[:, None]**2 * GRAVITY - case.shape @ rotation.T
    coefficients = np.linalg.lstsq(translation * root[:, None], remaining * root[:, None], rcond=1e-10)[0]
    return coefficients[0], coefficients[1], rotation, singular


def gravity_seed(case):
    """Use the moving-window mean specific force as a conventional gravity guess."""
    mean_force = np.mean(case.force, axis=0)
    local_vertical = mean_force / np.linalg.norm(mean_force)
    # This minimum-rotation alignment supplies no external yaw measurement.
    return Rotation.align_vectors(np.array([[0., 0., 1.]]), local_vertical[None, :])[0].as_matrix()


def compensated_gravity_seed(case):
    """Estimate initial gravity after undoing measured body rotation, without static data."""
    initial_force = np.einsum("nij,nj->ni", case.local_rotations[:-1], case.force)
    mean_force = np.mean(initial_force, axis=0)
    local_vertical = mean_force / np.linalg.norm(mean_force)
    return Rotation.align_vectors(np.array([[0., 0., 1.]]), local_vertical[None, :])[0].as_matrix()


def rotation_error_deg(estimated, truth):
    """Return geodesic SO(3) error, including initial tilt as well as yaw."""
    return float(np.rad2deg(Rotation.from_matrix(estimated.T @ truth).magnitude()))
