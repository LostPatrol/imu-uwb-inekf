"""Conventional 15-state TWR ESKF with the common frozen-R IMU step.

Errors are truth minus estimate: body-right attitude, world velocity/position,
then body gyro/accelerometer biases. Initial covariance uses these coordinates.
The historical noise parameters are per-sample variances, including bias rates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Iterable

import numpy as np


# SI conversion used by the common TWR implementation.
GRAVITY_MAGNITUDE = 9.81
# IMU angular measurements arrive in degrees per second.
DEG_TO_RAD = math.pi / 180.0
# Navigation/error-state dimension and order are fixed for all experiments.
ERROR_DIM = 15
# Right attitude error is expressed in the nominal body frame.
THETA = slice(0, 3)
# Velocity error is additive in the world frame.
VELOCITY = slice(3, 6)
# Position error is additive in the world frame.
POSITION = slice(6, 9)
# Gyro bias error has units of rad/s.
GYRO_BIAS = slice(9, 12)
# Accelerometer bias error has units of m/s^2.
ACCEL_BIAS = slice(12, 15)
# Avoid evaluating the direction of a zero length TWR prediction.
MIN_RANGE = 1e-12
# Taylor expansion avoids cancellation in SO(3) scalar coefficients.
SMALL_ANGLE = 1e-4


def skew(vector: np.ndarray) -> np.ndarray:
    """Return the cross-product matrix of a three-vector."""
    x, y, z = np.asarray(vector, dtype=float).reshape(3)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def rotation_terms(phi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return Exp(phi) and its SO(3) right Jacobian."""
    phi = np.asarray(phi, dtype=float).reshape(3)
    angle = float(np.linalg.norm(phi))
    angle2 = angle * angle
    if angle < SMALL_ANGLE:
        angle4 = angle2 * angle2
        sinc = 1.0 - angle2 / 6.0 + angle4 / 120.0
        cosc = 0.5 - angle2 / 24.0 + angle4 / 720.0
        third = 1.0 / 6.0 - angle2 / 120.0 + angle4 / 5040.0
    else:
        sinc = math.sin(angle) / angle
        cosc = (1.0 - math.cos(angle)) / angle2
        third = (angle - math.sin(angle)) / (angle2 * angle)
    wedge = skew(phi)
    wedge2 = wedge @ wedge
    identity = np.eye(3)
    return identity + sinc * wedge + cosc * wedge2, identity - cosc * wedge + third * wedge2


def symmetrize(matrix: np.ndarray) -> np.ndarray:
    """Remove numerical asymmetry without clipping covariance eigenvalues."""
    return 0.5 * (matrix + matrix.T)


def anchor_parameter(value: float | Iterable[float], name: str, positive: bool = False):
    """Validate a scalar or per-anchor bias/noise parameter."""
    array = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(array)) or array.size == 0:
        raise ValueError(f"{name} must contain finite values")
    if positive and np.any(array <= 0.0):
        raise ValueError(f"{name} must be positive")
    return float(array) if array.ndim == 0 else array.reshape(-1).copy()


def for_anchor(value: float | np.ndarray, index: int | None) -> float:
    """Resolve an indexed measurement parameter, allowing scalar defaults."""
    if np.isscalar(value):
        return float(value)
    if index is None:
        if value.size == 1:
            return float(value[0])
        raise ValueError("anchor_index is required for per-anchor parameters")
    index = int(index)
    if index < 0 or index >= value.size:
        raise ValueError("anchor_index is outside per-anchor parameters")
    return float(value[index])


@dataclass
class UpdateStats:
    """Measurement accounting compatible with the common TWR filters."""

    accepted: int = 0
    rejected: int = 0
    invalid: int = 0
    nis_sum: float = 0.0
    nis_max: float = 0.0
    downweighted: int = 0
    weight_sum: float = 0.0
    per_anchor: dict[int, dict[str, int]] = field(default_factory=dict)

    @property
    def mean_nis(self) -> float:
        """Return nominal NIS averaged over valid attempted measurements."""
        count = self.accepted + self.rejected
        return self.nis_sum / count if count else math.nan

    @property
    def mean_accepted_weight(self) -> float:
        """Return the mean Huber weight among accepted measurements."""
        return self.weight_sum / self.accepted if self.accepted else math.nan

    def record(self, nis: float, accepted: bool, weight: float, anchor_index: int | None) -> None:
        """Record a valid measurement and its nominal innovation statistic."""
        self.nis_sum += float(nis)
        self.nis_max = max(self.nis_max, float(nis))
        downweighted = accepted and weight < 1.0 - 1e-12
        if accepted:
            self.accepted += 1
            self.weight_sum += float(weight)
            self.downweighted += int(downweighted)
        else:
            self.rejected += 1
        if anchor_index is not None:
            counts = self.per_anchor.setdefault(
                int(anchor_index), {"accepted": 0, "rejected": 0, "downweighted": 0}
            )
            counts["accepted" if accepted else "rejected"] += 1
            counts["downweighted"] += int(downweighted)


class ESKF15:
    """Body-right attitude ESKF with additive world translation and biases.

    ``covariance`` is physical [dtheta_body, dv_world, dp_world, dbg, dba].
    It must not be copied directly from an invariant filter's internal P.
    """

    def __init__(
        self,
        state: np.ndarray,
        covariance: np.ndarray,
        bias: np.ndarray,
        range_sigma: float | Iterable[float] = 0.20,
        range_bias: float | Iterable[float] = -0.18,
        gate_threshold: float = 25.0,
        gate_warmup_updates: int = 0,
        huber_threshold: float | None = None,
        *,
        ground_clamp: bool = False,
    ) -> None:
        """Initialize the shared physical prior and scalar/per-anchor TWR model."""
        self.X = np.asarray(state, dtype=float).copy()
        self.P = symmetrize(np.asarray(covariance, dtype=float).copy())
        self.bias = np.asarray(bias, dtype=float).reshape(6).copy()
        if self.X.shape != (5, 5) or self.P.shape != (ERROR_DIM, ERROR_DIM):
            raise ValueError("state and covariance must have shapes (5,5) and (15,15)")
        if not all(np.all(np.isfinite(item)) for item in (self.X, self.P, self.bias)):
            raise ValueError("initial state, covariance, and bias must be finite")
        if not np.isfinite(gate_threshold) or gate_threshold <= 0.0:
            raise ValueError("gate_threshold must be positive and finite")
        if gate_warmup_updates < 0:
            raise ValueError("gate_warmup_updates must be non-negative")
        if huber_threshold is not None and (
            not np.isfinite(huber_threshold) or huber_threshold <= 0.0
        ):
            raise ValueError("huber_threshold must be positive and finite")
        self.range_sigma = anchor_parameter(range_sigma, "range_sigma", positive=True)
        self.range_bias = anchor_parameter(range_bias, "range_bias")
        self.gate_threshold = float(gate_threshold)
        self.gate_warmup_updates = int(gate_warmup_updates)
        self.huber_threshold = huber_threshold
        self.ground_clamp = bool(ground_clamp)

        # Match the TWR InEKF's SI per-sample IMU variances, not continuous PSDs.
        self.w_gyr = np.full(3, 0.1**2)
        self.w_acc = np.full(3, 2.0**2)
        # Historical bias-rate samples produce end-step increments proportional to dt.
        self.w_gyr_b = np.full(3, 0.000001**2)
        self.w_acc_b = np.full(3, 0.0001**2)
        self.stats = UpdateStats()
        self.last_nis = math.nan
        self.last_innovation = math.nan
        self.rej_cnt = 0

    @staticmethod
    def transition_and_noise(
        rotation: np.ndarray, acceleration: np.ndarray, omega: np.ndarray, dt: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the exact nominal-step Jacobian and independent sample-noise map."""
        increment, right_jacobian = rotation_terms(omega * dt)
        transition = np.eye(ERROR_DIM)
        transition[THETA, THETA] = increment.T
        transition[THETA, GYRO_BIAS] = -right_jacobian * dt
        attitude_acceleration = -rotation @ skew(acceleration)
        transition[VELOCITY, THETA] = attitude_acceleration * dt
        transition[VELOCITY, ACCEL_BIAS] = -rotation * dt
        transition[POSITION, THETA] = attitude_acceleration * (0.5 * dt**2)
        transition[POSITION, VELOCITY] = np.eye(3) * dt
        transition[POSITION, ACCEL_BIAS] = -rotation * (0.5 * dt**2)

        # Columns are held gyro/accel measurement errors and end-step bias rates.
        noise = np.zeros((ERROR_DIM, 12))
        noise[THETA, :3] = -right_jacobian * dt
        noise[VELOCITY, 3:6] = -rotation * dt
        noise[POSITION, 3:6] = -rotation * (0.5 * dt**2)
        noise[GYRO_BIAS, 6:9] = np.eye(3) * dt
        noise[ACCEL_BIAS, 9:12] = np.eye(3) * dt
        return transition, noise

    def predict(self, imu: Iterable[float], dt: float):
        """Propagate [acc_g, gyro_deg/s] using the common pre-step frozen-R model."""
        sample = np.asarray(imu, dtype=float).reshape(6)
        dt = float(dt)
        if not np.isfinite(dt) or dt < 0.0 or not np.all(np.isfinite(sample)):
            raise ValueError("IMU and non-negative dt must be finite")
        if dt == 0.0:
            return self.X, self.P, self.bias
        transition, noise = discrete_error_matrices(
            self.X, self.bias, sample, dt, self.ground_clamp
        )
        self.X = nominal_step(self.X, self.bias, sample, dt, self.ground_clamp)
        variances = np.concatenate((self.w_gyr, self.w_acc, self.w_gyr_b, self.w_acc_b))
        process_covariance = (noise * variances[np.newaxis, :]) @ noise.T
        self.P = symmetrize(transition @ self.P @ transition.T + process_covariance)
        return self.X, self.P, self.bias

    @staticmethod
    def reset_jacobian(delta: np.ndarray) -> np.ndarray:
        """Map posterior additive errors after R <- R Exp(delta_theta)."""
        reset = np.eye(ERROR_DIM)
        reset[THETA, THETA] = rotation_terms(np.asarray(delta)[:3])[1]
        return reset

    def inject(self, delta: np.ndarray) -> None:
        """Inject a truth-minus-estimate correction into the nominal state."""
        self.X[:3, :3] = self.X[:3, :3] @ rotation_terms(delta[THETA])[0]
        self.X[:3, 3] += delta[VELOCITY]
        self.X[:3, 4] += delta[POSITION]
        self.bias += delta[9:15]

    def range_model(self, anchor: Iterable[float]) -> tuple[float, np.ndarray]:
        """Return zero-lever-arm TWR and its conventional error Jacobian."""
        anchor = np.asarray(anchor, dtype=float).reshape(3)
        displacement = self.X[:3, 4] - anchor
        predicted = float(np.linalg.norm(displacement))
        if not np.isfinite(predicted) or predicted < MIN_RANGE:
            raise ValueError("TWR direction is undefined")
        row = np.zeros((1, ERROR_DIM))
        row[0, POSITION] = displacement / predicted
        return predicted, row

    def correct_twr(
        self, anchor: Iterable[float], measurement: float, anchor_index: int | None = None
    ) -> bool:
        """Apply common warm-up/NIS/Huber policy, Joseph update, and attitude reset."""
        self.last_nis = math.nan
        self.last_innovation = math.nan
        try:
            sigma = for_anchor(self.range_sigma, anchor_index)
            measurement = float(measurement) - for_anchor(self.range_bias, anchor_index)
            if not np.isfinite(measurement) or measurement <= 0.0:
                raise ValueError("bias-corrected TWR must be positive and finite")
            predicted, row = self.range_model(anchor)
        except ValueError:
            self.stats.invalid += 1
            return False
        variance = sigma**2
        state_variance = float((row @ self.P @ row.T).item())
        innovation_variance = state_variance + variance
        if not np.isfinite(innovation_variance) or innovation_variance <= 0.0:
            self.stats.invalid += 1
            return False
        innovation = measurement - predicted
        nis = innovation**2 / innovation_variance
        self.last_nis = nis
        self.last_innovation = innovation
        gate_active = self.stats.accepted + self.stats.rejected >= self.gate_warmup_updates
        accepted = not gate_active or nis <= self.gate_threshold
        weight = 1.0 if accepted else 0.0
        if accepted and gate_active and self.huber_threshold is not None:
            standardized = math.sqrt(max(0.0, nis))
            if standardized > self.huber_threshold:
                weight = self.huber_threshold / standardized
        self.stats.record(nis, accepted, weight, anchor_index)
        if not accepted:
            self.rej_cnt += 1
            return False

        effective_variance = variance / weight
        gain = self.P @ row.T / (state_variance + effective_variance)
        delta = gain[:, 0] * innovation
        residual_map = np.eye(ERROR_DIM) - gain @ row
        posterior = residual_map @ self.P @ residual_map.T + effective_variance * gain @ gain.T
        self.inject(delta)
        # Jr(delta_theta) is the exact local derivative of Log(Exp(-delta) Exp(delta+e)).
        reset = self.reset_jacobian(delta)
        self.P = symmetrize(reset @ posterior @ reset.T)
        return True


def nominal_step(
    state: np.ndarray, bias: np.ndarray, imu: Iterable[float], dt: float,
    ground_clamp: bool = False,
) -> np.ndarray:
    """Return the common frozen-R IMU state step without mutating the inputs."""
    sample = np.asarray(imu, dtype=float).reshape(6)
    bias = np.asarray(bias, dtype=float).reshape(6)
    state = np.asarray(state, dtype=float)
    rotation = state[:3, :3]
    velocity = state[:3, 3]
    acceleration = sample[:3] * GRAVITY_MAGNITUDE - bias[3:]
    omega = sample[3:] * DEG_TO_RAD - bias[:3]
    world_acceleration = rotation @ acceleration + np.array([0.0, 0.0, -GRAVITY_MAGNITUDE])
    result = state.copy()
    result[:3, :3] = rotation @ rotation_terms(omega * dt)[0]
    result[:3, 3] = velocity + world_acceleration * dt
    result[:3, 4] += velocity * dt + world_acceleration * (0.5 * dt**2)
    if ground_clamp and result[2, 4] < 0.0:
        result[2, 4] = 0.0
        result[:3, 3] = 0.0
    return result


def discrete_error_matrices(
    state: np.ndarray, bias: np.ndarray, imu: Iterable[float], dt: float,
    ground_clamp: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact F and G; noise order is [gyro,accel,bg_rate,ba_rate]."""
    sample = np.asarray(imu, dtype=float).reshape(6)
    bias = np.asarray(bias, dtype=float).reshape(6)
    acceleration = sample[:3] * GRAVITY_MAGNITUDE - bias[3:]
    omega = sample[3:] * DEG_TO_RAD - bias[:3]
    transition, noise = ESKF15.transition_and_noise(state[:3, :3], acceleration, omega, dt)
    if ground_clamp and nominal_step(state, bias, sample, dt)[2, 4] < 0.0:
        # Differentiate the optional deterministic clamp away from its boundary.
        transition[VELOCITY, :] = 0.0
        transition[8, :] = 0.0
        noise[VELOCITY, :] = 0.0
        noise[8, :] = 0.0
    return transition, noise
