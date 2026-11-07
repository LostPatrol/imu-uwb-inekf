"""Raw and lifted TWR filters with shared calibration and robust weighting.

The lifted observer uses an exact linear transition for world acceleration;
its translation state remains separate from the IMU attitude and bias core.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Callable, Iterable

import numpy as np

from inekf import (
    GRAVITY_MAGNITUDE,
    InEKF,
    SE3_2_adjoint,
    SE3_2_exp,
    SE3_2_inv,
)


Q = 0  # Half squared position norm.
S = 1  # Position-velocity inner product.
H = 2  # Squared velocity norm.
P_SLICE = slice(3, 6)  # World position coordinates.
V_SLICE = slice(6, 9)  # World velocity coordinates.
ONE = 9  # Exact homogeneous coordinate.
LIFTED_DIM = 10  # Size of the homogeneous lifted state.


def _as_vector3(value: Iterable[float], name: str) -> np.ndarray:
    """Validate a finite vector with three coordinates."""

    result = np.asarray(value, dtype=np.float64).reshape(-1)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain three finite values")
    return result


def _symmetrize(matrix: np.ndarray) -> np.ndarray:
    """Remove round-off asymmetry without hiding a genuinely indefinite P."""

    return 0.5 * (matrix + matrix.T)


def _anchor_parameter(
    value: float | Iterable[float], name: str, *, positive: bool = False
) -> float | np.ndarray:
    """Validate a scalar or per-anchor measurement-model parameter."""

    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        result = float(array)
        if not np.isfinite(result) or (positive and result <= 0.0):
            qualifier = "positive and " if positive else ""
            raise ValueError(f"{name} must be {qualifier}finite")
        return result
    result = array.reshape(-1).copy()
    if result.size == 0 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} per-anchor array must contain finite values")
    if positive and np.any(result <= 0.0):
        raise ValueError(f"{name} per-anchor values must be positive")
    return result


def _for_anchor(
    value: float | np.ndarray, anchor_index: int | None, name: str
) -> float:
    """Return a scalar parameter, requiring an index for a parameter array."""

    if np.isscalar(value):
        return float(value)
    if anchor_index is None:
        if value.size == 1:
            return float(value[0])
        raise ValueError(f"anchor_index is required for per-anchor {name}")
    index = int(anchor_index)
    if index < 0 or index >= value.size:
        raise ValueError(f"anchor_index {index} is outside per-anchor {name}")
    return float(value[index])


def _validate_robust_settings(
    gate_threshold: float,
    gate_warmup_updates: int,
    huber_threshold: float | None,
) -> None:
    """Validate the hard gate, initialization warmup, and optional Huber rule."""

    if not np.isfinite(gate_threshold) or gate_threshold <= 0.0:
        raise ValueError("gate_threshold must be positive and finite")
    if gate_warmup_updates < 0:
        raise ValueError("gate_warmup_updates must be non-negative")
    if huber_threshold is not None and (
        not np.isfinite(huber_threshold) or huber_threshold <= 0.0
    ):
        raise ValueError("huber_threshold must be positive and finite")


def _robust_measurement_variance(
    variance: float,
    nis: float,
    *,
    gate_is_active: bool,
    gate_threshold: float,
    huber_threshold: float | None,
) -> tuple[bool, float, float]:
    """Apply nominal-NIS gating and Huber R/w weighting after warmup."""

    if not gate_is_active:
        return True, variance, 1.0
    if nis > gate_threshold:
        return False, variance, 0.0
    if huber_threshold is None:
        return True, variance, 1.0
    standardized = math.sqrt(max(0.0, nis))
    if standardized <= huber_threshold:
        return True, variance, 1.0
    weight = huber_threshold / standardized
    return True, variance / weight, weight


@dataclass
class UpdateStats:
    """Accumulate nominal NIS, robust weights, and per-anchor decisions."""

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
        """Return nominal NIS averaged over valid measurement decisions."""

        if self.accepted + self.rejected == 0:
            return math.nan
        return self.nis_sum / (self.accepted + self.rejected)

    @property
    def mean_accepted_weight(self) -> float:
        """Return the mean Huber weight among accepted measurements."""

        if self.accepted == 0:
            return math.nan
        return self.weight_sum / self.accepted

    def record(
        self,
        nis: float,
        accepted: bool,
        *,
        weight: float = 1.0,
        anchor_index: int | None = None,
    ) -> None:
        """Record one valid update and its optional anchor-specific weight."""

        self.nis_sum += float(nis)
        self.nis_max = max(self.nis_max, float(nis))
        if accepted:
            self.accepted += 1
            self.weight_sum += float(weight)
            if weight < 1.0 - 1e-12:
                self.downweighted += 1
        else:
            self.rejected += 1
        if anchor_index is not None:
            counts = self.per_anchor.setdefault(
                int(anchor_index),
                {"accepted": 0, "rejected": 0, "downweighted": 0},
            )
            key = "accepted" if accepted else "rejected"
            counts[key] += 1
            if accepted and weight < 1.0 - 1e-12:
                counts["downweighted"] += 1


class LiftedRangeKF:
    """Kalman observer for the lifted half-squared TWR model.

    The final homogeneous coordinate is kept exactly equal to one.  It has
    zero covariance and is used only to express the affine acceleration and
    anchor terms in matrix form.
    """

    def __init__(
        self,
        position: Iterable[float],
        velocity: Iterable[float] | None = None,
        *,
        position_std: float = 8.0,
        velocity_std: float = 2.0,
        q_std: float = 60.0,
        s_std: float = 30.0,
        h_std: float = 20.0,
        accel_noise_std: float = 1.0,
        range_sigma: float | Iterable[float] = 0.20,
        range_bias: float | Iterable[float] = -0.18,
        gate_threshold: float = 25.0,
        gate_warmup_updates: int = 64,
        huber_threshold: float | None = None,
        project_constraints: bool = False,
    ) -> None:
        """Initialize the lifted prior, indexed calibration, and robust policy."""

        p = _as_vector3(position, "position")
        v = np.zeros(3) if velocity is None else _as_vector3(velocity, "velocity")
        if min(position_std, velocity_std, q_std, s_std, h_std) <= 0:
            raise ValueError("initial standard deviations must be positive")
        if accel_noise_std < 0:
            raise ValueError("acceleration noise must be non-negative")
        _validate_robust_settings(
            gate_threshold, gate_warmup_updates, huber_threshold
        )

        self.chi = np.zeros(LIFTED_DIM, dtype=np.float64)
        self.chi[Q] = 0.5 * p @ p
        self.chi[S] = p @ v
        self.chi[H] = v @ v
        self.chi[P_SLICE] = p
        self.chi[V_SLICE] = v
        self.chi[ONE] = 1.0

        # Map an ordinary [p, v] prior into the lifted coordinates, then add
        # independent uncertainty for the three auxiliary coordinates.  The
        # latter is important near p=v=0, where a first-order mapping alone
        # would incorrectly make q, s, and h deterministic.
        jacobian = np.zeros((LIFTED_DIM, 6), dtype=np.float64)
        jacobian[Q, :3] = p
        jacobian[S, :3] = v
        jacobian[S, 3:] = p
        jacobian[H, 3:] = 2.0 * v
        jacobian[P_SLICE, :3] = np.eye(3)
        jacobian[V_SLICE, 3:] = np.eye(3)
        base_covariance = np.diag(
            [position_std**2] * 3 + [velocity_std**2] * 3
        )
        self.P = jacobian @ base_covariance @ jacobian.T
        self.P[Q, Q] += q_std**2
        self.P[S, S] += s_std**2
        self.P[H, H] += h_std**2
        self.P[ONE, :] = 0.0
        self.P[:, ONE] = 0.0

        self.accel_noise_std = float(accel_noise_std)
        self.range_sigma = _anchor_parameter(
            range_sigma, "range_sigma", positive=True
        )
        self.range_bias = _anchor_parameter(range_bias, "range_bias")
        self.gate_threshold = float(gate_threshold)
        self.gate_warmup_updates = int(gate_warmup_updates)
        self.huber_threshold = (
            None if huber_threshold is None else float(huber_threshold)
        )
        self.project_constraints = bool(project_constraints)
        self.stats = UpdateStats()
        self.last_nis = math.nan
        self.last_innovation = math.nan

    @property
    def position(self) -> np.ndarray:
        return self.chi[P_SLICE].copy()

    @property
    def velocity(self) -> np.ndarray:
        return self.chi[V_SLICE].copy()

    @staticmethod
    def continuous_matrix(world_acceleration: Iterable[float]) -> np.ndarray:
        """Return A(u) for dot(chi)=A(u)chi."""

        u = _as_vector3(world_acceleration, "world_acceleration")
        matrix = np.zeros((LIFTED_DIM, LIFTED_DIM), dtype=np.float64)
        matrix[Q, S] = 1.0
        matrix[S, H] = 1.0
        matrix[S, P_SLICE] = u
        matrix[H, V_SLICE] = 2.0 * u
        matrix[P_SLICE, V_SLICE] = np.eye(3)
        matrix[V_SLICE, ONE] = u
        return matrix

    @staticmethod
    def transition_matrix(
        world_acceleration: Iterable[float], dt: float
    ) -> np.ndarray:
        """Exact exp(A(u) dt) for a piecewise-constant acceleration."""

        u = _as_vector3(world_acceleration, "world_acceleration")
        if not np.isfinite(dt) or dt < 0:
            raise ValueError("dt must be finite and non-negative")

        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt2 * dt2
        u2 = float(u @ u)
        phi = np.eye(LIFTED_DIM, dtype=np.float64)

        phi[Q, S] = dt
        phi[Q, H] = 0.5 * dt2
        phi[Q, P_SLICE] = 0.5 * u * dt2
        phi[Q, V_SLICE] = 0.5 * u * dt3
        phi[Q, ONE] = 0.125 * u2 * dt4

        phi[S, H] = dt
        phi[S, P_SLICE] = u * dt
        phi[S, V_SLICE] = 1.5 * u * dt2
        phi[S, ONE] = 0.5 * u2 * dt3

        phi[H, V_SLICE] = 2.0 * u * dt
        phi[H, ONE] = u2 * dt2

        phi[P_SLICE, V_SLICE] = np.eye(3) * dt
        phi[P_SLICE, ONE] = 0.5 * u * dt2
        phi[V_SLICE, ONE] = u * dt
        return phi

    def _acceleration_noise_jacobian(
        self, world_acceleration: np.ndarray, dt: float
    ) -> np.ndarray:
        """First-order sensitivity of the exact step to acceleration noise."""

        p = self.chi[P_SLICE]
        v = self.chi[V_SLICE]
        u = world_acceleration
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt2 * dt2

        jacobian = np.zeros((LIFTED_DIM, 3), dtype=np.float64)
        jacobian[Q, :] = 0.5 * p * dt2 + 0.5 * v * dt3 + 0.25 * u * dt4
        jacobian[S, :] = p * dt + 1.5 * v * dt2 + u * dt3
        jacobian[H, :] = 2.0 * v * dt + 2.0 * u * dt2
        jacobian[P_SLICE, :] = np.eye(3) * (0.5 * dt2)
        jacobian[V_SLICE, :] = np.eye(3) * dt
        return jacobian

    def predict(self, world_acceleration: Iterable[float], dt: float) -> None:
        u = _as_vector3(world_acceleration, "world_acceleration")
        phi = self.transition_matrix(u, dt)
        noise_jacobian = self._acceleration_noise_jacobian(u, dt)
        process_covariance = (
            self.accel_noise_std**2 * noise_jacobian @ noise_jacobian.T
        )

        self.chi = phi @ self.chi
        self.P = _symmetrize(phi @ self.P @ phi.T + process_covariance)
        self.chi[ONE] = 1.0
        self.P[ONE, :] = 0.0
        self.P[:, ONE] = 0.0

    def project_onto_manifold(self) -> None:
        """Project q/s/h and P onto the six-dimensional physical manifold.

        This optional diagnostic deliberately sacrifices the independent
        lifted coordinates (and therefore the large-error linear-observer
        mechanism).  It is useful for separating errors caused by the
        half-squared observation from errors caused by drift away from
        q=0.5*p.T*p, s=p.T*v, and h=v.T*v.
        """

        p = self.chi[P_SLICE].copy()
        v = self.chi[V_SLICE].copy()
        projection_jacobian = np.zeros(
            (LIFTED_DIM, LIFTED_DIM), dtype=np.float64
        )
        projection_jacobian[Q, P_SLICE] = p
        projection_jacobian[S, P_SLICE] = v
        projection_jacobian[S, V_SLICE] = p
        projection_jacobian[H, V_SLICE] = 2.0 * v
        projection_jacobian[P_SLICE, P_SLICE] = np.eye(3)
        projection_jacobian[V_SLICE, V_SLICE] = np.eye(3)
        projection_jacobian[ONE, ONE] = 1.0

        self.chi[Q] = 0.5 * p @ p
        self.chi[S] = p @ v
        self.chi[H] = v @ v
        self.P = _symmetrize(
            projection_jacobian @ self.P @ projection_jacobian.T
        )
        self.chi[ONE] = 1.0
        self.P[ONE, :] = 0.0
        self.P[:, ONE] = 0.0

    @staticmethod
    def observation_row(anchor: Iterable[float]) -> np.ndarray:
        anchor_vector = _as_vector3(anchor, "anchor")
        row = np.zeros((1, LIFTED_DIM), dtype=np.float64)
        row[0, Q] = 1.0
        row[0, P_SLICE] = -anchor_vector
        row[0, ONE] = 0.5 * anchor_vector @ anchor_vector
        return row

    def _processed_measurement(
        self, measured_range: float, anchor_index: int | None = None
    ) -> tuple[float, float]:
        """Debias half squared range using the selected anchor noise model."""

        range_bias = _for_anchor(self.range_bias, anchor_index, "range_bias")
        range_sigma = _for_anchor(self.range_sigma, anchor_index, "range_sigma")
        corrected_range = float(measured_range) - range_bias
        if not np.isfinite(corrected_range) or corrected_range <= 0:
            raise ValueError("TWR range must be finite and positive after bias correction")
        sigma2 = range_sigma**2
        measurement = 0.5 * (corrected_range**2 - sigma2)
        variance = corrected_range**2 * sigma2 + 0.5 * sigma2**2
        return measurement, max(variance, np.finfo(np.float64).eps)

    def correct(
        self,
        anchor: Iterable[float],
        measured_range: float,
        anchor_index: int | None = None,
    ) -> bool:
        """Apply one calibrated half squared range with optional Huber weighting."""

        try:
            measurement, variance = self._processed_measurement(
                measured_range, anchor_index
            )
        except ValueError:
            self.stats.invalid += 1
            return False

        row = self.observation_row(anchor)
        innovation = float(measurement - (row @ self.chi).item())
        innovation_variance = float((row @ self.P @ row.T).item() + variance)
        if not np.isfinite(innovation_variance) or innovation_variance <= 0:
            self.stats.invalid += 1
            return False

        nis = innovation * innovation / innovation_variance
        self.last_nis = nis
        self.last_innovation = innovation
        gate_is_active = (
            self.stats.accepted + self.stats.rejected >= self.gate_warmup_updates
        )
        accepted, effective_variance, weight = _robust_measurement_variance(
            variance,
            nis,
            gate_is_active=gate_is_active,
            gate_threshold=self.gate_threshold,
            huber_threshold=self.huber_threshold,
        )
        self.stats.record(
            nis,
            accepted,
            weight=weight,
            anchor_index=anchor_index,
        )
        if not accepted:
            return False

        innovation_variance = float(
            (row @ self.P @ row.T).item() + effective_variance
        )
        gain = self.P @ row.T / innovation_variance
        identity = np.eye(LIFTED_DIM)
        residual_map = identity - gain @ row
        self.chi = self.chi + gain[:, 0] * innovation
        self.P = _symmetrize(
            residual_map @ self.P @ residual_map.T
            + effective_variance * gain @ gain.T
        )
        self.chi[ONE] = 1.0
        self.P[ONE, :] = 0.0
        self.P[:, ONE] = 0.0
        if self.project_constraints:
            self.project_onto_manifold()
        return True

    def correct_batch(
        self,
        anchors: np.ndarray,
        ranges: np.ndarray,
        changed: np.ndarray | None = None,
    ) -> int:
        """Update changed anchor ranges using their calibration indices."""

        anchor_array = np.asarray(anchors, dtype=np.float64)
        range_array = np.asarray(ranges, dtype=np.float64).reshape(-1)
        if anchor_array.shape != (range_array.size, 3):
            raise ValueError("anchors must have shape (number of ranges, 3)")
        if changed is None:
            changed_array = np.ones(range_array.size, dtype=bool)
        else:
            changed_array = np.asarray(changed, dtype=bool).reshape(-1)
            if changed_array.shape != range_array.shape:
                raise ValueError("changed mask has the wrong shape")

        accepted = 0
        for anchor_index, (anchor, measured_range, is_new) in enumerate(
            zip(anchor_array, range_array, changed_array)
        ):
            if is_new and self.correct(
                anchor, float(measured_range), anchor_index=anchor_index
            ):
                accepted += 1
        return accepted

    def constraint_residual(self) -> np.ndarray:
        """Return the three lift-manifold residuals for diagnostics only."""

        p = self.chi[P_SLICE]
        v = self.chi[V_SLICE]
        return np.array(
            [
                self.chi[Q] - 0.5 * p @ p,
                self.chi[S] - p @ v,
                self.chi[H] - v @ v,
            ]
        )


class DirectRangeEKF:
    """Conventional translational TWR EKF with a position-dependent Jacobian."""

    def __init__(
        self,
        position: Iterable[float],
        velocity: Iterable[float] | None = None,
        *,
        position_std: float = 8.0,
        velocity_std: float = 2.0,
        accel_noise_std: float = 1.0,
        range_sigma: float | Iterable[float] = 0.20,
        range_bias: float | Iterable[float] = -0.18,
        gate_threshold: float = 25.0,
        gate_warmup_updates: int = 64,
        huber_threshold: float | None = None,
    ) -> None:
        """Initialize the direct range prior and shared measurement policy."""

        p = _as_vector3(position, "position")
        v = np.zeros(3) if velocity is None else _as_vector3(velocity, "velocity")
        self.x = np.concatenate((p, v))
        self.P = np.diag([position_std**2] * 3 + [velocity_std**2] * 3)
        self.accel_noise_std = float(accel_noise_std)
        _validate_robust_settings(
            gate_threshold, gate_warmup_updates, huber_threshold
        )
        self.range_sigma = _anchor_parameter(
            range_sigma, "range_sigma", positive=True
        )
        self.range_bias = _anchor_parameter(range_bias, "range_bias")
        self.gate_threshold = float(gate_threshold)
        self.gate_warmup_updates = int(gate_warmup_updates)
        self.huber_threshold = (
            None if huber_threshold is None else float(huber_threshold)
        )
        self.stats = UpdateStats()
        self.last_nis = math.nan
        self.last_innovation = math.nan

    @property
    def position(self) -> np.ndarray:
        return self.x[:3].copy()

    @property
    def velocity(self) -> np.ndarray:
        return self.x[3:].copy()

    def predict(self, world_acceleration: Iterable[float], dt: float) -> None:
        u = _as_vector3(world_acceleration, "world_acceleration")
        if not np.isfinite(dt) or dt < 0:
            raise ValueError("dt must be finite and non-negative")
        transition = np.eye(6)
        transition[:3, 3:] = np.eye(3) * dt
        input_map = np.vstack((0.5 * dt * dt * np.eye(3), dt * np.eye(3)))
        self.x[:3] += self.x[3:] * dt + 0.5 * u * dt * dt
        self.x[3:] += u * dt
        self.P = _symmetrize(
            transition @ self.P @ transition.T
            + self.accel_noise_std**2 * input_map @ input_map.T
        )

    def correct(
        self,
        anchor: Iterable[float],
        measured_range: float,
        anchor_index: int | None = None,
    ) -> bool:
        """Apply one indexed raw range using a hard gate and optional Huber rule."""

        anchor_vector = _as_vector3(anchor, "anchor")
        try:
            range_bias = _for_anchor(self.range_bias, anchor_index, "range_bias")
            range_sigma = _for_anchor(
                self.range_sigma, anchor_index, "range_sigma"
            )
        except ValueError:
            self.stats.invalid += 1
            return False
        corrected_range = float(measured_range) - range_bias
        if not np.isfinite(corrected_range) or corrected_range <= 0:
            self.stats.invalid += 1
            return False
        displacement = self.x[:3] - anchor_vector
        predicted_range = float(np.linalg.norm(displacement))
        if predicted_range < 1e-12:
            self.stats.invalid += 1
            return False

        row = np.zeros((1, 6), dtype=np.float64)
        row[0, :3] = displacement / predicted_range
        innovation = corrected_range - predicted_range
        variance = range_sigma**2
        innovation_variance = float((row @ self.P @ row.T).item() + variance)
        nis = innovation * innovation / innovation_variance
        self.last_nis = nis
        self.last_innovation = innovation
        gate_is_active = (
            self.stats.accepted + self.stats.rejected >= self.gate_warmup_updates
        )
        accepted, effective_variance, weight = _robust_measurement_variance(
            variance,
            nis,
            gate_is_active=gate_is_active,
            gate_threshold=self.gate_threshold,
            huber_threshold=self.huber_threshold,
        )
        self.stats.record(
            nis,
            accepted,
            weight=weight,
            anchor_index=anchor_index,
        )
        if not accepted:
            return False

        innovation_variance = float(
            (row @ self.P @ row.T).item() + effective_variance
        )
        gain = self.P @ row.T / innovation_variance
        identity = np.eye(6)
        residual_map = identity - gain @ row
        self.x = self.x + gain[:, 0] * innovation
        self.P = _symmetrize(
            residual_map @ self.P @ residual_map.T
            + effective_variance * gain @ gain.T
        )
        return True

    def correct_batch(
        self,
        anchors: np.ndarray,
        ranges: np.ndarray,
        changed: np.ndarray | None = None,
    ) -> int:
        """Update changed anchor ranges with matching per-anchor parameters."""

        anchor_array = np.asarray(anchors, dtype=np.float64)
        range_array = np.asarray(ranges, dtype=np.float64).reshape(-1)
        if anchor_array.shape != (range_array.size, 3):
            raise ValueError("anchors must have shape (number of ranges, 3)")
        if changed is None:
            changed_array = np.ones(range_array.size, dtype=bool)
        else:
            changed_array = np.asarray(changed, dtype=bool).reshape(-1)
        accepted = 0
        for anchor_index, (anchor, measured_range, is_new) in enumerate(
            zip(anchor_array, range_array, changed_array)
        ):
            if is_new and self.correct(
                anchor, float(measured_range), anchor_index=anchor_index
            ):
                accepted += 1
        return accepted


class SquaredRangeEKF(DirectRangeEKF):
    """Six-state EKF using the same debiased half-squared TWR observation.

    This diagnostic keeps only [p, v].  Its observation Jacobian depends on
    position, so it is not the proposed constant-H observer.  Comparing it
    with :class:`LiftedRangeKF` isolates the effect of squaring the range from
    the effect of treating q/s/h as independent states.
    """

    def correct(
        self,
        anchor: Iterable[float],
        measured_range: float,
        anchor_index: int | None = None,
    ) -> bool:
        """Correct six physical states using a debiased half squared range."""

        anchor_vector = _as_vector3(anchor, "anchor")
        try:
            range_bias = _for_anchor(self.range_bias, anchor_index, "range_bias")
            range_sigma = _for_anchor(
                self.range_sigma, anchor_index, "range_sigma"
            )
        except ValueError:
            self.stats.invalid += 1
            return False
        corrected_range = float(measured_range) - range_bias
        if not np.isfinite(corrected_range) or corrected_range <= 0:
            self.stats.invalid += 1
            return False

        displacement = self.x[:3] - anchor_vector
        row = np.zeros((1, 6), dtype=np.float64)
        row[0, :3] = displacement
        sigma2 = range_sigma**2
        measurement = 0.5 * (corrected_range**2 - sigma2)
        predicted = 0.5 * float(displacement @ displacement)
        innovation = measurement - predicted
        variance = corrected_range**2 * sigma2 + 0.5 * sigma2**2
        innovation_variance = float((row @ self.P @ row.T).item() + variance)
        if not np.isfinite(innovation_variance) or innovation_variance <= 0:
            self.stats.invalid += 1
            return False

        nis = innovation * innovation / innovation_variance
        self.last_nis = nis
        self.last_innovation = innovation
        gate_is_active = (
            self.stats.accepted + self.stats.rejected >= self.gate_warmup_updates
        )
        accepted, effective_variance, weight = _robust_measurement_variance(
            variance,
            nis,
            gate_is_active=gate_is_active,
            gate_threshold=self.gate_threshold,
            huber_threshold=self.huber_threshold,
        )
        self.stats.record(
            nis,
            accepted,
            weight=weight,
            anchor_index=anchor_index,
        )
        if not accepted:
            return False

        innovation_variance = float(
            (row @ self.P @ row.T).item() + effective_variance
        )
        gain = self.P @ row.T / innovation_variance
        residual_map = np.eye(6) - gain @ row
        self.x = self.x + gain[:, 0] * innovation
        self.P = _symmetrize(
            residual_map @ self.P @ residual_map.T
            + effective_variance * gain @ gain.T
        )
        return True


class TWRInEKF(InEKF):
    """Minimal single-range adaptation of the author's TDoA InEKF."""

    def __init__(
        self,
        X0: np.ndarray,
        P0: np.ndarray,
        bias: np.ndarray,
        *,
        range_sigma: float | Iterable[float] = 0.20,
        range_bias: float | Iterable[float] = -0.18,
        gate_threshold: float = 25.0,
        gate_warmup_updates: int = 0,
        huber_threshold: float | None = None,
        lever_arm: Iterable[float] = (0.0, 0.0, 0.0),
    ) -> None:
        """Initialize the joint InEKF and scalar or per-anchor range model."""

        super().__init__(X0, P0, bias)
        _validate_robust_settings(
            gate_threshold, gate_warmup_updates, huber_threshold
        )
        self.range_sigma = _anchor_parameter(
            range_sigma, "range_sigma", positive=True
        )
        self.range_bias = _anchor_parameter(range_bias, "range_bias")
        self.gate_threshold = float(gate_threshold)
        self.gate_warmup_updates = int(gate_warmup_updates)
        self.huber_threshold = (
            None if huber_threshold is None else float(huber_threshold)
        )
        self.t_uv = _as_vector3(lever_arm, "lever_arm")
        self.stats = UpdateStats()
        self.last_nis = math.nan

    def correct_twr(
        self,
        anchor: Iterable[float],
        measured_range: float,
        anchor_index: int | None = None,
    ) -> bool:
        """Correct the joint inertial state using one calibrated robust raw range."""

        anchor_vector = _as_vector3(anchor, "anchor")
        try:
            range_bias = _for_anchor(self.range_bias, anchor_index, "range_bias")
            range_sigma = _for_anchor(
                self.range_sigma, anchor_index, "range_sigma"
            )
        except ValueError:
            self.stats.invalid += 1
            return False
        corrected_range = float(measured_range) - range_bias
        if not np.isfinite(corrected_range) or corrected_range <= 0:
            self.stats.invalid += 1
            return False

        rotation = self.X_hat[:3, :3]
        position = self.X_hat[:3, 4]
        tag_position = position + rotation @ self.t_uv
        displacement = anchor_vector - tag_position
        predicted_range = float(np.linalg.norm(displacement))
        if predicted_range < 1e-12:
            self.stats.invalid += 1
            return False

        # The author's residual convention is z = y - y_hat while the
        # invariant error is estimate minus truth, hence this is the negative
        # physical range gradient.
        left_row = np.zeros((1, 15), dtype=np.float64)
        left_row[0, 6:9] = displacement / predicted_range

        adjoint = np.zeros((15, 15), dtype=np.float64)
        adjoint[:9, :9] = SE3_2_adjoint(SE3_2_inv(self.X_hat))
        adjoint[9:, 9:] = np.eye(6)
        row = left_row @ adjoint

        innovation = corrected_range - predicted_range
        variance = range_sigma**2
        innovation_variance = float((row @ self.P_hat @ row.T).item() + variance)
        nis = innovation * innovation / innovation_variance
        self.last_nis = nis
        gate_is_active = (
            self.stats.accepted + self.stats.rejected >= self.gate_warmup_updates
        )
        accepted, effective_variance, weight = _robust_measurement_variance(
            variance,
            nis,
            gate_is_active=gate_is_active,
            gate_threshold=self.gate_threshold,
            huber_threshold=self.huber_threshold,
        )
        self.stats.record(
            nis,
            accepted,
            weight=weight,
            anchor_index=anchor_index,
        )
        if not accepted:
            self.X = self.X_hat
            self.P = self.P_hat
            self.rej_cnt += 1
            return False

        innovation_variance = float(
            (row @ self.P_hat @ row.T).item() + effective_variance
        )
        gain = self.P_hat @ row.T / innovation_variance
        delta = gain[:, 0] * innovation
        residual_map = np.eye(15) - gain @ row
        self.P = _symmetrize(
            residual_map @ self.P_hat @ residual_map.T
            + effective_variance * gain @ gain.T
        )
        self.X = SE3_2_exp(-delta[:9]) @ self.X_hat
        self.bias = self.bias - delta[9:15]
        # Allow several anchor ranges at the same timestamp to update
        # sequentially rather than repeatedly starting from the same prior.
        self.X_hat = self.X.copy()
        self.P_hat = self.P.copy()
        return True


class LiftedTWRInEKF:
    """Staged IMU-InEKF -> world acceleration -> lifted-range observer.

    UWB updates affect the lifted position/velocity state but do not feed
    attitude or bias covariance back into the IMU core.  This is an explicit
    first-stage research architecture, not a unified group-affine filter.
    """

    def __init__(
        self,
        X0: np.ndarray,
        P0: np.ndarray,
        bias: np.ndarray,
        *,
        imu_core_factory: Callable[..., InEKF] | None = None,
        **lifted_kwargs,
    ) -> None:
        """Construct a lifted observer and a compatible injectable IMU core."""

        core_factory = InEKF if imu_core_factory is None else imu_core_factory
        self.imu_core = core_factory(X0.copy(), P0.copy(), bias.copy())
        self.lifted = LiftedRangeKF(
            X0[:3, 4], X0[:3, 3], **lifted_kwargs
        )
        self._synchronize_translation()

    def _synchronize_translation(self) -> None:
        """Synchronize translation through the core's coordinate-aware hook."""

        set_translation = getattr(self.imu_core, "set_translation", None)
        if set_translation is not None:
            set_translation(self.lifted.velocity, self.lifted.position)
        else:
            self.imu_core.X[:3, 3] = self.lifted.velocity
            self.imu_core.X[:3, 4] = self.lifted.position
            self.imu_core.X_hat = self.imu_core.X.copy()

    @property
    def X(self) -> np.ndarray:
        result = self.imu_core.X.copy()
        result[:3, 3] = self.lifted.velocity
        result[:3, 4] = self.lifted.position
        return result

    @property
    def bias(self) -> np.ndarray:
        return self.imu_core.bias.copy()

    def predict(self, imu_sample: Iterable[float], dt: float) -> np.ndarray:
        sample = np.asarray(imu_sample, dtype=np.float64).reshape(-1)
        if sample.shape != (6,):
            raise ValueError("imu_sample must be [acc_g, gyro_deg_s]")
        rotation = self.imu_core.X[:3, :3]
        body_acceleration = (
            sample[:3] * GRAVITY_MAGNITUDE - self.imu_core.bias[3:6]
        )
        world_acceleration = (
            rotation @ body_acceleration
            + np.array([0.0, 0.0, -GRAVITY_MAGNITUDE])
        )
        self.imu_core.predict(sample, dt)
        self.lifted.predict(world_acceleration, dt)
        self._synchronize_translation()
        return world_acceleration

    def correct_twr(
        self,
        anchor: Iterable[float],
        measured_range: float,
        anchor_index: int | None = None,
    ) -> bool:
        """Update lifted translation with the selected anchor calibration."""

        accepted = self.lifted.correct(
            anchor, measured_range, anchor_index=anchor_index
        )
        if accepted:
            self._synchronize_translation()
        return accepted


def paper_consistent_initial_covariance(
    *,
    attitude_std: Iterable[float] = (0.05, 0.05, 0.5),
    velocity_std: float = 2.0,
    position_std: float = 8.0,
    gyro_bias_std: float = 0.01,
    accel_bias_std: float = 0.10,
) -> np.ndarray:
    """Create P in paper order [theta, v, p, b_g, b_a]."""

    attitude = _as_vector3(attitude_std, "attitude_std")
    diagonal = np.concatenate(
        (
            attitude**2,
            np.full(3, velocity_std**2),
            np.full(3, position_std**2),
            np.full(3, gyro_bias_std**2),
            np.full(3, accel_bias_std**2),
        )
    )
    return np.diag(diagonal)


def imu_world_acceleration(
    rotation: np.ndarray, imu_sample: Iterable[float], bias: Iterable[float]
) -> np.ndarray:
    """Paper-consistent helper used by tests and experiment tooling."""

    sample = np.asarray(imu_sample, dtype=np.float64).reshape(-1)
    bias_vector = np.asarray(bias, dtype=np.float64).reshape(-1)
    if sample.shape != (6,) or bias_vector.shape != (6,):
        raise ValueError("imu_sample and bias must both contain six values")
    body_acceleration = (
        sample[:3] * GRAVITY_MAGNITUDE - bias_vector[3:6]
    )
    return (
        np.asarray(rotation, dtype=np.float64) @ body_acceleration
        + np.array([0.0, 0.0, -GRAVITY_MAGNITUDE])
    )
