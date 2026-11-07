"""Single-covariance body-frame lifted TWR filter (CLIF).

One error covariance carries the 15-dim invariant error ``[theta, v, p, b_g,
b_a]`` together with the quadratic lifted states, so a TWR update can correct
attitude, position and both biases through the cross-covariance instead of only
the translation block.  The lifted block is propagated with the exact nilpotent
series from :mod:`clif_core`, never with forward Euler.

Error convention: the truth state is ``X_true = Exp(xi) X_hat``.  A measurement
with innovation ``y - y_hat`` therefore updates ``X <- Exp(+K (y - y_hat)) X``
and ``b <- b + K_b (y - y_hat)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import linalg

from inekf import SE3_2_adjoint, SE3_2_exp

import clif_core as core


@dataclass
class ClifParameters:
    """Noise levels and gates of the CLIF filter."""

    gyro_noise_std: float = 0.1  # rad/s, white noise on the rate measurement
    accel_noise_std: float = 2.0  # m/s^2, white noise on the specific force
    gyro_bias_walk_std: float = 3e-4  # rad/s/sqrt(s)
    accel_bias_walk_std: float = 3e-2  # m/s^2/sqrt(s)
    quadratic_walk_std: float = 1.0  # m^2/sqrt(s), lifted process noise
    range_sigma: float = 0.20  # m, TWR noise
    range_bias: float = -0.18  # m, TWR bias
    gate_threshold: float = 25.0  # NIS gate
    gate_warmup_updates: int = 64  # updates skipped by the gate
    # The gate stays open while the position閺嶅洤鍣?is above this level, so a
    # large initial offset is not rejected during the capture transient.
    gate_position_std_m: float = 1.0
    # Innovations beyond this half-squared-range scale (m^2) are down-weighted.
    # During capture the redundancy Jacobian is evaluated at a badly wrong
    # estimate, so trusting it fully collapses the covariance and the filter
    # stops listening; inflating the variance keeps the gain honest.
    innovation_scale_m2: float = 20.0
    max_quadratic_step: float = 5.0  # m, guard against absurd innovations
    # Rate at which the unobserved auxiliary quadratics are pulled back to the
    # value implied by the nominal state.  Without it their variance grows like
    # t^8 because the chain is a fourth-order integrator driven by s22.
    auxiliary_relaxation_rate: float = 20.0  # 1/s


@dataclass
class ClifStats:
    """Update counters used by the experiment harness."""

    accepted: int = 0
    rejected: int = 0
    invalid: int = 0
    last_nis: float = math.nan


@dataclass
class ClifTWRInEKF:
    """Joint invariant/quadratic TWR filter with a single covariance."""

    X0: np.ndarray
    P0: np.ndarray
    bias: np.ndarray
    anchors: np.ndarray
    parameters: ClifParameters = field(default_factory=ClifParameters)
    stats: ClifStats = field(default_factory=ClifStats)

    def __post_init__(self) -> None:
        self.X = np.array(self.X0, dtype=np.float64, copy=True)
        self.bias = np.array(self.bias, dtype=np.float64, copy=True).reshape(6)
        self.anchors = np.array(self.anchors, dtype=np.float64).reshape(-1, 3)
        self.anchor_count = self.anchors.shape[0]
        # Layout: [xi (15), s00_1..A, s01_1..A, s02_1..A, s11, s12, s22].
        self.state_dim = core.ERROR_DIM + 3 * self.anchor_count + 3
        self.s00_offset = core.ERROR_DIM
        self.s01_offset = self.s00_offset + self.anchor_count
        self.s02_offset = self.s01_offset + self.anchor_count
        self.s11_index = self.s02_offset + self.anchor_count
        self.s12_index = self.s11_index + 1
        self.s22_index = self.s11_index + 2

        quantities = core.lifted_quantities(self.X, np.zeros(3), self.anchors)
        self.quadratic = core.quadratic_from_quantities(quantities)
        self.P = np.zeros((self.state_dim, self.state_dim))
        self.P[: core.ERROR_DIM, : core.ERROR_DIM] = np.array(self.P0, dtype=np.float64)
        self._initialise_quadratic_covariance(quantities)

    # ---------------------------------------------------------------- helpers

    def quadratic_vector(self) -> np.ndarray:
        """Return the quadratic block in the state layout."""
        return np.concatenate(
            [
                self.quadratic["s00"],
                self.quadratic["s01"],
                self.quadratic["s02"],
                [self.quadratic["s11"], self.quadratic["s12"], self.quadratic["s22"]],
            ]
        )

    def _set_quadratic_vector(self, values: np.ndarray) -> None:
        """Write the quadratic block back into the quadratic dictionary."""
        count = self.anchor_count
        self.quadratic["s00"] = np.array(values[:count], copy=True)
        self.quadratic["s01"] = np.array(values[count : 2 * count], copy=True)
        self.quadratic["s02"] = np.array(values[2 * count : 3 * count], copy=True)
        self.quadratic["s11"] = float(values[3 * count])
        self.quadratic["s12"] = float(values[3 * count + 1])
        self.quadratic["s22"] = float(values[3 * count + 2])

    def _initialise_quadratic_covariance(self, quantities: dict) -> None:
        """Map the state uncertainty onto the quadratic block at start-up.

        The quadratic states are pure functions of the nominal state, so the
        Jacobian is taken by central differences of that map.  Doing it
        numerically removes a whole class of hand-derivation sign errors.
        """
        def quadratic_vector_at(state: np.ndarray, bias_value: np.ndarray) -> np.ndarray:
            quantities_here = core.lifted_quantities(state, -bias_value[3:6], self.anchors)
            block = core.quadratic_from_quantities(quantities_here)
            return np.concatenate(
                [
                    block["s00"],
                    block["s01"],
                    block["s02"],
                    [block["s11"], block["s12"], block["s22"]],
                ]
            )

        base = quadratic_vector_at(self.X, self.bias)
        jacobian = np.zeros((3 * self.anchor_count + 3, core.ERROR_DIM))
        epsilon = 1e-6
        for index in range(core.ERROR_DIM):
            step = np.zeros(core.ERROR_DIM)
            step[index] = epsilon
            plus = quadratic_vector_at(SE3_2_exp(step[:9]) @ self.X, self.bias + step[9:15])
            minus = quadratic_vector_at(SE3_2_exp(-step[:9]) @ self.X, self.bias - step[9:15])
            jacobian[:, index] = (plus - minus) / (2.0 * epsilon)

        state_block = self.P[: core.ERROR_DIM, : core.ERROR_DIM]
        lifted = jacobian @ state_block
        self.P[core.ERROR_DIM :, : core.ERROR_DIM] = lifted
        self.P[: core.ERROR_DIM, core.ERROR_DIM :] = lifted.T
        quadratic_covariance = jacobian @ state_block @ jacobian.T
        floor = 1e-6 * np.eye(3 * self.anchor_count + 3)
        self.P[core.ERROR_DIM :, core.ERROR_DIM :] = quadratic_covariance + floor
        del base

    def _quadratic_jacobian(self) -> np.ndarray:
        """Jacobian of the quadratic block with respect to the invariant error."""

        def quadratic_vector_at(state: np.ndarray, bias_value: np.ndarray) -> np.ndarray:
            quantities_here = core.lifted_quantities(state, -bias_value[3:6], self.anchors)
            block = core.quadratic_from_quantities(quantities_here)
            return np.concatenate(
                [
                    block["s00"],
                    block["s01"],
                    block["s02"],
                    [block["s11"], block["s12"], block["s22"]],
                ]
            )

        jacobian = np.zeros((3 * self.anchor_count + 3, core.ERROR_DIM))
        epsilon = 1e-6
        for index in range(core.ERROR_DIM):
            step = np.zeros(core.ERROR_DIM)
            step[index] = epsilon
            plus = quadratic_vector_at(SE3_2_exp(step[:9]) @ self.X, self.bias + step[9:15])
            minus = quadratic_vector_at(SE3_2_exp(-step[:9]) @ self.X, self.bias - step[9:15])
            jacobian[:, index] = (plus - minus) / (2.0 * epsilon)
        return jacobian

    def _enforce_quadratic_constraint(self, anchor_index: int) -> None:
        """Drive s00 back onto 0.5 |a_i - p|^2 by moving the state, not only P.

        The range update corrects s00 almost completely while p barely moves, so
        the two disagree afterwards.  Writing the redundancy into the covariance
        alone then makes the filter believe p is as well known as s00.  This
        scalar pseudo-update transfers that disagreement into the state.
        """
        jacobian = self._quadratic_jacobian()
        displacement = self.X[:3, 4] - self.anchors[anchor_index]
        implied = 0.5 * float(displacement @ displacement)
        residual = float(self.quadratic["s00"][anchor_index]) - implied

        row = np.zeros(self.state_dim)
        row[: core.ERROR_DIM] = -jacobian[anchor_index]
        row[self.s00_offset + anchor_index] += 1.0
        noise = 0.05
        variance = float(row @ self.P @ row) + noise
        gain = self.P @ row / variance
        # The pseudo-measurement is "0 = s00 - implied", so innovation = -residual.
        self._apply_correction(-gain * residual)
        residual_map = np.eye(self.state_dim) - np.outer(gain, row)
        self.P = residual_map @ self.P @ residual_map.T + noise * np.outer(gain, gain)
        self.P = 0.5 * (self.P + self.P.T)

    def _reset_redundant_covariance(self) -> None:
        """Re-inject the redundancy relation after every measurement.

        The measured quadratics are functions of the nominal state.  Between two
        ranges of the same anchor the cross-covariance to the state decays by
        dt^2 per step, so without this reset the next range carries almost no
        gain and the filter stops listening to position.
        """
        jacobian = self._quadratic_jacobian()
        state_block = self.P[: core.ERROR_DIM, : core.ERROR_DIM]
        cross = jacobian @ state_block
        self.P[core.ERROR_DIM :, : core.ERROR_DIM] = cross
        self.P[: core.ERROR_DIM, core.ERROR_DIM :] = cross.T
        floor = 1e-6 * np.eye(3 * self.anchor_count + 3)
        self.P[core.ERROR_DIM :, core.ERROR_DIM :] = jacobian @ state_block @ jacobian.T + floor
        self.P = 0.5 * (self.P + self.P.T)

    def invariant_transition(self, dt: float) -> np.ndarray:
        """15x15 error transition of the right-invariant IMU propagation."""
        rotation = self.X[:3, :3]
        gravity_cross = core.skew(core.GRAVITY_WORLD)
        adjoint = SE3_2_adjoint(self.X)
        zeros = np.zeros((3, 3))
        identity = np.eye(3)
        system = np.block(
            [
                [zeros, zeros, zeros, -rotation, zeros],
                [gravity_cross, zeros, zeros, -adjoint[3:6, 0:3], -rotation],
                [zeros, identity, zeros, -adjoint[6:9, 0:3], zeros],
                [zeros, zeros, zeros, zeros, zeros],
                [zeros, zeros, zeros, zeros, zeros],
            ]
        )
        return linalg.expm(system * dt)

    def lifted_transition(self, dt: float) -> np.ndarray:
        """Transition of the quadratic block, including the state coupling."""
        count = self.anchor_count
        block = np.zeros((3 * count + 3, 3 * count + 3))
        chain = core.nilpotent_transition(dt)
        for index in range(count):
            for row in range(3):
                block[index + row * count, index + row * count] = chain[row, row]
            # Coupling to the three global states through the chain rows.
            for row in range(3):
                for column in range(3):
                    block[index + row * count, 3 * count + column] = chain[row, 3 + column]
            for other in range(3):
                for row in range(3):
                    block[index + row * count, other * count + index] = chain[row, other]
        block[3 * count :, 3 * count :] = chain[3:, 3:]
        return block

    def lifted_state_coupling(self, dt: float, quantities: dict, rotation: np.ndarray) -> np.ndarray:
        """d(quadratic_{k+1}) / d(xi_k), shape (3A+3, 15)."""
        count = self.anchor_count
        input_map = core.nilpotent_input_map(dt)
        driving = core.anchor_driving_jacobian(quantities, rotation, self.anchors)
        coupling = np.zeros((3 * count + 3, core.ERROR_DIM))
        row_s01 = core.CHAIN_INDEX["s01"]
        row_s11 = core.CHAIN_INDEX["s11"]
        row_s12 = core.CHAIN_INDEX["s12"]
        for index in range(count):
            per_anchor = (
                np.outer(input_map[0:3, row_s01], driving[index, row_s01])
                + np.outer(input_map[0:3, row_s11], driving[index, row_s11])
                + np.outer(input_map[0:3, row_s12], driving[index, row_s12])
            )
            coupling[index : 3 * count : count] = per_anchor
        global_rows = np.outer(input_map[3:6, row_s11], driving[0, row_s11]) + np.outer(
            input_map[3:6, row_s12], driving[0, row_s12]
        )
        coupling[3 * count :] = global_rows
        return coupling

    def predict(self, gyro, accel, dt: float) -> None:
        """Propagate the nominal state and the joint covariance by one step."""
        rate = np.asarray(gyro, dtype=float).reshape(3) - self.bias[0:3]
        specific_force = np.asarray(accel, dtype=float).reshape(3) - self.bias[3:6]
        rotation = self.X[:3, :3]
        velocity = self.X[:3, 3]
        position = self.X[:3, 4]

        acceleration = rotation @ specific_force + core.GRAVITY_WORLD
        new_rotation = rotation @ core.so3_exp(rate * dt)
        new_velocity = velocity + acceleration * dt
        new_position = position + velocity * dt + 0.5 * acceleration * dt**2

        predicted = np.array(self.X, copy=True)
        predicted[:3, :3] = new_rotation
        predicted[:3, 3] = new_velocity
        predicted[:3, 4] = new_position

        # Trapezoidal inputs: averaging the start and end lifted quantities
        # makes the frozen-input step second order, which keeps the manifold
        # residual from growing like n^2 dt^2.
        start_quantities = core.lifted_quantities(self.X, specific_force, self.anchors)
        end_quantities = core.lifted_quantities(predicted, specific_force, self.anchors)
        quantities = {
            key: 0.5 * (start_quantities[key] + end_quantities[key])
            for key in ("z_anchor", "z_velocity", "z_gravity")
        }
        quantities["z_force"] = specific_force

        self.quadratic = core.exact_lifted_step(self.quadratic, quantities, dt)
        self._relax_auxiliary_quadratic(predicted, dt)

        transition = np.eye(self.state_dim)
        transition[: core.ERROR_DIM, : core.ERROR_DIM] = self.invariant_transition(dt)
        transition[core.ERROR_DIM :, core.ERROR_DIM :] = self.lifted_transition(dt)
        transition[core.ERROR_DIM :, : core.ERROR_DIM] = self.lifted_state_coupling(dt, quantities, rotation)

        self.X = predicted

        self.P = transition @ self.P @ transition.T + self.process_noise(dt, quantities)

    def _relax_auxiliary_quadratic(self, state: np.ndarray, dt: float) -> None:
        """Pull the unobserved quadratics back to the state-implied values.

        s00 stays free because it is the measured quantity.  The other five are
        only integrated, so leaving them free makes their variance grow like
        t^8 and the gain on the state explodes.  The same weight is applied to
        their covariance through a diagonal congruence, which preserves PSD.
        """
        rate = self.parameters.auxiliary_relaxation_rate
        if rate <= 0.0:
            return
        weight = math.exp(-rate * dt)
        implied = core.quadratic_from_quantities(core.lifted_quantities(state, np.zeros(3), self.anchors))
        for name in ("s01", "s02"):
            self.quadratic[name] = implied[name] + weight * (self.quadratic[name] - implied[name])
        for name in ("s11", "s12", "s22"):
            self.quadratic[name] = float(implied[name]) + weight * (self.quadratic[name] - float(implied[name]))

        leak = np.ones(self.state_dim)
        leak[self.s00_offset : self.s01_offset] = 1.0
        leak[self.s01_offset :] = weight
        self.P = self.P * np.outer(leak, leak)

    def process_noise(self, dt: float, quantities: dict) -> np.ndarray:
        """Diagonal process noise with a physically scaled lifted block.

        The quadratic states are driven by the specific force, so their noise
        scales with |z|, the accelerometer noise and the integration order.
        A constant walk level either hides the real injection or over-trusts
        the lifting; this version follows the same chain the states do.
        """
        noise = np.zeros(self.state_dim)
        sigma = self.parameters.accel_noise_std
        noise[0:3] = (self.parameters.gyro_noise_std * dt) ** 2
        noise[3:6] = (sigma * dt) ** 2
        noise[6:9] = (0.5 * sigma * dt**2) ** 2
        noise[9:12] = (self.parameters.gyro_bias_walk_std * dt) ** 2
        noise[12:15] = (self.parameters.accel_bias_walk_std * dt) ** 2

        z_anchor = quantities["z_anchor"]
        anchor_norm = np.linalg.norm(z_anchor, axis=1)
        velocity_norm = float(np.linalg.norm(quantities["z_velocity"]))
        gravity_norm = float(np.linalg.norm(quantities["z_gravity"]))
        floor = (self.parameters.quadratic_walk_std * dt) ** 2
        count = self.anchor_count
        noise[self.s00_offset : self.s01_offset] = np.maximum(
            (0.5 * anchor_norm * sigma * dt * dt) ** 2, floor
        )
        noise[self.s01_offset : self.s02_offset] = np.maximum(
            (0.5 * anchor_norm * sigma * dt) ** 2, floor
        )
        noise[self.s02_offset : self.s02_offset + count] = floor
        noise[self.s11_index] = max((velocity_norm * sigma * dt) ** 2, floor)
        noise[self.s12_index] = max((0.5 * gravity_norm * sigma * dt) ** 2, floor)
        noise[self.s22_index] = floor
        return np.diag(noise)

    def correct_twr(self, anchor_index: int, measured_range: float) -> bool:
        """Update with one de-biased half-squared TWR measurement."""
        corrected = float(measured_range) - self.parameters.range_bias
        if not np.isfinite(corrected) or corrected <= 0.0:
            self.stats.invalid += 1
            return False

        index = self.s00_offset + int(anchor_index)
        row = np.zeros(self.state_dim)
        row[index] = 1.0
        sigma2 = self.parameters.range_sigma**2
        measurement = 0.5 * (corrected**2 - sigma2)
        variance = corrected**2 * sigma2 + 0.5 * sigma2**2
        predicted = float(self.quadratic["s00"][anchor_index])
        innovation = measurement - predicted

        # Robust inflation: a large half-squared-range innovation means the
        # linearisation point is unreliable, so the measurement is trusted less.
        scale = max(self.parameters.innovation_scale_m2, 1e-9)
        inflation = 1.0 + (innovation / scale) ** 2

        innovation_variance = (float(row @ self.P @ row.T) + variance) * inflation
        nis = innovation * innovation / innovation_variance
        self.stats.last_nis = nis
        position_std = math.sqrt(max(float(np.trace(self.P[6:9, 6:9])) / 3.0, 0.0))
        gate_active = (
            self.stats.accepted + self.stats.rejected >= self.parameters.gate_warmup_updates
            and position_std < self.parameters.gate_position_std_m
        )
        if gate_active and nis > self.parameters.gate_threshold:
            self.stats.rejected += 1
            return False

        gain = self.P @ row / innovation_variance
        # Posterior error mean is +K (y - h(x_hat)); the state parameterisation
        # is X_true = Exp(xi) X_hat, so the correction is Exp(+K innovation).
        state_correction = gain * innovation
        self._apply_correction(state_correction)

        residual_map = np.eye(self.state_dim) - np.outer(gain, row)
        self.P = residual_map @ self.P @ residual_map.T + variance * np.outer(gain, gain)
        self.P = 0.5 * (self.P + self.P.T)
        self._repair_covariance()
        self._reset_redundant_covariance()
        self.stats.accepted += 1
        return True

    def _repair_covariance(self) -> None:
        """Keep the joint covariance positive definite after a rank-one update.

        The lifted block is redundant with the nominal state, so a Joseph update
        can leave it numerically indefinite; without this repair the nilpotent
        chain amplifies the negative directions on the next prediction.
        """
        if not np.all(np.isfinite(self.P)):
            raise FloatingPointError("covariance became non-finite")
        smallest = float(np.min(np.linalg.eigvalsh(self.P)))
        if smallest < 1e-9:
            self.P = self.P + (1e-9 - smallest) * np.eye(self.state_dim)

    def _apply_correction(self, correction: np.ndarray) -> None:
        """Inject the error correction back into the nominal state."""
        xi = correction[: core.ERROR_DIM]
        if not np.all(np.isfinite(xi)):
            return
        self.X = SE3_2_exp(xi) @ self.X
        self.bias = self.bias + correction[9:15]
        quadratic = self.quadratic_vector() + correction[core.ERROR_DIM :]
        self._set_quadratic_vector(quadratic)

    # -------------------------------------------------------------- accessors

    @property
    def position(self) -> np.ndarray:
        """Current world-frame position estimate."""
        return self.X[:3, 4].copy()

    @property
    def velocity(self) -> np.ndarray:
        """Current world-frame velocity estimate."""
        return self.X[:3, 3].copy()

    @property
    def rotation(self) -> np.ndarray:
        """Current attitude estimate."""
        return self.X[:3, :3].copy()

    def manifold_residual(self) -> np.ndarray:
        """Per-anchor residual s00 - 0.5 z_i^T z_i in m^2."""
        z_anchor = core.lifted_quantities(self.X, np.zeros(3), self.anchors)["z_anchor"]
        return self.quadratic["s00"] - 0.5 * np.sum(z_anchor**2, axis=1)


def skew_helper(vector: np.ndarray) -> np.ndarray:
    """Small alias so the covariance bootstrap reads symmetrically."""
    return core.skew(vector)
