"""Invariant EKF primitives and a verified IMU--UWB estimator.

The public implementation mixed the nominal bias order ``[b_a, b_g]``
with the error-state order ``[b_g, b_a]`` used by its propagation matrix.
This module uses the paper's order everywhere for the verified filter:

    [theta, velocity, position, gyro bias, accelerometer bias].

``FilterConfig.legacy()`` intentionally retains the public implementation's
behaviour so experiments can perform a reproducible ablation.
"""

from dataclasses import dataclass
import math

import numpy as np
from scipy import linalg
from scipy.spatial.transform import Rotation


GRAVITY_MAGNITUDE = 9.81
DEG_TO_RAD = math.pi / 180.0

THETA = slice(0, 3)
VEL = slice(3, 6)
POS = slice(6, 9)
BG = slice(9, 12)
BA = slice(12, 15)


def SO3_wedge(phi):
    phi = np.asarray(phi, dtype=float).reshape(3)
    return np.array(
        [[0.0, -phi[2], phi[1]],
         [phi[2], 0.0, -phi[0]],
         [-phi[1], phi[0], 0.0]]
    )


def SO3_vee(Phi):
    Phi = np.asarray(Phi, dtype=float)
    return np.array([Phi[2, 1], Phi[0, 2], Phi[1, 0]])


def SO3_exp(phi):
    """Numerically stable exponential map from so(3) to SO(3)."""
    phi = np.asarray(phi, dtype=float).reshape(3)
    angle = np.linalg.norm(phi)
    Phi = SO3_wedge(phi)
    if angle < 1e-8:
        angle2 = angle * angle
        a = 1.0 - angle2 / 6.0 + angle2 * angle2 / 120.0
        b = 0.5 - angle2 / 24.0 + angle2 * angle2 / 720.0
    else:
        a = math.sin(angle) / angle
        b = (1.0 - math.cos(angle)) / (angle * angle)
    return np.eye(3) + a * Phi + b * (Phi @ Phi)


def SO3_log(Phi):
    """Stable principal logarithm with a fast path away from pi."""
    Phi = np.asarray(Phi, dtype=float).reshape(3, 3)
    cosine = float(np.clip(0.5 * np.trace(Phi) - 0.5, -1.0, 1.0))
    angle = math.acos(cosine)
    antisymmetric = SO3_vee(Phi - Phi.T)
    if angle < 1e-7:
        # angle/(2 sin(angle)) = 1/2 + angle^2/12 + O(angle^4)
        scale = 0.5 + angle * angle / 12.0
        return scale * antisymmetric
    if math.pi - angle < 1e-5:
        # The antisymmetric part vanishes at pi; SciPy's quaternion-based
        # branch is robust there and is rare in normal estimator logging.
        return Rotation.from_matrix(Phi).as_rotvec()
    return (angle / (2.0 * math.sin(angle))) * antisymmetric


def SO3_left_jacobian(phi):
    """Left Jacobian of SO(3), with a continuous small-angle series."""
    phi = np.asarray(phi, dtype=float).reshape(3)
    angle = np.linalg.norm(phi)
    Phi = SO3_wedge(phi)
    if angle < 1e-6:
        angle2 = angle * angle
        a = 0.5 - angle2 / 24.0 + angle2 * angle2 / 720.0
        b = 1.0 / 6.0 - angle2 / 120.0 + angle2 * angle2 / 5040.0
    else:
        a = (1.0 - math.cos(angle)) / (angle * angle)
        b = (angle - math.sin(angle)) / (angle**3)
    return np.eye(3) + a * Phi + b * (Phi @ Phi)


def SO3_gamma2(phi):
    """Second integral of ``Exp(phi)`` for constant-rate IMU propagation.

    This is

    ``Gamma2(phi) = integral_0^1 (1-s) Exp(s phi^) ds``.
    """
    phi = np.asarray(phi, dtype=float).reshape(3)
    angle = np.linalg.norm(phi)
    Phi = SO3_wedge(phi)
    angle2 = angle * angle
    if angle < 1e-4:
        c1 = 1.0 / 6.0 - angle2 / 120.0 + angle2 * angle2 / 5040.0
        c2 = 1.0 / 24.0 - angle2 / 720.0 + angle2 * angle2 / 40320.0
    else:
        c1 = (angle - math.sin(angle)) / (angle**3)
        c2 = (angle2 + 2.0 * math.cos(angle) - 2.0) / (2.0 * angle**4)
    return 0.5 * np.eye(3) + c1 * Phi + c2 * (Phi @ Phi)


def SO3_motion_terms(phi):
    """Return ``Exp(phi)``, ``Gamma1(phi)`` and ``Gamma2(phi)`` together."""
    phi = np.asarray(phi, dtype=float).reshape(3)
    angle = np.linalg.norm(phi)
    angle2 = angle * angle
    Phi = SO3_wedge(phi)
    Phi2 = Phi @ Phi
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
        gamma1_2 = (angle - sine) / (angle**3)
        gamma2_2 = (angle2 + 2.0 * cosine - 2.0) / (2.0 * angle**4)
    identity = np.eye(3)
    rotation = identity + exp_1 * Phi + exp_2 * Phi2
    gamma1 = identity + exp_2 * Phi + gamma1_2 * Phi2
    gamma2 = 0.5 * identity + gamma1_2 * Phi + gamma2_2 * Phi2
    return rotation, gamma1, gamma2


def SO3_inv_left_jacobian(phi):
    """Inverse left Jacobian of SO(3)."""
    phi = np.asarray(phi, dtype=float).reshape(3)
    angle = np.linalg.norm(phi)
    Phi = SO3_wedge(phi)
    if angle < 1e-6:
        angle2 = angle * angle
        c = 1.0 / 12.0 + angle2 / 720.0 + angle2 * angle2 / 30240.0
    else:
        c = (1.0 - 0.5 * angle / math.tan(0.5 * angle)) / (angle * angle)
    return np.eye(3) - 0.5 * Phi + c * (Phi @ Phi)


def SE3_2_wedge(xi):
    """SE_2(3) wedge operation."""
    xi = np.asarray(xi, dtype=float).reshape(9)
    Xi = np.zeros((5, 5))
    Xi[:3, :3] = SO3_wedge(xi[THETA])
    Xi[:3, 3] = xi[VEL]
    Xi[:3, 4] = xi[POS]
    return Xi


def SE3_2_vee(Xi):
    return np.concatenate((SO3_vee(Xi[:3, :3]), Xi[:3, 3], Xi[:3, 4]))


def SE3_2_vec(Xi):
    return np.concatenate((SO3_log(Xi[:3, :3]), Xi[:3, 3], Xi[:3, 4]))


def SE3_2_exp(xi):
    xi = np.asarray(xi, dtype=float).reshape(9)
    T = np.eye(5)
    T[:3, :3] = SO3_exp(xi[THETA])
    T[:3, 3] = SO3_left_jacobian(xi[THETA]) @ xi[VEL]
    T[:3, 4] = SO3_left_jacobian(xi[THETA]) @ xi[POS]
    return T


def SE3_2_inv(Xi):
    T = np.eye(5)
    T[:3, :3] = Xi[:3, :3].T
    T[:3, 3] = -T[:3, :3] @ Xi[:3, 3]
    T[:3, 4] = -T[:3, :3] @ Xi[:3, 4]
    return T


def SE3_2_log(T):
    phi = SO3_log(T[:3, :3])
    translated = SO3_inv_left_jacobian(phi) @ T[:3, 3:5]
    return np.concatenate((phi, translated[:, 0], translated[:, 1]))


def SE3_2_adjoint(T):
    Adjoint = np.zeros((9, 9))
    R = T[:3, :3]
    Adjoint[THETA, THETA] = R
    Adjoint[VEL, VEL] = R
    Adjoint[POS, POS] = R
    Adjoint[VEL, THETA] = SO3_wedge(T[:3, 3]) @ R
    Adjoint[POS, THETA] = SO3_wedge(T[:3, 4]) @ R
    return Adjoint


def SE3_2_ad(xi):
    """Lie-algebra adjoint satisfying ``ad_xi eta = [xi, eta]``."""
    xi = np.asarray(xi, dtype=float).reshape(9)
    algebra_adjoint = np.zeros((9, 9))
    algebra_adjoint[THETA, THETA] = SO3_wedge(xi[THETA])
    algebra_adjoint[VEL, THETA] = SO3_wedge(xi[VEL])
    algebra_adjoint[VEL, VEL] = SO3_wedge(xi[THETA])
    algebra_adjoint[POS, THETA] = SO3_wedge(xi[POS])
    algebra_adjoint[POS, POS] = SO3_wedge(xi[THETA])
    return algebra_adjoint


def SE3_2_right_jacobian(xi, terms=18):
    """Right Jacobian computed from its stable matrix-power series."""
    ad_xi = SE3_2_ad(xi)
    jacobian = np.eye(9)
    power = np.eye(9)
    factorial = 1.0
    for order in range(1, terms):
        power = power @ ad_xi
        factorial *= order + 1
        jacobian += ((-1.0) ** order / factorial) * power
    return jacobian


def project_to_so3(R):
    """Remove round-off drift while preserving a proper rotation."""
    u, _, vh = np.linalg.svd(R)
    projected = u @ vh
    if np.linalg.det(projected) < 0.0:
        u[:, -1] *= -1.0
        projected = u @ vh
    return projected


@dataclass(frozen=True)
class FilterConfig:
    """Ablation switches for the public and verified estimators."""

    name: str = "verified"
    midpoint_integration: bool = True
    van_loan_discretization: bool = True
    trapezoidal_discretization: bool = False
    direct_lever_arm_jacobian: bool = True
    consistent_bias_order: bool = True
    ground_clamp: bool = False
    covariance_reset: bool = True
    static_bias_initialization: bool = True
    stationary_inertial_update: bool = False
    coordinate_consistent_covariance: bool = True
    map_physical_initial_covariance: bool = True
    initial_yaw_std: float = 1.0
    initial_position_std: float = 0.1
    structured_transition: bool = False
    closed_form_imu_integration: bool = False
    rank_one_scalar_update: bool = False
    direct_transformed_jacobian: bool = False
    project_rotation_each_step: bool = True

    @classmethod
    def verified(cls):
        return cls()

    @classmethod
    def legacy(cls):
        return cls(
            name="legacy",
            midpoint_integration=False,
            van_loan_discretization=False,
            trapezoidal_discretization=False,
            direct_lever_arm_jacobian=False,
            consistent_bias_order=False,
            ground_clamp=True,
            covariance_reset=False,
            static_bias_initialization=False,
            stationary_inertial_update=False,
            coordinate_consistent_covariance=False,
            map_physical_initial_covariance=False,
            structured_transition=False,
            closed_form_imu_integration=False,
            rank_one_scalar_update=False,
            direct_transformed_jacobian=False,
            project_rotation_each_step=True,
        )

    @classmethod
    def proposed(cls):
        """Low-overhead algorithmic improvement used for task01.

        The paper-consistent bias order and stable Lie operations are retained.
        Expensive strict event splitting, midpoint propagation, Van Loan process
        noise and the still research-only direct TDoA Jacobian stay outside the
        production path.  A stationary zero-angular-rate measurement adds
        observable IMU bias information without injecting false yaw information.
        """
        return cls(
            name="proposed",
            midpoint_integration=False,
            van_loan_discretization=False,
            trapezoidal_discretization=False,
            direct_lever_arm_jacobian=False,
            consistent_bias_order=True,
            ground_clamp=False,
            covariance_reset=False,
            static_bias_initialization=True,
            stationary_inertial_update=True,
            coordinate_consistent_covariance=True,
            map_physical_initial_covariance=True,
            structured_transition=True,
            closed_form_imu_integration=True,
            rank_one_scalar_update=True,
            direct_transformed_jacobian=True,
            project_rotation_each_step=False,
            # For an unknown heading over the full circle, sigma=1 rad would
            # make a pi-radian error a 3.14-sigma event.  Two radians is close to
            # pi/sqrt(3), the standard deviation of a uniform angular prior.
            initial_yaw_std=2.0,
        )

    @classmethod
    def task03_compiled(cls):
        """Production equations executed by the fixed-size Numba replay core."""
        proposed = cls.proposed()
        return cls(**{**proposed.__dict__, "name": "task03_compiled"})

    @classmethod
    def pre_optimization(cls):
        """Coordinate-correct task01 path before the task02 compute kernels."""
        proposed = cls.proposed()
        return cls(
            **{
                **proposed.__dict__,
                "name": "pre_optimization",
                "structured_transition": False,
                "closed_form_imu_integration": False,
                "rank_one_scalar_update": False,
                "direct_transformed_jacobian": False,
                "project_rotation_each_step": True,
            }
        )

    @classmethod
    def pre_coordinate_fix(cls):
        """Task01 proposed path before invariant covariance/gating repair."""
        proposed = cls.proposed()
        return cls(
            **{
                **proposed.__dict__,
                "name": "pre_coordinate_fix",
                "coordinate_consistent_covariance": False,
                "map_physical_initial_covariance": False,
                "initial_yaw_std": 1.0,
            }
        )

    @classmethod
    def aligned_legacy(cls):
        """Only replace the public timestamp scheduler (diagnostic ablation)."""
        legacy = cls.legacy()
        return cls(**{**legacy.__dict__, "name": "aligned_legacy"})

    @classmethod
    def proposed_euler(cls):
        proposed = cls.proposed()
        return cls(
            **{
                **proposed.__dict__,
                "name": "proposed_euler",
                "midpoint_integration": False,
                "closed_form_imu_integration": False,
            }
        )

    @classmethod
    def proposed_no_bias_init(cls):
        proposed = cls.proposed()
        return cls(
            **{
                **proposed.__dict__,
                "name": "proposed_no_bias_init",
                "static_bias_initialization": False,
            }
        )

    @classmethod
    def proposed_no_stationary(cls):
        proposed = cls.proposed()
        return cls(
            **{
                **proposed.__dict__,
                "name": "proposed_no_stationary",
                "stationary_inertial_update": False,
            }
        )


class InEKF:
    """World-centric right-invariant EKF for IMU and UWB TDoA."""

    def __init__(self, X0, P0, bias, config=None):
        self.config = config or FilterConfig.verified()
        self.X = SE3_2_exp(X0) if np.shape(X0) == (9,) else np.array(X0, dtype=float)
        self.X_hat = self.X.copy()
        self.P = np.array(P0, dtype=float)
        self.P_hat = self.P.copy()
        self.bias = np.array(bias, dtype=float).reshape(6)

        # These are per-sample standard deviations, matching the public code.
        self.w_gyr = np.ones(3) * 0.1**2
        self.w_acc = np.ones(3) * 2.0**2
        self.w_tdoa = np.array([math.sqrt(0.05)])
        self.w_gyr_b = np.ones(3) * 0.000001**2
        self.w_acc_b = np.ones(3) * 0.0001**2
        self.imu_sample_period = 0.01

        self.t_uv = np.array([-0.01245, 0.00127, 0.0908])
        self.rej_cnt = 0
        self.nis_history = []
        self.last_nis = math.nan
        self.stationary_update_count = 0

        # Conservative pseudo-measurement uncertainties.  Updates are rate
        # limited by the replay scheduler, so these are not repeatedly applied
        # at the raw 1 kHz IMU rate.
        self.w_zero_angular_rate = math.radians(0.5) ** 2

    def _imu_si(self, u):
        u = np.asarray(u, dtype=float).reshape(6)
        if self.config.consistent_bias_order:
            b_g = self.bias[:3]
            b_a = self.bias[3:]
        else:
            # Public nominal state: [b_a, b_g].
            b_a = self.bias[:3]
            b_g = self.bias[3:]
        acceleration = u[:3] * GRAVITY_MAGNITUDE - b_a
        angular_rate = u[3:] * DEG_TO_RAD - b_g
        return acceleration, angular_rate

    def _state_matrix(self, R, v, p):
        X = np.eye(5)
        X[:3, :3] = (
            project_to_so3(R)
            if self.config.project_rotation_each_step
            else R
        )
        X[:3, 3] = v
        X[:3, 4] = p
        return X

    @staticmethod
    def _augmented_adjoint(X):
        adjoint = np.eye(15)
        adjoint[:9, :9] = SE3_2_adjoint(X)
        return adjoint

    def initialize_right_invariant_covariance(self, physical_covariance):
        """Map a physical/local covariance into this filter's error coordinates.

        The public implementation inserted independent physical attitude,
        velocity, and position variances directly into the right-invariant
        covariance.  For ``H_r = H_l Ad(X^-1)``, the corresponding covariance
        is ``P_r = Ad(X) P_l Ad(X)'``.  The position-attitude cross terms are
        essential for translation-invariant gating when yaw is uncertain.
        """
        physical_covariance = np.asarray(physical_covariance, dtype=float)
        if physical_covariance.shape != (15, 15):
            raise ValueError("Initial covariance must have shape (15, 15)")
        augmented = self._augmented_adjoint(self.X)
        covariance = augmented @ physical_covariance @ augmented.T
        covariance = 0.5 * (covariance + covariance.T)
        self.P = covariance
        self.P_hat = covariance.copy()
        return covariance

    def _error_dynamics(self, X):
        gravity_cross = SO3_wedge(np.array([0.0, 0.0, -GRAVITY_MAGNITUDE]))
        R = X[:3, :3]
        A = np.zeros((15, 15))
        A[THETA, BG] = -R
        A[VEL, THETA] = gravity_cross
        A[VEL, BG] = -SO3_wedge(X[:3, 3]) @ R
        A[VEL, BA] = -R
        A[POS, VEL] = np.eye(3)
        A[POS, BG] = -SO3_wedge(X[:3, 4]) @ R
        return A

    def _sample_noise_variances(self):
        if self.config.consistent_bias_order:
            bias_1, bias_2 = self.w_gyr_b, self.w_acc_b
        else:
            bias_1, bias_2 = self.w_acc_b, self.w_gyr_b
        return np.concatenate(
            (
                self.w_gyr,
                self.w_acc,
                np.zeros(3),
                bias_1,
                bias_2,
            )
        )

    def _sample_noise_covariance(self):
        return np.diag(self._sample_noise_variances())

    @staticmethod
    def structured_transition(A, dt):
        """Exact ``exp(A dt)`` for the frozen InEKF error dynamics.

        The upper-left navigation block is nilpotent of degree three and the
        bias rows are zero, hence the complete augmented matrix satisfies
        ``A**4 == 0`` even though its bias-coupling columns depend on the state.
        """
        A_dt = A * float(dt)
        A_dt2 = A_dt @ A_dt
        return np.eye(A.shape[0]) + A_dt + 0.5 * A_dt2 + (A_dt2 @ A_dt) / 6.0

    @staticmethod
    def van_loan_discretization(A, continuous_covariance, dt):
        """Exact LTI covariance integral for a constant A over ``dt``."""
        n = A.shape[0]
        transition_generator = A * dt
        # Solve dQ/ds = FQ + QF' + Qc*dt by its converged matrix series for
        # the very small event intervals in UTIL.  This evaluates the same
        # covariance integral as Van Loan while using 15x15 products instead
        # of a generic 30x30 exponential.
        if np.linalg.norm(transition_generator, ord=np.inf) <= 0.05:
            Phi = linalg.expm(transition_generator)
            term = continuous_covariance * dt
            Qd = term.copy()
            for order in range(1, 9):
                term = (
                    transition_generator @ term
                    + term @ transition_generator.T
                ) / (order + 1)
                Qd += term
            return Phi, 0.5 * (Qd + Qd.T)

        van_loan = np.zeros((2 * n, 2 * n))
        van_loan[:n, :n] = A
        van_loan[:n, n:] = continuous_covariance
        van_loan[n:, n:] = -A.T
        exponential = linalg.expm(van_loan * dt)
        Phi = exponential[:n, :n]
        Qd = exponential[:n, n:] @ Phi.T
        return Phi, 0.5 * (Qd + Qd.T)

    def predict(self, u, dt, u_next=None):
        """Propagate to the next event; optionally use a midpoint IMU sample."""
        dt = float(dt)
        if dt < 0.0:
            raise ValueError("IMU propagation dt must be non-negative")
        if dt == 0.0:
            return self.X_hat, self.P_hat, self.bias

        R = self.X[:3, :3]
        v = self.X[:3, 3]
        p = self.X[:3, 4]
        a0, omega0 = self._imu_si(u)

        if self.config.midpoint_integration and u_next is not None:
            a1, omega1 = self._imu_si(u_next)
            acceleration = 0.5 * (a0 + a1)
            angular_rate = 0.5 * (omega0 + omega1)
            R_mid = R @ SO3_exp(0.5 * angular_rate * dt)
        else:
            acceleration = a0
            angular_rate = omega0
            R_mid = R

        gravity = np.array([0.0, 0.0, -GRAVITY_MAGNITUDE])
        if (
            self.config.closed_form_imu_integration
            and not (self.config.midpoint_integration and u_next is not None)
        ):
            phi = angular_rate * dt
            rotation_increment, gamma1, gamma2 = SO3_motion_terms(phi)
            R_new = R @ rotation_increment
            delta_velocity_body = gamma1 @ acceleration
            delta_position_body = gamma2 @ acceleration
            v_new = v + gravity * dt + R @ delta_velocity_body * dt
            p_new = (
                p
                + v * dt
                + 0.5 * gravity * dt**2
                + R @ delta_position_body * dt**2
            )
        else:
            R_new = R @ SO3_exp(angular_rate * dt)
            world_acceleration = R_mid @ acceleration + gravity
            v_new = v + world_acceleration * dt
            p_new = p + v * dt + 0.5 * world_acceleration * dt**2

        if self.config.ground_clamp and p_new[2] < 0.0:
            p_new[2] = 0.0
            v_new = np.zeros(3)

        self.X_hat = self._state_matrix(R_new, v_new, p_new)
        A = self._error_dynamics(self.X_hat)
        augmented_adjoint = self._augmented_adjoint(self.X_hat)

        if self.config.van_loan_discretization or self.config.trapezoidal_discretization:
            # The public parameters are per-IMU-sample variances.  Use the IMU
            # sample period (not an asynchronous UWB sub-step) to construct a
            # continuous density, so splitting propagation at a UWB timestamp
            # does not artificially reduce process noise.
            continuous_covariance = (
                augmented_adjoint
                @ self._sample_noise_covariance()
                @ augmented_adjoint.T
                * self.imu_sample_period
            )
            if self.config.van_loan_discretization:
                Phi, Qd = self.van_loan_discretization(A, continuous_covariance, dt)
            else:
                Phi = linalg.expm(A * dt)
                propagated_covariance = Phi @ continuous_covariance @ Phi.T
                Qd = 0.5 * (continuous_covariance + propagated_covariance) * dt
            self.P_hat = Phi @ self.P @ Phi.T + Qd
        else:
            if self.config.structured_transition:
                Phi = self.structured_transition(A, dt)
                noise_map = Phi @ augmented_adjoint
                noise_variances = self._sample_noise_variances() * dt**2
                process_covariance = (
                    noise_map * noise_variances[np.newaxis, :]
                ) @ noise_map.T
                self.P_hat = Phi @ self.P @ Phi.T + process_covariance
            else:
                Phi = linalg.expm(A * dt)
                sample_covariance = self._sample_noise_covariance()
                Q = sample_covariance * dt**2
                self.P_hat = (
                    Phi @ self.P @ Phi.T
                    + Phi @ augmented_adjoint @ Q @ augmented_adjoint.T @ Phi.T
                )

        self.P_hat = 0.5 * (self.P_hat + self.P_hat.T)
        self.X = self.X_hat.copy()
        self.P = self.P_hat.copy()
        return self.X_hat, self.P_hat, self.bias

    def _measurement_geometry(self, uwb, anchor_position, X):
        """Return prediction, tag position and prediction position gradient."""
        an_A = anchor_position[int(uwb[0])]
        an_B = anchor_position[int(uwb[1])]
        p_tag = X[:3, 4] + X[:3, :3] @ self.t_uv
        delta_A = p_tag - an_A
        delta_B = p_tag - an_B
        d_A = linalg.norm(delta_A)
        d_B = linalg.norm(delta_B)
        if min(d_A, d_B) < 1e-9:
            raise ValueError("UWB tag position coincides with an anchor")
        gradient = delta_B / d_B - delta_A / d_A
        return d_B - d_A, p_tag, gradient

    def measurement(self, uwb, anchor_position, X=None):
        X = self.X_hat if X is None else X
        prediction, _, _ = self._measurement_geometry(uwb, anchor_position, X)
        return prediction

    def measurement_jacobian(
        self,
        uwb,
        anchor_position,
        X=None,
        geometry=None,
    ):
        """Jacobian of ``y-h(Xhat)`` w.r.t. the right-invariant error.

        With ``X_true = Exp(-xi) Xhat``, the first-order tag displacement is
        ``delta p_tag = p_tag^ theta - delta p``.  This directly includes the
        complete lever-arm attitude sensitivity.
        """
        X = self.X_hat if X is None else X
        if geometry is None:
            geometry = self._measurement_geometry(uwb, anchor_position, X)
        _, p_tag, gradient = geometry

        H = np.zeros((1, 15))
        H[0, THETA] = gradient @ SO3_wedge(p_tag)
        H[0, POS] = -gradient
        return H

    def _legacy_measurement_jacobian(
        self,
        uwb,
        anchor_position,
        X,
        geometry=None,
    ):
        if geometry is None:
            geometry = self._measurement_geometry(uwb, anchor_position, X)
        _, _, gradient = geometry
        H = np.zeros((1, 15))
        H[0, POS] = -gradient
        return H

    def _direct_transformed_measurement_jacobian(
        self,
        uwb,
        anchor_position,
        X,
        geometry=None,
    ):
        """Compute the existing ``H_left Ad(X^-1)`` without dense Adjoint work.

        This deliberately preserves the task01/legacy observation model.  It is
        not the research-only complete lever-arm Jacobian returned by
        :meth:`measurement_jacobian`.
        """
        if geometry is None:
            geometry = self._measurement_geometry(uwb, anchor_position, X)
        _, _, gradient = geometry
        h_position = -gradient
        rotation_transpose = X[:3, :3].T
        position = X[:3, 4]
        H = np.zeros((1, 15))
        H[0, THETA] = -h_position @ rotation_transpose @ SO3_wedge(position)
        H[0, POS] = h_position @ rotation_transpose
        return H

    def correct(self, uwb, anchor_position):
        geometry = self._measurement_geometry(uwb, anchor_position, self.X_hat)
        y_hat = geometry[0]
        residual = np.array([float(uwb[2]) - y_hat])

        if self.config.direct_lever_arm_jacobian:
            H = self.measurement_jacobian(
                uwb,
                anchor_position,
                geometry=geometry,
            )
        elif self.config.direct_transformed_jacobian:
            H = self._direct_transformed_measurement_jacobian(
                uwb,
                anchor_position,
                self.X_hat,
                geometry=geometry,
            )
        else:
            H_left = self._legacy_measurement_jacobian(
                uwb,
                anchor_position,
                self.X_hat,
                geometry=geometry,
            )
            if not self.config.coordinate_consistent_covariance:
                # Reproduce the public implementation: this preliminary gate
                # mixes a left-coordinate Jacobian with a right-invariant P.
                gate_variance = float(H_left @ self.P_hat @ H_left.T + self.w_tdoa**2)
                gate_nis = float(residual[0] ** 2 / gate_variance)
                if gate_nis >= 25.0:
                    self.last_nis = gate_nis
                    self.nis_history.append(gate_nis)
                    self.rej_cnt += 1
                    self.X = self.X_hat.copy()
                    self.P = self.P_hat.copy()
                    return self.X, self.P, self.bias, 1
            augmented = np.eye(15)
            augmented[:9, :9] = SE3_2_adjoint(SE3_2_inv(self.X_hat))
            H = H_left @ augmented

        measurement_variance = float(self.w_tdoa[0] ** 2)
        covariance_projection = np.asarray(self.P_hat @ H.T).reshape(15)
        innovation_variance = float(H @ covariance_projection + measurement_variance)
        if innovation_variance <= np.finfo(float).tiny:
            raise FloatingPointError("UWB innovation variance must be positive")
        nis = float(residual[0] ** 2 / innovation_variance)
        self.last_nis = nis
        self.nis_history.append(nis)
        if nis >= 25.0:
            self.rej_cnt += 1
            self.X = self.X_hat.copy()
            self.P = self.P_hat.copy()
            return self.X, self.P, self.bias, 1

        K = covariance_projection / innovation_variance
        delta = K * residual[0]
        if self.config.rank_one_scalar_update:
            self.P = self.P_hat - np.outer(
                covariance_projection,
                covariance_projection,
            ) / innovation_variance
        else:
            K_column = K.reshape(15, 1)
            identity = np.eye(15)
            IKH = identity - K_column @ H
            self.P = (
                IKH @ self.P_hat @ IKH.T
                + measurement_variance * np.outer(K, K)
            )
        self.X = SE3_2_exp(-delta[:9]) @ self.X_hat
        if self.config.project_rotation_each_step:
            self.X[:3, :3] = project_to_so3(self.X[:3, :3])
        self.bias = self.bias - delta[9:15]
        if self.config.covariance_reset:
            reset = np.eye(15)
            reset[:9, :9] = SE3_2_right_jacobian(delta[:9])
            self.P = reset @ self.P @ reset.T
        self.P = 0.5 * (self.P + self.P.T)
        self.X_hat = self.X.copy()
        self.P_hat = self.P.copy()
        return self.X, self.P, self.bias, 0

    def correct_stationary(self, gyroscope_degrees_per_second):
        """Fuse a zero-angular-rate observation during detected initial rest.

        At rest, the raw gyro measurement observes the gyroscope bias directly.
        A zero-velocity row is intentionally excluded: with large initial yaw
        uncertainty, its right-invariant velocity/attitude coupling can inject
        spurious yaw information even though yaw is unobservable at rest.
        """
        if not self.config.consistent_bias_order:
            raise ValueError("Stationary update requires [b_g, b_a] bias order")

        gyro_measurement = (
            np.asarray(gyroscope_degrees_per_second, dtype=float).reshape(3)
            * DEG_TO_RAD
        )
        residual = gyro_measurement - self.bias[:3]

        H = np.zeros((3, 15))
        H[:, BG] = -np.eye(3)
        measurement_covariance = np.eye(3) * self.w_zero_angular_rate

        innovation_covariance = H @ self.P_hat @ H.T + measurement_covariance
        K = linalg.solve(
            innovation_covariance,
            H @ self.P_hat,
            assume_a="pos",
        ).T
        delta = np.asarray(K @ residual).reshape(15)
        identity = np.eye(15)
        IKH = identity - K @ H
        self.P = (
            IKH @ self.P_hat @ IKH.T
            + K @ measurement_covariance @ K.T
        )
        self.X = SE3_2_exp(-delta[:9]) @ self.X_hat
        self.X[:3, :3] = project_to_so3(self.X[:3, :3])
        self.bias = self.bias - delta[9:15]
        self.P = 0.5 * (self.P + self.P.T)
        self.X_hat = self.X.copy()
        self.P_hat = self.P.copy()
        self.stationary_update_count += 1
        return self.X, self.P, self.bias
