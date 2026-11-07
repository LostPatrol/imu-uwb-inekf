"""Right-invariant TWR filter using the ESKF's exact discrete physical model.

The covariance is expressed in [theta_world, invariant_v, invariant_p, bg, ba].
Coordinate changes share the same nominal step and independent noise samples.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm

from eskf15 import ESKF15, discrete_error_matrices, nominal_step, rotation_terms, skew, symmetrize

# A matrix exponential avoids truncated-series cancellation for large injections.
RESET_SERIES_MAX_ANGLE_RAD = 0.5


def coordinate_map(state: np.ndarray) -> np.ndarray:
    """Map body attitude/additive world errors to left-injected group errors."""
    mapping = np.eye(15)
    rotation = state[:3, :3]
    mapping[:3, :3] = rotation
    mapping[3:6, :3] = skew(state[:3, 3]) @ rotation
    mapping[6:9, :3] = skew(state[:3, 4]) @ rotation
    return mapping


def inverse_coordinate_map(state: np.ndarray) -> np.ndarray:
    """Invert the physical-to-invariant map without a dense matrix solve."""
    mapping = np.eye(15)
    mapping[:3, :3] = state[:3, :3].T
    mapping[3:6, :3] = -skew(state[:3, 3])
    mapping[6:9, :3] = -skew(state[:3, 4])
    return mapping


def group_exp(delta: np.ndarray) -> np.ndarray:
    """Exponentiate an SE_2(3) increment in [theta,v,p] order."""
    rotation, left_jacobian = rotation_terms(-delta[:3])
    increment = np.eye(5)
    increment[:3, :3] = rotation.T
    increment[:3, 3] = left_jacobian @ delta[3:6]
    increment[:3, 4] = left_jacobian @ delta[6:9]
    return increment


class Invariant15(ESKF15):
    """Right-invariant error with left group injection and additive IMU bias."""

    def __init__(self, state, covariance, bias, *, lever_arm=(0.0, 0.0, 0.0), **kwargs):
        """Accept invariant P; the protocol converts the common physical prior."""
        super().__init__(state, covariance, bias, **kwargs)
        self.t_uv = np.asarray(lever_arm, dtype=float)
        if np.any(self.t_uv != 0):
            raise ValueError("The common TWR protocol uses zero lever arm")
        self.X_hat = self.X.copy()
        self.P_hat = self.P.copy()

    def physical_covariance(self):
        """Return covariance in the ESKF's body/world physical error coordinates."""
        inverse = inverse_coordinate_map(self.X)
        return symmetrize(inverse @ self.P @ inverse.T)

    def set_translation(self, velocity, position):
        """Re-express the core covariance when the lifted branch changes its origin."""
        physical = self.physical_covariance()
        self.X[:3, 3] = velocity
        self.X[:3, 4] = position
        mapping = coordinate_map(self.X)
        self.P = symmetrize(mapping @ physical @ mapping.T)
        self.X_hat = self.X.copy()
        self.P_hat = self.P.copy()

    def predict(self, imu, dt):
        """Transform the common exact nominal Jacobian and sample-noise map."""
        physical = self.physical_covariance()
        transition, noise = discrete_error_matrices(self.X, self.bias, imu, dt)
        self.X = nominal_step(self.X, self.bias, imu, dt)
        variances = np.concatenate((self.w_gyr, self.w_acc, self.w_gyr_b, self.w_acc_b))
        physical = transition @ physical @ transition.T + (noise * variances) @ noise.T
        mapping = coordinate_map(self.X)
        self.P = symmetrize(mapping @ physical @ mapping.T)
        self.X_hat = self.X.copy()
        self.P_hat = self.P.copy()
        return self.X, self.P, self.bias

    def range_model(self, anchor):
        """Differentiate range under a full left SE_2(3) perturbation."""
        predicted, physical_row = super().range_model(anchor)
        gradient = physical_row[0, 6:9]
        row = physical_row.copy()
        row[0, :3] = -gradient @ skew(self.X[:3, 4])
        return predicted, row

    def inject(self, delta):
        """Apply a left group correction and an additive bias correction."""
        self.X = group_exp(delta[:9]) @ self.X
        self.bias += delta[9:15]

    @staticmethod
    def reset_jacobian(delta):
        """Differentiate Log(Exp(delta+e) Exp(-delta)) for a left injection."""
        algebra = np.zeros((9, 9))
        for start in (0, 3, 6):
            algebra[start:start + 3, start:start + 3] = skew(delta[:3])
        algebra[3:6, :3] = skew(delta[3:6])
        algebra[6:9, :3] = skew(delta[6:9])
        if np.linalg.norm(delta[:3]) > RESET_SERIES_MAX_ANGLE_RAD:
            # The upper-right block integrates exp(ad(delta)*s) for s in [0,1].
            augmented = np.zeros((18, 18))
            augmented[:9, :9] = algebra
            augmented[:9, 9:] = np.eye(9)
            jacobian = expm(augmented)[:9, 9:]
        else:
            jacobian = np.eye(9)
            term = np.eye(9)
            for order in range(1, 40):
                term = term @ algebra / (order + 1)
                jacobian += term
                if np.max(np.abs(term)) < 1e-14:
                    break
        result = np.eye(15)
        result[:9, :9] = jacobian
        return result

    def correct_twr(self, anchor, measurement, anchor_index=None):
        """Use the common scalar policy and keep the legacy supervisor views current."""
        accepted = super().correct_twr(anchor, measurement, anchor_index)
        self.X_hat = self.X.copy()
        self.P_hat = self.P.copy()
        return accepted


def handoff_covariance(capture, position_floor=0.75, velocity_floor=0.75):
    """Build physical block covariance, then map it at the new capture state."""
    source = capture.imu_core.physical_covariance()
    physical = np.zeros((15, 15))
    retained = np.r_[0:3, 9:15]
    physical[np.ix_(retained, retained)] = source[np.ix_(retained, retained)]
    lifted = capture.lifted.P
    # The homogeneous lift uses [q,s,h,p_world,v_world,1], not [p,v,q,s,h].
    physical[3:6, 3:6] = lifted[6:9, 6:9] + velocity_floor**2 * np.eye(3)
    physical[6:9, 6:9] = lifted[3:6, 3:6] + position_floor**2 * np.eye(3)
    physical[3:6, 6:9] = lifted[6:9, 3:6]
    physical[6:9, 3:6] = lifted[3:6, 6:9]
    mapping = coordinate_map(capture.X)
    return symmetrize(mapping @ physical @ mapping.T)
