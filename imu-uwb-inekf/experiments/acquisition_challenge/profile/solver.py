"""Sensor-seeded Profile-1 and rotating body-bias acquisition with optional MAP.

Every fit uses the existing robust CV seed and one exact lifted yaw candidate.
An optional absolute position/SO(3) prior applies at the physical window start.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import time

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

# Reuse the audited candidate and physical raw-range models without changing them.
HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parents[1]
sys.path.insert(0, str(EXPERIMENTS / "recovery_bootstrap"))
sys.path.insert(0, str(EXPERIMENTS / "radio_inertial_models" / "shape_branch"))
import initializer as init
import physical_shape as shape

# Fixed, per-component accelerometer calibration prior from the physical study.
BIAS_STD_MPS2 = 0.10


@dataclass
class StartPrior:
    """Absolute position and full attitude soft priors at the window start."""

    position: np.ndarray
    rotation: np.ndarray
    position_std: float
    attitude_std: float


@dataclass
class Window:
    """Only arrived sensor samples plus their endpoint-anchored IMU geometry."""

    values: tuple
    sensitivity: np.ndarray
    start_tau: float
    start_displacement: np.ndarray
    start_sensitivity: np.ndarray
    start_rotation: np.ndarray
    end_rotation: np.ndarray


def start_state(x, window):
    """Return physical position and full attitude at the actual window start."""
    yaw = init.yaw_rotation(x[6])
    displacement = window.start_displacement.copy()
    if len(x) == 10:
        displacement -= window.start_sensitivity @ x[7:10]
    position = x[:3] + window.start_tau * x[3:6] + yaw @ displacement
    return position, yaw @ window.start_rotation


def start_prior_terms(x, window, prior):
    """Differentiate a position Gaussian and principal-log SO(3) Gaussian."""
    position, rotation = start_state(x, window)
    vector = Rotation.from_matrix(prior.rotation.T @ rotation).as_rotvec()
    residual = np.r_[(position-prior.position)/prior.position_std,
                     vector/prior.attitude_std]
    derivative = np.zeros((6, len(x)))
    derivative[:3, :3] = np.eye(3)/prior.position_std
    derivative[:3, 3:6] = window.start_tau*np.eye(3)/prior.position_std
    displacement = window.start_displacement.copy()
    yaw = init.yaw_rotation(x[6])
    if len(x) == 10:
        displacement -= window.start_sensitivity @ x[7:10]
        derivative[:3, 7:10] = -yaw @ window.start_sensitivity/prior.position_std
    rotated = yaw @ displacement
    derivative[:3, 6] = np.array([-rotated[1], rotated[0], 0.])/prior.position_std
    theta = np.linalg.norm(vector)
    cross = shape.skew(vector)
    coefficient = 1/12 if theta < 1e-5 else (1-.5*theta/np.tan(.5*theta))/theta**2
    inverse_right = np.eye(3)+.5*cross+coefficient*cross@cross
    derivative[3:, 6] = inverse_right @ rotation.T @ np.array([0., 0., 1.])/prior.attitude_std
    return residual, derivative


def sensor_seed(window):
    """Use the unchanged two-start CV translation and single lifted yaw."""
    tau, displacement, anchors, ranges, sigma = window.values
    started = time.perf_counter()
    cv = init.cv_candidate(tau, anchors, ranges, sigma)[0]
    lifted = init.lifted_candidate(tau, displacement, anchors, ranges, sigma)[0]
    return np.r_[cv, lifted[6]], time.perf_counter()-started


def solve(window, model="profile1", prior=None, seed=None):
    """Refine once from sensors; native fits never consume a previous state."""
    started = time.perf_counter()
    if seed is None:
        initial, candidate_s = sensor_seed(window)
    else:
        initial, candidate_s = np.asarray(seed).copy(), 0.
    physical = model == "body_bias_prior"
    if model not in ("profile1", "body_bias_prior"):
        raise ValueError(model)
    initial = np.r_[initial[:7], np.zeros(3)] if physical else initial[:7]
    bias_derivative = np.zeros((3 if physical else 0, len(initial)))
    if physical:
        bias_derivative[:, 7:10] = np.eye(3)/BIAS_STD_MPS2

    def raw(x):
        """Evaluate the frozen original-range model for this method."""
        return (shape.raw_model(x, window.values, window.sensitivity, False) if physical
                else init.raw_residual_jacobian(x, *window.values))

    def extras(x):
        """Keep assumed bias and external state information distinct from data."""
        residual = bias_derivative @ x
        derivative = bias_derivative
        if prior is not None:
            state_residual, state_derivative = start_prior_terms(x, window, prior)
            residual = np.r_[residual, state_residual]
            derivative = np.vstack((derivative, state_derivative))
        return residual, derivative

    def residual(x):
        """Append Gaussian priors after the robust radio residuals."""
        return np.r_[raw(x)[0], extras(x)[0]]

    def jacobian(x):
        """Append exact bias and physical-start state prior derivatives."""
        return np.vstack((raw(x)[1], extras(x)[1]))

    fit = least_squares(residual, initial, jac=jacobian,
        loss=lambda z: shape.mixed_loss(z, len(window.values[0])),
        max_nfev=init.MAX_EVALUATIONS)
    x = fit.x.copy()
    x[6] = np.arctan2(np.sin(x[6]), np.cos(x[6]))
    errors, derivative = raw(x)
    weighted = derivative*np.sqrt(np.minimum(1., init.HUBER_SCALE/
        np.maximum(np.abs(errors), 1e-12)))[:, None]
    nuisance = [index for index in range(len(x)) if index != 6]
    data_projected, _ = init.project_out(weighted[:, nuisance], weighted[:, 6:7])
    augmented = np.vstack((weighted, extras(x)[1]))
    map_projected, _ = init.project_out(augmented[:, nuisance], augmented[:, 6:7])
    rotation = init.yaw_rotation(x[6]) @ window.end_rotation
    return dict(x=x, position=x[:3].copy(), velocity=x[3:6].copy(), rotation=rotation,
        yaw_information_data=float(np.sum(data_projected**2)),
        yaw_information_assumed_prior=float(np.sum(map_projected**2)),
        normalized_radio_rms=float(np.sqrt(np.mean(errors**2))),
        radio_objective=init.robust_cost(errors), penalized_objective=float(2*fit.cost),
        solver_success=bool(fit.success), solver_status=int(fit.status), nfev=int(fit.nfev),
        candidate_s=candidate_s, fit_elapsed_s=time.perf_counter()-started,
        prior_consumed=prior is not None, refinement_starts=1,
        bias_correction_norm_mps2=float(np.linalg.norm(x[7:10])) if physical else 0.)
