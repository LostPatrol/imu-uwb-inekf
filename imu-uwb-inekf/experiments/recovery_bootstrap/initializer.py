"""Causal range/IMU window initialization with an exact yaw-profile relaxation.

The algebra assumes known IMU biases and gravity alignment over the window.
The final estimate minimizes raw ranges; the lift is only a candidate generator.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
from scipy.optimize import least_squares

# Raw-range residuals beyond this normalized magnitude receive Huber weights.
HUBER_SCALE = 2.0
# Number of uniformly spaced yaw seeds in the strong nonlinear reference.
YAW_STARTS = 12
# Relative rank tolerance used by every nuisance projection.
RANK_RTOL = 1e-10
# Bound solver work independently of the numerical quality of an initial guess.
MAX_EVALUATIONS = 120
# Broad, explicit physical regularization; a pseudoinverse would erase null variances.
CURVATURE_PRIOR_STD = np.r_[np.full(3, 8.0), np.full(3, 2.0), np.pi/np.sqrt(3.0)]


def yaw_rotation(angle):
    """Return a world-vertical rotation in radians."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def project_out(design, values):
    """Orthogonally remove a possibly rank-deficient nuisance column space."""
    u, singular, _ = np.linalg.svd(design, full_matrices=False)
    rank = int(np.sum(singular > (singular[0] * RANK_RTOL))) if singular.size else 0
    basis = u[:, :rank]
    return values - basis @ (basis.T @ values), rank


def circle_minimum(hessian, gradient):
    """Find the global minimum of u'Hu-2g'u on the unit circle via a quartic."""
    h11, h12, h22 = hessian[0, 0], hessian[0, 1], hessian[1, 1]
    g1, g2 = gradient
    difference = h22 - h11
    coefficients = np.array([h12 + g2, -2 * difference + 2 * g1,
                             -6 * h12, 2 * difference + 2 * g1, h12 - g2])
    scale = max(np.max(np.abs(coefficients)), np.finfo(float).tiny)
    coefficients = np.trim_zeros(np.where(np.abs(coefficients) > scale * 1e-13,
                                          coefficients / scale, 0.0), "f")
    candidates = [0.0, np.pi]
    if coefficients.size > 1:
        for root in np.roots(coefficients):
            if abs(root.imag) < 1e-7 * (1 + abs(root.real)):
                candidates.append(float(2 * np.arctan(root.real)))
    angles = np.array(candidates)
    vectors = np.column_stack((np.cos(angles), np.sin(angles)))
    costs = np.einsum("ni,ij,nj->n", vectors, hessian, vectors) - 2 * vectors @ gradient
    order = np.argsort(costs)
    return float(angles[order[0]]), angles[order], costs[order]


def lifted_candidate(tau, displacement, anchors, ranges, sigmas):
    """Eliminate seven norm/cross nuisances, then solve the relaxed yaw globally.

    p(t)=p_end+v_end*tau+Rz(yaw)*displacement(t); the expansion is exact.
    Free lifted nuisance variables relax their nonlinear consistency constraints.
    """
    t = np.asarray(tau)
    d = np.asarray(displacement)
    a = np.asarray(anchors)
    # Squared-range mean correction and first-order heteroscedastic whitening.
    response = 0.5 * (ranges**2 - sigmas**2 - np.sum(a*a, axis=1)
                      - np.sum(d*d, axis=1)) + a[:, 2] * d[:, 2]
    nuisance = np.column_stack((np.ones(t.size), t, t*t, d[:, :2], t[:, None]*d[:, :2]))
    translation = np.column_stack((-a, -t[:, None]*a))
    translation[:, 2] += d[:, 2]
    translation[:, 5] += t*d[:, 2]
    circular = np.column_stack((-(a[:, 0]*d[:, 0] + a[:, 1]*d[:, 1]),
                                a[:, 0]*d[:, 1] - a[:, 1]*d[:, 0]))
    sigma_squared_range = np.sqrt(np.maximum(ranges, sigmas)**2 * sigmas**2 + 0.5*sigmas**4)
    weights = 1.0 / sigma_squared_range
    linear = np.column_stack((nuisance, translation)) * weights[:, None]
    projected, rank = project_out(linear, np.column_stack((response, circular))*weights[:, None])
    y, design = projected[:, 0], projected[:, 1:]
    angle, candidates, costs = circle_minimum(design.T @ design, design.T @ y)
    u = np.array([np.cos(angle), np.sin(angle)])
    solution = np.linalg.lstsq(linear, (response - circular@u)*weights, rcond=RANK_RTOL)[0]
    return np.r_[solution[7:13], angle], dict(nuisance_rank=rank,
        projected_yaw_singular=np.linalg.svd(design, compute_uv=False).tolist(),
        candidate_angles=candidates.tolist(), candidate_costs=costs.tolist())


def raw_residual_jacobian(parameters, tau, displacement, anchors, ranges, sigmas):
    """Evaluate the original asynchronous range model and its analytic Jacobian."""
    rotation = yaw_rotation(parameters[6])
    rotated = displacement @ rotation.T
    position = parameters[:3] + tau[:, None]*parameters[3:6] + rotated
    difference = position - anchors
    distance = np.maximum(np.linalg.norm(difference, axis=1), 1e-12)
    directions = difference / distance[:, None]
    yaw_derivative = np.column_stack((-rotated[:, 1], rotated[:, 0], np.zeros(tau.size)))
    jacobian = np.column_stack((directions, tau[:, None]*directions,
                               np.sum(directions*yaw_derivative, axis=1))) / sigmas[:, None]
    return (distance-ranges)/sigmas, jacobian


def robust_cost(residual):
    """Return the same Huber objective used by scipy least_squares."""
    absolute = np.abs(residual)
    return float(np.sum(np.where(absolute <= HUBER_SCALE, residual**2,
                                 2*HUBER_SCALE*absolute - HUBER_SCALE**2)))


def cv_candidate(tau, anchors, ranges, sigmas):
    """Fit a robust constant-velocity range window with no position truth."""
    # A squared-range relaxation supplies a geometry-only, translation-free seed.
    design = np.column_stack((np.ones(tau.size), tau, tau*tau, -anchors,
                              -tau[:, None]*anchors))
    response = 0.5*(ranges**2 - sigmas**2 - np.sum(anchors*anchors, axis=1))
    weights = 1 / np.maximum(ranges*sigmas, sigmas**2)
    initial = np.linalg.lstsq(design*weights[:, None], response*weights, rcond=RANK_RTOL)[0][3:9]
    def residual(x):
        """Evaluate constant-velocity residuals at the actual transaction times."""
        difference = x[:3] + tau[:, None]*x[3:] - anchors
        return (np.linalg.norm(difference, axis=1)-ranges)/sigmas
    def jacobian(x):
        """Differentiate the six physical translation variables."""
        difference = x[:3] + tau[:, None]*x[3:] - anchors
        direction = difference / np.maximum(np.linalg.norm(difference, axis=1)[:, None], 1e-12)
        return np.column_stack((direction, tau[:, None]*direction))/sigmas[:, None]
    # The centroid fallback prevents a nearly singular polynomial seed dominating.
    seeds = [initial, np.r_[np.mean(anchors, axis=0), np.zeros(3)]]
    fitted = [least_squares(residual, seed, jac=jacobian, loss="huber", f_scale=HUBER_SCALE,
                            max_nfev=MAX_EVALUATIONS) for seed in seeds]
    best = min(fitted, key=lambda item: robust_cost(item.fun))
    return best.x, best.fun, jacobian(best.x)


@dataclass
class WindowEstimate:
    """Endpoint state, local information, runtime, and diagnostic provenance."""
    parameters: np.ndarray
    covariance: np.ndarray
    yaw_information: float
    residual_rms: float
    robust_objective: float
    elapsed_s: float
    evaluations: int
    diagnostics: dict


def solve_window(tau, displacement, anchors, ranges, sigmas, method="profile"):
    """Compare CV, one-start inertial, multistart inertial, and profiled-lift seeds."""
    started = time.perf_counter()
    cv, cv_residual, cv_jacobian = cv_candidate(tau, anchors, ranges, sigmas)
    diagnostics = {}
    if method == "cv":
        parameters = np.r_[cv, 0.0]
        residual = cv_residual
        jacobian = np.column_stack((cv_jacobian, np.zeros(tau.size)))
        evaluations = 0
    else:
        def fun(x):
            """Return raw metric residuals normalized by frozen range scales."""
            return raw_residual_jacobian(x, tau, displacement, anchors, ranges, sigmas)[0]
        def jac(x):
            """Return the analytic raw-range Jacobian."""
            return raw_residual_jacobian(x, tau, displacement, anchors, ranges, sigmas)[1]
        if method == "profile":
            seed, diagnostics = lifted_candidate(tau, displacement, anchors, ranges, sigmas)
            # The profile only determines yaw; robust CV translation is better conditioned.
            seeds = [np.r_[cv, seed[6]]]
            diagnostics["lift_translation"] = seed[:6].tolist()
        elif method == "multistart":
            seeds = [np.r_[cv, angle] for angle in np.linspace(-np.pi, np.pi, YAW_STARTS, endpoint=False)]
        elif method == "local":
            seeds = [np.r_[cv, 0.0]]
        else:
            raise ValueError(f"Unknown method {method}")
        fits = [least_squares(fun, seed, jac=jac, loss="huber", f_scale=HUBER_SCALE,
                              max_nfev=MAX_EVALUATIONS) for seed in seeds]
        fit = min(fits, key=lambda item: robust_cost(item.fun))
        parameters, residual, jacobian = fit.x, fit.fun, jac(fit.x)
        evaluations = sum(item.nfev for item in fits)
        diagnostics["solver_success"] = bool(fit.success)
        diagnostics["seed_costs"] = [robust_cost(item.fun) for item in fits]
    # This is local robust curvature, not a calibrated posterior covariance.
    huber_weights = np.minimum(1.0, HUBER_SCALE/np.maximum(np.abs(residual), 1e-12))
    weighted = jacobian*np.sqrt(huber_weights[:, None])
    projected, _ = project_out(weighted[:, :6], weighted[:, 6:7])
    yaw_information = float(np.sum(projected**2))
    # A broad prior preserves finite uncertainty in every unobserved physical direction.
    # It is used only for handoff curvature, never for the data-only information gate.
    prior_precision = np.diag(1/CURVATURE_PRIOR_STD**2)
    covariance = np.linalg.inv(weighted.T@weighted + prior_precision)
    diagnostics["curvature_prior_std"] = CURVATURE_PRIOR_STD.tolist()
    parameters[6] = np.arctan2(np.sin(parameters[6]), np.cos(parameters[6]))
    return WindowEstimate(parameters, covariance, yaw_information,
        float(np.sqrt(np.mean(residual**2))), robust_cost(residual),
        time.perf_counter()-started, evaluations, diagnostics)
