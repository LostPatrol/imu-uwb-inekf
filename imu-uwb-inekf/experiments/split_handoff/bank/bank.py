"""Causal circular heading bank with nonoverlapping raw-radio prediction blocks.

Large heading basins remain separate from local p/v and body-bias estimation.
"""

from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import least_squares

# Rules are frozen independently of any UTIL truth-based evaluation.
RULE_PATH = Path(__file__).with_name("frozen_rules.json")
RULES = json.loads(RULE_PATH.read_text(encoding="utf-8"))
# The navigation frame has known upward z and gravitational acceleration in SI.
GRAVITY = np.array([0., 0., -9.81])
# Fixed circular particles include the retained heading at index zero.
ANGLES = np.deg2rad(np.asarray(RULES["heading_offsets_deg"], dtype=float))


def yaw_rotation(angle):
    """Return a proper rotation about the known world vertical."""
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array([[cosine, -sine, 0.], [sine, cosine, 0.], [0., 0., 1.]])


def radio_geometry(position, plus, minus):
    """Evaluate directed TDoA or TWR and the corresponding position gradient."""
    delta = position-plus
    distance = np.maximum(np.linalg.norm(delta, axis=1), 1e-10)
    gradient = delta/distance[:, None]
    if minus is not None:
        other = position-minus
        other_distance = np.maximum(np.linalg.norm(other, axis=1), 1e-10)
        gradient -= other/other_distance[:, None]
        distance -= other_distance
    return distance, gradient


def huber_cost(residual):
    """Return twice the fixed normalized Huber objective per observation."""
    absolute = np.abs(residual)
    threshold = RULES["normalized_Huber_threshold"]
    return np.where(absolute <= threshold, residual**2, 2*threshold*absolute-threshold**2)


def mixed_loss(z, count):
    """Use Huber for raw radio and a Gaussian quadratic body-bias prior."""
    out = np.vstack((z.copy(), np.ones_like(z), np.zeros_like(z)))
    threshold = RULES["normalized_Huber_threshold"]
    indices = np.flatnonzero((np.arange(len(z)) < count) & (z > threshold**2))
    root = np.sqrt(z[indices])
    out[0, indices] = 2*threshold*root-threshold**2
    out[1, indices] = threshold/root
    out[2, indices] = -.5*threshold/root**3
    return out


def cv_seed(tau, plus, minus, measurement, sigma, previous=None):
    """Fit only the already-arrived radio prefix, without a future path initializer."""
    initial = (np.r_[np.mean(plus if minus is None else np.vstack((plus, minus)), axis=0), np.zeros(3)]
               if previous is None else previous)
    def residual(x):
        """Evaluate normalized prefix-only constant-velocity radio residuals."""
        predicted, _ = radio_geometry(x[:3]+tau[:, None]*x[3:6], plus, minus)
        return (predicted-measurement)/sigma
    def jacobian(x):
        """Differentiate prefix-only translation variables."""
        _, gradient = radio_geometry(x[:3]+tau[:, None]*x[3:6], plus, minus)
        return np.column_stack((gradient, tau[:, None]*gradient))/sigma[:, None]
    fit = least_squares(residual, initial, jac=jacobian, loss="huber",
        f_scale=RULES["normalized_Huber_threshold"], max_nfev=RULES["maximum_function_evaluations_per_prefix_fit"])
    return fit.x, int(fit.nfev)


def prefix_gauge(tau, force, bias, prefix):
    """Remove endpoint-dependent affine IMU terms using prefix samples only."""
    affine = np.column_stack((np.ones(len(tau)), tau))
    gravity = .5*tau[:, None]**2*GRAVITY
    # Any IMU after the prefix enters endpoint-reanchoring only through these affine terms.
    clean_force = force-affine@np.linalg.lstsq(affine[prefix], force[prefix], rcond=1e-10)[0]
    flat_bias = bias.reshape(len(tau), 9)
    clean_bias = (flat_bias-affine@np.linalg.lstsq(affine[prefix], flat_bias[prefix], rcond=1e-10)[0]).reshape(-1, 3, 3)
    clean_gravity = gravity-affine@np.linalg.lstsq(affine[prefix], gravity[prefix], rcond=1e-10)[0]
    return clean_force, clean_bias, clean_gravity


def fit_particle(tau, force, bias, gravity, plus, minus, measurement, sigma, rotation, seed):
    """Fit p/v and a weakly regularized body bias while holding one heading fixed."""
    bias_std = RULES["body_bias_prior_std_mps2"]
    count = len(tau)
    def evaluate(x, need_jacobian=False):
        """Return original radio factors or analytic translation/bias derivatives."""
        corrected = force-np.einsum("nij,j->ni", bias, x[6:9])
        position = x[:3]+tau[:, None]*x[3:6]+gravity+corrected@rotation.T
        predicted, gradient = radio_geometry(position, plus, minus)
        if not need_jacobian:
            return np.r_[(predicted-measurement)/sigma, x[6:9]/bias_std]
        jacobian = np.column_stack((gradient, tau[:, None]*gradient,
            -np.einsum("ni,ij,njk->nk", gradient, rotation, bias)))/sigma[:, None]
        return np.vstack((jacobian, np.column_stack((np.zeros((3, 6)), np.eye(3)/bias_std))))
    fit = least_squares(evaluate, np.r_[seed, np.zeros(3)], jac=lambda x:evaluate(x, True),
        loss=lambda z:mixed_loss(z, count), max_nfev=RULES["maximum_function_evaluations_per_prefix_fit"])
    return fit


def posterior(log_weights):
    """Normalize circular particle weights without overflowing likelihood ratios."""
    weights = np.exp(log_weights-np.max(log_weights))
    return weights/weights.sum()


def decide(tau, force, B, plus, minus, y, sigma, path, Cref):
    """Return a truth-free yaw correction after causal out-of-prefix prediction."""
    started = time.perf_counter()
    tau, force, B, plus, y = map(np.asarray, (tau, force, B, plus, y))
    sigma = np.broadcast_to(np.asarray(sigma, dtype=float), y.shape)
    order = np.argsort(tau, kind="stable")
    tau, force, B, plus, y, sigma = (item[order] for item in (tau, force, B, plus, y, sigma))
    minus = None if minus is None else np.asarray(minus)[order]
    log_weights = np.zeros(len(ANGLES))
    log_weights[0] += RULES["initial_log_prior_advantage_for_retaining_heading"]
    trace, total_nfev, nonconverged, previous_cv = [], 0, 0, None
    span = float(tau[-1]-tau[0]) if len(tau) else 0.
    block_width = max(RULES["minimum_prediction_block_s"],
                      (span-RULES["initial_prefix_s"])/RULES["maximum_prediction_blocks"])
    left = float(tau[0])+RULES["initial_prefix_s"] if len(tau) else 0.
    last_winner, streak = -1, 0
    rotations = [yaw_rotation(angle)@Cref for angle in ANGLES]
    while len(trace) < RULES["maximum_prediction_blocks"] and left < (float(tau[-1]) if len(tau) else 0.)-1e-9:
        right = min(left+block_width, float(tau[-1])+1e-9)
        prefix = tau < left
        validation = (tau >= left) & (tau < right)
        if prefix.sum() < RULES["minimum_prefix_observations"] or validation.sum() < RULES["minimum_validation_observations"]:
            left = right
            continue
        previous_cv, evaluations = cv_seed(tau[prefix], plus[prefix], None if minus is None else minus[prefix],
            y[prefix], sigma[prefix], previous_cv)
        total_nfev += evaluations
        clean_force, clean_bias, clean_gravity = prefix_gauge(tau, force, B, prefix)
        losses, fit_flags = [], []
        for rotation in rotations:
            fit = fit_particle(tau[prefix], clean_force[prefix], clean_bias[prefix], clean_gravity[prefix],
                plus[prefix], None if minus is None else minus[prefix], y[prefix], sigma[prefix], rotation, previous_cv)
            total_nfev += int(fit.nfev)
            nonconverged += int(not fit.success)
            fit_flags.append(bool(fit.success))
            corrected = clean_force[validation]-np.einsum("nij,j->ni", clean_bias[validation], fit.x[6:9])
            predicted_position = fit.x[:3]+tau[validation, None]*fit.x[3:6]+clean_gravity[validation]+corrected@rotation.T
            predicted, _ = radio_geometry(predicted_position, plus[validation], None if minus is None else minus[validation])
            residual = (predicted-y[validation])/sigma[validation]
            losses.append(float(np.mean(np.minimum(huber_cost(residual), RULES["per_observation_prediction_loss_cap"]))))
        # Only measurements excluded from every fit above update heading evidence.
        log_weights -= .5*RULES["effective_independent_observations_per_block"]*np.asarray(losses)
        weights = posterior(log_weights)
        winner = int(np.argmax(weights))
        streak = streak+1 if winner == last_winner else 1
        last_winner = winner
        trace.append(dict(prefix_end_tau=float(left), prediction_end_tau=float(right),
            prefix_count=int(prefix.sum()), prediction_count=int(validation.sum()),
            mean_prediction_losses=losses, weights=weights.tolist(), winner_deg=float(np.rad2deg(ANGLES[winner])),
            winner_streak=streak, fit_success=fit_flags))
        left = right
    weights = posterior(log_weights)
    winner = int(np.argmax(weights))
    log_odds = float(log_weights[winner]-log_weights[0])
    accepted = bool(winner != 0 and len(trace) >= RULES["minimum_valid_prediction_blocks_to_switch"]
        and streak >= RULES["minimum_consecutive_winning_blocks_to_switch"]
        and weights[winner] >= RULES["minimum_winner_posterior_mass"]
        and log_odds >= RULES["minimum_log_odds_against_retained_heading"])
    resultant = float(abs(np.sum(weights*np.exp(1j*ANGLES))))
    grid_std = (2*np.pi/len(ANGLES))/np.sqrt(12.)
    circular_std = np.sqrt(max(0., -2*np.log(max(resultant, 1e-12)))+grid_std**2)
    candidate = float(ANGLES[winner])
    return dict(delta=candidate if accepted else 0., candidate=candidate, accepted=accepted,
        sd_deg=float(np.rad2deg(circular_std)), method="causal_heading_bank8", posterior_mass=float(weights[winner]),
        log_odds_vs_retain=log_odds, blocks=len(trace), streak=streak, weights=weights.tolist(), trace=trace,
        total_nfev=total_nfev, nonconverged_prefix_fits=nonconverged,
        elapsed_s=time.perf_counter()-started, full_window_radio_path_used=False,
        explanation="switch" if accepted else "retain: insufficient stable predictive evidence")
