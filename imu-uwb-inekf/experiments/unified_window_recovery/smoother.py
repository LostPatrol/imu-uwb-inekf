"""Conditional full-state IEKS replacement using the native discrete IMU model.

Reference trajectories are computational seeds, never independent observations.
Each pass restarts from the same checkpoint distribution and packet ledger.
"""

import sys
import time
import importlib.util
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

# Import the exact established physical model and coordinate conversion.
CODE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE / "experiments/roadmap_stage1"))
sys.path.insert(0, str(CODE / "experiments/recovery_bootstrap"))
from eskf15 import discrete_error_matrices, nominal_step, rotation_terms, symmetrize
from invariant15 import coordinate_map
from initializer import solve_window, yaw_rotation

# Reuse this round's body-bias window model exclusively as an optimization seed.
_SEED_PATH = CODE / "experiments/radio_inertial_models/joint_branch/study.py"
_SEED_SPEC = importlib.util.spec_from_file_location("unified_joint_seed", _SEED_PATH)
joint_seed = importlib.util.module_from_spec(_SEED_SPEC)
_SEED_SPEC.loader.exec_module(joint_seed)

# Native per-sample variances, not continuous-time noise densities.
NOISE = np.array([.1**2]*3 + [2.**2]*3 + [1e-6**2]*3 + [1e-4**2]*3)
# The iteration budget is frozen before any estimator scoring.
PASSES = 4
# The existing raw-window robust loss changes only optimization linearizations.
HUBER = 2.


def difference(reference, reference_bias, target, target_bias):
    """Express target minus reference in physical body-right error coordinates."""
    return np.r_[Rotation.from_matrix(reference[:3, :3].T @ target[:3, :3]).as_rotvec(),
                 target[:3, 3]-reference[:3, 3], target[:3, 4]-reference[:3, 4],
                 target_bias-reference_bias]


def inject(reference, bias, delta):
    """Retract a physical local increment into full state and additive biases."""
    state = reference.copy()
    state[:3, :3] = state[:3, :3] @ rotation_terms(delta[:3])[0]
    state[:3, 3] += delta[3:6]
    state[:3, 4] += delta[6:9]
    return state, bias+delta[9:]


def tangent_transport(delta):
    """Differentiate Log(Exp(delta)Exp(e)) for dynamics and prior charts."""
    transport = np.eye(15)
    transport[:3, :3] = np.linalg.inv(rotation_terms(delta[:3])[1])
    return transport


def reference_seed(times, imu, prior, packets, anchors, profile, acquire_start):
    """Generate deterministic IMU reference, optionally using Profile-1 p/v/yaw."""
    initial, covariance, bias = prior
    states = np.empty((len(times), 5, 5))
    biases = np.tile(bias, (len(times), 1))
    states[0] = initial
    for k in range(1, len(times)):
        states[k] = nominal_step(states[k-1], bias, imu[k-1], times[k]-times[k-1])
    diagnostic = {"profile_requested": profile, "profile_used": False}
    if not profile:
        return states, biases, diagnostic
    selected = [p for p in packets if times[p[0]] >= acquire_start]
    rows = np.array([p[0] for p in selected], dtype=int)
    ids = np.array([p[1] for p in selected], dtype=int)
    ranges = np.array([p[2] for p in selected])
    sigmas = np.array([p[3] for p in selected])
    tau = times[rows]-times[-1]
    displacement = states[rows, :3, 4]-states[-1, :3, 4]-tau[:, None]*states[-1, :3, 3]
    estimate = solve_window(tau, displacement, anchors[ids], ranges, sigmas, method="profile")
    sensitivity = joint_seed.shape.reanchor_integral(states[:-1, :3, :3], times, times[rows])
    calibration = {"static": np.zeros(len(anchors)), "sigma": np.ones(len(anchors))}
    for anchor, sigma in zip(ids, sigmas):
        calibration["sigma"][anchor] = sigma
    values = (tau, displacement, anchors[ids], ranges, ids)
    fitted, bias_diagnostic = joint_seed.solve(values, sensitivity, "BiasPrior", "static",
        calibration, [np.r_[estimate.parameters, np.zeros(3)]])
    parameters = fitted[:7]
    corrected_bias = bias.copy()
    corrected_bias[3:] += fitted[7:10]
    biases[:] = corrected_bias
    for k in range(1, len(times)):
        states[k] = nominal_step(states[k-1], corrected_bias, imu[k-1], times[k]-times[k-1])
    rotation = yaw_rotation(parameters[6])
    tau_all = times-times[-1]
    displacement_all = states[:, :3, 4]-states[-1, :3, 4]-tau_all[:, None]*states[-1, :3, 3]
    velocity_all = states[:, :3, 3]-states[-1, :3, 3]
    states[:, :3, :3] = np.einsum("ij,njk->nik", rotation, states[:, :3, :3])
    states[:, :3, 4] = parameters[:3]+tau_all[:, None]*parameters[3:6]+displacement_all@rotation.T
    states[:, :3, 3] = parameters[3:6]+velocity_all@rotation.T
    diagnostic.update(profile_used=True, parameters=parameters.tolist(),
                      seed_objective=estimate.robust_objective, seed_elapsed_s=estimate.elapsed_s,
                      seed_yaw_information=estimate.yaw_information,
                      seed_body_accel_bias_delta=fitted[7:10].tolist(),
                      seed_bias_fit=bias_diagnostic,
                      bias_seed="Profile-1 + Gaussian body-bias seed; free during full-state IEKS")
    return states, biases, diagnostic


def conditional_replace(times, imu, prior, packets, anchors, profile, acquire_start):
    """Recompute interval posterior and return its full endpoint conditional state."""
    begun = time.perf_counter()
    states, biases, diagnostic = reference_seed(times, imu, prior, packets, anchors,
                                                profile, acquire_start)
    count = len(times)
    packet_ids = [(p[0], p[1]) for p in packets]
    assert len(set(packet_ids)) == len(packet_ids), "Duplicate packet factor"
    by_row = [[] for _ in times]
    for packet in packets:
        by_row[packet[0]].append(packet)
    # No diagonalization: every stored covariance retains all cross terms.
    filtered_cov = np.empty((count, 15, 15))
    predicted_cov = np.empty_like(filtered_cov)
    transitions = np.empty_like(filtered_cov)
    filtered_mean = np.empty((count, 15))
    predicted_mean = np.empty_like(filtered_mean)
    iteration_rows = []
    for iteration in range(PASSES):
        delta_prior = difference(states[0], biases[0], prior[0], prior[2])
        # Linearize the original prior residual Log(R_prior.T R_ref Exp(e)).
        # Its inverse residual Jacobian is Jr(-delta), not a pushed mean chart.
        transport = np.eye(15)
        transport[:3, :3] = rotation_terms(-delta_prior[:3])[1]
        covariance = symmetrize(transport @ prior[1] @ transport.T)
        mean = delta_prior.copy()
        downweighted = 0
        for k in range(count):
            if k:
                dt = float(times[k]-times[k-1])
                transition, noise = discrete_error_matrices(states[k-1], biases[k-1], imu[k-1], dt)
                predicted = nominal_step(states[k-1], biases[k-1], imu[k-1], dt)
                defect = difference(states[k], biases[k], predicted, biases[k-1])
                chart = tangent_transport(defect)
                transition = chart @ transition
                noise = chart @ noise
                covariance = symmetrize(transition @ covariance @ transition.T + (noise*NOISE)@noise.T)
                mean = defect+transition@mean
                transitions[k] = transition
            predicted_cov[k], predicted_mean[k] = covariance, mean
            for _, anchor_id, measured, sigma in by_row[k]:
                difference_p = states[k, :3, 4]-anchors[anchor_id]
                distance = max(float(np.linalg.norm(difference_p)), 1e-12)
                row = np.zeros(15)
                row[6:9] = difference_p/distance
                residual = measured-distance
                # IRLS weights belong to the fixed reference, never a second factor.
                weight = min(1., HUBER/max(abs(residual)/sigma, 1e-12))
                downweighted += int(weight < 1.)
                variance = sigma**2/weight
                projected = covariance@row
                gain = projected/(row@projected+variance)
                mean = mean+gain*(residual-row@mean)
                residual_map = np.eye(15)-np.outer(gain, row)
                covariance = symmetrize(residual_map@covariance@residual_map.T+variance*np.outer(gain, gain))
            filtered_cov[k], filtered_mean[k] = covariance, mean
        smoothed_mean = filtered_mean.copy()
        smoothed_cov = filtered_cov.copy()
        for k in range(count-2, -1, -1):
            cross = filtered_cov[k]@transitions[k+1].T
            gain = np.linalg.solve(predicted_cov[k+1], cross.T).T
            smoothed_mean[k] += gain@(smoothed_mean[k+1]-predicted_mean[k+1])
            smoothed_cov[k] = symmetrize(filtered_cov[k]+gain@(smoothed_cov[k+1]-predicted_cov[k+1])@gain.T)
        endpoint_delta = smoothed_mean[-1].copy()
        for k in range(count):
            states[k], biases[k] = inject(states[k], biases[k], smoothed_mean[k])
        reset = np.eye(15)
        reset[:3, :3] = rotation_terms(endpoint_delta[:3])[1]
        endpoint_cov = symmetrize(reset@smoothed_cov[-1]@reset.T)
        iteration_rows.append(dict(iteration=iteration+1, packet_factors=len(packets),
            downweighted=downweighted, endpoint_delta_norm=float(np.linalg.norm(endpoint_delta)),
            endpoint_rotation_delta_deg=float(np.rad2deg(np.linalg.norm(endpoint_delta[:3])))))
        if not np.isfinite(states).all() or not np.isfinite(endpoint_cov).all():
            raise FloatingPointError("Nonfinite IEKS interval posterior")
    minimum_eigenvalue = float(np.linalg.eigvalsh(endpoint_cov)[0])
    if minimum_eigenvalue <= 0:
        raise FloatingPointError(f"Endpoint covariance not SPD: {minimum_eigenvalue}")
    mapping = coordinate_map(states[-1])
    diagnostic.update(elapsed_s=time.perf_counter()-begun, iterations=iteration_rows,
        unique_packet_factors=len(packets), covariance_min_eigenvalue=minimum_eigenvalue,
        full_cross_block_norm=float(np.linalg.norm(endpoint_cov[:9, 9:])),
        prior_centre=prior[0].tolist(), prior_bias=prior[2].tolist(),
        prior_covariance=prior[1].tolist(), checkpoint_time_s=float(times[0]),
        endpoint_time_s=float(times[-1]), endpoint_bias=biases[-1].tolist(),
        endpoint_physical_covariance=endpoint_cov.tolist())
    return states[-1], mapping@endpoint_cov@mapping.T, biases[-1], diagnostic
