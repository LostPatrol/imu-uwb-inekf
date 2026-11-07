"""Numerical checks for the conventional 15-state TWR ESKF.

Tests target physical propagation, held-sample noise, error reset, and robustness.
They deliberately avoid dataset replay and redundant implementation fixtures.
"""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

# Permit direct unittest discovery without changing the project's import paths.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eskf15 import (
    DEG_TO_RAD, GRAVITY_MAGNITUDE, ESKF15, discrete_error_matrices,
    nominal_step, rotation_terms,
)


def make_state() -> np.ndarray:
    """Return a nontrivial nominal state for geometric derivative checks."""
    state = np.eye(5)
    state[:3, :3] = Rotation.from_rotvec([0.3, -0.2, 0.7]).as_matrix()
    state[:3, 3] = [0.6, -0.4, 0.2]
    state[:3, 4] = [1.5, -1.0, 2.0]
    return state


def perturb(state: np.ndarray, bias: np.ndarray, error: np.ndarray):
    """Apply a physical body-right attitude/additive translation perturbation."""
    result = state.copy()
    result[:3, :3] = state[:3, :3] @ Rotation.from_rotvec(error[:3]).as_matrix()
    result[:3, 3] += error[3:6]
    result[:3, 4] += error[6:9]
    return result, bias + error[9:15]


def physical_error(state: np.ndarray, bias: np.ndarray, reference: np.ndarray, reference_bias: np.ndarray):
    """Recover the conventional error using an independent SO(3) logarithm."""
    return np.concatenate((
        Rotation.from_matrix(reference[:3, :3].T @ state[:3, :3]).as_rotvec(),
        state[:3, 3] - reference[:3, 3],
        state[:3, 4] - reference[:3, 4],
        bias - reference_bias,
    ))


class ESKF15Tests(unittest.TestCase):
    """Meaningful first-order consistency and repeated-update checks."""

    def test_discrete_jacobians_against_physical_finite_differences(self):
        """All state and noise columns agree with the actual frozen-R step."""
        state = make_state()
        bias = np.array([0.01, -0.03, 0.02, 0.1, -0.15, 0.05])
        imu = np.array([0.12, -0.08, 1.02, 55.0, -35.0, 28.0])
        dt = 0.13
        transition, noise = discrete_error_matrices(state, bias, imu, dt)
        reference = nominal_step(state, bias, imu, dt)
        epsilon = 1e-6
        state_fd = np.zeros((15, 15))
        for column in range(15):
            error = np.eye(15)[column] * epsilon
            plus_state, plus_bias = perturb(state, bias, error)
            minus_state, minus_bias = perturb(state, bias, -error)
            plus = physical_error(nominal_step(plus_state, plus_bias, imu, dt), plus_bias, reference, bias)
            minus = physical_error(nominal_step(minus_state, minus_bias, imu, dt), minus_bias, reference, bias)
            state_fd[:, column] = (plus - minus) / (2.0 * epsilon)
        np.testing.assert_allclose(transition, state_fd, atol=4e-9, rtol=4e-8)

        noise_fd = np.zeros((15, 12))
        for column in range(12):
            outputs = []
            for sign in (1.0, -1.0):
                sample_noise = np.eye(12)[column] * epsilon * sign
                # A measured noise sample is subtracted to obtain the true input.
                ideal_imu = imu.copy()
                ideal_imu[3:] -= sample_noise[:3] / DEG_TO_RAD
                ideal_imu[:3] -= sample_noise[3:6] / GRAVITY_MAGNITUDE
                true_bias = bias + sample_noise[6:] * dt
                true_state = nominal_step(state, bias, ideal_imu, dt)
                outputs.append(physical_error(true_state, true_bias, reference, bias))
            noise_fd[:, column] = (outputs[0] - outputs[1]) / (2.0 * epsilon)
        np.testing.assert_allclose(noise, noise_fd, atol=4e-9, rtol=4e-8)

    def test_measurement_and_attitude_reset_jacobians(self):
        """TWR has only position sensitivity; a finite attitude reset is nonidentity."""
        state = make_state()
        bias = np.zeros(6)
        estimator = ESKF15(state, np.eye(15), bias, range_bias=0.0)
        anchor = np.array([-3.0, 2.0, 0.5])
        _, row = estimator.range_model(anchor)
        epsilon = 1e-6
        numerical = np.zeros(15)
        for column in range(15):
            error = np.eye(15)[column] * epsilon
            plus, _ = perturb(state, bias, error)
            minus, _ = perturb(state, bias, -error)
            numerical[column] = (
                np.linalg.norm(plus[:3, 4] - anchor) - np.linalg.norm(minus[:3, 4] - anchor)
            ) / (2.0 * epsilon)
        np.testing.assert_allclose(row[0], numerical, atol=2e-9)

        delta = np.array([0.8, -0.4, 0.6] + [0.0] * 12)
        rotation_inverse = Rotation.from_rotvec(-delta[:3]).as_matrix()
        reset_fd = np.zeros((3, 3))
        for column in range(3):
            increment = np.eye(3)[column] * epsilon
            plus = Rotation.from_matrix(rotation_inverse @ Rotation.from_rotvec(delta[:3] + increment).as_matrix()).as_rotvec()
            minus = Rotation.from_matrix(rotation_inverse @ Rotation.from_rotvec(delta[:3] - increment).as_matrix()).as_rotvec()
            reset_fd[:, column] = (plus - minus) / (2.0 * epsilon)
        reset = estimator.reset_jacobian(delta)
        np.testing.assert_allclose(reset[:3, :3], reset_fd, atol=2e-9)
        self.assertGreater(np.linalg.norm(reset[:3, :3] - np.eye(3)), 0.5)

    def test_joseph_update_injects_bias_and_resets_covariance(self):
        """A cross-correlated range update corrects all states and preserves PSD."""
        generator = np.random.default_rng(24)
        factor = generator.normal(size=(15, 15))
        covariance = factor @ factor.T * 0.05 + np.eye(15) * 0.01
        state = make_state()
        bias = np.zeros(6)
        estimator = ESKF15(state, covariance, bias, range_sigma=0.2, range_bias=0.0)
        anchor = np.array([-3.0, 2.0, 0.5])
        prediction, row = estimator.range_model(anchor)
        innovation = 0.7
        gain = covariance @ row.T / float((row @ covariance @ row.T).item() + 0.2**2)
        delta = gain[:, 0] * innovation
        expected_state, expected_bias = perturb(state, bias, delta)
        residual_map = np.eye(15) - gain @ row
        posterior = residual_map @ covariance @ residual_map.T + 0.2**2 * gain @ gain.T
        reset = np.eye(15)
        reset[:3, :3] = rotation_terms(delta[:3])[1]
        expected_covariance = reset @ posterior @ reset.T
        self.assertTrue(estimator.correct_twr(anchor, prediction + innovation))
        np.testing.assert_allclose(estimator.X, expected_state, atol=1e-13)
        np.testing.assert_allclose(estimator.bias, expected_bias, atol=1e-13)
        np.testing.assert_allclose(estimator.P, expected_covariance, atol=1e-13)
        self.assertGreater(np.linalg.norm(estimator.bias), 0.01)
        self.assertGreater(np.linalg.eigvalsh(estimator.P).min(), 0.0)

    def test_robust_policy_and_per_anchor_invalid_measurements(self):
        """Warm-up precedes gating/Huber and per-anchor parameters are enforced."""
        estimator = ESKF15(
            make_state(), np.eye(15) * 1e-6, np.zeros(6),
            range_sigma=[1.0, 2.0], range_bias=[0.3, -0.4],
            gate_threshold=12.25, gate_warmup_updates=1, huber_threshold=2.0,
        )
        anchor = np.array([-3.0, 2.0, 0.5])
        prediction, _ = estimator.range_model(anchor)
        self.assertTrue(estimator.correct_twr(anchor, prediction + 10.3, anchor_index=0))
        self.assertEqual(estimator.stats.downweighted, 0)
        prediction, _ = estimator.range_model(anchor)
        self.assertTrue(estimator.correct_twr(anchor, prediction + 3.3, anchor_index=0))
        self.assertEqual(estimator.stats.downweighted, 1)
        self.assertAlmostEqual(estimator.stats.weight_sum, 1.0 + 2.0 / np.sqrt(estimator.last_nis))
        previous_state = estimator.X.copy()
        self.assertFalse(estimator.correct_twr(anchor, prediction + 10.3, anchor_index=0))
        np.testing.assert_array_equal(estimator.X, previous_state)
        self.assertFalse(estimator.correct_twr(anchor, prediction, anchor_index=10))
        self.assertFalse(estimator.correct_twr(anchor, np.nan, anchor_index=0))
        self.assertFalse(estimator.correct_twr(anchor, prediction))
        self.assertEqual(estimator.stats.invalid, 3)
        self.assertEqual(estimator.stats.per_anchor[0], {"accepted": 2, "rejected": 1, "downweighted": 1})

    def test_repeated_stationary_updates_preserve_covariance_and_pose(self):
        """A bounded IMU/TWR sequence remains finite, stationary, and PSD."""
        state = np.eye(5)
        state[:3, 4] = [0.7, -0.4, 1.2]
        estimator = ESKF15(state, np.eye(15) * 0.1, np.zeros(6), range_bias=0.0)
        anchors = np.array([[-3, -2, 0], [3, -2, 1], [-2, 3, 2], [2, 3, 0]])
        for index in range(240):
            estimator.predict([0.0, 0.0, 1.0, 0.0, 0.0, 0.0], 0.01)
            anchor = anchors[index % len(anchors)]
            estimator.correct_twr(anchor, np.linalg.norm(state[:3, 4] - anchor), index % len(anchors))
        np.testing.assert_allclose(estimator.X, state, atol=1e-13)
        np.testing.assert_allclose(estimator.bias, np.zeros(6), atol=1e-13)
        np.testing.assert_allclose(estimator.P, estimator.P.T, atol=1e-13)
        self.assertGreaterEqual(np.linalg.eigvalsh(estimator.P).min(), -1e-12)
        self.assertEqual(estimator.stats.accepted, 240)

    def test_optional_ground_projection_is_differentiated(self):
        """A requested ground clamp projects translation and its uncertainty."""
        state = make_state()
        state[2, 4] = -0.1
        estimator = ESKF15(state, np.eye(15), np.zeros(6), ground_clamp=True)
        estimator.predict([0, 0, 1, 0, 0, 0], 0.01)
        self.assertEqual(estimator.X[2, 4], 0.0)
        np.testing.assert_array_equal(estimator.X[:3, 3], np.zeros(3))
        np.testing.assert_array_equal(estimator.P[3:6], np.zeros((3, 15)))
        np.testing.assert_array_equal(estimator.P[8], np.zeros(15))


if __name__ == "__main__":
    unittest.main()
