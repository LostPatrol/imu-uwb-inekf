"""Tests for the CLIF core: exact chain discretisation and driving Jacobians.

The Jacobian test perturbs the invariant error directly, so a sign or frame
mistake in ``anchor_driving_jacobian`` cannot pass unnoticed.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from scipy import linalg

ESTIMATION_DIR = Path(__file__).resolve().parents[1] / "scripts" / "estimation"
if str(ESTIMATION_DIR) not in sys.path:
    sys.path.insert(0, str(ESTIMATION_DIR))

import clif_core as core  # noqa: E402
from inekf import SE3_2_exp  # noqa: E402


ANCHORS = np.array(
    [
        [-5.0, -5.0, 0.0],
        [5.0, -5.0, 0.0],
        [-5.0, 5.0, 0.0],
        [5.0, 5.0, 0.0],
        [-5.0, -5.0, 3.0],
        [5.0, -5.0, 3.0],
        [-5.0, 5.0, 3.0],
        [5.0, 5.0, 3.0],
    ]
)


def nominal_state() -> np.ndarray:
    """A nominal SE_2(3) state with a non-trivial attitude and position."""
    yaw, pitch = np.radians(35.0), np.radians(-7.0)
    rz = np.array([[np.cos(yaw), -np.sin(yaw), 0.0], [np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[np.cos(pitch), 0.0, np.sin(pitch)], [0.0, 1.0, 0.0], [-np.sin(pitch), 0.0, np.cos(pitch)]])
    state = np.zeros((5, 5))
    state[3, 3] = 1.0
    state[4, 4] = 1.0
    state[:3, :3] = rz @ ry
    state[:3, 3] = np.array([1.1, -0.4, 0.35])
    state[:3, 4] = np.array([0.8, 0.3, 1.6])
    return state


class ChainDiscretisationTests(unittest.TestCase):
    """The nilpotent series must equal the matrix exponential exactly."""

    def test_transition_matches_expm(self) -> None:
        for dt in (1e-4, 1e-3, 5e-3, 0.02):
            np.testing.assert_allclose(core.nilpotent_transition(dt), linalg.expm(core.CHAIN * dt), atol=1e-15)

    def test_input_map_matches_quadrature(self) -> None:
        dt = 2e-3
        steps = 2000
        reference = np.zeros((core.QUADRATIC_COUNT, core.QUADRATIC_COUNT))
        for k in range(steps):
            tau = dt * (k + 0.5) / steps
            reference += linalg.expm(core.CHAIN * (dt - tau)) * (dt / steps)
        np.testing.assert_allclose(core.nilpotent_input_map(dt), reference, atol=1e-12)

    def test_exact_step_matches_fine_rk4(self) -> None:
        state = nominal_state()
        quantities = core.lifted_quantities(state, np.array([0.3, -0.2, 9.9]), ANCHORS)
        quadratic = core.quadratic_from_quantities(quantities)
        dt = 4e-3

        exact = core.exact_lifted_step(quadratic, quantities, dt)

        # Independent RK4 reference on the continuous chain.
        reference = core.stacked_quadratic(quadratic)
        driving = np.zeros_like(reference)
        inputs = core.quadratic_inputs(quantities)
        driving[:, core.CHAIN_INDEX["s01"]] = inputs["u01"]
        driving[:, core.CHAIN_INDEX["s11"]] = inputs["u11"]
        driving[:, core.CHAIN_INDEX["s12"]] = inputs["u12"]
        substeps = 50
        h = dt / substeps
        for _ in range(substeps):
            rate = lambda value: value @ core.CHAIN.T + driving  # noqa: E731
            k1 = rate(reference)
            k2 = rate(reference + 0.5 * h * k1)
            k3 = rate(reference + 0.5 * h * k2)
            k4 = rate(reference + h * k3)
            reference = reference + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        expected = core.unstack_quadratic(reference)

        for name in ("s00", "s01", "s02"):
            np.testing.assert_allclose(exact[name], expected[name], rtol=1e-10, atol=1e-12)
        for name in ("s11", "s12", "s22"):
            self.assertAlmostEqual(exact[name], expected[name], places=10)


class DrivingJacobianTests(unittest.TestCase):
    """Central differences of the composed map must reproduce d u / d xi."""

    def _perturbed_inputs(self, xi: np.ndarray, rotation_bias: np.ndarray, force: np.ndarray) -> np.ndarray:
        state = SE3_2_exp(xi[:9]) @ nominal_state()
        bias = rotation_bias + np.concatenate([xi[9:12], xi[12:15]])
        quantities = core.lifted_quantities(state, force - bias[3:6], ANCHORS)
        inputs = core.quadratic_inputs(quantities)
        per_anchor = np.zeros((len(ANCHORS), core.QUADRATIC_COUNT))
        per_anchor[:, core.CHAIN_INDEX["s01"]] = inputs["u01"]
        per_anchor[:, core.CHAIN_INDEX["s11"]] = inputs["u11"]
        per_anchor[:, core.CHAIN_INDEX["s12"]] = inputs["u12"]
        return per_anchor

    def test_jacobian_matches_finite_difference(self) -> None:
        state = nominal_state()
        bias = np.array([0.01, -0.02, 0.03, 0.10, -0.05, 0.02])
        force = np.array([0.4, -0.3, 9.7])
        quantities = core.lifted_quantities(state, force - bias[3:6], ANCHORS)
        analytic = core.anchor_driving_jacobian(quantities, state[:3, :3], ANCHORS)

        baseline = self._perturbed_inputs(np.zeros(15), bias, force)
        epsilon = 1e-6
        numeric = np.zeros_like(analytic)
        for index in range(core.ERROR_DIM):
            step = np.zeros(15)
            step[index] = epsilon
            plus = self._perturbed_inputs(step, bias, force)
            minus = self._perturbed_inputs(-step, bias, force)
            numeric[:, :, index] = (plus - minus) / (2.0 * epsilon)

        np.testing.assert_allclose(analytic, numeric, rtol=2e-5, atol=2e-6)
        # Guard against a vacuous pass when every entry is tiny.
        self.assertGreater(float(np.max(np.abs(analytic))), 1.0)
        del baseline


class LiftedInvarianceTests(unittest.TestCase):
    """The measurement row itself must not see the attitude error."""

    def test_half_squared_range_row_has_no_state_dependence(self) -> None:
        state = nominal_state()
        quantities = core.lifted_quantities(state, np.array([0.2, 0.1, 9.8]), ANCHORS)
        quadratic = core.quadratic_from_quantities(quantities)
        # A perturbation of xi leaves the predicted half-squared range
        # unchanged at first order because only s00 is read.
        predicted = core.half_squared_range(quadratic, 2)
        self.assertAlmostEqual(predicted, 0.5 * float(np.sum(quantities["z_anchor"][2] ** 2)), places=12)

    def test_chain_is_nilpotent_of_index_five(self) -> None:
        self.assertFalse(np.allclose(np.linalg.matrix_power(core.CHAIN, 4), 0.0))
        self.assertTrue(np.allclose(np.linalg.matrix_power(core.CHAIN, 5), 0.0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
