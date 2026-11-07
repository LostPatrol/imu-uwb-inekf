"""Regression tests for the body-frame lifted TWR audit.

The tests are deliberately short (3 s of truth) so they run in a few seconds;
the numerical claims themselves do not depend on the horizon.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ESTIMATION_DIR = Path(__file__).resolve().parents[1] / "scripts" / "estimation"
if str(ESTIMATION_DIR) not in sys.path:
    sys.path.insert(0, str(ESTIMATION_DIR))

import clif_audit as audit  # noqa: E402


class LifterdAuditTests(unittest.TestCase):
    """Checks the five structural claims and the integrator sanity."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.truth = audit.build_truth(audit.Scenario(duration_s=3.0))
        cls.quad = audit.quadratic_states(cls.truth["z_anchor"], cls.truth["z_velocity"], cls.truth["z_gravity"])

    def test_closed_form_matches_finite_difference(self) -> None:
        report = audit.check_closed_form(self.truth, self.quad)
        self.assertTrue(report["pass"], msg=str(report))

    def test_angular_rate_terms_cancel(self) -> None:
        report = audit.check_omega_cancellation(self.truth, self.quad)
        self.assertTrue(report["pass"], msg=str(report))
        # At least one cross state must carry a macroscopic pair term, otherwise
        # the cancellation claim would be vacuous.
        self.assertGreater(report["s01"]["omega_term_magnitude"], 1e-3)

    def test_quadratic_subsystem_is_constant(self) -> None:
        report = audit.check_constant_subsystem()
        self.assertTrue(report["pass"], msg=str(report))

    def test_propagators_reproduce_truth_without_perturbation(self) -> None:
        report = audit.check_zero_perturbation(self.truth, audit.COMPARISON_DT)
        self.assertTrue(report["pass"], msg=str(report))

    def test_initial_attitude_error_cannot_reach_body_lifting(self) -> None:
        report = audit.check_attitude_immunity(self.truth)
        self.assertTrue(report["pass"], msg=str(report["attitude_error_only"]))
        self.assertLess(report["attitude_error_only"]["body_over_world_rmse"], 1e-3)

    def test_exact_propagation_keeps_the_manifold(self) -> None:
        report = audit.check_redundancy_drift(self.truth)
        self.assertTrue(report["pass"], msg=str(report))
        # Euler is expected to diverge; compare it with RK4 rather than with a
        # fixed bound, because the residual grows quickly with the horizon.
        self.assertGreater(report["euler_peak_residual_m2"], 1000.0 * max(report["rk4_final_residual_m2"], 1e-12))


if __name__ == "__main__":
    unittest.main(verbosity=2)
