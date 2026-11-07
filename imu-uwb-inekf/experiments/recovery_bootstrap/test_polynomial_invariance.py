"""Independent checks of polynomial-contamination invariance and yaw ambiguity.

The lifted objective is reconstructed with redundant pairwise polynomial
products, rather than the implementation's compact monomial basis.
"""

import unittest

import numpy as np

from initializer import HUBER_SCALE, project_out, yaw_rotation
from polynomial_bias import polynomial_lift, polynomial_residual_jacobian

# Fixed trial count keeps the property checks fast while testing every K <= 2.
RANDOM_TRIALS = 15
# Dense enough to verify the entire circular objective, not only its minimizer.
PROFILE_ANGLES = np.linspace(-np.pi, np.pi, 137, endpoint=False)


def independent_profile(t, d, anchors, ranges, sigma, degree):
    """Eliminate redundant quadratic nuisances by least squares at every yaw."""
    basis = np.column_stack([t**j for j in range(degree + 1)])
    # Pairwise products deliberately retain algebraically duplicate columns.
    products = [basis[:, j] * basis[:, k]
                for j in range(degree + 1) for k in range(j, degree + 1)]
    cross_terms = [basis[:, j] * d[:, axis]
                   for j in range(degree + 1) for axis in (0, 1)]
    nuisance = np.column_stack(products + cross_terms)
    translation_blocks = []
    for j in range(degree + 1):
        block = -basis[:, j:j + 1] * anchors
        block[:, 2] += basis[:, j] * d[:, 2]
        translation_blocks.append(block)
    linear = np.column_stack([nuisance] + translation_blocks)
    response = (0.5 * (ranges**2 - sigma**2 - np.sum(anchors**2, axis=1)
                       - np.sum(d**2, axis=1)) + anchors[:, 2] * d[:, 2])
    circular = np.column_stack((
        -(anchors[:, 0] * d[:, 0] + anchors[:, 1] * d[:, 1]),
        anchors[:, 0] * d[:, 1] - anchors[:, 1] * d[:, 0]))
    # These weights depend only on measured ranges and their supplied scales.
    weight = 1.0 / np.sqrt(np.maximum(ranges, sigma)**2 * sigma**2 + 0.5 * sigma**4)
    vectors = np.array([np.cos(PROFILE_ANGLES), np.sin(PROFILE_ANGLES)])
    target = (response[:, None] - circular @ vectors) * weight[:, None]
    weighted = linear * weight[:, None]
    solution = np.linalg.lstsq(weighted, target, rcond=1e-12)[0]
    residual = target - weighted @ solution
    return np.sum(residual**2, axis=0)


def data_yaw_information(jacobian, residual):
    """Remove all polynomial coefficients from the Huber IRLS information."""
    weights = np.minimum(1.0, HUBER_SCALE / np.maximum(np.abs(residual), 1e-12))
    weighted = jacobian * np.sqrt(weights[:, None])
    remaining, _ = project_out(weighted[:, :-1], weighted[:, -1:])
    return float(np.sum(remaining**2))


class PolynomialInvarianceTests(unittest.TestCase):
    """Check exact model equivalence for arbitrary data, not just fitted truth."""

    def test_raw_residual_and_schur_invariance(self):
        """Polynomial contamination preserves residuals and profiled yaw information."""
        rng = np.random.default_rng(9041)
        for degree in range(3):
            for _ in range(RANDOM_TRIALS):
                t = np.sort(rng.uniform(-2, 0, 140))
                basis = np.column_stack([t**j for j in range(degree + 1)])
                d = rng.normal(size=(t.size, 3))
                anchors = rng.uniform(-4, 4, (t.size, 3))
                coefficients = rng.normal(size=(degree + 1, 3))
                contamination = rng.normal(scale=0.3, size=(degree + 1, 3))
                yaw = rng.uniform(-np.pi, np.pi)
                rotation = yaw_rotation(yaw)
                ranges = rng.uniform(1, 8, t.size)
                sigma = rng.uniform(0.08, 0.4, t.size)
                original = np.r_[coefficients.ravel(), yaw]
                transformed = np.r_[(coefficients - contamination @ rotation.T).ravel(), yaw]
                polluted = d + basis @ contamination
                r0, j0 = polynomial_residual_jacobian(original, t, d, anchors, ranges, sigma, degree)
                r1, j1 = polynomial_residual_jacobian(transformed, t, polluted, anchors, ranges, sigma, degree)
                np.testing.assert_allclose(r0, r1, atol=1e-11, rtol=1e-12)
                np.testing.assert_allclose(data_yaw_information(j0, r0),
                                           data_yaw_information(j1, r1), rtol=2e-11, atol=1e-9)

    def test_entire_lifted_circle_objective(self):
        """Every yaw cost and the implemented candidate survive polynomial contamination."""
        rng = np.random.default_rng(9042)
        for degree in range(3):
            for _ in range(RANDOM_TRIALS):
                t = np.sort(rng.uniform(-2, 0, 140))
                basis = np.column_stack([t**j for j in range(degree + 1)])
                d = rng.normal(size=(t.size, 3))
                anchors = rng.uniform(-4, 4, (t.size, 3))
                coefficients = rng.normal(size=(degree + 1, 3))
                contamination = rng.normal(scale=0.3, size=(degree + 1, 3))
                yaw = rng.uniform(-np.pi, np.pi)
                positions = basis @ coefficients + d @ yaw_rotation(yaw).T
                sigma = rng.uniform(0.08, 0.4, t.size)
                ranges = np.linalg.norm(positions - anchors, axis=1) + rng.normal(0, sigma)
                polluted = d + basis @ contamination
                original_cost = independent_profile(t, d, anchors, ranges, sigma, degree)
                polluted_cost = independent_profile(t, polluted, anchors, ranges, sigma, degree)
                np.testing.assert_allclose(original_cost, polluted_cost, rtol=2e-10, atol=1e-8)
                x0, _ = polynomial_lift(t, d, anchors, ranges, sigma, degree)
                x1, _ = polynomial_lift(t, polluted, anchors, ranges, sigma, degree)
                difference = np.arctan2(np.sin(x1[-1] - x0[-1]), np.cos(x1[-1] - x0[-1]))
                self.assertLess(abs(difference), 1e-8)

    def test_horizontal_polynomial_exact_yaw_gauge(self):
        """Polynomial horizontal displacement hides yaw even with nonpolynomial altitude."""
        rng = np.random.default_rng(9043)
        t = np.linspace(-2, 0, 130)
        for degree in range(3):
            basis = np.column_stack([t**j for j in range(degree + 1)])
            horizontal = rng.normal(size=(degree + 1, 3))
            horizontal[:, 2] = 0.0
            d = basis @ horizontal
            d[:, 2] = np.sin(3.1 * t)
            coefficients = rng.normal(size=(degree + 1, 3))
            yaw0, yaw1 = rng.uniform(-np.pi, np.pi, 2)
            r0, r1 = yaw_rotation(yaw0), yaw_rotation(yaw1)
            transformed = coefficients + horizontal @ (r0 - r1).T
            original_positions = basis @ coefficients + d @ r0.T
            equivalent_positions = basis @ transformed + d @ r1.T
            np.testing.assert_allclose(original_positions, equivalent_positions, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
