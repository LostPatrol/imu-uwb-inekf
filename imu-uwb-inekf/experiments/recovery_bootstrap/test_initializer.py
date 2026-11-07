"""Independent algebra, derivative, and observability checks for the initializer."""

import unittest
import numpy as np
from initializer import circle_minimum, lifted_candidate, raw_residual_jacobian, project_out, yaw_rotation


class InitializerTests(unittest.TestCase):
    """Test identities against direct geometry, not stored expected implementation output."""

    def test_circle_global(self):
        """The quartic solution must dominate a dense independent angular grid."""
        rng = np.random.default_rng(41)
        angles = np.linspace(-np.pi, np.pi, 100001)
        vectors = np.column_stack((np.cos(angles), np.sin(angles)))
        for _ in range(50):
            a = rng.normal(size=(2, 2))
            h = a.T@a
            g = rng.normal(size=2)
            angle, _, _ = circle_minimum(h, g)
            u = np.array([np.cos(angle), np.sin(angle)])
            grid = np.einsum("ni,ij,nj->n", vectors, h, vectors)-2*vectors@g
            self.assertLessEqual(u@h@u-2*g@u, grid.min()+1e-9)

    def test_exact_lift(self):
        """Rich asynchronous noiseless geometry must recover translation and yaw."""
        rng = np.random.default_rng(42)
        t = np.sort(rng.uniform(-4, 0, 400))
        d = np.column_stack((np.sin(2*t), np.cos(1.7*t)-1, 0.2*np.sin(0.7*t)))
        a = rng.uniform(-5, 5, (t.size, 3))
        parameters = np.array([1, -2, 1.5, 0.3, -0.2, 0.1, 2.4])
        p = parameters[:3]+t[:, None]*parameters[3:6]+d@yaw_rotation(parameters[6]).T
        ranges = np.linalg.norm(p-a, axis=1)
        estimate, _ = lifted_candidate(t, d, a, ranges, np.full(t.size, 1e-7))
        np.testing.assert_allclose(estimate, parameters, atol=1e-8)

    def test_raw_jacobian(self):
        """All seven columns agree with central differences of the metric residual."""
        rng = np.random.default_rng(43)
        t = np.linspace(-3, 0, 80)
        d, a = rng.normal(size=(80, 3)), rng.normal(size=(80, 3))*5
        x = np.array([1, 2, 1, 0.2, 0.3, -0.1, 2.8])
        ranges, sigma = np.full(80, 3.0), np.full(80, 0.1)
        _, jac = raw_residual_jacobian(x, t, d, a, ranges, sigma)
        for column in range(7):
            delta = np.zeros(7)
            delta[column] = 1e-6
            plus = raw_residual_jacobian(x+delta, t, d, a, ranges, sigma)[0]
            minus = raw_residual_jacobian(x-delta, t, d, a, ranges, sigma)[0]
            np.testing.assert_allclose(jac[:, column], (plus-minus)/2e-6, atol=2e-8)

    def test_yaw_bias_ambiguity(self):
        """Constant acceleration yaw information disappears if acceleration bias is free."""
        rng = np.random.default_rng(44)
        t = np.linspace(-3, 0, 200)
        d = 0.5*t[:, None]**2*np.array([0.4, -0.3, 0.1])
        a = rng.uniform(-5, 5, (200, 3))
        x = np.array([1, 2, 1, 0.2, 0.3, -0.1, 1.8])
        _, jac = raw_residual_jacobian(x, t, d, a, np.ones(200), np.ones(200))
        free_bias = -0.5*t[:, None]**2*jac[:, :3]@yaw_rotation(x[6])
        projected, _ = project_out(np.column_stack((jac[:, :6], free_bias)), jac[:, 6:])
        known_bias, _ = project_out(jac[:, :6], jac[:, 6:])
        self.assertGreater(np.linalg.norm(known_bias), 0.1)
        self.assertLess(np.linalg.norm(projected), 1e-10)


if __name__ == "__main__":
    unittest.main()
