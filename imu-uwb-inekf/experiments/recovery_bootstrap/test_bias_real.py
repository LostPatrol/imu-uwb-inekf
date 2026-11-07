"""Finite-difference and held-sample integration checks for rotating bias correction."""

import unittest
import numpy as np
from bias_real import residual_jacobian,bias_preintegration,mixed_loss,solve_bias_window
from initializer import yaw_rotation


class RotatingBiasTests(unittest.TestCase):
    """Verify exact local derivatives and endpoint bias sensitivity independently."""

    def test_range_jacobian(self):
        """All ten Jacobian columns match raw geometric finite differences."""
        rng=np.random.default_rng(112)
        t=np.linspace(-3,0,100)
        d,a=rng.normal(size=(100,3)),rng.uniform(-5,5,(100,3))
        b=rng.normal(size=(100,3,3))
        x=np.r_[1,2,1,0.3,-0.4,0.1,2.3,0.03,-0.04,0.02]
        ranges,sigma=np.full(100,3.),np.full(100,0.1)
        _,jac=residual_jacobian(x,t,d,a,ranges,sigma,b)
        for column in range(10):
            delta=np.zeros(10)
            delta[column]=1e-6
            plus=residual_jacobian(x+delta,t,d,a,ranges,sigma,b)[0]
            minus=residual_jacobian(x-delta,t,d,a,ranges,sigma,b)[0]
            np.testing.assert_allclose(jac[:,column],(plus-minus)/2e-6,atol=4e-8)

    def test_rotating_bias_integral(self):
        """Directly propagated bias error equals the matrix double integral."""
        t=np.linspace(0,3,101)
        rotations=np.array([yaw_rotation(0.7*stamp) for stamp in t])
        first,second=bias_preintegration(t,rotations)
        bias=np.array([0.1,-0.2,0.03])
        velocity=np.zeros(3)
        position=np.zeros(3)
        for index in range(1,t.size):
            dt=t[index]-t[index-1]
            acceleration=rotations[index-1]@bias
            position+=velocity*dt+0.5*acceleration*dt**2
            velocity+=acceleration*dt
        np.testing.assert_allclose(position,second[-1]@bias,atol=1e-13)
        np.testing.assert_allclose(velocity,first[-1]@bias,atol=1e-13)

    def test_prior_is_quadratic(self):
        """Large bias-prior residuals are never silently Huber-downweighted."""
        result=mixed_loss(np.array([100.,100.]),1)
        self.assertEqual(result[0,1],100.)
        self.assertEqual(result[1,1],1.)
        self.assertEqual(result[2,1],0.)
        self.assertLess(result[1,0],1.)

    def test_unobserved_translation_retains_variance(self):
        """Single-line range geometry must not produce zero orthogonal position variance."""
        t=np.linspace(-2,0,60)
        d=np.zeros((t.size,3))
        anchors=np.tile(np.array([10.,0.,0.]),(t.size,1))
        sensitivity=0.5*t[:,None,None]**2*np.eye(3)[None,:,:]
        fit=solve_bias_window(t,d,anchors,np.full(t.size,10.),np.full(t.size,0.1),sensitivity)
        self.assertGreater(np.linalg.eigvalsh(fit.covariance[:3,:3]).max(),63.9)
        self.assertAlmostEqual(fit.covariance[6,6],np.pi**2/3,places=8)
        self.assertLess(fit.yaw_information_data,1e-12)


if __name__=="__main__":
    unittest.main()
