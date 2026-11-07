"""Verify the generalized lift and its exact constant-bias degeneracy."""

import unittest
import numpy as np
from initializer import yaw_rotation
from polynomial_bias import polynomial_lift, polynomial_residual_jacobian


class PolynomialTests(unittest.TestCase):
    """Independently generate physical positions before calling the lifted solver."""

    def test_exact_quadratic_lift(self):
        """Rich motion identifies polynomial translation and yaw without initial guesses."""
        rng=np.random.default_rng(71)
        t=np.sort(rng.uniform(-4,0,500))
        d=np.column_stack((np.sin(2*t),np.cos(1.7*t)-1,0.2*np.sin(0.7*t)))
        a=rng.uniform(-5,5,(t.size,3))
        x=np.array([1,-2,1.5,0.3,-0.2,0.1,0.03,0.07,-0.02,2.4])
        p=x[:3]+t[:,None]*x[3:6]+t[:,None]**2*x[6:9]+d@yaw_rotation(x[-1]).T
        ranges=np.linalg.norm(p-a,axis=1)
        fitted,_=polynomial_lift(t,d,a,ranges,np.full(t.size,1e-7))
        np.testing.assert_allclose(fitted,x,atol=1e-7)

    def test_quadratic_yaw_nullspace(self):
        """The projected yaw columns vanish for any quadratic inertial displacement."""
        rng=np.random.default_rng(72)
        t=np.sort(rng.uniform(-4,0,500))
        d=0.5*t[:,None]**2*np.array([0.4,-0.3,0.1])
        a=rng.uniform(-5,5,(t.size,3))
        p=np.array([1,-2,1.5])+d@yaw_rotation(2.4).T
        _,diagnostic=polynomial_lift(t,d,a,np.linalg.norm(p-a,axis=1),np.full(t.size,0.1))
        self.assertLess(np.max(diagnostic["singular"]),1e-8)

    def test_polynomial_jacobian(self):
        """Check every polynomial and yaw derivative by central differences."""
        rng=np.random.default_rng(73)
        t=np.linspace(-3,0,90)
        d,a=rng.normal(size=(90,3)),rng.normal(size=(90,3))*5
        x=np.array([1,2,1,0.2,0.3,-0.1,0.1,0.05,-0.02,2.8])
        ranges,sigma=np.full(90,3.0),np.full(90,0.1)
        _,jac=polynomial_residual_jacobian(x,t,d,a,ranges,sigma)
        for column in range(10):
            delta=np.zeros(10)
            delta[column]=1e-6
            plus=polynomial_residual_jacobian(x+delta,t,d,a,ranges,sigma)[0]
            minus=polynomial_residual_jacobian(x-delta,t,d,a,ranges,sigma)[0]
            np.testing.assert_allclose(jac[:,column],(plus-minus)/2e-6,atol=3e-8)


if __name__=="__main__":
    unittest.main()
