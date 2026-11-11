"""Full-attitude acquisition from one moving window, without a prior tilt or yaw.

A radio-only trajectory supplies an orthogonal Procrustes seed after eliminating
affine translation. Raw radio factors are used only once in the final objective.
"""

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from window_dynamics import MAX_NFEV, defect_basis, fit_track, radio_geometry, robust_rows

# Survey coordinates use metres and an upward z axis; the gravity vector is known.
GRAVITY = np.array([0., 0., -9.81])
# The optional isotropic body bias prior is in m/s^2, fixed before real-window scoring.
BIAS_SIGMA = .10


def local_shape(t, acc, gyro, rows, lever=None):
    """Return endpoint-reanchored force, bias integral, radio-time Q and terminal Q (SI)."""
    dt = np.diff(t)
    dq = Rotation.from_rotvec(dt[:, None]*gyro[:-1]).as_matrix()
    q = np.empty((len(t), 3, 3)); q[0] = np.eye(3)
    for k, delta in enumerate(dq):
        q[k+1] = q[k]@delta

    def integral(field):
        """Double-integrate held inputs and remove endpoint position and velocity."""
        h = dt.reshape((-1,)+(1,)*(field.ndim-1))
        zero = np.zeros_like(field[:1])
        vel = np.concatenate((zero, np.cumsum(h*field[:-1], axis=0)))
        disp = np.concatenate((zero, np.cumsum(h*vel[:-1]+.5*h*h*field[:-1], axis=0)))
        tau = (t[rows]-t[-1]).reshape((-1,)+(1,)*(field.ndim-1))
        return disp[rows]-disp[-1]-tau*vel[-1]

    body_force = np.einsum('nij,nj->ni', q, acc)
    force = integral(body_force)
    if lever is not None:
        force += q[rows]@lever
    return force, integral(q), q[rows], q[-1]


def radio_seed_path(tau, plus, minus, measurement, sigma, cv):
    """Fit one radio-only acceleration trajectory solely to initialize the nonlinear solve."""
    phi = defect_basis(tau)
    x, _, _, _ = fit_track(tau, None, plus, minus, measurement, sigma, cv, phi, False)
    return x[:3]+tau[:, None]*x[3:6]+phi@x[6:].reshape(-1, 3)


def wahba_seed(tau, force, path_seed, weights=None):
    """Estimate a proper rotation after removing unknown position and velocity from both sides."""
    weight = np.ones((len(tau), 1)) if weights is None else np.sqrt(weights)[:, None]
    affine = np.column_stack((np.ones(len(tau)), tau))*weight
    basis = np.linalg.qr(affine, mode='reduced')[0]
    x = force*weight
    y = (path_seed-.5*tau[:, None]**2*GRAVITY)*weight
    xc = x-basis@(basis.T@x); yc = y-basis@(basis.T@y)
    u, singular, vt = np.linalg.svd(yc.T@xc)
    handedness = 1. if np.linalg.det(u@vt) >= 0 else -1.
    return (u*np.array([1., 1., handedness]))@vt, singular


def right_jacobian(theta):
    """Differentiate the SO(3) exponential using its continuous small-angle limit."""
    x,y,z = theta
    skew = np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])
    n = np.linalg.norm(theta)
    if n < 1e-5:
        return np.eye(3)-.5*skew+(skew@skew)/6
    return np.eye(3)-(1-np.cos(n))/n**2*skew+(n-np.sin(n))/n**3*(skew@skew)


def solve_attitude(tau, force, plus, minus, measurement, sigma, path_seed,
                   bias_sensitivity=None, seed_rotation=None, bias_sigma=BIAS_SIGMA):
    """Fit p_T,v_T,C on SO(3), optionally a constant body bias; C maps initial body to world."""
    rotation, singular = wahba_seed(tau, force, path_seed)
    if seed_rotation is not None:
        rotation = seed_rotation
    affine = np.column_stack((np.ones(len(tau)), tau))
    gravity = .5*tau[:, None]**2*GRAVITY
    translation = np.linalg.lstsq(affine, path_seed-gravity-force@rotation.T, rcond=None)[0]
    with_bias = bias_sensitivity is not None
    seed = np.r_[translation.ravel(), np.zeros(6 if with_bias else 3)]

    def evaluate(x, jac=False):
        """Evaluate original radio readings and differentiate rotation in a right tangent chart."""
        c = rotation@Rotation.from_rotvec(x[6:9]).as_matrix()
        local = force-(bias_sensitivity@x[9:]) if with_bias else force
        pos = x[:3]+tau[:, None]*x[3:6]+gravity+local@c.T
        predicted, tangent = radio_geometry(pos, plus, minus)
        r, deriv = robust_rows((predicted-measurement)/sigma)
        if not jac:
            return np.r_[r, x[9:]/bias_sigma] if with_bias else r
        grad = tangent*(deriv/sigma)[:, None]
        local_grad = grad@c
        rotblock = np.cross(local, local_grad)@right_jacobian(x[6:9])
        j = np.column_stack((grad, tau[:, None]*grad, rotblock))
        if with_bias:
            biasblock = -np.einsum('ni,nij->nj', local_grad, bias_sensitivity)
            j = np.vstack((np.column_stack((j, biasblock)),
                           np.column_stack((np.zeros((3, 9)), np.eye(3)/bias_sigma))))
        return j

    fit = least_squares(evaluate, seed, jac=lambda x:evaluate(x, True), max_nfev=MAX_NFEV)
    return dict(x=fit.x, C=rotation@Rotation.from_rotvec(fit.x[6:9]).as_matrix(),
                C0=rotation, singular=singular, cost=2*fit.cost, nfev=fit.nfev, success=fit.success)
