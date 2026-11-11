"""Causal range-inertial window with piecewise acceleration defects and raw TWR/TDoA.

The IMU supplies the mean trajectory; independent Gaussian acceleration increments
give its translation finite compliance without treating radio ranges as positions.
"""

import numpy as np
from scipy.optimize import least_squares

# Unit Huber innovations become linear beyond two calibrated radio standard deviations.
HUBER = 2.0
# One-second acceleration cells represent low-frequency IMU shape discrepancies.
CELL_SECONDS = 1.0
# Rate of velocity-error variance, in m^2/s^3; shared by both radio modalities.
DEFECT_PSD = 1.0
# A modest radio-only motion prior accounts for physical acceleration without IMU inputs.
RADIO_PSD = 1.0
# Each trajectory fit receives the same evaluation limit, regardless of model dimension.
MAX_NFEV = 120


def defect_basis(tau, cell_s=CELL_SECONDS):
    """Integrate unit backward acceleration increments exactly over each knot cell."""
    age = -np.asarray(tau)
    duration = max(float(age.max()), 1e-5)
    edges = np.linspace(0, duration, max(1, int(np.ceil(duration/cell_s-1e-10)))+1)
    # Division by sqrt(cell duration) turns a unit normal coefficient into white-rate forcing.
    phi = .5 * (np.maximum(age[:, None]-edges[:-1], 0)**2
                - np.maximum(age[:, None]-edges[1:], 0)**2) / np.sqrt(np.diff(edges))
    return phi


def radio_geometry(position, anchors, minus):
    """Evaluate raw distances or directed differences and their position Jacobian."""
    delta = position-anchors
    dist = np.maximum(np.linalg.norm(delta, axis=1), 1e-10)
    grad = delta/dist[:, None]
    if minus is not None:
        other = position-minus
        dist_other = np.maximum(np.linalg.norm(other, axis=1), 1e-10)
        grad = grad-other/dist_other[:, None]
        dist = dist-dist_other
    return dist, grad


def robust_rows(residual):
    """Transform only radio residuals so squared least squares equals twice Huber loss."""
    absolute = np.abs(residual)
    high = absolute > HUBER
    value = residual.copy()
    value[high] = np.sign(residual[high])*np.sqrt(2*HUBER*absolute[high]-HUBER**2)
    slope = np.ones_like(residual)
    slope[high] = HUBER/np.abs(value[high])
    return value, slope


def fit_track(tau, shape, plus, minus, measurement, sigma, start, basis=None, inertial=True):
    """Fit endpoint translation, optional yaw and Gaussian acceleration corrections."""
    block = 7 if inertial else 6
    basis = np.empty((len(tau), 0)) if basis is None else basis
    count = basis.shape[1]
    seed = np.r_[start[:block], np.zeros(3*count)]

    def parts(x, need_jac):
        """Evaluate robust raw-radio factors and unmodified quadratic process factors."""
        p = x[:3]+tau[:, None]*x[3:6]
        if inertial:
            c, s = np.cos(x[6]), np.sin(x[6])
            dx, dy = c*shape[:, 0]-s*shape[:, 1], s*shape[:, 0]+c*shape[:, 1]
            p = p+np.column_stack((dx, dy, shape[:, 2]))
        if count:
            p = p+basis@x[block:].reshape(count, 3)
        h, direction = radio_geometry(p, plus, minus)
        rr, derivative = robust_rows((h-measurement)/sigma)
        if not need_jac:
            return np.r_[rr, x[block:]]
        grad = direction*(derivative/sigma)[:, None]
        jac = np.zeros((len(tau)+3*count, len(x)))
        jac[:len(tau), :3] = grad
        jac[:len(tau), 3:6] = tau[:, None]*grad
        if inertial:
            jac[:len(tau), 6] = -grad[:, 0]*dy+grad[:, 1]*dx
        if count:
            jac[:len(tau), block:] = (basis[:, :, None]*grad[:, None, :]).reshape(len(tau), -1)
            jac[len(tau):, block:] = np.eye(3*count)
        return jac

    result = least_squares(lambda x: parts(x, False), seed, jac=lambda x: parts(x, True),
                           max_nfev=MAX_NFEV)
    return result.x, float(2*result.cost), int(result.nfev), bool(result.success)
