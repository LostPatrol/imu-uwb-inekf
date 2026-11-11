"""Perform one physical horizontal translation of a strong InEKF state.

Callers own packet order, perform at most one nonzero reset, and subsequently use
the native estimator. There is no covariance reduction or pseudo-measurement.
"""

import numpy as np

from radio_bounds import bound_xy


def single_offset(stamp, p, lever_radius, plus, minus, y, sigma):
    """Compute the frozen bound displacement from this one just-arrived packet."""
    time = np.array([stamp], dtype=float)
    position = np.asarray(p, dtype=float).reshape(1, 3)
    far = np.asarray(plus, dtype=float).reshape(1, 3)
    near = None if minus is None else np.asarray(minus, dtype=float).reshape(1, 3)
    result = bound_xy(time, position, lever_radius, time, far, near, [y], sigma, stamp)
    return result["pos"][0] - position[0]


def translate_original(estimator, delta):
    """Rebase the right-invariant chart, retaining physical covariance at this instant."""
    delta = np.asarray(delta, dtype=float).reshape(3)
    if not np.any(delta):
        return
    if delta[2] != 0 or not np.all(np.isfinite(delta)):
        raise ValueError("Expected a finite purely horizontal translation.")
    dx, dy, dz = delta
    cross = np.array([[0., -dz, dy], [dz, 0., -dx], [-dy, dx, 0.]])
    transform = np.eye(15)
    transform[6:9, 0:3] = cross
    estimator.X = estimator.X.copy()
    estimator.X[:3, 4] += delta
    estimator.P = transform @ estimator.P @ transform.T
    # Synchronize the author's stored correction prior only on an actual reset.
    estimator.X_hat, estimator.P_hat = estimator.X.copy(), estimator.P.copy()
