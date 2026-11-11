"""Paper's nine-state quadratic immersion and a matched six-state range EKF.

The real-data continuous/discrete adaptation keeps every asynchronous range.
Neither estimator has an attitude state or receives position ground truth.
"""

import numpy as np
from numba import njit

# The paper's navigation immersion adds three scalar quadratic states to p/v.
DIM = 9
# Held IMU acceleration sample noise matches the shared replay's 2 m/s^2 scale.
ACCEL_VARIANCE = 4.0
# The paper's positive Riccati forcing is retained per second in both marginals.
RICCATI_FLOOR = 0.0001


def lift_prior(mean, covariance):
    """Return exact first/second moments of a Gaussian p/v prior under the lift."""
    hessians = np.zeros((3, 6, 6))
    hessians[0, :3, :3] = np.eye(3)
    hessians[1, :3, 3:] = np.eye(3)
    hessians[1, 3:, :3] = np.eye(3)
    hessians[2, 3:, 3:] = 2*np.eye(3)
    jacobian = np.einsum("ijk,k->ij", hessians, mean)
    scalars = .5*jacobian@mean + .5*np.einsum("ijk,kj->i", hessians, covariance)
    result = np.zeros((9, 9))
    result[3:, 3:] = covariance
    result[:3, 3:] = jacobian@covariance
    result[3:, :3] = result[:3, 3:].T
    for i in range(3):
        for j in range(3):
            result[i, j] = (jacobian[i]@covariance@jacobian[j]
                + .5*np.trace(hessians[i]@covariance@hessians[j]@covariance))
    return np.r_[scalars, mean], result


@njit
def frozen_step(state, acceleration, dt):
    """Exactly integrate the affine immersion with held world acceleration."""
    u, h = acceleration, dt
    transition = np.eye(9)
    transition[0, 1], transition[0, 2] = h, .5*h*h
    transition[0, 3:6], transition[0, 6:9] = .5*h*h*u, .5*h**3*u
    transition[1, 2] = h
    transition[1, 3:6], transition[1, 6:9] = h*u, 1.5*h*h*u
    transition[2, 6:9] = 2*h*u
    transition[3:6, 6:9] = h*np.eye(3)
    uu = u@u
    offset = np.zeros(9)
    offset[:3] = np.array([.125*h**4*uu, .5*h**3*uu, h*h*uu])
    offset[3:6], offset[6:9] = .5*h*h*u, h*u
    # Differentiate the exact held-input step to share physical input noise.
    noise = np.zeros((9, 3))
    noise[0] = .5*h*h*state[3:6]+.5*h**3*state[6:9]+.25*h**4*u
    noise[1] = h*state[3:6]+1.5*h*h*state[6:9]+h**3*u
    noise[2] = 2*h*state[6:9]+2*h*h*u
    noise[3:6], noise[6:9] = .5*h*h*np.eye(3), h*np.eye(3)
    return transition@state+offset, transition, noise


@njit
def quadratic_predict(state, covariance, acceleration, dt):
    """Propagate the immersion and its matched held-input acceleration uncertainty."""
    result, transition, noise = frozen_step(state, acceleration, dt)
    posterior = transition@covariance@transition.T + ACCEL_VARIANCE*noise@noise.T
    posterior += RICCATI_FLOOR*dt*np.eye(9)
    return result, .5*(posterior+posterior.T)


@njit
def quadratic_update(state, covariance, anchor, calibrated_range, sigma):
    """Fuse one debiased half-squared range, returning its conditional Gaussian update."""
    variance = sigma*sigma
    row = np.zeros(9)
    row[0], row[3:6] = 1., -anchor
    observation = .5*(calibrated_range*calibrated_range-variance-anchor@anchor)
    difference = state[3:6]-anchor
    radio_variance = (difference@difference)*variance+.5*variance*variance
    cross = covariance@row
    gain = cross/(row@cross+radio_variance)
    result = state+gain*(observation-row@state)
    joseph = np.eye(9)-np.outer(gain, row)
    posterior = joseph@covariance@joseph.T+np.outer(gain, gain)*radio_variance
    return result, .5*(posterior+posterior.T)


@njit
def replay(times, acceleration, ranges, changed, anchors, biases, sigmas,
           start, initial, covariance, active):
    """Run same-input paired modules; preserve nonfinite failures as NaN tails."""
    state, lifted_cov = initial.copy(), covariance.copy()
    base, base_cov = initial[3:].copy(), covariance[3:, 3:].copy()
    lifted_history = np.full((len(times), 6), np.nan)
    base_history = np.full((len(times), 6), np.nan)
    latent_history = np.full((len(times), 3), np.nan)
    counts = np.zeros(2, dtype=np.int64)
    failed = np.zeros(2, dtype=np.bool_)
    failure_time = np.full(2, np.nan)
    lifted_history[start], base_history[start] = state[3:], base
    latent_history[start] = state[:3]
    for k in range(start+1, len(times)):
        h = times[k]-times[k-1]
        u = acceleration[k-1]
        if not failed[0]:
            state, lifted_cov = quadratic_predict(state, lifted_cov, u, h)
        if not failed[1]:
            transition6 = np.eye(6)
            transition6[:3, 3:] = h*np.eye(3)
            noise6 = np.empty((6, 3))
            noise6[:3], noise6[3:] = .5*h*h*np.eye(3), h*np.eye(3)
            base = transition6@base+noise6@u
            base_cov = transition6@base_cov@transition6.T+ACCEL_VARIANCE*noise6@noise6.T
            base_cov += RICCATI_FLOOR*h*np.eye(6)
        if not (15 <= times[k] < 25):
            for j in range(len(anchors)):
                if not (active[j] and changed[k, j] and ranges[k, j] > 0):
                    continue
                measured = ranges[k, j]-biases[j]
                variance = sigmas[j]**2
                if not failed[0]:
                    state, lifted_cov = quadratic_update(state, lifted_cov, anchors[j], measured, sigmas[j])
                    counts[0] += 1
                if not failed[1]:
                    difference = base[:3]-anchors[j]
                    expected = np.sqrt(difference@difference)
                    row6 = np.zeros(6)
                    row6[:3] = difference/max(expected, 1e-12)
                    cross = base_cov@row6
                    gain6 = cross/(row6@cross+variance)
                    base += gain6*(measured-expected)
                    joseph6 = np.eye(6)-np.outer(gain6, row6)
                    base_cov = joseph6@base_cov@joseph6.T+np.outer(gain6, gain6)*variance
                    counts[1] += 1
        lifted_cov = .5*(lifted_cov+lifted_cov.T)
        base_cov = .5*(base_cov+base_cov.T)
        for method in range(2):
            candidate = state if method == 0 else base
            candidate_cov = lifted_cov if method == 0 else base_cov
            if not failed[method] and (not np.all(np.isfinite(candidate))
                    or not np.all(np.isfinite(candidate_cov))):
                failed[method], failure_time[method] = True, times[k]
        if not failed[0]:
            lifted_history[k], latent_history[k] = state[3:], state[:3]
        if not failed[1]:
            base_history[k] = base
    return lifted_history, base_history, latent_history, counts, failed, failure_time


