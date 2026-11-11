"""Decide whether radio and IMU support changing an existing heading about gravity.

Fit a periodic profile in raw range coordinates, then test unseen time blocks.
No attitude, position ground truth, flight labels, or device-specific headings enter.
"""

from pathlib import Path
import sys

import numpy as np
from scipy.optimize import minimize_scalar

# The raw TWR/TDoA model and the radio-only trajectory are shared with the prior work.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'radio_inertial_models'))
from window_dynamics import defect_basis, fit_track, radio_geometry

# Survey coordinates have an upward vertical axis; the input shape excludes gravity.
GRAVITY = np.array([0., 0., -9.81])
# A 15-degree mesh brackets competing circular modes; the winner is refined continuously.
GRID = np.linspace(-np.pi, np.pi, 24, endpoint=False)
# Four contiguous validation blocks expose changes in the motion or the radio environment.
BLOCKS = 4
# Reference quantiles are fixed before inspecting real errors; these are screening margins.
NORMAL_95 = 1.95996398454
T_ONE_SIDED_95_DF3 = 2.35336343480
# Physical scales regularize nuisance fits for prediction, never the data-only curvature.
BIAS_STD = .20  # m/s^2, a constant body-frame accelerometer error.
TILT_STD = np.deg2rad(5.)  # radians, local tilt error in the gravity-aligned reference.
# Robust clipping is in radio standard deviations.
HUBER = 2.


def wrap(angle):
    """Return principal circular differences in radians, including at the branch cut."""
    return np.arctan2(np.sin(angle), np.cos(angle))


def huber(x):
    """Return twice the Huber negative log likelihood, before aggregation."""
    return np.where(np.abs(x) <= HUBER, x*x, 2*HUBER*np.abs(x)-HUBER**2)


def basis_inverse(design):
    """Factor an unpenalized nuisance design with scale-independent rank decisions."""
    scale = np.linalg.norm(design, axis=0)
    scale[scale < 1e-12] = 1.
    u, s, vh = np.linalg.svd(design/scale, full_matrices=False)
    take = s > (s[0]*1e-9 if len(s) else 0.)
    return u[:, take], (vh[take].T/scale[:, None])/s[take]


def radio_path(tau, plus, minus, y, sigma, use, basis):
    """Fit a radio-only trajectory on training rows; extrapolate to the validation times."""
    anchors = plus[use] if minus is None else np.vstack((plus[use], minus[use]))
    center = np.unique(anchors, axis=0).mean(0)
    # A coplanar starting point has a zero height Jacobian; the sign is unidentifiable there.
    if np.ptp(anchors[:, 2]) < 1e-5:
        center[2] += 1.
    other = None if minus is None else minus[use]
    cv, *_ = fit_track(tau[use], None, plus[use], other, y[use], sigma[use],
                       np.r_[center, np.zeros(3)], inertial=False)
    fit, *_ = fit_track(tau[use], None, plus[use], other, y[use], sigma[use],
                        cv, basis[use], inertial=False)
    return fit[:3]+tau[:, None]*fit[3:6]+basis@fit[6:].reshape(-1, 3)


class Profile:
    """Raw radio linearization with affine motion, acceleration and channel offsets profiled."""

    def __init__(self, tau, force, bias, plus, minus, y, sigma, path, cref, channels, use):
        """Pre-project the shared nuisance space; rotating bias/tilt are refit at every angle."""
        pred, grad = radio_geometry(path, plus, minus)
        self.use = use
        self.target = (y-pred+np.sum(grad*(path-.5*tau[:, None]**2*GRAVITY), axis=1))/sigma
        self.base = np.column_stack((grad, tau[:, None]*grad, .5*tau[:, None]**2*grad, channels))/sigma[:, None]
        # Coefficients rotate around world z. Two horizontal tilt directions are nuisance only.
        u = force@cref.T
        vectors = np.concatenate((u[:, :, None], -np.einsum('ij,njk->nik', cref, bias),
                                  np.cross([1., 0., 0.], u)[:, :, None],
                                  np.cross([0., 1., 0.], u)[:, :, None]), axis=2)
        nx, ny, nz = (grad/sigma[:, None]).T[:, :, None]
        self.parts = np.stack((nx*vectors[:, 0]+ny*vectors[:, 1],
                               ny*vectors[:, 0]-nx*vectors[:, 1], nz*vectors[:, 2]), axis=0)
        # Validation values must never influence training weights or their normalization.
        self.w = np.sqrt(np.minimum(1., HUBER/np.maximum(np.abs((y[use]-pred[use])/sigma[use]), 1e-12)))
        self.q, self.v = basis_inverse(self.base[use]*self.w[:, None])
        stacked = self.parts[:, use].transpose(1, 0, 2).reshape(len(self.w), -1)
        cols = np.column_stack((self.target[use], stacked))*self.w[:, None]
        orthogonal = cols-self.q@(self.q.T@cols)
        self.response = orthogonal[:, 0]
        self.projected = orthogonal[:, 1:].reshape(len(self.w), 3, -1).transpose(1, 0, 2)
        self.prior = np.diag(1./np.r_[np.full(3, BIAS_STD), np.full(2, TILT_STD)])

    def fit(self, angle, details=False):
        """Profile all nuisance terms at one circular heading, optionally retaining predictions."""
        co, si = np.cos(angle), np.sin(angle)
        p = co*self.projected[0]+si*self.projected[1]+self.projected[2]
        d, target = p[:, 1:], self.response-p[:, 0]
        # The diagonal prior makes this small normal system nonsingular for stationary data.
        beta = np.linalg.solve(d.T@d+self.prior.T@self.prior, d.T@target)
        residual = target-d@beta
        cost = np.dot(residual, residual)+np.dot(self.prior@beta, self.prior@beta)
        if not details:
            return float(cost)
        vec = np.r_[1., beta]
        local = (co*self.parts[0]+si*self.parts[1]+self.parts[2])@vec
        base_coef = self.v@(self.q.T@((self.target[self.use]-local[self.use])*self.w))
        # Heading curvature is data-only: freely project nuisance, including bias and tilt.
        derivative = (-si*self.projected[0]+co*self.projected[1])@vec
        if d.shape[1]:
            norms = np.maximum(np.linalg.norm(d, axis=0), 1e-14)
            scaled = d/norms
            derivative -= scaled@np.linalg.lstsq(scaled, derivative, rcond=1e-9)[0]
        info = float(derivative@derivative)
        return float(cost), self.target-self.base@base_coef-local, info

    def search(self):
        """Find the best circular grid basin, then refine it on both sides of the seam."""
        cost = np.array([self.fit(x) for x in GRID])
        i = int(np.argmin(cost)); step = 2*np.pi/len(GRID)
        sol = minimize_scalar(self.fit, bounds=(GRID[i]-step, GRID[i]+step), method='bounded',
                              options={'xatol': 2e-4})
        return float(wrap(sol.x)), cost


def decide(tau, force, B, plus, minus, y, sigma, path, Cref):
    """Return a world-z yaw correction in radians; return delta=0 whenever evidence fails."""
    tau, force, B, plus, y, path, Cref = [np.asarray(x, dtype=float) for x in (tau, force, B, plus, y, path, Cref)]
    minus = None if minus is None else np.asarray(minus, dtype=float)
    sigma = np.broadcast_to(np.asarray(sigma, dtype=float), tau.shape)
    out = dict(delta=0., candidate=0., accepted=False, sd_deg=float('inf'),
               info=0., profile_arc_deg=180., cv_gain=float('nan'), cv_lower=float('nan'),
               fold_delta_deg=[], fold_gain=[], reason='insufficient_rows')
    if len(tau) < 2*BLOCKS+6 or np.ptp(tau) < 1e-4 or np.any(sigma <= 0):
        return out
    ids = plus if minus is None else np.column_stack((plus, minus))
    _, group = np.unique(ids, axis=0, return_inverse=True)
    channels = np.eye(group.max()+1)[group]
    all_rows = np.ones(len(tau), dtype=bool)

    def make(use, anchor_path):
        """Construct the training-only profile about the supplied antenna path."""
        return Profile(tau, force, B, plus, minus, y, sigma, anchor_path, Cref, channels, use)

    full = make(all_rows, path)
    angle, grid_cost = full.search()
    cost, residual, info = full.fit(angle, details=True)
    # Never infer sub-radio noise from an overfitted path; large residual scale is adverse.
    scale = max(1., float(1.4826*np.median(np.abs(residual-np.median(residual)))))
    sd = scale/np.sqrt(max(info, 1e-24))
    plausible = GRID[grid_cost <= cost+(NORMAL_95*scale)**2]
    arc = float(np.max(np.abs(wrap(plausible-angle)))) if len(plausible) else 0.
    out.update(candidate=angle, info=info, sd_deg=float(np.rad2deg(sd)), profile_arc_deg=float(np.rad2deg(arc)))
    # A competing mode in the other half-plane or a correction compatible with zero is insufficient.
    if arc >= np.pi/2 or abs(angle) <= NORMAL_95*sd:
        out['reason'] = 'ambiguous_profile' if arc >= np.pi/2 else 'small_against_uncertainty'
        return out
    # Leave out contiguous time blocks. Refit even the radio linearization path without those readings.
    tbin = np.minimum(((tau-tau.min())/np.ptp(tau)*BLOCKS).astype(int), BLOCKS-1)
    basis = defect_basis(tau)
    gains, choices = [], []
    for block in range(BLOCKS):
        hold = tbin == block
        if hold.sum() < 2 or (~hold).sum() <= full.q.shape[1]+2:
            out['reason'] = 'insufficient_blocks'
            return out
        subpath = radio_path(tau, plus, minus, y, sigma, ~hold, basis)
        train = make(~hold, subpath)
        choice, _ = train.search()
        _, old, _ = train.fit(0., details=True)
        _, new, _ = train.fit(choice, details=True)
        # An unobserved link has no estimated offset under either hypothesis; exclude it from comparison.
        scored = hold & np.isin(group, np.unique(group[~hold]))
        if scored.sum() < 2:
            out['reason'] = 'unseen_links'
            return out
        gains.append(float(np.mean(huber(old[scored])-huber(new[scored]))))
        choices.append(choice)
    choices = np.array(choices)
    offsets = wrap(choices-angle)
    jack_sd = float(np.sqrt((BLOCKS-1)/BLOCKS*np.sum((offsets-offsets.mean())**2)))
    sd = max(sd, jack_sd)
    avg = float(np.mean(gains)); lower = float(avg-T_ONE_SIDED_95_DF3*np.std(gains, ddof=1)/np.sqrt(BLOCKS))
    out.update(cv_gain=avg, cv_lower=lower, sd_deg=float(np.rad2deg(sd)),
               fold_delta_deg=np.rad2deg(choices).tolist(), fold_gain=gains)
    if np.max(np.abs(offsets)) >= np.pi/2 or abs(angle) <= NORMAL_95*sd:
        out['reason'] = 'unstable_across_blocks'
    elif lower <= 0.:
        out['reason'] = 'no_predictive_advantage'
    else:
        out.update(accepted=True, delta=angle, reason='accepted')
    return out
