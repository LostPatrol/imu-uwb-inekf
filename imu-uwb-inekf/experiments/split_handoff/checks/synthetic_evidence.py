"""Label-free synthetic yaw-evidence challenges and exact nuisance counterexamples.

No real recordings, calibration or ground truth files are accessed by this module.
"""

import argparse
import importlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

# All challenge settings are fixed independently of real-data results.
DURATION = 6.0
RADIO_HZ = 30
SIGMA = .10
SEED = 81126
GRAVITY = np.array([0., 0., -9.81])
# Eight well-spread anchors retain both near and far directions.
ANCHORS = np.array([[x, y, z] for x in (-4., 4.) for y in (-4., 4.) for z in (.2, 3.)])
# Artifacts are confined to the authorized process-output directory.
ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT/'agent'/'codex'/'split_checks'


def rotation(yaw):
    """Return a world-vertical rotation for one radian yaw coordinate."""
    return Rotation.from_euler('z', yaw).as_matrix()


def reanchored_basis(tau):
    """Construct a non-polynomial displacement with zero endpoint position and velocity."""
    return np.column_stack((np.sin(1.6*tau)-1.6*tau,
                            .7*(np.cos(1.1*tau)-1),
                            .2*(np.sin(.8*tau)-.8*tau)))


def bias_integral(tau, rotating):
    """Integrate a known rotation field, retaining genuine rotating body-bias columns."""
    if not rotating:
        return .5*tau[:, None, None]**2*np.eye(3)[None]
    dense = np.linspace(-DURATION, 0., 1201)
    q = Rotation.from_euler('xyz', np.column_stack((.2*np.sin(1.3*dense),
                                                   .15*np.sin(.9*dense),
                                                   .35*np.sin(.7*dense)))).as_matrix()
    dt = dense[1]-dense[0]
    vel = np.concatenate((np.zeros((1, 3, 3)), np.cumsum(q[:-1]*dt, axis=0)))
    disp = np.concatenate((np.zeros((1, 3, 3)), np.cumsum(vel[:-1]*dt+.5*q[:-1]*dt**2, axis=0)))
    integral = disp-disp[-1]-dense[:, None, None]*vel[-1]
    return np.array([[np.interp(tau, dense, integral[:, a, b]) for b in range(3)]
                      for a in range(3)]).transpose(2, 0, 1)


def radio(position, plus, minus):
    """Return directed raw distances and the local world-position gradient."""
    delta = position-plus
    distance = np.linalg.norm(delta, axis=1)
    grad = delta/distance[:, None]
    if minus is not None:
        other = position-minus
        other_distance = np.linalg.norm(other, axis=1)
        distance -= other_distance
        grad -= other/other_distance[:, None]
    return distance, grad


def build(name, seed=SEED):
    """Create one controlled API input without exposing synthetic truth to decide()."""
    rng = np.random.default_rng(seed)
    tau = np.linspace(-DURATION, 0., int(DURATION*RADIO_HZ)+1)
    amplitude = .003 if name == 'weak' else .8
    dynamic = amplitude*reanchored_basis(tau)
    if name == 'constant_velocity':
        dynamic[:] = 0.
    if name == 'constant_acceleration':
        dynamic = .5*tau[:, None]**2*np.array([.4, -.2, .1])
    force = dynamic-.5*tau[:, None]**2*GRAVITY
    rotating = name in ('rotating_bias', 'wrong_bias_model', 'bias_only_excitation')
    b = bias_integral(tau, rotating)
    bias = np.array([.12, -.08, .06]) if rotating else np.zeros(3)
    if name == 'bias_only_excitation':
        force = -.5*tau[:, None]**2*GRAVITY+b@bias
    yaw = np.radians(179.) if name == 'near_180' else .7
    endpoint, velocity = np.array([1.2, .7, 1.4]), np.array([.10, -.08, .02])
    translation = endpoint+tau[:, None]*velocity+.5*tau[:, None]**2*GRAVITY
    physical = translation+(force-b@bias)@rotation(yaw).T
    anchors = ANCHORS.copy()
    if name == 'coplanar_weak_z':
        anchors[:, 2] = 0.
        physical[:, 2] = .03
    ids = np.arange(len(tau)) % len(anchors)
    plus = anchors[ids]
    minus = anchors[(ids+3) % len(anchors)] if name.startswith('tdoa') else None
    clean, _ = radio(physical, plus, minus)
    y = clean+rng.normal(0., SIGMA, len(tau))
    # This ideal independent path isolates the evidence model from radio-path fit errors.
    path = physical.copy()
    if name == 'systematic_yaw_alias':
        wrong = yaw+1.2
        path = translation+(force-b@bias)@rotation(wrong).T
        y = radio(path, plus, minus)[0]+rng.normal(0., SIGMA, len(tau))
    if name in ('link_offsets', 'tdoa_link_offsets'):
        y += np.array([.6, -.4, .3, -.5, .2, -.3, .5, -.2])[ids]
    supplied_b = bias_integral(tau, False) if name == 'wrong_bias_model' else b
    public = dict(tau=tau, force=force, B=supplied_b, plus=plus, minus=minus,
                  y=y, sigma=SIGMA, path=path, Cref=np.eye(3))
    private = dict(true_yaw=yaw, true_bias=bias, physical_path=physical, clean=clean,
                   expected='nonidentifiable' if name in ('constant_velocity', 'constant_acceleration', 'bias_only_excitation')
                   else 'coherent_error_indistinguishable' if name == 'systematic_yaw_alias'
                   else 'weak_or_model_error' if name in ('weak', 'link_offsets', 'tdoa_link_offsets', 'wrong_bias_model')
                   else 'excited_local_model')
    return public, private


def profile(public, values, fixed_bias_projector=False):
    """Compute a diagnostic full-circle linearized profile with yaw-dependent bias nuisances."""
    tau, path = public['tau'], public['path']
    predicted, direction = radio(path, public['plus'], public['minus'])
    target = values-predicted+np.einsum('ni,ni->n', direction, path)
    polynomial = np.column_stack((direction, tau[:, None]*direction,
                                  .5*tau[:, None]**2*direction))
    angles = np.linspace(-np.pi, np.pi, 361)
    costs = []
    for angle in angles:
        c = rotation(angle)@public['Cref']
        bias_c = public['Cref'] if fixed_bias_projector else c
        nuisance = np.column_stack((polynomial, -np.einsum('ni,nij->nj', direction@bias_c, public['B'])))
        residual = target-np.einsum('ni,ni->n', direction, public['force']@c.T)
        residual -= nuisance@np.linalg.lstsq(nuisance, residual, rcond=1e-10)[0]
        costs.append(float(residual@residual/public['sigma']**2))
    return angles, np.asarray(costs)


def audit_schur_and_handoff():
    """Exhibit XY conditional-confidence inflation and forbid implicit yaw updates from position."""
    h = np.array([[4., 0., 1.9], [0., 3., 0.], [1.9, 0., 1.]])
    conditional = np.linalg.inv(h[:2, :2])
    marginal = np.linalg.inv(h)[:2, :2]
    schur = np.linalg.inv(h[:2, :2]-h[:2, 2:]@np.linalg.inv(h[2:, 2:])@h[2:, :2])
    p = np.array([[1., .8], [.8, 1.]])
    measurement = np.array([[0., 1.]])
    gain = p@measurement.T/float((measurement@p@measurement.T+.1).item())
    frozen = gain.copy(); frozen[0] = 0.
    identity = np.eye(2)
    joseph = (identity-frozen@measurement)@p@(identity-frozen@measurement).T+frozen@frozen.T*.1
    return dict(xy_conditional_cov=conditional.tolist(), xy_marginal_cov=marginal.tolist(),
                schur_matches_full_inverse=bool(np.allclose(marginal, schur)),
                ordinary_position_update_yaw_gain=float(gain[0, 0]),
                frozen_yaw_gain=float(frozen[0, 0]), frozen_yaw_variance=float(joseph[0, 0]),
                frozen_covariance_min_eigenvalue=float(np.linalg.eigvalsh(joseph)[0]),
                once_used_unit_measurement_variance=.5,
                wrongly_repeated_same_measurement_variance=1/3)


def serializable(value):
    """Convert public API outputs into reviewable JSON, preserving nested diagnostics."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def main():
    """Save challenge inputs and optional current-algorithm decisions, without tuning thresholds."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--module', default=None)
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    decide = importlib.import_module(args.module).decide if args.module else None
    names = ('strong', 'weak', 'constant_velocity', 'constant_acceleration', 'near_180',
             'coplanar_weak_z', 'rotating_bias', 'wrong_bias_model', 'tdoa_strong',
             'systematic_yaw_alias', 'link_offsets', 'tdoa_link_offsets', 'bias_only_excitation')
    result = dict(algebra=audit_schur_and_handoff(), cases=[])
    for name in names:
        public, private = build(name)
        angles, costs = profile(public, public['y'])
        _, ideal_costs = profile(public, radio(public['path'], public['plus'], public['minus'])[0])
        row = dict(name=name, expected=private['expected'], true_yaw_deg=float(np.degrees(private['true_yaw'])),
                   grid_candidate_deg=float(np.degrees(angles[np.argmin(costs)])),
                   noisy_profile_span=float(np.ptp(costs)), ideal_profile_span=float(np.ptp(ideal_costs)),
                   ideal_minimum=float(ideal_costs.min()),
                   ideal_candidate_deg=float(np.degrees(angles[np.argmin(ideal_costs)])))
        if name in ('constant_velocity', 'constant_acceleration'):
            assert np.ptp(costs) < 1e-8, 'An exact polynomial nuisance must leave yaw unidentifiable.'
        if name == 'bias_only_excitation':
            _, fixed_ideal = profile(public, radio(public['path'], public['plus'], public['minus'])[0], True)
            row['incorrect_fixed_bias_projector_ideal_span'] = float(np.ptp(fixed_ideal))
            assert np.ptp(ideal_costs) < 1e-8, 'Body bias alone must not manufacture physical yaw information.'
        if name == 'strong':
            contaminated = dict(public)
            contaminated['force'] = public['force']+.5*public['tau'][:, None]**2*np.array([.3, -.2, .1])
            _, contaminated_costs = profile(contaminated, public['y'])
            row['polynomial_contamination_profile_max_difference'] = float(np.max(abs(costs-contaminated_costs)))
            assert row['polynomial_contamination_profile_max_difference'] < 1e-6
        if decide:
            row['decision'] = decide(**public)
        result['cases'].append(row)
        np.savez_compressed(OUTPUT/(name+'.npz'), **{k:v for k,v in public.items() if v is not None},
                            angles=angles, costs=costs, ideal_costs=ideal_costs)
    (OUTPUT/'synthetic_checks.json').write_text(json.dumps(result, indent=2, default=serializable), encoding='utf-8')
    print(json.dumps(result, indent=2, default=serializable))


if __name__ == '__main__':
    main()
