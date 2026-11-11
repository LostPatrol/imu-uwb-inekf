"""Check public profile algebra, held-radio isolation and circular search on synthetic inputs."""

import hashlib
import json
from pathlib import Path
import sys

import numpy as np

# Only the current public implementation and the synthetic builder are imported.
BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
import yaw_evidence as evidence
from synthetic_evidence import OUTPUT, build, radio, rotation


def make(public, use):
    """Build the exact public nuisance design with the same known link inventory."""
    ids = public['plus'] if public['minus'] is None else np.column_stack((public['plus'], public['minus']))
    _, group = np.unique(ids, axis=0, return_inverse=True)
    channels = np.eye(group.max()+1)[group]
    sigma = np.full(len(public['tau']), public['sigma'])
    return evidence.Profile(public['tau'], public['force'], public['B'], public['plus'], public['minus'],
                            public['y'], sigma, public['path'], public['Cref'], channels, use)


def main():
    """Assert exact algebra and isolation; record dense-grid checks without modifying decisions."""
    public, _ = build('strong')
    all_rows = np.ones(len(public['tau']), dtype=bool)
    profile = make(public, all_rows)
    angle = .83
    beta = np.array([.12, -.08, .06, .02, -.03])
    u = public['force']@public['Cref'].T
    vector = (u-np.einsum('ij,njk,k->ni', public['Cref'], public['B'], beta[:3])
              +beta[3]*np.cross([1., 0., 0.], u)+beta[4]*np.cross([0., 1., 0.], u))
    _, grad = radio(public['path'], public['plus'], public['minus'])
    direct = np.einsum('ni,ni->n', grad, vector@rotation(angle).T)/public['sigma']
    decomposed = (np.cos(angle)*profile.parts[0]+np.sin(angle)*profile.parts[1]+profile.parts[2])@np.r_[1., beta]
    part_error = float(np.max(abs(direct-decomposed)))
    assert part_error < 1e-10
    use = public['tau'] < -1.5
    modified = dict(public)
    modified['y'] = public['y'].copy()
    modified['y'][~use] += 1000.
    train, changed = make(public, use), make(modified, use)
    cost_difference = max(abs(train.fit(a)-changed.fit(a)) for a in (-2., 0., 1.4))
    assert cost_difference < 1e-10
    basis = evidence.defect_basis(public['tau'])
    inputs = (public['tau'], public['plus'], public['minus'])
    sigma = np.full(len(public['tau']), public['sigma'])
    first = evidence.radio_path(*inputs, public['y'], sigma, use, basis)
    second = evidence.radio_path(*inputs, modified['y'], sigma, use, basis)
    path_difference = float(np.max(abs(first-second)))
    assert path_difference < 1e-10
    rows = []
    for name in ('strong', 'weak', 'near_180', 'rotating_bias', 'wrong_bias_model',
                 'tdoa_strong', 'systematic_yaw_alias', 'link_offsets', 'bias_only_excitation'):
        p, _ = build(name)
        fit = make(p, np.ones(len(p['tau']), dtype=bool))
        candidate, _ = fit.search()
        angles = np.linspace(-np.pi, np.pi, 1441, endpoint=False)
        costs = np.array([fit.fit(a) for a in angles])
        dense = int(np.argmin(costs))
        rows.append(dict(name=name, public_candidate_deg=float(np.degrees(candidate)),
                         dense_grid_candidate_deg=float(np.degrees(angles[dense])),
                         public_cost_minus_dense_grid_min=float(fit.fit(candidate)-costs[dense])))
    result = dict(public_sha256=hashlib.sha256((BASE/'yaw_evidence.py').read_bytes()).hexdigest(),
                  trigonometric_model_max_error=part_error,
                  held_radio_perturbation_training_cost_change=cost_difference,
                  held_radio_perturbation_training_path_change=path_difference,
                  dense_search=rows)
    (OUTPUT/'implementation_audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
