"""Swap only the two-second Delft position frontend and retain one InEKF backend.

Theodosis' quadratic lift uses a gyro-propagated world input without test truth.
A separate height-profiled squared-range solve tests an XY-specific alternative.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from cpu_replay import replay_pool

# Import the checked, shared Delft replay and the external observer adaptation.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE/'quadratic_observer'))
from observer import lift_prior, quadratic_predict, quadratic_update
from delft import study as nav

# This bridge reuses the existing handoff clock and endpoint score from the Delft comparison.
FRONTENDS = ('radio_cv', 'quadratic_multi', 'height_profile_xy')
# Rank and Huber clipping are numerical and radio-noise scales; no test label selects them.
RANK_TOL = 1e-9
HUBER = 2.


def projected_xy(tau, anchors, radii, sigma, native):
    """Eliminate height and squared norm; retain native z/vz and fall back if XY is unidentified."""
    center = np.mean(np.unique(anchors, axis=0), axis=0)
    a = anchors-center
    h = np.column_stack((a[:, :2], tau[:, None]*a[:, :2]))
    nuisance = np.column_stack((np.ones(len(tau)), tau, tau*tau, a[:, 2], tau*a[:, 2]))
    y = .5*(np.sum(a*a, axis=1)+sigma*sigma-radii*radii)
    # A radio-only CV predicts distance for weighting, avoiding weights inversely tied to a noisy radius.
    mean_r = np.linalg.norm(native[:3]+tau[:, None]*native[3:]-anchors, axis=1)
    scale = np.sqrt(mean_r**2*sigma**2+.5*sigma**4)
    weights = np.ones(len(y))
    solution = native.copy()
    for _ in range(4):
        factor = np.sqrt(weights)/np.maximum(scale, 1e-9)
        n = nuisance*factor[:, None]; x = h*factor[:, None]; target = y*factor
        # Equilibrate nuisance columns so surveying coordinate units cannot change rank.
        ns = np.linalg.norm(n, axis=0); ns[ns < 1e-10] = 1.
        n /= ns
        u, s, _ = np.linalg.svd(n, full_matrices=False)
        q = u[:, s > RANK_TOL*s[0]]
        xp = x-q@(q.T@x); yp = target-q@(q.T@target)
        beta, _, rank, singular = np.linalg.lstsq(xp, yp, rcond=RANK_TOL)
        if rank < 4:
            return native.copy(), np.inf, True
        residual = (yp-xp@beta)/np.maximum(np.sqrt(weights), 1e-9)
        weights = np.minimum(1., HUBER/np.maximum(np.abs(residual), 1e-12))
    solution[:2] = center[:2]+beta[:2]
    solution[3:5] = beta[2:]
    return solution, float(singular[0]/singular[-1]), False


def record_errors(predicted, truth):
    """Measure radial horizontal and separate vertical errors without aligning trajectories."""
    err = predicted-truth
    xy = np.sum(err[:, :2]**2, axis=1)
    return np.sqrt(np.mean(xy)), np.sqrt(np.mean(err[:, 2]**2)), np.sqrt(np.max(xy))


def one_case(args):
    """Replay a common prefix, fork position once, and score the fixed future for all frontends."""
    cache, name, layout, offset = args
    data = dict(np.load(cache/(Path(name).stem+'.npz')))
    model = json.loads((cache/'protocol.json').read_text(encoding='utf-8'))['model']
    rb, rs = np.asarray(model['range_bias']), np.asarray(model['range_sigma'])
    t, imu, anchors = data['time'], data['imu'], data['anchors']
    good = data['changed'] & (data['ranges']>0)
    active = np.ones(len(anchors), dtype=bool)
    if layout == 'lower4':
        active[:] = False; active[np.argsort(anchors[:, 2])[:4]] = True
    i15, i25, i27, i31, i36 = np.searchsorted(t, (15.,25.,27.,31.,36.))
    parity = np.cumsum(good & ~((t >= 15)&(t < 25))[:, None], axis=0)%2
    order = np.r_[6:9, 3:6]
    core = None
    for k in range(i27+1):
        if core is not None and k:
            if i25 < k <= i27:
                # A finite increment of the very same blind InEKF gives its held world input exactly.
                before = core.X[:3, 3].copy()
            core.predict(imu[k-1], float(t[k]-t[k-1]))
            if i25 < k <= i27:
                dt = float(t[k]-t[k-1]); acc = (core.X[:3, 3]-before)/dt
                aug, paug = quadratic_predict(aug, paug, acc, dt)
        if core is None:
            if t[k] >= 2.:
                seed = nav.solve_window(*nav.window_data(data,0,k,model), method='profile')
                core = nav.tracker_from_estimate(seed, data['rotations'][k], data['bias'], model)
            continue
        if k == i15:
            physical = core.physical_covariance()
            core.X[:3, :3] = nav.yaw_rotation(np.deg2rad(offset))@core.X[:3, :3]
            chart = nav.coordinate_map(core.X); core.P = nav.symmetrize(chart@physical@chart.T)
        if k == i25:
            aug, paug = lift_prior(np.r_[core.X[:3, 4], core.X[:3, 3]], core.physical_covariance()[np.ix_(order, order)])
        for j in np.flatnonzero(good[k]&active):
            if k < i15:
                core.correct_twr(anchors[j], data['ranges'][k, j], int(j))
            elif k >= i25:
                aug, paug = quadratic_update(aug, paug, anchors[j], data['ranges'][k,j]-rb[j], rs[j])
    reference, _ = nav.position_handoff(core,data,i25,i27,active,model)
    native = np.r_[reference.X[:3, 4],reference.X[:3, 3]]
    idx, j = nav.range_events(data, i25, i27, active)
    planar, cond, fallback = projected_xy(t[idx]-t[i27], anchors[j], data['ranges'][idx,j]-rb[j], rs[j], native)
    values = [native, aug[3:], planar]
    branches, positions = {}, {}
    for label, pv in zip(FRONTENDS, values):
        state = reference.X.copy(); state[:3, 4], state[:3, 3] = pv[:3], pv[3:]
        branches[label] = nav.tracker(state, reference.physical_covariance(), reference.bias.copy(), model)
        positions[label] = [state[:3, 4].copy()]
    for k in range(i27+1,i36):
        ids = np.flatnonzero(good[k]&active&((parity[k]==1) if k <= i31 else True))
        for label, branch in branches.items():
            branch.predict(imu[k-1],float(t[k]-t[k-1]))
            for j in ids:
                branch.correct_twr(anchors[j],data['ranges'][k,j],int(j))
            if k == i31:
                # Every frontend uses the exact retained-heading handoff of the shared experiment.
                physical, world = nav.conservative_attitude_covariance(branch)
                rot = branch.X[:3, :3]
                physical[:3,:3] = rot.T@world@rot
                branches[label] = nav.tracker(branch.X.copy(),physical,branch.bias.copy(),model)
            positions[label].append(branch.X[:3,4].copy())
    # Test truth is opened only after all branches have finished.
    truth = np.load(cache/(Path(name).stem+'_scoring.npz'))['position']
    rows = []
    for (label, pv) in zip(FRONTENDS, values):
        pos = np.array(positions[label]); err = pv[:3]-truth[i27]
        xy,z,peak = record_errors(pos[i31-i27:],truth[i31:i36])
        early_xy, early_z, _ = record_errors(pos[:i31-i27],truth[i27:i31])
        rows.append(dict(sequence=name, layout=layout, yaw_offset_deg=offset, frontend=label,
                         position_xy_m=np.linalg.norm(err[:2]), position_z_m=abs(err[2]),
                         early_xy_rmse_m=early_xy,early_z_rmse_m=early_z,fixed_xy_rmse_m=xy,
                         fixed_z_rmse_m=z,fixed_peak_xy_m=peak,n_range=len(idx),
                         height_profile_condition=cond,height_profile_fallback=fallback))
    return rows


def main():
    """Write all matched trials and verify the common native control against the published replay."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--delft',type=Path,default=HERE/'delft'/'results')
    parser.add_argument('--output',type=Path,default=HERE/'bridge_results')
    parser.add_argument('--workers',type=int,help='Optional worker limit; default uses all available CPUs.')
    args = parser.parse_args(); cache = args.delft/'cache'
    protocol = json.loads((cache/'protocol.json').read_text(encoding='utf-8'))
    jobs = [(cache,n,layout,angle) for n in protocol['evaluation'] for layout in ('all8','lower4') for angle in (0,90,-90,180)]
    with replay_pool(len(jobs),args.workers) as pool:
        frame = pd.DataFrame([r for batch in pool.map(one_case,jobs) for r in batch])
    args.output.mkdir(parents=True,exist_ok=True)
    frame.to_csv(args.output/'paired.csv',index=False)
    original = pd.read_csv(args.delft/'metrics.csv').query("method == 'position_only'")
    join = original.merge(frame.query("frontend == 'radio_cv'"),on=['sequence','layout','yaw_offset_deg'],validate='one_to_one')
    diff = float(np.max(np.abs(join.fixed_xy_rmse_m_x-join.fixed_xy_rmse_m_y)))
    if len(join)!=48 or diff>1e-7:
        raise AssertionError(f'Common-backend control mismatch ({len(join)} pairs), max {diff} m')
    measures = ['position_xy_m','early_xy_rmse_m','fixed_xy_rmse_m','fixed_z_rmse_m']
    grouped = frame.groupby(['layout','frontend'])[measures].median()
    grouped.to_csv(args.output/'median.csv')
    print(grouped.to_string()); print('Matched native control max difference:',diff,flush=True)


if __name__ == '__main__':
    main()
