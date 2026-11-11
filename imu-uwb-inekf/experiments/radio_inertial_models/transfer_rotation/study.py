"""Predetermined UTIL trial3 dynamic-window transfer, with truth used only for scoring.

All methods share in-window SI inertial samples and directed raw TDoA events.
"""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation, Slerp

# Sibling modules supply the common current experimental estimator.
BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from rotation_window import local_shape, solve_attitude, wahba_seed
from window_dynamics import defect_basis, fit_track, radio_geometry, robust_rows

# Settings were fixed before observing this method's trial3 scores.
DATA = Path('E:/datasets/dataset/flight-dataset')
WINDOWS = ((10., 16.), (25., 31.))
IMU_DT = .01
RADIO_DT = .05
SIGMA = .30
LEVER = np.array([-.01245, .00127, .0908])
BIAS_SIGMA = .10


def read_stream(frame, columns):
    """Read and sort one raw stream independently, retaining its original clock."""
    data = frame[columns].dropna().to_numpy(float)
    return data[np.argsort(data[:, 0], kind='stable')]


def prepare(frame, anchors, start, end, ring):
    """Crop before interpolation; align selected asynchronous radio at its actual time."""
    acc = read_stream(frame, ['t_acc', 'acc_x', 'acc_y', 'acc_z'])
    gyr = read_stream(frame, ['t_gyro', 'gyro_x', 'gyro_y', 'gyro_z'])
    radio = read_stream(frame, ['t_tdoa', 'idA', 'idB', 'tdoa_meas'])
    origin = max(acc[0, 0], gyr[0, 0], radio[0, 0])
    left, right = origin+start, origin+end
    acc = acc[(acc[:, 0] >= left) & (acc[:, 0] <= right)]
    gyr = gyr[(gyr[:, 0] >= left) & (gyr[:, 0] <= right)]
    left, right = max(acc[0, 0], gyr[0, 0]), min(acc[-1, 0], gyr[-1, 0])
    radio = radio[(radio[:, 0] >= left) & (radio[:, 0] <= right)]
    if ring:
        pairs = {(j, (j+1) % 8) for j in range(8)}
        radio = radio[np.array([tuple(row) in pairs for row in radio[:, 1:3]])]
    # Keep the first observed event in each fixed time bin, without using measurement values.
    bins = np.floor((radio[:, 0]-left)/RADIO_DT).astype(int)
    _, first = np.unique(bins, return_index=True)
    radio = radio[first]
    t = np.unique(np.r_[np.arange(left, right, IMU_DT), radio[:, 0], right])
    values = np.column_stack([np.interp(t, raw[:, 0], raw[:, k])
                              for raw in (acc, gyr) for k in (1, 2, 3)])
    values[:, :3] *= 9.81
    values[:, 3:] = np.radians(values[:, 3:])
    rows = np.searchsorted(t, radio[:, 0])
    force, bias, event_q, terminal_q = local_shape(t, values[:, :3], values[:, 3:], rows, LEVER)
    plus, minus = anchors[radio[:, 2].astype(int)], anchors[radio[:, 1].astype(int)]
    return dict(t=t, imu=values, radio=radio, tau=radio[:, 0]-t[-1], plus=plus, minus=minus,
                z=radio[:, 3], force=force, bias=bias, event_q=event_q, terminal_q=terminal_q)


def endpoint_truth(frame, stamp):
    """Access truth only after fitting, to score the absolute body endpoint."""
    truth = read_stream(frame, ['t_pose', 'pose_x', 'pose_y', 'pose_z',
                                'pose_qx', 'pose_qy', 'pose_qz', 'pose_qw'])
    position = np.array([np.interp(stamp, truth[:, 0], truth[:, k]) for k in (1, 2, 3)])
    rotation = Slerp(truth[:, 0], Rotation.from_quat(truth[:, 4:]))(stamp)
    return position, rotation


def score(position, rotation, truth_position, truth_rotation):
    """Compute absolute position, full rotation angle and wrapped yaw without alignment."""
    out = dict(position_m=float(np.linalg.norm(position-truth_position)))
    if rotation is not None:
        estimated = Rotation.from_matrix(rotation)
        yaw = estimated.as_euler('xyz')[2]-truth_rotation.as_euler('xyz')[2]
        out.update(so3_deg=float(np.degrees((truth_rotation.inv()*estimated).magnitude())),
                   yaw_deg=float(np.degrees(abs(np.arctan2(np.sin(yaw), np.cos(yaw))))))
    return out


def rotation_information(data, fit):
    """Profile translation and bias from robust local radio curvature, with and without its prior."""
    x, c, tau = fit['x'], fit['C'], data['tau']
    local = data['force']-data['bias']@x[9:]
    pos = x[:3]+tau[:, None]*x[3:6]+.5*tau[:, None]**2*np.array([0., 0., -9.81])+local@c.T
    predicted, direction = radio_geometry(pos, data['plus'], data['minus'])
    _, derivative = robust_rows((predicted-data['z'])/SIGMA)
    grad = direction*(derivative/SIGMA)[:, None]
    local_grad = grad@c
    attitude = np.cross(local, local_grad)
    nuisance = np.column_stack((grad, tau[:, None]*grad,
                               -np.einsum('ni,nij->nj', local_grad, data['bias'])))
    def eigenvalues(n, a):
        """Return the rotation Schur-complement eigenvalues without a confidence claim."""
        residual = a-n@np.linalg.lstsq(n, a, rcond=1e-10)[0]
        return np.linalg.eigvalsh(residual.T@residual).tolist()
    prior_n = np.vstack((nuisance, np.column_stack((np.zeros((3, 6)), np.eye(3)/BIAS_SIGMA))))
    return dict(rotation_info_free_bias=eigenvalues(nuisance, attitude),
                rotation_info_bias_prior=eigenvalues(prior_n, np.vstack((attitude, np.zeros((3, 3))))))


def run_window(frame, anchors, name, start, end):
    """Run the fixed independent seeds and then score; no truth enters any estimator."""
    data = prepare(frame, anchors, start, end, 'tdoa2' in name)
    tau, plus, minus, z = (data[k] for k in ('tau', 'plus', 'minus', 'z'))
    cv = np.r_[anchors.mean(axis=0), np.zeros(3)]
    started = time.perf_counter()
    cv_fit = fit_track(tau, None, plus, minus, z, SIGMA, cv, inertial=False)
    phi = defect_basis(tau)
    radio_fit = fit_track(tau, None, plus, minus, z, SIGMA, cv_fit[0], phi, False)
    path = radio_fit[0][:3]+tau[:, None]*radio_fit[0][3:6]+phi@radio_fit[0][6:].reshape(-1, 3)
    common_seconds = time.perf_counter()-started
    average = data['imu'][:, :3].mean(axis=0)
    gravity_seed = Rotation.align_vectors([[0., 0., 1.]], [average/np.linalg.norm(average)])[0].as_matrix()
    seeds = [('identity', np.eye(3)), ('window_gravity', gravity_seed), ('projected_wahba', None)]
    fitted = []
    for method, initial in seeds:
        begun = time.perf_counter()
        fit = solve_attitude(tau, data['force'], plus, minus, z, SIGMA, path,
                             bias_sensitivity=data['bias'], seed_rotation=initial, bias_sigma=BIAS_SIGMA)
        fit.update(method=method, seconds=time.perf_counter()-begun+common_seconds)
        fitted.append(fit)
    # Four fixed yaw rotations of the same in-window gravity seed provide a stronger local baseline.
    begun = time.perf_counter()
    uniform = [solve_attitude(tau, data['force'], plus, minus, z, SIGMA, path,
                             bias_sensitivity=data['bias'], bias_sigma=BIAS_SIGMA,
                             seed_rotation=Rotation.from_euler('z', angle).as_matrix()@gravity_seed)
               for angle in (0., np.pi/2, np.pi, 3*np.pi/2)]
    best = min(uniform, key=lambda value: value['cost']).copy()
    best.update(method='uniform4_gravity', seconds=time.perf_counter()-begun+common_seconds,
                nfev=sum(value['nfev'] for value in uniform))
    fitted.append(best)
    truth_position, truth_rotation = endpoint_truth(frame, data['t'][-1])
    common = dict(name=name, start=start, end=end, actual_end=float(data['t'][-1]),
                  radio_count=len(z), imu_count=len(data['t']), common_seconds=common_seconds,
                  force_singular=wahba_seed(tau, data['force'], path)[1].tolist())
    rows = []
    for fit in fitted:
        rotation = fit['C']@data['terminal_q']
        rows.append(dict(**common, method=fit['method'], cost=fit['cost'], nfev=fit['nfev'],
                         seconds=fit['seconds'], success=bool(fit['success']),
                         bias_norm=float(np.linalg.norm(fit['x'][9:])),
                         initial_so3_deg=float(np.degrees((truth_rotation.inv()*Rotation.from_matrix(fit['C0']@data['terminal_q'])).magnitude())),
                         **rotation_information(data, fit),
                         **score(fit['x'][:3], rotation, truth_position, truth_rotation)))
    # The radio-only path estimates the antenna position; no attitude is available to remove its lever arm.
    rows.append(dict(**common, method='radio_only_tag', cost=radio_fit[1], nfev=radio_fit[2],
                     seconds=common_seconds,
                     position_m=float(np.linalg.norm(radio_fit[0][:3]-truth_position-truth_rotation.apply(LEVER))),
                     position_target='antenna; truth lever used only to define scoring target'))
    return rows


def main():
    """Save each completed window immediately for partial-run review and reproducibility."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--names', nargs='+')
    args = parser.parse_args()
    names = args.names or [f'const{geometry}-trial3-tdoa{protocol}'+('-traj1' if geometry == 4 else '')
                           for geometry in range(1, 5) for protocol in (2, 3)]
    output = Path(__file__).parent
    rows = []
    for name in names:
        file = DATA/'csv-data'/name.split('-')[0]/(name+'.csv')
        if not file.exists():
            rows.append(dict(name=name, missing=str(file)))
            continue
        frame = pd.read_csv(file)
        anchors = np.load(DATA/'survey-results'/('anchor_'+name.split('-')[0]+'.npz'))['an_pos']
        for start, end in WINDOWS:
            result = run_window(frame, anchors, name, start, end)
            rows.extend(result)
            (output/(name+f'_{int(start)}_{int(end)}.json')).write_text(json.dumps(result, indent=2), encoding='utf-8')
            print(json.dumps(result), flush=True)
    (output/'metrics.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
