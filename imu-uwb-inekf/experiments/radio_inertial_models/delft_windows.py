"""Rebuild matched Delft TWR and native TDoA windows from the original CSV files.

Training uses earlier separate recordings. Test truth is used for clock cropping
and final scoring only, following the paper's convention.
"""

import argparse
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

# Existing physics and the paper initializer are reused, but no saved estimates are read.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'recovery_bootstrap'))
from run_research import DATASET, SENSOR_DELAY_S, INITIAL_IMU_PREFIX_S, estimate_initial_rotation_and_bias
from run_research import preintegrate, load_anchors, load_real_log
from initializer import cv_candidate, lifted_candidate, solve_window
from window_dynamics import DEFECT_PSD, RADIO_PSD, defect_basis, fit_track

# Match the paper's eleven physical windows whenever the recording is long enough.
WINDOWS = [(t, w) for t in (5., 10., 20., 30.) for w in (2., 4., 8.) if w <= t]
# The training/test split by recording excludes the publisher's incomplete files.
MOTIONS = ('triangle', 'octagon', 'hourglass', 'random', 'star', 'square')
# Bound channel noise in metres as in the TWR study, including for the TDoA transfer.
SIGMA_LIMITS = (.08, .50)


def recordings(root, mode):
    """Reserve the earlier member of each available pair for calibration only."""
    train, evaluation = [], []
    for m in MOTIONS:
        files = sorted((root/mode).glob(f'*_{m}.csv'))
        if len(files) > 1:
            train.append(files[0])
        if files:
            evaluation.append(files[-1])
    return train, evaluation


def read_record(root, path, mode):
    """Decode changed entries, their directed anchor IDs and the common score clock."""
    if mode == 'twr':
        rec = load_real_log(root, path.name, sensor_delay_s=SENSOR_DELAY_S)
        rec_valid = rec.changed & (rec.ranges > 0)
        return rec, rec.ranges, rec_valid, np.arange(8), None
    raw = pd.read_csv(path, skipinitialspace=True)
    keys = [c for c in raw.columns if c.startswith('tdoa')]
    # The publisher's simulator maps CSV tdoaij to rho_j - rho_i (both in metres).
    left = np.array([int(k[-1]) for k in keys]); right = np.array([int(k[-2]) for k in keys])
    z = raw[keys].to_numpy(float)
    imu = raw[['accX', 'accY', 'accZ', 'gyroX', 'gyroY', 'gyroZ']].to_numpy(float)
    ref = raw[['otX', 'otY', 'otZ']].to_numpy(float)
    t = raw.timeTick.to_numpy(float)/1000
    valid = np.isfinite(ref).all(1) & (np.linalg.norm(ref, axis=1) > 1e-9)
    sensed = np.isfinite(imu).all(1) & np.isfinite(z).all(1)
    t0 = t[np.flatnonzero(valid & sensed)[0]]
    ts = t-SENSOR_DELAY_S
    selected = sensed & (ts >= t0-1e-12) & (ts <= t[valid][-1]+1e-12)
    gt = np.stack([np.interp(ts[selected], t[valid], c[valid]) for c in ref.T], axis=1)
    rec = SimpleNamespace(time=ts[selected]-t0, imu=imu[selected], ground_truth_position=gt)
    # A held value does not identify a new radio transaction, even if it is nonzero.
    changed = np.zeros_like(z, dtype=bool); changed[1:] = np.abs(np.diff(z, axis=0)) > 1e-12
    return rec, z[selected], changed[selected], left, right


def calibration(root, files, mode, anchors):
    """Estimate each channel median and inlier scale on training recordings alone."""
    pieces = []
    for path in files:
        rec, radio, valid, left, right = read_record(root, path, mode)
        row, chan = np.nonzero(valid)
        pos = rec.ground_truth_position[row]
        expected = np.linalg.norm(pos-anchors[left[chan]], axis=1)
        if right is not None:
            expected -= np.linalg.norm(pos-anchors[right[chan]], axis=1)
        pieces.append(pd.DataFrame({'channel':chan, 'residual':radio[row, chan]-expected}))
    by = pd.concat(pieces).groupby('channel').residual
    median = by.median().reindex(range(len(left))).fillna(0.)
    scale = by.apply(lambda x: 1.4826*np.median(np.abs(x-np.median(x))))
    scale = scale.reindex(median.index).fillna(SIGMA_LIMITS[1]).clip(*SIGMA_LIMITS)
    return median.to_numpy(), scale.to_numpy()


def run_record(root, path, mode, anchors, cal, layouts, fast=False, sensitivity=False):
    """Use the same event set, endpoint and starting yaw in each matched comparison."""
    rec, radio, valid, left, right = read_record(root, path, mode)
    rot, bias = estimate_initial_rotation_and_bias(rec)
    keep = rec.time >= INITIAL_IMU_PREFIX_S
    t = rec.time[keep]-rec.time[keep][0]
    _, dp, dv = preintegrate(t, rec.imu[keep], rot, bias)
    ref = rec.ground_truth_position[keep]; z = radio[keep]; valid = valid[keep]
    bias_z, sigma = cal
    result = []
    for endpoint, width in ([(10.,4.),(20.,8.)] if fast else WINDOWS):
        if endpoint > t[-1]:
            continue
        end = np.searchsorted(t, endpoint, side='right')-1
        start = np.searchsorted(t, t[end]-width)
        for layout in layouts:
            active = np.ones(len(left), dtype=bool)
            if layout == 'lower4':
                low = np.argsort(anchors[:, 2])[:4]
                active = np.isin(left, low)
                if right is not None:
                    active &= np.isin(right, low)
            idx, channel = np.nonzero(valid[start:end+1] & active)
            idx += start
            if len(idx) < 8:
                raise ValueError(f'Too few observations: {path.name}/{endpoint}/{width}/{layout}')
            tau = t[idx]-t[end]
            d = dp[idx]-dp[end]-tau[:, None]*dv[end]
            a = anchors[left[channel]]; b = None if right is None else anchors[right[channel]]
            y = z[idx, channel]-bias_z[channel]; scale = sigma[channel]
            if mode == 'twr':
                cv = cv_candidate(tau, a, y, scale)[0]
                yaw = lifted_candidate(tau, d, a, y, scale)[0][6]
                paper = solve_window(tau, d, a, y, scale, 'profile')
            else:
                # Native differences have no TWR square lift. Fit a radio CV from the surveyed centroid.
                cv = fit_track(tau, d, a, b, y, scale, np.r_[anchors.mean(0), np.zeros(3)], inertial=False)[0]
                yaw = 0.
            seed = np.r_[cv, yaw]
            base = defect_basis(tau)
            specs = [('rigid', None, True), ('constant_defect', (.5*tau**2)[:, None], True),
                     ('radio_spline', np.sqrt(RADIO_PSD)*base, False), ('imu_spline', np.sqrt(DEFECT_PSD)*base, True)]
            if sensitivity:
                specs += [('imu_q025', .5*base, True), ('imu_q4', 2*base, True)]
            fits = [(label, *fit_track(tau, d, a, b, y, scale, seed, B, use)) for label, B, use in specs]
            if mode == 'twr':
                fits.append(('paper_profile', paper.parameters, paper.robust_objective, paper.evaluations,
                             paper.diagnostics['solver_success']))
            for label, x, cost, nfev, success in fits:
                err = x[:3]-ref[end]
                result.append(dict(mode=mode, sequence=path.stem, layout=layout, end_s=endpoint,
                    actual_end_s=t[end], window_s=width, events=len(y), method=label, cost=cost, nfev=nfev,
                    success=success, error_m=np.linalg.norm(err), dx_m=err[0], dy_m=err[1], dz_m=err[2],
                    px_m=x[0], py_m=x[1], pz_m=x[2]))
    return result


def main():
    """Score physical time windows once and preserve paired, per-recording outcomes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DATASET)
    parser.add_argument('--output', type=Path, default=HERE/'results_windows')
    parser.add_argument('--modes', nargs='+', choices=['twr','tdoa'], default=['twr','tdoa'])
    parser.add_argument('--fast', action='store_true'); parser.add_argument('--sensitivity', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    anchors = load_anchors(args.root); rows = []; used = []
    for mode in args.modes:
        train, test = recordings(args.root, mode)
        cal = calibration(args.root, train, mode, anchors)
        for split, files in [('calibration',train), ('evaluation',test)]:
            used += [dict(mode=mode, split=split, file=p.name) for p in files]
        for path in test:
            # All8 is the cross-modality test; the lower4 TWR geometry reproduces the paper stress.
            layouts = ('all8','lower4') if mode == 'twr' else ('all8',)
            rows.extend(run_record(args.root, path, mode, anchors, cal, layouts, args.fast, args.sensitivity))
            print(f'{path.stem}: {len(rows)} cumulative fits', flush=True)
    df = pd.DataFrame(rows); df.to_csv(args.output/'windows.csv', index=False)
    pd.DataFrame(used).to_csv(args.output/'recordings.csv', index=False)
    def summarize(errors):
        """Summarize metres without treating overlapping windows as independent trials."""
        return pd.Series(dict(n=len(errors),median=np.median(errors),p95=np.quantile(errors,.95),
                              rmse=np.sqrt(np.mean(errors**2))))
    for keys, name in [(('mode','layout','method'),'summary'), (('mode','layout','sequence','method'),'per_recording')]:
        stats = df.groupby(list(keys)).error_m.apply(summarize).unstack()
        stats.to_csv(args.output/(name+'.csv'))
        if name == 'summary':
            print(stats.round(3).to_string(), flush=True)


if __name__ == '__main__':
    main()
