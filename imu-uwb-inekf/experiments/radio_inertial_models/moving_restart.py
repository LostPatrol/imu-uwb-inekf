"""Test full SO(3) restart using only inertial samples inside each Delft radio window."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from delft_windows import (DATASET, HERE, INITIAL_IMU_PREFIX_S, WINDOWS, calibration, cv_candidate,
                           fit_track, load_anchors, read_record, recordings)
from rotation_window import GRAVITY, local_shape, radio_seed_path, solve_attitude


def gravity_seed(force, dt):
    """Align the window's mean local force to up, leaving heading at minimum rotation."""
    mean = np.sum(dt[:, None]*force, axis=0)
    if np.linalg.norm(mean) < 1e-10:
        return np.eye(3)
    c, _ = Rotation.align_vectors([(-GRAVITY)/np.linalg.norm(GRAVITY)], [mean/np.linalg.norm(mean)])
    return c.as_matrix()


def evaluate(root, path, mode, anchors, model):
    """Score non-overlapping data dependencies: the pre-window IMU never enters a solve."""
    rec, z, valid, left, right = read_record(root, path, mode)
    # The 1.5 s offset reproduces the paper's scoring clock only; no alignment prefix is used.
    origin = rec.time[np.flatnonzero(rec.time >= INITIAL_IMU_PREFIX_S)[0]]
    time = rec.time-origin
    bias_z, std = model; output=[]
    for endpoint, width in WINDOWS:
        if endpoint > time[-1]:
            continue
        e = np.searchsorted(time, endpoint, side='right')-1
        k = np.searchsorted(time, time[e]-width)
        ts = time[k:e+1]; imu=rec.imu[k:e+1]
        raw_acc = imu[:,:3]*9.81; raw_gyro=np.deg2rad(imu[:,3:])
        full = np.arange(len(ts))
        f, bmat, q, _ = local_shape(ts, raw_acc, raw_gyro, full)
        tilt = gravity_seed(np.einsum('nij,nj->ni',q[:-1],raw_acc[:-1]), np.diff(ts))
        for layout in (('all8','lower4') if mode == 'twr' else ('all8',)):
            active=np.ones(len(left),bool) if layout=='all8' else np.isin(left,np.argsort(anchors[:,2])[:4])
            j, chan = np.nonzero(valid[k:e+1] & active)
            tau = ts[j]-ts[-1]; a=anchors[left[chan]]
            b = None if right is None else anchors[right[chan]]
            y=z[k+j,chan]-bias_z[chan]; sigma=std[chan]
            if b is None:
                cv=cv_candidate(tau,a,y,sigma)[0]
            else:
                cv=fit_track(tau,None,a,b,y,sigma,np.r_[anchors.mean(0),np.zeros(3)],inertial=False)[0]
            path_seed = radio_seed_path(tau,a,b,y,sigma,cv)
            for label, seed, ba in [('procrustes',None,None),('procrustes_bias',None,bmat[j]),
                                    ('gravity_bias',tilt,bmat[j]),('identity_bias',np.eye(3),bmat[j])]:
                fit=solve_attitude(tau,f[j],a,b,y,sigma,path_seed,ba,seed)
                err=fit['x'][:3]-rec.ground_truth_position[e]
                xyz=fit['x'][:3]; vel=fit['x'][3:6]
                output.append(dict(mode=mode,sequence=path.stem,layout=layout,end_s=endpoint,window_s=width,
                    actual_end_s=time[e],method=label,events=len(y),error_m=np.linalg.norm(err),
                    px_m=xyz[0],py_m=xyz[1],pz_m=xyz[2],vx=vel[0],vy=vel[1],vz=vel[2],
                    dz_m=err[2],success=fit['success'],nfev=fit['nfev'],cost=fit['cost'],
                    seed_s2=fit['singular'][1],min_radio_tau=tau.min(),available_imu_s=ts[-1]-ts[0]))
    return output


def main():
    """Rebuild all calibrated channels and score original TWR and TDoA recordings."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=DATASET)
    parser.add_argument('--output',type=Path,default=HERE/'results_restart')
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    anchors=load_anchors(args.root); rows=[]
    for mode in ('twr','tdoa'):
        train,test=recordings(args.root,mode); cal=calibration(args.root,train,mode,anchors)
        for path in test:
            rows+=evaluate(args.root,path,mode,anchors,cal)
            print(f'{path.stem}: {len(rows)} cumulative fits',flush=True)
    frame=pd.DataFrame(rows); frame.to_csv(args.output/'windows.csv',index=False)
    for keys,name in [(['mode','layout','method'],'summary'),(['mode','layout','sequence','method'],'per_recording')]:
        table=frame.groupby(keys).agg(rmse=('error_m',lambda v:np.sqrt(np.mean(v**2))),
            median=('error_m','median'),p95=('error_m',lambda v:np.quantile(v,.95)),
            n=('error_m','size'),capped=('success',lambda v:np.sum(~v)))
        table.to_csv(args.output/(name+'.csv'))
        if name=='summary':
            print(table.round(3).to_string())


if __name__=='__main__':
    main()
