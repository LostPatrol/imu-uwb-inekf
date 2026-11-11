"""Test whether an effective anchor geometry learned with TWR transfers to native TDoA.

Apparent offsets are shrinkage estimates of direction-dependent range error,
not an independent physical survey. Evaluation records never enter calibration.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from delft_windows import (DATASET, HERE, INITIAL_IMU_PREFIX_S, WINDOWS, calibration, cv_candidate,
                           fit_track, load_anchors, read_record, recordings)
from rotation_window import local_shape, solve_attitude
from window_dynamics import defect_basis, robust_rows

# Fit at most one transaction per link per quarter-second to reduce held-channel redundancy.
CAL_BIN_S = .25
# The apparent displacement is softly shrunk toward the survey, in metres per component.
OFFSET_STD = .20
# Fixed radio scale for the TDoA test that receives no target-mode calibration at all.
UNCAL_STD = .30


def training_events(root, mode, paths):
    """Collect geometries at real training transactions; retain at most one per link per bin."""
    parts=[]
    for path in paths:
        rec,z,valid,plus,minus = read_record(root,path,mode)
        rows,channel=np.nonzero(valid)
        key=np.column_stack(((rec.time[rows]/CAL_BIN_S).astype(int),channel))
        index=np.sort(np.unique(key,axis=0,return_index=True)[1])
        r=rows[index]; c=channel[index]
        neg=np.full(len(c),-1) if minus is None else minus[c]
        parts.append((rec.ground_truth_position[r],z[r,c],plus[c],neg,c))
    return tuple(np.concatenate(s) for s in zip(*parts))


def fit_geometry(anchors, blocks):
    """Estimate shared apparent anchor locations and distinct per-channel offsets on training data."""
    indexed=[]; means=[]; stds=[]
    for (position,z,plus,minus,channel), (bias,scale) in blocks:
        indexed.append((position,z,plus,minus,channel+len(means)))
        means.extend(bias); stds.extend(scale)
    position,z,plus,minus,channel=(np.concatenate(x) for x in zip(*indexed))
    std=np.array(stds)[channel]; diff=minus>=0; idx=np.arange(len(z))

    def factor(x, jac=False):
        """Differentiate exact Euclidean radio factors and the Gaussian survey prior."""
        apparent=anchors+x[:anchors.size].reshape(anchors.shape)
        v=position-apparent[plus]; norm=np.maximum(np.linalg.norm(v,axis=1),1e-12)
        pred=norm+x[anchors.size:][channel]
        vm=position[diff]-apparent[minus[diff]]; nm=np.maximum(np.linalg.norm(vm,axis=1),1e-12)
        pred[diff]-=nm
        residual, deriv=robust_rows((pred-z)/std)
        if not jac:
            return np.r_[residual,x[:anchors.size]/OFFSET_STD]
        j=np.zeros((len(z)+anchors.size,len(x)))
        for axis in range(3):
            j[idx,3*plus+axis]=-v[:,axis]/norm
            j[idx[diff],3*minus[diff]+axis]=vm[:,axis]/nm
        j[idx,anchors.size+channel]=1.
        j[:len(z)]*=(deriv/std)[:,None]
        j[len(z):,:anchors.size]=np.eye(anchors.size)/OFFSET_STD
        return j

    opt=least_squares(factor,np.r_[np.zeros(anchors.size),means],jac=lambda x:factor(x,True),max_nfev=80)
    if not opt.success:
        raise RuntimeError(opt.message)
    return anchors+opt.x[:anchors.size].reshape(anchors.shape)


def score_record(root,path,mode,reference,scenarios):
    """Use a single fixed yaw-free SO(3) solve and matched radio-only control per scenario."""
    rec,z,valid,left,right=read_record(root,path,mode)
    origin=rec.time[np.flatnonzero(rec.time>=INITIAL_IMU_PREFIX_S)[0]]
    t=rec.time-origin; output=[]
    for end,width in WINDOWS:
        if end>t[-1]:
            continue
        e=np.searchsorted(t,end,side='right')-1; k=np.searchsorted(t,t[e]-width)
        ts=t[k:e+1]; imu=rec.imu[k:e+1]; full=np.arange(len(ts))
        f,_,_,_=local_shape(ts,imu[:,:3]*9.81,np.deg2rad(imu[:,3:]),full)
        for layout in (('all8','lower4') if mode=='twr' else ('all8',)):
            # Membership always comes from the physical survey, never the fitted offsets.
            active=np.ones(len(left),bool) if layout=='all8' else np.isin(left,np.argsort(reference[:,2])[:4])
            j,c=np.nonzero(valid[k:e+1]&active); tau=ts[j]-ts[-1]
            phi=defect_basis(tau)
            for label,apparent,bias,scale in scenarios:
                a=apparent[left[c]]; b=None if right is None else apparent[right[c]]
                y=z[k+j,c]-bias[c]; s=scale[c]
                if b is None:
                    cv=cv_candidate(tau,a,y,s)[0]
                else:
                    cv=fit_track(tau,None,a,b,y,s,np.r_[reference.mean(0),np.zeros(3)],inertial=False)[0]
                # This one fitted path is both the radio-only comparator and the inertial solver's seed.
                radio,_,ne,ok=fit_track(tau,None,a,b,y,s,cv,phi,False)
                path_seed=radio[:3]+tau[:,None]*radio[3:6]+phi@radio[6:].reshape(-1,3)
                fit=solve_attitude(tau,f[j],a,b,y,s,path_seed)
                for model,point,steps,success in [('radio_only',radio[:3],ne,ok),
                                                  ('so3',fit['x'][:3],fit['nfev'],fit['success'])]:
                    err=point-rec.ground_truth_position[e]
                    output.append(dict(mode=mode,sequence=path.stem,layout=layout,end_s=end,window_s=width,
                        method=model,geometry=label,error_m=np.linalg.norm(err),dx_m=err[0],dy_m=err[1],
                        dz_m=err[2],nfev=steps,success=success,events=len(c)))
    return output


def main():
    """Fit TWR-only, TDoA-only and pooled geometry before evaluating any switched model."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=DATASET)
    parser.add_argument('--output',type=Path,default=HERE/'results_geometry_transfer')
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    nominal=load_anchors(args.root); modes=('twr','tdoa')
    splits={m:recordings(args.root,m) for m in modes}
    cal={m:calibration(args.root,splits[m][0],m,nominal) for m in modes}
    blocks={m:(training_events(args.root,m,splits[m][0]),cal[m]) for m in modes}
    geometry={'survey':nominal,'twr_fit':fit_geometry(nominal,[blocks['twr']]),
              'tdoa_fit':fit_geometry(nominal,[blocks['tdoa']]),'joint_fit':fit_geometry(nominal,list(blocks.values()))}
    offsets=[dict(source=key,anchor=i,dx=dr[0],dy=dr[1],dz=dr[2],norm_m=np.linalg.norm(dr))
             for key,val in geometry.items() for i,dr in enumerate(val-nominal)]
    pd.DataFrame(offsets).to_csv(args.output/'offsets.csv',index=False); rows=[]
    for mode in modes:
        scenarios=[]
        for key,loc in geometry.items():
            # Recenter per channel on identical training flights; keep the nominal scales unchanged.
            med=cal[mode][0] if key=='survey' else calibration(args.root,splits[mode][0],mode,loc)[0]
            scenarios.append((key,loc,med,cal[mode][1]))
        if mode=='tdoa':
            for key in ('survey','twr_fit'):
                scenarios.append((key+'_no_target',geometry[key],np.zeros_like(cal[mode][0]),
                                  np.full_like(cal[mode][1],UNCAL_STD)))
        for path in splits[mode][1]:
            rows.extend(score_record(args.root,path,mode,nominal,scenarios))
            print(f'{path.stem}: {len(rows)} cumulative fits',flush=True)
    frame=pd.DataFrame(rows); frame.to_csv(args.output/'windows.csv',index=False)
    for extra,name in [([],'summary'),(['sequence'],'per_recording')]:
        stats=frame.groupby(['mode','layout',*extra,'method','geometry']).agg(n=('error_m','size'),
            rmse=('error_m',lambda x:np.sqrt(np.mean(x**2))),median=('error_m','median'),
            p95=('error_m',lambda x:x.quantile(.95)),capped=('success',lambda x:np.sum(~x)))
        stats.to_csv(args.output/(name+'.csv'))
        if not extra:
            print(stats.round(3).to_string())


if __name__=='__main__':
    main()
