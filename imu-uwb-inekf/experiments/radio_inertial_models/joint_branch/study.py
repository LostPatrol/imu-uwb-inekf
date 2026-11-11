"""Test frozen radio models jointly with K2 and rotating-body bias correction.

Every model uses the same static-radio CV translation and four yaw starts.
Only separate training recordings supplied radio parameters; truth only scores.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import time

# Limit BLAS concurrency before importing numerical libraries on the desktop.
for _name in ("OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","OMP_NUM_THREADS"):
    os.environ[_name]="1"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.optimize._numdiff import approx_derivative

# Only the new completed branches are imported as integration dependencies.
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT/"radio_branch"))
import study as radio
sys.path.insert(0,str(ROOT/"shape_branch"))
import physical_shape as shape

# Match the frozen radio comparison rather than searching joint hyperparameters.
MAX_NFEV=120
STARTS=4
TRAJECTORIES=("K2","BiasPrior")
RADIO_MODELS=("static","geometry","static_ar","geometry_ar")


def positions(x,tau,displacement,sensitivity,trajectory):
    """Evaluate the shared endpoint chart and each model's position derivatives."""
    rotation=radio.yaw_rotation(x[6])
    corrected=displacement.copy()
    if trajectory=="BiasPrior":
        corrected-=np.einsum("nij,j->ni",sensitivity,x[7:10])
    rotated=corrected@rotation.T
    position=x[:3]+tau[:,None]*x[3:6]+rotated
    derivative=np.zeros((len(tau),3,10))
    derivative[:,:,:3]=np.eye(3)
    derivative[:,:,3:6]=tau[:,None,None]*np.eye(3)
    derivative[:,:,6]=np.column_stack((-rotated[:,1],rotated[:,0],np.zeros(len(tau))))
    if trajectory=="BiasPrior":
        derivative[:,:,7:10]=-np.einsum("ij,njk->nik",rotation,sensitivity)
    else:
        # K2 is an unconstrained world-frame constant acceleration defect.
        position+=.5*tau[:,None]**2*x[7:10]
        derivative[:,:,7:10]=.5*tau[:,None,None]**2*np.eye(3)
    return position,derivative


def residual_jacobian(x,values,sensitivity,trajectory,radio_model,calibration):
    """Compose motion derivatives with trained radio mean and same-anchor whitening."""
    tau,displacement,anchors,ranges,ids=values
    position,derivative=positions(x,tau,displacement,sensitivity,trajectory)
    delta=position-anchors
    distance=np.maximum(np.linalg.norm(delta,axis=1),1e-9)
    direction=delta/distance[:,None]
    gradient=direction.copy()
    mean=calibration["static"][ids]
    if radio_model.startswith("geometry"):
        beta=calibration["beta"][ids]
        mean=beta[:,0]+beta[:,1]*(distance-calibration["centers"][ids])+np.sum(beta[:,2:]*direction,axis=1)
        gradient+=beta[:,1,None]*direction+(beta[:,2:]-np.sum(beta[:,2:]*direction,axis=1)[:,None]*direction)/distance[:,None]
    sigma=calibration["sigma"][ids]
    residual=(distance+mean-ranges)/sigma
    jacobian=np.einsum("ni,nij->nj",gradient,derivative)/sigma[:,None]
    if radio_model.endswith("_ar"):
        raw_residual,raw_jacobian=residual.copy(),jacobian.copy()
        for anchor in np.unique(ids):
            indices=np.flatnonzero(ids==anchor)
            current,previous=indices[1:],indices[:-1]
            factor=calibration["rho"][anchor]**((tau[current]-tau[previous])/calibration["interval"][anchor])
            denominator=np.sqrt(np.maximum(1-factor**2,.05))
            residual[current]=(raw_residual[current]-factor*raw_residual[previous])/denominator
            jacobian[current]=(raw_jacobian[current]-factor[:,None]*raw_jacobian[previous])/denominator[:,None]
    return residual,jacobian


def solve(values,sensitivity,trajectory,radio_model,calibration,seeds):
    """Use identical four-start caps while preserving the independent bias prior."""
    prior=np.zeros((3 if trajectory=="BiasPrior" else 0,10))
    if len(prior):
        prior[:,7:10]=np.eye(3)/shape.BIAS_STD
    def fun(x):
        """Append Gaussian calibration priors after the radio observations."""
        return np.r_[residual_jacobian(x,values,sensitivity,trajectory,radio_model,calibration)[0],prior@x]
    def jac(x):
        """Append exact fixed prior derivatives to the analytic radio Jacobian."""
        return np.vstack((residual_jacobian(x,values,sensitivity,trajectory,radio_model,calibration)[1],prior))
    fits=[least_squares(fun,seed,jac=jac,loss=lambda z:shape.mixed_loss(z,len(values[0])),
                        max_nfev=MAX_NFEV) for seed in seeds]
    best=min(fits,key=lambda fit:fit.cost)
    return best.x,dict(nfev=sum(fit.nfev for fit in fits),objective=2*best.cost,
        success=bool(best.success),seed_costs=[float(2*fit.cost) for fit in fits])


def verify(calibrations):
    """Check joint derivatives and exact agreement with the completed bias module."""
    rng=np.random.default_rng(91011)
    tau=np.linspace(-4.,0.,100)
    displacement=rng.normal(size=(100,3))*.2
    sensitivity=rng.normal(size=(100,3,3))*.1
    ids=np.arange(100)%8
    anchors=radio.load_anchors(radio.DATASET)[ids]
    ranges=np.linalg.norm(np.array([0.,0.,1.])-anchors,axis=1)
    values=(tau,displacement,anchors,ranges,ids)
    x=np.array([0.,0.,1.,.1,.2,.3,.7,.02,-.01,.03])
    checks={}
    for trajectory in TRAJECTORIES:
        for radio_model in RADIO_MODELS:
            cal=calibrations[radio_model]
            def fun(value):
                """Evaluate the joint factor for independent finite differences."""
                return residual_jacobian(value,values,sensitivity,trajectory,radio_model,cal)[0]
            analytic=residual_jacobian(x,values,sensitivity,trajectory,radio_model,cal)[1]
            checks[trajectory+"__"+radio_model]=float(np.max(np.abs(analytic-approx_derivative(fun,x))))
    cal=calibrations["static"]
    old_values=(tau,displacement,anchors,ranges-cal["static"][ids],cal["sigma"][ids])
    reference=shape.raw_model(x,old_values,sensitivity,False)
    current=residual_jacobian(x,values,sensitivity,"BiasPrior","static",cal)
    checks["bias_module_residual_match"]=float(np.max(np.abs(reference[0]-current[0])))
    checks["bias_module_jacobian_match"]=float(np.max(np.abs(reference[1]-current[1])))
    assert max(checks.values())<1e-6
    return checks


def main():
    """Freeze integration inputs, run 1056 paired fits and retain physical errors."""
    output=HERE/"results"
    output.mkdir(parents=True,exist_ok=True)
    previous=ROOT/"radio_branch"/"results"
    original=json.loads((previous/"protocol.json").read_text())
    raw_cal=json.loads((previous/"calibration.json").read_text())
    base={key:np.asarray(value) if isinstance(value,list) else value for key,value in raw_cal.items()}
    static_cal=json.loads((ROOT/"radio_branch"/"results_ar_ablation"/"calibration.json").read_text())
    static_ar={key:np.asarray(value) if isinstance(value,list) else value for key,value in static_cal.items()}
    calibrations={name:static_ar if name=="static_ar" else base for name in RADIO_MODELS}
    checks=verify(calibrations)
    anchors=radio.load_anchors(radio.DATASET)
    metadata=dict(training=original["training"],evaluation=original["evaluation"],
        trajectories=TRAJECTORIES,radio_models=RADIO_MODELS,
        common_sensor_delay_s=.20,static_prefix_s=1.5,starts=STARTS,max_nfev_per_start=MAX_NFEV,
        seeds="Identical static-calibrated CV p/v, uniform yaw at -pi,-pi/2,0,pi/2; nuisance zero",
        bias_prior_std_mps2=shape.BIAS_STD,truth_used_only_for_scoring=True,
        k2_formula="p=p_end+tau*v_end+Rz(psi)*d+.5*tau^2*c_world; no prior on c_world",
        bias_formula="p=p_end+tau*v_end+Rz(psi)*(d-B(tau)*delta_b); delta_b~N(0,.1^2 I)",
        radio_formula="y=distance+trained_mean(anchor,distance,LOS)+correlated_error; frozen Huber AR innovations",
        no_post_result_parameter_changes=True,independent_blind_test=False,
        checks=checks,source_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in (
            Path(__file__),Path(inspect.getfile(radio)),Path(inspect.getfile(shape)),
            previous/"calibration.json",ROOT/"radio_branch"/"results_ar_ablation"/"calibration.json")})
    radio.dump(output/"protocol.json",metadata)
    rows=[]
    for sequence in original["evaluation"]:
        log=radio.load_real_log(radio.DATASET,sequence,sensor_delay_s=.20)
        rotation,bias=radio.estimate_initial_rotation_and_bias(log)
        selected=log.time>=1.5
        times=log.time[selected]-log.time[selected][0]
        ranges,changed=log.ranges[selected],log.changed[selected]
        rotations,d,v=radio.preintegrate(times,log.imu[selected],rotation,bias)
        for endpoint in (5.,10.,20.,30.):
            for duration in (2.,4.,8.):
                if duration>endpoint:
                    continue
                end=np.searchsorted(times,endpoint,side="right")-1
                start=np.searchsorted(times,times[end]-duration,side="left")
                for layout in ("all8","lower4"):
                    mask=changed[start:end+1]&(ranges[start:end+1]>0)
                    if layout=="lower4":
                        mask[:,4:]=False
                    indices,ids=np.nonzero(mask)
                    indices+=start
                    tau=times[indices]-times[end]
                    displacement=d[indices]-d[end]-tau[:,None]*v[end]
                    sensitivity=shape.reanchor_integral(rotations[start:end],times[start:end+1],times[indices])
                    observations=ranges[indices,ids]
                    values=(tau,displacement,anchors[ids],observations,ids)
                    cv=radio.cv_candidate(tau,anchors[ids],observations-base["static"][ids],base["sigma"][ids])[0]
                    seeds=[np.r_[cv,angle,np.zeros(3)] for angle in np.linspace(-np.pi,np.pi,STARTS,endpoint=False)]
                    truth=log.ground_truth_position[selected][end]
                    for trajectory in TRAJECTORIES:
                        for radio_model in RADIO_MODELS:
                            started=time.perf_counter()
                            x,fit=solve(values,sensitivity,trajectory,radio_model,calibrations[radio_model],seeds)
                            rows.append(dict(sequence=sequence,endpoint_s=endpoint,actual_endpoint_s=times[end],
                                duration_s=duration,layout=layout,trajectory=trajectory,radio_model=radio_model,
                                events=len(tau),position_error_m=np.linalg.norm(x[:3]-truth),
                                elapsed_s=time.perf_counter()-started,x=x[0],y=x[1],z=x[2],
                                nuisance_norm=np.linalg.norm(x[7:10]),**fit))
        pd.DataFrame(rows).to_csv(output/"metrics.csv",index=False)
        print(f"DONE {sequence}: {len(rows)} fits",flush=True)
    frame=pd.DataFrame(rows)
    summarize(frame,output)


def summarize(frame,output):
    """Report paired windows and per-log wins without treating overlaps as iid."""
    for group,name in ((["trajectory","layout","radio_model"],"overall"),
        (["trajectory","duration_s","layout","radio_model"],"duration"),
        (["trajectory","sequence","layout","radio_model"],"per_log")):
        frame.groupby(group).position_error_m.agg(n="size",median="median",p95=lambda x:x.quantile(.95),
            rmse=lambda x:np.sqrt(np.mean(x*x)),maximum="max").to_csv(output/(name+"_summary.csv"))
    keys=["trajectory","sequence","endpoint_s","duration_s","layout"]
    paired=frame.pivot(index=keys,columns="radio_model",values="position_error_m")
    paired.to_csv(output/"paired_metrics.csv")
    rows=[]
    for key,part in paired.groupby(level=["trajectory","sequence","layout"]):
        for model in RADIO_MODELS[1:]:
            difference=part[model]-part.static
            rows.append(dict(zip(("trajectory","sequence","layout"),key),radio_model=model,
                wins=int((difference<0).sum()),windows=len(part),median_delta_m=difference.median(),
                rmse_static_m=np.sqrt(np.mean(part.static**2)),rmse_new_m=np.sqrt(np.mean(part[model]**2)),
                max_degradation_m=difference.max(),max_improvement_m=difference.min()))
    pd.DataFrame(rows).to_csv(output/"per_log_paired.csv",index=False)
    frame.groupby(["trajectory","radio_model"])[["elapsed_s","nfev"]].agg(["median","mean","max"]).to_csv(output/"costs.csv")
    fig,axes=plt.subplots(2,2,figsize=(11,8))
    for ax,((trajectory,layout),part) in zip(axes.flat,paired.groupby(level=["trajectory","layout"])):
        for model in RADIO_MODELS[1:]:
            ax.scatter(part.static,part[model],s=16,alpha=.65,label=model)
        ax.plot([.02,10],[.02,10],color="black",linewidth=.8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(trajectory+" / "+layout)
        ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    fig.supxlabel("Static radio endpoint error (m)")
    fig.supylabel("Candidate radio endpoint error (m)")
    fig.tight_layout()
    fig.savefig(output/"joint_comparison.png",dpi=180)
    plt.close(fig)


if __name__=="__main__":
    main()
