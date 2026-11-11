"""Transfer frozen Delft early-XY requirements to native UTIL TDoA observations.

Reserved recovery packets are never scalar-fused before their XY/PV handoff.
All trajectories finish sensor-only generation before truth-based scoring.
"""

from __future__ import annotations

import argparse
import ast
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

# Each independent recording runs one numerical thread.
for _name in ("OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","OMP_NUM_THREADS"):
    os.environ[_name]="1"

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

# Reuse the native physical filter, without importing Delft's TWR data frontend.
HERE=Path(__file__).resolve().parent
UTIL=HERE.parent/"util"
DELFT=HERE.parent/"delft"/"early_position.py"
sys.path.insert(0,str(UTIL))
spec=importlib.util.spec_from_file_location("early_tdoa_util",UTIL/"run.py")
util=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=util
spec.loader.exec_module(util)

# These literal requirements are read verbatim from the frozen Delft source.
THRESHOLD_NAMES=("CHECK_INTERVAL_S","RANGE_STD_FLOOR_M","POSITION_STD_FLOOR_M",
    "UNMODELED_ACCEL_MPS2","MAX_95_HORIZONTAL_RADIUS_M","CHI2_95_2D",
    "MAX_NORMALIZED_RMS","MAX_SPEED_MPS","MIN_DISTINCT_ANCHORS",
    "MIRROR_XY_TOLERANCE_M","RANK_TOLERANCE")
FROZEN={node.targets[0].id:ast.literal_eval(node.value) for node in ast.parse(DELFT.read_text(encoding="utf8")).body
    if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name)
    and node.targets[0].id in THRESHOLD_NAMES}
assert len(FROZEN)==len(THRESHOLD_NAMES)
globals().update(FROZEN)
# This complete trial was chosen before any of its scores were examined.
TRIAL=1
GEOMETRIES=(1,2,3,4)
PROTOCOLS=(2,3)
YAWS=(0.,180.)
DURATION=32.
OUTAGE=(10.,20.)
SCORE_WINDOWS={"early":(20.,22.),"future":(26.,31.)}
METHODS=("native","fixed_2s_pv","early_xy","early_switch")
# The two native CV seeds and reflected fit retain Delft's evaluation caps.
CV_CAP=120
MIRROR_CAP=80
HUBER_SCALE=2.
VELOCITY_STD_FLOOR_MPS=1.


def save_json(path,value):
    """Serialize diagnostic arrays without consulting scoring truth."""
    path.write_text(json.dumps(util.safe_json(value),indent=2),encoding="utf8")


def physical_covariance(state,covariance):
    """Recover physical body-right attitude, world velocity and world position errors."""
    inverse=util.kernel.inverse_mapping(state)
    result=inverse@covariance@inverse.T
    return .5*(result+result.T)


def transport(state,physical):
    """Map a physical covariance into the native invariant state chart."""
    mapping=util.kernel.mapping(state)
    result=mapping@physical@mapping.T
    return .5*(result+result.T)


def native_model(parameters,tau,minus,plus,measurements,sigma):
    """Return native rho_b-rho_a CV residuals and their exact translation Jacobian."""
    tag=parameters[:3]+tau[:,None]*parameters[3:]
    da,db=tag-minus,tag-plus
    ra=np.maximum(np.linalg.norm(da,axis=1),1e-12)
    rb=np.maximum(np.linalg.norm(db,axis=1),1e-12)
    gradient=db/rb[:,None]-da/ra[:,None]
    return (rb-ra-measurements)/sigma,np.column_stack((gradient,tau[:,None]*gradient))/sigma[:,None]


def robust_cost(residual):
    """Use the same twice-Huber objective as the frozen Delft comparison."""
    absolute=np.abs(residual)
    return float(np.sum(np.where(absolute<=2.,residual**2,4.*absolute-4.)))


def cv_candidate(observations,anchors,endpoint,sigma):
    """Fit two native CV starts without inertial position or evaluation truth seeds."""
    tau=observations[:,0]-endpoint
    a,b=observations[:,1:3].astype(int).T
    minus,plus=anchors[a],anchors[b]
    y=observations[:,3]
    # Squared-hyperbola linearization supplies only a seed; its free per-anchor
    # ranges are discarded before the original TDoA likelihood is optimized.
    links=np.eye(len(anchors))[a]
    design=np.column_stack((-(plus-minus),-y[:,None]*links))
    response=.5*(y*y-sigma*sigma-np.sum(plus*plus,axis=1)+np.sum(minus*minus,axis=1))
    position=np.linalg.lstsq(design/sigma[:,None],response/sigma,rcond=1e-10)[0][:3]
    seeds=(np.r_[position,np.zeros(3)],np.r_[np.mean(anchors,axis=0),np.zeros(3)])
    def model(parameters):
        """Evaluate the frozen native residual and analytic Jacobian."""
        return native_model(parameters,tau,minus,plus,y,sigma)
    fits=[least_squares(lambda x:model(x)[0],seed,jac=lambda x:model(x)[1],
        loss="huber",f_scale=HUBER_SCALE,max_nfev=CV_CAP) for seed in seeds]
    fit=min(fits,key=lambda item:robust_cost(item.fun))
    return fit.x,fit.fun,model(fit.x)[1],dict(seed_costs=[robust_cost(item.fun) for item in fits],
        nfev=[item.nfev for item in fits],success=bool(fit.success))


def horizontal_candidate(data,start,end,state,covariance):
    """Apply Delft's XY Schur, absolute ellipse, prediction and relevant-mirror gates."""
    observations=data["radio"][(data["radio"][:,0]>=start)&(data["radio"][:,0]<=end)]
    ids=observations[:,1:3].astype(int)
    distinct=len(np.unique(ids))
    report=dict(time_s=float(end-data["times"][0]),events=len(observations),
        anchors=distinct,accepted=False)
    if distinct<MIN_DISTINCT_ANCHORS:
        return None,dict(report,reason="insufficient_anchors")
    tau=observations[:,0]-end
    nominal=np.sqrt(.22 if "const4" in data["name"] else .14)
    sigma=np.sqrt(max(nominal,RANGE_STD_FLOOR_M)**2+(.5*UNMODELED_ACCEL_MPS2*tau*tau)**2)
    estimate,residual,jacobian,fit=cv_candidate(observations,data["anchors"],end,sigma)
    used_anchors=data["anchors"][np.unique(ids)]
    mirror=estimate.copy()
    mirror[2]=2*np.mean(used_anchors[:,2])-mirror[2]
    mirror[5]*=-1
    minus,plus=data["anchors"][ids[:,0]],data["anchors"][ids[:,1]]
    def model(parameters):
        """Evaluate the reflected solution under exactly the same native likelihood."""
        return native_model(parameters,tau,minus,plus,observations[:,3],sigma)
    alternative=least_squares(lambda x:model(x)[0],mirror,jac=lambda x:model(x)[1],
        loss="huber",f_scale=HUBER_SCALE,max_nfev=MIRROR_CAP)
    comparable=2*alternative.cost<=robust_cost(residual)+2.
    mirror_xy=float(np.linalg.norm(alternative.x[:2]-estimate[:2]))
    weights=np.minimum(1.,2./np.maximum(np.abs(residual),1e-12))
    weighted=jacobian*np.sqrt(weights[:,None])
    nuisance=weighted[:,[2,3,4,5]]
    left,singular,_=np.linalg.svd(nuisance,full_matrices=False)
    rank=int(np.sum(singular>singular[0]*RANK_TOLERANCE)) if singular.size else 0
    target=weighted[:,:2]
    projected=target-left[:,:rank]@(left[:,:rank].T@target)
    info=projected.T@projected
    info=.5*(info+info.T)
    eigenvalues=np.linalg.eigvalsh(info)
    report.update(nuisance_rank=rank,horizontal_information_eigenvalues=eigenvalues,
        normalized_rms=float(np.sqrt(np.mean(residual**2))),speed_mps=float(np.linalg.norm(estimate[3:])),
        mirror_xy_separation_m=mirror_xy,comparable_mirror=bool(comparable),
        fitted_z_m=float(estimate[2]),fitted_velocity=estimate[3:],fit=fit,
        anchor_height_span_m=float(np.ptp(used_anchors[:,2])),
        candidate_latest_time_s=float(observations[-1,0]-data["times"][0]))
    if eigenvalues[0]<=max(eigenvalues[-1],1.)*RANK_TOLERANCE:
        return None,dict(report,reason="horizontal_rank_deficient")
    xy_covariance=np.linalg.inv(info)+np.eye(2)*POSITION_STD_FLOOR_M**2
    radius=float(np.sqrt(CHI2_95_2D*np.linalg.eigvalsh(xy_covariance)[-1]))
    prior=physical_covariance(state,covariance)[6:8,6:8]
    advantage=float(np.linalg.eigvalsh(.5*((prior-xy_covariance)+(prior-xy_covariance).T))[0])
    # The native frontend fits antenna position; the filter state is the IMU origin.
    position=estimate[:2]-(state[:3,:3]@util.LEVER)[:2]
    report.update(horizontal_covariance=xy_covariance,radius_95_m=radius,
        prior_advantage_min_eigenvalue_m2=advantage,position_xy=position,
        fitted_antenna_xy=estimate[:2])
    reason="accepted"
    if report["normalized_rms"]>MAX_NORMALIZED_RMS:
        reason="radio_model_mismatch"
    elif report["speed_mps"]>MAX_SPEED_MPS:
        reason="velocity_nuisance_implausible"
    elif comparable and mirror_xy>MIRROR_XY_TOLERANCE_M:
        reason="horizontal_mirror_ambiguity"
    elif radius>MAX_95_HORIZONTAL_RADIUS_M:
        reason="horizontal_usage_requirement"
    elif advantage<=0:
        reason="not_better_than_prediction"
    report.update(reason=reason,accepted=reason=="accepted")
    return (position,xy_covariance) if report["accepted"] else None,report


def handoff_xy(state,covariance,bias,candidate):
    """Replace XY and its covariance while retaining every other physical marginal."""
    old_state,old_bias=state.copy(),bias.copy()
    physical=physical_covariance(state,covariance)
    retained=np.r_[0:6,8:15]
    old_retained=physical[np.ix_(retained,retained)].copy()
    state=state.copy()
    state[:2,4]=candidate[0]
    physical[6:8,:]=0.
    physical[:,6:8]=0.
    physical[6:8,6:8]=candidate[1]
    covariance=transport(state,physical)
    recovered=physical_covariance(state,covariance)
    checks=dict(rotation=float(np.max(np.abs(state[:3,:3]-old_state[:3,:3]))),
        velocity=float(np.max(np.abs(state[:3,3]-old_state[:3,3]))),
        vertical_position=abs(float(state[2,4]-old_state[2,4])),
        bias=float(np.max(np.abs(bias-old_bias))),
        retained_covariance=float(np.max(np.abs(recovered[np.ix_(retained,retained)]-old_retained))))
    return state,covariance,bias,checks


def fixed_pv(data,start,end,state,covariance,bias):
    """Set Delft's fixed two-second radio PV with the same uncertainty floors."""
    observations=data["radio"][(data["radio"][:,0]>=start)&(data["radio"][:,0]<=end)]
    nominal=np.sqrt(.22 if "const4" in data["name"] else .14)
    estimate,residual,jacobian,fit=cv_candidate(observations,data["anchors"],end,
        np.full(len(observations),nominal))
    weights=np.minimum(1.,2./np.maximum(np.abs(residual),1e-12))
    curvature=np.linalg.inv((jacobian.T*weights)@jacobian+
        np.diag(np.r_[np.full(3,1/8.**2),np.full(3,1/2.**2)]))
    source=physical_covariance(state,covariance)
    physical=np.zeros((15,15))
    retained=np.r_[0:3,9:15]
    physical[np.ix_(retained,retained)]=source[np.ix_(retained,retained)]
    order=np.r_[6:9,3:6]
    physical[np.ix_(order,order)]=curvature+np.diag(np.r_[np.full(3,POSITION_STD_FLOOR_M**2),
        np.full(3,VELOCITY_STD_FLOOR_MPS**2)])
    state=state.copy()
    state[:3,4]=estimate[:3]-state[:3,:3]@util.LEVER
    state[:3,3]=estimate[3:]
    return state,transport(state,physical),bias,dict(events=len(observations),
        fitted_pv=estimate,normalized_rms=float(np.sqrt(np.mean(residual**2))),fit=fit)


def row_at(state,bias,stamp):
    """Store one post-event physical state in the shared trace format."""
    return np.r_[stamp,state[:3,4],state[:3,3],state[:3,:3].ravel(),bias[3:]]


def run_case(data,common,yaw,output):
    """Keep a shared native prefix, then evaluate reserved-packet early recovery."""
    origin=float(data["times"][0])
    k10,k20,k22=[int(np.searchsorted(data["times"],origin+value)) for value in (10.,20.,22.)]
    t10,t20,t22=data["times"][[k10,k20,k22]]
    state,covariance,bias,prefix=common
    state,covariance,bias=state.copy(),covariance.copy(),bias.copy()
    physical=physical_covariance(state,covariance)
    state[:3,:3]=Rotation.from_euler("z",yaw,degrees=True).as_matrix()@state[:3,:3]
    covariance=transport(state,physical)
    empty=dict(data,radio=np.empty((0,4)))
    state,covariance,bias,gap,_,_,failed=util.bounded_replay(empty,state,covariance,bias,t10,t20,None)
    common_rows=np.vstack((prefix,gap))
    states={method:(state.copy(),covariance.copy(),bias.copy()) for method in METHODS[:3]}
    histories={method:[common_rows] for method in METHODS[:3]}
    failures={method:bool(failed) for method in METHODS[:3]}
    trigger=None
    attempts=[]
    preservation={}
    fixed_diagnostic=None
    previous=t20
    # Existing 100 Hz IMU ticks define the original 50 ms physical check schedule.
    check_indices=np.r_[np.arange(k20,len(data["times"]),5),k22,len(data["times"])-1]
    check_indices=np.unique(check_indices)
    for index in check_indices:
        stamp=float(data["times"][index])
        for method in METHODS[:3]:
            if failures[method]:
                continue
            X,P,b=states[method]
            blocked=(method=="fixed_2s_pv" and previous<t22) or (method=="early_xy" and trigger is None)
            X,P,b,rows,_,_,bad=util.bounded_replay(empty if blocked else data,X,P,b,previous,stamp,None)
            states[method]=(X,P,b)
            histories[method].append(rows)
            failures[method]=bool(bad)
        if trigger is None and not failures["early_xy"]:
            X,P,b=states["early_xy"]
            window_start=float(data["times"][max(k20,int(np.searchsorted(data["times"],stamp-2.)))])
            candidate,report=horizontal_candidate(data,window_start,stamp,X,P)
            attempts.append(report)
            if candidate is not None:
                X,P,b,preservation=handoff_xy(X,P,b,candidate)
                states["early_xy"]=(X,P,b)
                trigger=stamp
                rows=histories["early_xy"][-1]
                if len(rows) and abs(rows[-1,0]-stamp)<1e-9:
                    rows[-1]=row_at(X,b,stamp)
        if stamp==t22 and not failures["fixed_2s_pv"]:
            X,P,b=states["fixed_2s_pv"]
            X,P,b,fixed_diagnostic=fixed_pv(data,t20,t22,X,P,b)
            states["fixed_2s_pv"]=(X,P,b)
            rows=histories["fixed_2s_pv"][-1]
            if len(rows) and abs(rows[-1,0]-stamp)<1e-9:
                rows[-1]=row_at(X,b,stamp)
        previous=stamp
    arrays={method:np.vstack(histories[method]) for method in METHODS[:3]}
    # This optional display switches outputs only; the reserved filter is unchanged.
    native,early=arrays["native"],arrays["early_xy"]
    switched=native.copy()
    if trigger is not None:
        switched=np.vstack((native[native[:,0]<trigger],early[early[:,0]>=trigger]))
    arrays["early_switch"]=switched
    failures["early_switch"]=failures["early_xy"] if trigger is not None else failures["native"]
    identifier=f"{data['name']}__yaw{yaw:g}"
    np.savez_compressed(output/(identifier+".npz"),**arrays)
    record=dict(name=data["name"],yaw_offset_deg=yaw,origin=origin,trigger_s=None if trigger is None else trigger-origin,
        attempts=attempts,preservation=preservation,fixed_handoff=fixed_diagnostic,failures=failures,
        artifact=str(output/(identifier+".npz")),common_prefix_identical=True,
        latest_reserved_packet_s=None if trigger is None else float(data["radio"][
            data["radio"][:,0]<=trigger][-1,0]-origin))
    save_json(output/(identifier+".json"),record)
    return record


def run_record(job):
    """Generate both yaw scenarios using sensors only and a common native 10 s prefix."""
    name,output_root=job
    output=Path(output_root)/"histories"
    output.mkdir(parents=True,exist_ok=True)
    try:
        data=util.load(name,duration=DURATION)
    except FileNotFoundError as error:
        return [dict(name=name,missing=True,error=str(error))]
    data.pop("truth")
    initial_time,state,bias=util.initial_state(data)
    covariance=transport(state,util.PHYSICAL)
    t10=float(data["times"][np.searchsorted(data["times"],data["times"][0]+10.)])
    state,covariance,bias,rows,_,_,failed=util.bounded_replay(data,state,covariance,bias,initial_time,t10,None)
    if failed:
        return [dict(name=name,prefix_failed=True)]
    common=(state,covariance,bias,rows)
    return [run_case(data,common,yaw,output) for yaw in YAWS]


def score(output,records):
    """Use tracking truth only after all sensor-only cases have completed."""
    truths={}
    metrics=[]
    for record in records:
        if record.get("missing") or record.get("prefix_failed"):
            continue
        name=record["name"]
        if name not in truths:
            truths[name]=util.load(name,duration=DURATION)["truth"]
        truth=truths[name]
        arrays=np.load(record["artifact"])
        for method in METHODS:
            rows=arrays[method]
            metric=dict(name=name,yaw_offset_deg=record["yaw_offset_deg"],method=method,
                trigger_s=record["trigger_s"],triggered=record["trigger_s"] is not None,
                failed=record["failures"][method],last_output_s=float(rows[-1,0]-record["origin"]))
            for label,(start,end) in SCORE_WINDOWS.items():
                chosen=rows[(rows[:,0]>=record["origin"]+start)&(rows[:,0]<record["origin"]+end)]
                values=util.score_rows(truth,chosen)
                metric.update({label+"_"+key:value for key,value in values.items()})
                actual=np.column_stack([np.interp(chosen[:,0],truth[:,0],truth[:,k]) for k in (1,2)])
                errors=np.linalg.norm(chosen[:,1:3]-actual,axis=1)
                metric[label+"_xy_p95"]=float(np.percentile(errors,95)) if len(errors) else None
                metric[label+"_xy_peak"]=float(np.max(errors)) if len(errors) else None
                metric[label+"_coverage"]=min(1.,len(chosen)/round((end-start)/.05))
            metrics.append(metric)
    pd.DataFrame(metrics).to_csv(output/"metrics.csv",index=False)


def main():
    """Write immutable settings before processing the entire unseen trial-1 matrix."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers",type=int,default=2)
    parser.add_argument("--score-only",action="store_true")
    args=parser.parse_args()
    output=HERE/"results"
    output.mkdir(parents=True,exist_ok=True)
    if args.score_only:
        score(output,json.loads((output/"records.json").read_text(encoding="utf8")))
        return
    names=[util.record_name(g,TRIAL,p) for g in GEOMETRIES for p in PROTOCOLS]
    save_json(output/"protocol.json",dict(trial=TRIAL,names=names,yaws_deg=YAWS,
        outage_s=OUTAGE,score_windows=SCORE_WINDOWS,thresholds=FROZEN,methods=METHODS,
        yaw_perturbation_time_s=10.,duration_s=DURATION,cv_seed_caps=[CV_CAP,CV_CAP],mirror_cap=MIRROR_CAP,
        sigma="Native std sqrt(.14) / sqrt(.22) for geometry4; frozen .30m floor and 3m/s2 mismatch inflation",
        handoff="XY only: antenna-to-IMU lever subtraction; retain z/v/R/bias and their complete physical covariance block; clear XY cross blocks",
        information="Native fuses each recovery packet once. Fixed and early reserve recovery packets until their handoff; thereafter only strictly later raw packets are fused.",
        trigger="First frozen acceptance at existing 100Hz IMU ticks every5 samples after20; Delft rolling maximum2s buffer; no truth selection",
        no_accept="Early remains prediction-only through end; optional early_switch retains native output throughout",
        initial="First .6s sensor-only PV, .2s gravity, common native prefix to10; world-z stress at10",
        truth="Discard loader truth immediately; score only after all requested trajectories finish",
        site_claim="Cross-source TWR-to-TDoA transfer, distinct UTIL trial1; not a blinded multi-site deployment",
        expected_cases=16,expected_traces=64,workers=args.workers,
        source_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in
            (Path(__file__),DELFT,UTIL/"run.py",UTIL/"kernel.py",UTIL/"protocol.json")}))
    records=[]
    started=time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(run_record,(name,str(output))):name for name in names}
        for future in as_completed(futures):
            records.extend(future.result())
            save_json(output/"records.json",records)
            print(f"DONE {futures[future]} {len(records)}/16 cases {time.perf_counter()-started:.1f}s",flush=True)
    print("Sensor-only generation finished; starting fixed-window truth scoring.",flush=True)
    score(output,records)


if __name__=="__main__":
    main()
