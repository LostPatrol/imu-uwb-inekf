"""Bias-aware raw-range initialization using actual rotating-IMU preintegration.

The optimization and curvature distinguish a calibrated accelerometer-bias prior
from information supplied by ranges. This module leaves the frozen base study intact.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from initializer import (solve_window, yaw_rotation, project_out, HUBER_SCALE,
                         MAX_EVALUATIONS, RANK_RTOL)
from run_research import (PROJECT, HERE, window_data, GAP_TRIGGER_S, WINDOWS_S,
                          YAW_STD_TARGET_RAD, MAX_NORMALIZED_RMS, POSITION_FLOOR_M,
                          VELOCITY_FLOOR_MPS, YAW_FLOOR_RAD, COMMON_SCORE_START_S,
                          Invariant15, coordinate_map, recovery_metrics, write_json)

# A fixed predeclared static-calibration prior; this is not range-only information.
BIAS_PRIOR_STD_MPS2 = 0.10
# Retained gyro-bias uncertainty is the same as the parent common protocol.
GYRO_BIAS_STD_RADPS = 0.01
# Accelerometer model uncertainty is retained after fitting a finite short window.
BIAS_COVARIANCE_FLOOR_MPS2 = 0.03
# Physical p/v priors preserve uncertainty in all data-unobservable directions.
POSITION_PRIOR_STD_M = 8.0
VELOCITY_PRIOR_STD_MPS = 2.0
# Independent process-level parallelism is bounded to leave room for other research.
DEFAULT_WORKERS = 12
# Each method is specified before evaluation; no outcome-dependent parameter selection.
METHODS = ("bias_2", "bias_4", "bias_8", "bias_multi_2", "bias_gate")


def bias_preintegration(times, rotations):
    """Integrate Q and its integral with the same held-rotation convention as IMU."""
    first = np.zeros((times.size, 3, 3))
    second = np.zeros_like(first)
    for index in range(1, times.size):
        dt = times[index] - times[index - 1]
        second[index] = second[index - 1] + first[index - 1]*dt + 0.5*rotations[index - 1]*dt**2
        first[index] = first[index - 1] + rotations[index - 1]*dt
    return first, second


def bias_window(data, start, end, model, first, second):
    """Build endpoint-anchored inertial and bias integrals at arrived transactions."""
    values = window_data(data, start, end, model)
    tau = values[0]
    rows, _ = np.nonzero(data["changed"][start:end+1] & (data["ranges"][start:end+1] > 0))
    rows += start
    # Reanchoring cancels any integration before this window exactly.
    sensitivity = second[rows] - second[end] - tau[:, None, None]*first[end]
    return (*values, sensitivity)


def residual_jacobian(x, tau, displacement, anchors, ranges, sigmas, sensitivity):
    """Differentiate original ranges in [p_end,v_end,yaw,delta_ba] coordinates."""
    rotation = yaw_rotation(x[6])
    corrected = displacement - np.einsum("nij,j->ni", sensitivity, x[7:10])
    rotated = corrected @ rotation.T
    position = x[:3] + tau[:, None]*x[3:6] + rotated
    difference = position - anchors
    distance = np.maximum(np.linalg.norm(difference, axis=1), 1e-12)
    directions = difference/distance[:, None]
    yaw_derivative = np.column_stack((-rotated[:, 1], rotated[:, 0], np.zeros(tau.size)))
    bias_derivative = -np.einsum("ni,ij,njk->nk", directions, rotation, sensitivity)
    jacobian = np.column_stack((directions, tau[:, None]*directions,
        np.sum(directions*yaw_derivative, axis=1), bias_derivative))/sigmas[:, None]
    return (distance-ranges)/sigmas, jacobian


def mixed_loss(squared_residuals, range_count):
    """Apply Huber to ranges and a Gaussian quadratic to bias-prior residuals."""
    z = squared_residuals
    loss = np.vstack((z, np.ones_like(z), np.zeros_like(z)))
    robust = np.arange(z.size) < range_count
    robust &= z > HUBER_SCALE**2
    root = np.sqrt(z[robust])
    loss[0, robust] = 2*HUBER_SCALE*root - HUBER_SCALE**2
    loss[1, robust] = HUBER_SCALE/root
    loss[2, robust] = -0.5*HUBER_SCALE/root**3
    return loss


@dataclass
class BiasEstimate:
    """Endpoint estimate plus data-only and prior-conditioned yaw information."""
    parameters: np.ndarray
    covariance: np.ndarray
    yaw_information_data: float
    yaw_information_with_prior: float
    normalized_rms: float
    objective: float
    elapsed_s: float
    evaluations: int
    solver_success: bool


def solve_bias_window(tau, displacement, anchors, ranges, sigmas, sensitivity, multistart=False):
    """Refine a lifted seed with rotating-IMU bias correction and explicit prior."""
    started = time.perf_counter()
    initial = solve_window(tau, displacement, anchors, ranges, sigmas, method="profile")
    seed = np.r_[initial.parameters, np.zeros(3)]
    # Gaussian priors are not included when reporting data-only identifiability.
    prior_jacobian = np.zeros((3, 10))
    prior_jacobian[:, 7:10] = np.eye(3)/BIAS_PRIOR_STD_MPS2
    def fun(x):
        """Concatenate raw-range residuals and the independent static bias prior."""
        return np.r_[residual_jacobian(x,tau,displacement,anchors,ranges,sigmas,sensitivity)[0],
                     x[7:10]/BIAS_PRIOR_STD_MPS2]
    def jac(x):
        """Provide the exact range and prior Jacobians to the trust-region solver."""
        return np.vstack((residual_jacobian(x,tau,displacement,anchors,ranges,sigmas,sensitivity)[1],
                          prior_jacobian))
    seeds = [seed]
    if multistart:
        seeds = [np.r_[seed[:6], angle, np.zeros(3)]
                 for angle in np.linspace(-np.pi,np.pi,12,endpoint=False)]
    fits = [least_squares(fun, candidate, jac=jac,
        loss=lambda z:mixed_loss(z,tau.size), max_nfev=MAX_EVALUATIONS,
        x_scale=np.r_[np.ones(7),np.full(3,BIAS_PRIOR_STD_MPS2)]) for candidate in seeds]
    fit = min(fits,key=lambda item:item.cost)
    residual, jacobian = residual_jacobian(fit.x,tau,displacement,anchors,ranges,sigmas,sensitivity)
    weights = np.minimum(1.0,HUBER_SCALE/np.maximum(np.abs(residual),1e-12))
    weighted = jacobian*np.sqrt(weights[:,None])
    nuisance = np.r_[0:6,7:10]
    projected,_ = project_out(weighted[:,nuisance],weighted[:,6:7])
    augmented = np.vstack((weighted,prior_jacobian))
    conditional,_ = project_out(augmented[:,nuisance],augmented[:,6:7])
    # This is a local curvature proxy with physical floors, not a calibrated posterior.
    regularizer = np.diag(np.r_[np.full(3,1/POSITION_PRIOR_STD_M**2),
        np.full(3,1/VELOCITY_PRIOR_STD_MPS**2),3/np.pi**2,np.zeros(3)])
    covariance = np.linalg.inv(augmented.T@augmented+regularizer)
    parameters = fit.x.copy()
    parameters[6] = np.arctan2(np.sin(parameters[6]),np.cos(parameters[6]))
    return BiasEstimate(parameters,covariance,float(np.sum(projected**2)),
        float(np.sum(conditional**2)),float(np.sqrt(np.mean(residual**2))),2*float(fit.cost),
        time.perf_counter()-started,sum(item.nfev for item in fits),bool(fit.success))


def create_tracker(estimate, rotation, bias, model):
    """Preserve joint p/v/yaw/ba curvature while moving into the invariant chart."""
    x = estimate.parameters
    state = np.eye(5)
    state[:3,:3] = yaw_rotation(x[6])@rotation
    state[:3,3],state[:3,4] = x[3:6],x[:3]
    body_vertical = state[:3,:3].T@np.array([0.,0.,1.])
    mapping = np.zeros((15,10))
    mapping[6:9,:3],mapping[3:6,3:6] = np.eye(3),np.eye(3)
    mapping[:3,6],mapping[12:15,7:10] = body_vertical,np.eye(3)
    physical = mapping@estimate.covariance@mapping.T
    physical[:3,:3] += 0.05**2*(np.eye(3)-np.outer(body_vertical,body_vertical))
    physical[:3,:3] += YAW_FLOOR_RAD**2*np.outer(body_vertical,body_vertical)
    physical[3:6,3:6] += VELOCITY_FLOOR_MPS**2*np.eye(3)
    physical[6:9,6:9] += POSITION_FLOOR_M**2*np.eye(3)
    physical[9:12,9:12] = GYRO_BIAS_STD_RADPS**2*np.eye(3)
    physical[12:15,12:15] += BIAS_COVARIANCE_FLOOR_MPS2**2*np.eye(3)
    chart = coordinate_map(state)
    corrected_bias = bias.copy()
    corrected_bias[3:] += x[7:10]
    return Invariant15(state,chart@physical@chart.T,corrected_bias,
        range_bias=model["range_bias"],range_sigma=model["range_sigma"],
        gate_threshold=12.25,gate_warmup_updates=0,huber_threshold=2.0)


def replay(job):
    """Run one causal cold-start/outage case and retain every emitted sample."""
    cache,output,name,method,outage = job
    data = dict(np.load(Path(cache)/(Path(name).stem+".npz")))
    manifest = json.loads((Path(cache)/"manifest.json").read_text(encoding="utf-8"))
    model = manifest["model"]
    times = data["time"]
    first,second = bias_preintegration(times,data["rotations"])
    history = np.full((times.size,6),np.nan)
    bias_history = np.full((times.size,3),np.nan)
    estimates,events_log = [],[]
    tracker,last_event,capture_start,candidate_index = None,None,0,0
    fixed_window = float(method.rsplit("_",1)[-1]) if method!="bias_gate" else None
    started = time.perf_counter()
    for index,stamp in enumerate(times):
        if tracker is not None and index:
            tracker.predict(data["imu"][index-1],float(stamp-times[index-1]))
        usable = data["changed"][index] & (data["ranges"][index]>0)
        if outage and 15<=stamp<25:
            usable[:] = False
        events = np.flatnonzero(usable)
        if events.size:
            if last_event is not None and stamp-last_event>GAP_TRIGGER_S:
                capture_start,candidate_index = index,0
                events_log.append(dict(event="gap_resume",time_s=float(stamp),gap_s=float(stamp-last_event)))
            last_event = stamp
        handed_off = False
        duration = fixed_window if fixed_window is not None else WINDOWS_S[candidate_index]
        if capture_start is not None and stamp-times[capture_start]>=duration:
            fit = solve_bias_window(*bias_window(data,capture_start,index,model,first,second),
                                    multistart=method=="bias_multi_2")
            accepted = method!="bias_gate" or (fit.yaw_information_data>=1/YAW_STD_TARGET_RAD**2
                                               and fit.normalized_rms<=MAX_NORMALIZED_RMS)
            estimates.append(dict(time_s=float(stamp),window_s=float(stamp-times[capture_start]),
                accepted=accepted,parameters=fit.parameters,covariance=fit.covariance,
                yaw_information_data=fit.yaw_information_data,
                yaw_information_with_prior=fit.yaw_information_with_prior,
                normalized_rms=fit.normalized_rms,objective=fit.objective,
                elapsed_s=fit.elapsed_s,evaluations=fit.evaluations,solver_success=fit.solver_success))
            if accepted:
                tracker = create_tracker(fit,data["rotations"][index],data["bias"],model)
                capture_start = None
                handed_off = True
            elif candidate_index<len(WINDOWS_S)-1:
                candidate_index += 1
            else:
                capture_start = None
                events_log.append(dict(event="no_qualified_handoff",time_s=float(stamp)))
        # No initialization-window observation is applied twice to a new tracker.
        if tracker is not None and capture_start is None and not handed_off:
            for anchor in events:
                tracker.correct_twr(data["anchors"][anchor],data["ranges"][index,anchor],int(anchor))
        if tracker is not None:
            history[index] = np.r_[tracker.X[:3,4],tracker.X[:3,3]]
            bias_history[index] = tracker.bias[3:]
    errors = np.linalg.norm(history[:,:3]-data["truth_position"],axis=1)
    available = np.all(np.isfinite(history),axis=1)
    common,tail = times>=COMMON_SCORE_START_S,times>=times[-1]-5
    def rmse(mask):
        """Score finite outputs separately from their explicitly reported availability."""
        return float(np.sqrt(np.mean(errors[mask]**2))) if mask.any() else float("nan")
    row = dict(sequence=name,method=method,outage=outage,elapsed_s=time.perf_counter()-started,
        duration_s=float(times[-1]),availability=float(available.mean()),
        common_availability=float(available[common].mean()),
        startup_unavailable_s=float(times[np.flatnonzero(available)[0]]) if available.any() else float(times[-1]),
        common_rmse_m=rmse(common&available),output_rmse_m=rmse(available),tail5_rmse_m=rmse(tail&available),
        p95_m=float(np.percentile(errors[available],95)) if available.any() else float("nan"),
        max_m=float(np.max(errors[available])) if available.any() else float("nan"),
        attempts=len(estimates),handoffs=sum(item["accepted"] for item in estimates),
        initializer_s=sum(item["elapsed_s"] for item in estimates))
    row.update(recovery_metrics(times,errors,25.0 if outage else 0.0))
    identifier=f"{Path(name).stem}__{method}__outage{int(outage)}"
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output/(identifier+".npz"),time=times,output=history,
                        truth=data["truth_position"],accelerometer_bias=bias_history)
    write_json(output/(identifier+".json"),dict(metrics=row,estimates=estimates,events=events_log))
    return row


def main():
    """Freeze the bias study configuration and dispatch independent paired replays."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache",type=Path,default=PROJECT/"results/recovery_bootstrap_20261107/cache")
    parser.add_argument("--output",type=Path,default=PROJECT/"results/recovery_bias_20261107")
    parser.add_argument("--workers",type=int,default=DEFAULT_WORKERS)
    parser.add_argument("--methods",nargs="+",default=list(METHODS),choices=METHODS)
    args=parser.parse_args()
    base=json.loads((args.cache/"manifest.json").read_text(encoding="utf-8"))
    args.output.mkdir(parents=True,exist_ok=True)
    protocol=dict(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        cache_manifest_sha256=hashlib.sha256((args.cache/"manifest.json").read_bytes()).hexdigest(),
        methods=args.methods,bias_prior_std_mps2=BIAS_PRIOR_STD_MPS2,
        bias_covariance_floor_mps2=BIAS_COVARIANCE_FLOOR_MPS2,
        yaw_information_gate="Raw-range robust curvature marginalized over p,v,accelerometer bias; no prior",
        yaw_prior_for_covariance="3/pi^2 precision only; not an objective yaw prior",
        translation_prior_for_covariance=dict(position_std_m=POSITION_PRIOR_STD_M,
                                              velocity_std_mps=VELOCITY_PRIOR_STD_MPS),
        initial_yaw_and_bias_truth=False,independent_blind_data=False,
        max_workers=args.workers,rotation_model="Held-sample Q from gyro; gyro bias is fixed within each window",
        covariance_status="Uncalibrated local curvature plus explicit floors")
    write_json(args.output/"protocol.json",protocol)
    jobs=[(str(args.cache),str(args.output/"histories"),name,method,outage)
          for name in base["evaluation"] for method in args.methods for outage in (False,True)]
    rows=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(replay,job):job for job in jobs}
        for future in as_completed(futures):
            rows.append(future.result())
            pd.DataFrame(rows).to_csv(args.output/"metrics.csv",index=False)
            print(f"DONE {len(rows)}/{len(jobs)} {futures[future][2:]}",flush=True)


if __name__=="__main__":
    main()
