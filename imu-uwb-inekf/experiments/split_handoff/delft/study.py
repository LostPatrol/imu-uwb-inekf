"""Paired Delft split position/heading handoffs without evaluation attitude truth.

Radio arrivals are disjoint between live tracking and heading evidence. All
strategies share the same state until one common terminal attitude decision.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

# Current source implementations supply common inertial dynamics and raw-radio fits.
HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parents[1]
RADIO = EXPERIMENTS / "radio_inertial_models"
for folder in (EXPERIMENTS/"roadmap_stage1", EXPERIMENTS/"recovery_bootstrap", RADIO, HERE.parent):
    sys.path.insert(0, str(folder))
from run import (DATASET, MOTIONS, calibration, load_anchors, load_real_log,
                 estimate_initial_rotation_and_bias, write_json)
from invariant15 import Invariant15, coordinate_map
from eskf15 import symmetrize
from initializer import cv_candidate, solve_window, yaw_rotation
from run_research import preintegrate, window_data, tracker_from_estimate
from rotation_window import local_shape, radio_seed_path, solve_attitude, wahba_seed

# All experiments use the same observable radio gate after startup.
POLICY = dict(gate_threshold=12.25, gate_warmup_updates=0, huber_threshold=2.0)
# Sensor timing and initial static prefix match the current physical replay model.
DELAY_S = 0.20
PREFIX_S = 1.50
# Artificial silence, common translation handoff, and later independent heading window.
OUTAGE = (15.0, 25.0)
POSITION_TIME = 27.0
ATTITUDE_TIME = 31.0
# The shortest late recording lasts 38.42 s, so this fixed future interval is available in all six.
SCORE_INTERVAL = (31.0, 36.0)
# Controlled mean perturbations are world-z left rotations at the blackout onset.
YAW_OFFSETS_DEG = (0.0, 90.0, -90.0, 180.0)
# A physical body-acceleration bias prior is a nuisance of the full SO(3) fit, not truth.
BIAS_STD = 0.10
# Common conservative uncertainty floors after independent position and heading handoffs.
POSITION_STD = 0.50
VELOCITY_STD = 1.00
TILT_STD_RAD = np.deg2rad(10.0)
YAW_STD_RAD = np.deg2rad(45.0)
# Four requested strategies plus a cheap projection of the full-rotation competitor.
METHODS = ("position_only", "full_attitude", "forced_yaw", "evidence_yaw", "projected_full_yaw")


def prepare(cache):
    """Build sensor-only caches and separate scoring arrays directly from raw recordings."""
    cache.mkdir(parents=True, exist_ok=True)
    anchors = load_anchors(DATASET)
    names = sorted(p.name for p in (DATASET/"twr").glob("*.csv"))
    training, evaluation = [], []
    for motion in MOTIONS:
        pair = [n for n in names if n.endswith(f"_{motion}.csv")]
        training.append(pair[0]); evaluation.append(pair[1])
    train_logs = {n:load_real_log(DATASET,n,sensor_delay_s=DELAY_S) for n in training}
    model = calibration(train_logs, anchors, training)
    for name in evaluation:
        log = load_real_log(DATASET,name,sensor_delay_s=DELAY_S)
        rotation, bias = estimate_initial_rotation_and_bias(log)
        selected = log.time >= PREFIX_S
        stamps = log.time[selected]-log.time[selected][0]
        imu = log.imu[selected]
        rotations, displacement, velocity = preintegrate(stamps,imu,rotation,bias)
        np.savez_compressed(cache/(Path(name).stem+".npz"),time=stamps,imu=imu,
            ranges=log.ranges[selected],changed=log.changed[selected],anchors=anchors,
            bias=bias,rotations=rotations,displacement=displacement,integrated_velocity=velocity)
        np.savez_compressed(cache/(Path(name).stem+"_scoring.npz"),position=log.ground_truth_position[selected])
    write_json(cache/"protocol.json",dict(training=training,evaluation=evaluation,model=model,
        sensor_delay_s=DELAY_S,prefix_s=PREFIX_S,outage=OUTAGE,position_time_s=POSITION_TIME,
        attitude_time_s=ATTITUDE_TIME,score_interval_s=SCORE_INTERVAL,yaw_offsets_deg=YAW_OFFSETS_DEG,
        yaw_perturbation_time_s=OUTAGE[0],radio_policy=POLICY,
        parity="Observed per-anchor arrivals, excluding masked transactions: odd live, even evidence during [27,31]",
        evaluation_truth_in_estimator=False,evaluation_yaw_truth_available=False,
        training_truth_for_radio_calibration=True,truth_used_for_valid_interval=True,
        independent_test_date=False,dataset_sha256={n:hashlib.sha256((DATASET/"twr"/n).read_bytes()).hexdigest()
                                                    for n in training+evaluation}))


def tracker(state, physical, bias, model):
    """Create the same 15-state invariant tracker from a physical covariance."""
    chart = coordinate_map(state)
    return Invariant15(state,chart@physical@chart.T,bias,
        range_bias=model["range_bias"],range_sigma=model["range_sigma"],**POLICY)


def clone(core, model):
    """Clone a common physical state without sharing mutable covariance or bias arrays."""
    return tracker(core.X.copy(),core.physical_covariance(),core.bias.copy(),model)


def conservative_attitude_covariance(core):
    """Return one common world-attitude uncertainty with unsupported cross-information removed."""
    physical = core.physical_covariance()
    world = core.X[:3,:3]@physical[:3,:3]@core.X[:3,:3].T
    world += np.diag([TILT_STD_RAD**2,TILT_STD_RAD**2,YAW_STD_RAD**2])
    physical[:3,:] = 0.0; physical[:,:3] = 0.0
    return physical, symmetrize(world)


def range_events(data, start, end, active, parity=None):
    """Collect actual positive transactions, optionally selecting one fixed per-anchor parity."""
    valid = data["changed"][start:end+1] & (data["ranges"][start:end+1]>0) & active
    if parity is not None:
        valid &= (data["parity"][start:end+1] == parity)
    rows, ids = np.nonzero(valid)
    rows += start
    return rows, ids


def position_handoff(core, data, start, end, active, model):
    """Set a common radio-only translation, without injecting this window twice."""
    rows, ids = range_events(data,start,end,active)
    tau = data["time"][rows]-data["time"][end]
    y = data["ranges"][rows,ids]-np.asarray(model["range_bias"])[ids]
    sigma = np.asarray(model["range_sigma"])[ids]
    estimate, residual, jacobian = cv_candidate(tau,data["anchors"][ids],y,sigma)
    weights = np.minimum(1.0,2/np.maximum(np.abs(residual),1e-12))
    curvature = np.linalg.inv((jacobian.T*weights)@jacobian + np.diag(np.r_[np.full(3,1/8**2),np.full(3,1/2**2)]))
    source = core.physical_covariance()
    physical = np.zeros((15,15))
    retained = np.r_[0:3,9:15]
    physical[np.ix_(retained,retained)] = source[np.ix_(retained,retained)]
    order = np.r_[6:9,3:6]
    physical[np.ix_(order,order)] = curvature+np.diag(np.r_[np.full(3,POSITION_STD**2),np.full(3,VELOCITY_STD**2)])
    state = core.X.copy()
    state[:3,4],state[:3,3] = estimate[:3],estimate[3:6]
    return tracker(state,physical,core.bias.copy(),model),dict(ranges=len(rows),position=estimate[:3],
        velocity=estimate[3:6],normalized_rms=float(np.sqrt(np.mean(residual**2))))


def heading_evidence(data, start, end, active, model, start_bias, live_end_rotation):
    """Fit disjoint raw radio headings and evaluate the shared bias-aware yaw gate."""
    from yaw_evidence import decide
    rows,ids = range_events(data,start,end,active,parity=0)
    local_rows = rows-start
    stamps = data["time"][start:end+1]
    imu = data["imu"][start:end+1]
    acc = imu[:,:3]*9.81-start_bias[3:]
    gyro = np.deg2rad(imu[:,3:])-start_bias[:3]
    force,B,_,Qend = local_shape(stamps,acc,gyro,local_rows)
    tau = data["time"][rows]-stamps[-1]
    plus = data["anchors"][ids]
    y = data["ranges"][rows,ids]-np.asarray(model["range_bias"])[ids]
    sigma = np.asarray(model["range_sigma"])[ids]
    cv = cv_candidate(tau,plus,y,sigma)[0]
    path = radio_seed_path(tau,plus,None,y,sigma,cv)
    # Re-reference to the actual live terminal attitude, including intervening radio corrections.
    Cref = live_end_rotation@Qend.T
    begun = time.perf_counter()
    decision = decide(tau,force,B,plus,None,y,sigma,path,Cref)
    gate_seconds = time.perf_counter()-begun
    # Raw-radio full-rotation refinement receives both an independent Procrustes seed and Cref.
    begun = time.perf_counter()
    fits = [solve_attitude(tau,force,plus,None,y,sigma,path,bias_sensitivity=B,
                          seed_rotation=seed,bias_sigma=BIAS_STD) for seed in (None,Cref)]
    fit = min(fits,key=lambda value:value["cost"])
    full_rotation = fit["C"]@Qend
    relative = full_rotation@live_end_rotation.T
    projected_delta = float(np.arctan2(relative[1,0]-relative[0,1],relative[0,0]+relative[1,1]))
    diagnostics = dict(decision=decision,events=len(rows),gate_seconds=gate_seconds,
        full_fit_seconds=time.perf_counter()-begun,full_cost=fit["cost"],full_success=bool(fit["success"]),
        full_nfev=sum(f["nfev"] for f in fits),full_seed_costs=[f["cost"] for f in fits],
        full_fit_bias=fit["x"][9:],full_tilt_change_deg=float(np.degrees(np.arccos(np.clip(
            live_end_rotation[:,2]@full_rotation[:,2],-1,1)))),projected_full_delta_rad=projected_delta,
        Cref=Cref,Qend=Qend,live_terminal_rotation=live_end_rotation,full_terminal_rotation=full_rotation,
        force_singular=wahba_seed(tau,force,path)[1])
    window = dict(tau=tau,force=force,B=B,plus=plus,y=y,sigma=sigma,path=path,rows=rows,
                  anchors=ids,Cref=Cref,Qend=Qend)
    return decision,full_rotation,projected_delta,diagnostics,window


def run_case(job):
    """Share the prefix and position handoff, fork only at the common heading decision."""
    cache,output,name,layout,yaw_deg = job
    cache,output = Path(cache),Path(output)
    data = dict(np.load(cache/(Path(name).stem+".npz")))
    model = json.loads((cache/"protocol.json").read_text(encoding="utf-8"))["model"]
    stamps = data["time"]
    # Only received events advance the runtime split counter; masked packets are absent.
    positive = data["changed"] & (data["ranges"]>0)
    received = positive & ~((stamps>=OUTAGE[0])&(stamps<OUTAGE[1]))[:,None]
    data["parity"] = np.cumsum(received,axis=0)%2
    active = np.ones(len(data["anchors"]),dtype=bool)
    if layout == "lower4":
        active[:] = False
        active[np.argsort(data["anchors"][:,2])[:4]] = True
    k15 = int(np.searchsorted(stamps,15.0))
    k25 = int(np.searchsorted(stamps,25.0))
    k27 = int(np.searchsorted(stamps,27.0))
    k31 = int(np.searchsorted(stamps,31.0))
    # Preserve complete state outputs rather than only aggregate errors.
    history = np.full((len(stamps),6),np.nan)
    rotations = np.full((len(stamps),3,3),np.nan)
    biases = np.full((len(stamps),6),np.nan)
    core = None
    captures = {}
    live_count = evidence_count = 0
    position_bias = None
    begun = time.perf_counter()
    for k in range(k31+1):
        stamp = stamps[k]
        if core is not None and k:
            core.predict(data["imu"][k-1],float(stamp-stamps[k-1]))
        if k == k15:
            physical = core.physical_covariance()
            core.X[:3,:3] = yaw_rotation(np.deg2rad(yaw_deg))@core.X[:3,:3]
            chart = coordinate_map(core.X)
            core.P = symmetrize(chart@physical@chart.T)
        handed = False
        if core is None and stamp >= 2.0:
            # Startup is a shared all-anchor Profile-1, before the independent layout stress.
            estimate = solve_window(*window_data(data,0,k,model),method="profile")
            core = tracker_from_estimate(estimate,data["rotations"][k],data["bias"],model)
            handed = True
        if k == k27:
            core,captures["position"] = position_handoff(core,data,k25,k27,active,model)
            position_bias = core.bias.copy()
            handed = True
        blocked = 15.0 <= stamp < 25.0 or k25 <= k < k27
        valid = positive[k] & active
        if core is not None and not blocked and not handed:
            for anchor in np.flatnonzero(valid):
                # The heading window starts strictly after the position handoff sample.
                if k27 < k <= k31 and data["parity"][k,anchor] == 0:
                    evidence_count += 1
                    continue
                core.correct_twr(data["anchors"][anchor],data["ranges"][k,anchor],int(anchor))
                if k27 < k <= k31:
                    live_count += 1
        if core is not None:
            history[k] = np.r_[core.X[:3,4],core.X[:3,3]]
            rotations[k],biases[k] = core.X[:3,:3],core.bias
    # Start after the common position-window endpoint to keep radio groups disjoint.
    decision,full,projected,diagnostics,window = heading_evidence(
        data,k27+1,k31,active,model,position_bias,core.X[:3,:3].copy())
    common_physical,world_angle = conservative_attitude_covariance(core)
    branches = {}
    for method in METHODS:
        state = core.X.copy()
        if method == "full_attitude":
            state[:3,:3] = full
        elif method == "forced_yaw":
            state[:3,:3] = yaw_rotation(float(decision["candidate"]))@state[:3,:3]
        elif method == "evidence_yaw":
            state[:3,:3] = yaw_rotation(float(decision["delta"]))@state[:3,:3]
        elif method == "projected_full_yaw":
            state[:3,:3] = yaw_rotation(projected)@state[:3,:3]
        physical = common_physical.copy()
        physical[:3,:3] = state[:3,:3].T@world_angle@state[:3,:3]
        branches[method] = tracker(state,physical,core.bias.copy(),model)
    outputs = {m:history.copy() for m in METHODS}
    attitudes = {m:rotations.copy() for m in METHODS}
    bias_outputs = {m:biases.copy() for m in METHODS}
    handoff_covariances = {m:branch.physical_covariance().copy() for m,branch in branches.items()}
    for method,branch in branches.items():
        attitudes[method][k31] = branch.X[:3,:3]
    for k in range(k31+1,len(stamps)):
        dt = float(stamps[k]-stamps[k-1])
        events = np.flatnonzero(positive[k]&active)
        for method,branch in branches.items():
            branch.predict(data["imu"][k-1],dt)
            for anchor in events:
                branch.correct_twr(data["anchors"][anchor],data["ranges"][k,anchor],int(anchor))
            outputs[method][k] = np.r_[branch.X[:3,4],branch.X[:3,3]]
            attitudes[method][k],bias_outputs[method][k] = branch.X[:3,:3],branch.bias
    elapsed = time.perf_counter()-begun
    # Evaluation positions are accessed only after every branch has finished.
    truth = np.load(cache/(Path(name).stem+"_scoring.npz"))["position"]
    identifier = Path(name).stem+"__"+layout+"__yaw"+str(int(yaw_deg))
    output.mkdir(parents=True,exist_ok=True)
    payload = dict(time=stamps,truth=truth,common_position_handoff=history[k27],
                   common_attitude_entry=rotations[k31],common_physical_covariance=common_physical,
                   common_world_attitude_covariance=world_angle,
                   active_anchors=np.flatnonzero(active),live_events=live_count,evidence_events=evidence_count)
    for method in METHODS:
        payload[method] = outputs[method]
        payload[method+"_R"] = attitudes[method]
        payload[method+"_bias"] = bias_outputs[method]
        payload[method+"_handoff_physical_covariance"] = handoff_covariances[method]
    np.savez_compressed(output/(identifier+".npz"),**payload)
    np.savez_compressed(output/(identifier+"_evidence.npz"),**window)
    write_json(output/(identifier+".json"),dict(sequence=name,layout=layout,yaw_offset_deg=yaw_deg,
        yaw_truth_available=False,position=captures["position"],heading=diagnostics,
        actual_position_time_s=stamps[k27],actual_attitude_time_s=stamps[k31],elapsed_s=elapsed,
        live_events=live_count,evidence_events=evidence_count))
    rows = []
    for method in METHODS:
        error = outputs[method][:,:3]-truth
        xy2 = np.sum(error[:,:2]**2,axis=1)
        z2 = error[:,2]**2
        row = dict(sequence=name,layout=layout,yaw_offset_deg=yaw_deg,method=method,
            accepted=bool(decision["accepted"]),candidate_delta_deg=float(np.rad2deg(decision["candidate"])),
            applied_evidence_delta_deg=float(np.rad2deg(decision["delta"])),sd_deg=decision.get("sd_deg"),
            elapsed_case_s=elapsed,finite=bool(np.all(np.isfinite(outputs[method][stamps>=2.1]))),
            yaw_truth_available=False,live_events=live_count,evidence_events=evidence_count,
            full_tilt_change_deg=diagnostics["full_tilt_change_deg"],
            full_success=diagnostics["full_success"],full_nfev=diagnostics["full_nfev"])
        for label,mask in dict(common_prefix=(stamps>=8.1)&(stamps<15),
                gap=(stamps>=15)&(stamps<25),early_blind=(stamps>=25)&(stamps<27),
                live_wait=(stamps>=27)&(stamps<31),
                fixed=(stamps>=SCORE_INTERVAL[0])&(stamps<SCORE_INTERVAL[1]),
                terminal=stamps>=31,post_radio=stamps>=25,tail=stamps>=stamps[-1]-5).items():
            row[label+"_xy_rmse_m"] = float(np.sqrt(np.mean(xy2[mask])))
            row[label+"_z_rmse_m"] = float(np.sqrt(np.mean(z2[mask])))
            row[label+"_xy_max_m"] = float(np.sqrt(np.max(xy2[mask])))
        rows.append(row)
    return rows


def main():
    """Run predetermined independent recordings and retain every completed degeneracy."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=HERE/"results")
    parser.add_argument("--prepare",action="store_true")
    parser.add_argument("--prepare-only",action="store_true")
    parser.add_argument("--workers",type=int,default=6)
    parser.add_argument("--sequence",default="")
    parser.add_argument("--layouts",nargs="+",default=["all8","lower4"])
    parser.add_argument("--yaw",nargs="+",type=float,default=YAW_OFFSETS_DEG)
    args = parser.parse_args()
    cache = args.output/"cache"
    if args.prepare or args.prepare_only:
        prepare(cache)
    if args.prepare_only:
        return
    protocol = json.loads((cache/"protocol.json").read_text(encoding="utf-8"))
    jobs = [(str(cache),str(args.output/"histories"),name,layout,yaw)
            for name in protocol["evaluation"] if args.sequence in name
            for layout in args.layouts for yaw in args.yaw]
    rows=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(run_case,job):job for job in jobs}
        for number,future in enumerate(as_completed(futures),1):
            rows.extend(future.result())
            pd.DataFrame(rows).to_csv(args.output/"metrics.csv",index=False)
            print(f"DONE {number}/{len(jobs)} {futures[future][2:]} gate={rows[-1]['accepted']}",flush=True)


if __name__=="__main__":
    main()
