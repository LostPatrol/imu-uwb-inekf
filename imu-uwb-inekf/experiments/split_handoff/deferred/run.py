"""Deferred heading handoff during complete causal UTIL TDoA trajectories.

Pair-wise packet splitting separates the live InEKF from heading evidence.
The fixed screen changes yaw only and never shrinks physical covariance.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

# Independent recordings use processes; each numerical library keeps one thread.
for _name in ("OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","OMP_NUM_THREADS"):
    os.environ[_name]="1"

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

# Reuse the new UTIL implementation and the unchanged shared evidence screen.
HERE=Path(__file__).resolve().parent
UTIL=HERE.parent/"util"
sys.path.insert(0,str(UTIL))
spec=importlib.util.spec_from_file_location("deferred_util_source",UTIL/"run.py")
util=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=util
spec.loader.exec_module(util)
import yaw_evidence

# Trial 6 is excluded from the other UTIL group's trial 3/4/5 experiment.
TRIAL=6
GEOMETRIES=(1,2,3,4)
RADIO_PROTOCOLS=(2,3)
YAWS=(0.,180.)
DURATION=45.
POLICIES=("retain","first_pass","forced_first","forced_each")
# The schedule is physical time from sensor origin; no observed motion chooses it.
SCENARIOS={"cold":dict(window_start=0.,attempts=(2.,6.,10.,14.),outage=None),
           "outage":dict(window_start=20.,attempts=(22.,26.,30.,34.),outage=(10.,20.))}


def save_json(path,value):
    """Retain diagnostic arrays and non-finite screening margins without truth access."""
    path.write_text(json.dumps(util.safe_json(value),indent=2),encoding="utf-8")


def partition_radio(data,initial_time):
    """Reserve alternating transactions independently within every directed link."""
    radio=data["radio"]
    live,candidate=[],[]
    counts={}
    for index,row in enumerate(radio):
        pair=(int(row[1]),int(row[2]))
        number=counts.get(pair,0)
        counts[pair]=number+1
        if row[0]<=initial_time:
            continue
        (live if number%2==0 else candidate).append(index)
    live,candidate=np.asarray(live,dtype=int),np.asarray(candidate,dtype=int)
    assert not np.intersect1d(live,candidate).size
    navigation=dict(data,radio=radio[live])
    evidence=dict(data,radio=radio[candidate])
    audit=dict(initial_packets=int(np.sum(radio[:,0]<=initial_time)),live_packets=len(live),
        candidate_packets=len(candidate),intersection_packets=0,
        first_live_time=float(radio[live[0],0]),first_candidate_time=float(radio[candidate[0],0]),
        live_indices_sha256=hashlib.sha256(live.tobytes()).hexdigest(),
        candidate_indices_sha256=hashlib.sha256(candidate.tobytes()).hexdigest())
    return navigation,evidence,audit


def aligned_time(data,relative):
    """Snap a predetermined clock to the next existing IMU sample only."""
    times=data["times"]
    return float(times[np.searchsorted(times,times[0]+relative)])


def replay(data,state,covariance,bias,start,end,outage):
    """Run strictly later live packets with the common compiled invariant filter."""
    return util.bounded_replay(data,state,covariance,bias,start,end,outage)


def decide(evidence,path,start,end,bias,state):
    """Use only arrived candidate packets and the actual live reference rotation."""
    available_path=path[path[:,0]<=end]
    inputs=util.window_inputs(evidence,start,end,bias,available_path)
    cref=state[:3,:3]@inputs["Qend"].T
    result=util.invoke_decision(yaw_evidence,inputs,cref)
    return result,dict(candidate_packets=len(inputs["y"]),source_packets=inputs["source_packets"],
        candidate_latest_time=float(end+inputs["tau"].max()) if len(inputs["tau"]) else None,
        path_latest_time=float(available_path[-1,0]),cref=cref,Qend=inputs["Qend"])


def change_yaw(state,covariance,delta):
    """Move the mean about world gravity and preserve physical covariance exactly."""
    old=state.copy()
    inverse=util.kernel.inverse_mapping(old)
    physical=inverse@covariance@inverse.T
    state=state.copy()
    state[:3,:3]=Rotation.from_euler("z",delta).as_matrix()@state[:3,:3]
    mapping=util.kernel.mapping(state)
    covariance=mapping@physical@mapping.T
    inverse_new=util.kernel.inverse_mapping(state)
    recovered=inverse_new@covariance@inverse_new.T
    audit=dict(position_change_m=float(np.linalg.norm(state[:3,4]-old[:3,4])),
        velocity_change_mps=float(np.linalg.norm(state[:3,3]-old[:3,3])),
        body_vertical_change=float(np.linalg.norm(state[:3,:3].T[:,2]-old[:3,:3].T[:,2])),
        physical_covariance_relative_change=float(np.linalg.norm(recovered-physical)/max(np.linalg.norm(physical),1e-12)))
    return state,.5*(covariance+covariance.T),audit


def run_case(data,navigation,evidence,path,initial_time,initial_state,initial_bias,scenario,yaw,output):
    """Keep a common normal prefix, then replay all policies over every fixed clock."""
    origin=float(data["times"][0])
    settings=SCENARIOS[scenario]
    attempts=[aligned_time(data,value) for value in settings["attempts"]]
    window_start=aligned_time(data,settings["window_start"])
    state=initial_state.copy()
    state[:3,:3]=Rotation.from_euler("z",yaw,degrees=True).as_matrix()@state[:3,:3]
    mapping=util.kernel.mapping(state)
    covariance=mapping@util.PHYSICAL@mapping.T
    bias=initial_bias.copy()
    prefix=[]
    failed=False
    if window_start>initial_time:
        state,covariance,bias,rows,_,_,failed=replay(navigation,state,covariance,bias,
            initial_time,window_start,settings["outage"])
        prefix.append(rows)
        start=window_start
    else:
        start=initial_time
    window_bias=bias.copy()
    state,covariance,bias,rows,_,_,first_failed=replay(navigation,state,covariance,bias,
        start,attempts[0],settings["outage"])
    prefix.append(rows)
    failed=failed or first_failed
    common_rows=np.vstack(prefix)
    first_result,first_audit=(None,None) if failed else decide(
        evidence,path,window_start,attempts[0],window_bias,state)
    records=[]
    for policy in POLICIES:
        current,P,ba=state.copy(),covariance.copy(),bias.copy()
        history=[common_rows]
        events=[]
        first_accept=None
        rejected_before=0
        passed=False
        policy_failed=failed
        accepted_radio=rejected_radio=0
        previous=attempts[0]
        for number,stamp in enumerate(attempts):
            if number:
                current,P,ba,rows,yes,no,bad=replay(navigation,current,P,ba,previous,stamp,settings["outage"])
                history.append(rows)
                accepted_radio+=yes
                rejected_radio+=no
                policy_failed=policy_failed or bad
            previous=stamp
            if policy_failed:
                break
            active=(policy=="forced_each" or (policy=="forced_first" and number==0)
                    or (policy=="first_pass" and not passed))
            if not active:
                continue
            result,audit=(first_result,first_audit) if number==0 else decide(
                evidence,path,window_start,stamp,window_bias,current)
            result=dict(result)
            old_rotation=current[:3,:3].copy()
            apply=bool(result["accepted"]) if policy=="first_pass" else True
            delta=float(result["delta"] if policy=="first_pass" else result["candidate"])
            change=None
            if apply:
                current,P,change=change_yaw(current,P,delta)
            if policy=="first_pass":
                if result["accepted"]:
                    first_accept=float(stamp-origin)
                    passed=True
                else:
                    rejected_before+=1
            events.append(dict(time_s=float(stamp-origin),attempt=number+1,decision=result,
                applied=apply,applied_delta_deg=float(np.degrees(delta)) if apply else 0.,
                old_rotation=old_rotation,new_rotation=current[:3,:3].copy(),
                actual_reference_rotation=audit["cref"],audit=audit,change=change))
        if not policy_failed:
            current,P,ba,rows,yes,no,bad=replay(navigation,current,P,ba,previous,
                float(data["times"][-1]),settings["outage"])
            history.append(rows)
            accepted_radio+=yes
            rejected_radio+=no
            policy_failed=policy_failed or bad
        all_rows=np.vstack(history)
        key=f"{data['name']}__{scenario}__yaw{yaw:g}__{policy}"
        np.savez_compressed(output/(key+".npz"),rows=all_rows,final=current,final_covariance=P,
            final_bias=ba,initial_state=initial_state,initial_time=initial_time)
        records.append(dict(key=key,name=data["name"],scenario=scenario,yaw_offset_deg=yaw,
            policy=policy,origin=origin,initial_position_time_s=initial_time-origin,
            common_future_start_s=settings["attempts"][-1],first_accept_s=first_accept,
            rejected_before_first_accept=rejected_before,late_accept=bool(first_accept is not None and rejected_before>0),
            heading_attempts=len(events),failed=bool(policy_failed),accepted_radio_after_first_attempt=accepted_radio,
            rejected_radio_after_first_attempt=rejected_radio,last_output_s=float(all_rows[-1,0]-origin) if len(all_rows) else None,
            artifact=str(output/(key+".npz")),events=events))
    return records


def run_record(job):
    """Generate one independent trial file without retaining or using its truth."""
    name,output_root=job
    output=Path(output_root)/name
    output.mkdir(parents=True,exist_ok=True)
    data=util.load(name,duration=DURATION)
    data.pop("truth")
    initial_time,state,bias=util.initial_state(data)
    navigation,evidence,audit=partition_radio(data,initial_time)
    records=[]
    for scenario in SCENARIOS:
        scenario_evidence=evidence
        outage=SCENARIOS[scenario]["outage"]
        if outage is not None:
            origin=float(data["times"][0])
            radio=evidence["radio"]
            keep=(radio[:,0]<origin+outage[0]) | (radio[:,0]>=origin+outage[1])
            scenario_evidence=dict(evidence,radio=radio[keep])
        path=util.radio_path(scenario_evidence)
        for yaw in YAWS:
            records.extend(run_case(data,navigation,scenario_evidence,path,initial_time,state,bias,scenario,yaw,output))
            save_json(output/"records.json",records)
    save_json(output/"radio_split.json",audit)
    return records


def score(output,records):
    """Read truth only after generation and compare fixed absolute future intervals."""
    metrics,event_scores=[],[]
    truths={}
    for record in records:
        name=record["name"]
        if name not in truths:
            truths[name]=util.load(name,duration=DURATION)["truth"]
        truth=truths[name]
        artifact=np.load(record["artifact"])
        rows=artifact["rows"]
        future=rows[rows[:,0]>record["origin"]+record["common_future_start_s"]]
        row={key:value for key,value in record.items() if key not in ("events","artifact")}
        row.update({"whole_"+key:value for key,value in util.score_rows(truth,rows).items()})
        row.update({"future_"+key:value for key,value in util.score_rows(truth,future).items()})
        expected=int(np.sum((np.arange(record["origin"],record["origin"]+DURATION,.01)[::5])>
            record["origin"]+record["common_future_start_s"]))
        row["future_output_fraction"]=min(1.,len(future)/max(expected,1))
        metrics.append(row)
        for event in record["events"]:
            stamp=record["origin"]+event["time_s"]
            errors={}
            for label in ("old_rotation","new_rotation"):
                value=np.r_[stamp,np.zeros(6),np.asarray(event[label]).ravel(),np.zeros(3)][None]
                errors[label+"_yaw_error_deg"]=util.score_rows(truth,value)["yaw_rmse_deg"]
            event_scores.append(dict(key=record["key"],name=name,scenario=record["scenario"],
                yaw_offset_deg=record["yaw_offset_deg"],policy=record["policy"],time_s=event["time_s"],
                reason=event["decision"]["reason"],accepted=event["decision"]["accepted"],
                applied=event["applied"],applied_delta_deg=event["applied_delta_deg"],**errors))
    frame=pd.DataFrame(metrics)
    frame.to_csv(output/"metrics.csv",index=False)
    pd.DataFrame(event_scores).to_csv(output/"event_scoring.csv",index=False)
    frame.groupby(["scenario","yaw_offset_deg","policy"]).agg(
        cases=("key","size"),failures=("failed","sum"),late_accepts=("late_accept","sum"),
        xy_median=("future_xy_rmse","median"),xy_max=("future_xy_rmse","max"),
        yaw_median=("future_yaw_rmse_deg","median"),coverage_min=("future_output_fraction","min"),
        whole_xy_median=("whole_xy_rmse","median")).to_csv(output/"summary.csv")


def main():
    """Freeze the complete trial-6 protocol, generate sensor-only traces, then score."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers",type=int,default=2)
    parser.add_argument("--score-only",action="store_true")
    args=parser.parse_args()
    output=HERE/"results"
    output.mkdir(parents=True,exist_ok=True)
    if args.score_only:
        score(output,json.loads((output/"records.json").read_text(encoding="utf-8")))
        return
    names=[util.record_name(g,TRIAL,p) for g in GEOMETRIES for p in RADIO_PROTOCOLS]
    protocol=dict(trial=TRIAL,names=names,geometries=GEOMETRIES,radio_protocols=RADIO_PROTOCOLS,
        yaws_deg=YAWS,scenarios=SCENARIOS,policies=POLICIES,duration_s=DURATION,
        position_handoff="Shared first .6s raw radio CV and .2s gravity tilt; no later p/v reset",
        radio_partition="Per-directed-link transaction parity after .6s; live even, heading candidate odd; fixed across all policies",
        candidate_prefix_policy="All initial <=.6s radio excluded from heading candidates and future live updates",
        outage_candidate_policy="Physically remove all outage candidate packets before radio-only path construction",
        deferred_policy="First accepted fixed yaw_evidence screen changes world-z yaw once; keep normal InEKF between attempts",
        force_policy="Forced-first applies candidate at first scheduled clock; forced-each applies at every clock, including rejected candidates",
        covariance="Only invariant-chart transport; physical covariance, p/v and IMU bias preserved at yaw change",
        reference="Actual current R @ Qend.T at each attempt; fixed known window-start IMU bias",
        primary_score="Every case future XY after the last absolute clock: 14s cold, 34s outage; never choose baseline by trigger",
        truth_usage="Raw loader returns truth then immediately discards it; generation uses no truth value or truth crop. Scoring starts after all names finish.",
        date_claim="Held-out trial relative to trial3/4/5 group; no independent recording date established",
        expected_base_cases=32,expected_trajectories=128,workers=args.workers,
        source_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in
            (Path(__file__),UTIL/"run.py",UTIL/"kernel.py",HERE.parent/"yaw_evidence.py")})
    save_json(output/"protocol.json",protocol)
    records=[]
    started=time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(run_record,(name,str(output))):name for name in names}
        for future in as_completed(futures):
            records.extend(future.result())
            save_json(output/"records.json",records)
            print(f"DONE {futures[future]}: {len(records)}/128 trajectories; {time.perf_counter()-started:.1f}s",flush=True)
    print("Sensor-only generation complete; loading truth for fixed-clock scoring.",flush=True)
    score(output,records)


if __name__=="__main__":
    main()
