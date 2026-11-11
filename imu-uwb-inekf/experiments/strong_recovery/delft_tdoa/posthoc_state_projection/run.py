"""Explore one state translation followed by genuine original Delft recursion.

The previously scored output overlays and their frozen sources are untouched.
"""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import warnings

# Keep the original small-matrix CPU workload bounded.
for variable in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[variable]="1"
sys.dont_write_bytecode=True

import numpy as np
import pandas as pd

# Reuse the same Delft sensor adapter and untouched author object factory.
HERE=Path(__file__).resolve().parent
DELFT=HERE.parent
STRONG=DELFT.parent
sys.path.insert(0,str(DELFT))
SPEC=importlib.util.spec_from_file_location("delft_native_feedback",DELFT/"run.py")
native=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(native)
import adapter
sys.path.insert(0,str(STRONG))
import feedback


def physical_mapping(state):
    """Map invariant chart errors to physical world attitude/velocity/position."""
    mapping=np.eye(15)
    mapping[3:6,:3]=-native.AUTHOR.SO3_wedge(state[:3,3])
    mapping[6:9,:3]=-native.AUTHOR.SO3_wedge(state[:3,4])
    return mapping


def replay(data,length):
    """Translate once after a newly arrived native packet, then retain recurrence."""
    start,state,bias=adapter.initial_state(data)
    estimator=native.make_filter(state,native.P0,bias)
    times,imu=(data[key] for key in ("times","imu"))
    radio=data["radio"]
    # This is equivalent to K's skip operation, but hidden packets are physically absent.
    radio=radio[~((radio[:,0]>=25-length)&(radio[:,0]<25))]
    j=np.searchsorted(radio[:,0],start,side="right")
    previous=start
    output=[]
    accepted=rejected=0
    trigger=None
    failure=None
    started=time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore",DeprecationWarning)
        try:
            for k in range(np.searchsorted(times,start,side="right"),len(times)):
                stamp=times[k]
                estimator.predict(imu[k-1],float(stamp-previous))
                while j<len(radio) and radio[j,0]<=stamp+1e-10:
                    packet=radio[j]
                    _,_,_,discard=estimator.correct(packet[1:],data["anchors"])
                    accepted+=int(not discard)
                    rejected+=int(discard)
                    if trigger is None and 25<=packet[0]<27:
                        delta=feedback.single_offset(float(packet[0]),estimator.X[:3,4],0.,
                            data["anchors"][int(packet[2])],data["anchors"][int(packet[1])],
                            float(packet[3]),np.sqrt(.13))
                        if np.any(delta):
                            old_X,old_P,old_bias=estimator.X.copy(),estimator.P.copy(),estimator.bias.copy()
                            mapping=physical_mapping(old_X)
                            old_physical=mapping@old_P@mapping.T
                            feedback.translate_original(estimator,delta)
                            mapping=physical_mapping(estimator.X)
                            new_physical=mapping@estimator.P@mapping.T
                            covariance_difference=float(np.max(np.abs(new_physical-old_physical)))
                            invariant_error=max(float(np.max(np.abs(estimator.X[:3,:4]-old_X[:3,:4]))),
                                                abs(float(estimator.X[2,4]-old_X[2,4])),
                                                float(np.max(np.abs(estimator.bias-old_bias))))
                            trigger=dict(time=float(packet[0]),pair=packet[1:3].astype(int).tolist(),
                                observation_m=float(packet[3]),native_packet_rejected=bool(discard),
                                delta_m=delta.tolist(),shift_m=float(np.linalg.norm(delta)),
                                before_position_m=old_X[:3,4].tolist(),after_position_m=estimator.X[:3,4].tolist(),
                                instant_attitude_height_velocity_bias_max_change=invariant_error,
                                physical_covariance_max_difference=covariance_difference,
                                physical_covariance_relative_difference=covariance_difference/max(float(np.max(np.abs(old_physical))),1e-15),
                                snapshot_state_max_difference=float(np.max(np.abs(estimator.X-estimator.X_hat))),
                                snapshot_covariance_max_difference=float(np.max(np.abs(estimator.P-estimator.P_hat))))
                    j+=1
                output.append(np.r_[stamp,estimator.X[:3,4],estimator.X[:3,3],estimator.X[:3,:3].ravel()])
                previous=stamp
                if not np.isfinite(estimator.X).all() or not np.isfinite(estimator.P).all():
                    failure="nonfinite_state_or_covariance"
                    break
                if np.max(np.abs(estimator.X[:3,4]))>1000:
                    failure="position_exceeds_1000m"
                    break
        except Exception as exception:
            failure=type(exception).__name__+": "+str(exception)
    return np.asarray(output).reshape(-1,16),dict(triggered=trigger is not None,trigger=trigger,
        accepted_radio=accepted,rejected_radio=rejected,failed=failure is not None,failure=failure,
        last_time=float(previous),runtime_s=time.perf_counter()-started,return_to_K=False)


def evaluate(output,records,calibration):
    """Score complete periods and explicitly preserve partial failure scores."""
    metrics=[]
    for record in records:
        key=record["key"]
        estimate=np.load(output/(key+".npz"))["rows"]
        baseline=np.load(DELFT/"baseline"/(key+".npz"))["rows"]
        reference=adapter.truth(record["name"],record["origin"],calibration["lag_s"])
        for policy,rows in [("K",baseline),("single_state_projection",estimate)]:
            for left,right in adapter.PROTOCOL["post_score_intervals_s"]:
                score=native.score(rows,reference,25+left,25+right)
                complete=score["coverage"]>=1-1e-9
                metrics.append({**record,"policy":policy,"interval":f"{left}-{right}",
                    "failed":bool(record["failed"]) if policy=="single_state_projection" else False,
                    "xy_rmse":score["xy_rmse"] if complete else np.nan,
                    "z_rmse":score["z_rmse"] if complete else np.nan,
                    "available_xy_rmse":score["xy_rmse"],"available_z_rmse":score["z_rmse"],
                    "samples":score["samples"],"coverage":score["coverage"],"complete":complete})
    frame=pd.DataFrame(metrics)
    frame.to_csv(output/"metrics.csv",index=False)
    frame.groupby(["outage_s","interval","policy"])[["xy_rmse","z_rmse","coverage"]].agg(["median","max"]).to_csv(output/"summary.csv")
    paired=frame.pivot(index=["name","outage_s","interval"],columns="policy",values="xy_rmse")
    paired["delta_m"]=paired.single_state_projection-paired.K
    paired.to_csv(output/"paired_xy.csv")
    rows=[]
    for interval,group in paired.groupby(level="interval"):
        delta=group.delta_m.dropna()
        rows.append(dict(interval=interval,cases=len(group),complete_pairs=len(delta),
            wins=int((delta< -1e-9).sum()),ties=int((np.abs(delta)<=1e-9).sum()),losses=int((delta>1e-9).sum()),
            median_paired_delta_m=float(delta.median()),best_delta_m=float(delta.min()),worst_delta_m=float(delta.max())))
    pd.DataFrame(rows).to_csv(output/"paired_differences.csv",index=False)


def main():
    """Generate the fixed 24 trajectories without truth, then score all of them."""
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,default=HERE/"results")
    parser.add_argument("--score-only",action="store_true")
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    calibration=json.loads((DELFT/"calibration.json").read_text(encoding="utf8"))
    if args.score_only:
        records=json.loads((args.output/"records.json").read_text(encoding="utf8"))
    else:
        records=[]
        for name in adapter.PROTOCOL["test_names"]:
            data=native.queued(adapter.load(name,calibration),"first_per_stamp")
            for length in (2,5,10,20):
                key=name+f"__first_per_stamp__L{length:g}"
                artifact=args.output/(key+".npz")
                if artifact.exists():
                    cached=np.load(artifact)
                    rows=cached["rows"]
                    info=json.loads(str(cached["info"]))
                else:
                    rows,info=replay(data,length)
                    np.savez_compressed(artifact,rows=rows,info=json.dumps(info))
                baseline=np.load(DELFT/"baseline"/(key+".npz"))["rows"]
                count=min(len(rows),len(baseline))
                if not info["triggered"]:
                    equality=float(np.max(np.abs(rows[:count]-baseline[:count])))
                else:
                    prefix=rows[:count,0]<info["trigger"]["time"]
                    equality=float(np.max(np.abs(rows[:count][prefix]-baseline[:count][prefix])))
                record=dict(key=key,name=name,outage_s=length,origin=data["origin"],
                            pretrigger_or_nontrigger_max_difference=equality,**info)
                records.append(record)
                (args.output/"records.json").write_text(json.dumps(records,indent=2),encoding="utf8")
                print(json.dumps(record),flush=True)
        sources=[HERE/"run.py",HERE/"protocol.json",STRONG/"feedback.py",STRONG/"feedback_protocol.json",
                 STRONG/"radio_bounds.py",DELFT/"adapter.py",DELFT/"run.py",DELFT/"calibration.json",native.AUTHOR_PATH]
        (args.output/"provenance.json").write_text(json.dumps(dict(
            sources={str(path):adapter.digest(path) for path in sources},
            test_truth_used_for_generation=False,exploratory=True),indent=2),encoding="utf8")
    print("All 24 exploratory sensor-only recursions complete; scoring truth now.",flush=True)
    evaluate(args.output,records,calibration)


if __name__=="__main__":
    main()
