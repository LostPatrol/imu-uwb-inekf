"""Compare real author, production and research kernels with one-switch controls."""

import argparse
from concurrent.futures import as_completed
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import time
import warnings

# Scientific imports inherit one numerical thread in every independent worker.
for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name]="1"
sys.dont_write_bytecode=True
import audit
import numpy as np
import pandas as pd

# This is a mechanism-development set, never an independent test claim.
HERE=Path(__file__).resolve().parent
NAMES=audit.NAMES
YAWS=audit.YAWS
VARIANTS=("author_original", "production_proposed", "research_util", "production_full_geometry")
sys.path.insert(0,str(audit.EXPERIMENTS/"split_handoff"))
from cpu_replay import replay_pool


def save_json(path,value):
    """Write exact settings and array diagnostics without truth-dependent selection."""
    path.write_text(json.dumps(audit.native.ours.util.safe_json(value),indent=2),encoding="utf8")


def replay_class(data,X,P,bias,start,variant):
    """Use the unchanged class and the same literal native event calls for all."""
    if variant=="author_original":
        estimator=audit.native.ours.make_filter(X,P,bias)
    else:
        config=audit.current.FilterConfig.proposed()
        if variant=="production_full_geometry":
            config=replace(config,direct_lever_arm_jacobian=True,name=variant)
        estimator=audit.current.InEKF(X,P,bias,config)
        estimator.w_tdoa=np.array([np.sqrt(.13)])
    times,imu,radio=(data[key] for key in ("times","imu","radio"))
    events=np.unique(np.r_[times,radio[:,0]])
    imu_at,radio_at={},{}
    for index,stamp in enumerate(times):
        imu_at.setdefault(float(stamp),index)
    for index,stamp in enumerate(radio[:,0]):
        radio_at.setdefault(float(stamp),index)
    previous_imu_time,last_output=start,-np.inf
    rows=[]
    accepted,rejected,predicts,corrections=0,0,0,0
    failure=None
    started=time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore",DeprecationWarning)
        try:
            for k in range(max(1,np.searchsorted(events,start,side="right")),len(events)):
                previous,stamp=float(events[k-1]),float(events[k])
                if previous<=start:
                    continue
                if previous in imu_at:
                    dt=stamp-previous_imu_time
                    previous_imu_time=stamp
                    estimator.predict(imu[imu_at[previous]],dt)
                    predicts+=1
                if previous in radio_at:
                    _,_,_,discard=estimator.correct(radio[radio_at[previous],1:],data["anchors"])
                    accepted+=int(not discard)
                    rejected+=int(discard)
                    corrections+=1
                if stamp-last_output>=.01:
                    rows.append(np.r_[stamp,estimator.X[:3,4],estimator.X[:3,3],estimator.X[:3,:3].ravel(),np.zeros(3)])
                    last_output=stamp
                if not np.all(np.isfinite(estimator.X)) or not np.all(np.isfinite(estimator.P)) or np.max(np.abs(estimator.X[:3,4]))>1000:
                    failure="nonfinite_or_position_exceeds_1000m"
                    break
        except Exception as exception:
            failure=type(exception).__name__+": "+str(exception)
    return np.array(rows).reshape(-1,19),dict(failed=failure is not None,failure=failure,
        predict_calls=predicts,correction_calls=corrections,accepted_radio=accepted,rejected_radio=rejected,
        runtime_seconds=time.perf_counter()-started),estimator.X,estimator.P,estimator.bias


def run_case(job):
    """Generate a single frozen real recording/method/initial angle without scoring."""
    name,yaw,variant=job
    output=HERE/"diagnostic"
    key=f"{name}__yaw{yaw:g}__{variant}"
    if (output/(key+".json")).exists():
        return json.loads((output/(key+".json")).read_text(encoding="utf8"))
    data,start,X,P,bias=audit.initial(name,yaw)
    if variant=="research_util":
        rows,info=audit.native.replay(data,X,P,bias,start,"current_kernel",None,False)
        final_X,final_P,final_bias=np.empty((0,)),np.empty((0,)),np.empty((0,))
    else:
        rows,info,final_X,final_P,final_bias=replay_class(data,X,P,bias,start,variant)
    np.savez_compressed(output/(key+".npz"),rows=rows,initial_X=X,initial_P=P,initial_bias=bias,
                        final_X=final_X,final_P=final_P,final_bias=final_bias)
    record=dict(name=name,yaw_offset_deg=yaw,variant=variant,origin=data["common_origin"],key=key,**info)
    save_json(output/(key+".json"),record)
    return record


def score(records):
    """Read truth only after all fixed trajectories have completed generation."""
    metrics=[]
    truths={}
    for record in records:
        name=record["name"]
        if name not in truths:
            truths[name]=audit.native.ours.util.load(name,duration=45.)["truth"]
        rows=np.load(HERE/"diagnostic"/(record["key"]+".npz"))["rows"]
        metric=dict(record)
        for label,left,right in (("whole",.65,44.95),("later",5.,44.95),("startup",.65,5.)):
            values=audit.native.ours.util.score_rows(truths[name],rows,record["origin"]+left,record["origin"]+right)
            metric.update({label+"_"+key:value for key,value in values.items()})
        metrics.append(metric)
    frame=pd.DataFrame(metrics)
    frame.to_csv(HERE/"diagnostic"/"metrics.csv",index=False)
    frame.groupby("variant")[[label+"_"+metric for label in ("whole","later","startup")
        for metric in ("xy_rmse","z_rmse","yaw_rmse_deg")]].median().to_csv(HERE/"diagnostic"/"summary.csv")
    print(frame[["name","yaw_offset_deg","variant","later_xy_rmse","later_yaw_rmse_deg","failed"]].to_string(index=False))


def main():
    """Freeze the small mechanistic matrix before replay and shared-clock scoring."""
    parser=argparse.ArgumentParser()
    parser.add_argument("--workers",type=int,default=16)
    parser.add_argument("--score-only",action="store_true")
    args=parser.parse_args()
    output=HERE/"diagnostic"
    output.mkdir(parents=True,exist_ok=True)
    if args.score_only:
        score(json.loads((output/"records.json").read_text(encoding="utf8")))
        return
    protocol=dict(names=NAMES,yaws_deg=YAWS,variants=VARIANTS,workers=args.workers,duration_s=45.,
        independent_test=False,selection="fixed original mechanism-development recordings; no data selection by error",
        initial="same sensor-only .6s X/bias; original invariant P exactly; world-z yaw perturbation at initialization",
        observations="same actual TDoA2 ring/TDoA3 full protocol; rhoB-rhoA; raw acc g and interpolated raw gyro deg/s",
        queue="literal previous-event/next-event author dt; first packet per duplicate timestamp; no outage",
        changes={"author_original":"untouched original class; original bias convention and .13 main variance",
                 "production_proposed":"actual current source FilterConfig.proposed; no extra stationary pseudo update",
                 "research_util":"standalone split_handoff/util/kernel, complete world Jacobian+Huber+reset; distinct from production",
                 "production_full_geometry":"production_proposed with ONLY direct_lever_arm_jacobian=True"},
        scoring="common20Hz ZOH; wrapped Euler yaw and relative SO3; no alignment; original paper SO3-log component score is distinct",
        source_sha256={str(path):audit.native.ours.digest(path) for path in (audit.native.ours.AUTHOR,Path(audit.current.__file__),
            Path(audit.kernel.__file__),Path(__file__),Path(audit.__file__))})
    save_json(HERE/"protocol.json",protocol)
    jobs=[(name,yaw,variant) for name in NAMES for yaw in YAWS for variant in VARIANTS]
    records=[]
    started=time.perf_counter()
    with replay_pool(len(jobs),args.workers) as pool:
        for future in as_completed([pool.submit(run_case,job) for job in jobs]):
            records.append(future.result())
            save_json(output/"records.json",records)
            print(f"DONE {len(records)}/{len(jobs)} {records[-1]['key']} wall={time.perf_counter()-started:.1f}s",flush=True)
    save_json(output/"provenance.json",dict(wall_seconds=time.perf_counter()-started,
        source_modified={path:audit.native.ours.digest(Path(path))!=digest for path,digest in protocol["source_sha256"].items()}))
    score(records)


if __name__=="__main__":
    main()
