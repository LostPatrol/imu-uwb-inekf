"""Test a separately registered single XY state translation on supplementary flights.

The untouched original class remains the backend. Generation and truth scoring
are separate stages; the original eight-flight output-only study is unchanged.
"""

import argparse
from concurrent.futures import as_completed
import json
from pathlib import Path
import time

import independent as experiment
import numpy as np
import pandas as pd

# These four supplementary recordings were named before their scores were read.
NAMES=("const4-trial2-tdoa2-traj2","const4-trial2-tdoa2-traj3",
       "const4-trial2-tdoa3-traj2","const4-trial2-tdoa3-traj3")
OUTPUT=experiment.HERE/"feedback_extension"
# Dataset location is fixed by the workspace instructions, with no substitutes.
DATA=Path("E:/datasets/dataset/flight-dataset/csv-data")
SHARED=experiment.audit.EXPERIMENTS/"strong_recovery"
experiment.sys.path.insert(0,str(SHARED))
from feedback import single_offset, translate_original


def physical_map(X):
    """Map the invariant position/velocity chart to physical additive errors."""
    mapping=np.eye(15)
    mapping[3:6,:3]=-experiment.audit.current.SO3_wedge(X[:3,3])
    mapping[6:9,:3]=-experiment.audit.current.SO3_wedge(X[:3,4])
    return mapping


def freeze():
    """Record timestamp coverage, file names and all fixed parameters before scoring."""
    OUTPUT.mkdir(parents=True,exist_ok=True)
    rows=[]
    for name in NAMES:
        path=DATA/"const4"/(name+".csv")
        item=dict(name=name,path=str(path),exists=path.exists(),sufficient_length=False)
        if path.exists():
            table=pd.read_csv(path,usecols=["t_acc","t_gyro","t_tdoa","t_pose"])
            streams={column:table[column].dropna().to_numpy(float) for column in table}
            origin=max(streams[column][0] for column in ("t_acc","t_gyro","t_tdoa"))
            item.update(origin=float(origin),minimum_end_s=float(min(values[-1] for values in streams.values())-origin),
                sufficient_length=bool(all(values[-1]>=origin+experiment.END for values in streams.values())))
        rows.append(item)
    pd.DataFrame(rows).to_csv(OUTPUT/"inventory.csv",index=False)
    experiment.save_json(OUTPUT/"protocol.json",dict(names=NAMES,stage="supplementary prospective extension; not balanced primary cohort",
        gaps_s=experiment.GAPS,resume_s=experiment.RESUME,end_s=experiment.END,windows=experiment.WINDOWS,
        workers_max=6,BLAS_threads=1,initial="same sensor-only X and original recommended invariant P; nominal yaw0",
        variance_m2=experiment.VARIANCE,sigma_m=float(experiment.SIGMA),lever_radius_m=float(np.linalg.norm(experiment.fast.LEVER)),
        trigger="first currently offered native packet in [25,27), after its original correction, with nonzero single_offset",
        update="single XY translation; unchanged instantaneous R/v/z/bias and physical additive covariance; no forced return",
        queue="original first-per-stamp native queue, same clock; actual source class for both arms",
        truth="freeze reads timestamps only; scoring starts only after all16 paired trajectories are generated",
        source_hashes={str(path):experiment.audit.native.ours.digest(path) for path in
            (SHARED/"feedback.py",SHARED/"feedback_protocol.json",SHARED/"radio_bounds.py",experiment.HERE/"feedback_runner.py",experiment.audit.native.ours.AUTHOR)}))
    print(pd.DataFrame(rows).to_string(index=False),flush=True)


def replay_record(name):
    """Reuse exact cached author split states including P and continue its native queue."""
    experiment.OUTPUT=OUTPUT
    baselines=experiment.generate_record(name)
    data=np.load(OUTPUT/(name+"__data.npz"))
    native=experiment.audit.native
    raw=native.load_raw(name,False)
    raw.pop("truth")
    events,ii,ri=experiment.inputs(raw)
    keep=events<=float(data["origin"])+experiment.END
    events,ii,ri=events[keep],ii[keep],ri[keep]
    target=OUTPUT/"feedback"
    target.mkdir(parents=True,exist_ok=True)
    results=[]
    for baseline in baselines:
        key=baseline["key"]
        cached=np.load(OUTPUT/"K"/(key+".npz"))
        estimator=native.ours.make_filter(cached["split_X"],cached["split_P"],np.zeros(6))
        for attribute in ("X","P","X_hat","P_hat","bias"):
            setattr(estimator,attribute,cached["split_"+attribute].copy())
        previous_imu=float(cached["split_previous_imu_time"])
        split=float(cached["split_time"])
        rows=list(cached["rows"][cached["rows"][:,0]<=split])
        last_output=rows[-1][0] if rows else -np.inf
        trigger=None
        info=dict(failed=False,failure=None,post_split_predicts=0,post_split_corrections=0)
        for k in range(max(1,np.searchsorted(events,split,side="left")+1),len(events)):
            previous,stamp=float(events[k-1]),float(events[k])
            try:
                if ii[k-1]>=0:
                    estimator.predict(raw["imu"][ii[k-1]],stamp-previous_imu)
                    previous_imu=stamp
                    info["post_split_predicts"]+=1
                off=baseline["resume"]-baseline["gap_s"]<=previous<baseline["resume"]
                if ri[k-1]>=0 and not off:
                    packet=raw["radio"][ri[k-1]]
                    estimator.correct(packet[1:],raw["anchors"])
                    info["post_split_corrections"]+=1
                    if trigger is None and baseline["resume"]<=previous<baseline["resume"]+2.:
                        delta=single_offset(previous,estimator.X[:3,4],float(np.linalg.norm(experiment.fast.LEVER)),
                            raw["anchors"][int(packet[2])],raw["anchors"][int(packet[1])],packet[3],experiment.SIGMA)
                        if np.any(delta):
                            before_X,before_P,before_bias=estimator.X.copy(),estimator.P.copy(),estimator.bias.copy()
                            physical_before=physical_map(before_X)@before_P@physical_map(before_X).T
                            translate_original(estimator,delta)
                            physical_after=physical_map(estimator.X)@estimator.P@physical_map(estimator.X).T
                            trigger=dict(packet_time=previous,output_time=stamp,relative_s=previous-baseline["resume"],
                                delta=delta.tolist(),distance_m=float(np.linalg.norm(delta)),
                                unchanged_R_v_z_max_abs=float(max(np.max(np.abs(estimator.X[:3,:4]-before_X[:3,:4])),abs(estimator.X[2,4]-before_X[2,4]))),
                                unchanged_bias_max_abs=float(np.max(np.abs(estimator.bias-before_bias))),
                                physical_covariance_max_abs=float(np.max(np.abs(physical_after-physical_before))))
                if experiment.failure(estimator):
                    info.update(failed=True,failure="nonfinite_or_position_exceeds_1000m")
                    break
            except Exception as error:
                info.update(failed=True,failure=type(error).__name__+": "+str(error))
                break
            if stamp-last_output>=.01:
                rows.append(experiment.row(estimator,stamp))
                last_output=stamp
        rows=np.asarray(rows).reshape(-1,19)
        same_shape=rows.shape==cached["rows"].shape
        no_trigger_max=None if trigger is not None or not same_shape else float(np.max(np.abs(rows-cached["rows"])))
        result=dict(baseline,key=key,trigger=trigger,trigger_count=int(trigger is not None),
            nontrigger_rows_max_abs_vs_K=no_trigger_max,row_shape_equal=same_shape,**info)
        np.savez_compressed(target/(key+".npz"),rows=rows,final_X=estimator.X,final_P=estimator.P,
            final_X_hat=estimator.X_hat,final_P_hat=estimator.P_hat,final_bias=estimator.bias)
        experiment.save_json(target/(key+".json"),result)
        results.append(result)
    return results


def run():
    """Generate every prelisted condition before any supplementary truth evaluation."""
    inventory=pd.read_csv(OUTPUT/"inventory.csv")
    names=inventory.loc[inventory.exists & inventory.sufficient_length,"name"].tolist()
    started=time.perf_counter()
    records=[]
    with experiment.replay_pool(len(names),6) as pool:
        for future in as_completed([pool.submit(replay_record,name) for name in names]):
            records.extend(future.result())
            experiment.save_json(OUTPUT/"records.json",records)
            print(f"GENERATED {len(records)}/16 wall={time.perf_counter()-started:.1f}s",flush=True)
    protocol=json.loads((OUTPUT/"protocol.json").read_text(encoding="utf8"))
    experiment.save_json(OUTPUT/"generation_provenance.json",dict(wall_seconds=time.perf_counter()-started,
        original_source_unchanged=experiment.audit.native.ours.digest(experiment.audit.native.ours.AUTHOR)==protocol["source_hashes"][str(experiment.audit.native.ours.AUTHOR)],
        actual_original_class_arms=2*len(records),workers=min(6,len(names)),BLAS_threads=1))


def score():
    """Evaluate full native feedback histories, with no two-second return-to-K insertion."""
    records=json.loads((OUTPUT/"records.json").read_text(encoding="utf8"))
    if len(records)!=16:
        raise RuntimeError("All16 registered supplementary conditions must finish before scoring")
    metrics=[]
    for record in records:
        truth=np.load(OUTPUT/(record["name"]+"__data.npz"))["truth"]
        K=np.load(OUTPUT/"K"/(record["key"]+".npz"))["rows"]
        feedback=np.load(OUTPUT/"feedback"/(record["key"]+".npz"))["rows"]
        for method,rows in (("K",K),("single_XY_feedback",feedback)):
            item={key:record[key] for key in ("name","key","gap_s","failed","failure","trigger_count")}
            item.update(method=method,trigger_shift_m=0. if record["trigger"] is None else record["trigger"]["distance_m"])
            for label,(left,right) in experiment.WINDOWS.items():
                values=experiment.audit.native.ours.util.score_rows(truth,rows,record["resume"]+left,record["resume"]+right-.05)
                item.update({label+"_"+key:value for key,value in values.items()})
                item[label+"_coverage"]=values["samples"]/round((right-left)/.05)
            phase=(rows[1:,0]>=record["resume"])&(rows[1:,0]<=record["resume"]+11.)
            item["maximum_postresume_xy_step_m"]=float(np.max(np.linalg.norm(np.diff(rows[:,1:3],axis=0)[phase],axis=1)))
            if rows.shape==K.shape:
                item["maximum_added_xy_step_m"]=float(np.max(np.linalg.norm(np.diff(rows[:,1:3]-K[:,1:3],axis=0),axis=1)))
            metrics.append(item)
    frame=pd.DataFrame(metrics)
    baseline=frame[frame.method=="K"].set_index("key")
    for window in experiment.WINDOWS:
        for metric in ("xy_rmse","yaw_rmse_deg"):
            column=window+"_"+metric
            frame[column+"_minus_K"]=frame.apply(lambda row:row[column]-baseline.loc[row.key,column],axis=1)
    scoring=OUTPUT/"feedback_scores"
    scoring.mkdir(exist_ok=True)
    frame.to_csv(scoring/"paired.csv",index=False)
    columns=[window+"_"+metric for window in experiment.WINDOWS for metric in ("xy_rmse","yaw_rmse_deg","coverage")]
    frame.groupby(["gap_s","method"])[columns].median().to_csv(scoring/"summary.csv")
    print(frame.groupby(["gap_s","method"])[columns].median().to_string())


def main():
    """Make preregistration, sensor replay and later truth scoring explicit stages."""
    parser=argparse.ArgumentParser()
    parser.add_argument("--stage",choices=("freeze","run","score"),required=True)
    stage=parser.parse_args().stage
    {"freeze":freeze,"run":run,"score":score}[stage]()


if __name__=="__main__":
    main()
