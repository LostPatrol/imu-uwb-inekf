"""Quantify source-author changes separately on the fixed mechanism-development set."""

from concurrent.futures import as_completed
import json
import sys
import time
import warnings

import audit
from diagnose import save_json
import fast_original as fast
import numpy as np
import pandas as pd
sys.path.insert(0,str(audit.EXPERIMENTS/"split_handoff"))
from cpu_replay import replay_pool

# Each row changes one declared operation; no threshold is learned from scores.
MODES=("mirrored_source","complete_jacobian","consistent_gate","refresh_prior","zero_updated_bias",
       "no_ground_clamp","class_default_radio_variance","physical_covariance_propagation")
CACHE={}


def run(job):
    """Replay one small real condition with exactly one declared mechanism change."""
    name,yaw,mode=job
    output=audit.HERE/"ablation"
    key=f"{name}__yaw{yaw:g}__{mode}"
    if (output/(key+".json")).exists():
        return json.loads((output/(key+".json")).read_text(encoding="utf8"))
    if (name,yaw) not in CACHE:
        CACHE[(name,yaw)]=audit.initial(name,yaw)
    data,start,X,P,canonical=CACHE[(name,yaw)]
    X,P,bias=X.copy(),P.copy(),np.r_[canonical[3:],canonical[:3]]
    Xhat,Phat=X.copy(),P.copy()
    imu_at,radio_at={},{}
    for index,stamp in enumerate(data["times"]):
        imu_at.setdefault(float(stamp),index)
    for index,stamp in enumerate(data["radio"][:,0]):
        radio_at.setdefault(float(stamp),index)
    events=np.unique(np.r_[data["times"],data["radio"][:,0]])
    last_output,previous_imu=-np.inf,start
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
                    dt=stamp-previous_imu
                    previous_imu=stamp
                    sample=data["imu"][imu_at[previous]]
                    if mode=="physical_covariance_propagation":
                        _,physical_P=audit.kernel.predict(X,P,np.r_[bias[3:],bias[:3]],sample,dt)
                    Xhat,Phat=fast.predict(X,P,bias,sample,dt,mode!="no_ground_clamp")
                    if mode=="physical_covariance_propagation":
                        Phat=physical_P
                    X,P=Xhat.copy(),Phat.copy()
                    predicts+=1
                if previous in radio_at:
                    packet=data["radio"][radio_at[previous],1:]
                    X,P,bias,keep=fast.correction(Xhat,Phat,bias,packet,data["anchors"],
                        .05 if mode=="class_default_radio_variance" else .13,
                        int(mode=="complete_jacobian"),int(mode=="consistent_gate"),fast.LEVER)
                    accepted+=int(keep)
                    rejected+=int(not keep)
                    corrections+=1
                    if mode=="refresh_prior":
                        Xhat,Phat=X.copy(),P.copy()
                    if mode=="zero_updated_bias":
                        bias[:]=0.
                if stamp-last_output>=.01:
                    rows.append(np.r_[stamp,X[:3,4],X[:3,3],X[:3,:3].ravel(),np.zeros(3)])
                    last_output=stamp
                if not np.all(np.isfinite(X)) or not np.all(np.isfinite(P)) or np.max(np.abs(X[:3,4]))>1000:
                    failure="nonfinite_or_position_exceeds_1000m"
                    break
        except Exception as exception:
            failure=type(exception).__name__+": "+str(exception)
    rows=np.array(rows).reshape(-1,19)
    np.savez_compressed(output/(key+".npz"),rows=rows,final_X=X,final_P=P,final_bias=bias)
    record=dict(name=name,yaw_offset_deg=yaw,mode=mode,key=key,origin=data["common_origin"],
        failed=failure is not None,failure=failure,accepted_radio=accepted,rejected_radio=rejected,
        predict_calls=predicts,correction_calls=corrections,runtime_seconds=time.perf_counter()-started)
    if mode=="mirrored_source":
        oracle=np.load(audit.HERE/"diagnostic"/(f"{name}__yaw{yaw:g}__author_original.npz"))
        record["oracle_rows_max_abs"]=float(np.max(np.abs(rows-oracle["rows"])))
        record["oracle_P_max_abs"]=float(np.max(np.abs(P-oracle["final_P"])))
    save_json(output/(key+".json"),record)
    return record


def main():
    """Freeze these mechanism controls, then score every completed trajectory."""
    output=audit.HERE/"ablation"
    output.mkdir(parents=True,exist_ok=True)
    save_json(output/"protocol.json",dict(names=audit.NAMES,yaws=audit.YAWS,modes=MODES,workers=6,
        independent_test=False,backend="full-trajectory validated author mirror; source unchanged",
        changes={"complete_jacobian":"only author transformed H replaced by exact physical world H; original pre-gate retained",
            "consistent_gate":"only pre-gate variance uses transformed H; update unchanged",
            "refresh_prior":"only refresh Xhat/Phat after each correction; same packets",
            "zero_updated_bias":"only set native bias to zero after corrections; initial sensor bias unchanged",
            "no_ground_clamp":"only omit negative-z/vzero nominal clamp",
            "class_default_radio_variance":"only .13 main variance replaced by original class default .05; no fitted noise",
            "physical_covariance_propagation":"only propagated P from existing standalone frozen-R physical discrete model; author nominal mean retained"},
        initial="same sensor-only X and original author P, nominal/180, raw native clock, no outage",
        source_hashes={str(path):audit.native.ours.digest(path) for path in (audit.native.ours.AUTHOR,audit.HERE/"fast_original.py",audit.HERE/"ablate.py")}))
    jobs=[(name,yaw,mode) for name in audit.NAMES for yaw in audit.YAWS for mode in MODES]
    records=[]
    started=time.perf_counter()
    with replay_pool(len(jobs),6) as pool:
        for future in as_completed([pool.submit(run,job) for job in jobs]):
            records.append(future.result())
            save_json(output/"records.json",records)
            print(f"DONE {len(records)}/{len(jobs)} {records[-1]['key']} wall={time.perf_counter()-started:.1f}s",flush=True)
    metrics=[]
    truths={}
    for record in records:
        name=record["name"]
        if name not in truths:
            truths[name]=audit.native.ours.util.load(name,duration=45.)["truth"]
        rows=np.load(output/(record["key"]+".npz"))["rows"]
        metric=dict(record)
        metric.update(audit.native.ours.util.score_rows(truths[name],rows,record["origin"]+5.,record["origin"]+44.95))
        metrics.append(metric)
    frame=pd.DataFrame(metrics)
    frame.to_csv(output/"metrics.csv",index=False)
    frame.groupby(["mode","yaw_offset_deg"])[["xy_rmse","z_rmse","yaw_rmse_deg"]].median().to_csv(output/"summary.csv")
    print(frame[["name","yaw_offset_deg","mode","xy_rmse","yaw_rmse_deg","failed"]].to_string(index=False))


if __name__=="__main__":
    main()
