"""Run actual author K and frozen recovery candidates on independent trial-2 TDoA.

Ground truth is cached for later scoring only. All 32 K trajectories use the
untouched author class; the fast mirror independently checks their outputs.
"""

import argparse
import copy
from concurrent.futures import as_completed
import importlib.util
import json
import sys
import time
import warnings

import audit
from diagnose import save_json
import fast_original as fast
from validate_fast import inputs
import numpy as np
import pandas as pd
sys.path.insert(0,str(audit.EXPERIMENTS/"split_handoff"))
from cpu_replay import replay_pool

# This entire cohort was frozen from file availability before its scores were read.
HERE=audit.HERE
OUTPUT=HERE/"independent"
GAPS=(2.,5.,10.,20.)
RESUME=25.
END=36.
VARIANCE=.13
SIGMA=np.sqrt(VARIANCE)
WINDOWS={"early":(0.,2.),"middle":(2.,5.),"later":(6.,11.)}
METHODS=("K","always_C","guarded","radio_bounds")


def row(estimator,stamp):
    """Extract source physical state with the same trace columns as native replay."""
    return np.r_[stamp,estimator.X[:3,4],estimator.X[:3,3],estimator.X[:3,:3].ravel(),np.zeros(3)]


def failure(estimator):
    """Keep the existing explicit numerical-failure rule without excluding bad scores."""
    return (not np.all(np.isfinite(estimator.X)) or not np.all(np.isfinite(estimator.P))
            or np.max(np.abs(estimator.X[:3,4]))>1000)


def generate_record(name):
    """Share the exact original prefix, cloning full objects at each outage start."""
    target=OUTPUT/"K"
    target.mkdir(parents=True,exist_ok=True)
    existing=[target/(f"{name}__gap{gap:g}.json") for gap in GAPS]
    if all(path.exists() for path in existing):
        return [json.loads(path.read_text(encoding="utf8")) for path in existing]
    started=time.perf_counter()
    data=audit.native.load_raw(name,False)
    truth=data.pop("truth")
    common=audit.native.ours.util.load(name,duration=45.)
    common.pop("truth")
    start,X,bias=audit.native.ours.util.initial_state(common)
    P=audit.native.ORIGINAL_P.copy()
    normal=audit.native.ours.make_filter(X,P,bias)
    origin=data["common_origin"]
    resume=origin+RESUME
    events,ii,ri=inputs(data)
    keep=events<=origin+END
    events,ii,ri=events[keep],ii[keep],ri[keep]
    first_radio={}
    for index,stamp in enumerate(data["radio"][:,0]):
        first_radio.setdefault(float(stamp),index)
    used_radio=data["radio"][np.array(list(first_radio.values()),dtype=int)]
    used_radio=used_radio[used_radio[:,0]<=origin+END]
    np.savez_compressed(OUTPUT/(name+"__data.npz"),times=data["times"],imu=data["imu"],radio=data["radio"],
        used_radio=used_radio,anchors=data["anchors"],truth=truth,origin=origin,initial_time=start,
        initial_X=X,initial_P=P,initial_bias=bias)
    normal_history=[]
    normal_info=dict(predict_calls=0,correction_calls=0,accepted_radio=0,rejected_radio=0,failed=False,failure=None)
    branches,histories,infos={}, {}, {}
    previous_imu,last_output=start,-np.inf
    splits={}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore",DeprecationWarning)
        for k in range(max(1,np.searchsorted(events,start,side="right")),len(events)):
            previous,stamp=float(events[k-1]),float(events[k])
            if previous<=start:
                continue
            for gap in GAPS:
                if gap not in branches and previous>=resume-gap:
                    branches[gap]=copy.deepcopy(normal)
                    histories[gap]=list(normal_history)
                    infos[gap]=dict(normal_info)
                    splits[gap]=dict(X=normal.X.copy(),P=normal.P.copy(),bias=normal.bias.copy(),
                        X_hat=normal.X_hat.copy(),P_hat=normal.P_hat.copy(),previous_imu_time=previous_imu,time=previous)
            dt=None
            if ii[k-1]>=0:
                dt=stamp-previous_imu
                previous_imu=stamp
            active=[(gap,branches[gap],infos[gap]) for gap in branches]
            if len(branches)<len(GAPS):
                active.append((None,normal,normal_info))
            for gap,estimator,info in active:
                if info["failed"]:
                    continue
                try:
                    if dt is not None:
                        estimator.predict(data["imu"][ii[k-1]],dt)
                        info["predict_calls"]+=1
                    off=gap is not None and resume-gap<=previous<resume
                    if ri[k-1]>=0 and not off:
                        _,_,_,discard=estimator.correct(data["radio"][ri[k-1],1:],data["anchors"])
                        info["accepted_radio"]+=int(not discard)
                        info["rejected_radio"]+=int(discard)
                        info["correction_calls"]+=1
                    if failure(estimator):
                        info.update(failed=True,failure="nonfinite_or_position_exceeds_1000m")
                except Exception as exception:
                    info.update(failed=True,failure=type(exception).__name__+": "+str(exception))
            if stamp-last_output>=.01:
                for gap,estimator,info in active:
                    if not info["failed"]:
                        (normal_history if gap is None else histories[gap]).append(row(estimator,stamp))
                last_output=stamp
    records=[]
    for gap in GAPS:
        key=f"{name}__gap{gap:g}"
        rows=np.array(histories[gap]).reshape(-1,19)
        estimator=branches[gap]
        np.savez_compressed(target/(key+".npz"),rows=rows,final_X=estimator.X,final_P=estimator.P,final_bias=estimator.bias,
            final_X_hat=estimator.X_hat,final_P_hat=estimator.P_hat,
            **{"split_"+key:value for key,value in splits[gap].items()})
        mirrored=fast.replay(events,ii,ri,data["imu"],data["radio"],data["anchors"],X,P,np.r_[bias[3:],bias[:3]],
            start,start,resume-gap,resume)
        FX,FP,Fb,mirror_rows,accepted,rejected,*_=mirrored
        same_shape=rows.shape==mirror_rows.shape
        validation=dict(row_shape_equal=same_shape,rows_max_abs=None if not same_shape else float(np.max(np.abs(rows-mirror_rows))),
            P_max_abs=float(np.max(np.abs(estimator.P-FP))),bias_max_abs=float(np.max(np.abs(estimator.bias-Fb))),
            accepted_equal=accepted==infos[gap]["accepted_radio"],rejected_equal=rejected==infos[gap]["rejected_radio"])
        validation["validated"]=bool(same_shape and validation["rows_max_abs"]<1e-7 and validation["P_max_abs"]<1e-7
            and validation["bias_max_abs"]<1e-9 and validation["accepted_equal"] and validation["rejected_equal"])
        record=dict(name=name,gap_s=gap,key=key,origin=origin,resume=resume,nominal_yaw_offset_deg=0,
            backend="untouched author class",shared_exact_prefix=True,mirror_validation=validation,
            raw_radio_rows=int(np.sum(data["radio"][:,0]<=origin+END)),native_radio_rows=len(used_radio),
            recording_wall_seconds=time.perf_counter()-started,**infos[gap])
        save_json(target/(key+".json"),record)
        records.append(record)
    return records


def load_module(name,path):
    """Load a frozen shared candidate by its exact source file."""
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    return module


def candidates():
    """Generate both frozen candidates while leaving K and its yaw untouched."""
    shadow_path=audit.EXPERIMENTS/"strong_recovery"/"shadow.py"
    bounds_path=audit.EXPERIMENTS/"strong_recovery"/"radio_bounds.py"
    shadow=load_module("toronto_frozen_shadow",shadow_path)
    bounds=load_module("toronto_frozen_bounds",bounds_path)
    target=OUTPUT/"candidates"
    target.mkdir(parents=True,exist_ok=True)
    records=json.loads((OUTPUT/"K_records.json").read_text(encoding="utf8"))
    save_json(OUTPUT/"candidate_protocol.json",dict(methods=METHODS,sigma_m=float(SIGMA),author_variance_m2=VARIANCE,
        lever_radius_m=float(np.linalg.norm(fast.LEVER)),windows_after_resume_s=WINDOWS,
        source_hashes={str(path):audit.native.ours.digest(path) for path in (shadow_path,bounds_path,HERE/"independent.py")},
        truth="candidate generation never accesses cached truth array; scores only after all candidates finish"))
    results=[]
    for record in records:
        data=np.load(OUTPUT/(record["name"]+"__data.npz"))
        K=np.load(OUTPUT/"K"/(record["key"]+".npz"))["rows"]
        raw=data["used_radio"]
        raw=raw[raw[:,0]>=record["resume"]]
        plus=data["anchors"][raw[:,2].astype(int)]
        minus=data["anchors"][raw[:,1].astype(int)]
        lever_world=np.einsum("nij,j->ni",K[:,7:16].reshape(-1,3,3),fast.LEVER)
        shadow_result=shadow.recover(K[:,0],K[:,1:4],K[:,4:7],lever_world,raw[:,0],plus,minus,raw[:,3],SIGMA,record["resume"])
        bounded=bounds.bound_xy(K[:,0],K[:,1:4],float(np.linalg.norm(fast.LEVER)),raw[:,0],plus,minus,raw[:,3],SIGMA,record["resume"])
        arrays={method:K.copy() for method in METHODS}
        arrays["guarded"][:,1:4]=shadow_result["pos"]
        arrays["guarded"][:,4:7]=shadow_result["vel"]
        arrays["always_C"][:,1:4]=shadow_result["always_c_pos"]
        arrays["always_C"][:,4:7]=shadow_result["always_c_vel"]
        arrays["radio_bounds"][:,1:4]=bounded["pos"]
        np.savez_compressed(target/(record["key"]+".npz"),**arrays,use_c=shadow_result["use_c"],bounds_active=bounded["active"])
        details=dict(record,shadow=shadow_result["diagnostics"],bounds=bounded["diagnostics"],
            candidate_source="frozen shared modules; original K recursion unchanged",
            preservation={method:dict(rotation_max_abs=float(np.max(np.abs(rows[:,7:16]-K[:,7:16]))),
                z_max_abs=float(np.max(np.abs(rows[:,3]-K[:,3])))) for method,rows in arrays.items()})
        save_json(target/(record["key"]+".json"),details)
        results.append(details)
    save_json(OUTPUT/"candidate_records.json",results)


def score():
    """Use the common deadline scorer for both direct calls and the CLI."""
    from score_independent import main as score_with_deadline
    score_with_deadline()


def main():
    """Stage K, candidate construction and truth scoring as explicit separate steps."""
    parser=argparse.ArgumentParser()
    parser.add_argument("--stage",choices=["k","candidates","score"],required=True)
    args=parser.parse_args()
    OUTPUT.mkdir(parents=True,exist_ok=True)
    if args.stage=="candidates":
        candidates()
    elif args.stage=="score":
        score()
    else:
        inventory=pd.read_csv(OUTPUT/"inventory.csv")
        names=inventory.loc[inventory.exists & inventory.sufficient_length,"name"].tolist()
        save_json(OUTPUT/"K_protocol.json",dict(names=names,trial=2,gaps_s=GAPS,resume_s=RESUME,end_s=END,workers=6,
            original_class_sha256=audit.native.ours.digest(audit.native.ours.AUTHOR),source_sha256=audit.native.ours.digest(HERE/"independent.py"),
            backend="every K uses untouched source class; mirror only checks equivalence",
            initial="same sensor-only0.6s X/gyro bias; original invariant P exactly; nominal yaw0",
            radio_variance_m2=VARIANCE,candidate_radio_sigma_m=float(SIGMA),
            queue="original raw native queue, first-per-stamp, same event clocks during mask",
            reuse="common full author prefix; deepcopies include X/P/Xhat/Phat/bias and previous IMU timestamp",
            evaluation="no candidate scores consulted; GT stored only for later scoring"))
        records=[]
        started=time.perf_counter()
        with replay_pool(len(names),6) as pool:
            for future in as_completed([pool.submit(generate_record,name) for name in names]):
                records.extend(future.result())
                save_json(OUTPUT/"K_records.json",records)
                print(f"K DONE {len(records)}/32 {records[-1]['name']} wall={time.perf_counter()-started:.1f}s",flush=True)
        save_json(OUTPUT/"K_provenance.json",dict(wall_seconds=time.perf_counter()-started,
            original_class_modified=audit.native.ours.digest(audit.native.ours.AUTHOR)!=json.loads((OUTPUT/"K_protocol.json").read_text(encoding="utf8"))["original_class_sha256"]))


if __name__=="__main__":
    main()
