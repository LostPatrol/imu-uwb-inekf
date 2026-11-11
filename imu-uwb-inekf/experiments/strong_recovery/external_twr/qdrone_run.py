"""Freeze and run the complete public QDrone TWR cohort with matched displays.

Eligibility uses schema, timing and supplied truth coverage; no estimator error
is read during preparation. Uniform score times avoid weighting copied truth rows.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import traceback

import numpy as np
import pandas as pd

from adapter import CODE,ROOT,SensorRecord
from qdrone import inventory,make_record,prism_truth,truth_at
from replay import RESUME,LENGTHS,reference_replay,apply_shadow,write_json

# Report original packet logs separately from fixed-rate descriptive RMSE scores.
SCORE_DT = .05
# Primary summaries require at least 95% of a predeclared window's truth support.
MIN_SCORE_COVERAGE = .95
# Windows are relative to the same unshifted recording-clock T=25 s.
WINDOWS = (("first2",0.,2.),("two_to_five",2.,5.),("six_to_eleven",6.,11.))
# Complete raw input must retain 45 s regardless of published score availability.
MIN_DURATION = 45.


def cache_record(path,record):
    """Write only sensor arrays and explicit geometry into an estimator cache."""
    np.savez_compressed(path,name=record.name,imu_time=record.imu_time,imu=record.imu,
        radio_time=record.radio_time,radio_id=record.radio_id,radio_range=record.radio_range,
        anchors=record.anchors,lever=record.lever,initial_rotation=record.initial_rotation,
        initial_bias=record.initial_bias,origin_s=record.origin_s,
        metadata_json=json.dumps(record.metadata))


def load_record(path):
    """Recreate a truth-free sensor record inside an independent replay worker."""
    data=dict(np.load(path))
    data["name"]=str(data["name"])
    data["origin_s"]=float(data["origin_s"])
    data["metadata"]=json.loads(str(data.pop("metadata_json")))
    return SensorRecord(**data)


def prepare(dataset,output,anchors_path,clock_mode="primary"):
    """Freeze all public recordings and timing support without generating estimates."""
    output.mkdir(parents=True,exist_ok=True)
    cache=output/"cache"
    cache.mkdir(exist_ok=True)
    anchors=json.loads(anchors_path.read_text(encoding="utf-8"))
    cohort=inventory(dataset)
    for entry in cohort:
        entry["status"]="no_published_truth"
        if not entry["truth_path"]:
            continue
        try:
            geometry=anchors[entry["site"]]
            if isinstance(geometry,dict):
                geometry=[geometry[str(identifier)] for identifier in (100,101,102,103)]
            record=make_record(entry,geometry)
            resume=RESUME if clock_mode=="primary" else record.metadata["raw_common_midpoint_s"]
            entry["resume_s"]=float(resume)
            record.metadata["resume_s"]=float(resume)
            truth,truth_audit=prism_truth(Path(entry["truth_path"]))
            entry["sensor_metadata"]=record.metadata
            entry["truth_audit"]=truth_audit
            entry["duration_s"]=float(record.imu_time[-1])
            clock=resume+np.arange(0.,11.,SCORE_DT)
            _,supported=truth_at(clock+record.origin_s,truth)
            entry["truth_coverage"]={label:float(np.mean(supported[(clock>=resume+begin)&(clock<resume+end)]))
                for label,begin,end in WINDOWS}
            if not record.metadata["specific_force_supported"]:
                entry.update(status="unsupported_IMU_gravity_convention",
                    reason="Raw acceleration near zero rather than specific force; no quaternion/add-gravity reconstruction is used")
                continue
            if record.imu_time[-1]<MIN_DURATION:
                entry.update(status="unsupported_sensor_duration",reason="Less than fixed required 45 s")
                continue
            entry["status"]="sensor_eligible"
            identifier=entry["site"]+"__"+entry["name"]
            cache_record(cache/(identifier+".npz"),record)
            np.savez_compressed(cache/(identifier+"_scoring.npz"),truth=truth,
                truth_source_sha256=hashlib.sha256(Path(entry["truth_path"]).read_bytes()).hexdigest())
        except Exception:
            entry.update(status="failed_preflight",reason=traceback.format_exc())
    protocol=dict(dataset="QDrone",revision="cd3774c9db03899afde886a52f06687bbe4a1ed1",
        resume_s=RESUME if clock_mode=="primary" else "per-record raw common interval midpoint",clock_mode=clock_mode,
        lengths_s=LENGTHS,score_grid_dt_s=SCORE_DT,min_primary_score_coverage=MIN_SCORE_COVERAGE,
        score_windows=WINDOWS,clock="Original released recording seconds; no shift by 1.5 s or test GT onset",
        sensor_initialization="First 1.5 s raw static tilt/bias, yaw zero; replay starts after prefix on unchanged clock",
        input="All Datasets raw stream, never GT-trimmed sensor packets; gyro first3 rad/s and next3 specific force m/s^2",
        no_DJI_quaternion_or_height=True,no_test_GT_range_correction=True,
        geometry=json.loads(anchors_path.read_text(encoding="utf-8")),
        anchor_source_path=str(anchors_path),anchor_file_sha256=hashlib.sha256(anchors_path.read_bytes()).hexdigest(),
        original_complete_capture_tracker_preserved=True,radio_bias_m=0.,radio_sigma_m=.3,
        K_and_C_nominal_lever_m=[0.,0.,0.],bounds_engineering_lever_radius_m=1.,
        physical_lever_is_not_verified=True,
        truth="Official inserted prism XYZ at timestamped events only; no boundary extrapolation; published interpolation is not independent dense station measurements",
        no_estimated_attitude_ground_truth=True,provider_raw_total_station_clock_unavailable=True,
        cohort_frozen_before_any_estimator_scores=True,cohort=cohort)
    protocol["implementation_sha256"]={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
        for p in Path(__file__).parent.glob("*.py")}
    protocol["display_sha256"]={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (CODE/"experiments/strong_recovery/shadow.py",CODE/"experiments/strong_recovery/radio_bounds.py")}
    write_json(output/"protocol.json",protocol)
    write_json(output/"eligibility.json",cohort)
    return cohort


def score_outputs(record,result,truth,resume=RESUME):
    """Score each independent recording on a uniform common clock with support reported."""
    clock=resume+np.arange(0.,11.,SCORE_DT)
    ground,supported=truth_at(clock+record.origin_s,truth)
    indices=np.clip(np.searchsorted(result["time"],clock,side="right")-1,0,len(result["time"])-1)
    rows=[]
    for method in ("K","proposal","always_C","radio_bounds"):
        sampled=result[method][indices].copy()
        # The display contract ends at exactly T+2, including off-grid score times.
        if method!="K":
            sampled[clock>=resume+2.]=result["K"][indices[clock>=resume+2.]]
        error=sampled-ground
        finite=np.isfinite(error).all(axis=1)
        for label,begin,end in WINDOWS:
            window=(clock>=resume+begin)&(clock<resume+end)
            valid=supported&finite&window
            coverage=float(np.sum(valid)/max(1,np.sum(window)))
            values=error[valid]
            row=dict(method=method,window=label,truth_coverage=float(np.mean(supported[window])),
                evaluated_fraction=coverage,complete_window=bool(coverage>=MIN_SCORE_COVERAGE),
                samples=int(np.sum(valid)),scoring_grid_hz=1/SCORE_DT,
                xy_rmse_m=float(np.sqrt(np.mean(np.sum(values[:,:2]**2,axis=1)))) if len(values) else np.nan,
                z_rmse_m=float(np.sqrt(np.mean(values[:,2]**2))) if len(values) else np.nan,
                xy_peak_m=float(np.max(np.linalg.norm(values[:,:2],axis=1))) if len(values) else np.nan)
            rows.append(row)
    result["score_time"],result["score_truth"],result["score_truth_supported"]=clock,ground,supported
    return rows


def run_case(job):
    """Complete all estimators before opening the separate scoring cache."""
    output,entry,length=job
    output=Path(output)
    identifier=entry["site"]+"__"+entry["name"]
    record=load_record(output/"cache"/(identifier+".npz"))
    resume=float(entry.get("resume_s",RESUME))
    result,events,audit=reference_replay(record,length,resume=resume)
    case=identifier+"__L"+str(int(length))
    try:
        if audit["failure"] is None:
            audit["display_diagnostics"]=apply_shadow(record,result,length,resume=resume)
        else:
            raise RuntimeError("Native full reference failed; display output unavailable")
    except Exception:
        audit["display_failure"]=traceback.format_exc()
        for method in ("proposal","always_C","radio_bounds"):
            result[method]=np.full_like(result["K"],np.nan)
    pd.DataFrame(events).to_csv(output/(case+"_events.csv"),index=False)
    np.savez_compressed(output/(case+".npz"),**result)
    write_json(output/(case+".json"),audit)
    truth=np.load(output/"cache"/(identifier+"_scoring.npz"))["truth"]
    rows=score_outputs(record,result,truth,resume=resume)
    for row in rows:
        row.update(sequence=entry["name"],site=entry["site"],length_s=length,
            resume_s=resume,
            native_failure=audit["failure"] is not None,display_failure="display_failure" in audit)
    np.savez_compressed(output/(case+".npz"),**result)
    return rows


def main():
    """Prepare first, then execute every sensor-eligible case without selecting by errors."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",type=Path,default=Path("E:/datasets/qdrone-benchmark"))
    parser.add_argument("--anchors",type=Path,default=Path(__file__).parent/"data_provenance/qdrone_anchors_final.json")
    parser.add_argument("--output",type=Path,default=ROOT/"agent/codex/strong_external/qdrone_verified")
    parser.add_argument("--prepare-only",action="store_true")
    parser.add_argument("--run",action="store_true")
    parser.add_argument("--workers",type=int)
    parser.add_argument("--clock",choices=("primary","midpoint"),default="primary")
    args=parser.parse_args()
    if args.prepare_only or not args.run:
        prepare(args.dataset,args.output,args.anchors,args.clock)
        return
    cohort=json.loads((args.output/"eligibility.json").read_text(encoding="utf-8"))
    jobs=[(str(args.output),entry,length) for entry in cohort if entry["status"]=="sensor_eligible" for length in LENGTHS]
    sys.path.insert(0,str(CODE/"experiments/split_handoff"))
    from cpu_replay import replay_pool
    from concurrent.futures import as_completed
    rows=[]
    with replay_pool(len(jobs),workers=args.workers) as pool:
        futures={pool.submit(run_case,job):job for job in jobs}
        for number,future in enumerate(as_completed(futures),1):
            job=futures[future]
            try:
                rows.extend(future.result())
            except Exception:
                failure=dict(site=job[1]["site"],sequence=job[1]["name"],length_s=job[2],traceback=traceback.format_exc())
                write_json(args.output/(job[1]["site"]+"__"+job[1]["name"]+"__L"+str(int(job[2]))+"_failure.json"),failure)
            pd.DataFrame(rows).to_csv(args.output/"metrics.csv",index=False)
            print(f"DONE {number}/{len(jobs)} {job[1]['site']} {job[1]['name']} L={job[2]}",flush=True)
    write_json(args.output/"source_sha256.json",{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")})


if __name__=="__main__":
    main()
