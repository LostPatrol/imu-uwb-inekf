"""Compare matched raw-range solvers on actual data windows and yaw-frame offsets.

Yaw offsets are equivalent parameterizations, not newly collected trajectories.
All solvers share the same robust CV translation seed and original range objective.
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from initializer import solve_window,lifted_candidate,yaw_rotation
from run_research import PROJECT,window_data,write_json

# Endpoints and durations are frozen before looking at optimization outcomes.
ENDPOINTS_S=(5.,10.,20.,30.)
DURATIONS_S=(2.,4.,8.)
# Equivalent gravity-aligned frame rotations challenge initial yaw without corrupting position seeds.
YAW_FRAMES_DEG=(0.,90.,180.,270.)
# A missed optimum is measured against the best candidate found, not a global certificate.
RELATIVE_COST_TOLERANCE=1e-4
# Root and geometry studies share this host; cap process-level parallelism.
WORKERS=4
# Avoid repeated archive decompression in each worker while preserving immutable arrays.
_DATA={}


def load_data(cache,name):
    """Load immutable per-log data once per worker."""
    key=(cache,name)
    if key not in _DATA:
        _DATA[key]=dict(np.load(Path(cache)/(Path(name).stem+".npz")))
    return _DATA[key]


def arrays_for_job(job):
    """Extract arrived events and apply a physically equivalent yaw coordinate change."""
    cache,name,endpoint,duration,layout,frame_deg,order=job
    data=load_data(cache,name)
    model=json.loads((Path(cache)/"manifest.json").read_text(encoding="utf-8"))["model"]
    times=data["time"]
    end=int(np.searchsorted(times,endpoint,side="right")-1)
    start=int(np.searchsorted(times,times[end]-duration,side="left"))
    tau,d,a,r,sigma=window_data(data,start,end,model)
    if layout=="lower4":
        selected=data["anchors"][np.argsort(data["anchors"][:,2])[:4]]
        keep=np.any(np.all(a[:,None,:]==selected[None,:,:],axis=2),axis=1)
        tau,d,a,r,sigma=tau[keep],d[keep],a[keep],r[keep],sigma[keep]
    angle=np.deg2rad(frame_deg)
    rotated=d@yaw_rotation(angle).T
    # Numerically verify coordinate equivalence using a fixed test yaw.
    np.testing.assert_allclose(rotated@yaw_rotation(0.71-angle).T,
                               d@yaw_rotation(0.71).T,atol=2e-13)
    return (tau,rotated,a,r,sigma),data["truth_position"][end],float(times[end])


def evaluate(job):
    """Run three unchanged solvers with balanced order on identical window arrays."""
    values,truth,end=arrays_for_job(job)
    cache,name,endpoint,duration,layout,frame_deg,order=job
    methods=np.roll(np.array(["local","profile","multistart"]),order%3)
    rows=[]
    for method in methods:
        try:
            fit=solve_window(*values,method=str(method))
            row=dict(method=str(method),objective=fit.robust_objective,elapsed_s=fit.elapsed_s,
                evaluations=fit.evaluations,solver_success=fit.diagnostics.get("solver_success",False),
                finite=bool(np.isfinite(fit.parameters).all()),position_error_m=float(np.linalg.norm(fit.parameters[:3]-truth)),
                yaw_in_original_frame=float(np.arctan2(np.sin(fit.parameters[6]+np.deg2rad(frame_deg)),
                                                       np.cos(fit.parameters[6]+np.deg2rad(frame_deg)))),
                data_yaw_information=fit.yaw_information)
            if method=="profile":
                singular=np.asarray(fit.diagnostics["projected_yaw_singular"])
                row["profile_min_singular"]=float(singular.min())
                row["profile_condition"]=float(singular.max()/max(singular.min(),1e-30))
            else:
                row["profile_min_singular"]=float("nan")
                row["profile_condition"]=float("nan")
        except Exception as exc:
            row=dict(method=str(method),objective=float("nan"),elapsed_s=float("nan"),evaluations=0,
                     solver_success=False,finite=False,error=repr(exc))
        row.update(sequence=name,endpoint_requested_s=endpoint,endpoint_actual_s=end,duration_s=duration,
                   layout=layout,frame_deg=frame_deg,events=len(values[0]))
        rows.append(row)
    return rows


def summarize(frame,output):
    """Report all finite failures and numerical misses alongside median and p95 cost."""
    keys=["sequence","endpoint_requested_s","duration_s","layout","frame_deg"]
    best=frame.groupby(keys)["objective"].transform("min")
    frame["objective_gap"]=frame["objective"]-best
    frame["relative_objective_gap"]=frame["objective_gap"]/(1+best)
    frame["missed_best_found"]=(frame["relative_objective_gap"]>RELATIVE_COST_TOLERANCE)|(~frame["finite"])
    frame.to_csv(output/"metrics.csv",index=False)
    summaries=[]
    for (layout,method),part in frame.groupby(["layout","method"]):
        summaries.append(dict(layout=layout,method=method,cases=len(part),
            missed_best_found=int(part["missed_best_found"].sum()),
            solver_nonconverged=int((~part["solver_success"]).sum()),nonfinite=int((~part["finite"]).sum()),
            median_ms=float(part["elapsed_s"].median()*1000),p95_ms=float(part["elapsed_s"].quantile(.95)*1000),
            max_ms=float(part["elapsed_s"].max()*1000),median_nfev=float(part["evaluations"].median()),
            p95_nfev=float(part["evaluations"].quantile(.95)),total_nfev=int(part["evaluations"].sum()),
            p95_relative_gap=float(part["relative_objective_gap"].quantile(.95)),
            max_relative_gap=float(part["relative_objective_gap"].max()),
            median_position_error_m=float(part["position_error_m"].median())))
    summary=pd.DataFrame(summaries)
    summary.to_csv(output/"summary.csv",index=False)
    pairs=frame.pivot(index=keys,columns="method",values="missed_best_found")
    contrast=dict(profile_wins_over_local=int((~pairs["profile"]&pairs["local"]).sum()),
                  local_wins_over_profile=int((pairs["profile"]&~pairs["local"]).sum()),
                  both_miss=int((pairs["profile"]&pairs["local"]).sum()))
    write_json(output/"paired_decision.json",contrast)
    print(summary.to_string(index=False))
    print(json.dumps(contrast))


def main():
    """Freeze a real-window matrix and execute no more than four worker processes."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache",type=Path,default=PROJECT/"results/recovery_bootstrap_20261107_final/cache")
    parser.add_argument("--output",type=Path,default=PROJECT/"results/recovery_window_solver_20261107")
    parser.add_argument("--workers",type=int,default=WORKERS)
    args=parser.parse_args()
    base=json.loads((args.cache/"manifest.json").read_text(encoding="utf-8"))
    jobs=[]
    for name in base["evaluation"]:
        data=load_data(str(args.cache),name)
        for endpoint in ENDPOINTS_S:
            if endpoint>data["time"][-1]:
                continue
            for duration in DURATIONS_S:
                if duration>endpoint:
                    continue
                for layout in ("all8","lower4"):
                    for frame in YAW_FRAMES_DEG:
                        jobs.append((str(args.cache),name,endpoint,duration,layout,frame,len(jobs)))
    args.output.mkdir(parents=True,exist_ok=True)
    protocol=dict(endpoints_s=ENDPOINTS_S,durations_s=DURATIONS_S,yaw_frames_deg=YAW_FRAMES_DEG,
        layouts=["all eight anchors","four anchors with smallest known z coordinate"],
        same_cv_translation_seed=True,relative_cost_tolerance=RELATIVE_COST_TOLERANCE,
        reference="Best objective found by these three methods; not a certified global raw-range optimum",
        independent_real_logs=6,independent_blind_data=False,windows=len(jobs),workers=args.workers,
        runtime_note="Per-fit wall clock under parallel host load; not unloaded latency or WCET",
        nfev_note="Raw inertial-refinement nfev summed across yaw seeds; common CV candidate nfev excluded equally",
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                       (Path(__file__),Path(__file__).parent/"initializer.py",Path(__file__).parent/"run_research.py")})
    write_json(args.output/"protocol.json",protocol)
    rows=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(evaluate,job) for job in jobs]
        for number,future in enumerate(as_completed(futures),1):
            rows.extend(future.result())
            if number%40==0:
                print(f"DONE {number}/{len(jobs)}",flush=True)
                pd.DataFrame(rows).to_csv(args.output/"partial_metrics.csv",index=False)
    summarize(pd.DataFrame(rows),args.output)


if __name__=="__main__":
    main()
