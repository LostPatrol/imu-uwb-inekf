"""Frozen multi-candidate refinement and complete-cost comparison on real windows.

Profile candidates include the quartic stationary roots plus the explicit 0/pi
auxiliary seeds already returned by the lift. All methods keep the same CV seed.
"""

from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

import initializer
from initializer import (cv_candidate,lifted_candidate,raw_residual_jacobian,robust_cost,
    HUBER_SCALE,MAX_EVALUATIONS,project_out,CURVATURE_PRIOR_STD,solve_window)
from window_solver_study import (PROJECT,ENDPOINTS_S,DURATIONS_S,YAW_FRAMES_DEG,
    arrays_for_job,load_data,RELATIVE_COST_TOLERANCE)
from run_research import write_json

# These policies are frozen before examining multi-candidate outcomes.
VARIANTS=("profile_two","profile_all","uniform_four")
# Circular deduplication only removes numerically identical seeds, not nearby minima.
ANGLE_DUPLICATE_TOL_RAD=1e-7


def complete_cv(tau,anchors,ranges,sigmas):
    """Count both CV least-squares calls without changing their implementation."""
    original=initializer.least_squares
    counts=[]
    def counted(*args,**kwargs):
        """Record actual function-evaluation counts returned by scipy."""
        fit=original(*args,**kwargs)
        counts.append(fit.nfev)
        return fit
    try:
        initializer.least_squares=counted
        result=cv_candidate(tau,anchors,ranges,sigmas)
    finally:
        initializer.least_squares=original
    return result,sum(counts),len(counts)


def candidate_solve(values,method):
    """Run a fixed seed policy, including common CV construction and final curvature."""
    started=time.perf_counter()
    tau,d,a,r,sigma=values
    (cv,_,_),cv_nfev,cv_calls=complete_cv(tau,a,r,sigma)
    candidate_seconds=0.
    if method=="uniform_four":
        angles=np.array([0.,np.pi/2,np.pi,-np.pi/2])
    else:
        candidate_started=time.perf_counter()
        _,diagnostic=lifted_candidate(tau,d,a,r,sigma)
        unique=[]
        for angle in diagnostic["candidate_angles"]:
            if not any(abs(np.arctan2(np.sin(angle-old),np.cos(angle-old)))<ANGLE_DUPLICATE_TOL_RAD
                       for old in unique):
                unique.append(angle)
        angles=np.asarray(unique[:2] if method=="profile_two" else unique)
        candidate_seconds=time.perf_counter()-candidate_started
    def fun(x):
        """Evaluate the identical normalized original-range residual."""
        return raw_residual_jacobian(x,tau,d,a,r,sigma)[0]
    def jac(x):
        """Evaluate the identical analytic seven-variable Jacobian."""
        return raw_residual_jacobian(x,tau,d,a,r,sigma)[1]
    fits=[least_squares(fun,np.r_[cv,angle],jac=jac,loss="huber",f_scale=HUBER_SCALE,
                        max_nfev=MAX_EVALUATIONS) for angle in angles]
    best=min(fits,key=lambda fit:robust_cost(fit.fun))
    # Match the original solver's post-fit robust curvature work in timing comparisons.
    residual,jacobian=raw_residual_jacobian(best.x,tau,d,a,r,sigma)
    weights=np.minimum(1.,HUBER_SCALE/np.maximum(np.abs(residual),1e-12))
    weighted=jacobian*np.sqrt(weights[:,None])
    projected,_=project_out(weighted[:,:6],weighted[:,6:7])
    covariance=np.linalg.inv(weighted.T@weighted+np.diag(1/CURVATURE_PRIOR_STD**2))
    assert np.isfinite(covariance).all()
    return dict(objective=robust_cost(best.fun),elapsed_s=time.perf_counter()-started,
        raw_nfev=sum(fit.nfev for fit in fits),cv_nfev=cv_nfev,cv_calls=cv_calls,
        total_nfev=cv_nfev+sum(fit.nfev for fit in fits),seed_count=len(angles),
        candidate_s=candidate_seconds,solver_success=bool(best.success),finite=bool(np.isfinite(best.x).all()),
        data_yaw_information=float(np.sum(projected**2)),parameters=best.x)


def evaluate(job):
    """Compare the frozen policies on one unchanged actual-data window."""
    values,truth,end=arrays_for_job(job)
    rows=[]
    for method in np.roll(np.asarray(VARIANTS),job[-1]%len(VARIANTS)):
        fit=candidate_solve(values,str(method))
        x=fit.pop("parameters")
        row=dict(sequence=job[1],endpoint_requested_s=job[2],duration_s=job[3],layout=job[4],
            frame_deg=job[5],endpoint_actual_s=end,events=len(values[0]),method=str(method),
            position_error_m=float(np.linalg.norm(x[:3]-truth)),**fit)
        rows.append(row)
    return rows


def aggregate(base,new,output):
    """Compare six policies against their joint best found cost and group by actual log."""
    keys=["sequence","endpoint_requested_s","duration_s","layout","frame_deg"]
    common=new.groupby(keys)[["cv_nfev","cv_calls"]].first().reset_index()
    old=base.merge(common,on=keys,validate="many_to_one")
    old["raw_nfev"]=old["evaluations"]
    old["total_nfev"]=old["raw_nfev"]+old["cv_nfev"]
    old["seed_count"]=old["method"].map({"local":1,"profile":1,"multistart":12})
    combined=pd.concat([old,new],ignore_index=True)
    best=combined.groupby(keys)["objective"].transform("min")
    combined["relative_objective_gap"]=(combined["objective"]-best)/(1+best)
    combined["missed_best_found"]=(combined["relative_objective_gap"]>RELATIVE_COST_TOLERANCE)|(~combined["finite"])
    combined.to_csv(output/"combined_metrics.csv",index=False)
    rows=[]
    for (layout,method),part in combined.groupby(["layout","method"]):
        rows.append(dict(layout=layout,method=method,cases=len(part),misses=int(part["missed_best_found"].sum()),
            solver_nonconverged=int((~part["solver_success"]).sum()),nonfinite=int((~part["finite"]).sum()),
            total_nfev=int(part["total_nfev"].sum()),median_total_nfev=float(part["total_nfev"].median()),
            p95_total_nfev=float(part["total_nfev"].quantile(.95)),max_total_nfev=int(part["total_nfev"].max()),
            median_seeds=float(part["seed_count"].median()),max_seeds=int(part["seed_count"].max()),
            median_ms=float(part["elapsed_s"].median()*1000),p95_ms=float(part["elapsed_s"].quantile(.95)*1000)))
    summary=pd.DataFrame(rows)
    summary.to_csv(output/"summary.csv",index=False)
    combined.groupby(["sequence","layout","method"])["missed_best_found"].agg(["sum","count"]).to_csv(output/"per_log_misses.csv")
    sensitivity=[]
    for tolerance in (1e-4,1e-3,1e-2):
        for method,part in combined.groupby("method"):
            sensitivity.append(dict(relative_tolerance=tolerance,method=method,
                misses=int(((part["relative_objective_gap"]>tolerance)|(~part["finite"])).sum())))
    pd.DataFrame(sensitivity).to_csv(output/"tolerance_sensitivity.csv",index=False)
    print(summary.to_string(index=False))


def sequential_timing(cache,output):
    """Measure a predeclared twelve-window panel with ten interleaved repetitions."""
    names=json.loads((Path(cache)/"manifest.json").read_text(encoding="utf-8"))["evaluation"]
    jobs=[]
    for name in names:
        jobs.extend([(str(cache),name,10.,2.,"all8",0.,len(jobs)),
                     (str(cache),name,30.,8.,"lower4",180.,len(jobs)+1)])
    methods=("local","profile","profile_two","profile_all","uniform_four","multistart")
    values,_,_=arrays_for_job(jobs[0])
    for method in methods:
        candidate_solve(values,method) if method in VARIANTS else solve_window(*values,method=method)
    rows=[]
    for number,job in enumerate(jobs):
        values,_,_=arrays_for_job(job)
        for repeat in range(10):
            for method in np.roll(np.asarray(methods),(number+repeat)%len(methods)):
                if method in VARIANTS:
                    fit=candidate_solve(values,str(method))
                    elapsed,objective=fit["elapsed_s"],fit["objective"]
                else:
                    fit=solve_window(*values,method=str(method))
                    elapsed,objective=fit.elapsed_s,fit.robust_objective
                rows.append(dict(sequence=job[1],endpoint_s=job[2],duration_s=job[3],layout=job[4],
                    frame_deg=job[5],repeat=repeat,method=str(method),elapsed_ms=elapsed*1000,objective=objective))
    frame=pd.DataFrame(rows)
    frame.to_csv(output/"sequential_timing.csv",index=False)
    summary=frame.groupby("method")["elapsed_ms"].agg(median="median",p95=lambda x:x.quantile(.95),maximum="max")
    summary.to_csv(output/"sequential_timing_summary.csv")
    print(summary.to_string())


def main():
    """Freeze extension policies before running the complete original window matrix."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache",type=Path,default=PROJECT/"results/recovery_bootstrap_20261107_final/cache")
    parser.add_argument("--base",type=Path,default=PROJECT/"results/recovery_window_solver_20261107")
    parser.add_argument("--output",type=Path,default=PROJECT/"results/recovery_window_candidates_20261107")
    parser.add_argument("--workers",type=int,default=8)
    parser.add_argument("--timing-only",action="store_true")
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    if args.timing_only:
        sequential_timing(args.cache,args.output)
        return
    base=pd.read_csv(args.base/"metrics.csv")
    keys=["sequence","endpoint_requested_s","duration_s","layout","frame_deg"]
    jobs=[(str(args.cache),*row,index) for index,row in enumerate(base[keys].drop_duplicates().itertuples(index=False,name=None))]
    write_json(args.output/"protocol.json",dict(variants=VARIANTS,windows=len(jobs),workers=args.workers,
        candidate_definition="All distinct quartic candidates returned by lifted_candidate, including auxiliary 0/pi; profile_two takes two smallest relaxed costs",
        duplicate_angle_tolerance_rad=ANGLE_DUPLICATE_TOL_RAD,main_relative_cost_tolerance=RELATIVE_COST_TOLERANCE,
        timing_panel="Each of six logs: (end10,duration2,all8,yaw0) and (end30,duration8,lower4,yaw180);10 interleaved repetitions in one process",
        developmental_selection=True,independent_blind_data=False,cv_cost_included=True,
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                       (Path(__file__),Path(__file__).parent/"initializer.py",Path(__file__).parent/"window_solver_study.py")}))
    rows=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(evaluate,job) for job in jobs]
        for number,future in enumerate(as_completed(futures),1):
            rows.extend(future.result())
            if number%60==0:
                print(f"DONE {number}/{len(jobs)}",flush=True)
    new=pd.DataFrame(rows)
    new.to_csv(args.output/"candidate_metrics.csv",index=False)
    aggregate(base,new,args.output)


if __name__=="__main__":
    main()
