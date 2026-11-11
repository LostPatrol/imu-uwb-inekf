"""Score a common 20.2–22 s diagnostic interval without changing the experiment."""

import json

import numpy as np
import pandas as pd

import run

# This secondary absolute interval was requested after primary scores; no case is removed.
DIAGNOSTIC_INTERVAL=(20.2,22.)


def main():
    """Keep original complete intervals and report all-case and per-log comparisons."""
    output=run.HERE/"results"
    records=json.loads((output/"records.json").read_text(encoding="utf8"))
    rows=[]
    truths={}
    for record in records:
        name=record["name"]
        if name not in truths:
            truths[name]=run.util.load(name,duration=run.DURATION)["truth"]
        truth=truths[name]
        arrays=np.load(record["artifact"])
        for method in run.METHODS:
            trace=arrays[method]
            keep=(trace[:,0]>=record["origin"]+DIAGNOSTIC_INTERVAL[0])&(
                trace[:,0]<record["origin"]+DIAGNOSTIC_INTERVAL[1])
            trace=trace[keep]
            actual=np.column_stack([np.interp(trace[:,0],truth[:,0],truth[:,k]) for k in (1,2)])
            errors=np.linalg.norm(trace[:,1:3]-actual,axis=1)
            full=arrays[method]
            full=full[(full[:,0]>=record["origin"]+20.)&(full[:,0]<record["origin"]+22.)]
            full_truth=np.column_stack([np.interp(full[:,0],truth[:,0],truth[:,k]) for k in (1,2)])
            energy=np.sum((full[:,1:3]-full_truth)**2,axis=1)
            initial=full[:,0]<record["origin"]+20.2
            rows.append(dict(name=name,yaw_offset_deg=record["yaw_offset_deg"],method=method,
                diagnostic_xy_rmse=float(np.sqrt(np.mean(errors**2))),
                diagnostic_xy_p95=float(np.percentile(errors,95)),diagnostic_xy_peak=float(np.max(errors)),
                diagnostic_samples=len(errors),first_20hz_sample_energy_fraction=float(energy[0]/np.sum(energy)),
                first_point2s_energy_fraction=float(np.sum(energy[initial])/np.sum(energy))))
    frame=pd.read_csv(output/"paired_metrics.csv").merge(pd.DataFrame(rows),on=["name","yaw_offset_deg","method"])
    frame.to_csv(output/"complete_and_diagnostic_metrics.csv",index=False)
    frame[["name","yaw_offset_deg","method","first_20hz_sample_energy_fraction",
        "first_point2s_energy_fraction"]].to_csv(output/"early_error_energy.csv",index=False)
    labels=["early_xy_rmse","early_xy_p95","early_xy_peak","future_xy_rmse","future_xy_p95","future_xy_peak",
            "diagnostic_xy_rmse","diagnostic_xy_p95","diagnostic_xy_peak"]
    frame.groupby(["name","method"])[labels].median().to_csv(output/"per_log_medians.csv")
    frame.groupby("method")[labels].agg(["median","max"]).to_csv(output/"complete_and_diagnostic_summary.csv")
    pairs=frame.pivot(index=["name","yaw_offset_deg"],columns="method",values=labels)
    pairs.columns=[metric+"__"+method for metric,method in pairs.columns]
    for label in ("early","future","diagnostic"):
        for method in ("early_xy","early_switch"):
            delta=pairs[label+"_xy_rmse__"+method]-pairs[label+"_xy_rmse__native"]
            pairs[label+"_"+method+"_vs_native"]=np.where(delta<-1e-10,"win",np.where(delta>1e-10,"lose","tie"))
    pairs.to_csv(output/"native_pairs.csv")
    run.save_json(output/"additional_score_protocol.json",dict(interval=DIAGNOSTIC_INTERVAL,
        timing="Post-primary-score diagnostic requested by parent; not tuned and not a trigger-relative crop",
        cases=len(records),methods=run.METHODS,original_primary_intervals_retained=run.SCORE_WINDOWS))
    print(frame.groupby("method")[labels].agg(["median","max"]).to_string())
    print(pairs[["early_early_xy_vs_native","future_early_xy_vs_native","diagnostic_early_xy_vs_native"]].to_string())
    print(frame.groupby("method")[["first_20hz_sample_energy_fraction","first_point2s_energy_fraction"]].median().to_string())


if __name__=="__main__":
    main()
