"""Retain every early-TDoA case and compare common fixed scoring intervals."""

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import run

# All figures retain a stable policy order and explicit baseline labels.
COLORS={"native":"#56758c","fixed_2s_pv":"#b47a25","early_xy":"#16855b","early_switch":"#953e70"}


def main():
    """Save all-case paired tables, geometry strata, trigger audits and plots."""
    output=run.HERE/"results"
    records=json.loads((output/"records.json").read_text(encoding="utf8"))
    metrics=pd.read_csv(output/"metrics.csv")
    metrics["geometry"]=metrics.name.str.extract(r"const(\d+)").astype(int)
    metrics["protocol"]=metrics.name.str.extract(r"tdoa(\d+)").astype(int)
    keys=["name","yaw_offset_deg"]
    assert len(records)==16 and len(metrics)==64 and not metrics.duplicated(keys+["method"]).any()
    paired=metrics.copy()
    for baseline in ("fixed_2s_pv","native"):
        base=metrics[metrics.method==baseline].set_index(keys)
        for label in ("early_xy_rmse","future_xy_rmse","early_xy_p95","future_xy_p95"):
            paired[label+"_minus_"+baseline]=[row[label]-base.loc[tuple(row[k] for k in keys),label]
                for _,row in paired.iterrows()]
    paired.to_csv(output/"paired_metrics.csv",index=False)
    aggregate=metrics.groupby(["geometry","method"]).agg(cases=("name","size"),
        early_xy_median=("early_xy_rmse","median"),early_xy_worst=("early_xy_rmse","max"),
        early_p95_median=("early_xy_p95","median"),future_xy_median=("future_xy_rmse","median"),
        future_xy_worst=("future_xy_rmse","max"),future_p95_median=("future_xy_p95","median"),
        failures=("failed","sum"),early_coverage_min=("early_coverage","min"),
        future_coverage_min=("future_coverage","min"))
    aggregate.to_csv(output/"geometry_summary.csv")
    metrics.groupby("method").agg(cases=("name","size"),
        early_xy_median=("early_xy_rmse","median"),early_xy_worst=("early_xy_rmse","max"),
        early_p95_median=("early_xy_p95","median"),future_xy_median=("future_xy_rmse","median"),
        future_xy_worst=("future_xy_rmse","max"),future_p95_median=("future_xy_p95","median"),
        failures=("failed","sum")).to_csv(output/"summary.csv")
    triggers=[]
    attempts=[]
    preservation={}
    clocks=[]
    prefix_error=0.
    for record in records:
        identity={key:record[key] for key in keys}
        report=record["attempts"][-1]
        triggers.append(dict(identity,trigger_s=record["trigger_s"],checks=len(record["attempts"]),
            reason=report["reason"],radius_95_m=report.get("radius_95_m"),
            normalized_rms=report.get("normalized_rms"),speed_mps=report.get("speed_mps")))
        attempts.extend(dict(identity,**item) for item in record["attempts"])
        for key,value in record["preservation"].items():
            preservation[key]=max(preservation.get(key,0.),value)
        arrays=np.load(record["artifact"])
        reference=arrays["native"]
        for method in run.METHODS:
            actual=arrays[method]
            if not np.array_equal(reference[:,0],actual[:,0]):
                clocks.append(dict(identity,method=method))
            mask=reference[:,0]<record["origin"]+10.
            prefix_error=max(prefix_error,float(np.max(np.abs(reference[mask]-actual[mask]))))
    pd.DataFrame(triggers).to_csv(output/"triggers.csv",index=False)
    pd.DataFrame(attempts).to_csv(output/"attempts.csv",index=False)
    early=paired[paired.method=="early_xy"]
    comparisons={}
    for label in ("early","future"):
        for baseline in ("fixed_2s_pv","native"):
            delta=early[label+"_xy_rmse_minus_"+baseline]
            comparisons[label+"_vs_"+baseline]=dict(wins=int(np.sum(delta<-1e-10)),
                losses=int(np.sum(delta>1e-10)),ties=int(np.sum(np.abs(delta)<=1e-10)))
    protocol=json.loads((output/"protocol.json").read_text(encoding="utf8"))
    checks=dict(cases=len(records),traces=len(metrics),failures=int(metrics.failed.sum()),
        trigger_before_22=sum(item["trigger_s"]<22. for item in records),
        triggered_total=sum(item["trigger_s"] is not None for item in records),
        maximum_handoff_preservation_errors=preservation,common_prefix_max_error=prefix_error,
        output_clock_mismatches=clocks,minimum_early_coverage=float(metrics.early_coverage.min()),
        minimum_future_coverage=float(metrics.future_coverage.min()),paired_comparisons=comparisons,
        source_hash_matches={path:hashlib.sha256(Path(path).read_bytes()).hexdigest()==value
            for path,value in protocol["source_sha256"].items()})
    run.save_json(output/"validation.json",checks)
    assert prefix_error==0. and not clocks
    assert max(preservation.values())<1e-8
    # The same sixteen cases appear in both physical-time panels.
    figure,axes=plt.subplots(1,2,figsize=(12,5.5))
    fixed=metrics[metrics.method=="fixed_2s_pv"].set_index(keys)
    for axis,label,title in zip(axes,("early_xy_rmse","future_xy_rmse"),("20–22 s","26–31 s")):
        for method in ("native","early_xy","early_switch"):
            subset=metrics[metrics.method==method]
            x=[fixed.loc[tuple(row[k] for k in keys),label] for _,row in subset.iterrows()]
            axis.scatter(x,subset[label],label=method,color=COLORS[method],s=30,alpha=.8)
        bounds=np.r_[metrics[metrics.method=="fixed_2s_pv"][label].to_numpy(),metrics[label].to_numpy()]
        low=max(np.min(bounds)*.7,1e-4)
        high=np.max(bounds)*1.4
        axis.plot([low,high],[low,high],"k--",lw=.8)
        axis.set(xscale="log",yscale="log",xlim=(low,high),ylim=(low,high),
            xlabel="Fixed 2 s PV XY RMSE (m)",ylabel="Alternative XY RMSE (m)",title=title)
        axis.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    figure.suptitle("Frozen Delft XY requirements on unseen UTIL trial 1: all 16 cases")
    figure.tight_layout()
    figure.savefig(output/"paired_position.png",dpi=160)
    plt.close(figure)
    # Geometry/protocol/yaw order avoids selecting favorable trajectories for display.
    figure,axes=plt.subplots(4,4,figsize=(15,10),sharex=True)
    for axis,record in zip(axes.ravel(),sorted(records,key=lambda item:(item["name"],item["yaw_offset_deg"]))):
        arrays=np.load(record["artifact"])
        for method in run.METHODS:
            rows=arrays[method]
            selected=(rows[:,0]>=record["origin"]+19.5)&(rows[:,0]<=record["origin"]+24.)
            trace=rows[selected]
            axis.plot(trace[:,0]-record["origin"],trace[:,1],color=COLORS[method],label=method,lw=1)
        axis.axvline(record["trigger_s"],color="black",ls=":",lw=.8)
        axis.set_title(record["name"].replace("-trial1","")+f" / {record['yaw_offset_deg']:g}°",fontsize=8)
        axis.grid(alpha=.2)
    axes[0,0].legend(fontsize=6)
    figure.supxlabel("Physical time from sensor origin (s)")
    figure.supylabel("Estimated world X (m), no truth used for plot selection")
    figure.tight_layout()
    figure.savefig(output/"all_recovery_outputs.png",dpi=140)
    plt.close(figure)
    print(aggregate.to_string())
    print(json.dumps(checks,indent=2))


if __name__=="__main__":
    main()
