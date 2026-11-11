"""Summarize every paired QDrone case without changing the frozen estimators.

Incomplete truth support stays visible; paired primary summaries use only common
complete windows. Geometry containment is a post-run diagnostic, never tuning.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from qdrone import truth_at
from qdrone_run import load_record, WINDOWS
from replay import write_json

# Comparison tolerances distinguish floating point equality from real changes.
TIE_M = 1e-10
# The primary protocol fixes both radio sigma and the engineering lever radius.
SIGMA_M, LEVER_M = .3, 1.


def summarize(output):
    """Write per-case paired changes, all-site summaries, and post-hoc diagnostics."""
    protocol=json.loads((output/"protocol.json").read_text(encoding="utf-8"))
    frame=pd.read_csv(output/"metrics.csv")
    index=["site","sequence","length_s","window"]
    baseline=frame[frame.method=="K"].set_index(index)
    pairs=[]
    for method in ("proposal","always_C","radio_bounds"):
        selected=frame[frame.method==method].set_index(index)
        joined=selected.join(baseline,rsuffix="_K")
        for key,row in joined.iterrows():
            pairs.append(dict(zip(index,key),method=method,
                complete_pair=bool(row.complete_window and row.complete_window_K),
                truth_coverage=row.truth_coverage,K_xy_m=row.xy_rmse_m_K,
                method_xy_m=row.xy_rmse_m,delta_xy_m=row.xy_rmse_m-row.xy_rmse_m_K,
                delta_z_m=row.z_rmse_m-row.z_rmse_m_K))
    paired=pd.DataFrame(pairs)
    paired.to_csv(output/"paired.csv",index=False)
    complete=paired[paired.complete_pair]
    groups=[]
    for keys,group in complete.groupby(["length_s","window","method"]):
        delta=group.delta_xy_m
        groups.append(dict(length_s=keys[0],window=keys[1],method=keys[2],recordings=len(group),
            K_xy_median_m=float(group.K_xy_m.median()),method_xy_median_m=float(group.method_xy_m.median()),
            median_paired_change_m=float(delta.median()),worst_deterioration_m=float(delta.max()),
            best_change_m=float(delta.min()),wins=int((delta < -TIE_M).sum()),
            ties=int((delta.abs()<=TIE_M).sum()),losses=int((delta>TIE_M).sum())))
    pd.DataFrame(groups).to_csv(output/"summary.csv",index=False)
    full=frame[frame.complete_window]
    full.groupby(["site","length_s","window","method"]).agg(
        recordings=("sequence","count"),xy_median_m=("xy_rmse_m","median"),
        z_median_m=("z_rmse_m","median"),worst_xy_m=("xy_rmse_m","max")
        ).to_csv(output/"site_summary.csv")
    jumps,motion,containment,audits=[],[],[],[]
    for entry in protocol["cohort"]:
        if entry["status"]!="sensor_eligible":
            continue
        identifier=entry["site"]+"__"+entry["name"]
        record=load_record(output/"cache"/(identifier+".npz"))
        truth=np.load(output/"cache"/(identifier+"_scoring.npz"))["truth"]
        resume=float(entry.get("resume_s",25.))
        for label,begin,end in WINDOWS:
            clock=resume+np.arange(begin,end,.05)
            values,support=truth_at(clock+record.origin_s,truth)
            xyz=values[support]
            motion.append(dict(site=entry["site"],sequence=entry["name"],window=label,
                resume_s=resume,truth_coverage=float(support.mean()),
                horizontal_bounding_box_diagonal_m=float(np.linalg.norm(np.ptp(xyz[:,:2],axis=0))) if len(xyz) else np.nan,
                vertical_span_m=float(np.ptp(xyz[:,2])) if len(xyz) else np.nan))
        packets=(record.radio_time>=resume)&(record.radio_time<resume+2)
        ground,support=truth_at(record.radio_time[packets]+record.origin_s,truth)
        anchors=record.anchors[record.radio_id[packets]]
        radius=record.radio_range[packets]+3*SIGMA_M+LEVER_M
        excess=np.linalg.norm(ground[:,:2]-anchors[:,:2],axis=1)-radius
        containment.append(dict(site=entry["site"],sequence=entry["name"],resume_s=resume,
            received_packets=int(packets.sum()),supported_packets=int(support.sum()),
            contained_fraction=float(np.mean(excess[support]<=0)) if support.any() else np.nan,
            max_disk_violation_m=float(np.max(excess[support])) if support.any() else np.nan,
            radius_was_fixed_before_errors=True,physical_lever_unverified=True))
        for length in protocol["lengths_s"]:
            case=identifier+"__L"+str(int(length))
            audit=json.loads((output/(case+".json")).read_text(encoding="utf-8"))
            data=np.load(output/(case+".npz"))
            audits.append(dict(case=case,failure=audit["failure"],display_failure=audit.get("display_failure"),
                elapsed_s=audit["elapsed_s"],**{key:value for key,value in audit.get("display_diagnostics",{}).items()
                if isinstance(value,bool)}))
            time=data["time"]
            for method in ("K","proposal","always_C","radio_bounds"):
                pos=data[method]
                step=np.r_[0.,np.linalg.norm(np.diff(pos[:,:2],axis=0),axis=1)]
                offset=pos[:,:2]-data["K"][:,:2]
                offset_step=np.r_[0.,np.linalg.norm(np.diff(offset,axis=0),axis=1)]
                after=(time>=resume)&(time<resume+11)
                boundary=int(np.searchsorted(time,resume+2))
                jumps.append(dict(site=entry["site"],sequence=entry["name"],length_s=length,method=method,
                    max_post_xy_step_m=float(np.nanmax(step[after])),
                    max_added_offset_jump_m=float(np.nanmax(offset_step[after])),
                    return_boundary_s=float(time[boundary]),return_boundary_xy_step_m=float(step[boundary]),
                    return_boundary_added_offset_jump_m=float(offset_step[boundary])))
    pd.DataFrame(jumps).to_csv(output/"jumps.csv",index=False)
    pd.DataFrame(motion).to_csv(output/"truth_motion_coverage.csv",index=False)
    pd.DataFrame(containment).to_csv(output/"fixed_disk_containment_audit.csv",index=False)
    write_json(output/"validation.json",dict(cases=audits,
        expected_cases=40,actual_cases=len(audits),metric_rows=len(frame),
        native_failures=sum(a["failure"] is not None for a in audits),
        display_failures=sum(a["display_failure"] is not None for a in audits),
        incomplete_rows=int((~frame.complete_window).sum()),
        unknown_lever_approximation_remains=True,posthoc_containment_does_not_tune_parameters=True))
    plot_cases(output,protocol)


def plot_cases(output,protocol):
    """Show all ten eligible recordings at the predeclared longest outage."""
    eligible=[entry for entry in protocol["cohort"] if entry["status"]=="sensor_eligible"]
    figure,axes=plt.subplots(2,5,figsize=(17,6.6),sharex=True)
    for axis,entry in zip(axes.flat,eligible):
        identifier=entry["site"]+"__"+entry["name"]+"__L20"
        data=np.load(output/(identifier+".npz"))
        time=data["score_time"]-float(entry.get("resume_s",25.))
        truth=data["score_truth"]
        support=data["score_truth_supported"]
        indices=np.clip(np.searchsorted(data["time"],data["score_time"],side="right")-1,0,len(data["time"])-1)
        for method,label in (("K","Full original K"),("proposal","Selected C"),("always_C","Always C"),("radio_bounds","Radio bounds")):
            error=np.linalg.norm(data[method][indices,:2]-truth[:,:2],axis=1)
            error[~support]=np.nan
            axis.plot(time,error,label=label,lw=1.1)
        axis.axvline(2,color=".6",ls="--",lw=.8)
        short=entry["name"].replace("raw_data","").split("_site")[0]
        if "Flight" in short: short=short.split("_")[-1]
        if "exp" in short: short="exp"+short.split("exp")[-1].split("_")[0]
        axis.set_title(entry["site"]+" / "+short,fontsize=8)
        axis.set_yscale("symlog",linthresh=.1)
        axis.set_ylim(bottom=0.)
        axis.grid(alpha=.2)
        if not support.any(): axis.text(.5,.5,"No supplied GT in window",ha="center",transform=axis.transAxes,fontsize=8)
    axes[0,0].legend(fontsize=7)
    for axis in axes[-1]: axis.set_xlabel("Seconds after recovery")
    for axis in axes[:,0]: axis.set_ylabel("XY error (m; symlog)")
    figure.suptitle("QDrone: every sensor-eligible recording, 20 s outage; "+protocol.get("clock_mode","primary"))
    figure.tight_layout()
    figure.savefig(output/"all_recordings_L20.png",dpi=170)
    plt.close(figure)


def main():
    """Accept an explicit completed output directory, never the invalidated first batch."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    if (args.output/"INVALIDATED.json").exists():
        raise ValueError("This anchor-input batch is explicitly invalidated")
    summarize(args.output)


if __name__=="__main__":
    main()
