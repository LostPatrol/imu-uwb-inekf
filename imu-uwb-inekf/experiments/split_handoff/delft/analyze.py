"""Validate disjoint Delft handoffs and summarize fixed-future radial position errors."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Names and colors are fixed before selecting examples or plotting outcomes.
METHODS = ("position_only","full_attitude","forced_yaw","evidence_yaw","projected_full_yaw")
COLORS = ("#000000","#d55e00","#0072b2","#009e73","#cc79a7")
# These headings denote injected mean rotations, not measured yaw errors.
ANGLES = (0.0,90.0,-90.0,180.0)


def validate(output, frame):
    """Check shared prefixes, covariance semantics, radio disjointness, and preserved tilt."""
    maxima = dict(prefix=0.0,position_at_heading=0.0,bias_at_heading=0.0,
                  world_attitude_covariance=0.0,yaw_only_gravity_direction=0.0)
    case_count = 0
    prefix_by_geometry = {}
    for path in sorted((output/"histories").glob("*.npz")):
        if path.stem.endswith("_evidence"):
            continue
        data = np.load(path)
        stamps = data["time"]
        before = (stamps>=2.1)&(stamps<31.0)
        endpoint = int(np.searchsorted(stamps,31.0))
        base = data["position_only"]
        baseR = data["position_only_R"][endpoint]
        for method in METHODS:
            maxima["prefix"] = max(maxima["prefix"],float(np.max(np.abs(data[method][before]-base[before]))))
            maxima["position_at_heading"] = max(maxima["position_at_heading"],float(np.max(np.abs(data[method][endpoint]-base[endpoint]))))
            maxima["bias_at_heading"] = max(maxima["bias_at_heading"],float(np.max(np.abs(data[method+"_bias"][endpoint]-data["position_only_bias"][endpoint]))))
            covariance = data[method+"_handoff_physical_covariance"]
            rotation = data[method+"_R"][endpoint]
            world = rotation@covariance[:3,:3]@rotation.T
            maxima["world_attitude_covariance"] = max(maxima["world_attitude_covariance"],
                float(np.max(np.abs(world-data["common_world_attitude_covariance"]))))
            if method in ("forced_yaw","evidence_yaw","projected_full_yaw"):
                maxima["yaw_only_gravity_direction"] = max(maxima["yaw_only_gravity_direction"],
                    float(np.max(np.abs(rotation[2,:]-baseR[2,:]))))
        # The controlled 15 s perturbations must have exactly the same pre-blackout position prefix.
        geometry = path.stem.split("__yaw")[0]
        prefix = base[(stamps>=2.1)&(stamps<15)]
        if geometry in prefix_by_geometry:
            assert np.array_equal(prefix,prefix_by_geometry[geometry])
        else:
            prefix_by_geometry[geometry] = prefix
        evidence = np.load(path.with_name(path.stem+"_evidence.npz"))
        meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        sensor = np.load(output/"cache"/(meta["sequence"].removesuffix(".csv")+".npz"))
        assert "truth" not in sensor.files
        valid = sensor["changed"] & (sensor["ranges"]>0)
        valid &= ~((stamps>=15)&(stamps<25))[:,None]
        parity = np.cumsum(valid,axis=0)%2
        rows,anchors = evidence["rows"],evidence["anchors"]
        assert np.all(parity[rows,anchors] == 0)
        assert np.all((stamps[rows]>meta["actual_position_time_s"]) & (stamps[rows]<=meta["actual_attitude_time_s"]))
        case_count += 1
    assert all(value<1e-8 for value in maxima.values())
    assert frame.finite.all()
    checks = dict(case_count=case_count,trajectory_count=len(frame),maxima=maxima,
        all_finite=bool(frame.finite.all()),evaluation_yaw_truth_used=False,
        odd_even_radio_groups_disjoint=True,all_yaw_offsets_share_pre_15s_prefix=True,
        fixed_future_interval_s=[31,36],
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")})
    evidence_source = Path(__file__).parents[1]/"yaw_evidence.py"
    if evidence_source.exists():
        checks["yaw_evidence_sha256"] = hashlib.sha256(evidence_source.read_bytes()).hexdigest()
    (output/"validation.json").write_text(json.dumps(checks,indent=2),encoding="utf-8")
    return checks


def analyze(output):
    """Export paired changes without converting injected rotations into attitude truth."""
    frame = pd.read_csv(output/"metrics.csv")
    checks = validate(output,frame)
    keys = ["sequence","layout","yaw_offset_deg"]
    baseline = frame[frame.method=="position_only"].set_index(keys)["fixed_xy_rmse_m"]
    frame["fixed_xy_change_vs_position_m"] = [r.fixed_xy_rmse_m-baseline.loc[(r.sequence,r.layout,r.yaw_offset_deg)] for r in frame.itertuples()]
    frame.to_csv(output/"paired_metrics.csv",index=False)
    decisions=[]
    for path in sorted((output/"histories").glob("*.json")):
        metadata=json.loads(path.read_text(encoding="utf-8"))
        heading=metadata["heading"]
        decisions.append(dict(sequence=metadata["sequence"],layout=metadata["layout"],
            yaw_offset_deg=metadata["yaw_offset_deg"],**heading["decision"],
            full_success=heading["full_success"],full_nfev=heading["full_nfev"],
            full_tilt_change_deg=heading["full_tilt_change_deg"],full_cost=heading["full_cost"],
            full_seed_costs=heading["full_seed_costs"]))
    decision_frame=pd.DataFrame(decisions)
    decision_frame.to_csv(output/"decisions.csv",index=False)
    decision_frame.groupby(["layout","reason"]).size().to_csv(output/"gate_reasons.csv",header=["cases"])
    frame[(frame.method=="evidence_yaw")&frame.accepted].to_csv(output/"accepted_cases.csv",index=False)
    summary = frame.groupby(["layout","yaw_offset_deg","method"]).agg(
        n=("fixed_xy_rmse_m","size"),xy_median_m=("fixed_xy_rmse_m","median"),
        xy_mean_m=("fixed_xy_rmse_m","mean"),xy_worst_m=("fixed_xy_rmse_m","max"),
        z_median_m=("fixed_z_rmse_m","median"),delta_median_m=("fixed_xy_change_vs_position_m","median"),
        worse=("fixed_xy_change_vs_position_m",lambda v:int(np.sum(v>1e-8))),
        improved=("fixed_xy_change_vs_position_m",lambda v:int(np.sum(v<-1e-8))),
        accepted=("accepted","sum"),sd_median_deg=("sd_deg","median"))
    summary.to_csv(output/"summary.csv")
    frame.pivot(index=keys,columns="method",values="fixed_xy_rmse_m").to_csv(output/"fixed_xy_paired.csv")
    frame.groupby(["layout","method"]).agg(n=("fixed_xy_rmse_m","size"),
        xy_median_m=("fixed_xy_rmse_m","median"),xy_worst_m=("fixed_xy_rmse_m","max"),
        z_median_m=("fixed_z_rmse_m","median"),delta_median_m=("fixed_xy_change_vs_position_m","median"),
        worse=("fixed_xy_change_vs_position_m",lambda v:int(np.sum(v>1e-8))),
        improved=("fixed_xy_change_vs_position_m",lambda v:int(np.sum(v<-1e-8))),accepted=("accepted","sum")
    ).to_csv(output/"aggregate.csv")
    fig,axes=plt.subplots(1,2,figsize=(11,4.8),layout="constrained")
    for axis,layout in zip(axes,("all8","lower4")):
        selected=frame[frame.layout==layout]
        if selected.empty:
            axis.set_visible(False)
            continue
        for method,color in zip(METHODS[1:],COLORS[1:]):
            rows=selected[selected.method==method]
            x=[baseline.loc[(r.sequence,r.layout,r.yaw_offset_deg)] for r in rows.itertuples()]
            axis.scatter(x,rows.fixed_xy_rmse_m,s=25,color=color,alpha=.75,label=method)
        limits=[min(.03,float(selected.fixed_xy_rmse_m.min())*.8),float(selected.fixed_xy_rmse_m.max())*1.25]
        axis.plot(limits,limits,"--",color="#777777",linewidth=.8)
        axis.set_xscale("log");axis.set_yscale("log")
        axis.set_xlim(limits);axis.set_ylim(limits)
        axis.set_title(layout)
        axis.set_xlabel("Position-only handoff: XY radial RMSE (m)")
        axis.set_ylabel("Attitude strategy: XY radial RMSE (m)")
        axis.grid(alpha=.2)
    fig.legend(*axes[0].get_legend_handles_labels(),loc="outside upper center",ncol=4)
    fig.savefig(output/"paired_fixed_xy.png",dpi=170,bbox_inches="tight")
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.6),layout="constrained")
    for axis,layout in zip(axes,("all8","lower4")):
        for method,color in zip(METHODS,COLORS):
            values=[frame[(frame.layout==layout)&(frame.method==method)&(frame.yaw_offset_deg==angle)].fixed_xy_rmse_m.median() for angle in ANGLES]
            axis.plot(np.arange(4),values,"o-",color=color,label=method,linewidth=1.2)
        axis.set_xticks(np.arange(4),["0°","+90°","-90°","180°"])
        axis.set_xlabel("Injected world-z mean rotation at t=15 s")
        axis.set_ylabel("Median fixed-future XY radial RMSE (m)")
        axis.set_title(layout);axis.grid(alpha=.2)
    fig.legend(*axes[0].get_legend_handles_labels(),loc="outside upper center",ncol=3)
    fig.savefig(output/"yaw_offset_summary.png",dpi=170,bbox_inches="tight")
    plt.close(fig)
    print(summary.round(4).to_string())
    print(json.dumps(checks,indent=2))


def main():
    """Analyze the complete predetermined replay matrix."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=Path(__file__).parent/"results")
    analyze(parser.parse_args().output)


if __name__=="__main__":
    main()
