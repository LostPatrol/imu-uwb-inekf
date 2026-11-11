"""Compose a causal native-to-early display without changing any frozen estimate."""

from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# This is an output selector between independently run estimators, not a second update.
HERE = Path(__file__).resolve().parent
DISPLAY = "early_live_display"
KEYS = ["sequence","layout","yaw_offset_deg"]
COLORS = {"fixed_2s_pv":"#000000","native_gate":"#0072b2",
          "early_only_live":"#009e73","early_live_display":"#e69f00","continuous":"#cc79a7"}


def analyze():
    """Select by stored trigger time, score all cases, and retain separate derived histories."""
    destination=HERE/"display_histories"
    destination.mkdir(exist_ok=True)
    rows=[]; events=[]; prefix_max=0.; prefixes={}
    for file in sorted((HERE/"histories").glob("*.json")):
        meta=json.loads(file.read_text(encoding="utf-8"))
        data=np.load(file.with_suffix(".npz"))
        stamps=data["time"]
        trigger=meta["trigger_time_s"]
        use_early=np.zeros(len(stamps),dtype=bool) if trigger is None else stamps>=trigger
        payload=dict(time=stamps,uses_early=use_early)
        for suffix in ("","_R","_bias"):
            selected=data["native_gate"+suffix].copy()
            selected[use_early]=data["early_only_live"+suffix][use_early]
            payload[DISPLAY+suffix]=selected
            assert np.array_equal(selected[use_early],data["early_only_live"+suffix][use_early])
            # Before startup all methods deliberately have missing values.
            prior=(~use_early)&(stamps>=2.1)
            assert np.array_equal(selected[prior],data["native_gate"+suffix][prior])
        np.savez_compressed(destination/(file.stem+".npz"),**payload)
        prefix=(stamps>=2.1)&(stamps<15)
        prefix_max=max(prefix_max,float(np.max(np.abs(payload[DISPLAY][prefix]-data["fixed_2s_pv"][prefix]))))
        geometry=(meta["sequence"],meta["layout"])
        if geometry in prefixes:
            assert np.array_equal(prefixes[geometry],data["fixed_2s_pv"][prefix])
        else:
            prefixes[geometry]=data["fixed_2s_pv"][prefix].copy()
        error=payload[DISPLAY][:,:3]-data["truth"]
        xy=np.linalg.norm(error[:,:2],axis=1)
        record={key:meta[key] for key in KEYS}
        record.update(method=DISPLAY,trigger_time_s=trigger,trigger_delay_s=meta["trigger_delay_s"],
            triggered=trigger is not None,finite=bool(np.all(np.isfinite(payload[DISPLAY][stamps>=2.1]))))
        for label,mask in dict(early=(stamps>=25)&(stamps<27),fixed=(stamps>=31)&(stamps<36),
                               post=stamps>=25,gap=(stamps>=15)&(stamps<25)).items():
            record[label+"_xy_rmse_m"]=float(np.sqrt(np.mean(xy[mask]**2)))
            record[label+"_xy_peak_m"]=float(np.max(xy[mask]))
            record[label+"_z_rmse_m"]=float(np.sqrt(np.mean(error[mask,2]**2)))
        rows.append(record)
        first_packet=meta["attempts"][0]["time_s"] if meta["attempts"] else None
        event={key:meta[key] for key in KEYS}
        event.update(first_packet_time_s=first_packet,trigger_time_s=trigger,
            delay_from25_s=meta["trigger_delay_s"],
            delay_from_first_packet_s=None if trigger is None or first_packet is None else trigger-first_packet)
        if trigger is not None:
            k=int(np.searchsorted(stamps,trigger))
            event["switch_xy_jump_m"]=float(np.linalg.norm(data["early_only_live"][k,:2]-data["native_gate"][k,:2]))
        events.append(event)
    derived=pd.DataFrame(rows)
    derived.to_csv(HERE/"display_metrics.csv",index=False)
    combined=pd.concat([pd.read_csv(HERE/"metrics.csv"),derived],ignore_index=True)
    for baseline in ("fixed_2s_pv","native_gate"):
        indexed=combined[combined.method==baseline].set_index(KEYS)
        for metric in ("early_xy_rmse_m","early_xy_peak_m","fixed_xy_rmse_m","post_xy_rmse_m"):
            combined[metric+"_change_vs_"+baseline]=[
                row[metric]-indexed.loc[tuple(row[key] for key in KEYS),metric] for _,row in combined.iterrows()]
    combined.to_csv(HERE/"combined_paired_metrics.csv",index=False)
    combined.groupby(["layout","method"]).agg(cases=("sequence","size"),
        early_xy_median_m=("early_xy_rmse_m","median"),early_xy_worst_m=("early_xy_rmse_m","max"),
        early_peak_median_m=("early_xy_peak_m","median"),early_peak_worst_m=("early_xy_peak_m","max"),
        future_xy_median_m=("fixed_xy_rmse_m","median"),future_xy_worst_m=("fixed_xy_rmse_m","max"),
        post_xy_median_m=("post_xy_rmse_m","median"),
        early_paired_vs_fixed_m=("early_xy_rmse_m_change_vs_fixed_2s_pv","median"),
        early_paired_vs_native_m=("early_xy_rmse_m_change_vs_native_gate","median"),
        future_paired_vs_native_m=("fixed_xy_rmse_m_change_vs_native_gate","median")).to_csv(HERE/"combined_aggregate.csv")
    pd.DataFrame(events).to_csv(HERE/"packet_trigger_timing.csv",index=False)
    checks=dict(cases=len(rows),all_finite=bool(derived.finite.all()),common_prefix_max=prefix_max,
        selection_uses_truth=False,selection_uses_future_packets=False,
        estimator_states_modified=False,stored_trigger_selection_exact=True,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    checks["all_yaw_offsets_have_identical_pre15_prefix"]=True
    assert checks["all_finite"] and prefix_max==0 and len(rows)==48
    (HERE/"display_validation.json").write_text(json.dumps(checks,indent=2),encoding="utf-8")
    plot_methods=list(COLORS)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    for ax,layout in zip(axes,("all8","lower4")):
        for j,method in enumerate(plot_methods):
            values=combined[(combined.layout==layout)&(combined.method==method)].early_xy_rmse_m.to_numpy()
            ax.scatter(np.full(len(values),j),values,color=COLORS[method],alpha=.55,s=16)
            ax.plot([j-.2,j+.2],[np.median(values)]*2,color=COLORS[method],lw=3)
        ax.set_xticks(range(len(plot_methods)),["Wait 2 s PV","Native gate","Early XY","Live display","No silence"],rotation=25,ha="right")
        ax.set(title=layout,ylabel="25–27 s XY radial RMSE (m)",yscale="log")
        ax.grid(axis="y",alpha=.25)
    fig.savefig(HERE/"early_display_comparison.png",dpi=170); plt.close(fig)
    print((HERE/"combined_aggregate.csv").read_text(encoding="utf-8"))


if __name__=="__main__":
    analyze()
