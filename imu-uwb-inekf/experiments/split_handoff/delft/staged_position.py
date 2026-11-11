"""Add causal early XY followed by disjoint-radio PV, preserving live attitude.

The original early-position trigger and tables remain frozen. This supplement
replays physical covariance and never reuses early or live packets in the PV fit.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

import study as shared
from early_position import handoff_xy

# Supplement outputs are separate from both frozen matrices.
HERE=Path(__file__).resolve().parent
OUTPUT=HERE/"early_results"
METHODS=("staged_xy_pv","staged_live_display")


def score(stamps, prediction, truth, identity, method):
    """Use the same fixed early/future and complete-post intervals as the early matrix."""
    error=prediction[:,:3]-truth
    xy=np.linalg.norm(error[:,:2],axis=1)
    row=dict(identity,method=method,finite=bool(np.all(np.isfinite(prediction[stamps>=2.1]))))
    for label,mask in dict(early=(stamps>=25)&(stamps<27),fixed=(stamps>=31)&(stamps<36),
                           post=stamps>=25,gap=(stamps>=15)&(stamps<25)).items():
        row[label+"_xy_rmse_m"]=float(np.sqrt(np.mean(xy[mask]**2)))
        row[label+"_xy_peak_m"]=float(np.max(xy[mask]))
        row[label+"_z_rmse_m"]=float(np.sqrt(np.mean(error[mask,2]**2)))
    return row


def run_case(file):
    """Reserve even post-trigger packets for one PV overwrite and keep actual live R/bias."""
    file=Path(file)
    meta=json.loads(file.read_text(encoding="utf-8"))
    name,layout,yaw=meta["sequence"],meta["layout"],meta["yaw_offset_deg"]
    original=np.load(file.with_suffix(".npz"))
    cache=HERE/"results/cache"
    data=dict(np.load(cache/(Path(name).stem+".npz")))
    model=json.loads((cache/"protocol.json").read_text(encoding="utf-8"))["model"]
    stamps=data["time"]
    trigger=meta["trigger_time_s"]
    assert trigger is not None and trigger<27
    k15,k25,k27,kt=[int(np.searchsorted(stamps,value)) for value in (15.,25.,27.,trigger)]
    active=np.ones(len(data["anchors"]),dtype=bool)
    if layout=="lower4":
        active[:]=False; active[np.argsort(data["anchors"][:,2])[:4]]=True
    positive=data["changed"]&(data["ranges"]>0)
    after_trigger=positive.copy()
    after_trigger[:kt+1]=False
    # A predetermined runtime counter starts at the first genuinely new post-trigger packet.
    parity=np.cumsum(after_trigger,axis=0)%2
    reserved=data.copy()
    reserved["changed"]=after_trigger&(parity==0)
    estimate_history=np.full((len(stamps),6),np.nan)
    rotations=np.full((len(stamps),3,3),np.nan)
    biases=np.full((len(stamps),6),np.nan)
    core=None; first_checks={}; pv_checks={}; counts=dict(live_window=0,pv_window=0)
    accepted=[trial for trial in meta["attempts"] if trial["accepted"]]
    assert len(accepted)==1
    candidate=(np.asarray(accepted[0]["position_xy"]),np.asarray(accepted[0]["horizontal_covariance"]))
    begun=time.perf_counter()
    trigger_replay_error=0.
    for k,stamp in enumerate(stamps):
        if core is not None and k:
            core.predict(data["imu"][k-1],float(stamp-stamps[k-1]))
        startup=False
        if core is None and stamp>=2:
            estimate=shared.solve_window(*shared.window_data(data,0,k,model),method="profile")
            core=shared.tracker_from_estimate(estimate,data["rotations"][k],data["bias"],model)
            startup=True
        if core is None:
            continue
        if k==k15:
            physical=core.physical_covariance()
            core.X[:3,:3]=shared.yaw_rotation(np.deg2rad(yaw))@core.X[:3,:3]
            chart=shared.coordinate_map(core.X)
            core.P=shared.symmetrize(chart@physical@chart.T)
        if k==kt:
            core,first_checks=handoff_xy(core,candidate,model)
            trigger_replay_error=max(float(np.max(np.abs(np.r_[core.X[:3,4],core.X[:3,3]]-original["early_only_live"][k]))),
                float(np.max(np.abs(core.X[:3,:3]-original["early_only_live_R"][k]))),
                float(np.max(np.abs(core.bias-original["early_only_live_bias"][k]))))
        if not startup and (k<k15 or k>kt):
            for anchor in np.flatnonzero(positive[k]&active):
                if kt<k<=k27 and parity[k,anchor]==0:
                    counts["pv_window"]+=1
                    continue
                core.correct_twr(data["anchors"][anchor],data["ranges"][k,anchor],int(anchor))
                if kt<k<=k27:
                    counts["live_window"]+=1
        if k==k27:
            old_rotation,old_bias=core.X[:3,:3].copy(),core.bias.copy()
            before=core.physical_covariance()
            retained=np.r_[0:3,9:15]
            core,pv=shared.position_handoff(core,reserved,kt+1,k27,active,model)
            pv_checks=dict(rotation=float(np.max(np.abs(core.X[:3,:3]-old_rotation))),
                bias=float(np.max(np.abs(core.bias-old_bias))),
                retained_covariance=float(np.max(np.abs(core.physical_covariance()[np.ix_(retained,retained)]-
                                                        before[np.ix_(retained,retained)]))))
        estimate_history[k]=np.r_[core.X[:3,4],core.X[:3,3]]
        rotations[k],biases[k]=core.X[:3,:3],core.bias
    assert trigger_replay_error<1e-8
    assert all(value<1e-8 for value in pv_checks.values())
    # This selection uses only the already-frozen trigger, not the scoring positions.
    use_staged=stamps>=trigger
    display=original["native_gate"].copy()
    display[use_staged]=estimate_history[use_staged]
    display_R=original["native_gate_R"].copy(); display_R[use_staged]=rotations[use_staged]
    display_bias=original["native_gate_bias"].copy(); display_bias[use_staged]=biases[use_staged]
    payload=dict(time=stamps,staged_xy_pv=estimate_history,staged_xy_pv_R=rotations,staged_xy_pv_bias=biases,
        staged_live_display=display,staged_live_display_R=display_R,staged_live_display_bias=display_bias,
        trigger_time_s=trigger,pv_rows=shared.range_events(reserved,kt+1,k27,active)[0],
        pv_anchors=shared.range_events(reserved,kt+1,k27,active)[1])
    np.savez_compressed(OUTPUT/"staged_histories"/(file.stem+".npz"),**payload)
    shared.write_json(OUTPUT/"staged_histories"/file.name,dict(sequence=name,layout=layout,yaw_offset_deg=yaw,
        trigger_time_s=trigger,trigger_replay_error=trigger_replay_error,first_handoff_preservation=first_checks,
        pv_handoff_preservation=pv_checks,pv=pv,counts=counts,elapsed_s=time.perf_counter()-begun))
    # Evaluation positions enter only after the actual staged trajectory and selector are complete.
    truth=original["truth"]
    identity=dict(sequence=name,layout=layout,yaw_offset_deg=yaw,trigger_time_s=trigger)
    return [score(stamps,estimate_history,truth,identity,METHODS[0]),score(stamps,display,truth,identity,METHODS[1])]


def summarize():
    """Compare all staged outputs with the frozen early/native/fixed controls without overwriting them."""
    staged=pd.read_csv(OUTPUT/"staged_metrics.csv")
    combined=pd.concat([pd.read_csv(OUTPUT/"combined_paired_metrics.csv"),staged],ignore_index=True)
    keys=["sequence","layout","yaw_offset_deg"]
    for baseline in ("fixed_2s_pv","native_gate","early_only_live"):
        base=combined[combined.method==baseline].set_index(keys)
        for label in ("early_xy_rmse_m","early_xy_peak_m","fixed_xy_rmse_m","post_xy_rmse_m"):
            combined[label+"_change_vs_"+baseline]=[r[label]-base.loc[tuple(r[key] for key in keys),label] for _,r in combined.iterrows()]
    combined.to_csv(OUTPUT/"staged_combined_paired.csv",index=False)
    combined.groupby(["layout","method"]).agg(cases=("sequence","size"),
        early_xy_median_m=("early_xy_rmse_m","median"),early_xy_worst_m=("early_xy_rmse_m","max"),
        early_peak_worst_m=("early_xy_peak_m","max"),future_xy_median_m=("fixed_xy_rmse_m","median"),
        future_xy_worst_m=("fixed_xy_rmse_m","max"),post_xy_median_m=("post_xy_rmse_m","median"),
        post_xy_worst_m=("post_xy_rmse_m","max"),
        future_paired_vs_early_m=("fixed_xy_rmse_m_change_vs_early_only_live","median")).to_csv(OUTPUT/"staged_aggregate.csv")
    maxima={}; counts=[]
    for file in sorted((OUTPUT/"staged_histories").glob("*.json")):
        meta=json.loads(file.read_text(encoding="utf-8"))
        maxima["trigger_replay"]=max(maxima.get("trigger_replay",0),meta["trigger_replay_error"])
        for name,value in meta["pv_handoff_preservation"].items():
            maxima[name]=max(maxima.get(name,0),value)
        data=np.load(file.with_suffix(".npz"))
        assert np.all(data["time"][data["pv_rows"]]>meta["trigger_time_s"])
        assert len(data["pv_rows"])==meta["counts"]["pv_window"]
        counts.append(dict(sequence=meta["sequence"],layout=meta["layout"],yaw_offset_deg=meta["yaw_offset_deg"],**meta["counts"]))
    assert len(staged)==96 and staged.finite.all() and all(value<1e-8 for value in maxima.values())
    pd.DataFrame(counts).to_csv(OUTPUT/"staged_counts.csv",index=False)
    shared.write_json(OUTPUT/"staged_validation.json",dict(cases=len(counts),trajectories=len(staged),
        all_finite=True,maxima=maxima,radio_groups_disjoint=True,evaluation_yaw_truth_used=False,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))


def main():
    """Freeze the supplement and run every original condition with the stored trigger."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers",type=int,default=6)
    args=parser.parse_args()
    (OUTPUT/"staged_histories").mkdir(exist_ok=True)
    shared.write_json(OUTPUT/"staged_protocol.json",dict(trigger="Frozen early-position accepted timestamp",
        first_window="Restored radio through trigger: only early XY",
        split="Post-trigger received per-anchor count starts0: odd live, even reserved CV; through27s inclusive",
        pv_handoff="At27s once from reserved raw radio only; preserve actual live R/bias and retained covariance block",
        after27="All later native scalar updates; no heading or old-branch state replacement",
        display="Native until frozen trigger, then actual staged state; no posterior fusion",
        thresholds_changed=False,evaluation_truth_in_estimator=False,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    files=sorted((OUTPUT/"histories").glob("*.json")); assert len(files)==48
    rows=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending={pool.submit(run_case,str(file)):file for file in files}
        for number,future in enumerate(as_completed(pending),1):
            rows.extend(future.result())
            pd.DataFrame(rows).to_csv(OUTPUT/"staged_metrics.csv",index=False)
            print(f"STAGED {number}/48 {pending[future].stem}",flush=True)
    summarize()


if __name__=="__main__":
    main()
