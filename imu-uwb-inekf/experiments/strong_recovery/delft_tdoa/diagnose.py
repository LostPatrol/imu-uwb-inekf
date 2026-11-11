"""Score physical radio containment and return-to-K jumps after all decisions.

These truth-based diagnostics never feed calibration, candidates or gates.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import adapter
import run as native

# Match the frozen common radio bound assumptions exactly for diagnostic replay.
HERE=Path(__file__).resolve().parent
SIGMA=np.sqrt(.13)
MARGIN=3*SIGMA
AGE=.25
SPEED=10.


def truth_at(reference,time):
    """Interpolate only the independent scoring reference, never estimator states."""
    return np.column_stack([np.interp(time,reference[:,0],reference[:,k]) for k in (1,2,3)])


def boundary(rows,baseline):
    """Measure the sampled output discontinuity and its added K-relative jump."""
    index=np.searchsorted(rows[:,0],27.)
    if index==0 or index>=len(rows):
        return dict(return_display_step_m=np.nan,return_added_offset_jump_m=np.nan)
    shift=rows[:,1:4]-baseline[:,1:4]
    return dict(return_display_step_m=float(np.linalg.norm(rows[index,1:4]-rows[index-1,1:4])),
                return_added_offset_jump_m=float(np.linalg.norm(shift[index]-shift[index-1])))


def physical(data,reference,rows,active):
    """Audit native XYZ residuals and exact horizontal branch containment."""
    radio=data["radio"]
    radio=radio[(radio[:,0]>=25)&(radio[:,0]<27)]
    truth=truth_at(reference,radio[:,0])
    plus=data["anchors"][radio[:,2].astype(int)]
    minus=data["anchors"][radio[:,1].astype(int)]
    expected=np.linalg.norm(truth-plus,axis=1)-np.linalg.norm(truth-minus,axis=1)
    physical_valid=np.abs(radio[:,3]-expected)<=MARGIN
    prepared=[]
    for j in range(len(radio)):
        y=radio[j,3]
        far,near=(plus[j],minus[j]) if y>=0 else (minus[j],plus[j])
        delta=abs(y)-MARGIN
        if delta<=0 or delta>=np.linalg.norm(far-near):
            continue
        prepared.append((radio[j,0],tuple(radio[j,1:3]),near,far,delta))
    truth=truth_at(reference,rows[:,0])
    cursor=0
    latest={}
    samples=contained=constraints=satisfied=0
    for i,now in enumerate(rows[:,0]):
        if not(25<=now<27):
            continue
        while cursor<len(prepared) and prepared[cursor][0]<=now:
            latest[prepared[cursor][1]]=cursor
            cursor+=1
        if not active[i]:
            continue
        all_inside=True
        applied=0
        for j in sorted(latest.values()):
            stamp,_,near,far,delta=prepared[j]
            age=now-stamp
            if age>AGE:
                continue
            separation=delta-2*SPEED*age
            vertical=abs(far[2]-near[2])
            if separation<=vertical:
                continue
            horizontal=np.sqrt((separation-vertical)*(separation+vertical))
            displacement=truth[i,:2]-near[:2]
            normal=near[:2]-far[:2]
            violation=horizontal*np.linalg.norm(displacement)-normal@displacement-(normal@normal-horizontal**2)/2
            inside=violation<=1e-8
            all_inside &= inside
            applied+=1
            constraints+=1
            satisfied+=int(inside)
        if applied:
            samples+=1
            contained+=int(all_inside)
    grid=np.arange(25,27-1e-9,.01)
    indices=np.searchsorted(rows[:,0],grid,side="right")-1
    return dict(post_radio_packets=len(radio),xyz_native_3sigma_satisfied=int(physical_valid.sum()),
                xyz_native_3sigma_rate=float(physical_valid.mean()) if len(radio) else np.nan,
                bound_active_native_samples=int(active.sum()),bound_active_grid_fraction=float(active[indices].mean()),
                active_samples_with_horizontal_constraints=samples,horizontal_contained_active_samples=contained,
                active_horizontal_containment_rate=contained/samples if samples else np.nan,
                applied_horizontal_constraints=constraints,satisfied_horizontal_constraints=satisfied,
                horizontal_constraint_satisfaction_rate=satisfied/constraints if constraints else np.nan)


def main():
    """Preserve every case, including zero-active and constraint-violation cases."""
    calibration=json.loads((HERE/"calibration.json").read_text(encoding="utf8"))
    records=json.loads((HERE/"paired"/"records.json").read_text(encoding="utf8"))
    physical_rows=[]
    jumps=[]
    cache={}
    for record in records:
        name=record["name"]
        if name not in cache:
            data=adapter.load(name,calibration)
            cache[name]=(data,adapter.truth(name,data["origin"],calibration["lag_s"]))
        data,reference=cache[name]
        artifact=np.load(HERE/"paired"/(record["key"]+".npz"))
        rows=artifact["rows"]
        base_key=record["key"].rsplit("__",1)[0]
        baseline=np.load(HERE/"baseline"/(base_key+".npz"))["rows"]
        jumps.append(dict(key=record["key"],name=name,queue=record["queue"],outage_s=record["outage_s"],
                          policy=record["policy"],**boundary(rows,baseline)))
        if record["policy"]=="bound_xy":
            queued=native.queued(data,record["queue"])
            physical_rows.append(dict(name=name,queue=record["queue"],outage_s=record["outage_s"],
                                      **physical(queued,reference,rows,artifact["use_c"])))
    pd.DataFrame(physical_rows).to_csv(HERE/"paired"/"bound_containment.csv",index=False)
    pd.DataFrame(jumps).to_csv(HERE/"paired"/"return_jumps.csv",index=False)
    frame=pd.DataFrame(physical_rows)
    counts=["post_radio_packets","xyz_native_3sigma_satisfied","bound_active_native_samples",
            "active_samples_with_horizontal_constraints","horizontal_contained_active_samples",
            "applied_horizontal_constraints","satisfied_horizontal_constraints"]
    summary=frame.groupby("queue")[counts].sum()
    summary["mean_case_active_grid_fraction"]=frame.groupby("queue").bound_active_grid_fraction.mean()
    summary["max_case_active_grid_fraction"]=frame.groupby("queue").bound_active_grid_fraction.max()
    summary["pooled_xyz_native_3sigma_rate"]=summary.xyz_native_3sigma_satisfied/summary.post_radio_packets
    summary["pooled_active_horizontal_containment_rate"]=summary.horizontal_contained_active_samples/summary.active_samples_with_horizontal_constraints
    summary["pooled_horizontal_constraint_rate"]=summary.satisfied_horizontal_constraints/summary.applied_horizontal_constraints
    summary.to_csv(HERE/"paired"/"bound_containment_summary.csv")
    print(summary[["pooled_xyz_native_3sigma_rate","pooled_active_horizontal_containment_rate","pooled_horizontal_constraint_rate"]].to_string(),flush=True)


if __name__=="__main__":
    main()
