"""Score frozen Toronto candidates with an explicit common two-second output deadline.

This prevents a last pre-deadline candidate sample being held past its allowed
phase by the reporting ZOH clock. Source K and raw candidate caches stay intact.
"""

import json
import independent as experiment
import numpy as np
import pandas as pd


def with_deadline(rows,K,deadline):
    """Emit K's causal held value at the exact return-to-K output boundary."""
    index=np.searchsorted(K[:,0],deadline+1e-9,side="right")-1
    boundary=K[index].copy()
    boundary[0]=deadline
    return np.vstack((rows[rows[:,0]<deadline-1e-12],boundary,rows[rows[:,0]>deadline+1e-12]))


def main(expected_cases=32):
    """Keep every paired condition and score only after both candidates were frozen."""
    output=experiment.OUTPUT
    records=json.loads((output/"candidate_records.json").read_text(encoding="utf8"))
    if len(records)!=expected_cases:
        raise RuntimeError(f"All{expected_cases} frozen conditions must finish candidate generation before scoring")
    scoring=output/"scored"
    scoring.mkdir(parents=True,exist_ok=True)
    metrics=[]
    integrity=[]
    for record in records:
        truth=np.load(output/(record["name"]+"__data.npz"))["truth"]
        raw=np.load(output/"candidates"/(record["key"]+".npz"))
        source_K=raw["K"]
        arrays={method:with_deadline(raw[method],source_K,record["resume"]+2.) for method in experiment.METHODS}
        np.savez_compressed(scoring/(record["key"]+".npz"),**arrays)
        K=arrays["K"]
        for method in experiment.METHODS:
            rows=arrays[method]
            outside=(rows[:,0]<record["resume"])|(rows[:,0]>=record["resume"]+2.)
            integrity.append(dict(key=record["key"],method=method,outside_phase_max_abs=float(np.max(np.abs(rows[outside]-K[outside]))),
                rotation_max_abs=float(np.max(np.abs(rows[:,7:16]-K[:,7:16]))),z_max_abs=float(np.max(np.abs(rows[:,3]-K[:,3])))))
            entry={key:record[key] for key in ("name","gap_s","key","failed","failure")}
            entry["method"]=method
            for label,(left,right) in experiment.WINDOWS.items():
                values=experiment.audit.native.ours.util.score_rows(truth,rows,record["resume"]+left,record["resume"]+right-.05)
                entry.update({label+"_"+key:value for key,value in values.items()})
                entry[label+"_coverage"]=values["samples"]/round((right-left)/.05)
            for boundary in (0.,2.):
                index=np.searchsorted(rows[:,0],record["resume"]+boundary)
                label="entry" if boundary==0 else "exit"
                if 0<index<len(rows):
                    entry[label+"_xy_jump_m"]=float(np.linalg.norm(rows[index,1:3]-rows[index-1,1:3]))
                    extra=(rows[index,1:3]-rows[index-1,1:3])-(K[index,1:3]-K[index-1,1:3])
                    entry[label+"_extra_xy_jump_m"]=float(np.linalg.norm(extra))
                    rotations=rows[index-1:index+1,7:16].reshape(-1,3,3)
                    yaw=np.arctan2(rotations[:,1,0],rotations[:,0,0])
                    entry[label+"_yaw_jump_deg"]=float(abs(np.degrees(np.arctan2(np.sin(yaw[1]-yaw[0]),np.cos(yaw[1]-yaw[0])))))
            entry["maximum_xy_change_from_K_m"]=float(np.max(np.linalg.norm(rows[:,1:3]-K[:,1:3],axis=1)))
            offset=rows[:,1:3]-K[:,1:3]
            velocity_offset=rows[:,4:6]-K[:,4:6]
            entry["maximum_added_xy_step_m"]=float(np.max(np.linalg.norm(np.diff(offset,axis=0),axis=1)))
            entry["maximum_added_velocity_step_mps"]=float(np.max(np.linalg.norm(np.diff(velocity_offset,axis=0),axis=1)))
            phase=(rows[1:,0]>=record["resume"])&(rows[1:,0]<=record["resume"]+2.)
            entry["maximum_xy_step_during_recovery_m"]=float(np.max(np.linalg.norm(np.diff(rows[:,1:3],axis=0)[phase],axis=1)))
            if method in ("guarded","radio_bounds"):
                active=raw["use_c" if method=="guarded" else "bounds_active"].astype(bool)
                phase=(source_K[:,0]>=record["resume"])&(source_K[:,0]<record["resume"]+2.)
                entry["active_output_samples"]=int(np.sum(active&phase))
                entry["active_phase_fraction"]=float(np.mean(active[phase]))
                entry["first_active_s"]=float(source_K[active,0][0]-record["resume"]) if np.any(active) else None
            metrics.append(entry)
    frame=pd.DataFrame(metrics)
    baseline=frame[frame.method=="K"].set_index("key")
    for label in experiment.WINDOWS:
        for metric in ("xy_rmse","yaw_rmse_deg"):
            name=label+"_"+metric
            frame[name+"_minus_K"]=[row[name]-baseline.loc[row["key"],name] for row in metrics]
    frame.to_csv(output/"metrics.csv",index=False)
    frame.to_csv(output/"paired.csv",index=False)
    columns=[window+"_"+metric for window in experiment.WINDOWS for metric in ("xy_rmse","yaw_rmse_deg","coverage")]
    frame.groupby(["gap_s","method"])[columns].median().to_csv(output/"summary.csv")
    overview=[]
    for (gap,method),selected in frame.groupby(["gap_s","method"]):
        item=dict(gap_s=float(gap),method=method,cases=len(selected),failures=int(selected.failed.sum()))
        for window in experiment.WINDOWS:
            item[window+"_xy_median_m"]=float(selected[window+"_xy_rmse"].median())
            item[window+"_yaw_median_deg"]=float(selected[window+"_yaw_rmse_deg"].median())
            item[window+"_minimum_coverage"]=float(selected[window+"_coverage"].min())
            delta=selected[window+"_xy_rmse_minus_K"]
            item[window+"_paired_improved"]=int((delta< -1e-9).sum())
            item[window+"_paired_worse"]=int((delta>1e-9).sum())
            item[window+"_worst_delta_m"]=float(delta.max())
        item["exit_extra_xy_jump_max_m"]=float(selected.exit_extra_xy_jump_m.max())
        item["exit_extra_xy_jump_median_m"]=float(selected.exit_extra_xy_jump_m.median())
        item["maximum_added_xy_step_m"]=float(selected.maximum_added_xy_step_m.max())
        item["maximum_added_velocity_step_mps"]=float(selected.maximum_added_velocity_step_mps.max())
        overview.append(item)
    experiment.save_json(output/"overview.json",overview)
    experiment.save_json(output/"integrity.json",integrity)
    experiment.save_json(output/"score_provenance.json",dict(
        source_sha256=experiment.audit.native.ours.digest(experiment.HERE/"score_independent.py"),
        correction="Explicit common deadline at resume+2, emitting strict ZOH K value, to prevent reporting hold leakage",
        algorithms_unchanged=True,raw_candidate_cache_unchanged=True,
        exact_no_future_contract="every method equals K before resume and at/after resume+2; R/yaw/z identical at all times",
        windows_half_open_after_resume_s=experiment.WINDOWS,score_clock_hz=20))
    print(frame.groupby(["gap_s","method"])[columns].median().to_string())


if __name__=="__main__":
    main()
