"""Audit supplementary feedback preservation and plot every triggered condition."""

import json
import feedback_runner as experiment
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    """Check causal equality before reset and report benefits together with damage."""
    output=experiment.OUTPUT
    records=json.loads((output/"records.json").read_text(encoding="utf8"))
    checks=[]
    for record in records:
        K=np.load(output/"K"/(record["key"]+".npz"))["rows"]
        F=np.load(output/"feedback"/(record["key"]+".npz"))["rows"]
        limit=np.inf if record["trigger"] is None else record["trigger"]["output_time"]
        mask=K[:,0]<limit
        checks.append(dict(key=record["key"],trigger_count=record["trigger_count"],
            pretrigger_max_abs=float(np.max(np.abs(K[mask]-F[mask]))),
            nontrigger_max_abs=float(np.max(np.abs(K-F))) if record["trigger"] is None else None,
            mirror_valid=record["mirror_validation"]["validated"]))
    experiment.experiment.save_json(output/"feedback_scores"/"integrity.json",checks)
    frame=pd.read_csv(output/"feedback_scores"/"paired.csv")
    feedback=frame[frame.method=="single_XY_feedback"]
    summary=[]
    for gap,group in feedback.groupby("gap_s"):
        item=dict(gap_s=float(gap),cases=len(group),triggers=int(group.trigger_count.sum()),failures=int(group.failed.sum()),
            maximum_added_xy_step_m=float(group.maximum_added_xy_step_m.max()))
        for window in experiment.experiment.WINDOWS:
            for metric in ("xy_rmse","yaw_rmse_deg"):
                delta=group[window+"_"+metric+"_minus_K"]
                item[window+"_"+metric+"_improved"]=int((delta< -1e-9).sum())
                item[window+"_"+metric+"_worse"]=int((delta>1e-9).sum())
                item[window+"_"+metric+"_worst_delta"]=float(delta.max())
        summary.append(item)
    experiment.experiment.save_json(output/"feedback_scores"/"overview.json",summary)
    triggered=[record for record in records if record["trigger"] is not None]
    figure,axes=plt.subplots(len(triggered),2,figsize=(11,3*len(triggered)),squeeze=False)
    for index,record in enumerate(triggered):
        raw=np.load(output/(record["name"]+"__data.npz"))
        truth=raw["truth"]
        for folder,label,color in (("K","K","black"),("feedback","single XY feedback","tab:blue")):
            rows=np.load(output/folder/(record["key"]+".npz"))["rows"]
            mask=(rows[:,0]>=record["resume"]-.1)&(rows[:,0]<=record["resume"]+11)
            rows=rows[mask]
            reference=np.column_stack([np.interp(rows[:,0],truth[:,0],truth[:,column]) for column in (1,2)])
            axes[index,0].plot(rows[:,0]-record["resume"],np.linalg.norm(rows[:,1:3]-reference,axis=1),color=color,label=label)
            R=rows[:,7:16].reshape(-1,3,3)
            yaw=np.degrees(np.arctan2(R[:,1,0],R[:,0,0]))
            axes[index,1].plot(rows[:,0]-record["resume"],yaw,color=color,label=label)
        axes[index,0].set_ylabel("XY error (m)")
        axes[index,1].set_ylabel("World Euler yaw (deg)")
        for axis in axes[index]:
            axis.axvline(record["trigger"]["relative_s"],color="grey",ls="--")
            axis.set_title(record["key"])
            axis.grid(alpha=.25)
            axis.set_xlabel("Time after radio returns (s)")
    axes[0,0].legend()
    figure.tight_layout()
    figure.savefig(output/"feedback_scores"/"all_triggered.png",dpi=180)
    plt.close(figure)


if __name__=="__main__":
    main()
