"""Illustrate one sustained repair and two adverse exploratory feedback cases."""

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Examples are selected after scoring for explanation, never for algorithm selection.
HERE=Path(__file__).resolve().parent
DELFT=HERE.parent
sys.path.insert(0,str(DELFT))
import adapter
EXAMPLES=[("2021-02-12_14_00_34_tdoa_octagon",20,"Sustained repair"),
          ("2021-02-12_14_05_09_tdoa_star",20,"Late deterioration"),
          ("2021-02-12_14_13_51_tdoa_hourglass_BAD",10,"Early gain, late deterioration")]


def sample(rows,grid):
    """Query estimator position by the same causal hold used for all metrics."""
    return rows[np.searchsorted(rows[:,0],grid,side="right")-1,1:4]


def main():
    """Keep chosen successes and failures visible alongside the full-case tables."""
    calibration=json.loads((DELFT/"calibration.json").read_text(encoding="utf8"))
    records=json.loads((HERE/"results"/"records.json").read_text(encoding="utf8"))
    instant=[]
    for record in records:
        if not record["triggered"]:
            continue
        trigger=record["trigger"]
        reference=adapter.truth(record["name"],record["origin"],calibration["lag_s"])
        truth=np.array([np.interp(trigger["time"],reference[:,0],reference[:,k]) for k in (1,2,3)])
        before=np.asarray(trigger["before_position_m"])
        after=np.asarray(trigger["after_position_m"])
        instant.append(dict(name=record["name"],outage_s=record["outage_s"],time=trigger["time"],
                            before_xy_error_m=float(np.linalg.norm(before[:2]-truth[:2])),
                            after_xy_error_m=float(np.linalg.norm(after[:2]-truth[:2]))))
    pd.DataFrame(instant).to_csv(HERE/"results"/"trigger_truth_diagnostics.csv",index=False)
    fig,axes=plt.subplots(3,2,figsize=(10,10))
    grid=np.arange(25,36-1e-9,.01)
    for row,(name,length,title) in enumerate(EXAMPLES):
        key=name+f"__first_per_stamp__L{length}"
        record=next(item for item in records if item["key"]==key)
        base=np.load(DELFT/"baseline"/(key+".npz"))["rows"]
        changed=np.load(HERE/"results"/(key+".npz"))["rows"]
        reference=adapter.truth(name,record["origin"],calibration["lag_s"])
        truth=np.column_stack([np.interp(grid,reference[:,0],reference[:,k]) for k in (1,2,3)])
        k,p=sample(base,grid),sample(changed,grid)
        axis=axes[row,0]
        axis.plot(grid-25,np.linalg.norm(k[:,:2]-truth[:,:2],axis=1),label="Original K")
        axis.plot(grid-25,np.linalg.norm(p[:,:2]-truth[:,:2],axis=1),label="One state projection")
        axis.axvline(record["trigger"]["time"]-25,color="k",ls=":",lw=.9,label="Single feedback time")
        axis.axvline(2,color="gray",ls="--",lw=.7)
        axis.set(xlabel="Time after radio resumption (s)",ylabel="Absolute horizontal error (m)",
                 title=title+f"; L={length} s")
        axis.grid(alpha=.2)
        axis=axes[row,1]
        axis.plot(truth[:,0],truth[:,1],"k",lw=1.8,label="Truth")
        axis.plot(k[:,0],k[:,1],label="Original K")
        axis.plot(p[:,0],p[:,1],label="One state projection")
        axis.set(xlabel="X (m)",ylabel="Y (m)",title=name.split("_tdoa_")[-1]+"; t=25–36 s")
        axis.set_aspect("equal",adjustable="datalim")
        axis.grid(alpha=.2)
    axes[0,0].legend(fontsize=7)
    axes[0,1].legend(fontsize=7)
    fig.suptitle("Post-hoc illustrations: no forced return to K at 27 seconds")
    fig.tight_layout()
    fig.savefig(HERE/"results"/"illustrative_trajectories.png",dpi=150)
    fig.savefig(HERE/"results"/"illustrative_trajectories.svg")
    plt.close(fig)


if __name__=="__main__":
    main()
