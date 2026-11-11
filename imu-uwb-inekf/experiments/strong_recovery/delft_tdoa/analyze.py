"""Summarize all matched Delft cases and plot absolute native/overlay errors."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# All graphs use the fixed experimental scoring intervals and include BAD cases.
HERE=Path(__file__).resolve().parent
POLICIES=("K","guarded_C","always_C","bound_xy")


def main():
    """Keep native completeness and paired differences visible beside medians."""
    baseline=pd.read_csv(HERE/"baseline"/"metrics.csv")
    result=dict(native_traces=int(len(baseline)/3),native_failures=int(baseline.groupby("key").failed.first().sum()),
                minimum_score_coverage=float(baseline.coverage.min()))
    first=baseline[baseline.queue=="first_per_stamp"]
    fig,ax=plt.subplots(figsize=(7.4,4.2))
    for interval,group in first.groupby("interval",sort=False):
        med=group.groupby("outage_s").xy_rmse.median()
        ax.plot(med.index,med.values,"o-",label=interval+" s after resumption")
    ax.set(xlabel="Withheld native radio duration (s)",ylabel="Median absolute XY RMSE (m)",yscale="log",
           title="Delft: six complete held-out genuine TDoA recordings")
    ax.grid(alpha=.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(HERE/"native_summary.png",dpi=160)
    fig.savefig(HERE/"native_summary.svg")
    plt.close(fig)
    if (HERE/"paired"/"metrics.csv").exists():
        frame=pd.read_csv(HERE/"paired"/"metrics.csv")
        pairs=frame.pivot(index=["name","queue","outage_s","interval"],columns="policy",values="xy_rmse")
        rows=[]
        for (queue,interval),group in pairs.groupby(level=["queue","interval"],sort=False):
            for policy in POLICIES[1:]:
                delta=group[policy]-group.K
                rows.append(dict(queue=queue,interval=interval,policy=policy,cases=len(delta),
                                 wins=int((delta < -1e-9).sum()),ties=int((np.abs(delta)<=1e-9).sum()),losses=int((delta>1e-9).sum()),
                                 median_paired_delta_m=float(delta.median()),best_delta_m=float(delta.min()),worst_delta_m=float(delta.max())))
        pd.DataFrame(rows).to_csv(HERE/"paired"/"paired_differences.csv",index=False)
        records=json.loads((HERE/"paired"/"records.json").read_text(encoding="utf8"))
        result.update(paired_traces=len(records),maximum_yaw_change=max(row["yaw_max_change"] for row in records),
                      outside_overlay_max_position_change=max(row["outside_gate_position_max_change"] for row in records))
        later=pairs[pairs.index.get_level_values("interval")!="0-2"]
        result["later_xy_score_max_change"]=float(max((later[policy]-later.K).abs().max() for policy in POLICIES[1:]))
        height=frame.pivot(index=["name","queue","outage_s","interval"],columns="policy",values="z_rmse")
        result["height_score_max_change"]=float(max((height[policy]-height.K).abs().max() for policy in POLICIES[1:]))
        selected=pairs.xs(("first_per_stamp","0-2"),level=("queue","interval"))
        fig,axes=plt.subplots(1,3,figsize=(11,3.7),sharex=True,sharey=True)
        for ax,policy in zip(axes,POLICIES[1:]):
            for length,group in selected.groupby(level="outage_s"):
                ax.scatter(group.K,group[policy],label=f"L={length:g} s",s=25)
            ax.plot([.05,500],[.05,500],"k--",lw=.8)
            ax.set(xscale="log",yscale="log",xlabel="Native K XY RMSE (m)",title=policy)
            ax.grid(alpha=.2)
        axes[0].set_ylabel("Overlay XY RMSE (m)")
        axes[-1].legend(fontsize=7)
        fig.suptitle("First two seconds: every primary-queue recording/outage pair")
        fig.tight_layout()
        fig.savefig(HERE/"paired"/"paired_xy.png",dpi=160)
        fig.savefig(HERE/"paired"/"paired_xy.svg")
        plt.close(fig)
    (HERE/"audit.json").write_text(json.dumps(result,indent=2),encoding="utf8")
    print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__":
    main()
