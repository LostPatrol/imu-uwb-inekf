"""Plot paired independent recovery errors and explicitly show display return steps."""

import independent as experiment
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Consistent visual labels identify output candidates without implying state feedback.
METHODS=("K","always_C","guarded","radio_bounds")
COLORS=("black","tab:orange","tab:blue","tab:green")


def main():
    """Visualize every paired primary case and one predetermined first-flight example."""
    output=experiment.OUTPUT
    table=pd.read_csv(output/"paired.csv")
    figure,axes=plt.subplots(1,3,figsize=(14,4.1))
    for method,color in zip(METHODS,COLORS):
        rows=table[table.method==method]
        group=rows.groupby("gap_s")
        axes[0].plot(group.early_xy_rmse.median(),"o-",label=method,color=color)
        if method!="K":
            axes[1].scatter(rows.gap_s,rows.early_xy_rmse_minus_K,s=22,alpha=.7,color=color,label=method)
            axes[2].plot(group.maximum_added_xy_step_m.max(),"o-",color=color,label=method)
    axes[0].set(ylabel="Early XY RMSE median (m)",yscale="log",title="Eight balanced trial-2 recordings")
    axes[1].axhline(0,color="black",lw=.7)
    axes[1].set(ylabel="Paired early XY RMSE change (m)",yscale="symlog",title="Every case: below zero improves")
    axes[1].set_yscale("symlog",linthresh=.01)
    axes[2].set(ylabel="Largest added XY output step (m)",yscale="log",title="Includes switches and return to K")
    for axis in axes:
        axis.set_xlabel("Radio outage duration (s)")
        axis.grid(alpha=.25)
        axis.set_xticks(experiment.GAPS)
    axes[0].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output/"paired_overview.png",dpi=180)
    plt.close(figure)
    # First prelisted recording and longest outage are fixed diagnostic choices.
    name="const1-trial2-tdoa2"
    key=name+"__gap20"
    data=np.load(output/(name+"__data.npz"))
    trajectories=np.load(output/"scored"/(key+".npz"))
    origin=float(data["origin"])
    truth=data["truth"]
    figure,axes=plt.subplots(2,1,figsize=(10,6),sharex=True)
    for method,color in zip(METHODS,COLORS):
        rows=trajectories[method]
        selected=(rows[:,0]>=origin+24)&(rows[:,0]<=origin+36)
        rows=rows[selected]
        reference=np.column_stack([np.interp(rows[:,0],truth[:,0],truth[:,i]) for i in (1,2)])
        error=np.linalg.norm(rows[:,1:3]-reference,axis=1)
        axes[0].plot(rows[:,0]-origin,error,label=method,color=color)
        axes[1].plot(rows[:,0]-origin,rows[:,1],label=method,color=color)
    for axis in axes:
        axis.axvspan(25,27,color="grey",alpha=.15,label="output-only phase" if axis is axes[0] else None)
        axis.axvline(27,color="grey",linestyle="--")
        axis.grid(alpha=.25)
    axes[0].set(ylabel="XY error (m)",title="Predetermined example: first recording, 20 s outage")
    axes[1].set(ylabel="World x output (m)",xlabel="Time from common sensor origin (s)")
    axes[0].legend(ncol=3,fontsize=8)
    figure.tight_layout()
    figure.savefig(output/"deadline_example.png",dpi=180)
    plt.close(figure)


if __name__=="__main__":
    main()
