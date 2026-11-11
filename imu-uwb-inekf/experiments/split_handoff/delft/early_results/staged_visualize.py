"""Audit staged transaction separation and plot retained successes and failures."""

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# These methods and colors are selected by their roles, not measured performance.
HERE=Path(__file__).resolve().parent
METHODS=("fixed_2s_pv","native_gate","early_only_live","staged_xy_pv","continuous")
COLORS=("#000000","#0072b2","#009e73","#d55e00","#cc79a7")


def audit_and_plot():
    """Recompute received-packet parity independently and retain every staged case."""
    maxima=dict(prefix=0.,trigger=0.)
    counts=[]
    for file in sorted((HERE/"staged_histories").glob("*.json")):
        meta=json.loads(file.read_text(encoding="utf-8"))
        staged=np.load(file.with_suffix(".npz"))
        original=np.load(HERE/"histories"/(file.stem+".npz"))
        sensor=np.load(HERE.parent/"results/cache"/(Path(meta["sequence"]).stem+".npz"))
        stamps=sensor["time"]; kt=int(np.searchsorted(stamps,meta["trigger_time_s"]))
        k27=int(np.searchsorted(stamps,27.))
        active=np.ones(len(sensor["anchors"]),dtype=bool)
        if meta["layout"]=="lower4":
            active[:]=False; active[np.argsort(sensor["anchors"][:,2])[:4]]=True
        received=sensor["changed"]&(sensor["ranges"]>0)
        received[:kt+1]=False
        parity=np.cumsum(received,axis=0)%2
        rows,ids=staged["pv_rows"],staged["pv_anchors"]
        assert np.all(rows>kt) and np.all(rows<=k27)
        assert np.all(active[ids]) and np.all(received[rows,ids]) and np.all(parity[rows,ids]==0)
        mask=received.copy(); mask[:kt+1]=False; mask[k27+1:]=False; mask&=active
        odd_count=int(np.sum(mask&(parity==1)))
        even_count=int(np.sum(mask&(parity==0)))
        assert odd_count==meta["counts"]["live_window"]
        assert even_count==len(rows)==meta["counts"]["pv_window"]
        prefix=(stamps>=2.1)&(stamps<meta["trigger_time_s"])
        for suffix in ("","_R","_bias"):
            difference=staged["staged_xy_pv"+suffix][prefix]-original["early_only_live"+suffix][prefix]
            maxima["prefix"]=max(maxima["prefix"],float(np.max(np.abs(difference))))
            maxima["trigger"]=max(maxima["trigger"],float(np.max(np.abs(
                staged["staged_xy_pv"+suffix][kt]-original["early_only_live"+suffix][kt]))))
        counts.append(dict(sequence=meta["sequence"],layout=meta["layout"],yaw_offset_deg=meta["yaw_offset_deg"],
                           live=odd_count,reserved=even_count))
    assert len(counts)==48 and maxima["prefix"]==0 and maxima["trigger"]==0
    (HERE/"staged_packet_audit.json").write_text(json.dumps(dict(cases=48,maxima=maxima,
        early_xy_radio_disjoint=True,live_pv_radio_disjoint=True,received_packet_coverage_exact=True),indent=2),encoding="utf-8")
    frame=pd.read_csv(HERE/"staged_combined_paired.csv")
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    labels=["Wait 2 s PV","Native","Early XY","Staged XY/PV","No silence"]
    for row,layout in enumerate(("all8","lower4")):
        for column,metric in enumerate(("early_xy_rmse_m","fixed_xy_rmse_m")):
            ax=axes[row,column]
            for j,(method,color) in enumerate(zip(METHODS,COLORS)):
                values=frame[(frame.layout==layout)&(frame.method==method)][metric].to_numpy()
                ax.scatter(np.full(len(values),j),values,color=color,alpha=.55,s=16)
                ax.plot([j-.2,j+.2],[np.median(values)]*2,color=color,lw=3)
            ax.set_xticks(range(len(METHODS)),labels,rotation=20,ha="right")
            ax.set(title=layout+" / "+("25–27 s" if column==0 else "31–36 s"),
                   yscale="log",ylabel="XY radial RMSE (m)")
            ax.grid(axis="y",alpha=.25)
    fig.savefig(HERE/"staged_comparison.png",dpi=165); plt.close(fig)
    # This prespecified troublesome condition motivated the supplement; it is not a best-case selection.
    identifier="2021-02-12_09_51_57_twr_hourglass__lower4__yaw90"
    original=np.load(HERE/"histories"/(identifier+".npz"))
    staged=np.load(HERE/"staged_histories"/(identifier+".npz"))
    stamps=original["time"]; truth=original["truth"]
    fig,ax=plt.subplots(figsize=(11,4),constrained_layout=True)
    for method,color in zip(METHODS,COLORS):
        prediction=staged[method] if method=="staged_xy_pv" else original[method]
        error=np.linalg.norm(prediction[:,:2]-truth[:,:2],axis=1)
        ax.plot(stamps,error,color=color,label=method,lw=1.2)
    for stamp in (25.,27.):
        ax.axvline(stamp,color="0.5",ls="--",lw=.8)
    ax.set(xlim=(25,float(stamps[-1])),yscale="log",xlabel="Replay time (s)",ylabel="XY radial error (m)",
           title="Hourglass / lower4 / injected +90°: full post-outage history")
    ax.grid(alpha=.2); ax.legend(fontsize=8,ncol=3)
    fig.savefig(HERE/"staged_hourglass_failure.png",dpi=165); plt.close(fig)


if __name__=="__main__":
    audit_and_plot()
