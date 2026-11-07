"""Independently audit retained trajectories and create publication figures/tables."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Stable method order and labels across every exported figure.
METHODS=["cv","local","multistart","profile","adaptive","capture","capture_robust","centroid"]
LABELS=["CV","Local","12-start","Profile","Adaptive","Lift","Lift+gate","Direct"]
# Consistent accessible dark blue, rust, and teal throughout the manuscript.
BLUE,ORANGE,TEAL="#21618c","#bd5c2a","#218c74"


def save_figure(fig,directory,name):
    """Export vector PDF plus a high-resolution preview without resizing the canvas."""
    directory.mkdir(parents=True,exist_ok=True)
    fig.savefig(directory/(name+".pdf"),bbox_inches="tight",pad_inches=0.04)
    fig.savefig(directory/(name+".png"),dpi=220,bbox_inches="tight",pad_inches=0.04)
    plt.close(fig)


def audit(root,frame):
    """Recompute metrics from full arrays without calling the experiment's scorer."""
    differences=[]
    checked=0
    missing=[]
    for row in frame.itertuples():
        identifier=f"{Path(row.sequence).stem}__{row.method}__outage{int(row.outage)}"
        path=root/"histories"/(identifier+".npz")
        if not path.exists():
            missing.append(str(path))
            continue
        data=np.load(path)
        t,output,truth=data["time"],data["output"],data["truth"]
        finite=np.isfinite(output).all(axis=1)
        error=np.sqrt(np.sum((output[:,:3]-truth)**2,axis=1))
        score=(t>=8.1)&finite
        values={"availability":np.mean(finite),"common_availability":np.mean(finite[t>=8.1])}
        values["common_rmse_m"]=np.sqrt(np.mean(error[score]**2)) if score.any() else np.nan
        values["tail5_rmse_m"]=np.sqrt(np.mean(error[(t>=t[-1]-5)&finite]**2)) if ((t>=t[-1]-5)&finite).any() else np.nan
        values["max_m"]=np.max(error[finite]) if finite.any() else np.nan
        origin=25 if row.outage else 0
        window=np.flatnonzero(t>=origin-1e-10)
        violation=window[(~np.isfinite(error[window]))|(error[window]>0.5)]
        first=violation[-1]+1 if len(violation) else window[0]
        success=first<len(t) and t[-1]-t[first]>=5-1e-10
        if bool(row.success)!=bool(success):
            raise AssertionError((identifier,"success",row.success,success))
        for key,value in values.items():
            recorded=getattr(row,key)
            if np.isnan(value) and np.isnan(recorded):
                continue
            difference=abs(value-recorded)
            if not np.isfinite(difference) or difference>1e-8:
                raise AssertionError((identifier,key,value,recorded))
            differences.append(difference)
        checked+=1
    result=dict(cases_checked=checked,missing=missing,max_absolute_difference=max(differences),passed=not missing,
                real_metrics_sha256=hashlib.sha256((root/"real_metrics.csv").read_bytes()).hexdigest())
    (root/"independent_audit.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    return result


def figures(root,destination):
    """Generate comparable figures and machine-readable aggregates from frozen outputs."""
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8,"axes.spines.top":False,
                         "axes.spines.right":False,"axes.labelsize":8,"xtick.labelsize":7,
                         "ytick.labelsize":7,"legend.fontsize":7,"pdf.fonttype":42})
    real=pd.read_csv(root/"real_metrics.csv")
    synthetic=pd.read_csv(root/"synthetic_metrics.csv")
    bias=pd.read_csv(root/"bias_metrics.csv")
    verification=audit(root,real)
    grouped=real.groupby(["outage","method"])
    table=grouped.agg(successes=("success","sum"),runs=("success","size"),
        median_common_rmse_m=("common_rmse_m","median"),median_tail5_rmse_m=("tail5_rmse_m","median"),
        maximum_error_m=("max_m","max"),mean_availability=("availability","mean"),
        mean_common_availability=("common_availability","mean"),median_startup_unavailable_s=("startup_unavailable_s","median"))
    table.to_csv(root/"real_summary.csv")
    synth_summary=synthetic.groupby(["motion","contaminated","method"]).median(numeric_only=True)
    synth_summary.to_csv(root/"synthetic_summary.csv")
    bias_summary=bias.groupby(["motion","bias_std","method"]).agg(
        n=("seed","size"),median_yaw_deg=("yaw_error_deg","median"),median_position_m=("position_error_m","median"),
        median_information=("yaw_information","median"),accepted=("accepted","sum"),false_confident=("false_confident","sum"))
    bias_summary.to_csv(root/"bias_summary.csv")

    # Mean/variance-free display: report success counts together with full-output peaks.
    fig,axes=plt.subplots(2,1,figsize=(3.48,3.8),sharex=True,layout="constrained")
    x=np.arange(len(METHODS))
    for outage,offset,color,label in ((False,-.19,BLUE,"Normal"),(True,.19,ORANGE,"10 s outage")):
        values=[table.loc[(outage,m),"successes"] for m in METHODS]
        axes[0].bar(x+offset,values,.36,color=color,label=label)
    axes[0].set(ylabel="Stable success / 6",ylim=(0,6.65),yticks=[0,2,4,6])
    axes[0].legend(loc="lower left",ncol=1)
    peaks=[table.loc[(False,m),"maximum_error_m"] for m in METHODS]
    axes[1].scatter(x,peaks,color=BLUE,s=22,zorder=3)
    axes[1].set_yscale("log")
    axes[1].set(ylabel="Normal maximum error (m)",xticks=x,xticklabels=LABELS)
    axes[1].tick_params(axis="x",rotation=55)
    for ax in axes:
        ax.grid(axis="y",alpha=.2)
    save_figure(fig,destination,"real_summary")

    fig,axes=plt.subplots(2,1,figsize=(3.48,3.65),layout="constrained")
    for method,color,label in (("fixed_bias",ORANGE,"Bias fixed"),("free_bias",BLUE,"Quadratic nuisance")):
        data=bias[(bias.motion=="excited")&(bias.method==method)]
        group=data.groupby("bias_std").yaw_error_deg
        med=group.median()
        axes[0].plot(med.index,med.values,"o-",color=color,label=label,markersize=4)
        axes[0].fill_between(med.index,group.quantile(.25),group.quantile(.75),color=color,alpha=.14)
        subset=bias[(bias.motion=="constant_acceleration")&(bias.method==method)]
        rates=subset.groupby("bias_std").false_confident.mean()*100
        axes[1].plot(rates.index,rates.values,"o-",color=color,label=label,markersize=4)
    axes[0].set(ylabel="Yaw error (deg)",title="Excited motion: median and IQR")
    axes[0].legend(loc="upper left")
    axes[1].set(ylabel="Threshold violations (%)",xlabel="Accelerometer bias SD (m/s$^2$)",
                title="Constant acceleration: acceptance can mislead",ylim=(-3,70))
    axes[1].text(.01,58,"Quadratic nuisance rejects all yaw handoffs",fontsize=7,color=BLUE)
    for ax in axes:
        ax.set_xticks([0,.1,.3]); ax.grid(alpha=.2)
    save_figure(fig,destination,"bias_tradeoff")

    # Display a difficult recorded sequence without censoring large finite errors.
    name=next(name for name in real.sequence.unique() if "random" in name)
    fig,axes=plt.subplots(1,2,figsize=(7.1,2.5),layout="constrained")
    selected=[("centroid",ORANGE,"Direct"),("capture",TEAL,"Lift"),("cv","#7d3c98","CV"),("profile",BLUE,"Profile")]
    for column,outage in enumerate((False,True)):
        for method,color,label in selected:
            data=np.load(root/"histories"/f"{Path(name).stem}__{method}__outage{int(outage)}.npz")
            error=np.linalg.norm(data["output"][:,:3]-data["truth"],axis=1)
            axes[column].plot(data["time"],error,color=color,label=label,linewidth=.85)
        axes[column].set_yscale("log")
        axes[column].axhline(.5,color="#555555",linestyle=":",linewidth=.8)
        axes[column].axvspan(0,2,color="#dddddd",alpha=.4,label="Window unavailable" if column==0 else None)
        if outage: axes[column].axvspan(15,25,color=ORANGE,alpha=.12)
        axes[column].set(xlabel="Time after static prefix (s)",ylabel="Position error (m)",
                         title="Random: "+("normal" if not outage else "range outage"),ylim=(.01,1e5))
        axes[column].grid(alpha=.2)
    axes[0].legend(ncol=2,loc="upper right",fontsize=6.7)
    save_figure(fig,destination,"trajectories")

    fig,axes=plt.subplots(2,1,figsize=(3.48,3.7),layout="constrained")
    order=["cv","local","multistart","profile"]
    excited=synthetic[synthetic.motion=="excited"]
    for contaminated,offset,color,label in ((False,-.18,BLUE,"Gaussian"),(True,.18,ORANGE,"5% positive outliers")):
        values=[excited[(excited.method==m)&(excited.contaminated==contaminated)].position_error_m.median() for m in order]
        axes[0].bar(np.arange(4)+offset,values,.34,color=color,label=label)
    axes[0].set(ylabel="Endpoint error (m)",xticks=np.arange(4),xticklabels=["CV","Local","12-start","Profile"])
    axes[0].legend()
    for method,color in (("local",TEAL),("multistart",ORANGE),("profile",BLUE)):
        data=excited[(excited.method==method)&(~excited.contaminated)]
        sorted_times=np.sort(data.elapsed_s.to_numpy()*1000)
        axes[1].plot(sorted_times,np.arange(1,len(data)+1)/len(data),color=color,label=method)
    axes[1].set(xlabel="Initializer time during parallel study (ms)",ylabel="Empirical CDF")
    axes[1].legend(loc="lower right")
    for ax in axes: ax.grid(axis="y",alpha=.2)
    save_figure(fig,destination,"synthetic_summary")

    # Numerical information contrast exposes the bias-dependent null direction.
    fig,ax=plt.subplots(figsize=(3.48,2.5),layout="constrained")
    for method,color in (("fixed_bias",ORANGE),("free_bias",BLUE)):
        for index,motion in enumerate(("constant_acceleration","excited")):
            values=bias[(bias.method==method)&(bias.motion==motion)&(bias.bias_std==.3)].yaw_information
            shift=-.12 if method=="fixed_bias" else .12
            ax.scatter(np.full(len(values),index+shift),np.maximum(values,1e-28),s=3,color=color,alpha=.15)
            ax.plot(index+shift,max(values.median(),1e-28),"_",markersize=14,color=color,
                    label=method.replace("_"," ") if index==0 else None)
    ax.axhline(1/.35**2,color="#555555",linestyle=":",label="handoff threshold")
    ax.set(yscale="log",xticks=[0,1],xticklabels=["Constant acceleration","Excited motion"],ylabel="Data-only yaw information")
    ax.legend(loc="center right")
    save_figure(fig,destination,"information_geometry")

    paired=real[(~real.outage)&real.method.isin(["cv","profile"])].pivot(index="sequence",columns="method",values="common_rmse_m")
    delta=(paired["profile"]-paired["cv"]).to_numpy()
    rng=np.random.default_rng(20261107)
    bootstrap=delta[rng.integers(0,len(delta),(10000,len(delta)))].mean(axis=1)
    evidence=dict(audit=verification,real_rows=len(real),synthetic_rows=len(synthetic),bias_rows=len(bias),
        paired_log_rmse_difference_profile_minus_cv_mean_m=float(delta.mean()),
        descriptive_cluster_bootstrap_95_interval_m=np.quantile(bootstrap,[.025,.975]).tolist(),
        inference_limit="Six same-day development logs; interval is exploratory, not external generalization evidence")
    (root/"analysis.json").write_text(json.dumps(evidence,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(evidence,indent=2))
    print(table.to_string())
    print(bias_summary.to_string())


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results",type=Path,required=True)
    parser.add_argument("--figures",type=Path,required=True)
    arguments=parser.parse_args()
    figures(arguments.results,arguments.figures)
