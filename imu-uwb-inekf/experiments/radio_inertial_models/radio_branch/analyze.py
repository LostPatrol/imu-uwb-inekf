"""Audit transfer, timing and physical endpoint effects of frozen TWR models."""

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize._numdiff import approx_derivative

from study import (HERE, DATASET, DELAY, events, load_anchors, load_real_log,
                   diagnostics, raw_model, dump, MODELS)


def audit_events(output, calibration, protocol):
    """Retain split-specific residual evidence and train-only timing diagnostics."""
    anchors = load_anchors(DATASET)
    summaries, autocorrelation, radial, simultaneous, delays = [], [], [], [], []
    for split in ("training", "evaluation"):
        frames = []
        for sequence in protocol[split]:
            log = load_real_log(DATASET,sequence,sensor_delay_s=DELAY)
            frame = events(log,anchors)
            ids = frame.anchor.to_numpy(int)
            row = frame.row.to_numpy(int)
            direction = frame[["ux","uy","uz"]].to_numpy()
            frame["radial_speed_mps"] = np.sum(direction*log.ground_truth_velocity[row],axis=1)
            frame["split"] = split
            frames.append(frame)
        frame = pd.concat(frames,ignore_index=True)
        split_dir=output/split
        split_dir.mkdir(exist_ok=True)
        scored=diagnostics(frame,calibration,split_dir)
        for sequence, part in scored.groupby("sequence"):
            summaries.append(dict(split=split,sequence=sequence,
                static_mae_m=part.static_error.abs().mean(),
                geometry_mae_m=part.geometry_error.abs().mean(),
                static_rmse_m=np.sqrt(np.mean(part.static_error**2)),
                geometry_rmse_m=np.sqrt(np.mean(part.geometry_error**2))))
            # Coarse bins allow cross-anchor comparison despite asynchronous arrivals.
            grouped=part.assign(time_bin=np.floor(part.time/.1)).pivot_table(
                index="time_bin",columns="anchor",values="geometry_error",aggfunc="mean")
            correlation=grouped.corr()
            for a in range(8):
                for b in range(a+1,8):
                    simultaneous.append(dict(split=split,sequence=sequence,a=a,b=b,
                        correlation=correlation.loc[a,b]))
        for (sequence,anchor),part in scored.groupby(["sequence","anchor"]):
            error=part.geometry_error.to_numpy()
            times=part.time.to_numpy()
            radial.append(dict(split=split,sequence=sequence,anchor=anchor,
                residual_vs_radial_speed=np.corrcoef(part.static_error,part.radial_speed_mps)[0,1],
                remaining_vs_distance=np.corrcoef(error,part.distance)[0,1]))
            for lag in (1,2,4,8,16,32):
                autocorrelation.append(dict(split=split,sequence=sequence,anchor=anchor,lag_events=lag,
                    median_lag_s=np.median(times[lag:]-times[:-lag]),
                    correlation=np.corrcoef(error[:-lag],error[lag:])[0,1]))
        scored.to_csv(split_dir/"events.csv",index=False)
    # This is a sensitivity audit, never a new test-driven correction.
    for delay in (0.,.1,.2,.3,.4):
        all_events=pd.concat([events(load_real_log(DATASET,name,sensor_delay_s=delay),anchors)
            for name in protocol["training"]],ignore_index=True)
        centered=all_events.residual-all_events.groupby("anchor").residual.transform("median")
        delays.append(dict(delay_s=delay,training_mae_m=centered.abs().mean(),
            training_rmse_m=np.sqrt(np.mean(centered**2))))
    pd.DataFrame(summaries).to_csv(output/"transfer_summary.csv",index=False)
    pd.DataFrame(autocorrelation).to_csv(output/"lag_correlation.csv",index=False)
    pd.DataFrame(radial).to_csv(output/"radial_speed_correlation.csv",index=False)
    pd.DataFrame(simultaneous).to_csv(output/"cross_anchor_correlation_100ms.csv",index=False)
    pd.DataFrame(delays).to_csv(output/"training_timing_sensitivity.csv",index=False)


def audit_solver(output,calibration):
    """Check analytic derivatives and consolidate paired endpoint/cost evidence."""
    rng=np.random.default_rng(2701)
    tau=np.linspace(-2,0,100)
    displacement=rng.normal(size=(100,3))*.1
    ids=np.arange(100)%8
    anchors=load_anchors(DATASET)[ids]
    x=np.array([0.,0.,1.,.1,.2,.3,.7])
    ranges=np.linalg.norm(x[:3]-anchors,axis=1)+rng.normal(size=100)*.1
    checks={}
    for model in (*MODELS,"static_ar"):
        def fun(value):
            """Keep numerical Jacobian checks independent of optimization truth."""
            return raw_model(value,tau,displacement,anchors,ranges,ids,model,calibration)[0]
        jac=raw_model(x,tau,displacement,anchors,ranges,ids,model,calibration)[1]
        checks[model]=float(np.max(np.abs(jac-approx_derivative(fun,x))))
    frame=pd.read_csv(output/"endpoint_metrics.csv")
    keys=["sequence","endpoint_s","duration_s","layout"]
    paired=frame.pivot(index=keys,columns="model",values="position_error_m")
    rows=[]
    for (sequence,layout),part in paired.groupby(level=[0,3]):
        for model in MODELS[1:]:
            difference=part[model]-part.static
            rows.append(dict(sequence=sequence,layout=layout,model=model,windows=len(part),
                wins=int((difference<0).sum()),losses=int((difference>0).sum()),
                median_difference_m=difference.median(),mean_difference_m=difference.mean(),
                rmse_static_m=np.sqrt(np.mean(part.static**2)),
                rmse_new_m=np.sqrt(np.mean(part[model]**2)),
                worst_improvement_m=difference.min(),worst_degradation_m=difference.max()))
    pd.DataFrame(rows).to_csv(output/"paired_by_path.csv",index=False)
    frame.groupby("model")[["nfev","elapsed_s"]].agg(["median","mean","max","sum"]).to_csv(output/"solver_costs.csv")
    dump(output/"numerical_checks.json",dict(max_absolute_jacobian_error=checks,
        expected_rows=6*11*2*4,actual_rows=len(frame),finite_endpoints=bool(np.isfinite(frame.position_error_m).all()),
        max_actual_nfev=int(frame.nfev.max()),budget_cap_nfev=480,
        note="Seed CV has two common max120 solves and is shared across all four radio models; no truth seeds"))
    frame.groupby(["duration_s","layout","model"]).position_error_m.agg(
        median="median",p95=lambda x:x.quantile(.95),rmse=lambda x:np.sqrt(np.mean(x*x))).to_csv(output/"duration_summary.csv")
    fig,axes=plt.subplots(2,3,figsize=(12,7))
    for ax,(sequence,part) in zip(axes.flat,paired.groupby(level=0)):
        for model,color in zip(MODELS[1:],("#3366aa","#dd8822","#22aa66")):
            ax.scatter(part.static,part[model],s=15,alpha=.7,label=model)
        ax.plot([.03,10],[.03,10],color="black",linewidth=.8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(.03,10)
        ax.set_ylim(.03,10)
        ax.set_title(sequence.split("twr_")[-1].removesuffix(".csv"))
        ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=7)
    fig.supxlabel("Static median/MAD+Huber endpoint error (m)")
    fig.supylabel("Candidate endpoint error (m)")
    fig.tight_layout()
    fig.savefig(output/"paired_by_path.png",dpi=180)
    plt.close(fig)


def main():
    """Use saved frozen models and raw data solely for the declared audits."""
    output=HERE/"results"
    protocol=json.loads((output/"protocol.json").read_text())
    values=json.loads((output/"calibration.json").read_text())
    calibration={key:np.asarray(value) if isinstance(value,list) else value for key,value in values.items()}
    audit_events(output,calibration,protocol)
    audit_solver(output,calibration)


if __name__=="__main__":
    main()
