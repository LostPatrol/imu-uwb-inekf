"""Train raw-TWR observation models and compare paired inertial endpoint fits.

Only the chronological first log of each motion calibrates radio parameters.
Test truth is kept outside optimization and used only for descriptive scoring.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

# Existing loader, inertial integration and seeds are reused without modification.
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
sys.path.insert(0, str(PROJECT / "experiments" / "recovery_bootstrap"))
from run_research import (DATASET, load_real_log, load_anchors,
                          estimate_initial_rotation_and_bias, preintegrate)
from initializer import cv_candidate, yaw_rotation

# Motion pairing and time correction reproduce the currently loaded protocol.
MOTIONS = ("triangle", "octagon", "hourglass", "random", "star", "square")
DELAY = 0.20
PREFIX = 1.5
# The common finite budget includes four identical yaw starts per model.
STARTS = 4
MAX_NFEV = 120
# Training regularization is selected only by leave-one-training-log-out scores.
RIDGES = (1.0, 10.0, 100.0, 1000.0)
# Coarse variance bins avoid flexible variance fitting with little local evidence.
DISTANCE_EDGES = np.array([0.0, 4.0, 6.0, 8.0, np.inf])
MODELS = ("static", "geometry", "geometry_hetero", "geometry_ar")


def dump(path, value):
    """Save NumPy-compatible evidence as readable UTF-8 JSON."""
    def convert(item):
        """Convert arrays and scalar NumPy types recursively."""
        if isinstance(item, dict):
            return {str(k): convert(v) for k, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [convert(v) for v in item]
        if isinstance(item, np.ndarray):
            return convert(item.tolist())
        if isinstance(item, (np.integer, np.bool_)):
            return item.item()
        if isinstance(item, (float, np.floating)):
            return float(item) if np.isfinite(item) else None
        return item
    path.write_text(json.dumps(convert(value), indent=2) + "\n", encoding="utf-8")


def events(log, anchors):
    """Extract changed positive ranges with offline physical residual covariates."""
    rows, ids = np.nonzero(log.changed & (log.ranges > 0))
    delta = log.ground_truth_position[rows] - anchors[ids]
    distance = np.linalg.norm(delta, axis=1)
    return pd.DataFrame(dict(sequence=log.name, row=rows, anchor=ids,
        time=log.time[rows], distance=distance, ux=delta[:, 0]/distance,
        uy=delta[:, 1]/distance, uz=delta[:, 2]/distance,
        residual=log.ranges[rows, ids]-distance))


def fit_coefficients(frame, ridge):
    """Fit robust per-anchor mean corrections with a ridge on geometry slopes."""
    coefficients, centers = [], []
    for anchor in range(8):
        part = frame[frame.anchor == anchor]
        center = part.distance.median()
        design = np.column_stack((np.ones(len(part)), part.distance-center,
                                   part[["ux", "uy", "uz"]]))
        y = part.residual.to_numpy()
        beta = np.r_[np.median(y), np.zeros(4)]
        penalty = np.diag([0.0, ridge, ridge, ridge, ridge])
        for _ in range(10):
            error = y-design@beta
            scale = max(.08, 1.4826*np.median(np.abs(error-np.median(error))))
            weight = np.minimum(1., 2*scale/np.maximum(np.abs(error), 1e-12))
            beta = np.linalg.solve(design.T@(weight[:, None]*design)+penalty,
                                   design.T@(weight*y))
        coefficients.append(beta)
        centers.append(center)
    return np.asarray(coefficients), np.asarray(centers)


def predict_training(frame, beta, centers):
    """Predict means on offline covariates without updating fitted coefficients."""
    ids = frame.anchor.to_numpy(int)
    design = np.column_stack((np.ones(len(frame)), frame.distance-centers[ids],
                               frame[["ux", "uy", "uz"]]))
    return np.sum(beta[ids]*design, axis=1)


def calibrate(train, output):
    """Freeze static, geometric, heteroscedastic and correlation models on training."""
    cv = []
    for ridge in RIDGES:
        for sequence in train.sequence.unique():
            fitting = train[train.sequence != sequence]
            validation = train[train.sequence == sequence]
            beta, centers = fit_coefficients(fitting, ridge)
            error = validation.residual.to_numpy()-predict_training(validation, beta, centers)
            cv.append(dict(ridge=ridge, sequence=sequence, mae_m=np.mean(np.abs(error))))
    cv = pd.DataFrame(cv)
    cv.to_csv(output/"training_leave_log_out.csv", index=False)
    ridge = float(cv.groupby("ridge").mae_m.mean().idxmin())
    beta, centers = fit_coefficients(train, ridge)
    static, sigma, bins, rho, interval = [], [], [], [], []
    error = train.residual.to_numpy()-predict_training(train, beta, centers)
    train = train.assign(geometry_error=error)
    for anchor in range(8):
        part = train[train.anchor == anchor]
        bias = part.residual.median()
        scale = np.clip(1.4826*np.median(np.abs(part.residual-bias)), .08, .5)
        static.append(bias)
        sigma.append(scale)
        global_scale = np.clip(1.4826*np.median(np.abs(part.geometry_error-part.geometry_error.median())), .08, .5)
        anchor_bins = []
        for low, high in zip(DISTANCE_EDGES[:-1], DISTANCE_EDGES[1:]):
            local = part[(part.distance >= low) & (part.distance < high)].geometry_error
            local_scale = 1.4826*np.median(np.abs(local-local.median())) if len(local) >= 100 else global_scale
            anchor_bins.append(np.clip(local_scale, .08, .5))
        bins.append(anchor_bins)
        pairs, dts = [], []
        for _, block in part.groupby("sequence"):
            e = np.clip(block.geometry_error.to_numpy()/global_scale, -3., 3.)
            pairs.append(np.column_stack((e[:-1], e[1:])))
            dts.extend(np.diff(block.time))
        pair = np.vstack(pairs)
        rho.append(np.clip(np.sum(pair[:, 0]*pair[:, 1])/np.sum(pair[:, 0]**2), 0., .9))
        interval.append(np.median(dts))
    result = dict(static=static, sigma=sigma, beta=beta, centers=centers,
                  bins=bins, rho=rho, interval=interval, selected_ridge=ridge)
    dump(output/"calibration.json", result)
    return {key: np.asarray(value) if isinstance(value, (list, np.ndarray)) else value
            for key, value in result.items()}


def diagnostics(frame, calibration, output):
    """Describe timing, mean structure, tails and same-anchor error persistence."""
    rows = []
    for (sequence, anchor), part in frame.groupby(["sequence", "anchor"]):
        e = part.residual.to_numpy()-calibration["static"][anchor]
        sigma = calibration["sigma"][anchor]
        positive = e > 2*sigma
        run_lengths, length = [], 0
        for value in positive:
            length = length+1 if value else 0
            if length:
                run_lengths.append(length)
        dt = np.diff(part.time)
        rows.append(dict(sequence=sequence, anchor=anchor, events=len(part),
            median_dt_s=np.median(dt), p95_dt_s=np.quantile(dt,.95),
            bias_m=part.residual.median(), static_median_error_m=np.median(e),
            static_mae_m=np.mean(np.abs(e)), static_rmse_m=np.sqrt(np.mean(e**2)),
            mad_m=1.4826*np.median(np.abs(e-np.median(e))),
            lag1_correlation=np.corrcoef(e[:-1],e[1:])[0,1],
            distance_correlation=np.corrcoef(e,part.distance)[0,1],
            positive_tail_fraction=np.mean(positive),
            positive_tail_stay_probability=np.mean(positive[1:][positive[:-1]]) if positive[:-1].any() else 0,
            max_positive_run_events=max(run_lengths, default=0),
            max_positive_run_approx_s=max(run_lengths, default=0)*np.median(dt)))
    pd.DataFrame(rows).to_csv(output/"event_diagnostics.csv", index=False)
    mean = predict_training(frame,calibration["beta"],calibration["centers"])
    scored = frame.assign(geometry_error=frame.residual-mean,
        static_error=frame.residual-calibration["static"][frame.anchor.to_numpy(int)])
    scored.groupby(["sequence", "anchor"])[["geometry_error", "static_error"]].agg(
        ["median", lambda x:np.mean(np.abs(x)), lambda x:np.sqrt(np.mean(x*x))]).to_csv(output/"radio_prediction_scores.csv")
    return scored


def raw_model(x, tau, displacement, anchors, ranges, ids, model, calibration):
    """Differentiate original ranges including position-dependent radio means."""
    rotated = displacement@yaw_rotation(x[6]).T
    position = x[:3]+tau[:,None]*x[3:6]+rotated
    difference = position-anchors
    distance = np.maximum(np.linalg.norm(difference,axis=1),1e-9)
    direction = difference/distance[:,None]
    gradient = direction.copy()
    mean = calibration["static"][ids]
    if model.startswith("geometry"):
        beta = calibration["beta"][ids]
        mean = beta[:,0]+beta[:,1]*(distance-calibration["centers"][ids])+np.sum(beta[:,2:]*direction,axis=1)
        tangent = (beta[:,2:]-np.sum(beta[:,2:]*direction,axis=1)[:,None]*direction)/distance[:,None]
        gradient += beta[:,1,None]*direction+tangent
    sigma = calibration["sigma"][ids]
    if model == "geometry_hetero":
        # Bins depend on the observed range, so variance is fixed within a fit.
        bin_ids = np.clip(np.searchsorted(DISTANCE_EDGES,ranges,side="right")-1,0,3)
        sigma = calibration["bins"][ids,bin_ids]
    yaw_derivative = np.column_stack((-rotated[:,1],rotated[:,0],np.zeros(len(tau))))
    jac = np.column_stack((gradient,tau[:,None]*gradient,np.sum(gradient*yaw_derivative,axis=1)))/sigma[:,None]
    residual = (distance+mean-ranges)/sigma
    if model.endswith("_ar"):
        # A fixed trained OU approximation whitens only earlier same-anchor events.
        source_residual, source_jac = residual.copy(), jac.copy()
        for anchor in np.unique(ids):
            indices = np.flatnonzero(ids==anchor)
            current, previous = indices[1:],indices[:-1]
            factor = calibration["rho"][anchor]**((tau[current]-tau[previous])/calibration["interval"][anchor])
            denominator = np.sqrt(np.maximum(1-factor**2,.05))
            residual[current] = (source_residual[current]-factor*source_residual[previous])/denominator
            jac[current] = (source_jac[current]-factor[:,None]*source_jac[previous])/denominator[:,None]
    return residual,jac


def solve(values, calibration, model, seeds):
    """Apply an identical fixed four-start nonlinear budget to each observation model."""
    def fun(x):
        """Evaluate only arrived radio/IMU observations; truth is absent."""
        return raw_model(x,*values,model,calibration)[0]
    def jac(x):
        """Evaluate exact derivatives through radio geometry and fixed whitening."""
        return raw_model(x,*values,model,calibration)[1]
    fits = [least_squares(fun,seed,jac=jac,loss="huber",f_scale=2.,max_nfev=MAX_NFEV) for seed in seeds]
    fit = min(fits,key=lambda value:value.cost)
    return fit.x, sum(value.nfev for value in fits), 2*fit.cost


def experiment(logs, anchors, calibration, output, models=MODELS):
    """Compare physical endpoint errors on six fixed held-out path groups."""
    rows, predictions = [], []
    for sequence, log in logs.items():
        rotation, bias = estimate_initial_rotation_and_bias(log)
        selected = log.time>=PREFIX
        times = log.time[selected]-log.time[selected][0]
        _,d,v = preintegrate(times,log.imu[selected],rotation,bias)
        for endpoint in (5.,10.,20.,30.):
            for duration in (2.,4.,8.):
                if duration>endpoint:
                    continue
                end = np.searchsorted(times,endpoint,side="right")-1
                start = np.searchsorted(times,times[end]-duration,side="left")
                for layout in ("all8","lower4"):
                    mask = log.changed[selected][start:end+1] & (log.ranges[selected][start:end+1]>0)
                    if layout=="lower4":
                        mask[:,4:] = False
                    event_rows,ids = np.nonzero(mask)
                    event_rows += start
                    tau = times[event_rows]-times[end]
                    displacement = d[event_rows]-d[end]-tau[:,None]*v[end]
                    ranges = log.ranges[selected][event_rows,ids]
                    event_anchors = anchors[ids]
                    corrected = ranges-calibration["static"][ids]
                    cv,_,_ = cv_candidate(tau,event_anchors,corrected,calibration["sigma"][ids])
                    seeds = [np.r_[cv,angle] for angle in np.linspace(-np.pi,np.pi,STARTS,endpoint=False)]
                    values = (tau,displacement,event_anchors,ranges,ids)
                    truth = log.ground_truth_position[selected][end]
                    for model in models:
                        started = time.perf_counter()
                        x,nfev,cost = solve(values,calibration,model,seeds)
                        elapsed = time.perf_counter()-started
                        rows.append(dict(sequence=sequence,endpoint_s=endpoint,duration_s=duration,
                            layout=layout,model=model,position_error_m=np.linalg.norm(x[:3]-truth),
                            nfev=nfev,elapsed_s=elapsed,objective=cost,events=len(tau),
                            x=x[0],y=x[1],z=x[2],truth_x=truth[0],truth_y=truth[1],truth_z=truth[2]))
        print(f"DONE {sequence} ({len(rows)} fits)",flush=True)
        pd.DataFrame(rows).to_csv(output/"endpoint_metrics.csv",index=False)
    frame = pd.DataFrame(rows)
    summary = frame.groupby(["sequence","model"]).position_error_m.agg(
        median="median",p95=lambda x:x.quantile(.95),rmse=lambda x:np.sqrt(np.mean(x*x)),maximum="max")
    summary.to_csv(output/"per_path_summary.csv")
    frame.groupby(["layout","model"]).position_error_m.agg(
        median="median",p95=lambda x:x.quantile(.95),rmse=lambda x:np.sqrt(np.mean(x*x)),maximum="max").to_csv(output/"overall_summary.csv")
    if "static" not in models:
        return
    keys = ["sequence","endpoint_s","duration_s","layout"]
    paired = frame.pivot(index=keys,columns="model",values="position_error_m")
    for model in MODELS[1:]:
        paired[model+"_minus_static"] = paired[model]-paired.static
    paired.to_csv(output/"paired_differences.csv")
    fig,axes = plt.subplots(1,2,figsize=(11,4),sharey=True)
    for ax,layout in zip(axes,("all8","lower4")):
        part = frame[frame.layout==layout]
        for index,model in enumerate(MODELS):
            values = part[part.model==model].position_error_m
            ax.scatter(np.full(len(values),index),values,s=10,alpha=.5)
            ax.plot([index-.2,index+.2],[values.median()]*2,color="black")
        ax.set_xticks(range(len(MODELS)),MODELS,rotation=20)
        ax.set_title(layout)
        ax.set_yscale("log")
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Endpoint position error (m), 66 overlapping windows")
    fig.tight_layout()
    fig.savefig(output/"endpoint_comparison.png",dpi=180)
    plt.close(fig)


def main():
    """Freeze raw split and parameters before reading held-out evaluation truth."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=HERE/"results")
    parser.add_argument("--diagnostics-only",action="store_true")
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    anchors=load_anchors(DATASET)
    train_names,test_names=[],[]
    for motion in MOTIONS:
        pair=sorted(path.name for path in (DATASET/"twr").glob(f"*twr_{motion}.csv"))
        train_names.append(pair[0])
        test_names.append(pair[1])
    train_logs={name:load_real_log(DATASET,name,sensor_delay_s=DELAY) for name in train_names}
    train=pd.concat([events(log,anchors) for log in train_logs.values()],ignore_index=True)
    calibration=calibrate(train,args.output)
    diagnostics(train,calibration,args.output).to_csv(args.output/"training_events.csv",index=False)
    dump(args.output/"protocol.json",dict(training=train_names,evaluation=test_names,
        models=MODELS,common_delay_s=DELAY,prefix_s=PREFIX,max_nfev_per_start=MAX_NFEV,
        starts=STARTS,seed_policy="Shared static-calibrated CV translation and uniform four yaw seeds",
        windows="5/10/20/30 s endpoints, 2/4/8 s durations <= endpoint; all8/lower4",
        geometry_model="Per-anchor robust intercept + distance slope + world LOS direction; training-log CV ridge",
        hetero_model="Per-anchor 4 observed-range bins with MAD clipped to [0.08,0.50] m; >=100 training events/bin",
        ar_model="Fixed per-anchor clipped residual correlation, decay with actual elapsed same-anchor time",
        no_test_truth_in_estimator=True,blind_test=False,
        independent_units="Six same-day paired paths; overlapping windows are descriptive, not iid",
        static_reference="Per-anchor median, MAD x1.4826 clipped [0.08,0.50] m; Huber scale 2",
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    if args.diagnostics_only:
        return
    test_logs={name:load_real_log(DATASET,name,sensor_delay_s=DELAY) for name in test_names}
    scored=diagnostics(pd.concat([events(log,anchors) for log in test_logs.values()],ignore_index=True),calibration,args.output)
    scored.to_csv(args.output/"evaluation_events_scoring_only.csv",index=False)
    experiment(test_logs,anchors,calibration,args.output)


if __name__=="__main__":
    main()
