"""Test causal early horizontal handoff separately from the frozen yaw matrix.

Only received raw ranges determine availability. Evaluation positions enter
after replay; no heading, vertical position, or velocity is set by early XY.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

import study as shared

# Frozen operational requirements, not universal physical observability constants.
CHECK_INTERVAL_S = 0.05
RANGE_STD_FLOOR_M = 0.30
POSITION_STD_FLOOR_M = 0.50
UNMODELED_ACCEL_MPS2 = 3.0
MAX_95_HORIZONTAL_RADIUS_M = 2.0
CHI2_95_2D = 5.991464547
MAX_NORMALIZED_RMS = 2.0
MAX_SPEED_MPS = 8.0
MIN_DISTINCT_ANCHORS = 4
MIRROR_XY_TOLERANCE_M = 1.0
RANK_TOLERANCE = 1e-8
# Continuous is a no-silence reference; isolated shares the complete fixed state from 27 s.
METHODS = ("fixed_2s_pv", "native_gate", "early_only_live", "early_common27", "continuous")
HERE = Path(__file__).resolve().parent


def horizontal_candidate(data, start, end, active, model, core):
    """Eliminate z and all velocity nuisance directions before assessing horizontal use."""
    rows, ids = shared.range_events(data, start, end, active)
    report = dict(time_s=float(data["time"][end]), events=len(rows),
                  anchors=int(len(np.unique(ids))), accepted=False)
    if len(np.unique(ids)) < MIN_DISTINCT_ANCHORS:
        return None, dict(report, reason="insufficient_anchors")
    tau = data["time"][rows] - data["time"][end]
    anchors = data["anchors"][ids]
    ranges = data["ranges"][rows, ids] - np.asarray(model["range_bias"])[ids]
    # Absolute radio/model scales prevent tiny fitted noise from claiming availability.
    sigma = np.sqrt(np.maximum(np.asarray(model["range_sigma"])[ids], RANGE_STD_FLOOR_M)**2
                    + (0.5 * UNMODELED_ACCEL_MPS2 * tau**2)**2)
    estimate, residual, jacobian = shared.cv_candidate(tau, anchors, ranges, sigma)
    # Explicit reflected seed detects relevant z-mirror ambiguity without using truth.
    mirror = estimate.copy()
    mirror[2] = 2 * np.mean(anchors[:, 2]) - mirror[2]
    mirror[5] *= -1

    def residual_at(parameters):
        """Return radio-only CV residuals at actual packet times."""
        return (np.linalg.norm(parameters[:3] + tau[:, None]*parameters[3:] - anchors, axis=1)-ranges)/sigma

    alternative = least_squares(residual_at, mirror, loss="huber", f_scale=2.0, max_nfev=80)
    # Compare actual robust objectives, retaining plausible reflected solutions.
    primary_cost = float(np.sum(np.where(np.abs(residual)<=2, residual**2, 4*np.abs(residual)-4)))
    comparable_mirror = 2*alternative.cost <= primary_cost + 2.0
    mirror_xy = float(np.linalg.norm(alternative.x[:2]-estimate[:2]))
    weights = np.minimum(1.0, 2.0/np.maximum(np.abs(residual), 1e-12))
    weighted = jacobian * np.sqrt(weights[:, None])
    nuisance = weighted[:, [2,3,4,5]]
    left, singular, _ = np.linalg.svd(nuisance, full_matrices=False)
    rank = int(np.sum(singular > (singular[0]*RANK_TOLERANCE))) if singular.size else 0
    target = weighted[:, :2]
    projected = target - left[:, :rank] @ (left[:, :rank].T @ target)
    info = shared.symmetrize(projected.T @ projected)
    eigenvalues = np.linalg.eigvalsh(info)
    report.update(nuisance_rank=rank, horizontal_information_eigenvalues=eigenvalues.tolist(),
        normalized_rms=float(np.sqrt(np.mean(residual**2))), speed_mps=float(np.linalg.norm(estimate[3:])),
        mirror_xy_separation_m=mirror_xy, comparable_mirror=bool(comparable_mirror),
        fitted_z_m=float(estimate[2]), fitted_velocity=estimate[3:].tolist(),
        anchor_height_span_m=float(np.ptp(anchors[:,2])))
    if eigenvalues[0] <= max(eigenvalues[-1], 1.0)*RANK_TOLERANCE:
        return None, dict(report, reason="horizontal_rank_deficient")
    covariance = np.linalg.inv(info) + np.eye(2)*POSITION_STD_FLOOR_M**2
    radius = float(np.sqrt(CHI2_95_2D*np.linalg.eigvalsh(covariance)[-1]))
    prior = core.physical_covariance()[6:8,6:8]
    advantage = float(np.linalg.eigvalsh(shared.symmetrize(prior-covariance))[0])
    report.update(horizontal_covariance=covariance.tolist(), radius_95_m=radius,
                  prior_advantage_min_eigenvalue_m2=advantage,
                  position_xy=estimate[:2].tolist())
    reason = "accepted"
    if report["normalized_rms"] > MAX_NORMALIZED_RMS:
        reason = "radio_model_mismatch"
    elif report["speed_mps"] > MAX_SPEED_MPS:
        reason = "velocity_nuisance_implausible"
    elif comparable_mirror and mirror_xy > MIRROR_XY_TOLERANCE_M:
        reason = "horizontal_mirror_ambiguity"
    elif radius > MAX_95_HORIZONTAL_RADIUS_M:
        reason = "horizontal_usage_requirement"
    elif advantage <= 0:
        reason = "not_better_than_prediction"
    report.update(reason=reason, accepted=reason=="accepted")
    return (estimate[:2], covariance) if report["accepted"] else None, report


def handoff_xy(core, candidate, model):
    """Replace XY only and remove unsupported cross-information in physical coordinates."""
    position, covariance = candidate
    state, physical = core.X.copy(), core.physical_covariance()
    retained = np.r_[0:6,8:15]
    old_retained = physical[np.ix_(retained,retained)].copy()
    old_rotation, old_bias = state[:3,:3].copy(), core.bias.copy()
    old_pz, old_velocity = float(state[2,4]), state[:3,3].copy()
    state[:2,4] = position
    physical[6:8,:] = 0; physical[:,6:8] = 0
    physical[6:8,6:8] = covariance
    result = shared.tracker(state, physical, core.bias.copy(), model)
    checks = dict(rotation=float(np.max(np.abs(result.X[:3,:3]-old_rotation))),
        bias=float(np.max(np.abs(result.bias-old_bias))),
        velocity=float(np.max(np.abs(result.X[:3,3]-old_velocity))),
        vertical_position=abs(float(result.X[2,4])-old_pz),
        retained_covariance=float(np.max(np.abs(result.physical_covariance()[np.ix_(retained,retained)]-old_retained))))
    return result, checks


def run_case(job):
    """Replay common prefix, causal early evidence, and continuous native alternatives."""
    name, layout, yaw, cache, output = job
    cache, output = Path(cache), Path(output)
    data = dict(np.load(cache/(Path(name).stem+".npz")))
    model = json.loads((cache/"protocol.json").read_text(encoding="utf-8"))["model"]
    stamps = data["time"]
    k15,k25,k27 = [int(np.searchsorted(stamps,value)) for value in (15.,25.,27.)]
    active = np.ones(len(data["anchors"]),dtype=bool)
    if layout=="lower4":
        active[:] = False
        active[np.argsort(data["anchors"][:,2])[:4]] = True
    positive = data["changed"] & (data["ranges"]>0)
    outputs = {method:np.full((len(stamps),6),np.nan) for method in METHODS}
    rotations = {method:np.full((len(stamps),3,3),np.nan) for method in METHODS}
    biases = {method:np.full((len(stamps),6),np.nan) for method in METHODS}
    core, branches = None, None
    attempts, checks = [], {}
    trigger, next_check = None, 25.0
    start_time = time.perf_counter()
    counts = {method:0 for method in METHODS}
    common_prefix_error = 0.0
    for k,stamp in enumerate(stamps):
        if k < k15:
            if core is not None and k:
                core.predict(data["imu"][k-1],float(stamp-stamps[k-1]))
            startup = False
            if core is None and stamp>=2.0:
                estimate = shared.solve_window(*shared.window_data(data,0,k,model),method="profile")
                core = shared.tracker_from_estimate(estimate,data["rotations"][k],data["bias"],model)
                startup = True
            if core is not None:
                if not startup:
                    for anchor in np.flatnonzero(positive[k]&active):
                        core.correct_twr(data["anchors"][anchor],data["ranges"][k,anchor],int(anchor))
                for method in METHODS:
                    outputs[method][k] = np.r_[core.X[:3,4],core.X[:3,3]]
                    rotations[method][k],biases[method][k] = core.X[:3,:3],core.bias
            continue
        if branches is None:
            branches = {method:shared.clone(core,model) for method in METHODS if method!="early_common27"}
        for method,branch in branches.items():
            branch.predict(data["imu"][k-1],float(stamp-stamps[k-1]))
            if k==k15:
                physical = branch.physical_covariance()
                branch.X[:3,:3] = shared.yaw_rotation(np.deg2rad(yaw))@branch.X[:3,:3]
                chart = shared.coordinate_map(branch.X)
                branch.P = shared.symmetrize(chart@physical@chart.T)
        # Pending early evidence has never been scalar-updated into this branch.
        triggered_now = False
        if k>=k25 and trigger is None and stamp>=next_check and np.any(positive[k]&active):
            # A rolling two-second buffer limits CV mismatch if short-window evidence remains weak.
            window_start = max(k25,int(np.searchsorted(stamps,stamp-2.0)))
            candidate, report = horizontal_candidate(data,window_start,k,active,model,branches["early_only_live"])
            attempts.append(report)
            next_check = float(stamp+CHECK_INTERVAL_S)
            if candidate is not None:
                branches["early_only_live"], checks = handoff_xy(branches["early_only_live"],candidate,model)
                trigger = float(stamp)
                triggered_now = True
        for method,branch in list(branches.items()):
            if method!="continuous" and stamp<25.0:
                blocked = True
            elif method=="fixed_2s_pv" and k<k27:
                blocked = True
            elif method=="early_only_live" and (trigger is None or triggered_now):
                blocked = True
            else:
                blocked = False
            if method=="fixed_2s_pv" and k==k27:
                branch, fixed_handoff = shared.position_handoff(branch,data,k25,k27,active,model)
                branches[method] = branch
                blocked = True
            if not blocked:
                for anchor in np.flatnonzero(positive[k]&active):
                    branch.correct_twr(data["anchors"][anchor],data["ranges"][k,anchor],int(anchor))
                    counts[method] += 1
            outputs[method][k] = np.r_[branch.X[:3,4],branch.X[:3,3]]
            rotations[method][k],biases[method][k] = branch.X[:3,:3],branch.bias
        # The diagnostic isolation branch alone copies the fixed complete state at 27 s.
        reference = "early_only_live" if k<k27 else "fixed_2s_pv"
        outputs["early_common27"][k] = outputs[reference][k]
        rotations["early_common27"][k] = rotations[reference][k]
        biases["early_common27"][k] = biases[reference][k]
    truth = np.load(cache/(Path(name).stem+"_scoring.npz"))["position"]
    identifier = Path(name).stem+"__"+layout+"__yaw"+str(int(yaw))
    payload = dict(time=stamps,truth=truth,active_anchors=np.flatnonzero(active))
    metrics=[]
    for method in METHODS:
        payload[method],payload[method+"_R"],payload[method+"_bias"] = outputs[method],rotations[method],biases[method]
        error = outputs[method][:,:3]-truth
        xy = np.linalg.norm(error[:,:2],axis=1)
        record = dict(sequence=name,layout=layout,yaw_offset_deg=yaw,method=method,
            trigger_time_s=trigger,trigger_delay_s=None if trigger is None else trigger-25.,
            triggered=trigger is not None,finite=bool(np.all(np.isfinite(outputs[method][stamps>=2.1]))))
        for label,mask in dict(early=(stamps>=25)&(stamps<27),fixed=(stamps>=31)&(stamps<36),
                               post=stamps>=25,gap=(stamps>=15)&(stamps<25)).items():
            record[label+"_xy_rmse_m"] = float(np.sqrt(np.mean(xy[mask]**2)))
            record[label+"_xy_peak_m"] = float(np.max(xy[mask]))
            record[label+"_z_rmse_m"] = float(np.sqrt(np.mean(error[mask,2]**2)))
        metrics.append(record)
    prefix = (stamps>=2.1)&(stamps<15)
    for method in METHODS:
        common_prefix_error = max(common_prefix_error,float(np.max(np.abs(outputs[method][prefix]-outputs["fixed_2s_pv"][prefix]))))
    np.savez_compressed(output/"histories"/(identifier+".npz"),**payload)
    shared.write_json(output/"histories"/(identifier+".json"),dict(sequence=name,layout=layout,
        yaw_offset_deg=yaw,trigger_time_s=trigger,trigger_delay_s=None if trigger is None else trigger-25,
        attempts=attempts,handoff_preservation=checks,common_prefix_error=common_prefix_error,
        fixed_handoff=fixed_handoff,update_counts=counts,elapsed_s=time.perf_counter()-start_time))
    return metrics


def summarize(output):
    """Save paired scores and verify physical handoff/prefix invariants for all cases."""
    frame = pd.read_csv(output/"metrics.csv")
    keys = ["sequence","layout","yaw_offset_deg"]
    base = frame[frame.method=="fixed_2s_pv"].set_index(keys)
    for label in ("early_xy_rmse_m","early_xy_peak_m","fixed_xy_rmse_m","post_xy_rmse_m"):
        frame[label+"_change_vs_fixed_m"] = [r[label]-base.loc[tuple(r[key] for key in keys),label] for _,r in frame.iterrows()]
    frame.to_csv(output/"paired_metrics.csv",index=False)
    frame.groupby(["layout","method"]).agg(cases=("sequence","size"),
        early_xy_median_m=("early_xy_rmse_m","median"),early_xy_worst_m=("early_xy_rmse_m","max"),
        early_peak_worst_m=("early_xy_peak_m","max"),future_xy_median_m=("fixed_xy_rmse_m","median"),
        future_xy_worst_m=("fixed_xy_rmse_m","max"),early_delta_median_m=("early_xy_rmse_m_change_vs_fixed_m","median"),
        future_delta_median_m=("fixed_xy_rmse_m_change_vs_fixed_m","median")).to_csv(output/"aggregate.csv")
    diagnostics=[]; attempts=[]; max_checks={}
    for path in sorted((output/"histories").glob("*.json")):
        item=json.loads(path.read_text(encoding="utf-8"))
        identity={key:item[key] for key in keys}
        diagnostics.append(dict(identity,trigger_time_s=item["trigger_time_s"],trigger_delay_s=item["trigger_delay_s"],
            checks=len(item["attempts"]),last_reason=item["attempts"][-1]["reason"] if item["attempts"] else "no_check"))
        attempts.extend(dict(identity,**trial) for trial in item["attempts"])
        for key,value in item["handoff_preservation"].items():
            max_checks[key]=max(max_checks.get(key,0),value)
        assert item["common_prefix_error"]==0
        data=np.load(path.with_suffix(".npz")); stamps=data["time"]
        after=stamps>=27
        assert np.array_equal(data["early_common27"][after],data["fixed_2s_pv"][after])
        assert np.array_equal(data["early_common27_R"][after],data["fixed_2s_pv_R"][after])
    assert frame.finite.all() and all(value<1e-8 for value in max_checks.values())
    pd.DataFrame(diagnostics).to_csv(output/"triggers.csv",index=False)
    pd.DataFrame(attempts).to_csv(output/"attempts.csv",index=False)
    shared.write_json(output/"validation.json",dict(cases=len(diagnostics),trajectories=len(frame),
        handoff_preservation_maxima=max_checks,all_finite=True,common_prefix_exact=True,
        isolated_future_exact=True,evaluation_yaw_truth_used=False,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    colors={"native_gate":"#0072b2","early_only_live":"#009e73","continuous":"#cc79a7"}
    for row,layout in enumerate(("all8","lower4")):
        for column,label in enumerate(("early_xy_rmse_m","fixed_xy_rmse_m")):
            ax=axes[row,column]; subset=frame[frame.layout==layout]
            for method,color in colors.items():
                part=subset[subset.method==method]
                x=[base.loc[tuple(r[key] for key in keys),label] for _,r in part.iterrows()]
                ax.scatter(x,part[label],label=method,color=color,s=23,alpha=.8)
            limit=max(ax.get_xlim()[1],ax.get_ylim()[1]); ax.plot([0,limit],[0,limit],"k--",lw=.8)
            ax.set(title=layout+" / "+("25–27 s" if column==0 else "31–36 s"),
                   xlabel="Fixed 2 s PV baseline XY RMSE (m)",ylabel="Alternative XY RMSE (m)")
            ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    fig.savefig(output/"paired_position.png",dpi=160); plt.close(fig)


def main():
    """Freeze this protocol before running the complete independent early-position matrix."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers",type=int,default=6)
    parser.add_argument("--sequence",default="")
    parser.add_argument("--layouts",nargs="+",default=["all8","lower4"])
    parser.add_argument("--yaw",nargs="+",type=float,default=shared.YAW_OFFSETS_DEG)
    parser.add_argument("--summarize-only",action="store_true")
    args=parser.parse_args()
    output=HERE/"early_results"; cache=HERE/"results"/"cache"
    output.mkdir(exist_ok=True); (output/"histories").mkdir(exist_ok=True)
    if args.summarize_only:
        summarize(output); return
    protocol=json.loads((cache/"protocol.json").read_text(encoding="utf-8"))
    shared.write_json(output/"protocol.json",dict(check_interval_s=CHECK_INTERVAL_S,
        range_std_floor_m=RANGE_STD_FLOOR_M,position_std_floor_m=POSITION_STD_FLOOR_M,
        unmodeled_acceleration_mps2=UNMODELED_ACCEL_MPS2,max_95_horizontal_radius_m=MAX_95_HORIZONTAL_RADIUS_M,
        threshold_meaning="Horizontal usage requirement / reference operating point, not universal physical constant",
        maximum_normalized_rms=MAX_NORMALIZED_RMS,maximum_speed_mps=MAX_SPEED_MPS,
        minimum_distinct_anchors=MIN_DISTINCT_ANCHORS,mirror_xy_tolerance_m=MIRROR_XY_TOLERANCE_M,
        target="XY",nuisance="z,vx,vy,vz projected with rank-aware SVD; no pseudo-inverse null covariance",
        handoff="XY only; retain z,v,R,bias; clear XY cross covariance; no same-packet scalar reuse",
        early_after_handoff="Only later packets scalar-updated; no 27s reset",
        future="All actual post27 ranges used in every branch; no explicit heading fit or overwrite",
        isolated="Copy complete fixed baseline state at27, only diagnostic attribution of first2s benefit",
        continuous="Same world-z mean perturbation at15; radio remains available15–25",
        methods=METHODS,evaluation_yaw_truth_used=False,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    jobs=[(name,layout,yaw,str(cache),str(output)) for name in protocol["evaluation"] if args.sequence in name
          for layout in args.layouts for yaw in args.yaw]
    rows=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending={pool.submit(run_case,job):job for job in jobs}
        for number,future in enumerate(as_completed(pending),1):
            rows.extend(future.result())
            pd.DataFrame(rows).to_csv(output/"metrics.csv",index=False)
            print(f"DONE {number}/{len(jobs)} {pending[future][:3]}",flush=True)
    summarize(output)


if __name__=="__main__":
    main()
