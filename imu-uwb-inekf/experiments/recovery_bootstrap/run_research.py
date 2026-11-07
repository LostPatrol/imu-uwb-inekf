"""Parallel, reproducible cold-start/outage and synthetic initialization research.

No estimator receives runtime position truth. Existing six-log calibration is
frozen; old logs remain development data. Full output trajectories are retained.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

# Import the frozen, previously audited common 15-state implementation.
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent / "roadmap_stage1"))
from run import (load_real_log, load_anchors, estimate_initial_rotation_and_bias,
                 SENSOR_DELAY_S, INITIAL_IMU_PREFIX_S, recovery_metrics, write_json,
                 DATASET, PRIOR_STD, SETTINGS, CaptureThenTrack, LiftedTWRInEKF, handoff_covariance)
from eskf15 import rotation_terms
from invariant15 import Invariant15, coordinate_map
from initializer import solve_window, yaw_rotation, raw_residual_jacobian, project_out

# Existing frozen results supply calibration, never evaluation-position estimates.
FROZEN = PROJECT / "results" / "roadmap_stage1_20261104_final" / "protocol.json"
# Runtime range gaps larger than one second request recapture on the next event.
GAP_TRIGGER_S = 1.0
# Candidate windows and provisional data-only yaw precision target.
WINDOWS_S = (2.0, 4.0, 8.0)
YAW_STD_TARGET_RAD = 0.35
# A broad model-mismatch check; no truth-based acceptance gate is permitted.
MAX_NORMALIZED_RMS = 3.0
# Physical covariance floors acknowledge unmodeled preintegration and calibration error.
POSITION_FLOOR_M = 0.15
VELOCITY_FLOOR_MPS = 0.20
YAW_FLOOR_RAD = 0.10
# Shared evaluation time excludes every candidate's maximum startup latency.
COMMON_SCORE_START_S = 8.1


def preintegrate(times, imu, rotation, bias):
    """Integrate the same held-sample inertial model with fixed calibrated biases."""
    rotations = np.empty((times.size, 3, 3))
    displacement = np.zeros((times.size, 3))
    velocity = np.zeros_like(displacement)
    rotations[0] = rotation
    gravity = np.array([0, 0, -9.81])
    for k in range(1, times.size):
        dt = times[k]-times[k-1]
        acceleration = rotations[k-1]@(imu[k-1, :3]*9.81-bias[3:])+gravity
        displacement[k] = displacement[k-1]+velocity[k-1]*dt+0.5*acceleration*dt**2
        velocity[k] = velocity[k-1]+acceleration*dt
        rotations[k] = rotations[k-1]@rotation_terms((np.deg2rad(imu[k-1, 3:])-bias[:3])*dt)[0]
    return rotations, displacement, velocity


def prepare(output):
    """Cache sensor-only integration and separately stored truth for offline scoring."""
    protocol = json.loads(FROZEN.read_text(encoding="utf-8"))
    anchors = load_anchors(DATASET)
    output.mkdir(parents=True, exist_ok=True)
    for name in protocol["evaluation"]:
        log = load_real_log(DATASET, name, sensor_delay_s=SENSOR_DELAY_S)
        rotation, bias = estimate_initial_rotation_and_bias(log)
        selected = log.time >= INITIAL_IMU_PREFIX_S
        times = log.time[selected]-log.time[selected][0]
        imu, ranges, changed = log.imu[selected], log.ranges[selected], log.changed[selected]
        rotations, d, v = preintegrate(times, imu, rotation, bias)
        np.savez_compressed(output / (Path(name).stem+".npz"), time=times, imu=imu,
            ranges=ranges, changed=changed, rotations=rotations, displacement=d,
            integrated_velocity=v, bias=bias, anchors=anchors,
            truth_position=log.ground_truth_position[selected], truth_velocity=log.ground_truth_velocity[selected])
    manifest = dict(created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        frozen_protocol_sha256=hashlib.sha256(FROZEN.read_bytes()).hexdigest(),
        evaluation=protocol["evaluation"], model=protocol["range_models"]["holdout"],
        ground_truth_in_estimator=False, truth_used_to_crop_available_evaluation_interval=True,
        calibration_uses_other_logs_truth=True, independent_blind_test=False,
        windows_s=WINDOWS_S, gap_trigger_s=GAP_TRIGGER_S,
        yaw_std_target_rad=YAW_STD_TARGET_RAD, max_normalized_rms=MAX_NORMALIZED_RMS,
        covariance="Local robust curvature plus explicit heuristic floors; not proven calibrated",
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob("*.py")})
    write_json(output / "manifest.json", manifest)


def window_data(data, start, end, model):
    """Collect only changed positive transactions already arrived by the endpoint."""
    times = data["time"]
    rows, anchor_indices = np.nonzero(data["changed"][start:end+1] &
                                      (data["ranges"][start:end+1] > 0))
    rows += start
    tau = times[rows]-times[end]
    # Reanchor the inertial integral at the endpoint without a new integration.
    d = data["displacement"][rows]-data["displacement"][end]-tau[:, None]*data["integrated_velocity"][end]
    return (tau, d, data["anchors"][anchor_indices],
            data["ranges"][rows, anchor_indices]-np.asarray(model["range_bias"])[anchor_indices],
            np.asarray(model["range_sigma"])[anchor_indices])


def tracker_from_estimate(estimate, rotation, bias, model):
    """Map the endpoint's joint p/v/yaw curvature into the invariant error chart."""
    x = estimate.parameters
    state = np.eye(5)
    state[:3, :3] = yaw_rotation(x[6])@rotation
    state[:3, 3], state[:3, 4] = x[3:6], x[:3]
    vertical_body = state[:3, :3].T@np.array([0.0, 0.0, 1.0])
    mapping = np.zeros((15, 7))
    mapping[6:9, :3], mapping[3:6, 3:6] = np.eye(3), np.eye(3)
    mapping[:3, 6] = vertical_body
    physical = mapping@estimate.covariance@mapping.T
    physical[:3, :3] += 0.05**2*(np.eye(3)-np.outer(vertical_body, vertical_body))
    physical[:3, :3] += YAW_FLOOR_RAD**2*np.outer(vertical_body, vertical_body)
    physical[3:6, 3:6] += VELOCITY_FLOOR_MPS**2*np.eye(3)
    physical[6:9, 6:9] += POSITION_FLOOR_M**2*np.eye(3)
    physical[9:12, 9:12] = 0.01**2*np.eye(3)
    physical[12:15, 12:15] = 0.10**2*np.eye(3)
    chart = coordinate_map(state)
    return Invariant15(state, chart@physical@chart.T, bias,
        range_bias=model["range_bias"], range_sigma=model["range_sigma"],
        gate_threshold=12.25, gate_warmup_updates=0, huber_threshold=2.0)


def run_real_case(job):
    """Replay one method/log independently; identify resumption from actual arrivals."""
    cache, output, name, method, outage = job
    data = dict(np.load(Path(cache)/(Path(name).stem+".npz")))
    manifest = json.loads((Path(cache)/"manifest.json").read_text(encoding="utf-8"))
    model = manifest["model"]
    times = data["time"]
    history = np.full((times.size, 6), np.nan)
    yaw_variance = np.full(times.size, np.nan)
    captures, estimates = [], []
    tracker = None
    last_event = None
    capture_start, candidate_index = 0, 0
    started = time.perf_counter()
    baseline = method in ("centroid", "capture", "capture_robust")
    if baseline:
        state = np.eye(5)
        state[:3, :3] = data["rotations"][0]
        state[:3, 4] = data["anchors"].mean(axis=0)
        chart = coordinate_map(state)
        covariance = chart@np.diag(PRIOR_STD**2)@chart.T
        common = dict(range_bias=model["range_bias"], range_sigma=model["range_sigma"], **SETTINGS)
        if method == "centroid":
            tracker = Invariant15(state, covariance, data["bias"], **common)
        else:
            if method == "capture_robust":
                common["gate_warmup_updates"] = 0
            tracker = CaptureThenTrack(state, covariance, data["bias"], reset_on_resume=True,
                tracker_factory=Invariant15, imu_core_factory=Invariant15,
                capture_factory=LiftedTWRInEKF, handoff_covariance_factory=handoff_covariance, **common)
        capture_start = None
    for k, stamp in enumerate(times):
        if tracker is not None and k:
            tracker.predict(data["imu"][k-1], float(stamp-times[k-1]))
        usable = data["changed"][k] & (data["ranges"][k] > 0)
        if outage and 15 <= stamp < 25:
            usable[:] = False
        events = np.flatnonzero(usable)
        if events.size:
            if last_event is not None and stamp-last_event > GAP_TRIGGER_S:
                captures.append(dict(event="gap_resume", time_s=float(stamp), gap_s=float(stamp-last_event)))
                if baseline:
                    if hasattr(tracker, "on_ranges_resumed"):
                        tracker.on_ranges_resumed()
                else:
                    capture_start, candidate_index = k, 0
            last_event = stamp
        handed_off = False
        if capture_start is not None and stamp-times[capture_start] >= WINDOWS_S[candidate_index]:
            estimator_method = "profile" if method == "adaptive" else method
            estimate = solve_window(*window_data(data, capture_start, k, model), method=estimator_method)
            accepted = method != "adaptive" or (estimate.yaw_information >= 1/YAW_STD_TARGET_RAD**2
                                                and estimate.residual_rms <= MAX_NORMALIZED_RMS)
            estimates.append(dict(time_s=float(stamp), window_s=float(stamp-times[capture_start]),
                accepted=accepted, parameters=estimate.parameters, covariance=estimate.covariance,
                yaw_information=estimate.yaw_information, normalized_rms=estimate.residual_rms,
                objective=estimate.robust_objective, elapsed_s=estimate.elapsed_s,
                evaluations=estimate.evaluations, diagnostics=estimate.diagnostics))
            if accepted:
                tracker = tracker_from_estimate(estimate, data["rotations"][k], data["bias"], model)
                capture_start = None
                handed_off = True
            elif candidate_index < len(WINDOWS_S)-1:
                candidate_index += 1
            else:
                capture_start = None
                captures.append(dict(event="no_qualified_handoff", time_s=float(stamp)))
        # Window events have been consumed by initialization and are never replayed.
        if tracker is not None and not handed_off and (capture_start is None or baseline):
            for anchor in events:
                tracker.correct_twr(data["anchors"][anchor], data["ranges"][k, anchor], int(anchor))
        if tracker is not None:
            history[k] = np.r_[tracker.X[:3, 4], tracker.X[:3, 3]]
            if isinstance(tracker, Invariant15):
                physical = tracker.physical_covariance()
                body_vertical = tracker.X[:3, :3].T@np.array([0, 0, 1])
                yaw_variance[k] = body_vertical@physical[:3, :3]@body_vertical
    elapsed = time.perf_counter()-started
    available = np.all(np.isfinite(history), axis=1)
    errors = np.linalg.norm(history[:, :3]-data["truth_position"], axis=1)
    common = times >= COMMON_SCORE_START_S
    tail = times >= times[-1]-5
    score_mask = common & available
    def rmse(mask):
        """Return conditional finite-output RMSE with availability reported separately."""
        return float(np.sqrt(np.mean(errors[mask]**2))) if np.any(mask) else float("nan")
    row = dict(sequence=name, method=method, outage=outage, elapsed_s=elapsed,
        duration_s=float(times[-1]), startup_unavailable_s=float(times[np.flatnonzero(available)[0]]) if available.any() else float(times[-1]),
        availability=float(np.mean(available)), common_availability=float(np.mean(available[common])),
        common_rmse_m=rmse(score_mask), output_rmse_m=rmse(available), tail5_rmse_m=rmse(tail & available),
        p95_m=float(np.percentile(errors[available], 95)) if available.any() else float("nan"),
        max_m=float(np.max(errors[available])) if available.any() else float("nan"),
        handoffs=sum(item["accepted"] for item in estimates), attempts=len(estimates),
        initializer_s=sum(item["elapsed_s"] for item in estimates))
    row.update(recovery_metrics(times, errors, origin=25.0 if outage else 0.0))
    identifier = f"{Path(name).stem}__{method}__outage{int(outage)}"
    Path(output).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(Path(output)/(identifier+".npz"), time=times, output=history,
        truth=data["truth_position"], yaw_variance=yaw_variance)
    write_json(Path(output)/(identifier+".json"), dict(metrics=row, captures=captures, estimates=estimates))
    return row


def synthetic_case(job):
    """Test initialization with exact preintegration and controlled measurement noise."""
    seed, motion, method, contaminated = job
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(-4, 0, 240))
    if motion == "excited":
        d = np.column_stack((0.8*np.sin(1.7*t), 0.6*(np.cos(1.1*t)-1), 0.15*np.sin(0.7*t)))
        endpoint_inertial_velocity = np.array([0.8*1.7, 0.0, 0.15*0.7])
    elif motion == "constant_acceleration":
        d = 0.5*t[:, None]**2*np.array([0.4, -0.3, 0.1])
        endpoint_inertial_velocity = np.zeros(3)
    else:
        d = np.zeros((t.size, 3))
        endpoint_inertial_velocity = np.zeros(3)
    anchors = np.array([[-4,-4,0],[4,-4,0],[-4,4,0],[4,4,0],[-4,-4,3],[4,-4,3],[-4,4,3],[4,4,3]])
    a = anchors[rng.integers(0, 8, t.size)]
    truth = np.r_[rng.uniform(-2,2,2), 1.2, rng.uniform(-0.5,0.5,3), rng.uniform(-np.pi,np.pi)]
    p = truth[:3]+t[:, None]*truth[3:6]+d@yaw_rotation(truth[6]).T
    sigma = np.full(t.size, 0.10)
    ranges = np.linalg.norm(p-a, axis=1)+rng.normal(0, sigma)
    if contaminated:
        ranges[rng.random(t.size)<0.05] += 1.5
    fit = solve_window(t, d, a, ranges, sigma, method=method)
    yaw_error = float(abs(np.arctan2(np.sin(fit.parameters[6]-truth[6]), np.cos(fit.parameters[6]-truth[6]))))
    _, jac = raw_residual_jacobian(truth, t, d, a, ranges, sigma)
    bias_jac = -0.5*t[:, None]**2*jac[:, :3]@yaw_rotation(truth[6])
    residual, _ = project_out(np.column_stack((jac[:, :6], bias_jac)), jac[:, 6:])
    estimated_velocity = fit.parameters[3:6]+yaw_rotation(fit.parameters[6])@endpoint_inertial_velocity
    true_velocity = truth[3:6]+yaw_rotation(truth[6])@endpoint_inertial_velocity
    return dict(seed=seed, motion=motion, method=method, contaminated=contaminated,
        position_error_m=float(np.linalg.norm(fit.parameters[:3]-truth[:3])),
        velocity_error_mps=float(np.linalg.norm(estimated_velocity-true_velocity)),
        coefficient_velocity_error_mps=float(np.linalg.norm(fit.parameters[3:6]-truth[3:6])),
        yaw_error_deg=float(np.rad2deg(yaw_error)), yaw_information=fit.yaw_information,
        yaw_information_free_constant_bias=float(np.sum(residual**2)),
        objective=fit.robust_objective, elapsed_s=fit.elapsed_s, evaluations=fit.evaluations)


def main():
    """Launch bounded process-level parallelism and flush each completed result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=min(20, os.cpu_count() or 1))
    parser.add_argument("--suite", choices=("prepare", "real", "synthetic"), required=True)
    parser.add_argument("--methods", nargs="+", default=["cv","local","multistart","profile","adaptive","centroid","capture","capture_robust"])
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--sequence", default=None)
    args = parser.parse_args()
    cache = args.output/"cache"
    if args.suite == "prepare":
        prepare(cache)
        print("Prepared six evaluation logs", flush=True)
        return
    if args.suite == "real":
        manifest = json.loads((cache/"manifest.json").read_text(encoding="utf-8"))
        names = [name for name in manifest["evaluation"] if args.sequence is None or args.sequence in name]
        jobs = [(str(cache),str(args.output/"histories"),name,method,outage)
                for name in names for method in args.methods for outage in (False,True)]
        worker, destination = run_real_case, args.output/"real_metrics.csv"
    else:
        jobs = [(seed,motion,method,contaminated) for seed in range(args.seeds)
                for motion in ("excited","constant_acceleration","constant_velocity")
                for method in args.methods if method in ("cv","local","multistart","profile")
                for contaminated in (False,True)]
        worker, destination = synthetic_case, args.output/"synthetic_metrics.csv"
    rows = []
    args.output.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        pending = {executor.submit(worker, job):job for job in jobs}
        for future in as_completed(pending):
            job = pending[future]
            try:
                rows.append(future.result())
            except Exception as exc:
                write_json(args.output/"failure.json", dict(job=job, error=repr(exc)))
                raise
            pd.DataFrame(rows).to_csv(destination, index=False)
            if args.suite == "real" or len(rows) % 50 == 0:
                print(f"DONE {len(rows)}/{len(jobs)} {job[2:]}" if args.suite == "real"
                      else f"DONE {len(rows)}/{len(jobs)}", flush=True)


if __name__ == "__main__":
    main()
