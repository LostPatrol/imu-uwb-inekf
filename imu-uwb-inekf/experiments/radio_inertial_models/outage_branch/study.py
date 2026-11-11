"""Matched Delft outage replay with retained joint inertial priors.

Evaluation truth is isolated to scoring; calibration uses earlier paired logs.
The MAP window uses deterministic IMU preintegration, an explicit approximation.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from scipy.linalg import solve_triangular
from scipy.optimize import least_squares

# Limit nested BLAS workers when replaying independent recordings.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
# Source modules implement the current common physical 15-state model.
HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parents[1]
sys.path.insert(0, str(EXPERIMENTS / "roadmap_stage1"))
sys.path.insert(0, str(EXPERIMENTS / "recovery_bootstrap"))
from run import (DATASET, MOTIONS, SENSOR_DELAY_S, INITIAL_IMU_PREFIX_S,
                 calibration, estimate_initial_rotation_and_bias, load_real_log,
                 load_anchors, recovery_metrics, write_json)
from invariant15 import Invariant15, coordinate_map
from eskf15 import nominal_step, discrete_error_matrices, rotation_terms, symmetrize
from run_research import preintegrate, window_data, tracker_from_estimate
from initializer import solve_window

# The matched outage and acquisition durations are fixed before scoring.
OUTAGE = (15.0, 25.0)
WINDOW_S = 2.0
# All methods use the same robust radio policy after startup.
POLICY = dict(gate_threshold=12.25, gate_warmup_updates=0, huber_threshold=2.0)
# Range residuals alone receive Huber robustification; prior residuals stay Gaussian.
HUBER = 2.0
# This motion prior is a deliberately simple indoor-hover comparator, not sensing.
DAMPING_TIME_S = 2.0
# A radio-silence decision uses only elapsed time since the last actual arrival.
GAP_DETECT_S = 0.25


def prepare(cache):
    """Rebuild sensor caches and training-only radio calibration from raw data."""
    cache.mkdir(parents=True, exist_ok=True)
    anchors = load_anchors(DATASET)
    names = sorted(p.name for p in (DATASET / "twr").glob("*.csv"))
    training, evaluation = [], []
    for motion in MOTIONS:
        pair = [n for n in names if n.endswith(f"_{motion}.csv")]
        training.append(pair[0])
        evaluation.append(pair[1])
    logs = {n: load_real_log(DATASET, n, sensor_delay_s=SENSOR_DELAY_S) for n in training}
    model = calibration(logs, anchors, training)
    for name in evaluation:
        log = load_real_log(DATASET, name, sensor_delay_s=SENSOR_DELAY_S)
        rotation, bias = estimate_initial_rotation_and_bias(log)
        selected = log.time >= INITIAL_IMU_PREFIX_S
        stamps = log.time[selected] - log.time[selected][0]
        imu = log.imu[selected]
        rotations, displacement, velocity = preintegrate(stamps, imu, rotation, bias)
        np.savez_compressed(cache / (Path(name).stem + ".npz"), time=stamps,
            imu=imu, ranges=log.ranges[selected], changed=log.changed[selected],
            rotations=rotations, displacement=displacement, integrated_velocity=velocity,
            bias=bias, anchors=anchors)
        np.savez_compressed(cache / (Path(name).stem + "_scoring.npz"),
            position=log.ground_truth_position[selected], velocity=log.ground_truth_velocity[selected])
    write_json(cache / "protocol.json", dict(training=training, evaluation=evaluation,
        model=model, sensor_delay_s=SENSOR_DELAY_S, initial_prefix_s=INITIAL_IMU_PREFIX_S,
        outage_s=OUTAGE, acquisition_s=WINDOW_S, radio_policy=POLICY,
        evaluation_truth_in_estimator=False, training_truth_for_radio_calibration=True,
        truth_used_for_available_interval_and_shared_delay=True, independent_test_date=False,
        map_approximation="Deterministic 2 s IMU window; ignores temporal process correlation; adds endpoint process noise after curvature update"))
    protocol_path = cache / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol["dataset_sha256"] = {n:hashlib.sha256((DATASET/"twr"/n).read_bytes()).hexdigest()
                                  for n in training+evaluation}
    protocol["anchors_sha256"] = hashlib.sha256((DATASET/"anchor_positions.yaml").read_bytes()).hexdigest()
    write_json(protocol_path, protocol)


def make_tracker(state, physical, bias, model):
    """Convert one physical joint covariance to the invariant chart."""
    chart = coordinate_map(state)
    return Invariant15(state, chart @ physical @ chart.T, bias,
                       range_bias=model["range_bias"], range_sigma=model["range_sigma"], **POLICY)


def map_recapture(data, start, end, prior, model, diagonal_prior=False):
    """Fit all 15 initial errors to a prior and each new range exactly once."""
    times = data["time"]
    rows, ids = np.nonzero(data["changed"][start:end+1] & (data["ranges"][start:end+1] > 0))
    rows = rows + start
    ranges = data["ranges"][rows, ids] - np.asarray(model["range_bias"])[ids]
    sigmas = np.asarray(model["range_sigma"])[ids]
    covariance = prior.physical_covariance()
    if diagonal_prior:
        covariance = np.diag(np.diag(covariance))
    chol = np.linalg.cholesky(symmetrize(covariance) + 1e-10*np.eye(15))
    whitening = solve_triangular(chol, np.eye(15), lower=True)
    initial = prior.X.copy()
    initial_bias = prior.bias.copy()
    event_rows = {k: np.flatnonzero(rows == k) for k in np.unique(rows)}
    memo = {}

    def evaluate(delta):
        """Propagate a physical perturbation and exact held-sample sensitivities."""
        if "delta" in memo and np.array_equal(delta, memo["delta"]):
            return memo["result"]
        state = initial.copy()
        state[:3, :3] = state[:3, :3] @ rotation_terms(delta[:3])[0]
        state[:3, 3] += delta[3:6]
        state[:3, 4] += delta[6:9]
        bias = initial_bias + delta[9:15]
        mapping = np.eye(15)
        mapping[:3, :3] = rotation_terms(delta[:3])[1]
        residual = np.empty(15 + len(rows))
        jacobian = np.empty((len(residual), 15))
        residual[:15], jacobian[:15] = whitening @ delta, whitening
        noise_covariance = np.zeros((15, 15))
        variances = np.r_[prior.w_gyr, prior.w_acc, prior.w_gyr_b, prior.w_acc_b]
        for k in range(start, end+1):
            if k > start:
                dt = float(times[k]-times[k-1])
                transition, noise = discrete_error_matrices(state, bias, data["imu"][k-1], dt)
                mapping = transition @ mapping
                noise_covariance = transition @ noise_covariance @ transition.T + (noise*variances) @ noise.T
                state = nominal_step(state, bias, data["imu"][k-1], dt)
            if k in event_rows:
                indices = event_rows[k]
                difference = state[:3, 4] - data["anchors"][ids[indices]]
                distance = np.maximum(np.linalg.norm(difference, axis=1), 1e-12)
                residual[15+indices] = (distance-ranges[indices])/sigmas[indices]
                jacobian[15+indices] = (difference/distance[:, None]) @ mapping[6:9] / sigmas[indices, None]
        result = residual, jacobian, state, bias, mapping, noise_covariance
        memo["delta"], memo["result"] = delta.copy(), result
        return result

    def loss(squared):
        """Keep a Gaussian joint prior while robustifying only radio residuals."""
        output = np.array([squared, np.ones_like(squared), np.zeros_like(squared)])
        tail = squared[15:]
        outside = tail > HUBER**2
        root = np.sqrt(np.maximum(tail, 1e-30))
        output[0, 15:] = np.where(outside, 2*HUBER*root-HUBER**2, tail)
        output[1, 15:] = np.where(outside, HUBER/root, 1.0)
        output[2, 15:] = np.where(outside, -0.5*HUBER/root**3, 0.0)
        return output

    started = time.perf_counter()
    # SciPy robust loss modifies arrays in place; preserve the cached raw values.
    fit = least_squares(lambda x: evaluate(x)[0].copy(), np.zeros(15), jac=lambda x: evaluate(x)[1].copy(),
                        loss=loss, max_nfev=45, ftol=1e-6, xtol=1e-6, gtol=1e-6)
    residual, jacobian, state, bias, mapping, noise = evaluate(fit.x)
    weights = np.r_[np.ones(15), np.minimum(1.0, HUBER/np.maximum(np.abs(residual[15:]), 1e-12))]
    posterior = np.linalg.inv((jacobian.T*weights)@jacobian)
    physical = symmetrize(mapping @ posterior @ mapping.T + noise)
    tracker = make_tracker(state, physical, bias, model)
    diagnostics = dict(elapsed_s=time.perf_counter()-started, nfev=fit.nfev,
        success=bool(fit.success), normalized_range_rms=float(np.sqrt(np.mean(residual[15:]**2))),
        delta=fit.x.tolist(), prior_condition=float(np.linalg.cond(covariance)), ranges=len(rows))
    if os.environ.get("INEKF_MAP_JAC_CHECK") == "1":
        numeric = np.empty_like(jacobian)
        for column in range(15):
            perturbation = np.zeros(15)
            perturbation[column] = 1e-6
            numeric[:, column] = (evaluate(fit.x+perturbation)[0] - evaluate(fit.x-perturbation)[0])/(2e-6)
        diagnostics["jacobian_max_absolute_difference"] = float(np.max(np.abs(numeric-jacobian)))
        diagnostics["jacobian_relative_frobenius_difference"] = float(np.linalg.norm(numeric-jacobian)/np.linalg.norm(numeric))
    return tracker, diagnostics


def damped_gap_step(tracker, imu, dt):
    """Apply a bounded indoor-hover motion prior during missing UWB only."""
    position, velocity = tracker.X[:3, 4].copy(), tracker.X[:3, 3].copy()
    physical = tracker.physical_covariance()
    transition, noise = discrete_error_matrices(tracker.X, tracker.bias, imu, dt)
    decay = np.exp(-dt/DAMPING_TIME_S)
    # Translation follows an uncertain damped-velocity prior during radio silence.
    transition[3:9] = 0.0
    transition[3:6, 3:6] = decay*np.eye(3)
    transition[6:9, 3:6] = DAMPING_TIME_S*(1-decay)*np.eye(3)
    transition[6:9, 6:9] = np.eye(3)
    noise[3:9] = 0.0
    noise[3:6, 3:6] = dt*np.eye(3)
    noise[6:9, 3:6] = 0.5*dt**2*np.eye(3)
    variances = np.r_[tracker.w_gyr, np.full(3, 2.0**2), tracker.w_gyr_b, tracker.w_acc_b]
    physical = symmetrize(transition@physical@transition.T + (noise*variances)@noise.T)
    tracker.predict(imu, dt)
    tracker.set_translation(velocity*decay, position+DAMPING_TIME_S*(1-decay)*velocity)
    chart = coordinate_map(tracker.X)
    tracker.P = symmetrize(chart@physical@chart.T)


def replay(job):
    """Replay sensor arrays, then load separate truth for paired offline scoring."""
    cache, output, name, method, outage = job
    cache, output = Path(cache), Path(output)
    data = dict(np.load(cache / (Path(name).stem+".npz")))
    model = json.loads((cache / "protocol.json").read_text(encoding="utf-8"))["model"]
    times = data["time"]
    history = np.full((len(times), 6), np.nan)
    tracker, capture_start, prior = None, 0, None
    capture_event, last_arrival = None, None
    captures = []
    accepted, rejected = np.zeros(len(times), dtype=int), np.zeros(len(times), dtype=int)
    started = time.perf_counter()
    for k, stamp in enumerate(times):
        blocked = outage and OUTAGE[0] <= stamp < OUTAGE[1]
        usable = data["changed"][k] & (data["ranges"][k] > 0)
        events = np.array([], dtype=int) if blocked else np.flatnonzero(usable)
        # Detect recovery from actual arrivals, without consulting the mask boundary.
        range_resumed = len(events) and last_arrival is not None and stamp-last_arrival > 1.0
        if tracker is not None and k:
            dt = float(stamp-times[k-1])
            gap_detected = not len(events) and last_arrival is not None and stamp-last_arrival > GAP_DETECT_S
            if method == "damped_gap" and gap_detected:
                damped_gap_step(tracker, data["imu"][k-1], dt)
            else:
                tracker.predict(data["imu"][k-1], dt)
        if len(events):
            last_arrival = stamp
        if range_resumed:
            if method in ("profile_reset", "profile_reset_live", "joint_map", "diagonal_map", "joint_map_live", "diagonal_map_live"):
                capture_start, capture_event = k, "resume"
                if method in ("joint_map", "diagonal_map", "joint_map_live", "diagonal_map_live"):
                    prior = make_tracker(tracker.X.copy(), tracker.physical_covariance(), tracker.bias.copy(), model)
        handed = False
        if capture_start is not None and stamp-times[capture_start] >= WINDOW_S:
            if capture_event == "resume" and method in ("joint_map", "diagonal_map", "joint_map_live", "diagonal_map_live"):
                tracker, detail = map_recapture(data, capture_start, k, prior, model, method.startswith("diagonal"))
            else:
                estimate = solve_window(*window_data(data, capture_start, k, model), method="profile")
                tracker = tracker_from_estimate(estimate, data["rotations"][k], data["bias"], model)
                detail = dict(elapsed_s=estimate.elapsed_s, normalized_range_rms=estimate.residual_rms)
            captures.append(dict(time_s=float(stamp), event=capture_event or "startup", **detail))
            capture_start, handed = None, True
        live_capture = capture_event == "resume" and method.endswith("_live")
        if tracker is not None and (capture_start is None or live_capture) and not handed:
            for anchor in events:
                success = tracker.correct_twr(data["anchors"][anchor], data["ranges"][k, anchor], int(anchor))
                accepted[k] += int(success)
                rejected[k] += int(not success)
        if tracker is not None:
            history[k] = np.r_[tracker.X[:3, 4], tracker.X[:3, 3]]
    elapsed = time.perf_counter()-started
    truth = dict(np.load(cache / (Path(name).stem+"_scoring.npz")))
    error = np.linalg.norm(history[:, :3]-truth["position"], axis=1)
    available = np.all(np.isfinite(history), axis=1)
    row = dict(sequence=name, method=method, outage=outage, elapsed_s=elapsed,
               finite_after_start=bool(np.all(available[times>=2.1])), captures=captures,
               accepted=int(accepted.sum()), rejected=int(rejected.sum()))
    masks = dict(common=times>=8.1, pre=(times>=8.1)&(times<15),
                 gap=(times>=15)&(times<25), resume2=(times>=25)&(times<27.1),
                 post=times>=25, post_handoff=times>=27.1, tail=times>=times[-1]-5)
    for label, mask in masks.items():
        selected = mask & available
        row[label+"_rmse_m"] = float(np.sqrt(np.mean(error[selected]**2)))
        row[label+"_max_m"] = float(np.max(error[selected]))
    row["end_gap_error_m"] = float(error[np.searchsorted(times, 25)-1])
    row["post_accepted"] = int(accepted[times>=25].sum())
    row["post_rejected"] = int(rejected[times>=25].sum())
    row.update(recovery_metrics(times, error, origin=25.0 if outage else 0.0))
    identifier = Path(name).stem+"__"+method+"__"+str(int(outage))
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / (identifier+".npz"), time=times, history=history,
                        truth=truth["position"], error=error, accepted=accepted, rejected=rejected)
    write_json(output / (identifier+".json"), row)
    row.pop("captures")
    return row


def main():
    """Run independent matched cases with bounded process-level parallelism."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE/"results")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--append-metrics", action="store_true")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--sequence", default="")
    parser.add_argument("--methods", nargs="+", default=["profile_reset", "profile_reset_live", "continuous", "joint_map", "diagonal_map", "joint_map_live", "diagonal_map_live", "damped_gap"])
    args = parser.parse_args()
    cache = args.output / "cache"
    if args.prepare or args.prepare_only:
        prepare(cache)
    if args.prepare_only:
        return
    protocol = json.loads((cache / "protocol.json").read_text(encoding="utf-8"))
    jobs = [(str(cache),str(args.output/"histories"),n,m,o) for n in protocol["evaluation"]
            if args.sequence in n for m in args.methods for o in (False,True)]
    metrics_path = args.output/"metrics.csv"
    rows = pd.read_csv(metrics_path).to_dict("records") if args.append_metrics and metrics_path.exists() else []
    existing = {(r["sequence"], r["method"], r["outage"]) for r in rows}
    jobs = [j for j in jobs if (j[2],j[3],j[4]) not in existing]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(replay, j):j for j in jobs}
        for completed, future in enumerate(as_completed(futures), 1):
            rows.append(future.result())
            pd.DataFrame(rows).to_csv(args.output/"metrics.csv", index=False)
            print(f"DONE {completed}/{len(jobs)} {futures[future][2:]} {rows[-1]['elapsed_s']:.2f}s", flush=True)


if __name__ == "__main__":
    main()
