"""Freeze and replay a shared IMU/TWR recovery protocol on real TU Delft logs.

Calibration uses other logs or a fixed six-log training split. Runtime truth
is used only for controlled initial position and offline trajectory scoring.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

from eskf15 import ESKF15, rotation_terms
from invariant15 import Invariant15, coordinate_map, handoff_covariance

# Existing research components remain separate from the mature TDoA entry point.
PROJECT = Path(__file__).resolve().parents[2]
# An environment override keeps the same dataset layout portable across computers.
DATASET = Path(os.environ.get("INEKF_TWR_DATASET", r"E:\datasets\tu-delft-uwb-mhe\data_v1\UWB dataset"))
# The existing two-filter implementation now exposes explicit model factories.
DEEP_SOURCE = PROJECT.parent / "imu-uwb-state-augment-deep-coupled-two-filters" / "scripts" / "estimation"
sys.path.insert(0, str(DEEP_SOURCE))
from deep_coupled import DeepCoupledTwoFilter
from lifted_twr import LiftedTWRInEKF
from twr_experiment import load_anchors, load_real_log, estimate_initial_rotation_and_bias
from capture_baseline import CaptureThenTrack

# Six chronological pairs provide a fixed training/evaluation split by motion type.
MOTIONS = ("triangle", "octagon", "hourglass", "random", "star", "square")
# A frozen common calibration policy avoids method-specific parameter fitting.
SETTINGS = dict(gate_threshold=3.5**2, gate_warmup_updates=256, huber_threshold=2.0)
# All 15-state filters begin with the same physical covariance.
PRIOR_STD = np.r_[0.05, 0.05, 2.0, np.full(3, 2.0), np.full(3, 8.0), np.full(3, 0.01), np.full(3, 0.10)]
# Restore the full historical offset and test its opposite and a transverse direction.
DIRECTIONS = np.array([[2.0, -2.0, 1.0], [-2.0, 2.0, -1.0], [1.0, 1.0, 0.0]])
DIRECTIONS /= np.linalg.norm(DIRECTIONS, axis=1)[:, None]
# Recovery has to remain within this distance for the complete remaining record.
RECOVERY_THRESHOLD_M = 0.5
# A terminal crossing with less observed time is censored as unsuccessful.
MIN_STABLE_S = 5.0
# Historical time correction is common to IMU and TWR, not UWB-only compensation.
SENSOR_DELAY_S = 0.20
# The initial static IMU prefix is consumed before starting the scored replay.
INITIAL_IMU_PREFIX_S = 1.5
# A controlled loss of measured ranges probes recovery on actual recorded IMU.
OUTAGE = (15.0, 25.0)
# The robust Gaussian inlier scale is clipped before any evaluation replay.
SIGMA_LIMITS_M = (0.08, 0.50)
# Convert a Gaussian median absolute deviation into its standard deviation.
MAD_FACTOR = 1.4826


def write_json(path, value):
    """Write portable UTF-8 JSON, converting missing scalar values to null."""
    def convert(item):
        """Normalize NumPy values for the evidence files."""
        if isinstance(item, dict):
            return {key: convert(val) for key, val in item.items()}
        if isinstance(item, (tuple, list)):
            return [convert(val) for val in item]
        if isinstance(item, np.ndarray):
            return convert(item.tolist())
        if isinstance(item, (np.integer, np.bool_)):
            return item.item()
        if isinstance(item, (float, np.floating)):
            return float(item) if np.isfinite(item) else None
        return item
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(convert(value), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def calibration(logs, anchors, training):
    """Fit positive asynchronous ranges on training logs alone."""
    biases, sigmas, counts = [], [], []
    for anchor_index, anchor in enumerate(anchors):
        residuals = []
        for name in training:
            log = logs[name]
            mask = log.changed[:, anchor_index] & (log.ranges[:, anchor_index] > 0)
            geometry = np.linalg.norm(log.ground_truth_position[mask] - anchor, axis=1)
            residuals.append(log.ranges[mask, anchor_index] - geometry)
        pooled = np.concatenate(residuals)
        bias = np.median(pooled)
        sigma = np.clip(MAD_FACTOR * np.median(np.abs(pooled - bias)), *SIGMA_LIMITS_M)
        biases.append(bias)
        sigmas.append(sigma)
        counts.append(pooled.size)
    return dict(range_bias=np.array(biases), range_sigma=np.array(sigmas), training=list(training), event_counts=counts)


def initial_state(log, offset, yaw):
    """Use static IMU alignment, then a controlled first-position/yaw perturbation."""
    rotation, bias = estimate_initial_rotation_and_bias(log)
    selected = log.time >= INITIAL_IMU_PREFIX_S
    scored = replace(log, time=log.time[selected] - log.time[selected][0], imu=log.imu[selected],
                     ranges=log.ranges[selected], changed=log.changed[selected],
                     ground_truth_position=log.ground_truth_position[selected],
                     ground_truth_velocity=log.ground_truth_velocity[selected])
    state = np.eye(5)
    state[:3, :3] = rotation @ rotation_terms(np.array([0.0, 0.0, np.deg2rad(yaw)]))[0]
    state[:3, 4] = scored.ground_truth_position[0] + offset
    physical = np.diag(PRIOR_STD**2)
    mapping = coordinate_map(state)
    return scored, state, physical, mapping @ physical @ mapping.T, bias


def make_methods(state, physical, invariant, bias, model):
    """Create all five estimators with identical physical prior and TWR policy."""
    common = {**SETTINGS, "range_bias": model["range_bias"], "range_sigma": model["range_sigma"]}
    single_args = dict(tracker_factory=Invariant15, imu_core_factory=Invariant15,
                       capture_factory=LiftedTWRInEKF, handoff_covariance_factory=handoff_covariance)
    deep_args = dict(tracker_factory=Invariant15, imu_core_factory=Invariant15,
                     handoff_covariance_factory=handoff_covariance)
    deep = DeepCoupledTwoFilter(state.copy(), invariant.copy(), bias.copy(),
        range_bias=common["range_bias"], range_sigma=common["range_sigma"],
        gate_threshold=common["gate_threshold"], huber_threshold=common["huber_threshold"],
        capture_gate_warmup_updates=common["gate_warmup_updates"],
        tracker_gate_warmup_updates=common["gate_warmup_updates"], **deep_args)
    return dict(inekf15=Invariant15(state, invariant, bias, **common),
                eskf15=ESKF15(state, physical, bias, **common),
                capture_init=CaptureThenTrack(state, invariant, bias, **common, **single_args),
                capture_reset=CaptureThenTrack(state, invariant, bias, reset_on_resume=True, **common, **single_args),
                deep=deep)


def recovery_metrics(times, errors, origin=0.0):
    """Separate first one-second dwell from permanent terminal recovery."""
    valid = times >= origin - 1e-10
    times, errors = times[valid], errors[valid]
    below = np.isfinite(errors) & (errors <= RECOVERY_THRESHOLD_M)
    first = np.nan
    start = None
    for index, good in enumerate(below):
        if good and start is None:
            start = index
        if not good:
            start = None
        elif times[index] - times[start] >= 1.0 - 1e-10:
            first = times[start] - origin
            break
    bad = np.flatnonzero(~below)
    candidate = int(bad[-1] + 1) if bad.size else 0
    stable = np.nan
    if candidate < times.size and times[-1] - times[candidate] >= MIN_STABLE_S - 1e-10:
        stable = times[candidate] - origin
    return dict(first_dwell_s=first, stable_recovery_s=stable, success=bool(np.isfinite(stable)),
                threshold_coverage=float(np.mean(below)),
                last_exit_s=float(times[bad[-1]] - origin) if bad.size else 0.0)


def run_case(log, anchors, spec, model, history_dir):
    """Replay all estimators against one immutable IMU and changed-range schedule."""
    scored, state, physical, invariant, bias = initial_state(log, spec["offset"], spec["yaw_deg"])
    methods = make_methods(state, physical, invariant, bias, model)
    histories = {name: np.empty((scored.time.size, 6)) for name in methods}
    diagnostics = {name: dict(accepted=0, rejected=0, nis=[], elapsed_s=0.0, weights=[]) for name in methods}
    resumed = False
    updates = 0
    last_update_time = None
    longest_gap = 0.0
    for index, timestamp in enumerate(scored.time):
        has_outage = spec["suite"] == "outage"
        blocked = has_outage and OUTAGE[0] <= timestamp < OUTAGE[1]
        if has_outage and not resumed and timestamp >= OUTAGE[1]:
            for estimator in methods.values():
                if hasattr(estimator, "on_ranges_resumed"):
                    estimator.on_ranges_resumed()
            resumed = True
        # Held logger values of zero denote absent transactions, not zero metres.
        usable = scored.changed[index] & np.isfinite(scored.ranges[index]) & (scored.ranges[index] > 0)
        events = [] if blocked else np.flatnonzero(usable)
        updates += len(events)
        if len(events):
            if last_update_time is not None:
                longest_gap = max(longest_gap, timestamp - last_update_time)
            last_update_time = timestamp
        for name, estimator in methods.items():
            start_clock = time.perf_counter()
            if index:
                estimator.predict(scored.imu[index - 1], float(timestamp - scored.time[index - 1]))
            for anchor_index in events:
                accepted = estimator.correct_twr(anchors[anchor_index], float(scored.ranges[index, anchor_index]), int(anchor_index))
                if name == "deep":
                    branch = estimator.tracker if estimator.tracker_output_weight > 0 else estimator.capture.lifted
                    accepted = accepted[1] if estimator.tracker_output_weight > 0 else accepted[0]
                    nis = branch.last_nis
                else:
                    nis = estimator.last_nis
                diagnostics[name]["accepted" if accepted else "rejected"] += 1
                if np.isfinite(nis):
                    diagnostics[name]["nis"].append(nis)
            diagnostics[name]["elapsed_s"] += time.perf_counter() - start_clock
            if name in ("inekf15", "eskf15"):
                position, velocity = estimator.X[:3, 4], estimator.X[:3, 3]
            else:
                position, velocity = estimator.position, estimator.velocity
            histories[name][index] = np.r_[position, velocity]
            diagnostics[name]["weights"].append(float(getattr(estimator, "tracker_output_weight", 1.0)))
        if index and index % 1000 == 0:
            print(f"  {spec['suite']} {log.name} {spec['case']}: {timestamp:.1f}/{scored.time[-1]:.1f}s", flush=True)
    rows = []
    payload = {"time": scored.time, "truth_position": scored.ground_truth_position, "truth_velocity": scored.ground_truth_velocity}
    tail = scored.time >= scored.time[-1] - 5.0
    for name, history in histories.items():
        payload[name] = history
        payload[name + "_weight"] = np.array(diagnostics[name]["weights"])
        errors = np.linalg.norm(history[:, :3] - scored.ground_truth_position, axis=1)
        velocity_error = np.linalg.norm(history[:, 3:] - scored.ground_truth_velocity, axis=1)
        diag = diagnostics[name]
        estimator = methods[name]
        row = {key: val for key, val in spec.items() if key != "offset"}
        row.update(sequence=log.name, method=name, duration_s=float(scored.time[-1]),
                   position_rmse_m=float(np.sqrt(np.mean(errors**2))),
                   position_tail5_rmse_m=float(np.sqrt(np.mean(errors[tail]**2))),
                   position_p95_m=float(np.percentile(errors, 95)), position_max_m=float(np.max(errors)),
                   velocity_rmse_mps=float(np.sqrt(np.mean(velocity_error**2))),
                   velocity_tail5_rmse_mps=float(np.sqrt(np.mean(velocity_error[tail]**2))),
                   finite=bool(np.all(np.isfinite(history))), elapsed_s=diag["elapsed_s"],
                   range_updates=updates, accepted=diag["accepted"], rejected=diag["rejected"],
                   nis_mean=float(np.mean(diag["nis"])) if diag["nis"] else np.nan,
                   nis_p95=float(np.percentile(diag["nis"], 95)) if diag["nis"] else np.nan,
                   mean_tracker_weight=float(np.mean(diag["weights"])),
                   reseeds=getattr(estimator.stats, "tracker_reseeds", 0),
                   longest_range_gap_s=float(longest_gap),
                   resume_error_m=float(errors[np.searchsorted(scored.time, OUTAGE[1])]) if spec["suite"] == "outage" else np.nan)
        row.update(recovery_metrics(scored.time, errors, OUTAGE[1] if spec["suite"] == "outage" else 0.0))
        rows.append(row)
    history_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(history_dir / f"{spec['suite']}__{Path(log.name).stem}__{spec['case']}.npz", **payload)
    return rows


def specifications(names, evaluation, suite):
    """Declare the entire case matrix before loading any result."""
    specs = []
    if suite in ("all", "loso", "smoke"):
        for name in names[:1] if suite == "smoke" else names:
            for amplitude in (0, 9):
                specs.append(dict(suite="loso", sequence=name, case=f"p{amplitude}_d0_y0", initial_error_m=amplitude,
                                  yaw_deg=0, offset=DIRECTIONS[0] * amplitude))
    if suite in ("all", "holdout"):
        for name in evaluation:
            for amplitude in (0, 3, 6, 9, 12):
                for direction in range(1 if amplitude == 0 else 3):
                    specs.append(dict(suite="holdout", sequence=name, case=f"p{amplitude}_d{direction}_y0",
                                      initial_error_m=amplitude, yaw_deg=0, offset=DIRECTIONS[direction] * amplitude))
            for yaw in (90, 180):
                specs.append(dict(suite="holdout", sequence=name, case=f"p9_d0_y{yaw}", initial_error_m=9,
                                  yaw_deg=yaw, offset=DIRECTIONS[0] * 9))
    if suite in ("all", "outage"):
        for name in evaluation:
            for amplitude in (0, 9):
                specs.append(dict(suite="outage", sequence=name, case=f"p{amplitude}_d0_y0", initial_error_m=amplitude,
                                  yaw_deg=0, offset=DIRECTIONS[0] * amplitude))
    return specs


def main():
    """Freeze fingerprints, run resumable cases, and retain full error histories."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--suite", choices=("all", "loso", "holdout", "outage", "smoke"), default="all")
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--sequence", action="append", help="Run only named logs; calibration always uses the full frozen split")
    parser.add_argument("--prepare-only", action="store_true", help="Write the complete protocol before starting any replay")
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    names = sorted(path.name for path in (args.dataset / "twr").glob("*.csv"))
    if len(names) != 12:
        raise ValueError(f"Expected 12 restored logs, found {len(names)}")
    training, evaluation = [], []
    for motion in MOTIONS:
        paired = [name for name in names if name.endswith(f"_{motion}.csv")]
        if len(paired) != 2:
            raise ValueError(f"Expected two {motion} logs, found {paired}")
        training.append(paired[0])
        evaluation.append(paired[1])
    anchors = load_anchors(args.dataset)
    logs = {name: load_real_log(args.dataset, name, sensor_delay_s=SENSOR_DELAY_S) for name in names}
    manifest = {name: dict(sha256=hashlib.sha256((args.dataset / "twr" / name).read_bytes()).hexdigest(),
                          samples=log.time.size, duration_s=float(log.time[-1]), events=int(log.changed.sum())) for name, log in logs.items()}
    models = {name: calibration(logs, anchors, [other for other in names if other != name]) for name in names}
    heldout_model = calibration(logs, anchors, training)
    specs = specifications(names, evaluation, args.suite)
    if args.sequence:
        if set(args.sequence) - set(names):
            raise ValueError("Unknown --sequence value")
        specs = [spec for spec in specs if spec["sequence"] in args.sequence]
    source_paths = list(Path(__file__).parent.glob("*.py")) + [DEEP_SOURCE / "deep_coupled.py", DEEP_SOURCE / "lifted_twr.py", DEEP_SOURCE / "twr_experiment.py"]
    fingerprints = {str(path.relative_to(PROJECT.parent)): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths}
    protocol = dict(dataset=str(args.dataset), training=training, evaluation=evaluation,
                    common_delay_s=SENSOR_DELAY_S, settings=SETTINGS, physical_prior_std=PRIOR_STD,
                    prefix_excluded_s=INITIAL_IMU_PREFIX_S, sigma_limits_m=SIGMA_LIMITS_M,
                    range_models={"loso": models, "holdout": heldout_model},
                    sample_noise_variance=dict(gyro=0.1**2, accel=2.0**2, gyro_bias_rate=1e-6**2, accel_bias_rate=1e-4**2),
                    noise_semantics="Independent held IMU rate samples; bias-rate end-step increments scale dt^2; not continuous PSD",
                    outage_s=OUTAGE, stable_threshold_m=RECOVERY_THRESHOLD_M, minimum_terminal_stable_s=MIN_STABLE_S,
                    controlled_truth_initial_position=True, ground_clamp=False,
                    real_attitude_bias_truth_available=False, date_independent_holdout=False,
                    historical_supervisor_exposed_to_all_logs=True, duration_limit=args.duration,
                    dataset_manifest=manifest, source_sha256=fingerprints, cases=specs)
    protocol_path = output / "protocol.json"
    if protocol_path.exists():
        old = json.loads(protocol_path.read_text(encoding="utf-8"))
        if old["source_sha256"] != fingerprints or old["duration_limit"] != args.duration or old["cases"] != json.loads(json.dumps(specs, default=lambda val: val.tolist())):
            raise ValueError("Existing results use a different source/case protocol; choose a new output directory")
    write_json(protocol_path, protocol)
    if args.prepare_only:
        print(f"Frozen {len(specs)} cases: {protocol_path}", flush=True)
        return
    metrics_path = output / "metrics.csv"
    rows = pd.read_csv(metrics_path).to_dict("records") if metrics_path.exists() else []
    completed = {(row["suite"], row["sequence"], row["case"]) for row in rows}
    for number, spec in enumerate(specs, 1):
        key = (spec["suite"], spec["sequence"], spec["case"])
        if key in completed:
            continue
        print(f"CASE {number}/{len(specs)} {key}", flush=True)
        log = logs[spec["sequence"]]
        if args.duration is not None:
            mask = log.time <= args.duration + INITIAL_IMU_PREFIX_S
            log = replace(log, time=log.time[mask], imu=log.imu[mask], ranges=log.ranges[mask], changed=log.changed[mask],
                          ground_truth_position=log.ground_truth_position[mask], ground_truth_velocity=log.ground_truth_velocity[mask])
        started = time.perf_counter()
        rows.extend(run_case(log, anchors, spec, models[spec["sequence"]] if spec["suite"] == "loso" else heldout_model, output / "histories"))
        pd.DataFrame(rows).to_csv(metrics_path, index=False, encoding="utf-8")
        print(f"DONE {key} wall={time.perf_counter() - started:.2f}s", flush=True)
    print(f"Complete: {len(rows)} method-cases; {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
