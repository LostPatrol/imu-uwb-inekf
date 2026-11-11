"""Controlled real UTIL position/heading handoff and subsequent InEKF trajectories.

Sensor-only trajectory generation finishes before the separate truth-scoring phase.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp

# Existing source utilities are reused; previous outputs/documents are not read.
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "experiments" / "radio_inertial_models" / "tdoa_branch"))
sys.path.insert(0, str(HERE.parent))
from study import load
import kernel

# The immutable protocol is written before reading the additional holdout trials.
PROTOCOL = json.loads((HERE / "protocol.json").read_text(encoding="utf8"))
# Calibrated physical antenna offset, in metres in the body frame.
LEVER = np.array([-.01245, .00127, .0908])
# Common physical handoff uncertainty isolates nominal rotation changes.
PHYSICAL = np.diag(np.array(PROTOCOL["common_handoff_physical_std"]) ** 2)
# Candidate packets are temporally thinned within each directed pair.
CANDIDATE_PAIR_SPACING = .1
# Fixed candidate count limits nonlinear fitting without using truth or residual ranking.
MAX_CANDIDATE_PACKETS = 600


def visible_data(data, scenario_name, endpoint=None):
    """Physically remove unavailable radio before any independent reconstruction."""
    result = dict(data)
    radio = data["radio"]
    scenario = PROTOCOL["scenarios"][scenario_name]
    keep = np.ones(len(radio), dtype=bool)
    if scenario["outage_start"] is not None:
        origin = data["times"][0]
        keep &= ~((radio[:, 0] >= origin + scenario["outage_start"])
                  & (radio[:, 0] < origin + scenario["outage_end"]))
    if endpoint is not None:
        keep &= radio[:, 0] <= endpoint
    result["radio"] = radio[keep]
    return result


def safe_json(value):
    """Convert model diagnostics into JSON without silently dropping array evidence."""
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def record_name(geometry, trial, radio_protocol):
    """Use the predetermined const4 trajectory rather than selecting by observed errors."""
    suffix = "-traj1" if geometry == 4 else ""
    return f"const{geometry}-trial{trial}-tdoa{radio_protocol}{suffix}"


def fit_radio_pv(observations, anchors, endpoint, seed=None):
    """Fit antenna endpoint position/velocity from raw hyperbolas without IMU attitude."""
    tau = observations[:, 0] - endpoint
    a, b = observations[:, 1:3].astype(int).T
    def model(parameters):
        """Return the native constant-velocity TDoA residual and exact Jacobian."""
        tag = parameters[:3] + tau[:, None] * parameters[3:]
        da, db = tag - anchors[a], tag - anchors[b]
        ra, rb = np.linalg.norm(da, axis=1), np.linalg.norm(db, axis=1)
        gradient = db / rb[:, None] - da / ra[:, None]
        return (rb - ra - observations[:, 3]) / .4, np.column_stack([gradient, tau[:, None] * gradient]) / .4
    initial = np.r_[np.mean(anchors, axis=0), np.zeros(3)] if seed is None else seed
    fit = least_squares(lambda x: model(x)[0], initial, jac=lambda x: model(x)[1],
                        loss="huber", f_scale=2, max_nfev=60)
    return fit.x


def radio_path(data):
    """Build causal independent 200 ms radio-only antenna positions and CV velocities."""
    radio, anchors, origin = data["radio"], data["anchors"], data["times"][0]
    rows = []
    seed = np.r_[np.mean(anchors, axis=0), np.zeros(3)]
    for available in np.arange(origin + .2, data["times"][-1] + 1e-9, .2):
        observations = radio[(radio[:, 0] > available - .6) & (radio[:, 0] <= available)]
        # A new output requires new packets, including after a genuine radio outage.
        fresh = np.sum((radio[:, 0] > available - .2) & (radio[:, 0] <= available))
        if len(observations) < 10 or fresh < 3:
            continue
        seed = fit_radio_pv(observations, anchors, available, seed)
        rows.append(np.r_[available, seed])
    return np.array(rows)


def initial_state(data):
    """Initialize from the first 0.6 s radio and IMU only, then consume future radio once."""
    origin = data["times"][0]
    endpoint = origin + .6
    imu = data["imu"][data["times"] <= origin + .2]
    specific = np.mean(imu[:, :3], axis=0)
    rotation = Rotation.align_vectors([[0., 0., 1.]], [specific / np.linalg.norm(specific)])[0].as_matrix()
    observations = data["radio"][data["radio"][:, 0] <= endpoint]
    pv = fit_radio_pv(observations, data["anchors"], endpoint)
    state = np.eye(5)
    state[:3, :3] = rotation
    state[:3, 3] = pv[3:]
    state[:3, 4] = pv[:3] - rotation @ LEVER
    prefix = data["imu"][data["times"] <= endpoint]
    stationary = (np.max(np.abs(np.linalg.norm(prefix[:, :3], axis=1) - 1)) < .05
                  and np.max(np.linalg.norm(prefix[:, 3:], axis=1)) < 2.)
    bias = np.zeros(6)
    if stationary:
        bias[:3] = np.radians(np.mean(prefix[:, 3:], axis=0))
    return endpoint, state, bias


def bounded_replay(data, state, covariance, bias, start, end, outage):
    """Run the single compiled native TDoA implementation for every policy."""
    variance = .22 if "const4" in data["name"] else .14
    origin = data["times"][0]
    left, right = (-1., -1.) if outage is None else (origin + outage[0], origin + outage[1])
    return kernel.replay(data["times"], data["imu"], data["radio"], data["anchors"],
                         state, covariance, bias, start, end, variance, left, right)


def window_inputs(data, start, end, bias, path):
    """Supply endpoint-reanchored specific force and B in the window-start body frame."""
    times, imu = data["times"], data["imu"]
    first = int(np.searchsorted(times, start))
    last = int(np.searchsorted(times, end))
    Q = np.eye(3)
    displacement, velocity = np.zeros(3), np.zeros(3)
    B, Bdot = np.zeros((3, 3)), np.zeros((3, 3))
    samples, matrices, rotations = [], [], []
    for k in range(first, last + 1):
        samples.append(np.r_[times[k], displacement, velocity])
        matrices.append(B.copy())
        rotations.append(Q.copy())
        if k == last:
            break
        dt = times[k + 1] - times[k]
        value = .5 * (imu[k] + imu[k + 1])
        acceleration = Q @ (9.81 * value[:3] - bias[3:])
        displacement += velocity * dt + .5 * acceleration * dt**2
        velocity += acceleration * dt
        B += Bdot * dt + .5 * Q * dt**2
        Bdot += Q * dt
        Q = Q @ kernel.rotation_terms((np.radians(value[3:]) - bias[:3]) * dt)[0]
    samples, matrices, rotations = np.array(samples), np.array(matrices), np.array(rotations)
    endpoint = samples[-1, 0]
    # This path is a conditional linearization available at handoff, never beyond it.
    path = path[path[:, 0] <= endpoint]
    tau_grid = samples[:, 0] - endpoint
    force = samples[:, 1:4] - samples[-1, 1:4] - tau_grid[:, None] * samples[-1, 4:7]
    force += np.einsum("nij,j->ni", rotations, LEVER)
    matrices -= matrices[-1].copy() + tau_grid[:, None, None] * Bdot
    observations = data["radio"][(data["radio"][:, 0] >= samples[0, 0]) & (data["radio"][:, 0] <= endpoint)]
    # Thinning is determined by pair/time only, independently of measurement values.
    keep, previous = [], {}
    for index, row in enumerate(observations):
        pair = (int(row[1]), int(row[2]))
        if row[0] - previous.get(pair, -1e9) >= CANDIDATE_PAIR_SPACING:
            keep.append(index)
            previous[pair] = row[0]
    observations = observations[keep]
    if len(observations) > MAX_CANDIDATE_PACKETS:
        observations = observations[np.linspace(0, len(observations) - 1, MAX_CANDIDATE_PACKETS).astype(int)]
    stamps = observations[:, 0]
    local_force = np.column_stack([np.interp(stamps, samples[:, 0], force[:, k]) for k in range(3)])
    local_B = np.stack([np.interp(stamps, samples[:, 0], matrices[:, j, k]) for j in range(3) for k in range(3)], axis=1).reshape(-1, 3, 3)
    radio_only = np.column_stack([np.interp(stamps, path[:, 0], path[:, k]) for k in (1, 2, 3)])
    return dict(tau=stamps - endpoint, force=local_force, B=local_B,
                plus=data["anchors"][observations[:, 2].astype(int)],
                minus=data["anchors"][observations[:, 1].astype(int)], y=observations[:, 3],
                sigma=np.full(len(observations), np.sqrt(.22 if "const4" in data["name"] else .14)),
                path=radio_only, Qend=rotations[-1], endpoint=endpoint,
                source_packets=int(np.sum((data["radio"][:, 0] >= samples[0, 0]) & (data["radio"][:, 0] <= endpoint))))


def fit_full_attitude(inputs, Cref, pv):
    """Fit a strong eight-start raw-TDoA pose with common tilt and bias priors."""
    tau, force, B, plus, minus, y, sigma = (inputs[k] for k in ("tau", "force", "B", "plus", "minus", "y", "sigma"))
    gravity = .5 * tau[:, None]**2 * np.array([0., 0., -9.81])
    def residual_jacobian(parameters):
        """Differentiate native hyperbolas with a world yaw and two world tilt variables."""
        yaw = Rotation.from_euler("z", parameters[6]).as_matrix()
        tilt = Rotation.from_rotvec(np.r_[parameters[10:12], 0.]).as_matrix()
        C = yaw @ tilt @ Cref
        shaped = force - np.einsum("nij,j->ni", B, parameters[7:10])
        rotated = shaped @ C.T
        tag = parameters[:3] + tau[:, None] * parameters[3:6] + rotated + gravity
        da, db = tag - minus, tag - plus
        ra, rb = np.linalg.norm(da, axis=1), np.linalg.norm(db, axis=1)
        gradient = db / rb[:, None] - da / ra[:, None]
        residual = (rb - ra - y) / sigma
        jacobian = np.zeros((len(y), 12))
        jacobian[:, :3] = gradient / sigma[:, None]
        jacobian[:, 3:6] = tau[:, None] * gradient / sigma[:, None]
        jacobian[:, 6] = np.sum(gradient * np.cross([0., 0., 1.], rotated), axis=1) / sigma
        jacobian[:, 7:10] = -np.einsum("ni,ij,njk->nk", gradient, C, B) / sigma[:, None]
        for j in range(2):
            direction = np.zeros(2)
            direction[j] = 1e-6
            upper = yaw @ Rotation.from_rotvec(np.r_[parameters[10:12] + direction, 0.]).as_matrix() @ Cref
            lower = yaw @ Rotation.from_rotvec(np.r_[parameters[10:12] - direction, 0.]).as_matrix() @ Cref
            derivative = shaped @ ((upper - lower) / 2e-6).T
            jacobian[:, 10 + j] = np.sum(gradient * derivative, axis=1) / sigma
        prior = np.zeros((5, 12))
        prior[:3, 7:10] = np.eye(3) / .2
        prior[3:, 10:12] = np.eye(2) / .08
        return np.r_[residual, parameters[7:10] / .2, parameters[10:12] / .08], np.vstack([jacobian, prior])
    fits, evaluations = [], 0
    for yaw in np.arange(8) * np.pi / 4:
        initial = np.r_[pv, yaw, np.zeros(5)]
        fit = least_squares(lambda x: residual_jacobian(x)[0], initial,
                            jac=lambda x: residual_jacobian(x)[1], loss="huber", f_scale=2,
                            max_nfev=120, ftol=1e-8, xtol=1e-8, gtol=1e-8)
        fits.append(fit)
        evaluations += fit.nfev
    fit = min(fits, key=lambda value: value.cost)
    C = Rotation.from_euler("z", fit.x[6]).as_matrix() @ Rotation.from_rotvec(np.r_[fit.x[10:12], 0.]).as_matrix() @ Cref
    return C @ inputs["Qend"], dict(cost=float(fit.cost), evaluations=evaluations,
                                     fit_success=bool(fit.success), delta_deg=float(np.degrees(fit.x[6])),
                                     fitted_delta_bias=fit.x[7:10], tilt_deg=float(np.degrees(np.linalg.norm(fit.x[10:12]))))


def orientation_yaw(rotation):
    """Return the heading of the body x-axis in the surveyed world plane."""
    return float(np.arctan2(rotation[1, 0], rotation[0, 0]))


def invoke_decision(module, inputs, Cref):
    """Call the common no-truth heading interface and keep all diagnostics."""
    arguments = [inputs[k] for k in ("tau", "force", "B", "plus", "minus", "y", "sigma", "path")]
    result = module.decide(*arguments, Cref)
    if not isinstance(result, dict):
        raise TypeError("Heading evidence must return a diagnostic dictionary")
    return result


def run_case(data, path, scenario_name, yaw_offset, decisions, output):
    """Run shared prefix/window then independent matched post-handoff counterfactuals."""
    origin = data["times"][0]
    scenario = PROTOCOL["scenarios"][scenario_name]
    start, end = origin + scenario["capture_start"], origin + scenario["capture_end"]
    start = data["times"][np.searchsorted(data["times"], start)]
    end = data["times"][np.searchsorted(data["times"], end)]
    outage = None if scenario["outage_start"] is None else (scenario["outage_start"], scenario["outage_end"])
    initial_time, state, bias = initial_state(data)
    nominal_rotation = state[:3, :3].copy()
    state[:3, :3] = Rotation.from_euler("z", yaw_offset, degrees=True).as_matrix() @ state[:3, :3]
    change = kernel.mapping(state)
    covariance = change @ PHYSICAL @ change.T
    empty = np.empty((0, 19))
    if start > initial_time:
        state, covariance, bias, prefix, _, _, prefix_failed = bounded_replay(data, state, covariance, bias, initial_time, start, outage)
        window_start_state = state.copy()
        window_bias = bias.copy()
        shadow_start = start
    else:
        prefix, prefix_failed = empty, False
        window_start_state = state.copy()
        window_bias = bias.copy()
        shadow_start = initial_time
    state, covariance, bias, normal_window, _, _, window_failed = bounded_replay(data, state, covariance, bias, shadow_start, end, outage)
    retained, normal_covariance, common_bias = state.copy(), covariance.copy(), bias.copy()
    inputs = window_inputs(data, start, end, window_bias, path)
    end = inputs["endpoint"]
    # Radio-only reconstruction is shared across every yaw perturbation and policy.
    recent = data["radio"][(data["radio"][:, 0] > end - .6) & (data["radio"][:, 0] <= end)]
    pv = fit_radio_pv(recent, data["anchors"], end)
    shared_position = pv[:3] - nominal_rotation @ LEVER
    shared_velocity = pv[3:]
    Cref = retained[:3, :3] @ inputs["Qend"].T
    full_rotation, full_diagnostic = fit_full_attitude(inputs, Cref, np.r_[shared_position, shared_velocity])
    forced_delta = orientation_yaw(full_rotation) - orientation_yaw(retained[:3, :3])
    evidence = {key: invoke_decision(module, inputs, Cref) for key, module in decisions.items()}
    rotations = dict(retain_normal=retained[:3, :3].copy(), full_attitude=full_rotation,
                     forced_yaw_only=Rotation.from_euler("z", forced_delta).as_matrix() @ retained[:3, :3])
    for key, result in evidence.items():
        rotations[key] = Rotation.from_euler("z", float(result.get("delta", 0.))).as_matrix() @ retained[:3, :3]
    forced_evidence = dict(evidence["evidence_yaw"])
    forced_evidence.update(delta=float(forced_evidence.get("candidate", 0.)),
                           accepted=True, forced=True,
                           gate_accepted=bool(evidence["evidence_yaw"].get("accepted", False)))
    evidence["forced_evidence_yaw"] = forced_evidence
    rotations["forced_evidence_yaw"] = Rotation.from_euler("z", forced_evidence["delta"]).as_matrix() @ retained[:3, :3]
    # Independent positions are available as each bin arrives, before yaw waits finish.
    window_path = path[(path[:, 0] > start) & (path[:, 0] <= end)]
    common_window = np.zeros((len(window_path), 19))
    if len(window_path):
        common_window[:, 0] = window_path[:, 0]
        common_window[:, 1:4] = window_path[:, 1:4] - nominal_rotation @ LEVER
        common_window[:, 4:7] = window_path[:, 4:7]
        for index, stamp in enumerate(window_path[:, 0]):
            if len(normal_window):
                nearest = np.clip(np.searchsorted(normal_window[:, 0], stamp, side="right") - 1, 0, len(normal_window) - 1)
                common_window[index, 7:16] = normal_window[nearest, 7:16]
            else:
                common_window[index, 7:16] = retained[:3, :3].ravel()
    records = []
    policies = list(rotations) + ["normal_unmodified"]
    for policy in policies:
        if policy == "normal_unmodified":
            handoff, P = retained.copy(), normal_covariance.copy()
            before = np.vstack([prefix, normal_window])
        else:
            handoff = retained.copy()
            handoff[:3, :3] = rotations[policy]
            handoff[:3, 3], handoff[:3, 4] = shared_velocity, shared_position
            change = kernel.mapping(handoff)
            P = change @ PHYSICAL @ change.T
            before = np.vstack([prefix, common_window])
        final, final_P, final_bias, after, accepted, rejected, failed = bounded_replay(
            data, handoff, P, common_bias, end, data["times"][-1], None)
        key = f"{data['name']}__{scenario_name}__yaw{yaw_offset:g}__{policy}"
        np.savez_compressed(output / (key + ".npz"), before=before, after=after, handoff=handoff,
                            handoff_physical_covariance=PHYSICAL if policy != "normal_unmodified" else kernel.inverse_mapping(retained) @ P @ kernel.inverse_mapping(retained).T,
                            bias=common_bias, window_start_state=window_start_state,
                            retained=retained, final=final)
        record = dict(key=key, name=data["name"], scenario=scenario_name, yaw_offset_deg=yaw_offset,
                      policy=policy, origin=origin, window_start=start, handoff_time=end,
                      first_position_time=float(initial_time if policy == "normal_unmodified" else path[0, 0]),
                      first_recovery_position_time=float(window_path[0, 0]) if len(window_path) and policy != "normal_unmodified" else None,
                      candidate_packets=len(inputs["y"]), source_packets=inputs["source_packets"],
                      accepted_radio=accepted, rejected_radio=rejected,
                      failed=bool(prefix_failed or window_failed or failed),
                      full_candidate=full_diagnostic, decision=evidence.get(policy, None))
        records.append(record)
    return records


def score_rows(truth, rows, uniform_start=None, uniform_end=None):
    """Score unaligned world XY/Z and physical SO(3)/yaw only after generation."""
    if not len(rows):
        return dict(xy_rmse=None, z_rmse=None, yaw_rmse_deg=None, so3_rmse_deg=None, samples=0)
    if uniform_start is not None:
        # A common 20 Hz scoring clock prevents unequal output densities changing RMSE.
        grid = np.arange(uniform_start, min(uniform_end, rows[-1, 0]) + 1e-9, .05)
        # Nanosecond tolerance only reconciles equivalent floating-point tick formulas.
        indices = np.searchsorted(rows[:, 0], grid + 1e-9, side="right") - 1
        valid = indices >= 0
        rows = rows[indices[valid]].copy()
        rows[:, 0] = grid[valid]
    keep = (rows[:, 0] >= truth[0, 0]) & (rows[:, 0] <= truth[-1, 0])
    rows = rows[keep]
    if not len(rows):
        return dict(xy_rmse=None, z_rmse=None, yaw_rmse_deg=None, so3_rmse_deg=None, samples=0)
    actual_position = np.column_stack([np.interp(rows[:, 0], truth[:, 0], truth[:, k]) for k in (1, 2, 3)])
    actual_rotation = Slerp(truth[:, 0], Rotation.from_quat(truth[:, 4:8]))(rows[:, 0])
    estimated = Rotation.from_matrix(rows[:, 7:16].reshape(-1, 3, 3))
    difference = rows[:, 1:4] - actual_position
    yaw = estimated.as_euler("xyz")[:, 2] - actual_rotation.as_euler("xyz")[:, 2]
    yaw = np.degrees(np.arctan2(np.sin(yaw), np.cos(yaw)))
    angle = np.degrees((actual_rotation.inv() * estimated).magnitude())
    return dict(xy_rmse=float(np.sqrt(np.mean(np.sum(difference[:, :2]**2, axis=1)))),
                z_rmse=float(np.sqrt(np.mean(difference[:, 2]**2))),
                yaw_rmse_deg=float(np.sqrt(np.mean(yaw**2))), so3_rmse_deg=float(np.sqrt(np.mean(angle**2))),
                samples=len(rows), last_time=float(rows[-1, 0]))


def score_split(output, records):
    """Load truth only here and retain every trial/case/failure rather than successful subsets."""
    metrics, truths = [], {}
    for record in records:
        name = record["name"]
        if name not in truths:
            truths[name] = load(name, duration=PROTOCOL["duration_seconds"])["truth"]
        truth = truths[name]
        artifact = np.load(output / (record["key"] + ".npz"))
        after = artifact["after"]
        whole = np.vstack([artifact["before"], after])
        metric = {k: v for k, v in record.items() if k not in ["full_candidate", "decision"]}
        metric.update({"post_" + k: v for k, v in score_rows(truth, after).items()})
        metric.update({"whole_" + k: v for k, v in score_rows(truth, whole, record["origin"] + .65, record["origin"] + PROTOCOL["duration_seconds"]).items()})
        # The cold window begins before the 0.6 s filter initialization actually exists.
        arrival_stamp = max(record["window_start"], record["origin"] + .6)
        for label, state, stamp in [("arrival", artifact["window_start_state"], arrival_stamp),
                                    ("retained", artifact["retained"], record["handoff_time"]),
                                    ("handoff", artifact["handoff"], record["handoff_time"])]:
            row = np.r_[stamp, state[:3, 4], state[:3, 3], state[:3, :3].ravel(), np.zeros(3)][None]
            values = score_rows(truth, row)
            metric[label + "_yaw_error_deg"] = values["yaw_rmse_deg"]
            metric[label + "_xy_error"] = values["xy_rmse"]
        if record["decision"] is not None:
            metric["heading_accepted"] = bool(record["decision"].get("accepted", False))
            metric["candidate_delta_deg"] = float(np.degrees(record["decision"].get("candidate", 0.)))
            metric["applied_delta_deg"] = float(np.degrees(record["decision"].get("delta", 0.)))
            metric["sd_deg"] = record["decision"].get("sd_deg", None)
        metrics.append(metric)
    pd.DataFrame(metrics).to_csv(output / "metrics.csv", index=False)
    frame = pd.DataFrame(metrics)
    summary = frame.groupby(["scenario", "policy"])[["post_xy_rmse", "post_z_rmse", "post_yaw_rmse_deg", "whole_xy_rmse"]].agg(["median", "max"])
    summary.to_csv(output / "summary.csv")
    print(summary.to_string(), flush=True)


def main():
    """Execute the frozen sensor-only split, then start the explicit scoring phase."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["development", "holdout"], default="development")
    parser.add_argument("--names", nargs="+")
    parser.add_argument("--scenarios", nargs="+", default=list(PROTOCOL["scenarios"]))
    parser.add_argument("--yaws", type=float, nargs="+", default=PROTOCOL["initial_world_yaw_offsets_deg"])
    parser.add_argument("--output", default="development")
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    output = HERE / args.output
    output.mkdir(exist_ok=True)
    if args.score_only:
        records = json.loads((output / "records.json").read_text(encoding="utf8"))
        score_split(output, records)
        return
    decisions = dict(evidence_yaw=importlib.import_module("yaw_evidence"),
                     bank_yaw=importlib.import_module("bank.bank"))
    trials = PROTOCOL[args.split + "_trials"]
    names = args.names or [record_name(g, trial, radio) for trial in trials for g in PROTOCOL["geometries"] for radio in PROTOCOL["radio_protocols"]]
    records = []
    cases = output / "cases"
    cases.mkdir(exist_ok=True)
    (output / "arguments.json").write_text(json.dumps(vars(args), indent=2), encoding="utf8")
    (output / "protocol.json").write_text(json.dumps(PROTOCOL, indent=2), encoding="utf8")
    started = time.perf_counter()
    for name in names:
        data = load(name, duration=PROTOCOL["duration_seconds"])
        data.pop("truth")
        for scenario in args.scenarios:
            scenario_data = visible_data(data, scenario)
            requested_end = data["times"][0] + PROTOCOL["scenarios"][scenario]["capture_end"]
            actual_end = data["times"][np.searchsorted(data["times"], requested_end)]
            path = radio_path(visible_data(data, scenario, actual_end))
            for yaw in args.yaws:
                checkpoint = cases / f"{name}__{scenario}__yaw{yaw:g}.json"
                if args.resume and checkpoint.exists():
                    records.extend(json.loads(checkpoint.read_text(encoding="utf8")))
                    continue
                case_records = run_case(scenario_data, path, scenario, yaw, decisions, output)
                records.extend(case_records)
                # Each checkpoint is written once, avoiding repeated reader/sync contention.
                checkpoint.write_text(json.dumps(safe_json(case_records), indent=2), encoding="utf8")
                print(json.dumps(dict(name=name, scenario=scenario, yaw=yaw, trajectories=len(records),
                                      elapsed_seconds=time.perf_counter() - started)), flush=True)
    (output / "records.json").write_text(json.dumps(safe_json(records), indent=2), encoding="utf8")
    print("Sensor-only generation complete; beginning truth scoring.", flush=True)
    score_split(output, records)


if __name__ == "__main__":
    main()
