"""Replay shared SI-unit sensor streams for native acquisition and MAP snapshots.

Native reset references share the same sensor-only fit, never a fabricated run.
MAP snapshots retain t=0 as the physical prior origin at both 2 and 10 seconds.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import time

# Bound BLAS concurrency before importing numerical libraries in each worker.
for _name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from solver import Window, StartPrior, init, shape, sensor_seed, solve

# Gravity and frozen experimental schedules are shared across every recording.
GRAVITY = np.array([0., 0., -9.81])
NATIVE_ARRIVALS = np.arange(2., 15.001, .5)
STRESS_ARRIVALS = (2., 10.)
MODELS = ("profile1", "body_bias_prior")


def digest(path):
    """Hash the exact common input bytes consumed by this replay."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_episode(path):
    """Load the sensor interface without reading any truth or state-prior file."""
    meta = json.loads((path/"meta.json").read_text(encoding="utf-8"))
    imu = pd.read_csv(path/"imu.csv")
    imu = imu[imu.t >= -1e-10].reset_index(drop=True)
    times = imu.t.to_numpy(float)
    if not len(times) or abs(times[0]) > .02:
        raise ValueError("The common stream must include its t=0 IMU boundary")
    # The common R0 and biases already summarize the shared stationary prefix.
    times = times-times[0]
    samples = imu[["ax", "ay", "az", "gx", "gy", "gz"]].to_numpy(float)
    rotations = np.empty((len(times), 3, 3))
    rotations[0] = np.asarray(meta["R0"], float)
    velocity = np.zeros((len(times), 3))
    displacement = np.zeros_like(velocity)
    bias_velocity = np.zeros((len(times), 3, 3))
    bias_position = np.zeros_like(bias_velocity)
    gyro_bias = np.asarray(meta["gyro_bias"], float)
    accel_bias = np.asarray(meta["accel_bias"], float)
    integration_work = np.zeros(len(times))
    for k, dt in enumerate(np.diff(times)):
        integration_started = time.perf_counter()
        acceleration = rotations[k] @ (samples[k, :3]-accel_bias)+GRAVITY
        displacement[k+1] = displacement[k]+dt*velocity[k]+.5*dt**2*acceleration
        velocity[k+1] = velocity[k]+dt*acceleration
        bias_position[k+1] = bias_position[k]+dt*bias_velocity[k]+.5*dt**2*rotations[k]
        bias_velocity[k+1] = bias_velocity[k]+dt*rotations[k]
        rotations[k+1] = rotations[k] @ Rotation.from_rotvec((samples[k, 3:]-gyro_bias)*dt).as_matrix()
        integration_work[k+1] = time.perf_counter()-integration_started
    anchors = pd.read_csv(path/"anchors.csv").set_index("anchor_id")
    radio = pd.read_csv(path/"radio.csv")
    return dict(meta=meta, time=times, samples=samples, rotations=rotations,
        displacement=displacement, velocity=velocity, bias_position=bias_position,
        bias_velocity=bias_velocity, gyro_bias=gyro_bias, accel_bias=accel_bias,
        anchors=anchors, radio=radio, integration_work=integration_work)


def at_times(data, events):
    """Evaluate held IMU position, velocity, rotation and bias integrals causally."""
    grid = data["time"]
    index = np.clip(np.searchsorted(grid, events, side="right")-1, 0, len(grid)-1)
    dt = events-grid[index]
    q = data["rotations"][index]
    acceleration = np.einsum("nij,nj->ni", q, data["samples"][index, :3]-data["accel_bias"])+GRAVITY
    displacement = data["displacement"][index]+dt[:, None]*data["velocity"][index]+.5*dt[:, None]**2*acceleration
    velocity = data["velocity"][index]+dt[:, None]*acceleration
    bias_position = data["bias_position"][index]+dt[:, None, None]*data["bias_velocity"][index]+.5*dt[:, None, None]**2*q
    bias_velocity = data["bias_velocity"][index]+dt[:, None, None]*q
    rotations = q @ Rotation.from_rotvec((data["samples"][index, 3:]-data["gyro_bias"])*dt[:, None]).as_matrix()
    return displacement, velocity, bias_position, bias_velocity, rotations


def make_window(data, end, duration):
    """Build one arrived-only window and its exact physical-start geometry."""
    start = max(0., end-duration)
    radio = data["radio"]
    selected = radio[(radio.t >= start-1e-10) & (radio.t <= end+1e-10)]
    if len(selected) < 14:
        raise ValueError(f"Only {len(selected)} arrived range events")
    events = selected.t.to_numpy(float)
    points = np.r_[events, start, end]
    displacement, velocity, bp, bv, rotations = at_times(data, points)
    tau = points-end
    d = displacement-displacement[-1]-tau[:, None]*velocity[-1]
    sensitivity = bp-bp[-1]-tau[:, None, None]*bv[-1]
    # State coordinates describe the body origin; ranges describe the radio tag.
    lever = np.asarray(data["meta"].get("lever_m", [0., 0., 0.]), float)
    tag_d = d[:-2]+np.einsum("nij,j->ni", rotations[:-2], lever)
    anchors = data["anchors"].loc[selected.anchor_id, ["x", "y", "z"]].to_numpy(float)
    values = (events-end, tag_d, anchors, selected.range_m.to_numpy(float), selected.sigma_m.to_numpy(float))
    window = Window(values, sensitivity[:-2], start-end, d[-2], sensitivity[-2], rotations[-2], rotations[-1])
    packet_hash = hashlib.sha256(selected.to_csv(index=False).encode()).hexdigest()
    return window, len(selected), packet_hash, start


def to_row(episode, case, arrival, model, fit, events, packet_hash, start, wall_s):
    """Keep every finite/nonfinite estimate, solve failure and timing visible."""
    x = fit["x"]
    rotation = fit["rotation"]
    row = dict(episode=episode, case_id=case, arrival_s=arrival, state_s=arrival,
        method=model, x=x[0], y=x[1], z=x[2], vx=x[3], vy=x[4], vz=x[5],
        yaw_rad=float(np.arctan2(rotation[1, 0], rotation[0, 0])),
        wall_s=wall_s, status=("exception" if "failure" in fit else
            "ok" if fit["solver_success"] else "solver_limit"),
        events=events, packet_sha256=packet_hash, window_start_s=start,
        window_end_s=arrival, output_available=bool(np.isfinite(x).all()))
    row.update({key:value for key,value in fit.items()
                if key not in ("x", "position", "velocity", "rotation")})
    return row


def attempt(window, model, prior=None, seed=None):
    """Retain a failed numerical attempt as an explicit unavailable-state row."""
    started = time.perf_counter()
    try:
        return solve(window, model, prior, seed)
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
        return dict(x=np.full(10 if model == "body_bias_prior" else 7, np.nan),
            rotation=np.full((3, 3), np.nan), solver_success=False, solver_status=-999,
            failure=str(error), fit_elapsed_s=time.perf_counter()-started,
            prior_consumed=prior is not None, refinement_starts=1)


def run_native(episode, output):
    """Compute independent native fits; each model serially waits for readiness."""
    data = load_episode(episode)
    rows = []
    available_at = {model:0. for model in MODELS}
    previous_arrival = -1.
    for arrival in NATIVE_ARRIVALS:
        if arrival > data["time"][-1]+.02:
            continue
        window_started = time.perf_counter()
        window, events, packet_hash, start = make_window(data, float(arrival), min(8., arrival))
        window_s = time.perf_counter()-window_started
        incremental_integration_s = float(data["integration_work"][(data["time"] > previous_arrival)
            & (data["time"] <= arrival)].sum())
        for model in MODELS:
            fit = attempt(window, model)
            # Each method is a separate replay; only its own compute queue matters.
            compute_start = max(float(arrival), available_at[model])
            total_s = incremental_integration_s+window_s+fit["fit_elapsed_s"]
            available_at[model] = compute_start+total_s
            row = to_row(episode.name, "native", float(arrival), model, fit, events, packet_hash, start, total_s)
            row.update(compute_start_s=compute_start, publish_s=available_at[model],
                incremental_integration_s=incremental_integration_s, window_build_s=window_s,
                initial_no_output_until_s=2., native_priors_file_read=False)
            rows.append(row)
        previous_arrival = arrival
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output/(episode.name+"__native.csv"), index=False)
    return len(rows)


def run_stress(episode, output, arrival):
    """Fit all fixed wrong-prior cases once from the same sensor candidate."""
    data = load_episode(episode)
    priors = pd.read_csv(episode/"priors.csv")
    window_started = time.perf_counter()
    window, events, packet_hash, start = make_window(data, arrival, arrival)
    preparation_s = time.perf_counter()-window_started+float(data["integration_work"][data["time"] <= arrival].sum())
    seed, candidate_s = sensor_seed(window)
    rows = []
    # A ZYX heading changes the full start attitude while preserving common tilt.
    reference = window.start_rotation
    heading = np.arctan2(reference[1, 0], reference[0, 0])
    tilt = init.yaw_rotation(-heading) @ reference
    for model in MODELS:
        fit = attempt(window, model, seed=seed)
        row = to_row(episode.name, "native", arrival, model, fit, events, packet_hash,
            start, preparation_s+candidate_s+fit["fit_elapsed_s"])
        row.update(candidate_s=candidate_s, comparison_semantics="native_sensor_only_snapshot",
            prior_time_s=0., confidence="none")
        rows.append(row)
    for case in priors.itertuples(index=False):
        prior = StartPrior(np.array([case.px, case.py, case.pz]),
            init.yaw_rotation(case.yaw_rad) @ tilt, case.pos_std_m, case.yaw_std_rad)
        for model in MODELS:
            fit = attempt(window, model, prior, seed)
            wall_s = preparation_s+candidate_s+fit["fit_elapsed_s"]
            row = to_row(episode.name, str(case.case_id), arrival, model, fit, events, packet_hash, start, wall_s)
            row.update(candidate_s=candidate_s, comparison_semantics="common_soft_prior_MAP_extension",
                prior_time_s=0., pos_std_m=case.pos_std_m, yaw_std_rad=case.yaw_std_rad,
                confidence=case.confidence, magnitude_m=case.magnitude_m,
                direction=case.direction, yaw_delta_deg=case.yaw_delta_deg)
            rows.append(row)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output/(episode.name+f"__stress_{arrival:g}.csv"), index=False)
    return len(rows)


def summarize(streams, output):
    """Consolidate fits and explicit native-reset references without duplicating fits."""
    native_files = sorted(output.glob("*__native.csv"))
    stress_files = sorted(output.glob("*__stress_*.csv"))
    if native_files:
        native = pd.concat([pd.read_csv(path) for path in native_files], ignore_index=True)
        native.to_csv(output/"native.csv", index=False)
    if stress_files:
        pd.concat([pd.read_csv(path) for path in stress_files], ignore_index=True).to_csv(output/"stress.csv", index=False)
    stress_episodes = {path.name.split("__")[0] for path in stress_files}
    consumed_episodes = {path.name.split("__")[0] for path in native_files+stress_files}
    references = []
    for episode in sorted(streams.iterdir()):
        if episode.name not in stress_episodes or not (episode/"priors.csv").exists():
            continue
        for case in pd.read_csv(episode/"priors.csv").case_id:
            for arrival in STRESS_ARRIVALS:
                for model in MODELS:
                    references.append(dict(episode=episode.name, case_id=case, method=model,
                        arrival_s=arrival, native_case_id="native", source_csv="stress.csv",
                        semantics="native_unconditional_sensor_reset_ignores_prior",
                        independent_repeated_run=False))
    pd.DataFrame(references).to_csv(output/"native_reset_references.csv", index=False)
    hashes = {str(path):digest(path) for episode in sorted(streams.iterdir()) if episode.name in consumed_episodes
              for name in ("imu.csv", "radio.csv", "anchors.csv", "meta.json", "priors.csv")
              if (path := episode/name).exists()}
    protocol = dict(models=MODELS, native_arrivals=NATIVE_ARRIVALS.tolist(), native_window_s="min(arrival,8)",
        stress_arrivals=STRESS_ARRIVALS, stress_window="[0,arrival]; fixed t=0 prior",
        common_range_calibration_already_applied=True, raw_sensor_rotation=False,
        prior_type="Gaussian absolute position; principal SO(3) logarithm, common tilt preserved",
        cv_starts=2, nonlinear_refinements=1, max_nfev=120, bias_std_mps2=.10,
        native_reset_references_are_not_independent_runs=True, hashes=hashes,
        source_sha256={p.name:digest(p) for p in (Path(__file__), Path(__file__).with_name("solver.py"))})
    protocol["timing"] = "Measured held-IMU integration, window construction, CV/lift and one raw fit; file I/O and prefix calibration excluded. Native publish_s includes serial per-method queue."
    (output/"protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")


def main():
    """Run the authorized full paired matrix with a bounded worker pool."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--streams", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--mode", choices=("native", "stress", "all", "summarize"), default="all")
    parser.add_argument("--episodes", nargs="*")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    episodes = [path for path in sorted(args.streams.iterdir()) if (path/"imu.csv").exists()
                and (not args.episodes or path.name in args.episodes)]
    if args.mode != "summarize":
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            jobs = []
            for episode in episodes:
                if args.mode in ("native", "all"):
                    jobs.append(pool.submit(run_native, episode, args.output))
                if args.mode in ("stress", "all") and (episode/"priors.csv").exists():
                    jobs.extend(pool.submit(run_stress, episode, args.output, endpoint) for endpoint in STRESS_ARRIVALS)
            for future in as_completed(jobs):
                print("Completed", future.result(), "fits", flush=True)
    summarize(args.streams, args.output)


if __name__ == "__main__":
    main()
