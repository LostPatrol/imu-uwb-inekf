"""Independent UTIL TDoA physical-heading stress replay of the unchanged author core.

Truth is confined to exported evaluation poses and controlled XY/yaw prior centres.
"""

from __future__ import annotations

import argparse
from concurrent.futures import as_completed
import hashlib
import importlib.util
import itertools
import json
import os
from pathlib import Path
import sys
import time
import warnings
from functools import lru_cache
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

# Each worker uses one numerical thread and a preferred performance core.
for _variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_variable] = "1"
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp
from threadpoolctl import threadpool_limits

# Frozen paths, recordings, physical calibration and evaluator schedules.
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DATA = Path("E:/datasets/dataset/flight-dataset")
AUTHOR = REPO.parent / "imu-uwb-inekf - backup/scripts/estimation/inekf.py"
NAMES = tuple(f"const{k}-trial4-tdoa2" + ("-traj1" if k == 4 else "") for k in range(1, 5))
LEVER = np.array([-.01245, .00127, .0908])
RING = {(7, 0), (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)}
DURATION = 15.
# This release follows the shared 1.5 s sensor prefix, not a truth-aligned pose.
PREFIX = 1.5
GRID = np.arange(0., DURATION+.001, .5)
spec = importlib.util.spec_from_file_location("toronto_original_inekf", AUTHOR)
author = importlib.util.module_from_spec(spec)
spec.loader.exec_module(author)
THREAD_LIMIT = threadpool_limits(1)
sys.path.insert(0, str(REPO / "experiments/split_handoff"))
from cpu_replay import preferred_processors, _prefer_worker


def digest(path):
    """Hash immutable input and original source bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    """Write reproducible metadata in UTF-8."""
    path.write_text(json.dumps(value, indent=2, allow_nan=True), encoding="utf8")


@lru_cache(maxsize=4)
def load_sensors(name):
    """Read only IMU/radio columns and surveyed anchors; pose columns stay outside."""
    path = DATA / "csv-data" / name.split("-")[0] / (name + ".csv")
    columns = ["t_acc", "acc_x", "acc_y", "acc_z", "t_gyro", "gyro_x", "gyro_y", "gyro_z", "t_tdoa", "idA", "idB", "tdoa_meas"]
    frame = pd.read_csv(path, usecols=columns)
    acc = frame[columns[:4]].dropna().to_numpy(float)
    gyro = frame[columns[4:8]].dropna().to_numpy(float)
    radio = frame[columns[8:]].dropna().to_numpy(float)
    radio = radio[np.array([tuple(pair) in RING for pair in radio[:, 1:3]])]
    radio = radio[np.argsort(radio[:, 0], kind="stable")]
    imu = np.column_stack([acc[:, 1:], *[np.interp(acc[:, 0], gyro[:, 0], gyro[:, i]) for i in (1, 2, 3)]])
    anchors = np.load(DATA / "survey-results" / ("anchor_" + name.split("-")[0] + ".npz"))["an_pos"]
    return dict(name=name, times=acc[:, 0], imu=imu, radio=radio, anchors=anchors, origin=float(acc[0, 0]))


def sensor_initial(data):
    """Use 1.5 s arrived TDoA CV fit, gravity tilt and optional stationary gyro mean."""
    start = data["origin"] + PREFIX
    prefix = data["imu"][data["times"] <= start]
    force = np.mean(prefix[:max(1, len(prefix)//3), :3], axis=0)
    rotation = Rotation.align_vectors([[0., 0., 1.]], [force/np.linalg.norm(force)])[0].as_matrix()
    radio = data["radio"][data["radio"][:, 0] <= start]
    tau = radio[:, 0] - start
    a, b = data["anchors"][radio[:, 1].astype(int)], data["anchors"][radio[:, 2].astype(int)]
    def residual(x):
        """Fit hyperbolic antenna positions without position or heading truth."""
        tag = x[:3] + tau[:, None]*x[3:]
        return (np.linalg.norm(tag-b, axis=1)-np.linalg.norm(tag-a, axis=1)-radio[:, 3])/.4
    fit = least_squares(residual, np.r_[np.mean(data["anchors"], axis=0), np.zeros(3)], loss="huber", f_scale=2, max_nfev=60)
    state = np.eye(5)
    state[:3, :3], state[:3, 3], state[:3, 4] = rotation, fit.x[3:], fit.x[:3]-rotation@LEVER
    bias = np.zeros(6)
    stationary = np.max(np.abs(np.linalg.norm(prefix[:, :3], axis=1)-1)) < .05 and np.max(np.linalg.norm(prefix[:, 3:], axis=1)) < 2
    if stationary:
        bias[3:] = np.radians(np.mean(prefix[:, 3:], axis=0))
    return start, state, bias, dict(sensor_bootstrap_s=PREFIX, stationary=bool(stationary), radio_cv_success=bool(fit.success))


def covariance(state, mode):
    """Map physical world XY/yaw uncertainty into the author's invariant coordinates."""
    position_std, yaw_std = (20., 120.) if mode == "wide" else (.1, 5.)
    # World-additive velocity/position, world-left small rotation; bias order is native.
    std = np.r_[np.radians([5., 5., yaw_std]), [.5]*3, [position_std, position_std, .5], [1e-4]*3, [1e-6]*3]
    physical = np.diag(std**2)
    mapping = np.eye(15)
    mapping[3:6, :3] = author.SO3_wedge(state[:3, 3])
    mapping[6:9, :3] = author.SO3_wedge(state[:3, 4])
    return mapping@physical@mapping.T, physical


def replay(data, state, prior, bias, start, duration):
    """Call untouched predict/correct on literal author event scheduling, recording availability."""
    estimator = author.InEKF(state.copy(), prior.copy(), bias.copy())
    estimator.w_tdoa = np.array([np.sqrt(.13)])
    events = np.unique(np.r_[data["times"], data["radio"][:, 0]])
    events = events[events <= start+duration+1e-9]
    imu_at, radio_at = {}, {}
    for i, stamp in enumerate(data["times"]):
        imu_at.setdefault(float(stamp), i)
    for i, stamp in enumerate(data["radio"][:, 0]):
        radio_at.setdefault(float(stamp), i)
    previous_imu_time, last_output, available = start, -np.inf, start
    output, accepted, rejected, failure = [], 0, 0, None
    launched = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        try:
            for k in range(max(1, np.searchsorted(events, start, side="right")), len(events)):
                previous, stamp = float(events[k-1]), float(events[k])
                if previous <= start:
                    continue
                tick = time.perf_counter()
                if previous in imu_at:
                    dt = stamp-previous_imu_time
                    previous_imu_time = stamp
                    estimator.predict(data["imu"][imu_at[previous]], dt)
                if previous in radio_at:
                    _, _, _, discard = estimator.correct(data["radio"][radio_at[previous], 1:], data["anchors"])
                    accepted += int(not discard)
                    rejected += int(discard)
                available = max(stamp, available) + time.perf_counter()-tick
                if stamp-last_output >= .02:
                    output.append(np.r_[stamp-start, available-start, estimator.X[:3, 4], estimator.X[:3, 3], estimator.X[:3, :3].ravel()])
                    last_output = stamp
                if not np.all(np.isfinite(estimator.X)) or not np.all(np.isfinite(estimator.P)) or np.max(np.abs(estimator.X[:3, 4])) > 1000:
                    failure = "nonfinite_or_position_exceeds_1000m"
                    break
        except Exception as exception:
            failure = f"{type(exception).__name__}: {exception}"
    return np.asarray(output).reshape(-1, 17), estimator.X.copy(), estimator.bias.copy(), dict(accepted=accepted, rejected=rejected, failure=failure, runtime_s=time.perf_counter()-launched)


def export():
    """Export raw truth and sensor channels separately before controlled replay."""
    protocol = dict(recordings=NAMES, duration_s=DURATION, sensor_prefix_s=PREFIX, recovery_absolute_s=15., author_sha256=digest(AUTHOR), source_sha256=digest(Path(__file__)), data_kind="UTIL single-tag TDoA2; separate from TWR rankings", priors="4 XY magnitudes 1/5/10/20m x 4 directions +x/-x/+y/-y x yaw 0/90/180deg x wide/tight=96 per recording", truth_policy="Only XY and yaw construct controlled prior centres; all other state components from sensors", output_columns=["state_s", "available_s", "x", "y", "z", "vx", "vy", "vz"]+[f"R{i}{j}" for i in range(3) for j in range(3)])
    for name in NAMES:
        directory = HERE / "data" / name
        directory.mkdir(parents=True, exist_ok=True)
        raw = DATA / "csv-data" / name.split("-")[0] / (name+".csv")
        frame = pd.read_csv(raw)
        frame[["t_pose", "pose_x", "pose_y", "pose_z", "pose_qx", "pose_qy", "pose_qz", "pose_qw"]].dropna().to_csv(directory/"truth.csv", index=False)
        sensors = load_sensors(name)
        pd.DataFrame(np.column_stack([sensors["times"], sensors["imu"]]), columns=["t", "ax_g", "ay_g", "az_g", "gx_deg_s", "gy_deg_s", "gz_deg_s"]).to_csv(directory/"imu.csv", index=False)
        pd.DataFrame(sensors["radio"], columns=["t", "idA", "idB", "tdoa_m"]).to_csv(directory/"radio.csv", index=False)
        pd.DataFrame(sensors["anchors"], columns=["x", "y", "z"]).rename_axis("anchor_id").to_csv(directory/"anchors.csv")
        start, state, bias, bootstrap = sensor_initial(sensors)
        prior, _ = covariance(state, "wide")
        _, recovered, recovered_bias, info = replay(sensors, state, prior, bias, start, 15.-start)
        np.savez_compressed(directory/"sensor_base.npz", startup_state=state, startup_bias=bias, startup_time=start, recovery_state=recovered, recovery_bias=recovered_bias, recovery_time=15.)
        survey = DATA/"survey-results"/("anchor_"+name.split("-")[0]+"_survey.txt")
        (directory/"survey.txt").write_bytes(survey.read_bytes())
        dump(directory/"meta.json", dict(raw_source=str(raw), raw_sha256=digest(raw), survey_source=str(survey), survey_sha256=digest(survey), lever_m=LEVER.tolist(), vicon_frame="surveyed world", bootstrap=bootstrap, recovery_sensor_base=info))
    dump(HERE/"protocol.json", protocol)


@lru_cache(maxsize=4)
def truth_for(name):
    """Read exported evaluation poses only in the prior-builder and scorer."""
    truth = pd.read_csv(HERE/"data"/name/"truth.csv").to_numpy(float)
    unique = np.r_[True, np.diff(truth[:, 0]) > 0]
    truth = truth[unique]
    return truth, Slerp(truth[:, 0], Rotation.from_quat(truth[:, 4:8]))


def truth_at(name, times):
    """Interpolate unaligned Vicon position and body orientation on the evaluation clock."""
    truth, slerp = truth_for(name)
    times = np.clip(times, truth[0, 0], truth[-1, 0])
    position = np.column_stack([np.interp(times, truth[:, 0], truth[:, i]) for i in (1, 2, 3)])
    rotation = slerp(times).as_matrix()
    return position, np.arctan2(rotation[:, 1, 0], rotation[:, 0, 0])


def score(name, start, output):
    """Compare latest available held outputs against current GT every 0.5 s, without alignment."""
    gt_position, gt_yaw = truth_at(name, start+GRID)
    index = np.searchsorted(output[:, 1], GRID, side="right")-1 if len(output) else np.full(len(GRID), -1)
    valid = index >= 0
    xy, yaw, age = np.full(len(GRID), np.nan), np.full(len(GRID), np.nan), np.full(len(GRID), np.nan)
    if np.any(valid):
        rows = output[index[valid]]
        xy[valid] = np.linalg.norm(rows[:, 2:4]-gt_position[valid, :2], axis=1)
        rotations = rows[:, 8:17].reshape(-1, 3, 3)
        estimate_yaw = np.arctan2(rotations[:, 1, 0], rotations[:, 0, 0])
        yaw[valid] = np.degrees(np.abs(np.arctan2(np.sin(estimate_yaw-gt_yaw[valid]), np.cos(estimate_yaw-gt_yaw[valid]))))
        age[valid] = GRID[valid]-rows[:, 0]
    result = dict(first_output_state_s=float(output[0, 0]) if len(output) else None, first_output_available_s=float(output[0, 1]) if len(output) else None, samples_available=int(valid.sum()), max_output_age_s=float(np.nanmax(age)) if np.any(valid) else None)
    for threshold in (.25, .5, 1.):
        label = str(threshold).replace(".", "p")
        eligible = np.isfinite(xy) & (xy <= threshold)
        confirmed = [i for i in range(2, len(GRID)) if np.all(eligible[i-2:i+1])]
        i = confirmed[0] if confirmed else None
        result.update({f"capture_{label}":i is not None, f"enter_{label}_s":float(GRID[i-2]) if i is not None else None, f"confirm_{label}_s":float(GRID[i]) if i is not None else None})
    for endpoint in (2., 10., 15.):
        i = int(round(endpoint/.5))
        result.update({f"xy_{endpoint:g}s_m":float(xy[i]), f"yaw_{endpoint:g}s_deg":float(yaw[i])})
    return result, np.column_stack([GRID, valid, xy, yaw, age])


def run_case(job):
    """Supply only perturbed XY/yaw centres plus sensor-derived other state to the author core."""
    name, magnitude, direction, yaw_offset, mode = job
    key = f"{name}__{magnitude:g}m_{direction}_yaw{yaw_offset:g}_{mode}" if mode != "sensor_bootstrap" else name+"__sensor_bootstrap"
    directory = HERE/"results"
    checkpoint = directory/(key+".json")
    if checkpoint.exists():
        return json.loads(checkpoint.read_text(encoding="utf8"))
    data = load_sensors(name)
    base = np.load(HERE/"data"/name/"sensor_base.npz")
    if mode == "sensor_bootstrap":
        state, bias, start = base["startup_state"].copy(), base["startup_bias"].copy(), float(base["startup_time"])
    else:
        state, bias, start = base["recovery_state"].copy(), base["recovery_bias"].copy(), float(base["recovery_time"])
        gt_position, gt_yaw = truth_at(name, np.array([start]))
        offsets = {"px":(1., 0.), "nx":(-1., 0.), "py":(0., 1.), "ny":(0., -1.)}
        state[:2, 4] = gt_position[0, :2]+magnitude*np.array(offsets[direction])
        sensor_yaw = np.arctan2(state[1, 0], state[0, 0])
        state[:3, :3] = Rotation.from_euler("z", gt_yaw[0]+np.radians(yaw_offset)-sensor_yaw).as_matrix()@state[:3, :3]
    prior, physical = covariance(state, "wide" if mode == "sensor_bootstrap" else mode)
    outputs, _, _, info = replay(data, state, prior, bias, start, DURATION)
    metrics, grid = score(name, start, outputs)
    np.savez_compressed(directory/(key+".npz"), outputs=outputs, evaluation_grid=grid, initial_state=state, initial_bias=bias, initial_invariant_covariance=prior, physical_covariance=physical)
    row = dict(key=key, name=name, magnitude_m=magnitude, direction=direction, yaw_offset_deg=yaw_offset, mode=mode, start_absolute_s=start, **info, **metrics)
    dump(checkpoint, row)
    return row


def summarize(records):
    """Retain failure rows and visualize physical yaw/XY separately from TWR results."""
    frame = pd.DataFrame(records)
    frame.to_csv(HERE/"metrics.csv", index=False)
    frame.groupby(["mode", "yaw_offset_deg"], dropna=False).agg(cases=("key", "size"), captures=("capture_0p5", "sum"), median_xy_10s_m=("xy_10s_m", "median"), median_yaw_10s_deg=("yaw_10s_deg", "median"), median_runtime_s=("runtime_s", "median")).to_csv(HERE/"summary.csv")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for mode, group in frame.groupby("mode"):
        if mode == "sensor_bootstrap":
            continue
        stats = group.groupby("yaw_offset_deg")
        axes[0].plot(stats["capture_0p5"].mean().index, stats["capture_0p5"].mean(), "o-", label=mode)
        axes[1].plot(stats["xy_10s_m"].median().index, stats["xy_10s_m"].median(), "o-", label=mode)
        axes[2].plot(stats["yaw_10s_deg"].median().index, stats["yaw_10s_deg"].median(), "o-", label=mode)
    for axis, title in zip(axes, ("XY capture <=0.5m, held 1s", "10s XY error, median (m)", "10s true yaw error, median (deg)")):
        axis.set(title=title, xlabel="Injected yaw offset (deg)")
        axis.grid(alpha=.3)
        axis.legend()
    fig.suptitle("Independent UTIL single-tag TDoA2: unchanged Oursland core")
    fig.tight_layout()
    fig.savefig(HERE/"overview.png", dpi=160)
    plt.close(fig)


def main():
    """Export sources or run the frozen physical-intervention schedule."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    if args.export:
        export()
    if args.run or args.pilot:
        (HERE/"results").mkdir(parents=True, exist_ok=True)
        jobs = [(name, 0., "none", 0., "sensor_bootstrap") for name in NAMES]
        jobs += list(itertools.product(NAMES, (1., 5., 10., 20.), ("px", "nx", "py", "ny"), (0., 90., 180.), ("wide", "tight")))
        if args.pilot:
            jobs = jobs[:1]+jobs[4:6]
        # Pin worker numerical threads to distinct preferred fast physical cores.
        ctx = mp.get_context("spawn")
        processors = preferred_processors()[:4]
        records = []
        launched = time.perf_counter()
        with ProcessPoolExecutor(max_workers=4, mp_context=ctx, initializer=_prefer_worker, initargs=(ctx.Value("i", 0), processors, True)) as pool:
            for future in as_completed([pool.submit(run_case, job) for job in jobs]):
                records.append(future.result())
                print(f"DONE {len(records)}/{len(jobs)} {records[-1]['key']} elapsed={time.perf_counter()-launched:.1f}s", flush=True)
        dump(HERE/("pilot_execution.json" if args.pilot else "execution.json"), dict(workers=4, preferred_processors=processors, wall_s=time.perf_counter()-launched, cases=len(records), author_sha256_after=digest(AUTHOR)))
        summarize(records)
    if args.summary:
        summarize([json.loads(path.read_text(encoding="utf8")) for path in (HERE/"results").glob("*.json")])


if __name__ == "__main__":
    main()
