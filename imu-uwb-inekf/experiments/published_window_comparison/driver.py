"""Replay unmodified Pfeiffer author classes and matched Delft sensor windows.

Native author simulation uses truth-derived altitude and range variance. The
window adaptation explicitly removes both and preserves the author dynamics.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import time
import urllib.request

# Keep each worker's numerical kernels single threaded on the shared desktop.
for _name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np
import pandas as pd

# All new outputs remain beside this driver; upstream sources are read only.
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
AUTHOR = PROJECT.parent / "third_party" / "pfeiffer_uwb_simulator_2021"
MODELS = PROJECT / "experiments" / "radio_inertial_models"
DATASET = Path(r"E:\datasets\tu-delft-uwb-mhe\data_v1\UWB dataset")
# This is the public repository revision whose normalized sources are verified.
AUTHOR_REVISION = "cb816bdb291ac933577427448883acfa7c457c33"
# Physical windows and clock correction match the paired acquisition evidence.
WINDOWS = [(e, w) for e in (5., 10., 20., 30.) for w in (2., 4., 8.) if w <= e]
PREFIX_S = 1.5
SENSOR_DELAY_S = .20
MOTIONS = ("triangle", "octagon", "hourglass", "random", "star", "square")
# Use the paper's released Nanchors script, not differing GUI class defaults.
MHE_SETTINGS = dict(enable=True, rate=50, N_max=20, iterations=1, alpha=.015,
                    mu=10., ransac_iterations=10, ransac_threshold=1.7,
                    ransac_fraction=.4, attitude_rate=250)
EKF_SETTINGS = dict(enable=True, rate=100, outlierThreshold=1500)
# Native Crazyflie airframe defaults; thrust is accelerometer z times mass.
MASS_KG = .03
DRAG = [.15, .15, 0.]
# Fixed random seed controls author RANSAC and native synthetic altimeter noise.
RANDOM_SEED = 20261112
# Descriptive endpoint success thresholds, not estimator acceptance gates.
SUCCESS_M = .5

sys.path.insert(0, str(AUTHOR))
sys.path.insert(0, str(PROJECT / "experiments" / "recovery_bootstrap"))
from UWBsim.airframe.drone import Drone, DroneParams
from UWBsim.estimators import EstimatorParams
from UWBsim.estimators.mhe import MHE, MHE_Params
from UWBsim.estimators.ekf import EKF, EKF_Params
from UWBsim.utils.dataTypes import TWR_meas
from UWBsim.utils.uwb_ranging import RangingParams, RangingSource, RangingType
from run_research import load_real_log, load_anchors, estimate_initial_rotation_and_bias, preintegrate

# Load the existing joint control without confusing its radio module with itself.
_spec = importlib.util.spec_from_file_location("published_joint", MODELS / "joint_branch" / "study.py")
joint = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = joint
_spec.loader.exec_module(joint)


def dump(path, value):
    """Write UTF-8 protocol evidence with ordinary Python scalar values."""
    def convert(item):
        """Normalize nested NumPy values for strict JSON serialization."""
        if isinstance(item, dict):
            return {str(k): convert(v) for k, v in item.items()}
        if isinstance(item, (tuple, list, np.ndarray)):
            return [convert(x) for x in item]
        if isinstance(item, (np.integer, np.bool_)):
            return item.item()
        if isinstance(item, (float, np.floating)):
            return float(item) if np.isfinite(item) else None
        return item
    path.write_text(json.dumps(convert(value), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sources(output):
    """Verify all imported author Python files against one official revision."""
    rows = []
    for path in sorted((AUTHOR / "UWBsim").rglob("*.py")):
        relative = path.relative_to(AUTHOR).as_posix()
        url = f"https://raw.githubusercontent.com/SUPfeiffer/uwb-simulator/{AUTHOR_REVISION}/{relative}"
        remote = urllib.request.urlopen(url, timeout=30).read().decode("utf-8").replace("\r\n", "\n")
        local = path.read_text(encoding="utf-8")
        rows.append(dict(path=relative, url=url, normalized_equal=local == remote,
                         local_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                         normalized_sha256=hashlib.sha256(local.encode()).hexdigest()))
    assert all(row["normalized_equal"] for row in rows)
    pd.DataFrame(rows).to_csv(output / "author_sources.csv", index=False)
    relative = "publication/headless_Nanchors.py"
    url = f"https://raw.githubusercontent.com/SUPfeiffer/uwb-simulator/{AUTHOR_REVISION}/{relative}"
    (output / "official_headless_Nanchors.py.txt").write_bytes(urllib.request.urlopen(url, timeout=30).read())


def split():
    """Use first recordings for static calibration and second for evaluation."""
    train, test = [], []
    for motion in MOTIONS:
        files = sorted((DATASET / "twr").glob(f"*twr_{motion}.csv"))
        assert len(files) == 2
        train.append(files[0].name)
        test.append(files[1].name)
    return train, test


def calibrate(train, anchors):
    """Fit only per-anchor medians and MAD scales on six earlier recordings."""
    pieces = [[] for _ in range(8)]
    for name in train:
        log = load_real_log(DATASET, name, sensor_delay_s=SENSOR_DELAY_S)
        for anchor in range(8):
            keep = log.changed[:, anchor] & (log.ranges[:, anchor] > 0)
            pieces[anchor].extend(log.ranges[keep, anchor] - np.linalg.norm(
                log.ground_truth_position[keep] - anchors[anchor], axis=1))
    median = np.array([np.median(x) for x in pieces])
    sigma = np.clip([1.4826 * np.median(np.abs(np.asarray(x) - b))
                     for x, b in zip(pieces, median)], .08, .5)
    return dict(static=median, sigma=sigma)


def score(position, truth):
    """Return signed axis errors and descriptive endpoint success."""
    error = np.asarray(position) - truth
    finite = bool(np.isfinite(error).all())
    length = float(np.linalg.norm(error))
    return dict(px_m=position[0], py_m=position[1], pz_m=position[2],
                dx_m=error[0], dy_m=error[1], dz_m=error[2],
                xy_error_m=float(np.linalg.norm(error[:2])), z_error_m=float(abs(error[2])),
                error_3d_m=length, finite=finite, endpoint_below_0p5m=finite and length < SUCCESS_M)


def run_window(log, times, start, end, active, anchors, cal, method):
    """Call an unchanged author class with exactly the selected raw sensor rows."""
    estimator = (MHE(MHE_Params(**MHE_SETTINGS), MASS_KG, DRAG) if method == "Pfeiffer_MHE"
                 else EKF(EKF_Params(**EKF_SETTINGS), MASS_KG, DRAG))
    rate = MHE_SETTINGS["rate"] if method == "Pfeiffer_MHE" else EKF_SETTINGS["rate"]
    next_step = 1 / rate
    steps = 0
    events = 0
    last_time = float("nan")
    estimate = np.full(3, np.nan)
    failure = ""
    started = time.perf_counter()
    try:
        for k in range(start, end + 1):
            # The local nonnegative clock prevents a large first Mahony integration.
            local = float(times[k] - times[start])
            acc = log.imu[k, :3] * 9.813
            gyro = np.deg2rad(log.imu[k, 3:])
            estimator.addInputs(float(acc[2] * MASS_KG), gyro, acc, local)
            for anchor in np.flatnonzero(log.changed[k] & (log.ranges[k] > 0) & active):
                estimator.addMeasurement(TWR_meas(anchors[anchor], int(anchor),
                    float(log.ranges[k, anchor] - cal["static"][anchor]),
                    stdDev=float(cal["sigma"][anchor]), timestamp=local))
                events += 1
            if local + 1e-10 >= next_step or k == end:
                estimate = estimator.step(local).x
                next_step += 1 / rate
                steps += 1
                last_time = float(times[k])
    except (AssertionError, FloatingPointError, ValueError, np.linalg.LinAlgError) as error:
        failure = type(error).__name__ + ": " + str(error)
        estimate = np.full(3, np.nan)
    return dict(elapsed_s=time.perf_counter() - started, steps=steps, events_consumed=events,
                estimate_timestamp_s=last_time, completed=not failure,
                failure=failure, nfev=np.nan, **score(estimate, log.ground_truth_position[end]))


def solve_control(values, sensitivity, calibration):
    """Run the existing four-yaw BiasPrior solver and count its CV seed work."""
    tau, displacement, anchors, ranges, ids = values
    counts = []
    original = joint.shape.init.least_squares
    def counted(*args, **kwargs):
        """Include both constant-velocity solves in the reported evaluation count."""
        result = original(*args, **kwargs)
        counts.append(result.nfev)
        return result
    joint.shape.init.least_squares = counted
    try:
        cv = joint.radio.cv_candidate(tau, anchors, ranges - calibration["static"][ids],
                                      calibration["sigma"][ids])[0]
    finally:
        joint.shape.init.least_squares = original
    seeds = [np.r_[cv, yaw, np.zeros(3)] for yaw in np.linspace(-np.pi, np.pi, 4, endpoint=False)]
    x, fit = joint.solve(values, sensitivity, "BiasPrior", "static", calibration, seeds)
    fit["cv_nfev"] = sum(counts)
    fit["nfev"] += sum(counts)
    return x, fit


def common_record(name, calibration, output, limit_windows):
    """Evaluate all common-input arms on one flight with matching packet hashes."""
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    anchors = load_anchors(DATASET)
    full = load_real_log(DATASET, name, sensor_delay_s=SENSOR_DELAY_S)
    rotation, bias = estimate_initial_rotation_and_bias(full)
    selected = full.time >= PREFIX_S
    times = full.time[selected] - full.time[selected][0]
    # The author classes receive these same raw IMU/radio rows without correction.
    from types import SimpleNamespace
    log = SimpleNamespace(imu=full.imu[selected], ranges=full.ranges[selected],
                          changed=full.changed[selected], ground_truth_position=full.ground_truth_position[selected])
    rotations, displacement, velocity = preintegrate(times, log.imu, rotation, bias)
    raw = dict(static=np.zeros(8), sigma=np.full(8, .25))
    rows = []
    for endpoint, width in WINDOWS[:limit_windows]:
        end = np.searchsorted(times, endpoint, side="right") - 1
        start = np.searchsorted(times, times[end] - width, side="left")
        for layout in ("all8", "lower4"):
            active = np.ones(8, dtype=bool)
            if layout == "lower4":
                active[:] = False
                active[np.argsort(anchors[:, 2])[:4]] = True
            index, ids = np.nonzero(log.changed[start:end+1] & (log.ranges[start:end+1] > 0) & active)
            index += start
            packet_hash = hashlib.sha256(np.column_stack((index, ids, log.ranges[index, ids])).tobytes()).hexdigest()
            meta = dict(sequence=name, endpoint_s=endpoint, duration_s=width, layout=layout,
                actual_endpoint_s=float(times[end]), actual_start_s=float(times[start]),
                events=len(index), packet_sha256=packet_hash, imu_rows=end-start+1,
                imu_sha256=hashlib.sha256(log.imu[start:end+1].tobytes()).hexdigest(),
                protocol="B_common_windows", position_truth_in_estimator=False,
                initial_position="zero", initial_heading="author_zero_yaw_prior")
            for cal_name, cal in (("raw", raw), ("static", calibration)):
                for method in ("Pfeiffer_MHE", "Pfeiffer_EKF"):
                    random.seed(RANDOM_SEED)
                    result = run_window(log, times, start, end, active, anchors, cal, method)
                    rows.append(dict(**meta, method=method, calibration=cal_name, **result))
                tau = times[index] - times[end]
                d = displacement[index] - displacement[end] - tau[:, None] * velocity[end]
                sensitivity = joint.shape.reanchor_integral(rotations[start:end], times[start:end+1], times[index])
                started = time.perf_counter()
                x, fit = solve_control((tau, d, anchors[ids], log.ranges[index, ids], ids), sensitivity, cal)
                rows.append(dict(**{**meta, "initial_position": "range_CV_two_starts", "initial_heading": "unknown_four_yaw_starts"},
                    method="BiasPrior", calibration=cal_name, elapsed_s=time.perf_counter()-started,
                    steps=0, events_consumed=len(index), estimate_timestamp_s=float(times[end]),
                    completed=True, failure="", nfev=fit["nfev"], cv_nfev=fit["cv_nfev"],
                    solver_success=fit["success"], **score(x[:3], log.ground_truth_position[end])))
        pd.DataFrame(rows).to_csv(output / (Path(name).stem + "__B.csv"), index=False)
        print(f"B {name} endpoint={endpoint:g} duration={width:g}: {len(rows)} rows", flush=True)
    return len(rows)


def native_record(name, output):
    """Replay the actual author Drone scheduler through the last common endpoint."""
    anchors = load_anchors(DATASET)
    full = load_real_log(DATASET, name, sensor_delay_s=SENSOR_DELAY_S)
    # Native t=0 is the first OptiTrack fix; common t=0 excludes prefix and delay.
    common_zero_in_native = float(full.time[np.flatnonzero(full.time >= PREFIX_S)[0]] + SENSOR_DELAY_S)
    rows, traces = [], []
    for layout in ("all8", "lower4"):
        active = np.ones(8, dtype=bool)
        if layout == "lower4":
            active[:] = False
            active[np.argsort(anchors[:, 2])[:4]] = True
        for method in ("Pfeiffer_MHE", "Pfeiffer_EKF"):
            random.seed(RANDOM_SEED)
            np.random.seed(RANDOM_SEED)
            params = EstimatorParams(MHE_Params(**{**MHE_SETTINGS, "enable": method == "Pfeiffer_MHE"}),
                EKF_Params(**{**EKF_SETTINGS, "enable": method == "Pfeiffer_EKF"}))
            drone = Drone(DroneParams(logfile=str(DATASET / "twr" / name), altitude_enable=True), params,
                RangingParams(source=RangingSource.LOG, rtype=RangingType.TWR,
                              anchor_positions=anchors, anchor_enable=active))
            key = "mhe" if method == "Pfeiffer_MHE" else "ekf"
            estimator = drone.estimators[key]
            old_step = estimator.step
            timings = []
            def timed_step(*args):
                """Measure the author update without changing any estimator values."""
                tic = time.perf_counter()
                result = old_step(*args)
                timings.append(time.perf_counter() - tic)
                return result
            estimator.step = timed_step
            failure = ""
            trajectory = []
            previous = -1.
            started = time.perf_counter()
            try:
                for tick in range(int((30.02 + common_zero_in_native) * 1000) + 1):
                    if not drone.step(tick / 1000):
                        break
                    if drone.time != previous:
                        trajectory.append((drone.time, *drone.state_estimate[key].x, *drone.state_true.x,
                                           len(timings), sum(timings), drone.state_estimate[key].timestamp))
                        previous = drone.time
            except (AssertionError, ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
                failure = type(error).__name__ + ": " + str(error)
            finally:
                drone._stop()
            elapsed = time.perf_counter() - started
            trajectory = np.asarray(trajectory)
            for endpoint, width in WINDOWS:
                target = endpoint + common_zero_in_native
                k = np.searchsorted(trajectory[:, 0], target, side="right") - 1
                record = trajectory[k]
                covered = bool(target - record[0] <= .025)
                position = record[1:4] if covered else np.full(3, np.nan)
                # Native scores use the same author clock/truth as the official code.
                rows.append(dict(protocol="A_native_author", sequence=name, endpoint_s=endpoint,
                    duration_s=width, layout=layout, method=method, calibration="author_truth_sigma_plus_altitude",
                    actual_endpoint_s=record[0]-common_zero_in_native, actual_start_s=-common_zero_in_native,
                    estimate_timestamp_s=record[9]-common_zero_in_native,
                    native_time_s=record[0], native_elapsed_s=elapsed, elapsed_s=record[8], steps=int(record[7]),
                    completed=covered, failure=failure if not covered else "", position_truth_in_estimator=True,
                    initial_position="zero", initial_heading="author_zero_yaw_prior",
                    **score(position, record[4:7])))
            for record in trajectory:
                traces.append(dict(sequence=name, layout=layout, method=method, native_time_s=record[0],
                    native_estimate_time_s=record[9],
                    common_time_s=record[0]-common_zero_in_native, px_m=record[1], py_m=record[2], pz_m=record[3],
                    truth_x_m=record[4], truth_y_m=record[5], truth_z_m=record[6]))
            print(f"A {name} {layout} {method}: {elapsed:.2f}s {failure}", flush=True)
    pd.DataFrame(rows).to_csv(output / (Path(name).stem + "__A.csv"), index=False)
    pd.DataFrame(traces).to_csv(output / (Path(name).stem + "__A_trace.csv"), index=False)
    return len(rows)


def main():
    """Freeze provenance before dispatching independent flight-level jobs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--logs", type=int, default=6)
    parser.add_argument("--windows", type=int, default=len(WINDOWS))
    parser.add_argument("--arms", nargs="+", choices=["A", "B"], default=["A", "B"])
    parser.add_argument("--output", type=Path, default=HERE / "evidence")
    parser.add_argument("--verify-sources", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.verify_sources:
        sources(args.output)
    train, evaluation = split()
    calibration = calibrate(train, load_anchors(DATASET))
    dump(args.output / "protocol.json", dict(author_repo="https://github.com/SUPfeiffer/uwb-simulator",
        author_revision=AUTHOR_REVISION, paper_doi="10.1109/LRA.2021.3095519", dataset_doi="10.4121/14827680",
        training=train, evaluation=evaluation[:args.logs], windows=WINDOWS[:args.windows], executed_arms=args.arms,
        mhe=MHE_SETTINGS, ekf=EKF_SETTINGS, mass_kg=MASS_KG, drag=DRAG,
        static_calibration=calibration, random_seed=RANDOM_SEED, sensor_delay_s=SENSOR_DELAY_S,
        prefix_s=PREFIX_S, endpoint_success_m=SUCCESS_M, independent_blind_test=False,
        native_scope="Unmodified Drone and author classes; continuous first 31.7s; synthetic truth altimeter and online truth radio sigma; author native time scoring",
        common_scope="Unmodified classes, 2/4/8s cold replay, native .4s MHE horizon; no altimeter, no runtime truth sigma, zero position/velocity/yaw; raw or six-training-log static calibration",
        bias_control="Existing joint BiasPrior: four uniform yaw starts, two range CV fits, bias prior .10 m/s2, 120 evaluations per raw start; initial prefix roll/pitch and IMU bias calibration",
        coverage_scope="Success is finite endpoint below .5m; failure rows remain denominators; native repeated durations are same continuous endpoint",
        calibration_truth="Only six early logs for B calibration; later truth only valid-clock crop and scoring; A explicit truth aids",
        input_sha256={name:hashlib.sha256((DATASET / "twr" / name).read_bytes()).hexdigest() for name in train+evaluation},
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    jobs = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for name in evaluation[:args.logs]:
            if "B" in args.arms:
                jobs.append(pool.submit(common_record, name, calibration, args.output, args.windows))
            if "A" in args.arms:
                jobs.append(pool.submit(native_record, name, args.output))
        for future in as_completed(jobs):
            print("FINISHED", future.result(), flush=True)


if __name__ == "__main__":
    main()
