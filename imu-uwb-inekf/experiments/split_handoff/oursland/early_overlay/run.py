"""Overlay frozen early XY on the untouched author's native event backend.

Both branches share one full author object until recovery; reserved packets
are used only by the frozen frontend, then only strictly later packets recur.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import warnings

# Independent conditions use one BLAS thread and the shared CPU affinity pool.
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
OURS = HERE.parent
SPLIT = OURS.parent
sys.path.insert(0, str(OURS))
sys.path.insert(0, str(SPLIT))
import native_schedule as native
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from threadpoolctl import threadpool_limits

# These modules remain read-only, including every frozen frontend threshold.
spec = importlib.util.spec_from_file_location("frozen_early_overlay", SPLIT / "early_tdoa" / "run.py")
early = importlib.util.module_from_spec(spec)
spec.loader.exec_module(early)
THREAD_LIMIT = threadpool_limits(1)
# Preserve the primary matrix and label its complementary protocols separately.
PRIMARY_NAMES = ("const1-trial4-tdoa2", "const2-trial4-tdoa3", "const3-trial4-tdoa2", "const4-trial4-tdoa3-traj1")
EXTENSION_NAMES = ("const1-trial4-tdoa3", "const2-trial4-tdoa2", "const3-trial4-tdoa3", "const4-trial4-tdoa2-traj1")
NAMES = PRIMARY_NAMES + EXTENSION_NAMES
YAWS = (0., 180.)
WINDOWS = {"early": (20., 22.), "future": (26., 31.), "post": (26., 45.)}


def save_json(path, value):
    """Write sensor diagnostics and arrays as UTF-8 JSON."""
    path.write_text(json.dumps(native.ours.util.safe_json(value), indent=2), encoding="utf8")


def row(estimator, stamp):
    """Extract physical state without changing the imported author object."""
    return np.r_[stamp, estimator.X[:3, 4], estimator.X[:3, 3], estimator.X[:3, :3].ravel(), np.zeros(3)]


def bad(estimator):
    """Use the same explicit numerical-failure rule as the primary native run."""
    return (not np.all(np.isfinite(estimator.X)) or not np.all(np.isfinite(estimator.P))
            or np.max(np.abs(estimator.X[:3, 4])) > 1000)


def run_case(job):
    """Share the complete original prefix, then branch without reconstructing P."""
    name, yaw = job
    started = time.perf_counter()
    output = HERE / "results"
    key = f"{name}__yaw{yaw:g}"
    if (output / (key + ".json")).exists():
        return json.loads((output / (key + ".json")).read_text(encoding="utf8"))
    data = native.load_raw(name, False)
    data.pop("truth")
    common = native.ours.util.load(name, duration=45.)
    common.pop("truth")
    start, state, bias = native.ours.util.initial_state(common)
    state[:3, :3] = Rotation.from_euler("z", yaw, degrees=True).as_matrix() @ state[:3, :3]
    covariance = native.ORIGINAL_P.copy()
    baseline = native.ours.make_filter(state, covariance, bias)
    times, imu, radio = (data[k] for k in ("times", "imu", "radio"))
    events = np.unique(np.r_[times, radio[:, 0]])
    imu_at, radio_at = {}, {}
    for index, stamp in enumerate(times):
        imu_at.setdefault(float(stamp), index)
    for index, stamp in enumerate(radio[:, 0]):
        radio_at.setdefault(float(stamp), index)
    # Native original-main queue consumes only the first packet at a shared stamp.
    candidate_radio = radio[np.array(list(radio_at.values()), dtype=int)]
    candidate_data = dict(common, radio=candidate_radio, anchors=data["anchors"])
    origin = data["common_origin"]
    t10, t20 = origin + 10., origin + 20.
    check_indices = np.arange(np.searchsorted(common["times"], t20), len(common["times"]), 5)
    checks = common["times"][check_indices]
    check_index = 0
    previous_imu_time, last_output = start, -np.inf
    branch, trigger, failures, attempts = None, None, {"native": None, "early_reserved": None}, []
    histories = {"native": [], "early_reserved": []}
    preservation = {}
    snapshot = {}
    calls = {method: dict(predict=0, correct=0, accepted=0, rejected=0) for method in histories}
    first_post_packet = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        for k in range(max(1, np.searchsorted(events, start, side="right")), len(events)):
            previous, stamp = float(events[k - 1]), float(events[k])
            if previous <= start:
                continue
            if branch is None and previous >= t20:
                branch = copy.deepcopy(baseline)
                snapshot = dict(time=previous, X=branch.X.copy(), P=branch.P.copy(), bias=branch.bias.copy(),
                                X_hat=branch.X_hat.copy(), P_hat=branch.P_hat.copy(), previous_imu_time=previous_imu_time)
                calls["early_reserved"] = calls["native"].copy()
            dt = None
            if previous in imu_at:
                dt = stamp - previous_imu_time
                previous_imu_time = stamp
            for method, estimator in (("native", baseline), ("early_reserved", branch)):
                if estimator is None or failures[method] is not None:
                    continue
                try:
                    if dt is not None:
                        estimator.predict(imu[imu_at[previous]], dt)
                        calls[method]["predict"] += 1
                    if previous in radio_at and not t10 <= previous < t20:
                        permitted = method == "native" or (trigger is not None and previous > trigger)
                        if permitted:
                            if method == "early_reserved" and first_post_packet is None:
                                first_post_packet = previous
                            _, _, _, discarded = estimator.correct(radio[radio_at[previous], 1:], data["anchors"])
                            calls[method]["correct"] += 1
                            calls[method]["accepted"] += int(not discarded)
                            calls[method]["rejected"] += int(discarded)
                    if bad(estimator):
                        failures[method] = "nonfinite_or_position_exceeds_1000m"
                except Exception as exception:
                    failures[method] = type(exception).__name__ + ": " + str(exception)
            if branch is not None and trigger is None and failures["early_reserved"] is None:
                if check_index < len(checks) and stamp >= checks[check_index]:
                    while check_index < len(checks) and stamp >= checks[check_index]:
                        check_index += 1
                    window_start = float(common["times"][max(np.searchsorted(common["times"], t20),
                        np.searchsorted(common["times"], stamp - 2.))])
                    # Only raw packets already reached by the event queue can enter a check.
                    visible = dict(candidate_data, radio=candidate_radio[candidate_radio[:, 0] <= previous])
                    candidate, report = early.horizontal_candidate(visible, window_start, stamp, branch.X, branch.P)
                    report["author_queue_received_through_s"] = previous - origin
                    attempts.append(report)
                    if candidate is not None:
                        X, P, b, preservation = early.handoff_xy(branch.X, branch.P, branch.bias, candidate)
                        branch.X, branch.P, branch.bias = X, P, b
                        # This is the single external handoff boundary, not a change to author correct.
                        # Refresh its predicted chart here so a following radio-only event uses XY.
                        branch.X_hat, branch.P_hat = X.copy(), P.copy()
                        trigger = stamp
            if stamp - last_output >= .01:
                for method, estimator in (("native", baseline), ("early_reserved", branch)):
                    if estimator is None:
                        histories[method].append(row(baseline, stamp))
                    elif failures[method] is None:
                        histories[method].append(row(estimator, stamp))
                last_output = stamp
            if failures["native"] is not None and failures["early_reserved"] is not None:
                break
    arrays = {method: np.array(rows).reshape(-1, 19) for method, rows in histories.items()}
    arrays["early_switch"] = arrays["native"].copy() if trigger is None else np.vstack((
        arrays["native"][arrays["native"][:, 0] < trigger],
        arrays["early_reserved"][arrays["early_reserved"][:, 0] >= trigger]))
    artifact = output / (key + ".npz")
    np.savez_compressed(artifact, **arrays, **{"split_" + k: v for k, v in snapshot.items()},
                        initial_state=state, initial_covariance=covariance, initial_bias=bias)
    result = dict(name=name, yaw_offset_deg=yaw, origin=origin, trigger_s=None if trigger is None else trigger-origin,
        cohort="primary" if name in PRIMARY_NAMES else "extension",
        first_post_packet_s=None if first_post_packet is None else first_post_packet-origin,
        attempts=attempts, preservation=preservation, failures=failures, calls=calls, artifact=str(artifact),
        runtime_seconds=time.perf_counter()-started, pid=os.getpid())
    save_json(output / (key + ".json"), result)
    return native.ours.util.safe_json(result)


def score(records):
    """Evaluate all generated trajectories on fixed common grids after generation."""
    rows, validations = [], []
    for record in records:
        truth = native.ours.util.load(record["name"], duration=45.)["truth"]
        arrays = np.load(record["artifact"])
        old_path = OURS / "native_original_prior" / (f"{record['name']}__yaw{record['yaw_offset_deg']:g}__outage__sensor_only_original_prior__author.npz")
        current = arrays["native"]
        if old_path.exists():
            previous = np.load(old_path)["rows"]
            validations.append(dict(name=record["name"], yaw_offset_deg=record["yaw_offset_deg"],
                shape_equal=current.shape==previous.shape,
                max_abs_difference=None if current.shape!=previous.shape else float(np.max(np.abs(current-previous)))))
        for method in ("native", "early_reserved", "early_switch"):
            history = arrays[method]
            failure_method = "early_reserved" if method == "early_switch" and record["trigger_s"] is not None else method
            failed = record["failures"].get(failure_method, record["failures"]["native"]) is not None
            entry = dict(name=record["name"], yaw_offset_deg=record["yaw_offset_deg"], method=method,
                cohort=record["cohort"],
                triggered=record["trigger_s"] is not None, trigger_s=record["trigger_s"], failed=failed)
            for label, (left, right) in WINDOWS.items():
                values = native.ours.util.score_rows(truth, history, record["origin"]+left, record["origin"]+right-.05)
                entry.update({label+"_"+key: value for key, value in values.items()})
                entry[label+"_coverage"] = values["samples"] / round((right-left)/.05)
            rows.append(entry)
    frame = pd.DataFrame(rows)
    frame.to_csv(HERE / "results" / "metrics.csv", index=False)
    baseline = frame[frame.method=="native"].set_index(["name", "yaw_offset_deg"])
    for label in WINDOWS:
        for value in ("xy_rmse", "yaw_rmse_deg"):
            frame[label+"_"+value+"_minus_native"] = [item[label+"_"+value]-baseline.loc[(item["name"],item["yaw_offset_deg"]),label+"_"+value] for item in rows]
    frame.to_csv(HERE / "results" / "paired.csv", index=False)
    save_json(HERE / "results" / "baseline_validation.json", validations)
    columns = [label+"_"+value for label in WINDOWS for value in ("xy_rmse", "yaw_rmse_deg", "coverage")]
    frame.groupby(["cohort","method"])[columns].median().to_csv(HERE / "results" / "summary.csv")


def main():
    """Freeze rules before sensor generation and use all independent worker slots."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--score-only", action="store_true")
    args = parser.parse_args()
    output = HERE / "results"
    output.mkdir(parents=True, exist_ok=True)
    if args.score_only:
        score(json.loads((output / "records.json").read_text(encoding="utf8")))
        return
    protocol = dict(names=NAMES, primary_names=PRIMARY_NAMES, extension_names=EXTENSION_NAMES,
        yaws_deg=YAWS, workers=args.workers, backend="untouched original author class; original native queue",
        initial="same sensor-only X/bias and exact original invariant P; yaw world-z at initial0.6s; no truth",
        outage_s=[10.,20.], duration_s=45., score_windows_half_open_s=WINDOWS, scoring="common20Hz zero-order-hold; no alignment",
        thresholds=early.FROZEN, frontend_source=str(SPLIT/"early_tdoa"/"run.py"),
        frontend_sha256=native.ours.digest(SPLIT/"early_tdoa"/"run.py"), author_sha256=native.ours.digest(native.ours.AUTHOR),
        recovery="deepcopy complete author object when first native previous timestamp reaches20; preserve X/X_hat/P/P_hat/bias and event dt",
        buffer="first native packet per timestamp; recovery packets<=already reached previous event; rolling max2s; original100Hz checks every5 samples mapped to first native event >= check",
        gate="exact frozen early_tdoa.horizontal_candidate; original thresholds and native .14/.22 fit variance unchanged",
        handoff="exact frozen XY-only mean/covariance; retain R/v/z/bias and full nonXY physical covariance; clear XY cross; refresh X_hat/P_hat only at handoff",
        reuse="reserved branch never scalar-fuses20s..trigger; after handoff only packet timestamp strictly greater than trigger; baseline fuses native packets normally",
        fallback="if no trigger reserved branch remains inertial; optional display early_switch stays native before trigger or entire run if none",
        source_hashes={str(path):native.ours.digest(path) for path in (Path(__file__), OURS/"native_schedule.py", SPLIT/"delft"/"early_position.py")})
    save_json(HERE/"protocol.json", protocol)
    # Shared pool establishes P-core-first affinity and independent numerical threads.
    import cpu_replay
    from concurrent.futures import as_completed
    started = time.perf_counter()
    records = []
    with cpu_replay.replay_pool(args.workers) as pool:
        futures = [pool.submit(run_case, (name,yaw)) for name in NAMES for yaw in YAWS]
        for future in as_completed(futures):
            records.append(future.result())
            save_json(output/"records.json", records)
            print(f"DONE {len(records)}/16 {records[-1]['name']} yaw{records[-1]['yaw_offset_deg']:g} {time.perf_counter()-started:.1f}s",flush=True)
    save_json(output/"provenance.json", dict(wall_seconds=time.perf_counter()-started,workers=args.workers,
        author_source_modified=protocol["author_sha256"]!=native.ours.digest(native.ours.AUTHOR),
        frontend_modified=protocol["frontend_sha256"]!=native.ours.digest(SPLIT/"early_tdoa"/"run.py")))
    score(records)


if __name__ == "__main__":
    main()
