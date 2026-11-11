"""Replay the untouched Oursland backup on the frozen sensor-only UTIL holdout."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import warnings

# Imported implementations, including the author backup, remain read-only.
sys.dont_write_bytecode = True
# Keep scipy's small matrix exponentials practical and reproducible.
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

# Files are imported read-only; only this experiment's output is written.
HERE = Path(__file__).resolve().parent
UTIL = HERE.parent / "util"
REPO = HERE.parents[2]
AUTHOR = REPO.parent / "imu-uwb-inekf - backup" / "scripts" / "estimation" / "inekf.py"
sys.path.insert(0, str(UTIL))
spec = importlib.util.spec_from_file_location("split_util_run", UTIL / "run.py")
util = importlib.util.module_from_spec(spec)
spec.loader.exec_module(util)
spec = importlib.util.spec_from_file_location("untouched_oursland_inekf", AUTHOR)
author = importlib.util.module_from_spec(spec)
spec.loader.exec_module(author)
# Original main.py selects this variance, overriding the class default .05.
AUTHOR_VARIANCE = .13


def digest(path):
    """Return a file's hash to certify the imported source did not change."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_filter(state, covariance, bias):
    """Translate only the external canonical bias ordering for native prediction."""
    native_bias = np.r_[bias[3:], bias[:3]]
    estimator = author.InEKF(state.copy(), covariance.copy(), native_bias)
    estimator.w_tdoa = np.array([np.sqrt(AUTHOR_VARIANCE)])
    return estimator


def replay(data, state, covariance, bias, start, end, outage=None):
    """Call native author predict/correct without altering its internal behavior."""
    estimator = make_filter(state, covariance, bias)
    times, imu, radio = (data[k] for k in ("times", "imu", "radio"))
    begin, stop = np.searchsorted(times, [start, end], side="right")
    radio_index = np.searchsorted(radio[:, 0], start, side="right")
    output, accepted, rejected, skipped = [], 0, 0, 0
    previous, failure, last_stamp = start, None, start
    started = time.perf_counter()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for k in range(begin, stop):
                stamp = times[k]
                estimator.predict(.5 * (imu[k - 1] + imu[k]), stamp - previous)
                while radio_index < len(radio) and radio[radio_index, 0] <= stamp:
                    packet = radio[radio_index]
                    if outage is not None and outage[0] <= packet[0] < outage[1]:
                        skipped += 1
                    else:
                        _, _, _, discard = estimator.correct(packet[1:], data["anchors"])
                        rejected += int(discard)
                        accepted += int(not discard)
                    radio_index += 1
                if k % 5 == 0:
                    output.append(np.r_[stamp, estimator.X[:3, 4], estimator.X[:3, 3],
                                        estimator.X[:3, :3].ravel(), estimator.bias[:3]])
                previous, last_stamp = stamp, stamp
                if not np.all(np.isfinite(estimator.X)) or not np.all(np.isfinite(estimator.P)):
                    failure = "nonfinite_state_or_covariance"
                    break
                if np.max(np.abs(estimator.X[:3, 4])) > 1000:
                    failure = "position_exceeds_1000m"
                    break
            messages = sorted(set(str(item.message) for item in caught))
    except Exception as exception:
        failure = type(exception).__name__ + ": " + str(exception)
        messages = []
    return np.array(output).reshape(-1, 19), dict(
        accepted_radio=accepted, rejected_radio=rejected, outage_skipped_radio=skipped,
        failed=failure is not None, failure=failure, warnings=messages,
        runtime_seconds=time.perf_counter() - started, last_replay_time=float(last_stamp))


def generate(args):
    """Complete the fixed sensor-only set before exposing truth to evaluation."""
    output = HERE / args.output
    output.mkdir(parents=True, exist_ok=True)
    source_before = digest(AUTHOR)
    records = []
    names = args.names or [util.record_name(g, t, p) for t in [4, 5] for g in [1, 2, 3, 4] for p in [2, 3]]
    for name in names:
        data = util.load(name, duration=45.)
        data.pop("truth")
        origin = float(data["times"][0])
        start, initial, bias = util.initial_state(data)
        for yaw in args.yaws:
            state = initial.copy()
            state[:3, :3] = Rotation.from_euler("z", yaw, degrees=True).as_matrix() @ initial[:3, :3]
            change = util.kernel.mapping(state)
            covariance = change @ util.PHYSICAL @ change.T
            for schedule in ["ordinary", "outage"]:
                key = f"{name}__yaw{yaw:g}__{schedule}__author_continuous"
                artifact = output / (key + ".npz")
                if artifact.exists():
                    cached = np.load(artifact, allow_pickle=False)
                    info = json.loads(str(cached["info"]))
                else:
                    outage = None if schedule == "ordinary" else (origin + 10, origin + 20)
                    rows, info = replay(data, state, covariance, bias, start, data["times"][-1], outage)
                    np.savez_compressed(artifact, rows=rows, initial_state=state, initial_covariance=covariance,
                                        initial_bias=bias, info=json.dumps(info))
                records.append(dict(key=key, name=name, yaw_offset_deg=yaw, schedule=schedule,
                                    origin=origin, initial_time=float(start), **info))
                (output / "records.json").write_text(json.dumps(records, indent=2), encoding="utf8")
                print(json.dumps(dict(key=key, **info)), flush=True)
    provenance = dict(author_source=str(AUTHOR), source_sha256_before=source_before,
                      source_sha256_after=digest(AUTHOR), author_source_modified=source_before != digest(AUTHOR),
                      source_protocol_sha256=digest(UTIL / "protocol.json"),
                      protocol_sha256=digest(HERE / "protocol.json"), arguments=vars(args))
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf8")
    if provenance["author_source_modified"]:
        raise RuntimeError("Author source hash changed during replay")
    return records


def evaluate(output, records):
    """Score the continuous original on the exact paired 20Hz cut-point grids."""
    metrics, truths = [], {}
    for record in records:
        if record["name"] not in truths:
            truths[record["name"]] = util.load(record["name"], duration=45.)["truth"]
        truth = truths[record["name"]]
        rows = np.load(output / (record["key"] + ".npz"))["rows"]
        scenarios = [("outage", 26.)] if record["schedule"] == "outage" else [("cold_weak", 2.), ("dynamic", 16.)]
        for scenario, cut in scenarios:
            stamp = record["origin"] + cut
            # Match the util kernel's first k%5 output strictly after the handoff.
            after = rows[rows[:, 0] > stamp + 1e-9]
            metric = {**record, "scenario": scenario, "policy": "author_original_continuous"}
            metric.update({"post_" + k: v for k, v in util.score_rows(truth, after).items()})
            metric.update({"whole_" + k: v for k, v in util.score_rows(truth, rows, record["origin"] + .65, record["origin"] + 45.).items()})
            metric["post_expected_samples"] = int(round((45. - cut) / .05)) - 1
            metric["post_coverage"] = metric["post_samples"] / metric["post_expected_samples"]
            metrics.append(metric)
    frame = pd.DataFrame(metrics)
    frame.to_csv(output / "metrics.csv", index=False)
    frame.groupby(["scenario", "policy"])[["post_xy_rmse", "post_z_rmse", "post_yaw_rmse_deg", "whole_xy_rmse", "post_coverage"]].agg(["median", "max"]).to_csv(output / "summary.csv")


def main():
    """Generate or resume the frozen replay and then score its full requested set."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--names", nargs="+")
    parser.add_argument("--yaws", nargs="+", type=float, default=[0, 90, -90, 180])
    parser.add_argument("--output", default="holdout")
    parser.add_argument("--score-only", action="store_true")
    args = parser.parse_args()
    output = HERE / args.output
    records = json.loads((output / "records.json").read_text(encoding="utf8")) if args.score_only else generate(args)
    print("All requested sensor-only author trajectories generated; starting truth scoring.", flush=True)
    evaluate(output, records)


if __name__ == "__main__":
    main()
