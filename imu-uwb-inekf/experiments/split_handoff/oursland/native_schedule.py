"""Audit the literal author main event queue and replay both kernels on that queue."""

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time
import warnings

# Imported backup and current code are read-only.
sys.dont_write_bytecode = True
import run as ours
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from numba import njit
from threadpoolctl import threadpool_info, threadpool_limits

# Both BLAS libraries are limited even if an earlier Python import initialized them.
THREAD_LIMIT = threadpool_limits(limits=1)
# Preserve diagnostics so a slow run cannot be mistaken for an algorithmic failure.
THREAD_POOLS = threadpool_info()
# These are exactly the already compiled current kernel functions, without a new filter.
CURRENT_PREDICT = ours.util.kernel.predict
CURRENT_CORRECT = ours.util.kernel.correct

# The original main always sets TDOA2=True, even when the input filename says TDoA3.
RING = {(7, 0), (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)}
# Original main covariance order and standard deviations are preserved numerically.
ORIGINAL_P = np.diag(np.array([.01, .01, 1., .01, .01, .01, .1, .1, .1,
                               .0001, .0001, .0001, .000001, .000001, .000001]) ** 2)
# Locate original raw CSV data independently of any historic outputs.
DATA = Path("E:/datasets/dataset/flight-dataset")


@njit(cache=True)
def compiled_current(events, imu_indices, radio_indices, imu, radio, anchors, state, covariance, bias,
                     start, initial_previous, outage_start, outage_end, literal):
    """Eliminate Python per-packet overhead while preserving the literal event calls."""
    state, covariance, bias = state.copy(), covariance.copy(), bias.copy()
    rows = np.zeros((len(events), 19))
    output_index, accepted, rejected, skipped, predicts, corrections = 0, 0, 0, 0, 0, 0
    previous_imu_time, last_output, failed = initial_previous, -np.inf, False
    first = max(1, np.searchsorted(events, start, side="right"))
    for k in range(first, len(events)):
        previous, stamp = events[k - 1], events[k]
        if not literal and previous <= start:
            continue
        if imu_indices[k - 1] >= 0:
            dt = stamp - previous_imu_time
            previous_imu_time = stamp
            state, covariance = CURRENT_PREDICT(state, covariance, bias, imu[imu_indices[k - 1]], dt)
            predicts += 1
        if radio_indices[k - 1] >= 0:
            packet = radio[radio_indices[k - 1]]
            if outage_start <= previous < outage_end:
                skipped += 1
            else:
                observation = packet.copy()
                observation[0] = stamp
                state, covariance, bias, keep = CURRENT_CORRECT(state, covariance, bias, observation, anchors, stamp, .13)
                accepted += int(keep)
                rejected += int(not keep)
                corrections += 1
        if stamp - last_output >= .01:
            rows[output_index, 0] = stamp
            rows[output_index, 1:4] = state[:3, 4]
            rows[output_index, 4:7] = state[:3, 3]
            rows[output_index, 7:16] = state[:3, :3].flatten()
            output_index += 1
            last_output = stamp
        if not np.all(np.isfinite(state)) or not np.all(np.isfinite(covariance)) or np.max(np.abs(state[:3, 4])) > 1000:
            failed = True
            break
    return state, covariance, bias, rows[:output_index].copy(), accepted, rejected, skipped, predicts, corrections, failed


def load_raw(name, literal_ring):
    """Reproduce native raw acceleration time and gyro interpolation from author main."""
    geometry = name.split("-")[0]
    frame = pd.read_csv(DATA / "csv-data" / geometry / (name + ".csv"))
    def stream(columns):
        """Read each original complete sensor stream in CSV order."""
        return frame[columns].dropna().to_numpy(float)
    acc = stream(["t_acc", "acc_x", "acc_y", "acc_z"])
    gyro = stream(["t_gyro", "gyro_x", "gyro_y", "gyro_z"])
    radio = stream(["t_tdoa", "idA", "idB", "tdoa_meas"])
    truth = stream(["t_pose", "pose_x", "pose_y", "pose_z", "pose_qx", "pose_qy", "pose_qz", "pose_qw"])
    origin = float(min(acc[0, 0], radio[0, 0]))
    original_position, original_rotation = None, None
    if literal_ring:
        origin = float(min(origin, truth[0, 0]))
        # Only the separately labeled truth-assisted reference uses these fields.
        original_position = truth[truth[:, 0] > origin][0, 1:4]
        original_rotation = Rotation.from_quat(truth[0, 4:8]).as_matrix()
    imu = np.column_stack([acc[:, 1:4], *[np.interp(acc[:, 0], gyro[:, 0], gyro[:, k]) for k in (1, 2, 3)]])
    if literal_ring or "tdoa2" in name:
        radio = radio[np.array([tuple(row) in RING for row in radio[:, 1:3]])]
    radio = radio[np.argsort(radio[:, 0])]
    common_origin = max(acc[0, 0], gyro[0, 0], stream(["t_tdoa", "idA", "idB", "tdoa_meas"])[0, 0])
    end = common_origin + 45.
    acc_time = acc[acc[:, 0] < end, 0]
    imu = imu[:len(acc_time)]
    radio = radio[radio[:, 0] < end]
    anchors = np.load(DATA / "survey-results" / ("anchor_" + geometry + ".npz"))["an_pos"]
    return dict(name=name, times=acc_time, imu=imu, radio=radio, truth=truth, anchors=anchors,
                origin=origin, common_origin=float(common_origin), original_position=original_position,
                original_rotation=original_rotation)


def replay(data, state, covariance, bias, start, variant, outage, literal, end=None, return_final=False, use_compiled=True):
    """Apply author main's exact predict dt and first-packet-per-stamp call sequence."""
    estimator = ours.make_filter(state, covariance, bias) if variant == "author" else None
    state, covariance, bias = state.copy(), covariance.copy(), bias.copy()
    times, imu, radio = (data[k] for k in ("times", "imu", "radio"))
    events = np.unique(np.r_[times, radio[:, 0]])
    if end is not None:
        events = events[events <= end]
    # Original isin returns the first occurrence rather than all duplicate packets.
    imu_at = {}
    radio_at = {}
    for index, stamp in enumerate(times):
        imu_at.setdefault(float(stamp), index)
    for index, stamp in enumerate(radio[:, 0]):
        radio_at.setdefault(float(stamp), index)
    first = max(1, np.searchsorted(events, start, side="right"))
    previous_imu_time = data["origin"] if literal else start
    records, accepted, rejected, skipped, predicts, corrections = [], 0, 0, 0, 0, 0
    failure, last_output, started = None, -np.inf, time.perf_counter()
    if variant == "current_kernel" and use_compiled:
        imu_indices = np.array([imu_at.get(float(stamp), -1) for stamp in events], dtype=np.int64)
        radio_indices = np.array([radio_at.get(float(stamp), -1) for stamp in events], dtype=np.int64)
        left, right = (-1., -1.) if outage is None else outage
        state, covariance, bias, rows, accepted, rejected, skipped, predicts, corrections, failed = compiled_current(
            events, imu_indices, radio_indices, imu, radio, data["anchors"], state, covariance, bias,
            start, previous_imu_time, left, right, literal)
        info = dict(failed=bool(failed), failure="nonfinite_or_position_exceeds_1000m" if failed else None,
                    predict_calls=int(predicts), correction_calls=int(corrections), accepted_radio=int(accepted),
                    rejected_radio=int(rejected), outage_skipped=int(skipped),
                    duplicate_radio_not_consumed=len(radio) - len(radio_at), runtime_seconds=time.perf_counter() - started)
        if return_final:
            return state, covariance, bias, rows, accepted, rejected, bool(failed)
        return rows, info
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        try:
            for k in range(first, len(events)):
                previous, stamp = float(events[k - 1]), float(events[k])
                if not literal and previous <= start:
                    continue
                if previous in imu_at:
                    dt = stamp - previous_imu_time
                    previous_imu_time = stamp
                    sample = imu[imu_at[previous]]
                    if variant == "author":
                        estimator.predict(sample, dt)
                    else:
                        state, covariance = ours.util.kernel.predict(state, covariance, bias, sample, dt)
                    predicts += 1
                if previous in radio_at:
                    packet = radio[radio_at[previous]]
                    if outage is not None and outage[0] <= previous < outage[1]:
                        skipped += 1
                    elif variant == "author":
                        _, _, _, discard = estimator.correct(packet[1:], data["anchors"])
                        rejected += int(discard)
                        accepted += int(not discard)
                        corrections += 1
                    else:
                        # Match the original no-time-compensation observation at this call time.
                        observation = np.r_[stamp, packet[1:]]
                        state, covariance, bias, keep = ours.util.kernel.correct(state, covariance, bias, observation, data["anchors"], stamp, .13)
                        accepted += int(keep)
                        rejected += int(not keep)
                        corrections += 1
                if variant == "author":
                    state, covariance = estimator.X, estimator.P
                if stamp - last_output >= .01:
                    records.append(np.r_[stamp, state[:3, 4], state[:3, 3], state[:3, :3].ravel(), np.zeros(3)])
                    last_output = stamp
                if not np.all(np.isfinite(state)) or not np.all(np.isfinite(covariance)) or np.max(np.abs(state[:3, 4])) > 1000:
                    failure = "nonfinite_or_position_exceeds_1000m"
                    break
        except Exception as exception:
            failure = type(exception).__name__ + ": " + str(exception)
    rows = np.array(records).reshape(-1, 19)
    info = dict(failed=failure is not None, failure=failure,
        predict_calls=predicts, correction_calls=corrections, accepted_radio=accepted, rejected_radio=rejected,
        outage_skipped=skipped, duplicate_radio_not_consumed=len(radio) - len(radio_at), runtime_seconds=time.perf_counter() - started)
    if return_final:
        if variant == "author":
            bias = np.r_[estimator.bias[3:], estimator.bias[:3]]
        return state, covariance, bias, rows, accepted, rejected, failure is not None
    return rows, info


def main():
    """Run a clearly labeled truth-initialized reference or sensor-only paired queue."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--names", nargs="+", default=["const1-trial4-tdoa2"])
    parser.add_argument("--yaws", nargs="+", type=float, default=[0, 180])
    parser.add_argument("--initialization", choices=["original_truth", "sensor_only", "sensor_only_original_prior"], default="original_truth")
    parser.add_argument("--variants", nargs="+", choices=["author", "current_kernel"], default=["author", "current_kernel"])
    parser.add_argument("--output", default="native_reference")
    args = parser.parse_args()
    output = ours.HERE / args.output
    output.mkdir(parents=True, exist_ok=True)
    records, truths = [], {}
    cached = {}
    if (output / "records.json").exists():
        cached = {r["key"]: r for r in json.loads((output / "records.json").read_text(encoding="utf8"))}
    before_hash = ours.digest(ours.AUTHOR)
    for name in args.names:
        literal = args.initialization == "original_truth"
        data = load_raw(name, literal)
        truths[name] = data.pop("truth")
        common = ours.util.load(name, duration=45.)
        common.pop("truth")
        for yaw in args.yaws:
            if literal:
                start = data["origin"]
                state, bias = np.eye(5), np.zeros(6)
                state[:3, :3] = ours.author.SO3_exp(ours.author.SO3_log(data["original_rotation"]) + np.r_[0., 0., np.radians(yaw)])
                state[:3, 4] = data["original_position"]
                covariance = ORIGINAL_P.copy()
            else:
                start, state, bias = ours.util.initial_state(common)
                state[:3, :3] = Rotation.from_euler("z", yaw, degrees=True).as_matrix() @ state[:3, :3]
                change = ours.util.kernel.mapping(state)
                covariance = change @ ours.util.PHYSICAL @ change.T
                if args.initialization == "sensor_only_original_prior":
                    covariance = ORIGINAL_P.copy()
            schedules = ["ordinary"] if literal else ["ordinary", "outage"]
            for schedule in schedules:
                outage = None if schedule == "ordinary" else (data["common_origin"] + 10, data["common_origin"] + 20)
                for variant in args.variants:
                    key = f"{name}__yaw{yaw:g}__{schedule}__{args.initialization}__{variant}"
                    if key in cached and (output / (key + ".npz")).exists():
                        records.append(cached[key])
                        continue
                    rows, info = replay(data, state, covariance, bias, start, variant, outage, literal)
                    np.savez_compressed(output / (key + ".npz"), rows=rows, initial_state=state, initial_covariance=covariance, initial_bias=bias)
                    record = dict(key=key, name=name, yaw_offset_deg=yaw, schedule=schedule, variant=variant,
                                  initialization=args.initialization, origin=data["common_origin"], **info)
                    records.append(record)
                    (output / "records.json").write_text(json.dumps(records, indent=2), encoding="utf8")
                    print(json.dumps(record), flush=True)
    metrics = []
    for record in records:
        rows = np.load(output / (record["key"] + ".npz"))["rows"]
        scenarios = [("outage", 26.)] if record["schedule"] == "outage" else [("cold_weak", 2.), ("dynamic", 16.)]
        for scenario, cut in scenarios:
            metric = {**record, "scenario": scenario}
            metric.update({"post_" + k: v for k, v in ours.util.score_rows(truths[record["name"]], rows, record["origin"] + cut + .05, record["origin"] + 44.95).items()})
            metric.update({"whole_" + k: v for k, v in ours.util.score_rows(truths[record["name"]], rows, record["origin"] + .65, record["origin"] + 44.95).items()})
            metrics.append(metric)
    pd.DataFrame(metrics).to_csv(output / "metrics.csv", index=False)
    (output / "provenance.json").write_text(json.dumps(dict(arguments=vars(args), source_sha256_before=before_hash,
        source_sha256_after=ours.digest(ours.AUTHOR), author_source_modified=before_hash != ours.digest(ours.AUTHOR),
        truth_assisted_reference=args.initialization == "original_truth", native_main_queue=True,
        modifications="Output extraction and recording hashes; optimized identical timestamp lookup; current comparator receives identical call times/payload; original source untouched."), indent=2), encoding="utf8")


if __name__ == "__main__":
    main()
