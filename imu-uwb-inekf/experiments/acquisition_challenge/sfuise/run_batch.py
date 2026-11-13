"""Run pinned SFUISE on common streams, distinguishing native restart and iterates."""

import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

# A process per run resets function-local static variables in the author code.
DEFAULT_TIMEOUT_S = 90


def run_case(executable, dependency_bin, stream, target, case, quaternion, lever, duration, imu_path):
    """Execute one causal, fresh-process restart with a bounded wall-clock time."""
    native = case is None
    case_id = "native_restart" if native else str(case["case_id"])
    command = [str(executable)]
    arguments = {
        "imu": imu_path, "radio": stream / "radio.csv", "anchors": stream / "anchors.csv",
        "output": target, "episode": stream.name, "case": case_id,
        "duration": duration, "mode": "native" if native else "iterate",
        "lx": lever[0], "ly": lever[1], "lz": lever[2],
    }
    if not native:
        arguments.update(dict(zip(("qw", "qx", "qy", "qz"), quaternion)))
        arguments.update({name: case[name] for name in ("px", "py", "pz")})
    for key, value in arguments.items():
        command.extend(("--" + key, str(value)))
    environment = os.environ.copy()
    environment["PATH"] = str(dependency_bin) + os.pathsep + environment.get("PATH", "")
    for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "BLIS_NUM_THREADS"):
        environment[name] = "1"
    before = time.perf_counter()
    try:
        process = subprocess.run(command, capture_output=True, text=True, env=environment,
                                 timeout=DEFAULT_TIMEOUT_S)
        return {"case_id": case_id, "returncode": process.returncode,
                "process_wall_s": time.perf_counter() - before,
                "stdout": process.stdout.strip(), "stderr": process.stderr.strip()}
    except subprocess.TimeoutExpired:
        return {"case_id": case_id, "returncode": "timeout", "process_wall_s": DEFAULT_TIMEOUT_S}


def prepare_imu(stream, output, meta):
    """Subtract the common sensor-only fixed bias, retaining zero incremental bias."""
    imu = pd.read_csv(stream / "imu.csv")
    gyro_bias = np.asarray(meta.get("gyro_bias", [0.0, 0.0, 0.0]), dtype=float)
    accel_bias = np.asarray(meta.get("accel_bias", [0.0, 0.0, 0.0]), dtype=float)
    imu.loc[:, ["gx", "gy", "gz"]] -= gyro_bias
    imu.loc[:, ["ax", "ay", "az"]] -= accel_bias
    path = output / "imu_common_calibrated.csv"
    imu.to_csv(path, index=False)
    return path, {"gyro_bias_subtracted": gyro_bias.tolist(), "accel_bias_subtracted": accel_bias.tolist()}


def iterate_quaternion(meta, row):
    """Set absolute body attitude center in anchor frame, preserving the common tilt."""
    r0 = Rotation.from_matrix(np.asarray(meta.get("R0", np.eye(3)), dtype=float))
    yaw0 = r0.as_euler("ZYX")[0]
    # R0's arbitrary yaw is replaced by the supplied center yaw; body tilt is unchanged.
    r = Rotation.from_euler("Z", float(row["yaw_rad"]) - yaw0) * r0
    q = r.as_quat()
    return q[[3, 0, 1, 2]]


def run_episode(stream, output, executable, dependency_bin, duration, workers, native_only, stress_only=False):
    """Run native once and each distinct supplied initial center once."""
    output.mkdir(parents=True, exist_ok=True)
    meta = json.loads((stream / "meta.json").read_text(encoding="utf-8-sig"))
    lever = meta.get("lever_m", meta.get("body_frame_lever_m", [0.0, 0.0, 0.0]))
    imu_path, calibration = prepare_imu(stream, output, meta)
    records = []
    if not stress_only:
        native_record = run_case(executable, dependency_bin, stream, output / "native.csv", None,
                                 None, lever, duration, imu_path)
        records.append(native_record)
    aliases = []
    rows = []
    prior_path = stream / "priors.csv"
    if not native_only and prior_path.exists():
        priors = pd.read_csv(prior_path)
        grouped = priors.groupby(["px", "py", "pz", "yaw_rad"], sort=False, dropna=False)
        tasks = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(workers, 5)) as pool:
            for index, (_, group) in enumerate(grouped):
                row = group.iloc[0]
                case_file = output / f"iterate_{index:03d}.csv"
                quaternion = iterate_quaternion(meta, row)
                tasks.append((pool.submit(run_case, executable, dependency_bin, stream, case_file,
                                          row, quaternion, lever, duration, imu_path), case_file, group))
            for future, case_file, group in tasks:
                records.append(future.result())
                if not case_file.exists() or case_file.stat().st_size == 0:
                    continue
                result = pd.read_csv(case_file)
                for _, row in group.iterrows():
                    duplicate = result.copy()
                    duplicate["case_id"] = row["case_id"]
                    rows.append(duplicate)
                    aliases.append({"case_id": row["case_id"], "native_case_id": str(group.iloc[0]["case_id"]),
                                    "confidence": row.get("confidence", ""), "covariance_consumed": False})
                print(json.dumps({"episode": stream.name, "center_completed": len(records),
                                  "centers_total": len(grouped), "case": str(group.iloc[0]["case_id"]),
                                  "returncode": records[-1]["returncode"]}), flush=True)
        if rows:
            pd.concat(rows, ignore_index=True).to_csv(output / "stress.csv", index=False)
    audit = {
        "method": "KIT-ISAS/SFUISE 75bf5a32f1a8e5c3046a5bd1a1ddf659fa996f7d ToA author numerical core",
        "native": "Author zero local pose and identity nav-to-anchor transform; ignores external prior",
        "stress": "Adapter: nonzero nav-to-anchor initial iterate only; no MAP residual or covariance support",
        "same_center_confidence_runs_reused": True, "case_aliases": aliases,
        "input_stream": str(stream.resolve()), "fixed_bias_preprocessing": calibration,
        "range_offset_in_core_m": 0, "range_preprocessing": "radio.csv used as supplied; no extra correction",
        "gravity": "Author first available accelerometer sample average, norm 9.81; freely optimized",
        "positions": "IMU origin in surveyed anchor frame, full current estimated nav-to-anchor transform",
        "timing": "1 ms causal sensor-time polling, native optimized publication at latest supported spline time",
        "arrival_s": "logical stream arrival/poll timestamp; computational latency separately in wall_s",
        "state_s": "latest interpolatable spline state timestamp, never greater than arrival_s",
        "ground_truth_read": False, "duration_s": duration, "records": records,
        "preloaded_anchor_survey": True, "original_isas_anchor_warmup_replayed": False,
        "pose_covariance_consumed": False, "first_unoptimized_knot_published": False,
        "publication_clock_caveat": "arrival_s is the causal sensor-event clock; wall_s is cumulative accelerated-replay computation, not a full ROS wall-clock publication timestamp",
        "startup_portability_correction": "Repeat sole endpoint instead of reading absent second knot during first integration",
        "startup_validation": "Validated against a valid two-duplicate-knot spline: total pose/velocity/acceleration/bias/angular-rate difference exactly 0",
    }
    (output / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return {"episode": stream.name, "runs": len(records), "failures": sum(r["returncode"] != 0 for r in records)}


def main():
    """Read batch paths and execute independent cases with bounded concurrency."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--streams", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", nargs="*")
    parser.add_argument("--duration", type=float, default=15)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--native-only", action="store_true")
    parser.add_argument("--stress-only", action="store_true")
    parser.add_argument("--build-root", type=Path, default=Path("agent/codex/sfuise_challenge"))
    args = parser.parse_args()
    build = args.build_root.resolve()
    episodes = [args.streams / name for name in args.episodes] if args.episodes else sorted(
        path for path in args.streams.iterdir() if (path / "meta.json").exists())
    summary = []
    for stream in episodes:
        result = run_episode(stream, args.output / stream.name, build / "sfuise_native.exe",
                             build / "deps/Library/bin", args.duration, args.workers, args.native_only, args.stress_only)
        summary.append(result)
        print(json.dumps(result), flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
