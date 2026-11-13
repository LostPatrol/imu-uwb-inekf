"""Build fixed ISAS held-out episodes and independent training-calibrated scores.

All estimator inputs preserve physical IMU vectors and corrected scalar ranges.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

from calibrate_training import IMU_TAG_M, load_sequence, read_csv
from export_isas import save_csv

# The benchmark owns the common prefix estimator and prior intervention grid.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prepare import prefix_estimate, write_priors

# Predeclared acquisition/recovery duration and initial stationary-prefix duration.
EPISODE_SECONDS = 15.0
PREFIX_SECONDS = 1.5
# Standard z-up specific force; physical gravitational acceleration is its negative.
STANDARD_SPECIFIC_FORCE = np.array([0.0, 0.0, 9.81])


def calibrated_reference(sequence_path, calibration, up_rotation):
    """Transform held-out poses with frozen Walk1 extrinsics and clock lag."""
    imu, radio, anchors, gt_t, gt_pos, slerp = load_sequence(sequence_path)
    rtls_vive = Rotation.from_matrix(calibration["rtls_from_vive_rotation_matrix"])
    tracker_imu = Rotation.from_matrix(calibration["tracker_from_imu_rotation_matrix"])
    tracker_to_imu = np.asarray(calibration["tracker_to_imu_in_tracker_m"])
    rtls_t = np.asarray(calibration["rtls_from_vive_translation_m"])
    tracker_orientation = slerp(gt_t)
    imu_position = up_rotation.apply(rtls_vive.apply(gt_pos + tracker_orientation.apply(tracker_to_imu)) + rtls_t)
    imu_orientation = up_rotation * rtls_vive * tracker_orientation * tracker_imu
    # Only scoring timestamps receive the fixed training clock correction.
    score_t = gt_t + calibration["imu_time_minus_vive_time_s"]
    yaw = np.arctan2(imu_orientation.as_matrix()[:, 1, 0], imu_orientation.as_matrix()[:, 0, 0])
    return imu, radio, anchors, score_t, imu_position, yaw


def main():
    """Write the four predeclared episodes with score data outside inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    streams, scoring = args.root / "streams", args.root / "scoring"
    calibration_file = scoring / "isas_walk1_calibration.json"
    calibration = json.loads(calibration_file.read_text(encoding="utf-8"))
    training_imu, _, _, gt_t, _, qs = load_sequence(streams / "isas_walk1")
    train_mask = training_imu["t"] < 1.0
    train_times = training_imu["t"][train_mask] - calibration["imu_time_minus_vive_time_s"]
    training_acc = np.column_stack([training_imu[key][train_mask] for key in ("ax", "ay", "az")])
    train_orientation = Rotation.from_matrix(calibration["rtls_from_vive_rotation_matrix"]) * qs(train_times) * Rotation.from_matrix(calibration["tracker_from_imu_rotation_matrix"])
    training_force = train_orientation.apply(training_acc).mean(axis=0)
    up_rotation, _ = Rotation.align_vectors(STANDARD_SPECIFIC_FORCE[None], training_force[None])
    calibration["zup_from_native_rtls_rotation_matrix"] = up_rotation.as_matrix().tolist()
    calibration["zup_training_specific_force_rtls_mps2"] = training_force.tolist()
    calibration["zup_policy"] = "Single Walk1 first-1s mean specific-force direction; applied only to world coordinates, never IMU vectors."
    calibration_file.write_text(json.dumps(calibration, indent=2) + "\n", encoding="utf-8")
    summaries = []
    for number in (2, 3):
        sequence_id = f"isas_walk{number}"
        sequence_path = streams / sequence_id
        imu, radio, anchors, score_t, score_pos, score_yaw = calibrated_reference(sequence_path, calibration, up_rotation)
        common_start = max(imu["t"][0], radio["t"][0])
        common_end = min(imu["t"][-1], radio["t"][-1])
        starts = {"start": common_start + PREFIX_SECONDS, "recovery": (common_start + common_end) / 2}
        for kind, start in starts.items():
            episode_id = f"{sequence_id}_{kind}"
            target = streams / episode_id
            target.mkdir(parents=True, exist_ok=True)
            end = start + EPISODE_SECONDS
            imu_episode = imu[(imu["t"] >= start - PREFIX_SECONDS) & (imu["t"] <= end)]
            radio_episode = radio[(radio["t"] >= start) & (radio["t"] <= end)]
            imu_times = imu_episode["t"] - start
            imu_values = np.column_stack([imu_episode[key] for key in ("ax", "ay", "az", "gx", "gy", "gz")])
            held_boundary = not np.any(np.abs(imu_times) < 1e-10)
            if held_boundary:
                # Explicit zero-order hold at restart copies the most recent sensor vector.
                latest = np.flatnonzero(imu_times < 0)[-1]
                imu_times = np.insert(imu_times, latest + 1, 0.0)
                imu_values = np.insert(imu_values, latest + 1, imu_values[latest], axis=0)
            save_csv(target / "imu.csv", ["t", "ax", "ay", "az", "gx", "gy", "gz"],
                     [[f"{time:.9f}", *values] for time, values in zip(imu_times, imu_values)])
            save_csv(target / "radio.csv", ["t", "anchor_id", "range_m", "sigma_m"],
                     [[f"{row['t'] - start:.9f}", int(row["anchor_id"]), row["range_m"], row["sigma_m"]] for row in radio_episode])
            save_csv(target / "anchors.csv", ["anchor_id", "x", "y", "z"],
                     [[key, *up_rotation.apply(value)] for key, value in sorted(anchors.items())])
            keep = (score_t >= start - 0.1) & (score_t <= end + 0.1)
            save_csv(scoring / f"{episode_id}.csv", ["t", "x", "y", "z", "yaw_rad"],
                     [[f"{time - start:.9f}", *pos, yaw] for time, pos, yaw in zip(score_t[keep], score_pos[keep], score_yaw[keep])])
            prefix = prefix_estimate(imu_times, imu_values)
            meta = json.loads((sequence_path / "meta.json").read_text(encoding="utf-8"))
            meta.update({"dataset_id": episode_id, "dataset": "isas", "episode": episode_id,
                         "kind": kind, "source_sequence": sequence_id,
                         "source_start_s": float(start), "source_end_s": float(end),
                         "selection": "start = sensor common start + 1.5s; recovery = sensor common midpoint; fixed before test evaluation",
                         "t_epoch_ns": meta["t_epoch_ns"] + int(round(start * 1e9)),
                         "duration_s": EPISODE_SECONDS, "lever_m": IMU_TAG_M.tolist(),
                         **prefix,
                         "bias_units": {"gyro_bias": "rad/s", "accel_bias": "m/s^2"},
                         "R0_policy": "Shared prepare.prefix_estimate from this episode adjacent [-1.5,0) prefix only; no earlier history or GT",
                         "bias_policy": "Shared prepare.prefix_estimate stationary gate/mean and acceleration gravity residual",
                         "imu_time_zero_held_boundary": bool(held_boundary),
                         "gravity_world_mps2": [0.0, 0.0, -9.81],
                         "anchor_frame": "Walk1-training-defined z-up RTLS world; fixed across all test episodes",
                         "anchor_native_to_world_rotation": up_rotation.as_matrix().tolist(),
                         "physical_vectors_modified": False,
                         "counts": {"imu": len(imu_times), "radio": len(radio_episode), "ground_truth": int(np.sum(keep))},
                         "ground_truth": {"file": f"../../scoring/{episode_id}.csv", "origin": "IMU",
                                          "full_pose": True, "yaw_reference": "independent Walk1 training calibration",
                                          "test_alignment": False, "author_published_extrinsics": False,
                                          "calibration_file": "../../scoring/isas_walk1_calibration.json",
                                          "strict_author_calibrated_heading": False,
                                          "limitation": "Reference yaw and position depend on independent noisy Walk1 calibration; no test trajectory fitted."}})
            (target / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
            if kind == "recovery":
                xy_at_zero = [np.interp(start, score_t, score_pos[:, column]) for column in (0, 1)]
                yaw_at_zero = float(np.interp(start, score_t, np.unwrap(score_yaw)))
                survey_height = float(np.mean([up_rotation.apply(value)[2] for value in anchors.values()]))
                write_priors(target / "priors.csv", xy_at_zero, yaw_at_zero, survey_height)
            summaries.append({"id": episode_id, "start_s": start, **meta["counts"]})
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
