"""Inspect all preselected i-ASL sources without trajectory registration or fitting.

Raw IMU and TWR exports contain no motion-capture labels. Ground truth is read
separately only to report its supplied schema and clock, never to infer a map.
"""

from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore, get_types_from_msg

# All three official scenarios were selected before any method evaluation.
SOURCE = Path("E:/datasets/iasl-uwb-imu")
OUTPUT = Path(__file__).resolve().parents[4] / "agent/codex/strong_iasl"
SCENARIOS = (1, 2, 3)


def stamp_seconds(stamp):
    """Combine the ROS seconds and nanoseconds without truncation."""
    return stamp.sec + stamp.nanosec * 1e-9


def clock_summary(time):
    """Report measured event cadence, including duplicates and long gaps."""
    delta = np.diff(time)
    return dict(count=len(time), first_s=float(time[0]), last_s=float(time[-1]),
                duration_s=float(time[-1]-time[0]), median_dt_s=float(np.median(delta)),
                min_dt_s=float(np.min(delta)), max_dt_s=float(np.max(delta)),
                nonpositive_deltas=int(np.sum(delta <= 0)))


def inspect_scenario(index):
    """Decode real asynchronous IMU and TWR and retain separate label metadata."""
    bag_path = SOURCE / f"rosbag/0612_hw{index}.bag"
    store = get_typestore(Stores.ROS1_NOETIC)
    imu, radio, topics = [], [], []
    with Reader(bag_path) as reader:
        for connection in reader.connections:
            store.register(get_types_from_msg(connection.msgdef.data, connection.msgtype))
            topics.append(dict(topic=connection.topic, type=connection.msgtype,
                               count=connection.msgcount))
        for connection, bag_ns, raw in reader.messages():
            if connection.topic not in ("/imu/data", "/nlink_linktrack_tagframe0"):
                continue
            msg = store.deserialize_ros1(raw, connection.msgtype)
            if connection.topic == "/imu/data":
                imu.append([stamp_seconds(msg.header.stamp), bag_ns*1e-9,
                            *[getattr(msg.linear_acceleration, a) for a in "xyz"],
                            *[getattr(msg.angular_velocity, a) for a in "xyz"]])
            else:
                radio.append([bag_ns*1e-9, msg.local_time, msg.system_time,
                              *msg.dis_arr, *msg.pos_3d])
    imu, radio = np.asarray(imu, dtype=float), np.asarray(radio, dtype=float)
    imu_csv = pd.read_csv(SOURCE/f"scenario{index}/imu.csv", sep="\t")
    radio_csv = pd.read_csv(SOURCE/f"scenario{index}/uwb.csv", sep="\t")
    # Labels are deliberately excluded from the estimator-ready archive.
    np.savez_compressed(OUTPUT/f"scenario{index}_sensors.npz",
                        imu_header_time=imu[:, 0], imu_receive_time=imu[:, 1],
                        acc_mps2=imu[:, 2:5], gyro_radps=imu[:, 5:8],
                        radio_receive_time=radio[:, 0], radio_local_ms=radio[:, 1],
                        radio_system_ms=radio[:, 2], ranges_m=radio[:, 3:11])
    gt = pd.read_csv(SOURCE/f"scenario{index}/gt.csv", sep="\t")
    rotations = gt.iloc[:, 4:].to_numpy(float).reshape(-1, 3, 3)
    imu_equal = len(imu_csv) == len(imu)
    radio_equal = len(radio_csv) == len(radio)
    csv_difference = (np.max(np.abs(imu_csv.iloc[:, 1:].to_numpy(float)-imu[:, 2:8]), axis=0)
                      if imu_equal else np.full(6, np.nan))
    return dict(scenario=index, bag_sha256=hashlib.sha256(bag_path.read_bytes()).hexdigest(),
        topics=topics, imu_header=clock_summary(imu[:, 0]),
        imu_receive=clock_summary(imu[:, 1]), radio_receive=clock_summary(radio[:, 0]),
        radio_local_seconds=clock_summary(radio[:, 1]*1e-3),
        imu_receive_minus_header_median_s=float(np.median(imu[:, 1]-imu[:, 0])),
        imu_receive_minus_header_max_s=float(np.max(imu[:, 1]-imu[:, 0])),
        csv_imu_columns=list(imu_csv.columns), csv_imu_count=len(imu_csv),
        csv_imu_same_values=bool(imu_equal and np.allclose(imu_csv.iloc[:, 1:], imu[:, 2:8], atol=1e-12)),
        csv_imu_max_abs_difference_by_column=csv_difference.tolist(),
        csv_imu_times_are_header_integer_seconds=bool(imu_equal and np.allclose(imu_csv.iloc[:, 0], np.floor(imu[:, 0]), atol=0, rtol=0)),
        csv_imu_unique_time_count=int(imu_csv.iloc[:, 0].nunique()),
        csv_radio_columns=list(radio_csv.columns), csv_radio_count=len(radio_csv),
        csv_radio_same_ranges=bool(radio_equal and np.allclose(radio_csv.iloc[:, 5:13], radio[:, 3:11], atol=1e-8)),
        range_nonpositive_count=int(np.sum(radio[:, 3:11] <= 0)),
        acc_first_body_mps2=imu[0, 2:5].tolist(), gyro_first_body_radps=imu[0, 5:8].tolist(),
        acceleration_norm_median_mps2=float(np.median(np.linalg.norm(imu[:, 2:5], axis=1))),
        first_uwb_device_position_m=radio[0, 11:14].tolist(),
        gt_columns=list(gt.columns), gt_clock=clock_summary(gt.iloc[:, 0].to_numpy(float)),
        gt_first_position_m=gt.iloc[0, 1:4].to_numpy(float).tolist(),
        gt_first_rotation=rotations[0].tolist(),
        gt_rotation_max_orthogonality_error=float(np.max(np.linalg.norm(rotations@rotations.transpose(0, 2, 1)-np.eye(3), axis=(1, 2)))),
        gt_rotation_determinant_min=float(np.min(np.linalg.det(rotations))),
        gt_transform_or_sync_in_bag=False)


def main():
    """Write a compact source audit and preserve every preselected scenario."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    results = [inspect_scenario(i) for i in SCENARIOS]
    payload = dict(revision="41e585b47d84e810d7b448bdb21e3d98a8a9423e", scenarios=results,
        recovery_s=25, outage_lengths_s=[2, 5, 10, 20],
        benchmark_status="excluded_metadata",
        reference_replay_performed=False, shadow_replay_performed=False,
        cross_recording_calibration_performed=False,
        exclusion_slots=[dict(scenario=i, gap_s=gap, status="excluded_metadata")
                         for i in SCENARIOS for gap in (2, 5, 10, 20)],
        unresolved=["No published Qualisys-to-anchor rigid transform or surveyed IMU/tag lever arm.",
                    "GT is relative 0.1-second clock; no published epoch-to-GT trigger or clock relation.",
                    "No published IMU-to-Qualisys body rotation; sensor orientation quaternion is not used.",
                    "ROS receive times share the host clock; sensor acquisition delay is not calibrated."])
    (OUTPUT/"source_audit.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for item in results:
        print(item["scenario"], "imu", item["imu_header"], "radio", item["radio_receive"],
              "csv_match", item["csv_imu_same_values"], "gt", item["gt_clock"], flush=True)


if __name__ == "__main__":
    main()
