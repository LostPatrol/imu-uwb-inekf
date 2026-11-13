"""Export official ISAS-Walk bags without rotating physical sensor vectors.

VIVE pose remains in score/ and is never used to construct estimator streams.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import urllib.request

import numpy as np
from rosbags.highlevel import AnyReader

# Pin official source revision so later exports use identical recordings/configs.
SOURCE_COMMIT = "f29744d81e41266a10df72d22ffd39b841004b22"
# Official bag files are small enough to retain in their original form.
SOURCE = f"https://raw.githubusercontent.com/ASIG-X/SFUISE/{SOURCE_COMMIT}/"
# Official offset order is ascending anchor ID; residual = predicted - raw - offset.
RANGE_OFFSETS = dict(zip([7475, 9524, 10548, 15155, 20276], [-0.0700, 0.1539, -0.0751, 0.1409, -0.0247]))
# A conservative common measurement model avoids treating running variance as noise.
DEFAULT_SIGMA_M = 0.15


def save_csv(path, header, rows):
    """Write a UTF-8 standard stream with explicit column names."""
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def stamp_ns(message):
    """Read the published sensor timestamp, not rosbag reception time."""
    return message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nanosec


def vector(value):
    """Extract a ROS geometry vector without changing its axes."""
    return [value.x, value.y, value.z]


def export_one(bag_path, output_root):
    """Decode one complete recording and separate scoring observations."""
    sequence = "isas_" + bag_path.stem.lower().replace("isas-", "").replace("-", "_")
    target = output_root / sequence
    score = target / "score"
    score.mkdir(parents=True, exist_ok=True)
    imu, radio, quality, gt, clocks, anchors = [], [], [], [], [], {}
    anchor_updates = 0
    anchor_change_max = 0.0
    with AnyReader([bag_path]) as reader:
        connections = [{"topic": c.topic, "type": c.msgtype, "count": c.msgcount} for c in reader.connections]
        for connection, bag_ns, raw in reader.messages():
            message = reader.deserialize(raw, connection.msgtype)
            topic = connection.topic
            if hasattr(message, "header"):
                sensor_ns = stamp_ns(message)
                clocks.append([topic, sensor_ns, bag_ns])
            if topic == "/waveshare_sense_hat_b":
                imu.append([sensor_ns, *vector(message.linear_acceleration), *vector(message.angular_velocity)])
            elif topic == "/rtls_flares":
                for measurement in message.ranges:
                    quality.append([sensor_ns, measurement.id, measurement.range, measurement.ra,
                                    measurement.toc, measurement.var, measurement.fpp, measurement.rxp])
                    if measurement.ra and np.isfinite(measurement.range) and measurement.range > 0:
                        corrected = measurement.range + RANGE_OFFSETS[measurement.id]
                        radio.append([sensor_ns, measurement.id, corrected, DEFAULT_SIGMA_M])
            elif topic == "/anchor_list":
                anchor_updates += 1
                for anchor in message.anchor:
                    position = np.asarray(vector(anchor.position))
                    if anchor.id in anchors:
                        anchor_change_max = max(anchor_change_max, float(np.max(np.abs(position - anchors[anchor.id]))))
                    else:
                        anchors[anchor.id] = position
            elif topic == "/vive/transform/tracker_1_ref":
                q = message.transform.rotation
                gt.append([sensor_ns, *vector(message.transform.translation), q.x, q.y, q.z, q.w])
    # Subtract one integer epoch only; all original sample times remain available.
    epoch_ns = min(imu[0][0], radio[0][0])
    for rows in (imu, radio, quality, gt):
        rows.sort(key=lambda row: row[0])
        for row in rows:
            row[0] = f"{(row[0] - epoch_ns) / 1e9:.9f}"
    save_csv(target / "imu.csv", ["t", "ax", "ay", "az", "gx", "gy", "gz"], imu)
    save_csv(target / "radio.csv", ["t", "anchor_id", "range_m", "sigma_m"], radio)
    save_csv(target / "anchors.csv", ["anchor_id", "x", "y", "z"], [[key, *value] for key, value in sorted(anchors.items())])
    save_csv(score / "ground_truth_native.csv", ["t", "x", "y", "z", "qx", "qy", "qz", "qw"], gt)
    save_csv(score / "radio_quality.csv", ["t", "anchor_id", "range_raw_m", "ra", "toc", "running_var", "fpp", "rxp"], quality)
    save_csv(score / "original_clock.csv", ["topic", "header_stamp_ns", "bag_stamp_ns"], clocks)
    initial = np.asarray([[float(value) for value in row] for row in imu if float(row[0]) <= 1.0])
    metadata = {
        "dataset_id": sequence, "source_repository": "https://github.com/ASIG-X/SFUISE", "source_commit": SOURCE_COMMIT,
        "source_url": SOURCE + "dataset/" + bag_path.name,
        "source_bag": str(bag_path.resolve()), "source_sha256": hashlib.sha256(bag_path.read_bytes()).hexdigest(),
        "source_connections": connections, "radio_type": "absolute_range_toa",
        "t_units": "s", "t_epoch_ns": epoch_ns, "timestamp_source": "message.header.stamp",
        "clock_caveat": "Shared ROS epoch; independent hardware clock calibration is not published. No timestamp warp applied.",
        "imu_accel_units": "m/s^2", "imu_gyro_units": "rad/s", "imu_frame": "Waveshare_Sense_HAT_B/IMU",
        "physical_vectors_modified": False, "anchor_frame": "native RTLS coordinate frame",
        "anchor_frame_origin_id": 7475, "anchor_frame_positive_x_id": 20276,
        "anchor_coordinates_units": "m", "anchor_updates": anchor_updates,
        "anchor_change_max_m": anchor_change_max,
        "range_policy": "rg.range plus official per-anchor constant; ra=0 stale readings omitted; no running mean, interpolation, or GT corrections",
        "sigma_m": DEFAULT_SIGMA_M,
        "sigma_policy": "constant common nominal model; device running variance retained separately, not assumed sensor variance",
        "official_range_additive_offsets_m": RANGE_OFFSETS,
        "official_offset_application": "corrected_range = raw_range + offset; applied in radio.csv; identical official configs for Walk1/2/3",
        "body_frame_lever_m": [0.1, -0.025, 0.0],
        "lever_source": SOURCE + "sfuise/config/config_test_isas-walk1.yaml",
        "lever_caveat": "Published IMU-to-tag lever. Tracker-to-IMU extrinsic rotation/translation not supplied.",
        "gravity_prefix": {"duration_s": 1.0, "samples": len(initial),
                           "accel_mean_mps2": initial[:, 1:4].mean(axis=0).tolist(),
                           "accel_std_mps2": initial[:, 1:4].std(axis=0).tolist(),
                           "gyro_mean_radps": initial[:, 4:7].mean(axis=0).tolist(),
                           "gyro_std_radps": initial[:, 4:7].std(axis=0).tolist(),
                           "availability": "observed low-motion prefix; gravity direction available, absolute yaw unavailable"},
        "ground_truth": {"file": "score/ground_truth_native.csv", "instrument": "HTC VIVE tracker",
                         "frame": "vive/reference", "body": "vive/tracker_1", "full_pose": True,
                         "same_as_anchor_frame": False, "published_fixed_transform_to_anchor": False,
                         "published_tracker_to_imu_extrinsic": False,
                         "strict_absolute_heading_ready": False,
                         "warning": "Native VIVE poses are not in RTLS frame. Test-trajectory alignment is forbidden for initialization."},
        "counts": {"imu": len(imu), "radio": len(radio), "ground_truth": len(gt)},
        "duration_s": float(imu[-1][0]) - float(imu[0][0]),
    }
    (target / "meta.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return {"id": sequence, "directory": str(target.resolve()), **metadata["counts"], "duration_s": metadata["duration_s"]}


def main():
    """Download missing official bags and export all three recordings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for number in (1, 2, 3):
        bag = args.raw_dir / f"ISAS-Walk{number}.bag"
        if not bag.exists():
            urllib.request.urlretrieve(SOURCE + "dataset/" + bag.name, bag)
        summaries.append(export_one(bag, args.output_dir))
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
