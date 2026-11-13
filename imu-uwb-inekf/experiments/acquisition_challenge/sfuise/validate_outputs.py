"""Audit native result causality, pose consistency and initial-iterate centers."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from run_batch import iterate_quaternion


def validate(streams, results, output):
    """Check recorded outputs without reading any scoring truth."""
    records = []
    centers = []
    for directory in sorted(results.iterdir()):
        stream = streams / directory.name
        if not stream.is_dir() or not (directory / "audit.json").exists():
            continue
        meta = json.loads((stream / "meta.json").read_text(encoding="utf-8-sig"))
        lever = np.asarray(meta.get("lever_m", meta.get("body_frame_lever_m", [0, 0, 0])))
        paths = list(directory.glob("native.csv")) + list(directory.glob("iterate_*.csv"))
        for path in paths:
            if path.stat().st_size == 0:
                continue
            frame = pd.read_csv(path)
            if frame.empty:
                continue
            quaternion = frame[["qx", "qy", "qz", "qw"]].to_numpy()
            rotated_lever = Rotation.from_quat(quaternion).apply(np.tile(lever, (len(frame), 1)))
            measured_lever = frame[["tag_x", "tag_y", "tag_z"]].to_numpy() - frame[["x", "y", "z"]].to_numpy()
            record = {
                "episode": directory.name, "file": path.name, "rows": len(frame),
                "first_arrival_s": float(frame.arrival_s.iloc[0]), "last_arrival_s": float(frame.arrival_s.iloc[-1]),
                "first_state_s": float(frame.state_s.iloc[0]), "last_state_s": float(frame.state_s.iloc[-1]),
                "last_wall_s": float(frame.wall_s.iloc[-1]),
                "causal": bool((frame.arrival_s + 1e-12 >= frame.state_s).all()),
                "state_time_increasing": bool((np.diff(frame.state_s) > 0).all()),
                "wall_time_increasing": bool((np.diff(frame.wall_s) >= 0).all()),
                "quaternion_max_norm_error": float(np.abs(np.linalg.norm(quaternion, axis=1) - 1).max()),
                "tag_body_lever_max_error_m": float(np.linalg.norm(measured_lever - rotated_lever, axis=1).max()),
                "max_position_norm_m": float(np.linalg.norm(frame[["x", "y", "z"]], axis=1).max()),
            }
            records.append(record)
        prior_file = stream / "priors.csv"
        if prior_file.exists():
            for _, row in pd.read_csv(prior_file).iterrows():
                q = iterate_quaternion(meta, row)
                achieved_yaw = Rotation.from_quat(q[[1, 2, 3, 0]]).as_euler("ZYX")[0]
                yaw_error = np.arctan2(np.sin(achieved_yaw - row.yaw_rad), np.cos(achieved_yaw - row.yaw_rad))
                centers.append({"episode": directory.name, "case_id": row.case_id,
                                "initial_body_x": row.px, "initial_body_y": row.py, "initial_body_z": row.pz,
                                "initial_qw": q[0], "initial_qx": q[1], "initial_qy": q[2], "initial_qz": q[3],
                                "requested_yaw_rad": row.yaw_rad, "achieved_yaw_rad": achieved_yaw,
                                "yaw_error_rad": yaw_error, "covariance_consumed": False})
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(output / "output_checks.csv", index=False)
    pd.DataFrame(centers).to_csv(output / "initial_centers.csv", index=False)
    summary = {
        "files_checked": len(records), "causal_all": all(row["causal"] for row in records),
        "state_time_increasing_all": all(row["state_time_increasing"] for row in records),
        "quaternion_max_norm_error": max((row["quaternion_max_norm_error"] for row in records), default=0),
        "tag_body_lever_max_error_m": max((row["tag_body_lever_max_error_m"] for row in records), default=0),
        "initial_center_yaw_max_error_rad": max((abs(row["yaw_error_rad"]) for row in centers), default=0),
        "initial_center_note": "Local body pose is identity/zero, so configured nav transform is exactly the physical body center; no output initialization rows are counted as estimates",
        "ground_truth_read": False,
    }
    (output / "validation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main():
    """Accept common stream and completed-output directories."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--streams", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    validate(args.streams, args.results, args.output)


if __name__ == "__main__":
    main()
