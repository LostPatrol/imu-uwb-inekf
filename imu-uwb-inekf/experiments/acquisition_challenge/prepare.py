"""Build identical Delft input streams and physically separate scoring references.

Wrong-prior cases are constructed only by the benchmark, after stream extraction.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.spatial.transform import Rotation

# Paths and constants reproduce the registered shared protocol.
HERE = Path(__file__).resolve().parent
PROTOCOL = json.loads((HERE / "protocol.json").read_text(encoding="utf-8"))
OLD_CAL = HERE.parent / "radio_inertial_models/radio_branch/results"
G = 9.81


def prefix_estimate(t, imu):
    """Estimate common tilt/bias from the permitted prefix in the original body axes."""
    prefix = imu[(t >= -PROTOCOL["prefix_s"] - 1e-9) & (t < 0)]
    norm = np.linalg.norm(prefix[:, :3], axis=1)
    still = ((norm > .85 * G) & (norm < 1.15 * G)
             & (np.linalg.norm(prefix[:, 3:], axis=1) < np.deg2rad(8)))
    accepted = prefix[still] if still.sum() >= 20 else prefix
    a = accepted[:, :3].mean(axis=0)
    roll = np.arctan2(a[1], a[2])
    pitch = np.arctan2(-a[0], np.hypot(a[1], a[2]))
    rotation = Rotation.from_euler("xyz", [roll, pitch, 0]).as_matrix()
    return dict(R0=rotation.tolist(), gyro_bias=accepted[:, 3:].mean(axis=0).tolist(),
                accel_bias=(a - rotation.T @ [0, 0, G]).tolist(),
                prefix_samples=len(prefix), prefix_samples_used=len(accepted))


def write_priors(path, xyz, yaw, height):
    """Write only the intervened center and spreads; height comes from survey or sensors."""
    rows = []
    directions = {"+x": (1, 0), "-x": (-1, 0), "+y": (0, 1), "-y": (0, -1)}
    for m in PROTOCOL["prior_magnitude_xy_m"]:
        for direction in PROTOCOL["prior_direction"]:
            dx, dy = m * np.array(directions[direction])
            for delta in PROTOCOL["prior_yaw_deg"]:
                heading = np.arctan2(np.sin(yaw + np.deg2rad(delta)),
                                     np.cos(yaw + np.deg2rad(delta)))
                for label, (sp, sy) in PROTOCOL["prior_1sigma"].items():
                    rows.append(dict(case_id=f"{m}m_{direction}_{delta}d_{label}", dx=dx,
                                     dy=dy, magnitude_m=m, direction=direction, yaw_delta_deg=delta,
                                     confidence=label, px=xyz[0]+dx, py=xyz[1]+dy, pz=height,
                                     yaw_rad=heading, pos_std_m=sp, yaw_std_rad=np.deg2rad(sy)))
    pd.DataFrame(rows).to_csv(path, index=False, float_format="%.12g")


def at_zero(frame, columns):
    """Interpolate inside supported truth samples only for the intervention maker."""
    t = frame.t.to_numpy()
    if t[0] > 0 or t[-1] < 0:
        raise ValueError("Missing independent position truth at intervention")
    return [np.interp(0., t, frame[key]) for key in columns]


def delft_record(out, root, name, calibration, anchors):
    """Extract two sensor-selected episodes and never use truth to choose their origin."""
    raw_file = root / "twr" / name
    raw = pd.read_csv(raw_file, skipinitialspace=True)
    ticks = raw.timeTick.to_numpy(float) / 1000
    imu = raw[["accX", "accY", "accZ", "gyroX", "gyroY", "gyroZ"]].to_numpy(float)
    imu[:, :3] *= G
    imu[:, 3:] = np.deg2rad(imu[:, 3:])
    good = np.isfinite(imu).all(axis=1)
    accel_norm = np.linalg.norm(imu[:, :3], axis=1)
    first_imu = np.flatnonzero(good & (accel_norm > .5 * G) & (accel_norm < 1.5 * G))[0]
    ranges = raw[[f"twr{i}" for i in range(len(anchors))]].to_numpy(float)
    changed = np.r_[np.ones_like(ranges[:1], dtype=bool), np.abs(np.diff(ranges, axis=0)) > 1e-12]
    ti, ai = np.nonzero(changed & np.isfinite(ranges) & (ranges > 0))
    first = max(ticks[first_imu], ticks[ti[0]])
    last = min(ticks[good][-1], ticks[ti[-1]])
    truth_values = raw[["otX", "otY", "otZ"]].to_numpy(float)
    valid = np.isfinite(truth_values).all(axis=1) & (np.linalg.norm(truth_values, axis=1) > 1e-9)
    # Measured sensors run 0.20 s behind OptiTrack on the original logging axis.
    truth_times = ticks[valid] + .20
    truth_values = truth_values[valid]
    survey = pd.DataFrame(anchors, columns=["x", "y", "z"])
    survey.insert(0, "anchor_id", np.arange(len(anchors)))
    motion = Path(name).stem.rsplit("_", 1)[-1]
    source_hash = hashlib.sha256(raw_file.read_bytes()).hexdigest()
    episodes = []
    release_index = np.searchsorted(ticks, first + PROTOCOL["prefix_s"] - 1e-10)
    midpoint_index = np.searchsorted(ticks, (first + last) / 2, side="right") - 1
    for kind, index in (("start", release_index), ("recovery", midpoint_index)):
        origin = ticks[index]
        episode = f"delft_{motion}_{kind}"
        dest = out / "streams" / episode
        dest.mkdir(parents=True, exist_ok=True)
        t = np.round(ticks - origin, 9)
        cut = good & (t >= -PROTOCOL["prefix_s"] - 1e-9) & (t <= PROTOCOL["horizon_s"] + 1e-9)
        imf = pd.DataFrame(imu[cut], columns=["ax", "ay", "az", "gx", "gy", "gz"])
        imf.insert(0, "t", t[cut])
        imf.to_csv(dest / "imu.csv", index=False, float_format="%.12g")
        pick = (t[ti] >= 0) & (t[ti] <= PROTOCOL["horizon_s"] + 1e-9)
        j, ids = ti[pick], ai[pick]
        radio = pd.DataFrame(dict(t=t[j], anchor_id=ids,
            range_m=ranges[j, ids] - np.asarray(calibration["static"])[ids],
            sigma_m=np.asarray(calibration["sigma"])[ids]))
        radio.to_csv(dest / "radio.csv", index=False, float_format="%.12g")
        survey.to_csv(dest / "anchors.csv", index=False, float_format="%.12g")
        meta = dict(episode=episode, kind=kind, dataset="delft", motion=motion, source=name,
                    origin_sensor_tick_s=origin, lever_m=[0., 0., 0.], gravity_world_mps2=[0., 0., -G],
                    yaw_basis="sensor_zero_unobserved_physical_heading", source_sha256=source_hash,
                    common_range_correction="train-only per-anchor median, already subtracted",
                    **prefix_estimate(t[cut], imu[cut]))
        (dest / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        reference_t = truth_times - origin
        refcut = (reference_t >= -1.) & (reference_t <= PROTOCOL["horizon_s"] + 1.)
        reference = pd.DataFrame(truth_values[refcut], columns=["x", "y", "z"])
        reference.insert(0, "t", reference_t[refcut])
        reference.to_csv(out / "scoring" / (episode + ".csv"), index=False, float_format="%.12g")
        if kind == "recovery":
            write_priors(dest / "priors.csv", at_zero(reference, ["x", "y"]), 0., anchors[:, 2].mean())
        episodes.append(dict(episode=episode, file=name, release_s=origin, kind=kind, imu=len(imf),
                             radio=len(radio), prefix=meta["prefix_samples"], source_sha256=source_hash,
                             reference_first_s=reference.t.min(), reference_last_s=reference.t.max()))
    return episodes


def main():
    """Export the six existing evaluation flights without looking at estimator output."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--delft", type=Path, default=Path("E:/datasets/tu-delft-uwb-mhe/data_v1/UWB dataset"))
    args = parser.parse_args()
    (args.out / "scoring").mkdir(parents=True, exist_ok=True)
    calibration = json.loads((OLD_CAL / "calibration.json").read_text(encoding="utf-8"))
    lists = json.loads((OLD_CAL / "protocol.json").read_text(encoding="utf-8"))
    survey = yaml.safe_load((args.delft / "anchor_positions.yaml").read_text(encoding="utf-8"))
    anchors = np.array([[survey[i][axis] for axis in "xyz"] for i in range(8)])
    rows = []
    for name in lists["evaluation"]:
        rows.extend(delft_record(args.out, args.delft, name, calibration, anchors))
    manifest = pd.DataFrame(rows)
    manifest.to_csv(args.out / "delft_manifest.csv", index=False)
    print(manifest[["episode", "imu", "radio", "reference_first_s", "reference_last_s"]].to_string(index=False))


if __name__ == "__main__":
    main()
