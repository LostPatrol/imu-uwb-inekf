"""UTIL flight data with synthetic TWR: CLIF against the raw-range InEKF.

UTIL ships IMU and Vicon truth but only TDoA ranges, so absolute TWR is
synthesised from the truth and the surveyed anchors.  That is enough to check
the capture behaviour of the lifted filter on the author's own dataset, with a
raw-range TWR-InEKF on exactly the same measurement stream as the baseline.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ESTIMATION_DIR = Path(__file__).resolve().parent
if str(ESTIMATION_DIR) not in sys.path:
    sys.path.insert(0, str(ESTIMATION_DIR))

import clif_core as core  # noqa: E402
from clif import ClifParameters, ClifTWRInEKF  # noqa: E402
from inekf import InEKF, SE3_2_exp  # noqa: E402

# Default UTIL root; the AGENTS.md layout puts it here.
DEFAULT_ROOT = Path(r"E:\datasets\dataset\flight-dataset")

# Gravity, m/s^2.
G = 9.81

# Accelerometer values in the CSV are in g, gyro in deg/s.
ACCEL_TO_SI = 9.81
DEG_TO_RAD = math.pi / 180.0


@dataclass
class Flight:
    """One UTIL flight reduced to the arrays the filters need."""

    time: np.ndarray
    gyro: np.ndarray  # rad/s
    accel: np.ndarray  # m/s^2
    truth_position: np.ndarray
    truth_rotation: np.ndarray


def load_flight(root: Path, constellation: str, name: str, duration: float) -> tuple[Flight, np.ndarray]:
    """Read one UTIL CSV and resample IMU and truth onto a common grid."""
    frame = pd.read_csv(root / "csv-data" / constellation / name)
    survey = np.load(root / "survey-results" / f"anchor_{constellation}.npz")
    anchors = np.asarray(survey["an_pos"], dtype=float)

    accel_rows = frame[np.isfinite(frame["t_acc"])]
    gyro_rows = frame[np.isfinite(frame["t_gyro"])]
    pose_rows = frame[np.isfinite(frame["t_pose"])]

    t0 = float(min(accel_rows["t_acc"].iloc[0], gyro_rows["t_gyro"].iloc[0], pose_rows["t_pose"].iloc[0]))
    time = accel_rows["t_acc"].to_numpy() - t0
    keep = time <= duration
    time = time[keep]

    accel = accel_rows[["acc_x", "acc_y", "acc_z"]].to_numpy()[keep] * ACCEL_TO_SI
    gyro = np.column_stack(
        [
            np.interp(time, gyro_rows["t_gyro"].to_numpy() - t0, gyro_rows[axis].to_numpy())
            for axis in ("gyro_x", "gyro_y", "gyro_z")
        ]
    ) * DEG_TO_RAD

    pose_time = pose_rows["t_pose"].to_numpy() - t0
    position = np.column_stack(
        [np.interp(time, pose_time, pose_rows[axis].to_numpy()) for axis in ("pose_x", "pose_y", "pose_z")]
    )
    quaternion = np.column_stack(
        [
            np.interp(time, pose_time, pose_rows[axis].to_numpy())
            for axis in ("pose_qx", "pose_qy", "pose_qz", "pose_qw")
        ]
    )
    rotation = np.array([_quaternion_to_matrix(q) for q in quaternion])
    return Flight(time=time, gyro=gyro, accel=accel, truth_position=position, truth_rotation=rotation), anchors


def _quaternion_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    """Convert an (x, y, z, w) quaternion to a rotation matrix."""
    x, y, z, w = quaternion / max(float(np.linalg.norm(quaternion)), 1e-12)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def run_filter(flight: Flight, anchors: np.ndarray, offset: np.ndarray, range_sigma: float, which: str) -> dict:
    """Run one filter on synthetic TWR ranges and return its error statistics."""
    start = np.zeros((5, 5))
    start[3, 3] = 1.0
    start[4, 4] = 1.0
    start[:3, :3] = flight.truth_rotation[0]
    start[:3, 3] = np.zeros(3)
    start[:3, 4] = flight.truth_position[0] + offset

    P0 = np.diag(
        [0.05**2] * 3
        + [0.2**2] * 3
        + [max(1.0, float(np.linalg.norm(offset))) ** 2] * 3
        + [0.05**2] * 3
        + [0.2**2] * 3
    )
    rng = np.random.default_rng(11)
    if which == "clif":
        estimator = ClifTWRInEKF(start, P0, np.zeros(6), anchors, ClifParameters(gate_warmup_updates=64))
    else:
        estimator = InEKF(start, P0, np.zeros(6))
        estimator.w_gyr = np.ones(3) * 0.1**2
        estimator.w_acc = np.ones(3) * 2.0**2
        estimator.t_uv = np.zeros(3)

    errors = np.empty(len(flight.time))
    anchor_cursor = 0
    uwb_stride = max(1, int(round(0.05 / (flight.time[1] - flight.time[0]))))
    for k, t in enumerate(flight.time):
        gyro = flight.gyro[k]
        accel = flight.accel[k]
        if which == "clif":
            estimator.predict(gyro, accel, float(flight.time[1] - flight.time[0]))
        else:
            estimator.predict(np.concatenate([accel / G, gyro / DEG_TO_RAD]), float(flight.time[1] - flight.time[0]))
        if k % uwb_stride == 0:
            index = anchor_cursor % len(anchors)
            anchor_cursor += 1
            true_range = float(np.linalg.norm(anchors[index] - flight.truth_position[k]))
            measured = true_range + rng.normal(0.0, range_sigma)
            if which == "clif":
                estimator.correct_twr(index, measured)
            else:
                _raw_range_update(estimator, anchors[index], measured, range_sigma)
        errors[k] = float(np.linalg.norm(np.asarray(estimator.X[:3, 4]).reshape(3) - flight.truth_position[k]))

    return {
        "rmse_m": float(np.sqrt(np.mean(errors**2))),
        "p95_m": float(np.percentile(errors, 95)),
        "max_m": float(np.max(errors)),
        "tail_rmse_m": float(np.sqrt(np.mean(errors[-int(5.0 / (flight.time[1] - flight.time[0])) :] ** 2))),
    }


def _raw_range_update(estimator: InEKF, anchor: np.ndarray, measured: float, sigma: float) -> None:
    """Single raw-range update on the author's 15-dim invariant error."""
    tag = estimator.X_hat[:3, 4]
    displacement = anchor - tag
    predicted = float(np.linalg.norm(displacement))
    if predicted < 1e-9:
        return
    left = np.zeros((1, 15))
    left[0, 6:9] = displacement / predicted
    from inekf import SE3_2_adjoint, SE3_2_inv

    full = np.zeros((15, 15))
    full[:9, :9] = SE3_2_adjoint(SE3_2_inv(estimator.X_hat))
    full[9:, 9:] = np.eye(6)
    row = left @ full
    innovation = measured - predicted
    variance = float(row @ estimator.P_hat @ row.T) + sigma**2
    gain = estimator.P_hat @ row.T / variance
    delta = gain[:, 0] * innovation
    residual = np.eye(15) - gain @ row
    estimator.P = residual @ estimator.P_hat @ residual.T + sigma**2 * gain @ gain.T
    estimator.X = SE3_2_exp(-delta[:9]) @ estimator.X_hat
    estimator.bias = estimator.bias - delta[9:15]
    estimator.X_hat = estimator.X.copy()
    estimator.P_hat = estimator.P.copy()


def main() -> int:
    """Compare CLIF with the raw-range InEKF on synthetic UTIL ranges."""
    parser = argparse.ArgumentParser(description="UTIL synthetic TWR comparison")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--constellation", default="const1")
    parser.add_argument("--file", default="const1-trial1-tdoa2.csv")
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--range-sigma", type=float, default=0.10)
    parser.add_argument("--output", type=Path, default=Path("results/clif_util"))
    args = parser.parse_args()

    flight, anchors = load_flight(args.root, args.constellation, args.file, args.duration)
    report = {}
    for name, offset in (("nominal", np.zeros(3)), ("offset_3m", np.array([2.0, -2.0, 0.6]))):
        for which in ("clif", "raw_inekf"):
            result = run_filter(flight, anchors, offset, args.range_sigma, which)
            report[f"{name}/{which}"] = result
            print(f"{name:10s} {which:10s} {result}")

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "clif_util.json").open("w", encoding="utf-8") as handle:
        json.dump({"file": args.file, "duration_s": args.duration, "results": report}, handle, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
