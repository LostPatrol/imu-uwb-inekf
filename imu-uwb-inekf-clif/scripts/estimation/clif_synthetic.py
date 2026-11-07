"""Synthetic end-to-end check of the CLIF filter.

Runs the joint filter on a bounded truth trajectory with a large initial
position offset and reports whether it converges.  The point is not accuracy
against a tuned baseline but whether the single-covariance lifted structure
actually pulls the estimate in from far away instead of drifting.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ESTIMATION_DIR = Path(__file__).resolve().parent
if str(ESTIMATION_DIR) not in sys.path:
    sys.path.insert(0, str(ESTIMATION_DIR))

import clif_core as core  # noqa: E402
from clif import ClifParameters, ClifTWRInEKF  # noqa: E402
from inekf import SE3_2_exp  # noqa: E402

# Gravity magnitude reused for the truth model.
G = 9.81

# Eight anchors of a 10 m x 10 m x 3 m room, metres.
ANCHORS = np.array(
    [
        [-5.0, -5.0, 0.0],
        [5.0, -5.0, 0.0],
        [-5.0, 5.0, 0.0],
        [5.0, 5.0, 0.0],
        [-5.0, -5.0, 3.0],
        [5.0, -5.0, 3.0],
        [-5.0, 5.0, 3.0],
        [5.0, 5.0, 3.0],
    ]
)

# True IMU biases; the filter starts with zero estimates.
TRUE_GYRO_BIAS = np.array([0.012, -0.009, 0.007])
TRUE_ACCEL_BIAS = np.array([0.08, -0.05, 0.04])


def truth_position(t: float) -> np.ndarray:
    """Bounded trajectory well inside the anchor box."""
    return np.array([2.0 * math.sin(0.5 * t), 1.6 * math.sin(0.42 * t + 0.3), 1.5 + 0.4 * math.sin(0.7 * t)])


def truth_velocity(t: float) -> np.ndarray:
    """Analytic derivative of :func:`truth_position`."""
    return np.array([1.0 * math.cos(0.5 * t), 0.672 * math.cos(0.42 * t + 0.3), 0.28 * math.cos(0.7 * t)])


def truth_acceleration(t: float) -> np.ndarray:
    """Analytic second derivative of :func:`truth_position`."""
    return np.array([-0.5 * math.sin(0.5 * t), -0.28224 * math.sin(0.42 * t + 0.3), -0.196 * math.sin(0.7 * t)])


def truth_omega(t: float) -> np.ndarray:
    """Body angular rate, rad/s."""
    return np.array([0.35 * math.sin(0.6 * t), 0.25 * math.cos(0.45 * t), 0.6 + 0.15 * math.sin(0.25 * t)])


@dataclass
class CaseResult:
    """Summary of one synthetic run."""

    offset_m: float
    position_rmse_full_m: float
    position_rmse_tail_m: float
    initial_error_m: float
    convergence_time_s: float
    accepted: int
    rejected: int
    manifold_residual_m2: float
    worst_error_m: float = 0.0
    p95_error_m: float = 0.0


def run_case(offset: np.ndarray, duration: float = 30.0, seed: int = 7) -> CaseResult:
    """Run one synthetic case with the given initial position offset."""
    rng = np.random.default_rng(seed)
    dt_imu = 1.0e-3
    steps = int(round(duration / dt_imu))
    uwb_period = int(round(0.05 / dt_imu))  # 20 Hz aggregate range rate

    yaw = math.radians(40.0)
    rotation = np.array([[math.cos(yaw), -math.sin(yaw), 0.0], [math.sin(yaw), math.cos(yaw), 0.0], [0.0, 0.0, 1.0]])
    start = np.zeros((5, 5))
    start[3, 3] = 1.0
    start[4, 4] = 1.0
    start[:3, :3] = rotation
    start[:3, 3] = truth_velocity(0.0)
    start[:3, 4] = truth_position(0.0) + offset

    initial_position = np.zeros((5, 5))
    initial_position[3, 3] = 1.0
    initial_position[4, 4] = 1.0
    initial_position[:3, :3] = rotation
    initial_position[:3, 3] = truth_velocity(0.0)
    initial_position[:3, 4] = truth_position(0.0)

    # Initial covariance: position uncertainty scaled to the offset.
    P0 = np.diag(
        [0.1**2] * 3 + [0.2**2] * 3 + [max(1.0, float(np.linalg.norm(offset))) ** 2] * 3 + [0.05**2] * 3 + [0.2**2] * 3
    )
    parameters = ClifParameters(quadratic_walk_std=1.0, gate_warmup_updates=64)
    estimator = ClifTWRInEKF(start, P0, np.zeros(6), ANCHORS, parameters)

    position_error = np.empty(steps)
    truth_rotation = rotation.copy()
    truth_velocity_value = truth_velocity(0.0)

    anchor_cursor = 0
    for k in range(steps):
        t = k * dt_imu
        truth_rotation = truth_rotation @ core.so3_exp(truth_omega(t) * dt_imu)
        world_acceleration = truth_acceleration(t)
        specific_force = truth_rotation.T @ (world_acceleration - np.array([0.0, 0.0, -G]))
        gyro_measured = truth_omega(t) + TRUE_GYRO_BIAS + rng.normal(0.0, 0.002, 3)
        accel_measured = specific_force + TRUE_ACCEL_BIAS + rng.normal(0.0, 0.03, 3)
        estimator.predict(gyro_measured, accel_measured, dt_imu)
        truth_velocity_value = truth_velocity_value + world_acceleration * dt_imu

        if k % uwb_period == 0:
            anchor_index = anchor_cursor % len(ANCHORS)
            anchor_cursor += 1
            true_range = float(np.linalg.norm(ANCHORS[anchor_index] - truth_position(t)))
            measured = true_range + parameters.range_bias + rng.normal(0.0, parameters.range_sigma)
            estimator.correct_twr(anchor_index, measured)

        position_error[k] = float(np.linalg.norm(estimator.position - truth_position(t)))

    tail = position_error[-int(round(5.0 / dt_imu)) :]
    threshold = 1.0
    below = np.where(position_error < threshold)[0]
    convergence = float(below[0] * dt_imu) if below.size else float("nan")
    return CaseResult(
        offset_m=float(np.linalg.norm(offset)),
        position_rmse_full_m=float(np.sqrt(np.mean(position_error**2))),
        position_rmse_tail_m=float(np.sqrt(np.mean(tail**2))),
        worst_error_m=float(np.max(position_error)),
        p95_error_m=float(np.percentile(position_error, 95)),
        initial_error_m=float(position_error[0]),
        convergence_time_s=convergence,
        accepted=estimator.stats.accepted,
        rejected=estimator.stats.rejected,
        manifold_residual_m2=float(np.max(np.abs(estimator.manifold_residual()))),
    )


def main() -> int:
    """Run the nominal and the large-offset case and report both."""
    parser = argparse.ArgumentParser(description="Synthetic convergence check for CLIF")
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--output", type=Path, default=Path("results/clif_synthetic"))
    args = parser.parse_args()

    cases = {
        "nominal": np.zeros(3),
        "large_5m": np.array([4.0, -3.0, 2.0]),
    }
    report = {}
    for name, offset in cases.items():
        result = run_case(offset, duration=args.duration)
        report[name] = result.__dict__
        print(f"{name}: {result}")

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "clif_synthetic.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    converged = all(np.isfinite(report[name]["convergence_time_s"]) for name in report)
    return 0 if converged else 1


if __name__ == "__main__":
    raise SystemExit(main())
