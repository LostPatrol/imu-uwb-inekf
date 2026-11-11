"""Evaluate common full-SO(3) raw-radio acquisition on nonstatic physical windows."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

# Bound BLAS workers independently of the number of physical trials.
for _name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from cases import (GRAVITY, make_case, gravity_seed, compensated_gravity_seed, wahba_seed, profile_rotation_information,
                   residual_jacobian, rotation_error_deg)

# Shared solvers are consumed through the parent-provided API without modifications.
HERE = Path(__file__).resolve().parent
RADIO_SOURCE = HERE.parents[1]
sys.path.insert(0, str(RADIO_SOURCE))
import rotation_window as common
sys.path.insert(0, str(RADIO_SOURCE.parent / "recovery_bootstrap"))
import initializer

# The cube rotation group covers different yaw and tilt basins without truth access.
MULTI_ROTATIONS = Rotation.create_group("O").as_matrix()
# A strict geodesic threshold reports acquisition, not calibrated covariance coverage.
ROTATION_TOLERANCE_DEG = 10.0


def angle_between(left, right):
    """Return the angle between normalized three-vectors in degrees."""
    return float(np.rad2deg(np.arccos(np.clip(np.dot(left, right) /
        (np.linalg.norm(left) * np.linalg.norm(right)), -1., 1.))))


def verify():
    """Check exact physical reconstruction, arbitrary-tilt Wahba, and range Jacobians."""
    checks = []
    for motion in ("rich", "weak", "constant_acceleration", "constant_velocity"):
        for tilt in (0., 90., 179.):
            case = make_case(20261111, motion, tilt)
            p, v, c, singular = wahba_seed(case, case.truth_tag_positions)
            info = profile_rotation_information(case)
            error = rotation_error_deg(c, case.truth_rotation)
            if motion in ("rich", "weak"):
                assert error < 1e-5
            else:
                assert info[0] < 1e-7
            checks.append(dict(motion=motion, tilt_deg=tilt, oracle_path_rotation_error_deg=error,
                               oracle_path_cross_singular=singular.tolist(), information_eigenvalues=info.tolist(),
                               minimum_gyro_radps=float(np.linalg.norm(case.gyro, axis=1).min())))
    case = make_case(20261111, "rich", 150.)
    analytic = residual_jacobian(case.truth_position, case.truth_velocity, case.truth_rotation, case)[1]
    numeric = np.empty_like(analytic)
    for index in range(9):
        step = np.eye(9)[index] * 1e-6
        plus = residual_jacobian(case.truth_position+step[:3], case.truth_velocity+step[3:6],
            case.truth_rotation@Rotation.from_rotvec(step[6:]).as_matrix(), case)[0]
        minus = residual_jacobian(case.truth_position-step[:3], case.truth_velocity-step[3:6],
            case.truth_rotation@Rotation.from_rotvec(-step[6:]).as_matrix(), case)[0]
        numeric[:, index] = (plus-minus)/2e-6
    difference = float(np.max(np.abs(analytic-numeric)))
    np.testing.assert_allclose(analytic, numeric, rtol=1e-5, atol=1e-6)
    # Cross-check the parent's Wahba convention using exact positions only as a diagnostic.
    c, _ = common.wahba_seed(case.tau, case.shape, case.truth_tag_positions)
    assert rotation_error_deg(c, case.truth_rotation) < 1e-5
    return dict(cases=checks, analytic_jacobian_max_abs=difference,
                oracle_path_diagnostic_only=True, gyro_bias_radps=0., accel_bias_mps2=0.)


def evaluate(case, include_multi=True, only_local_gravity=False):
    """Use identical raw observations, radio path, and refinement model for all seeds."""
    start = time.perf_counter()
    cv, _, _ = initializer.cv_candidate(case.tau, case.anchors, case.ranges, case.sigmas)
    path = common.radio_seed_path(case.tau, case.anchors, None, case.ranges, case.sigmas, cv)
    shared_seconds = time.perf_counter()-start
    gravity = gravity_seed(case)
    seeds = [("single_identity", np.eye(3)), ("moving_gravity", gravity), ("radio_wahba", None)]
    if only_local_gravity:
        seeds = [("local_gravity", compensated_gravity_seed(case))]
    rows = []
    information = profile_rotation_information(case)
    def solve(rotation):
        """Call the parent's unchanged original-range SO(3) refinement."""
        return common.solve_attitude(case.tau, case.shape, case.anchors, None,
                                     case.ranges, case.sigmas, path, seed_rotation=rotation)
    fits = []
    for method, seed in seeds:
        started = time.perf_counter()
        fit = solve(seed)
        fits.append((method, fit, time.perf_counter()-started))
    if include_multi:
        started = time.perf_counter()
        alternatives = [solve(rotation) for rotation in MULTI_ROTATIONS]
        fit = min(alternatives, key=lambda item:item["cost"])
        fit = dict(fit, nfev=sum(item["nfev"] for item in alternatives),
                   nonconverged_starts=sum(not item["success"] for item in alternatives))
        fits.append(("cube24", fit, time.perf_counter()-started))
    best_cost = min(fit["cost"] for _,fit,_ in fits)
    for method, fit, seconds in fits:
        estimated = fit["C"]
        endpoint_rotation = estimated @ case.local_rotations[-1]
        true_endpoint_rotation = case.truth_rotation @ case.local_rotations[-1]
        heading = np.arctan2(endpoint_rotation[1, 0], endpoint_rotation[0, 0])
        truth_heading = np.arctan2(true_endpoint_rotation[1, 0], true_endpoint_rotation[0, 0])
        heading_error = abs(np.rad2deg(np.arctan2(np.sin(heading-truth_heading), np.cos(heading-truth_heading))))
        rotation_error = rotation_error_deg(estimated, case.truth_rotation)
        rows.append(dict(seed=case.seed, motion=case.motion, tilt_deg=case.tilt_deg, geometry=case.geometry,
            method=method, events=len(case.tau), position_error_m=float(np.linalg.norm(fit["x"][:3]-case.truth_position)),
            velocity_error_mps=float(np.linalg.norm(fit["x"][3:6]-case.truth_velocity)),
            rotation_error_deg=rotation_error, endpoint_heading_error_deg=float(heading_error),
            tilt_error_deg=angle_between(estimated.T@[0.,0.,1.], case.truth_rotation.T@[0.,0.,1.]),
            seed_rotation_error_deg=rotation_error_deg(fit["C0"], case.truth_rotation),
            success=bool(fit["success"]), acquired=bool(rotation_error <= ROTATION_TOLERANCE_DEG),
            nfev=fit["nfev"], nonconverged_starts=fit.get("nonconverged_starts", int(not fit["success"])),
            cost=fit["cost"], cost_miss=bool(fit["cost"]-best_cost > max(.001, 1e-5*abs(best_cost))),
            rotation_info_min=float(information[0]), rotation_info_mid=float(information[1]), rotation_info_max=float(information[2]),
            gyro_norm_min_radps=float(np.linalg.norm(case.gyro, axis=1).min()),
            moving_gravity_tilt_error_deg=angle_between(gravity.T@[0.,0.,1.], case.truth_rotation.T@[0.,0.,1.]),
            shared_seed_seconds=shared_seconds, refine_seconds=seconds))
    return rows


def summarize(frame):
    """Keep every physical regime and every failure in descriptive statistics."""
    rows = []
    for keys, part in frame.groupby(["motion", "geometry", "tilt_deg", "method"]):
        row = dict(zip(("motion", "geometry", "tilt_deg", "method"), keys))
        row.update(cases=len(part), acquired=int(part.acquired.sum()), cost_misses=int(part.cost_miss.sum()),
                   nonconverged=int((~part.success).sum()))
        for name in ("position_error_m", "velocity_error_mps", "rotation_error_deg", "endpoint_heading_error_deg",
                     "seed_rotation_error_deg", "rotation_info_min", "refine_seconds", "nfev"):
            row[name+"_median"] = float(part[name].median())
            row[name+"_p95"] = float(part[name].quantile(.95))
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    """Freeze an independent nonstatic arbitrary-attitude synthetic panel."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--tilts", type=float, nargs="+", default=[0., 60., 120., 179.])
    parser.add_argument("--motions", nargs="+", default=["rich", "weak", "constant_acceleration", "constant_velocity"])
    parser.add_argument("--geometries", nargs="+", default=["normal", "weak_height"])
    parser.add_argument("--no-multi", action="store_true")
    parser.add_argument("--output", type=Path, default=HERE / "results")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    checks = verify()
    rows = []
    for seed_index in range(args.seeds):
        for motion in args.motions:
            for geometry in args.geometries:
                for tilt in args.tilts:
                    rows.extend(evaluate(make_case(2026111100+seed_index, motion, tilt, geometry), not args.no_multi))
        print(f"Completed {seed_index+1}/{args.seeds} paired seed groups", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output / "metrics.csv", index=False)
    summary = summarize(frame)
    summary.to_csv(args.output / "summary.csv", index=False)
    metadata = dict(seeds=list(range(2026111100, 2026111100+args.seeds)), tilts=args.tilts,
        motions=args.motions, geometries=args.geometries, include_cube24=not args.no_multi,
        scope="Independent physical simulation of moving-window cold acquisition; no real-data generalization",
        no_static_prefix=True, local_initial_attitude="identity", sensor_bias="exactly zero", imu_white_noise=False,
        range_count=240, duration_s=4., imu_rate_hz=200., range_sigma_m=.1, stand_off_m=30., lever_arm_m=[0.,0.,0.],
        seeding="Same radio-only path and original observation budget for every method; cube24 spends 24 refinements",
        oracle_path="Used only in verification, never final raw-range refinement", verification=checks,
        source_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in
                       (Path(__file__), HERE / "cases.py", RADIO_SOURCE / "rotation_window.py", RADIO_SOURCE / "window_dynamics.py")})
    (args.output / "protocol.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    pooled = frame.groupby(["motion", "method"]).agg(cases=("acquired", "size"), acquired=("acquired", "sum"),
        rotation_median_deg=("rotation_error_deg", "median"), position_median_m=("position_error_m", "median"),
        cost_misses=("cost_miss", "sum"), nonconverged=("success", lambda x:int((~x).sum())))
    pooled.to_csv(args.output / "pooled_summary.csv")
    print(pooled.to_string())


if __name__ == "__main__":
    main()
