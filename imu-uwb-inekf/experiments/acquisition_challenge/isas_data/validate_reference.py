"""Check frozen ISAS train calibration and plot held-out data diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

from calibrate_training import fit_gyro, fit_world, load_sequence, read_csv


def main():
    """Report held-out residuals without changing any calibration parameter."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    streams, scoring = args.root / "streams", args.root / "scoring"
    calibration = json.loads((scoring / "isas_walk1_calibration.json").read_text(encoding="utf-8"))
    world = Rotation.from_matrix(calibration["rtls_from_vive_rotation_matrix"])
    tracker_imu = Rotation.from_matrix(calibration["tracker_from_imu_rotation_matrix"])
    zup = Rotation.from_matrix(calibration["zup_from_native_rtls_rotation_matrix"])
    translation = np.asarray(calibration["rtls_from_vive_translation_m"])
    lever = np.asarray(calibration["tracker_to_tag_in_tracker_m"])
    lag = calibration["imu_time_minus_vive_time_s"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    diagnostics = {"calibration_policy": "Walk1 training only, no Walk2/3 fitting", "sequences": {}, "episodes": {}, "limitations": [
        "Reference is HTC VIVE, not Vicon. Native VIVE and RTLS frames differ.",
        "Tracker-to-IMU and VIVE-to-RTLS transformations were estimated on Walk1, not supplied by authors.",
        "Clock lag is an effective training gyro/VIVE delay; no independent hardware timing calibration exists.",
        "Raw IMU/radio estimator timestamps remain unchanged; score timestamps add fixed training gyro lag.",
        "Range-derived training reference calibration inherits UWB bias and outliers; it is not metrology-grade absolute pose.",
        "Half-training fits probe stability only; they do not form statistical confidence intervals."]}
    for index, sequence in enumerate(("isas_walk1", "isas_walk2", "isas_walk3")):
        imu, radio, anchors, gt_t, gt_pos, qs = load_sequence(streams / sequence)
        times = radio["t"]
        poses = np.column_stack([np.interp(times, gt_t, gt_pos[:, j]) for j in range(3)])
        tag = world.apply(poses + qs(times).apply(lever)) + translation
        a = np.array([anchors[int(value)] for value in radio["anchor_id"]])
        residual = np.linalg.norm(tag - a, axis=1) - radio["range_m"]
        gyro_times = np.arange(max(imu["t"][0], gt_t[0]) + 0.2, min(imu["t"][-1], gt_t[-1]) - 0.2, 0.04)
        tracker_rate = (qs(gyro_times - 0.03).inv() * qs(gyro_times + 0.03)).as_rotvec() / 0.06
        gyro = np.column_stack([np.interp(gyro_times + lag, imu["t"], imu[key]) for key in ("gx", "gy", "gz")])
        gyro_residual = tracker_imu.apply(gyro - calibration["gyro_bias_radps"]) - tracker_rate
        prefix = imu["t"] <= imu["t"][0] + 1.0
        acceleration = np.column_stack([imu[key][prefix] for key in ("ax", "ay", "az")])
        orientations = zup * world * qs(imu["t"][prefix] - lag) * tracker_imu
        force = orientations.apply(acceleration).mean(axis=0)
        diagnostics["sequences"][sequence] = {
            "range_rmse_m": float(np.sqrt(np.mean(residual ** 2))),
            "range_median_abs_m": float(np.median(np.abs(residual))),
            "range_p95_abs_m": float(np.percentile(np.abs(residual), 95)),
            "range_max_abs_m": float(np.max(np.abs(residual))),
            "range_count_abs_above_0p6m": int(np.sum(np.abs(residual) > 0.6)),
            "gyro_component_rmse_radps": float(np.sqrt(np.mean(gyro_residual ** 2))),
            "prefix_specific_force_zup_mps2": force.tolist(),
            "prefix_tilt_from_up_deg": float(np.rad2deg(np.arccos(force[2] / np.linalg.norm(force)))),
        }
        xy = zup.apply(world.apply(gt_pos) + translation)
        axes[0, 0].plot(xy[:, 0], xy[:, 1], label=sequence + (" (train)" if index == 0 else " (held out)"))
        axes[0, 1].plot(times, residual, ".", ms=1.5, alpha=0.5, label=sequence)
        axes[1, 0].plot(gyro_times, np.linalg.norm(gyro_residual, axis=1), lw=0.6, label=sequence)
        if index == 0:
            middle = (imu["t"][0] + imu["t"][-1]) / 2
            stability = {}
            for label, imu_select, radio_select in (("early", imu["t"] < middle, radio["t"] < middle),
                                                    ("late", imu["t"] >= middle, radio["t"] >= middle)):
                sensor_rot, _, split_lag, _ = fit_gyro(imu[imu_select], gt_t, qs)
                parameters, _ = fit_world(radio[radio_select], anchors, gt_t, gt_pos, qs)
                stability[label] = {
                    "tracker_rotation_difference_from_full_deg": float(np.rad2deg((tracker_imu.inv() * sensor_rot).magnitude())),
                    "world_rotation_difference_from_full_deg": float(np.rad2deg((world.inv() * Rotation.from_rotvec(parameters[:3])).magnitude())),
                    "world_translation_difference_from_full_m": float(np.linalg.norm(parameters[3:6] - translation)),
                    "tag_lever_difference_from_full_m": float(np.linalg.norm(parameters[6:9] - lever)),
                    "imu_time_minus_vive_time_s": split_lag,
                }
            diagnostics["training_half_stability"] = stability
    for sequence in ("isas_walk2", "isas_walk3"):
        source_imu = read_csv(streams / sequence / "imu.csv")
        source_radio = read_csv(streams / sequence / "radio.csv")
        for kind in ("start", "recovery"):
            identifier = f"{sequence}_{kind}"
            directory = streams / identifier
            metadata = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
            imu, radio = read_csv(directory / "imu.csv"), read_csv(directory / "radio.csv")
            score = read_csv(scoring / f"{identifier}.csv")
            start, end = metadata["source_start_s"], metadata["source_end_s"]
            selected_imu = source_imu[(source_imu["t"] >= start - 1.5) & (source_imu["t"] <= end)]
            selected_radio = source_radio[(source_radio["t"] >= start) & (source_radio["t"] <= end)]
            original_imu = imu[imu["t"] != 0] if metadata["imu_time_zero_held_boundary"] else imu
            unchanged = all(np.array_equal(original_imu[key], selected_imu[key]) for key in ("ax", "ay", "az", "gx", "gy", "gz"))
            unchanged &= all(np.array_equal(radio[key], selected_radio[key]) for key in ("anchor_id", "range_m", "sigma_m"))
            assert unchanged, identifier + " modified physical measurements"
            assert np.all(np.diff(imu["t"]) > 0) and np.all(np.diff(radio["t"]) >= 0)
            assert np.min(imu["t"]) >= -1.5 and np.max(imu["t"]) <= 15
            assert np.sum(imu["t"] == 0) == 1
            assert score["t"][0] < 0 and score["t"][-1] > 15
            if metadata["imu_time_zero_held_boundary"]:
                boundary = np.flatnonzero(imu["t"] == 0)[0]
                assert all(imu[key][boundary] == imu[key][boundary - 1] for key in ("ax", "ay", "az", "gx", "gy", "gz"))
            if kind == "recovery":
                priors = read_csv(directory / "priors.csv")
                assert len(priors) == 96
                truth_xy = [np.interp(0, score["t"], score[key]) for key in ("x", "y")]
                truth_yaw = np.interp(0, score["t"], np.unwrap(score["yaw_rad"]))
                assert np.allclose(priors["px"] - priors["dx"], truth_xy[0], atol=1e-9)
                assert np.allclose(priors["py"] - priors["dy"], truth_xy[1], atol=1e-9)
                yaw_error = priors["yaw_rad"] - truth_yaw - np.deg2rad(priors["yaw_delta_deg"])
                assert np.max(np.abs(np.arctan2(np.sin(yaw_error), np.cos(yaw_error)))) < 1e-9
            diagnostics["episodes"][identifier] = {"sensor_values_exactly_preserved": bool(unchanged),
                "source_start_s": start, "samples_imu": len(imu), "samples_radio": len(radio),
                "score_rows": len(score), "reference_yaw_at_zero_deg": float(np.rad2deg(np.interp(0, score["t"], np.unwrap(score["yaw_rad"])))),
                "score_position_origin": "IMU", "test_fitting_or_alignment": False}
            axes[1, 1].plot(score["t"], np.rad2deg(np.unwrap(score["yaw_rad"])), label=identifier)
    axes[0, 0].set(xlabel="z-up world x (m)", ylabel="z-up world y (m)", title="Reference tracker trajectory, fixed Walk1 calibration")
    axes[0, 0].axis("equal")
    axes[0, 1].set(xlabel="sequence time (s)", ylabel="predicted - corrected range (m)", title="Range residuals including all outliers (symlog)")
    axes[0, 1].set_yscale("symlog", linthresh=0.3)
    axes[1, 0].set(xlabel="sequence time (s)", ylabel="gyro residual norm (rad/s)", title="Frozen tracker-to-IMU rotation and delay (symlog)")
    axes[1, 0].set_yscale("symlog", linthresh=0.1)
    axes[1, 1].set(xlabel="episode time (s)", ylabel="IMU yaw (deg)", title="Held-out heading reference, no test alignment")
    for axis in axes.flat:
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=8)
    fig.suptitle("ISAS-Walk independent training calibration diagnostics")
    fig.savefig(scoring / "isas_reference_diagnostics.png", dpi=160)
    plt.close(fig)
    (scoring / "isas_reference_diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
