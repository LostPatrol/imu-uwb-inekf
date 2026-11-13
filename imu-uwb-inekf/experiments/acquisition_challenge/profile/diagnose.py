"""Post-hoc initialization audit; frozen challenge inputs and results are untouched.

Reused rolling-eight-second outputs are single-recording unconditional resets.
Extra bias/history variants diagnose model limitations and cannot select a new main.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import json
from pathlib import Path
import time

from run import (np, pd, Rotation, GRAVITY, load_episode, make_window, attempt,
                 to_row, at_times, digest)
from scipy.spatial.transform import Slerp

# Only these endpoints and predeclared strategies are audited; no outcome tuning.
ENDPOINTS = (2., 10., 15.)
MODEL = "body_bias_prior"


def score_row(row, reference):
    """Interpolate held-out reference solely after estimation for error reporting."""
    t = row["state_s"]
    truth = np.array([np.interp(t, reference.t, reference[key]) for key in ("x", "y", "z")])
    error = np.array([row[key] for key in ("x", "y", "z")])-truth
    row.update(xy_m=float(np.linalg.norm(error[:2])), error_3d_m=float(np.linalg.norm(error)),
        z_error_m=float(error[2]), yaw_error_deg=np.nan)
    if "yaw_rad" in reference:
        yaw = np.interp(t, reference.t, np.unwrap(reference.yaw_rad))
        row["yaw_error_deg"] = float(np.rad2deg(abs(np.arctan2(np.sin(row["yaw_rad"]-yaw),
            np.cos(row["yaw_rad"]-yaw)))))
    return row


def reuse_baselines(root, results, output):
    """Reuse each existing native reset once instead of inventing 96 repetitions."""
    panels = {name:pd.read_csv(results/(name+".csv")) for name in ("native", "stress", "trajectory")}
    rows = []
    for name, panel in panels.items():
        keep = panel[(panel.method == MODEL) & panel.arrival_s.isin(ENDPOINTS)
                     & panel.episode.str.endswith("_recovery")]
        for source in keep.to_dict("records"):
            ref = pd.read_csv(root/"scoring"/(source["episode"]+".csv"))
            source["source_csv"] = name+".csv"
            source["window_policy"] = "rolling_max8" if name == "native" else "growing_from_zero"
            source["comparison"] = ("blind_discard_prior_sensor_reset" if source["case_id"] == "native"
                else "common_MAP_extension_"+source["confidence"])
            rows.append(score_row(source, ref))
    frame = pd.DataFrame(rows)
    frame.to_csv(output/"reused_single_record_baselines.csv", index=False)
    # Summaries preserve individual recordings and distinguish real fits from case dimensions.
    grouped = frame.groupby(["episode", "arrival_s", "source_csv", "comparison"])
    grouped.agg(rows=("xy_m", "size"), xy_median_m=("xy_m", "median"),
        xy_p95_m=("xy_m", lambda x:x.quantile(.95)), yaw_median_deg=("yaw_error_deg", "median"),
        solver_successes=("solver_success", "sum")).to_csv(output/"reused_per_recording.csv")
    selected = frame[frame.case_id == "native"]
    columns = ["episode", "arrival_s", "xy_m", "yaw_error_deg", "solver_success"]
    rolling = selected[selected.source_csv == "native.csv"][columns]
    growing = selected[selected.source_csv == "trajectory.csv"][columns]
    paired = rolling.merge(growing, on=["episode", "arrival_s"], suffixes=("_rolling8", "_growing"),
        validate="one_to_one")
    paired.to_csv(output/"blind_reset_paired_by_record.csv", index=False)
    summary = frame.copy()
    summary["dataset"] = summary.episode.str.split("_").str[0]
    per_record = summary.groupby(["dataset", "episode", "arrival_s", "source_csv", "comparison"]).xy_m.median()
    per_record.groupby(["dataset", "arrival_s", "source_csv", "comparison"]).agg(
        ["count", "median", "max"]).to_csv(output/"reused_dataset_summary.csv")
    start_rows = []
    start = panels["native"]
    start = start[(start.method == MODEL) & start.arrival_s.isin(ENDPOINTS) & start.episode.str.endswith("_start")]
    for row in start.to_dict("records"):
        start_rows.append(score_row(row, pd.read_csv(root/"scoring"/(row["episode"]+".csv"))))
    pd.DataFrame(start_rows).to_csv(output/"frozen_start_baselines.csv", index=False)


def reference_orientation(root, episode, meta):
    """Load the independent Walk1-calibrated full ISAS attitude for diagnostics only."""
    if meta["dataset"] != "isas":
        return None
    calibration = json.loads((root/"scoring/isas_walk1_calibration.json").read_text(encoding="utf-8"))
    gt = pd.read_csv(root/"streams"/meta["source_sequence"]/"score/ground_truth_native.csv")
    orientation = (Rotation.from_matrix(calibration["zup_from_native_rtls_rotation_matrix"])
        * Rotation.from_matrix(calibration["rtls_from_vive_rotation_matrix"])
        * Rotation.from_quat(gt[["qx", "qy", "qz", "qw"]].to_numpy())
        * Rotation.from_matrix(calibration["tracker_from_imu_rotation_matrix"]))
    times = gt.t.to_numpy()+calibration["imu_time_minus_vive_time_s"]-meta["source_start_s"]
    return Slerp(times, orientation)


def prefix_statistics(root, episode):
    """Describe the accepted prefix and compare actual motion where pose exists."""
    meta = json.loads((episode/"meta.json").read_text(encoding="utf-8"))
    imu = pd.read_csv(episode/"imu.csv")
    prefix = imu[(imu.t >= -1.5-1e-9) & (imu.t < 0)]
    force = prefix[["ax", "ay", "az"]].to_numpy()
    gyro = prefix[["gx", "gy", "gz"]].to_numpy()
    norms = np.linalg.norm(force, axis=1)
    still = ((norms > .85*9.81) & (norms < 1.15*9.81)
             & (np.linalg.norm(gyro, axis=1) < np.deg2rad(8)))
    accepted = still if still.sum() >= 20 else np.ones(len(still), bool)
    row = dict(episode=episode.name, dataset=meta["dataset"], prefix_samples=len(prefix),
        stationary_gate_samples=int(still.sum()), fallback_to_all_prefix=bool(still.sum() < 20),
        accepted_samples=int(accepted.sum()), accepted_fraction=float(accepted.mean()),
        force_norm_mean_mps2=float(norms.mean()), force_norm_std_mps2=float(norms.std()),
        accepted_force_norm_mean_mps2=float(norms[accepted].mean()),
        accepted_gyro_mean_norm_radps=float(np.linalg.norm(gyro[accepted].mean(axis=0))),
        accepted_gyro_rms_norm_radps=float(np.sqrt(np.mean(np.sum(gyro[accepted]**2, axis=1)))),
        gyro_component_std_radps=float(np.sqrt(np.mean(np.var(gyro[accepted], axis=0)))),
        accel_bias_norm_mps2=float(np.linalg.norm(meta["accel_bias"])))
    orientation = reference_orientation(root, episode, meta)
    if orientation is not None:
        times = prefix.t.to_numpy()[accepted]
        half_window = .03
        actual_rate = (orientation(times-half_window).inv()*orientation(times+half_window)).as_rotvec()/(2*half_window)
        residual = gyro[accepted]-actual_rate
        q0 = orientation([0.]).as_matrix()[0]
        r0 = np.asarray(meta["R0"])
        angle = np.arccos(np.clip((r0.T @ [0, 0, 1]) @ (q0.T @ [0, 0, 1]), -1., 1.))
        row.update(reference_accepted_rate_mean_norm_radps=float(np.linalg.norm(actual_rate.mean(axis=0))),
            reference_accepted_rate_rms_radps=float(np.sqrt(np.mean(np.sum(actual_rate**2, axis=1)))),
            measured_minus_reference_mean_norm_radps=float(np.linalg.norm(residual.mean(axis=0))),
            reference_tilt_error_at_restart_deg=float(np.rad2deg(angle)),
            reference_usage="Independent Walk1 extrinsics and lag; reference noise limits bias diagnosis")
        for j, axis in enumerate("xyz"):
            row["measured_gyro_mean_"+axis] = float(gyro[accepted, j].mean())
            row["reference_rate_mean_"+axis] = float(actual_rate[:, j].mean())
            row["measured_minus_reference_mean_"+axis] = float(residual[:, j].mean())
        original = load_episode(episode)
        for end in ENDPOINTS:
            qref = orientation([0., end]).as_matrix()
            nominal_end = at_times(original, np.array([end]))[-1][0]
            relative_error = (qref[0].T @ qref[1]).T @ (r0.T @ nominal_end)
            row[f"baseline_relative_attitude_drift_{end:g}s_deg"] = float(np.rad2deg(
                Rotation.from_matrix(relative_error).magnitude()))
    else:
        row["reference_usage"] = "No physical attitude truth; mean gyro cannot separate motion from bias"
    return row


def integrate_override(original, rotation, gyro_bias, accel_bias):
    """Use exactly the existing held-IMU dynamics with explicitly changed initialization."""
    data = copy.deepcopy(original)
    data["gyro_bias"] = np.asarray(gyro_bias)
    data["accel_bias"] = np.asarray(accel_bias)
    data["rotations"][0] = rotation
    data["velocity"][:] = 0.
    data["displacement"][:] = 0.
    data["bias_position"][:] = 0.
    data["bias_velocity"][:] = 0.
    for k, dt in enumerate(np.diff(data["time"])):
        q = data["rotations"][k]
        acceleration = q @ (data["samples"][k, :3]-data["accel_bias"])+GRAVITY
        data["displacement"][k+1] = data["displacement"][k]+dt*data["velocity"][k]+.5*dt**2*acceleration
        data["velocity"][k+1] = data["velocity"][k]+dt*acceleration
        data["bias_position"][k+1] = data["bias_position"][k]+dt*data["bias_velocity"][k]+.5*dt**2*q
        data["bias_velocity"][k+1] = data["bias_velocity"][k]+dt*q
        data["rotations"][k+1] = q @ Rotation.from_rotvec((data["samples"][k, 3:]-data["gyro_bias"])*dt).as_matrix()
    return data


def propagated_record_start(root, episode, meta, delft):
    """Use extra same-recording start history without scoring positions or attitudes."""
    start_episode = root/"streams"/(episode.name.rsplit("_", 1)[0]+"_start")
    start_meta = json.loads((start_episode/"meta.json").read_text(encoding="utf-8"))
    if meta["dataset"] == "delft":
        raw = pd.read_csv(delft/"twr"/meta["source"], skipinitialspace=True)
        grid = raw.timeTick.to_numpy()/1000
        gyro = np.deg2rad(raw[["gyroX", "gyroY", "gyroZ"]].to_numpy())
        begin, end = start_meta["origin_sensor_tick_s"], meta["origin_sensor_tick_s"]
    else:
        raw = pd.read_csv(root/"streams"/meta["source_sequence"]/"imu.csv")
        grid = raw.t.to_numpy()
        gyro = raw[["gx", "gy", "gz"]].to_numpy()
        begin, end = start_meta["source_start_s"], meta["source_start_s"]
    events = np.unique(np.r_[begin, grid[(grid > begin) & (grid < end)], end])
    indices = np.clip(np.searchsorted(grid, events[:-1], side="right")-1, 0, len(grid)-1)
    q = np.asarray(start_meta["R0"])
    bg = np.asarray(start_meta["gyro_bias"])
    for index, dt in zip(indices, np.diff(events)):
        q = q @ Rotation.from_rotvec((gyro[index]-bg)*dt).as_matrix()
    return start_meta, q


def run_episode(root, episode, output, delft):
    """Run fixed, post-hoc variants with unchanged data and exact one-start solver."""
    original = load_episode(episode)
    meta = original["meta"]
    r0 = original["rotations"][0]
    bg, ba = original["gyro_bias"], original["accel_bias"]
    variants = [("zero_gyro_bias_adjacent_tilt", r0, np.zeros(3), ba, "No extra past history")]
    if meta["kind"] == "recovery":
        start_meta, propagated = propagated_record_start(root, episode, meta, delft)
        variants += [("record_start_fixed_gyro_adjacent_tilt", r0, start_meta["gyro_bias"], ba,
            "Additional same-recording initial-prefix gyro calibration"),
            ("record_start_propagated_attitude_and_bias", propagated, start_meta["gyro_bias"],
             start_meta["accel_bias"], "Additional same-recording entire intervening IMU history")]
    if meta["dataset"] == "isas":
        cal = json.loads((root/"scoring/isas_walk1_calibration.json").read_text(encoding="utf-8"))
        variants.append(("Walk1_fixed_gyro_adjacent_tilt", r0, cal["gyro_bias_radps"], ba,
            "Independent Walk1 pose-calibrated gyro bias; training reference reused"))
    reference = pd.read_csv(root/"scoring"/(episode.name+".csv"))
    orientation = reference_orientation(root, episode, meta)
    rows = []
    for label, rotation, gyro_bias, accel_bias, information in variants:
        data = integrate_override(original, rotation, gyro_bias, accel_bias)
        for end in ENDPOINTS:
            for policy, duration in (("rolling_max8", min(end, 8.)), ("growing_from_zero", end)):
                window, events, packet_hash, start = make_window(data, end, duration)
                fit = attempt(window, MODEL)
                row = to_row(episode.name, "native", end, MODEL, fit, events, packet_hash, start, fit["fit_elapsed_s"])
                row.update(strategy=label, dataset=meta["dataset"], kind=meta["kind"], window_policy=policy,
                    additional_information=information, post_hoc_diagnostic=True,
                    gyro_bias_norm_radps=float(np.linalg.norm(gyro_bias)))
                if orientation is not None:
                    qref = orientation([0., end]).as_matrix()
                    relative_true = qref[0].T @ qref[1]
                    relative_nominal = data["rotations"][0].T @ window.end_rotation
                    row["relative_attitude_drift_deg"] = float(np.rad2deg(Rotation.from_matrix(
                        relative_true.T @ relative_nominal).magnitude()))
                    row["tilt_error_at_restart_deg"] = float(np.rad2deg(np.arccos(np.clip(
                        (rotation.T @ [0, 0, 1]) @ (qref[0].T @ [0, 0, 1]), -1., 1.))))
                rows.append(score_row(row, reference))
    pd.DataFrame(rows).to_csv(output/(episode.name+"__diagnostic.csv"), index=False)
    return len(rows)


def physics_reference_audit(root, episodes, output):
    """Measure oracle trajectory expressivity; test pose is never fed to a field fit."""
    calibration = json.loads((root/"scoring/isas_walk1_calibration.json").read_text(encoding="utf-8"))
    rows = []
    for episode in episodes:
        original = load_episode(episode)
        if original["meta"]["dataset"] != "isas":
            continue
        orientation = reference_orientation(root, episode, original["meta"])
        true_initial = orientation([0.]).as_matrix()[0]
        reference = pd.read_csv(root/"scoring"/(episode.name+".csv"))
        trained = integrate_override(original, original["rotations"][0],
            calibration["gyro_bias_radps"], original["accel_bias"])
        oracle = integrate_override(original, true_initial, calibration["gyro_bias_radps"],
            original["accel_bias"])
        for name, data in (("adjacent_prefix", original), ("Walk1_fixed_gyro_adjacent_tilt", trained),
                           ("ORACLE_test_initial_attitude_Walk1_gyro", oracle)):
            initial = data["rotations"][0]
            true_heading = np.arctan2(true_initial[1, 0], true_initial[0, 0])
            nominal_heading = np.arctan2(initial[1, 0], initial[0, 0])
            alignment = Rotation.from_euler("z", true_heading-nominal_heading).as_matrix()
            for end in ENDPOINTS:
                times = np.linspace(0., end, 101)
                displacement, _, sensitivity, _, _ = at_times(data, times)
                nominal = displacement @ alignment.T
                sensitivity = np.einsum("ij,njk->nik", alignment, sensitivity)
                truth = np.column_stack([np.interp(times, reference.t, reference[key]) for key in ("x", "y", "z")])
                affine = np.concatenate((np.tile(np.eye(3), (len(times), 1, 1)),
                    times[:, None, None]*np.eye(3)), axis=2).reshape(-1, 6)
                response = (truth-nominal).reshape(-1)
                design = np.column_stack((affine, -sensitivity.reshape(-1, 3)))
                affine_error = (affine @ np.linalg.lstsq(affine, response, rcond=None)[0]-response).reshape(-1, 3)
                fit = np.linalg.lstsq(design, response, rcond=None)[0]
                error = (design @ fit-response).reshape(-1, 3)
                rows.append(dict(episode=episode.name, strategy=name, endpoint_s=end,
                    oracle_position_and_initial_heading_used=True,
                    initial_full_attitude_from_test_pose=name.startswith("ORACLE"),
                    used_for_main_or_strategy_selection=False,
                    affine_only_shape_rms_m=float(np.sqrt(np.mean(np.sum(affine_error**2, axis=1)))),
                    free_body_bias_shape_rms_m=float(np.sqrt(np.mean(np.sum(error**2, axis=1)))),
                    free_body_bias_xy_rms_m=float(np.sqrt(np.mean(np.sum(error[:, :2]**2, axis=1)))),
                    required_bias_norm_mps2=float(np.linalg.norm(fit[6:9])),
                    required_bias_norm_in_point1_units=float(np.linalg.norm(fit[6:9])/.1)))
    pd.DataFrame(rows).to_csv(output/"oracle_physics_expressivity.csv", index=False)


def main():
    """Keep the audit separate from every frozen main comparison and protocol."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--delft", type=Path, default=Path("E:/datasets/tu-delft-uwb-mhe/data_v1/UWB dataset"))
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--stats-only", action="store_true")
    parser.add_argument("--physics-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    reuse_baselines(args.root, args.results, args.output)
    episodes = [path for path in sorted((args.root/"streams").iterdir()) if (path/"meta.json").exists()
                and (path.name.endswith("_recovery") or path.name in ("isas_walk2_start", "isas_walk3_start"))]
    if args.physics_only:
        physics_reference_audit(args.root, episodes, args.output)
        return
    stats = [prefix_statistics(args.root, episode) for episode in episodes]
    pd.DataFrame(stats).to_csv(args.output/"prefix_statistics.csv", index=False)
    if args.stats_only:
        return
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(run_episode, args.root, episode, args.output, args.delft) for episode in episodes]
        for job in as_completed(jobs):
            print("Diagnostic completed", job.result(), flush=True)
    frame = pd.concat([pd.read_csv(path) for path in sorted(args.output.glob("*__diagnostic.csv"))], ignore_index=True)
    frame.to_csv(args.output/"diagnostic_fits.csv", index=False)
    frame.groupby(["dataset", "kind", "strategy", "window_policy", "arrival_s"]).agg(
        records=("xy_m", "size"), xy_median_m=("xy_m", "median"),
        xy_max_m=("xy_m", "max"), yaw_median_deg=("yaw_error_deg", "median"),
        solver_successes=("solver_success", "sum")).to_csv(args.output/"diagnostic_summary.csv")
    protocol = dict(post_hoc=True, frozen_main_results_unchanged=True, model=MODEL, endpoints=ENDPOINTS,
        no_strategy_selected_or_tuned=True, no_test_pose_used_for_solver_initialization=True,
        same_recording_history_is_additional_information=True,
        no_Delft_attitude_truth=True, ISAS_reference="Independent noisy Walk1 pose calibration",
        source_sha256=digest(Path(__file__)),
        frozen_csv_sha256={name:digest(args.results/name) for name in ("native.csv", "stress.csv", "trajectory.csv")})
    (args.output/"diagnostic_protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print("Diagnostic fits", len(frame), flush=True)


if __name__ == "__main__":
    main()
