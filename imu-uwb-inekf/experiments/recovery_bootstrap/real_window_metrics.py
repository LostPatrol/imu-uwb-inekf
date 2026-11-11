"""Resummarize saved real-window endpoint errors and audit a small replay panel.

Saved errors remain historical results; verification timings are recorded separately.
"""

import argparse
import hashlib
from pathlib import Path
import platform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy

import initializer
from initializer import raw_residual_jacobian, solve_window, yaw_rotation
from run_research import PROJECT, DATASET, INITIAL_IMU_PREFIX_S, SENSOR_DELAY_S, load_real_log, write_json
from window_solver_study import arrays_for_job, load_data, RELATIVE_COST_TOLERANCE

# These keys identify the frozen physical window and its equivalent parameterization.
KEYS = ["sequence", "endpoint_requested_s", "duration_s", "layout", "frame_deg"]
# The original raw-range comparison is retained alongside the candidate extension.
METHODS = ("local", "profile", "profile_two", "profile_all", "uniform_four", "multistart")
# Display names keep the original one-candidate Profile distinct from Profile-2.
LABELS = dict(local="Local", profile="Profile-1", profile_two="Profile-2",
              profile_all="Profile-all", uniform_four="Uniform-4", multistart="Multi12")
# Six-log cached replay uses only this modest panel, never complete tracking.
PANEL = ((5., 2., "all8", 0.), (30., 8., "lower4", 180.))


def summarize_group(part):
    """Compute descriptive endpoint statistics without assuming independent frames."""
    errors = part["position_error_m"].to_numpy()
    physical = part[["sequence", "endpoint_requested_s", "duration_s"]].drop_duplicates()
    return dict(configurations=len(part), physical_time_windows=len(physical),
                median_position_error_m=float(np.median(errors)),
                p95_position_error_m=float(np.quantile(errors, .95)),
                endpoint_rmse_m=float(np.sqrt(np.mean(errors**2))),
                max_position_error_m=float(np.max(errors)),
                objective_misses=int(part["missed_best_found"].sum()),
                objective_miss_percent=float(part["missed_best_found"].mean()*100),
                median_total_nfev=float(part["total_nfev"].median()),
                p95_total_nfev=float(part["total_nfev"].quantile(.95)),
                total_nfev=int(part["total_nfev"].sum()),
                historical_parallel_median_ms=float(part["elapsed_s"].median()*1000),
                historical_parallel_p95_ms=float(part["elapsed_s"].quantile(.95)*1000))


def grouped_summary(frame, groups):
    """Build a flat summary for each requested grouping."""
    rows = []
    for values, part in frame.groupby(groups):
        values = values if isinstance(values, tuple) else (values,)
        rows.append(dict(zip(groups, values), **summarize_group(part)))
    return pd.DataFrame(rows)


def summarize(source, base, output):
    """Reuse all saved endpoint errors and keep historical cost provenance explicit."""
    frame = pd.read_csv(source/"combined_metrics.csv")
    assert len(frame) == 3168 and not frame.duplicated(KEYS+["method"]).any()
    assert np.isfinite(frame["position_error_m"]).all()
    reference = frame.groupby(KEYS)["objective"].transform("min")
    frame["relative_objective_gap"] = (frame["objective"]-reference)/(1+reference)
    frame["missed_best_found"] = (frame["relative_objective_gap"] > RELATIVE_COST_TOLERANCE) | ~frame["finite"]
    frame["endpoint_error_source"] = "saved original CSV; parameters not retained"
    frame.to_csv(output/"endpoint_metrics.csv", index=False)
    write_json(output/"endpoint_summary.json", dict(
        overall=grouped_summary(frame, ["method"]).to_dict("records"),
        per_log=grouped_summary(frame, ["sequence", "method"]).to_dict("records"),
        frame0=grouped_summary(frame[frame.frame_deg == 0], ["method"]).to_dict("records")))
    groups = (("per_log", ["sequence", "method"]),
              ("per_log_layout", ["sequence", "layout", "method"]),
              ("overall", ["method"]), ("layout", ["layout", "method"]))
    for suffix, group in groups:
        grouped_summary(frame, group).to_csv(output/f"{suffix}_summary.csv", index=False)
        grouped_summary(frame[frame.frame_deg == 0], group).to_csv(
            output/f"{suffix}_frame0_summary.csv", index=False)
    original = pd.read_csv(base/"metrics.csv")
    original = original.merge(frame[KEYS+["method", "total_nfev"]], on=KEYS+["method"], validate="one_to_one")
    original["missed_best_found"] = original.relative_objective_gap > RELATIVE_COST_TOLERANCE
    grouped_summary(original, ["sequence", "method"]).to_csv(output/"per_log_original_three_reference.csv", index=False)
    timing = pd.read_csv(source/"sequential_timing.csv")
    timing_summary = timing.groupby("method")["elapsed_ms"].agg(
        observations="size", median="median", p95=lambda x:x.quantile(.95), maximum="max")
    timing_summary.to_csv(output/"historical_sequential_timing_summary.csv")
    make_table(frame, output)
    make_figure(frame, output)
    return frame


def make_table(frame, output):
    """Write a compact LaTeX table of saved endpoint medians and tails per log."""
    summary = grouped_summary(frame, ["sequence", "method"]).set_index(["sequence", "method"])
    lines = [r"% Endpoint median / p95 in metres; 88 correlated configurations per log/method.",
             r"% Each log has 11 physical time windows x 2 anchor layouts x 4 yaw frames.",
             r"\begin{tabular}{lccc}", r"\hline", r"Log & Local & Profile-1 & Multi12 \\", r"\hline"]
    for name in sorted(frame.sequence.unique()):
        cells = []
        for method in ("local", "profile", "multistart"):
            row = summary.loc[(name, method)]
            cells.append(f"{row.median_position_error_m:.3f} / {row.p95_position_error_m:.3f}")
        lines.append(name.removesuffix(".csv").split("_twr_")[-1].capitalize()+" & "+" & ".join(cells)+r" \\")
    lines += [r"\hline", r"\end{tabular}"]
    (output/"endpoint_table.tex").write_text("\n".join(lines)+"\n", encoding="utf-8")


def make_figure(frame, output):
    """Plot endpoint median and p95 as descriptive values, never confidence bounds."""
    summary = grouped_summary(frame, ["sequence", "layout", "method"]).set_index(["sequence", "layout", "method"])
    names = sorted(frame.sequence.unique())
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), sharey=True)
    for ax, layout in zip(axes, ("all8", "lower4")):
        for offset, method, color in zip((-.18, 0, .18), ("local", "profile", "multistart"), ("#d95f02", "#1b9e77", "#7570b3")):
            rows = summary.loc[[(name, layout, method) for name in names]]
            x = np.arange(len(names))+offset
            ax.scatter(x, rows.median_position_error_m, s=24, color=color, label=LABELS[method])
            ax.scatter(x, rows.p95_position_error_m, s=38, marker="_", color=color)
            ax.vlines(x, rows.median_position_error_m, rows.p95_position_error_m, color=color, alpha=.6)
        ax.set_xticks(np.arange(len(names)), [n.split("_twr_")[-1].removesuffix(".csv") for n in names], rotation=30, ha="right")
        ax.set_title("All 8 anchors" if layout == "all8" else "Lower 4 anchors")
        ax.set_yscale("log")
        ax.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("Endpoint position error (m), median to p95")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output/"endpoint_position_by_log.pdf", bbox_inches="tight")
    fig.savefig(output/"endpoint_position_by_log.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def raw_truth_at_endpoint(name, endpoint, cache):
    """Independently interpolate raw OptiTrack samples at the corrected endpoint."""
    log = load_real_log(DATASET, name, sensor_delay_s=SENSOR_DELAY_S)
    prefix = float(log.time[log.time >= INITIAL_IMU_PREFIX_S][0])
    raw = pd.read_csv(DATASET/"twr"/name, skipinitialspace=True)
    xyz = raw[["otX", "otY", "otZ"]].to_numpy(float)
    imu_ranges = raw[["gyroX", "gyroY", "gyroZ", "accX", "accY", "accZ"]+
                     [f"twr{k}" for k in range(8)]].to_numpy(float)
    valid = np.isfinite(xyz).all(axis=1) & (np.linalg.norm(xyz, axis=1) > 1e-9)
    start = np.flatnonzero(valid & np.isfinite(imu_ranges).all(axis=1))[0]
    raw_seconds = raw.timeTick.to_numpy(float)/1000
    acquisition = raw_seconds[start]+prefix+endpoint
    truth = np.array([np.interp(acquisition, raw_seconds[valid], xyz[valid, k]) for k in range(3)])
    data = load_data(str(cache), name)
    end = np.searchsorted(data["time"], endpoint, side="right")-1
    np.testing.assert_allclose(truth, data["truth_position"][end], atol=2e-10, rtol=0)
    return truth, prefix, acquisition


def verify(cache, frame, output, log_limit=None, coordinates_only=False):
    """Replay twelve windows and audit frame equivalence and raw-truth matching."""
    rows, checks = [], []
    names = sorted(frame.sequence.unique())[:log_limit]
    for number, name in enumerate(names):
        for endpoint, duration, layout, angle in PANEL:
            job = (str(cache), name, endpoint, duration, layout, angle, number)
            values, truth, actual = arrays_for_job(job)
            raw_truth, prefix, acquisition = raw_truth_at_endpoint(name, actual, cache)
            tau, d, anchors, ranges, sigma = values
            data = load_data(str(cache), name)
            end = np.searchsorted(data["time"], endpoint, side="right")-1
            first_capture = int(np.searchsorted(data["time"], 2., side="left"))
            first_twr = int(np.flatnonzero(np.any(data["changed"] & (data["ranges"] > 0), axis=1))[0])
            end_d = data["displacement"][end]-data["displacement"][end]
            np.testing.assert_array_equal(end_d, np.zeros(3))
            x = np.r_[truth+[.1, -.2, .3], [.2, -.1, .05], .71-np.deg2rad(angle)]
            unrotated = d@yaw_rotation(-np.deg2rad(angle)).T
            original_x = x.copy()
            original_x[6] += np.deg2rad(angle)
            residual = raw_residual_jacobian(x, *values)[0]
            original_residual = raw_residual_jacobian(original_x, tau, unrotated, anchors, ranges, sigma)[0]
            np.testing.assert_allclose(residual, original_residual, atol=2e-12, rtol=0)
            endpoint_residual, endpoint_jacobian = raw_residual_jacobian(
                x, np.zeros(1), end_d[None, :], anchors[:1], ranges[:1], sigma[:1])
            direct_residual = (np.linalg.norm(x[:3]-anchors[0])-ranges[0])/sigma[0]
            np.testing.assert_allclose(endpoint_residual[0], direct_residual, atol=1e-13, rtol=0)
            np.testing.assert_array_equal(endpoint_jacobian[:, 3:], np.zeros((1, 4)))
            checks.append(dict(sequence=name, endpoint_actual_s=actual, duration_s=duration, layout=layout,
                               frame_deg=angle, prefix_end_loader_s=prefix,
                               first_usable_twr_cache_s=float(data["time"][first_twr]),
                               first_2s_candidate_cache_s=float(data["time"][first_capture]),
                               first_2s_candidate_loader_s=float(prefix+data["time"][first_capture]),
                               acquisition_timeTick_s=acquisition, raw_truth_error_m=float(np.linalg.norm(raw_truth-truth)),
                               frame_residual_max_abs=float(np.max(np.abs(residual-original_residual))),
                               endpoint_range_residual_abs=float(abs(endpoint_residual[0]-direct_residual)),
                               endpoint_displacement_norm_m=float(np.linalg.norm(end_d))))
            if coordinates_only:
                continue
            for method in ("local", "profile", "multistart"):
                counts = []
                original_solver = initializer.least_squares
                def counted(*args, **kwargs):
                    """Record unchanged CV and refinement function-evaluation counts."""
                    fit = original_solver(*args, **kwargs)
                    counts.append(fit.nfev)
                    return fit
                try:
                    initializer.least_squares = counted
                    fit = solve_window(*values, method=method)
                finally:
                    initializer.least_squares = original_solver
                saved = frame[(frame.sequence == name) & (frame.endpoint_requested_s == endpoint) &
                              (frame.duration_s == duration) & (frame.layout == layout) &
                              (frame.frame_deg == angle) & (frame.method == method)].iloc[0]
                error = float(np.linalg.norm(fit.parameters[:3]-truth))
                rows.append(dict(sequence=name, endpoint_requested_s=endpoint, duration_s=duration,
                                 layout=layout, frame_deg=angle, method=method,
                                 endpoint_actual_s=actual, endpoint_x_m=fit.parameters[0],
                                 endpoint_y_m=fit.parameters[1], endpoint_z_m=fit.parameters[2],
                                 truth_x_m=truth[0], truth_y_m=truth[1], truth_z_m=truth[2],
                                 position_error_m=error, saved_position_error_m=saved.position_error_m,
                                 absolute_error_difference_m=abs(error-saved.position_error_m),
                                 objective=fit.robust_objective, saved_objective=saved.objective,
                                 relative_objective_difference=abs(fit.robust_objective-saved.objective)/(1+saved.objective),
                                 total_nfev=sum(counts), cv_nfev=sum(counts[:2]), raw_nfev=fit.evaluations,
                                 replay_elapsed_ms=1000*fit.elapsed_s))
            pd.DataFrame(rows).to_csv(output/"verification_replays.csv", index=False)
            print(f"Verified {name} end={endpoint:g}s duration={duration:g}s {layout}", flush=True)
    pd.DataFrame(checks).to_csv(output/"endpoint_coordinate_checks.csv", index=False)
    write_json(output/"coordinate_verification.json", dict(
        windows=len(checks), max_endpoint_residual_difference=max(row["endpoint_range_residual_abs"] for row in checks),
        max_raw_truth_difference_m=max(row["raw_truth_error_m"] for row in checks),
        max_frame_residual_difference=max(row["frame_residual_max_abs"] for row in checks)))
    if coordinates_only:
        return
    replay = pd.DataFrame(rows)
    result = dict(windows=len(checks), fits=len(replay),
                  max_position_error_difference_m=float(replay.absolute_error_difference_m.max()),
                  max_relative_objective_difference=float(replay.relative_objective_difference.max()),
                  max_raw_truth_difference_m=max(row["raw_truth_error_m"] for row in checks),
                  max_frame_residual_difference=max(row["frame_residual_max_abs"] for row in checks),
                  replay_elapsed_s=float(replay.replay_elapsed_ms.sum()/1000),
                  current_python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                  timing_scope="Verification panel only, current host; not original full-panel latency")
    write_json(output/"verification.json", result)
    return result


def main():
    """Write reproducible descriptive summaries and optionally verify cached windows."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=PROJECT/"results/recovery_window_candidates_20261107")
    parser.add_argument("--base", type=Path, default=PROJECT/"results/recovery_window_solver_20261107")
    parser.add_argument("--output", type=Path, default=PROJECT/"results/recovery_real_windows_20261111")
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--verify-logs", type=int, help="Limit verification logs for a speed pilot")
    parser.add_argument("--coordinates-only", action="store_true", help="Check coordinate and truth identities without fits")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    frame = summarize(args.source, args.base, args.output)
    protocol = dict(configurations_per_method=528, method_count=6, physical_time_windows=66,
                    physical_time_windows_per_log=11, anchor_layout_windows=132,
                    configurations_per_log_method=88, frame0_configurations_per_log_method=22,
                    layouts=["all8", "lower4"], yaw_frames_deg=[0, 90, 180, 270],
                    endpoints_s=[5, 10, 20, 30], durations_s=[2, 4, 8],
                    duration_restriction="duration <= endpoint, giving 2+3+3+3=11 windows/log",
                    statistical_unit="Descriptive configurations; overlapping windows, layouts and yaw frames are correlated; no independent-sample inference",
                    endpoint_model="p(t)=p_end+v_end*tau+Rz(yaw)*d(t); tau=0 and d=0 at endpoint; p_end is directly in the anchor/OptiTrack world frame",
                    translation_model_order=1, nuisance_basis="1,tau,tau^2,d_x,d_y,tau*d_x,tau*d_y",
                    translation_seeds="Two CV solves: squared-range linear seed and anchor-centroid/zero-velocity fallback; choose robust best, then share one CV translation across all yaw candidates",
                    yaw_seed_counts={"local":1, "profile":1, "profile_two":2, "profile_all":"all distinct stationary plus 0/pi candidates", "uniform_four":4, "multistart":12},
                    yaw_frame_transform="d_frame=Rz(frame)*d; yaw_original=yaw_frame+frame; p_end, v_end, anchors and truth unchanged",
                    loss="Huber, normalized residual, f_scale=2; scipy least_squares analytic Jacobian, max_nfev=120 per solve; same original solvers",
                    error_definition="Euclidean endpoint error; RMSE=sqrt(mean(endpoint_error^2)) across the indicated configurations, not full-trajectory RMSE",
                    truth_matching="Last arrived cached sample <= requested endpoint; OptiTrack XYZ linearly interpolated at corrected sensor acquisition time; not interpolated to requested endpoint afterward",
                    time_origin="Loader t=0 is first valid OptiTrack time. IMU and TWR share a 0.20 s acquisition-time correction. Alignment/bias uses loader t<=1.5 s; cache t=0 is first retained sample at loader t>=1.5 s (square:1.51 s, others:1.50 s). Window endpoints are relative to this cache origin, not raw recording start.",
                    startup_note="2/4/8 s capture windows add to 1.5 s prefix; full-replay shared scoring begins cache t=8.1 s. This window study tests endpoints 5/10/20/30 s, not online startup latency.",
                    objective_reference="Joint best found over six policies per identical configuration, no global certificate; original-three misses also retained separately",
                    objective_relative_tolerance=RELATIVE_COST_TOLERANCE,
                    cost_nfev="CV nfev from both translation solves plus all raw yaw-refinement nfev; SVD/polynomial work included only in wall time",
                    historical_timing="Saved November 7 results on source-study host (documented Intel Core Ultra 7 265KF); parallel full-panel times and separate 12-window x 10-repeat interleaved sequential times are historical, not new local timings or WCET",
                    ground_truth_yaw_available=False, yaw_accuracy_claim=False, independent_blind_test=False,
                    source_files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                  [args.source/"combined_metrics.csv", args.source/"sequential_timing.csv", args.base/"metrics.csv", Path(__file__)]})
    write_json(args.output/"protocol.json", protocol)
    if args.cache:
        verify(args.cache, frame, args.output, args.verify_logs, args.coordinates_only)
    print(grouped_summary(frame, ["method"]).to_string(index=False))


if __name__ == "__main__":
    main()
