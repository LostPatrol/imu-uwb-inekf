#!/usr/bin/env python3
"""Command-line entry point and interactive InEKF comparison plots."""

import argparse
import gc
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pipeline import compute_metrics, load_dataset, run_dataset


VARIANTS = (
    "legacy",
    "aligned_legacy",
    "pre_coordinate_fix",
    "pre_optimization",
    "proposed",
    "task03_compiled",
    "proposed_euler",
    "proposed_no_bias_init",
    "proposed_no_stationary",
    "verified",
)


def save_plots(dataset, result, output_dir):
    """Save the compact two-figure report for a single estimator."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics = compute_metrics(dataset, result)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(dataset.p_gt[:, 0], dataset.p_gt[:, 1], "k--", lw=1.2, label="Vicon")
    axes[0].plot(result.states[:, 6], result.states[:, 7], lw=1.0, label=result.variant)
    axes[0].scatter(dataset.anchors[:, 0], dataset.anchors[:, 1], marker="^", s=28, label="anchors")
    axes[0].set(xlabel="x [m]", ylabel="y [m]", title="Horizontal trajectory")
    axes[0].axis("equal")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(
        diagnostics["times"],
        np.linalg.norm(diagnostics["position_error"], axis=1),
        lw=0.9,
    )
    axes[1].set(xlabel="time [s]", ylabel="position error [m]", title="Position error norm")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "trajectory_and_position_error.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=False)
    axes[0].plot(diagnostics["times"], np.degrees(diagnostics["yaw_error"]), lw=0.9)
    axes[0].axhline(5.0, color="tab:red", ls="--", lw=0.8)
    axes[0].axhline(-5.0, color="tab:red", ls="--", lw=0.8)
    axes[0].set(ylabel="yaw error [deg]", title="Geometric attitude evaluation")
    axes[0].grid(alpha=0.25)
    axes[1].plot(result.nis_times, result.nis, ".", ms=1.5)
    axes[1].axhline(25.0, color="tab:red", ls="--", lw=0.8, label="5-sigma gate")
    axes[1].set(xlabel="time [s]", ylabel="NIS", title="UWB innovation consistency")
    axes[1].set_ylim(0.0, min(50.0, max(30.0, np.nanpercentile(result.nis, 99))))
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "yaw_and_nis.png", dpi=180)
    plt.close(fig)


def _decimation_indices(length, max_points):
    if length <= max_points:
        return np.arange(length)
    stride = int(math.ceil(length / max_points))
    indices = np.arange(0, length, stride)
    if indices[-1] != length - 1:
        indices = np.append(indices, length - 1)
    return indices


def _comparison_series(dataset, result, max_points):
    """Copy a bounded plot series so the full covariance history can be freed."""
    diagnostics = compute_metrics(dataset, result)
    valid = (result.times >= dataset.t_gt[0]) & (result.times <= dataset.t_gt[-1])
    source_indices = np.flatnonzero(valid)
    sample = _decimation_indices(len(source_indices), max_points)
    source = source_indices[sample]

    biases = result.biases[source]
    if result.variant in {"legacy", "aligned_legacy"}:
        # Legacy nominal propagation interprets its six-vector as [b_a, b_g].
        accelerometer_bias = biases[:, :3]
        gyroscope_bias = biases[:, 3:6]
    else:
        gyroscope_bias = biases[:, :3]
        accelerometer_bias = biases[:, 3:6]

    nis_sample = _decimation_indices(len(result.nis), max_points)
    cumulative_rejections = np.cumsum(result.rejected, dtype=int)
    return {
        "variant": result.variant,
        "metrics": dict(result.metrics),
        "times": diagnostics["times"][sample].copy(),
        "states": result.states[source].copy(),
        "position_gt": diagnostics["position_gt"][sample].copy(),
        "velocity_gt": diagnostics["velocity_gt"][sample].copy(),
        "position_error": diagnostics["position_error"][sample].copy(),
        "velocity_error": diagnostics["velocity_error"][sample].copy(),
        "rotation_error_rotvec": diagnostics["rotation_error_rotvec"][sample].copy(),
        "geodesic_error": diagnostics["geodesic_error"][sample].copy(),
        "yaw_error": diagnostics["yaw_error"][sample].copy(),
        "gyroscope_bias": gyroscope_bias.copy(),
        "accelerometer_bias": accelerometer_bias.copy(),
        "nis_times": result.nis_times[nis_sample].copy(),
        "nis": result.nis[nis_sample].copy(),
        "cumulative_rejections": cumulative_rejections[nis_sample].copy(),
    }


def _set_window_title(fig, title):
    manager = getattr(fig.canvas, "manager", None)
    if manager is not None and hasattr(manager, "set_window_title"):
        manager.set_window_title(title)


def _finish_figure(fig, title, filename, output_dir, figures):
    _set_window_title(fig, title)
    fig.tight_layout()
    if output_dir is not None:
        fig.savefig(output_dir / filename, dpi=160)
    figures.append(fig)


def _plot_component_estimates(
    series,
    gt_field,
    state_slice,
    labels,
    units,
    title,
    filename,
    output_dir,
    figures,
):
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 8.2), sharex=True)
    first = next(iter(series.values()))
    for axis_index, axis in enumerate(axes):
        axis.plot(
            first["times"],
            first[gt_field][:, axis_index],
            "k--",
            lw=1.2,
            label="Vicon",
        )
        for item in series.values():
            axis.plot(
                item["times"],
                item["states"][:, state_slice.start + axis_index],
                lw=0.85,
                label=item["variant"],
            )
        axis.set_ylabel(f"{labels[axis_index]} [{units}]")
        axis.grid(alpha=0.25)
    axes[0].set_title(title)
    axes[0].legend(ncol=3)
    axes[-1].set_xlabel("time [s]")
    _finish_figure(fig, title, filename, output_dir, figures)


def _plot_component_errors(
    series,
    field,
    labels,
    units,
    title,
    filename,
    output_dir,
    figures,
):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)
    for axis_index, axis in enumerate(axes.flat[:3]):
        for item in series.values():
            axis.plot(
                item["times"],
                item[field][:, axis_index],
                lw=0.8,
                label=item["variant"],
            )
        axis.axhline(0.0, color="0.45", lw=0.6)
        axis.set(ylabel=f"{labels[axis_index]} error [{units}]")
        axis.grid(alpha=0.25)
    norm_axis = axes.flat[3]
    for item in series.values():
        norm_axis.plot(
            item["times"],
            np.linalg.norm(item[field], axis=1),
            lw=0.85,
            label=item["variant"],
        )
    norm_axis.set(ylabel=f"error norm [{units}]")
    norm_axis.grid(alpha=0.25)
    axes.flat[0].set_title(title)
    axes.flat[0].legend()
    for axis in axes[-1, :]:
        axis.set_xlabel("time [s]")
    _finish_figure(fig, title, filename, output_dir, figures)


def create_comparison_plots(dataset, series, output_dir=None):
    """Create the complete legacy/proposed comparison figure set."""
    output_dir = Path(output_dir) if output_dir else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
    figures = []

    # 1. Horizontal and three-dimensional trajectories.
    fig = plt.figure(figsize=(12, 5.3))
    axis_xy = fig.add_subplot(1, 2, 1)
    axis_3d = fig.add_subplot(1, 2, 2, projection="3d")
    gt_sample = _decimation_indices(len(dataset.t_gt), 6000)
    gt_position = dataset.p_gt[gt_sample]
    axis_xy.plot(gt_position[:, 0], gt_position[:, 1], "k--", lw=1.2, label="Vicon")
    axis_3d.plot(
        gt_position[:, 0],
        gt_position[:, 1],
        gt_position[:, 2],
        "k--",
        lw=1.2,
        label="Vicon",
    )
    for item in series.values():
        position = item["states"][:, 6:9]
        axis_xy.plot(position[:, 0], position[:, 1], lw=0.9, label=item["variant"])
        axis_3d.plot(position[:, 0], position[:, 1], position[:, 2], lw=0.85, label=item["variant"])
    motion_points = np.vstack(
        [gt_position] + [item["states"][:, 6:9] for item in series.values()]
    )
    motion_min = np.min(motion_points, axis=0)
    motion_max = np.max(motion_points, axis=0)
    motion_span = np.maximum(motion_max - motion_min, 0.1)
    motion_padding = 0.08 * motion_span
    axis_xy.set_xlim(motion_min[0] - motion_padding[0], motion_max[0] + motion_padding[0])
    axis_xy.set_ylim(motion_min[1] - motion_padding[1], motion_max[1] + motion_padding[1])
    axis_3d.set_xlim(motion_min[0] - motion_padding[0], motion_max[0] + motion_padding[0])
    axis_3d.set_ylim(motion_min[1] - motion_padding[1], motion_max[1] + motion_padding[1])
    axis_3d.set_zlim(motion_min[2] - motion_padding[2], motion_max[2] + motion_padding[2])
    lower_bound = motion_min - motion_padding
    upper_bound = motion_max + motion_padding
    visible_xy = np.all(
        (dataset.anchors[:, :2] >= lower_bound[:2])
        & (dataset.anchors[:, :2] <= upper_bound[:2]),
        axis=1,
    )
    visible_xyz = np.all(
        (dataset.anchors >= lower_bound) & (dataset.anchors <= upper_bound),
        axis=1,
    )
    if np.any(visible_xy):
        axis_xy.scatter(
            dataset.anchors[visible_xy, 0],
            dataset.anchors[visible_xy, 1],
            marker="^",
            s=28,
            label="anchors in view",
        )
    if np.any(visible_xyz):
        axis_3d.scatter(
            dataset.anchors[visible_xyz, 0],
            dataset.anchors[visible_xyz, 1],
            dataset.anchors[visible_xyz, 2],
            marker="^",
            s=24,
            label="anchors in view",
        )
    axis_xy.set(xlabel="x [m]", ylabel="y [m]", title="Horizontal trajectory")
    axis_xy.set_aspect("equal", adjustable="box")
    axis_xy.grid(alpha=0.25)
    axis_xy.legend()
    axis_3d.set(xlabel="x [m]", ylabel="y [m]", zlabel="z [m]", title="3D trajectory")
    axis_3d.legend()
    _finish_figure(
        fig,
        "01 - Trajectory comparison",
        "interactive_01_trajectory.png",
        output_dir,
        figures,
    )

    # 2-5. State estimates and component errors.
    _plot_component_estimates(
        series,
        "position_gt",
        slice(6, 9),
        ("x", "y", "z"),
        "m",
        "Position estimates",
        "interactive_02_position.png",
        output_dir,
        figures,
    )
    _plot_component_estimates(
        series,
        "velocity_gt",
        slice(3, 6),
        ("vx", "vy", "vz"),
        "m/s",
        "Velocity estimates",
        "interactive_03_velocity.png",
        output_dir,
        figures,
    )
    _plot_component_errors(
        series,
        "position_error",
        ("x", "y", "z"),
        "m",
        "Position errors",
        "interactive_04_position_error.png",
        output_dir,
        figures,
    )
    _plot_component_errors(
        series,
        "velocity_error",
        ("vx", "vy", "vz"),
        "m/s",
        "Velocity errors",
        "interactive_05_velocity_error.png",
        output_dir,
        figures,
    )

    # 6. Geometrically valid attitude diagnostics.
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)
    attitude_labels = ("rotation-vector x", "rotation-vector y", "rotation-vector z")
    for axis_index, axis in enumerate(axes.flat[:3]):
        for item in series.values():
            axis.plot(
                item["times"],
                np.degrees(item["rotation_error_rotvec"][:, axis_index]),
                lw=0.8,
                label=item["variant"],
            )
        axis.axhline(0.0, color="0.45", lw=0.6)
        axis.set_ylabel(f"{attitude_labels[axis_index]} error [deg]")
        axis.grid(alpha=0.25)
    summary_axis = axes.flat[3]
    for item in series.values():
        summary_axis.plot(
            item["times"],
            np.degrees(item["geodesic_error"]),
            lw=0.9,
            label=f"{item['variant']} geodesic",
        )
        summary_axis.plot(
            item["times"],
            np.abs(np.degrees(item["yaw_error"])),
            lw=0.75,
            ls="--",
            label=f"{item['variant']} |yaw|",
        )
    summary_axis.axhline(5.0, color="tab:red", lw=0.7, ls=":", label="5 deg")
    summary_axis.set_ylabel("attitude error [deg]")
    summary_axis.grid(alpha=0.25)
    axes.flat[0].set_title("SO(3) attitude errors")
    axes.flat[0].legend()
    summary_axis.legend(fontsize=8, ncol=2)
    for axis in axes[-1, :]:
        axis.set_xlabel("time [s]")
    _finish_figure(
        fig,
        "06 - Attitude and yaw errors",
        "interactive_06_attitude_error.png",
        output_dir,
        figures,
    )

    # 7. Canonically ordered nominal IMU bias estimates.
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.2), sharex=True)
    axis_names = ("x", "y", "z")
    for axis_index in range(3):
        for item in series.values():
            axes[0, axis_index].plot(
                item["times"],
                item["gyroscope_bias"][:, axis_index],
                lw=0.8,
                label=item["variant"],
            )
            axes[1, axis_index].plot(
                item["times"],
                item["accelerometer_bias"][:, axis_index],
                lw=0.8,
                label=item["variant"],
            )
        axes[0, axis_index].set_title(f"gyro bias {axis_names[axis_index]}")
        axes[0, axis_index].set_ylabel("rad/s")
        axes[1, axis_index].set_title(f"accel bias {axis_names[axis_index]}")
        axes[1, axis_index].set_ylabel("m/s^2")
        axes[1, axis_index].set_xlabel("time [s]")
        axes[0, axis_index].grid(alpha=0.25)
        axes[1, axis_index].grid(alpha=0.25)
    axes[0, 0].legend()
    _finish_figure(
        fig,
        "07 - IMU bias estimates",
        "interactive_07_biases.png",
        output_dir,
        figures,
    )

    # 8. UWB innovation consistency and rejection history.
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.2), sharex=True)
    for item in series.values():
        axes[0].semilogy(
            item["nis_times"],
            np.maximum(item["nis"], 1e-6),
            ".",
            ms=2.0,
            alpha=0.65,
            label=item["variant"],
        )
        axes[1].step(
            item["nis_times"],
            item["cumulative_rejections"],
            where="post",
            lw=1.0,
            label=item["variant"],
        )
    axes[0].axhline(25.0, color="tab:red", ls="--", lw=0.8, label="gate = 25")
    axes[0].set(ylabel="NIS (log scale)", title="UWB innovation consistency")
    axes[1].set(xlabel="time [s]", ylabel="cumulative rejects")
    maximum_rejections = max(
        (int(item["cumulative_rejections"][-1]) if len(item["cumulative_rejections"]) else 0)
        for item in series.values()
    )
    axes[1].set_ylim(-0.05, max(1.0, maximum_rejections * 1.05))
    axes[0].grid(alpha=0.25, which="both")
    axes[1].grid(alpha=0.25)
    axes[0].legend()
    axes[1].legend()
    _finish_figure(
        fig,
        "08 - UWB NIS and rejection",
        "interactive_08_nis.png",
        output_dir,
        figures,
    )

    # 9. Direct scalar comparison of every reported metric.
    metric_fields = (
        ("position_rmse_m", "position RMSE", "m"),
        ("velocity_rmse_mps", "velocity RMSE", "m/s"),
        ("orientation_geodesic_rmse_rad", "attitude RMSE", "rad"),
        ("yaw_rmse_rad", "yaw RMSE", "rad"),
        ("runtime_seconds", "runtime", "s"),
        ("rejection_rate", "UWB rejection rate", ""),
    )
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.2))
    variants = list(series)
    colors = ("tab:blue", "tab:orange")
    for axis, (field, label, unit) in zip(axes.flat, metric_fields):
        values = [series[variant]["metrics"][field] for variant in variants]
        bars = axis.bar(variants, values, color=colors[: len(variants)])
        axis.set_title(label)
        axis.set_ylabel(unit)
        axis.grid(axis="y", alpha=0.25)
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.6g}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        if max(values) == 0.0:
            axis.set_ylim(0.0, 1.0)
    _finish_figure(
        fig,
        "09 - Metric summary",
        "interactive_09_metrics.png",
        output_dir,
        figures,
    )
    return figures


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", nargs=2, required=True, metavar=("ANCHORS", "CSV"))
    parser.add_argument("--variant", choices=VARIANTS, default="proposed")
    parser.add_argument("--yaw-deg", type=float, default=0.0)
    parser.add_argument("--max-duration", type=float)
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--show-comparison",
        action="store_true",
        help=(
            "run legacy and --variant on the same data, then open the complete "
            "interactive comparison figure set"
        ),
    )
    parser.add_argument(
        "--plot-max-points",
        type=int,
        default=6000,
        help="maximum plotted points per curve in --show-comparison mode",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.plot_max_points < 100:
        raise ValueError("--plot-max-points must be at least 100")
    if args.show_comparison and args.variant == "legacy":
        raise ValueError("--show-comparison requires a non-legacy --variant")

    dataset = load_dataset(args.i[0], args.i[1], max_duration=args.max_duration)
    if not args.show_comparison:
        result = run_dataset(dataset, variant=args.variant, yaw_error_degrees=args.yaw_deg)
        if args.output_dir:
            save_plots(dataset, result, args.output_dir)
        print(json.dumps(result.metrics, ensure_ascii=False, indent=2, allow_nan=True))
        return

    # Copy bounded plot series and release each full covariance history before
    # running the next estimator. This keeps full-dataset comparison practical.
    print("Running comparison baseline: legacy", flush=True)
    baseline = run_dataset(dataset, variant="legacy", yaw_error_degrees=args.yaw_deg)
    series = {"legacy": _comparison_series(dataset, baseline, args.plot_max_points)}
    metrics = {"legacy": dict(baseline.metrics)}
    del baseline
    gc.collect()

    print(f"Running comparison target: {args.variant}", flush=True)
    improved = run_dataset(dataset, variant=args.variant, yaw_error_degrees=args.yaw_deg)
    series[args.variant] = _comparison_series(dataset, improved, args.plot_max_points)
    metrics[args.variant] = dict(improved.metrics)
    del improved
    gc.collect()

    print("Preparing interactive comparison windows...", flush=True)
    create_comparison_plots(dataset, series, output_dir=args.output_dir)
    print(json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=True))
    if args.output_dir:
        print(f"Comparison figures saved to: {Path(args.output_dir).resolve()}")
    print("Close all plot windows to finish the command.")
    plt.show()


if __name__ == "__main__":
    main()
