"""Matched before/after evaluation for invariant covariance/gating repair."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ESTIMATION_DIR = Path(__file__).resolve().parents[1] / "scripts" / "estimation"
sys.path.insert(0, str(ESTIMATION_DIR))

from pipeline import compute_metrics, load_dataset, run_dataset


DATASETS = (
    ("c3m1", 3, "const3/const3-trial7-tdoa2-manual1.csv"),
    ("c3m2", 3, "const3/const3-trial7-tdoa2-manual2.csv"),
    ("c3m3", 3, "const3/const3-trial7-tdoa3-manual3.csv"),
    ("c3m4", 3, "const3/const3-trial7-tdoa3-manual4.csv"),
    ("c4m1", 4, "const4/const4-trial7-tdoa2-manual1.csv"),
    ("c4m2", 4, "const4/const4-trial7-tdoa2-manual2.csv"),
)
VARIANTS = ("pre_coordinate_fix", "proposed")


def _finite(value):
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _run_metric_job(job):
    root, duration, tag, constellation, relative_csv, yaw, variant = job
    root = Path(root)
    dataset = load_dataset(
        root / f"survey-results/anchor_const{constellation}.npz",
        root / "csv-data" / relative_csv,
        max_duration=duration,
    )
    result = run_dataset(dataset, variant=variant, yaw_error_degrees=yaw)
    return {
        "dataset": tag,
        "constellation": constellation,
        "yaw_deg": yaw,
        "variant": variant,
        **result.metrics,
    }


def _run_shift_case(root, duration, tag, constellation, relative_csv, mode, variant):
    dataset = load_dataset(
        root / f"survey-results/anchor_const{constellation}.npz",
        root / "csv-data" / relative_csv,
        max_duration=duration,
    )
    if mode == "initial_origin":
        shift = dataset.p_gt[0].copy()
        dataset.anchors = dataset.anchors - shift
        dataset.p_gt = dataset.p_gt - shift
    elif mode == "const3_like_origin":
        target = np.array([1.489, 0.018, 0.023])
        shift = target - dataset.p_gt[0]
        dataset.anchors = dataset.anchors + shift
        dataset.p_gt = dataset.p_gt + shift
    result = run_dataset(dataset, variant=variant, yaw_error_degrees=180.0)
    return {
        "dataset": tag,
        "origin": mode,
        "variant": variant,
        **result.metrics,
    }


def _metric_matrix(frame, dataset_tags, yaws, variant, field):
    subset = frame[frame["variant"] == variant]
    return np.array(
        [
            [
                float(
                    subset[
                        (subset["dataset"] == dataset)
                        & (subset["yaw_deg"] == yaw)
                    ].iloc[0][field]
                )
                for yaw in yaws
            ]
            for dataset in dataset_tags
        ]
    )


def plot_grid(frame, yaws, output):
    datasets = [item[0] for item in DATASETS]
    pre_position = _metric_matrix(frame, datasets, yaws, VARIANTS[0], "position_rmse_m")
    fixed_position = _metric_matrix(frame, datasets, yaws, VARIANTS[1], "position_rmse_m")
    pre_rejection = 100.0 * _metric_matrix(frame, datasets, yaws, VARIANTS[0], "rejection_rate")
    fixed_rejection = 100.0 * _metric_matrix(frame, datasets, yaws, VARIANTS[1], "rejection_rate")

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.0), constrained_layout=True)
    position_norm = colors.LogNorm(
        vmin=max(0.05, min(pre_position.min(), fixed_position.min())),
        vmax=max(pre_position.max(), fixed_position.max()),
    )
    panels = (
        (pre_position, "Before: position RMSE [m]", position_norm, "magma"),
        (fixed_position, "Fixed: position RMSE [m]", position_norm, "magma"),
        (pre_rejection, "Before: UWB rejection [%]", colors.Normalize(0, 100), "viridis"),
        (fixed_rejection, "Fixed: UWB rejection [%]", colors.Normalize(0, 100), "viridis"),
    )
    for axis, (values, title, norm, cmap) in zip(axes.flat, panels):
        image = axis.imshow(values, aspect="auto", norm=norm, cmap=cmap)
        axis.set_xticks(np.arange(len(yaws)), [f"{yaw:g}" for yaw in yaws])
        axis.set_yticks(np.arange(len(datasets)), datasets)
        axis.set(xlabel="initial yaw error [deg]", title=title)
        for row in range(values.shape[0]):
            for column in range(values.shape[1]):
                value = values[row, column]
                label = f"{value:.2g}" if value >= 1.0 else f"{value:.3f}"
                axis.text(column, row, label, ha="center", va="center", color="white", fontsize=8)
        fig.colorbar(image, ax=axis, shrink=0.82)
    fig.suptitle("Coordinate-consistent covariance and gating: matched 30 s ablation", fontsize=14)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_shift_invariance(frame, output):
    labels = [
        ("c3m3", "survey_origin"),
        ("c3m3", "initial_origin"),
        ("c4m1", "survey_origin"),
        ("c4m1", "const3_like_origin"),
    ]
    display = ["c3 survey", "c3 shifted", "c4 survey", "c4 shifted"]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), constrained_layout=True)
    x = np.arange(len(labels))
    width = 0.36
    for offset, variant, name in (
        (-width / 2, VARIANTS[0], "before"),
        (width / 2, VARIANTS[1], "fixed"),
    ):
        subset = frame[frame["variant"] == variant]
        positions = [
            float(
                subset[
                    (subset["dataset"] == dataset)
                    & (subset["origin"] == origin)
                ].iloc[0]["position_rmse_m"]
            )
            for dataset, origin in labels
        ]
        rejections = [
            100.0
            * float(
                subset[
                    (subset["dataset"] == dataset)
                    & (subset["origin"] == origin)
                ].iloc[0]["rejection_rate"]
            )
            for dataset, origin in labels
        ]
        axes[0].bar(x + offset, positions, width, label=name)
        axes[1].bar(x + offset, rejections, width, label=name)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("position RMSE [m], log scale")
    axes[1].set_ylabel("UWB rejection [%]")
    for axis in axes:
        axis.set_xticks(x, display, rotation=15)
        axis.legend()
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_title("Physically identical coordinate translations")
    axes[1].set_title("Rejection decisions")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_representative(root, duration, output):
    dataset = load_dataset(
        root / "survey-results/anchor_const3.npz",
        root / "csv-data/const3/const3-trial7-tdoa3-manual3.csv",
        max_duration=duration,
    )
    results = {
        variant: run_dataset(dataset, variant=variant, yaw_error_degrees=180.0)
        for variant in VARIANTS
    }
    diagnostics = {variant: compute_metrics(dataset, result) for variant, result in results.items()}
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.2), constrained_layout=True)
    axes[0, 0].plot(dataset.p_gt[:, 0], dataset.p_gt[:, 1], "k--", lw=1.2, label="Vicon")
    for variant, label in ((VARIANTS[0], "before"), (VARIANTS[1], "fixed")):
        result = results[variant]
        diagnostic = diagnostics[variant]
        axes[0, 0].plot(result.states[:, 6], result.states[:, 7], lw=1.0, label=label)
        axes[0, 1].plot(
            diagnostic["times"],
            np.linalg.norm(diagnostic["position_error"], axis=1),
            lw=1.0,
            label=label,
        )
        axes[1, 0].plot(
            diagnostic["times"],
            np.degrees(diagnostic["yaw_error"]),
            lw=1.0,
            label=label,
        )
        axes[1, 1].plot(
            result.nis_times,
            np.cumsum(result.rejected, dtype=int),
            lw=1.0,
            label=label,
        )
    axes[0, 0].set(xlabel="x [m]", ylabel="y [m]", title="Horizontal trajectory")
    axes[0, 1].set(xlabel="time [s]", ylabel="position error [m]", title="Position error norm")
    axes[0, 1].set_yscale("log")
    axes[1, 0].set(xlabel="time [s]", ylabel="yaw error [deg]", title="Yaw error")
    axes[1, 1].set(xlabel="time [s]", ylabel="cumulative rejects", title="UWB gate lockout")
    for axis in axes.flat:
        axis.legend()
        axis.grid(alpha=0.25)
    fig.suptitle("const3 manual3, initial yaw error 180 deg", fontsize=14)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(r"E:\datasets\dataset\flight-dataset"),
    )
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--workers", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    parser.add_argument(
        "--yaw-deg",
        type=float,
        nargs="+",
        default=[0.0, 90.0, 120.0, 150.0, 180.0],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/coordinate_fix_ablation"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    jobs = [
        (
            str(args.dataset_root),
            args.duration,
            tag,
            constellation,
            relative_csv,
            yaw,
            variant,
        )
        for tag, constellation, relative_csv in DATASETS
        for yaw in args.yaw_deg
        for variant in VARIANTS
    ]
    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(_run_metric_job, job) for job in jobs]
        for completed, future in enumerate(as_completed(futures), start=1):
            record = future.result()
            records.append(record)
            print(
                f"[{completed:02d}/{len(jobs)}] "
                f"{record['dataset']} yaw={record['yaw_deg']:g} {record['variant']}",
                flush=True,
            )

    records.sort(key=lambda row: (row["dataset"], row["yaw_deg"], row["variant"]))
    serializable = [{key: _finite(value) for key, value in row.items()} for row in records]
    (args.output_dir / "metrics.json").write_text(
        json.dumps(serializable, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    frame = pd.DataFrame(records)
    frame.to_csv(args.output_dir / "metrics.csv", index=False)
    plot_grid(frame, args.yaw_deg, args.output_dir / "01_grid_metrics.png")

    shift_records = []
    for tag, constellation, relative_csv, modes in (
        ("c3m3", 3, "const3/const3-trial7-tdoa3-manual3.csv", ("survey_origin", "initial_origin")),
        ("c4m1", 4, "const4/const4-trial7-tdoa2-manual1.csv", ("survey_origin", "const3_like_origin")),
    ):
        for mode in modes:
            for variant in VARIANTS:
                shift_records.append(
                    _run_shift_case(
                        args.dataset_root,
                        args.duration,
                        tag,
                        constellation,
                        relative_csv,
                        mode,
                        variant,
                    )
                )
                print(f"[shift] {tag} {mode} {variant}", flush=True)
    shift_frame = pd.DataFrame(shift_records)
    shift_frame.to_csv(args.output_dir / "translation_invariance.csv", index=False)
    (args.output_dir / "translation_invariance.json").write_text(
        json.dumps(
            [{key: _finite(value) for key, value in row.items()} for row in shift_records],
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    plot_shift_invariance(shift_frame, args.output_dir / "02_translation_invariance.png")
    plot_representative(
        args.dataset_root,
        args.duration,
        args.output_dir / "03_representative_manual3.png",
    )
    print(f"Saved results to {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
