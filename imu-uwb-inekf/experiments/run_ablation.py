#!/usr/bin/env python3
"""Run matched legacy/verified InEKF ablations and save inspectable results."""

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ESTIMATION_DIR = Path(__file__).resolve().parents[1] / "scripts" / "estimation"
sys.path.insert(0, str(ESTIMATION_DIR))

from pipeline import compute_metrics, load_dataset, run_dataset  # noqa: E402


def _finite_json(value):
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def plot_pair(dataset, legacy, verified, yaw_degrees, output_file):
    legacy_diag = compute_metrics(dataset, legacy)
    verified_diag = compute_metrics(dataset, verified)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

    axes[0].plot(dataset.p_gt[:, 0], dataset.p_gt[:, 1], "k--", lw=1.1, label="Vicon")
    for result in (legacy, verified):
        axes[0].plot(result.states[:, 6], result.states[:, 7], lw=0.9, label=result.variant)
    axes[0].set(xlabel="x [m]", ylabel="y [m]", title=f"Trajectory, initial yaw {yaw_degrees:g} deg")
    axes[0].axis("equal")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    for result, diag in ((legacy, legacy_diag), (verified, verified_diag)):
        axes[1].plot(
            diag["times"],
            np.linalg.norm(diag["position_error"], axis=1),
            lw=0.8,
            label=result.variant,
        )
        axes[2].plot(diag["times"], np.degrees(diag["yaw_error"]), lw=0.8, label=result.variant)
    axes[1].set(xlabel="time [s]", ylabel="position error [m]", title="Position error norm")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    axes[2].axhline(5.0, color="0.5", ls="--", lw=0.7)
    axes[2].axhline(-5.0, color="0.5", ls="--", lw=0.7)
    axes[2].set(xlabel="time [s]", ylabel="yaw error [deg]", title="Yaw convergence")
    axes[2].grid(alpha=0.25)
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(output_file, dpi=180)
    plt.close(fig)


def plot_summary(table, output_file):
    yaws = sorted(table["yaw_error_degrees"].unique())
    variants = list(table["variant"].drop_duplicates())
    x = np.arange(len(yaws))
    width = 0.72 / len(variants)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.0))
    fields = [
        ("position_rmse_m", "position RMSE [m]"),
        ("velocity_rmse_mps", "velocity RMSE [m/s]"),
        ("orientation_geodesic_rmse_rad", "attitude RMSE [rad]"),
        ("runtime_seconds", "replay runtime [s]"),
    ]
    for axis, (field, label) in zip(axes.flat, fields):
        for variant_index, variant in enumerate(variants):
            offset = (variant_index - (len(variants) - 1) / 2) * width
            subset = table[table["variant"] == variant].set_index("yaw_error_degrees")
            values = [subset.loc[yaw, field] for yaw in yaws]
            axis.bar(x + offset, values, width, label=variant)
        axis.set_xticks(x, [f"{yaw:g}" for yaw in yaws])
        axis.set(xlabel="initial yaw error [deg]", ylabel=label)
        axis.grid(axis="y", alpha=0.25)
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(output_file, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("anchors")
    parser.add_argument("csv")
    parser.add_argument("--yaw-deg", type=float, nargs="+", default=[0.0, 90.0, 180.0])
    parser.add_argument("--max-duration", type=float)
    parser.add_argument("--variants", nargs="+", default=["legacy", "proposed"])
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(args.anchors, args.csv, max_duration=args.max_duration)
    records = []

    for yaw in args.yaw_deg:
        if len(args.variants) != 2:
            raise ValueError("Comparison plotting currently requires exactly two variants")
        legacy = run_dataset(dataset, variant=args.variants[0], yaw_error_degrees=yaw)
        verified = run_dataset(dataset, variant=args.variants[1], yaw_error_degrees=yaw)
        for result in (legacy, verified):
            records.append(
                {"dataset": dataset.name, "yaw_error_degrees": yaw, "variant": result.variant, **result.metrics}
            )
        plot_pair(
            dataset,
            legacy,
            verified,
            yaw,
            output_dir / f"comparison_yaw_{int(round(yaw)):03d}.png",
        )

    table = pd.DataFrame.from_records(records)
    table.to_csv(output_dir / "metrics.csv", index=False, encoding="utf-8")
    serializable = [{key: _finite_json(value) for key, value in row.items()} for row in records]
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(serializable, stream, ensure_ascii=False, indent=2)
    plot_summary(table, output_dir / "summary_metrics.png")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
