"""Validate paired outage replays and export inspectable plots and tables."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# The original initializer, same-gate continuation, and two practical comparators.
PLOT_METHODS = ("profile_reset", "continuous", "profile_reset_live", "damped_gap")
# Distinct colors preserve method identity across the paired panels.
COLORS = ("#d55e00", "#0072b2", "#009e73", "#cc79a7")
# Friendly names explicitly distinguish the motion prior from an inertial sensor.
LABELS = ("Profile-1 recapture", "InEKF continuation", "Profile-1, live output", "Damped-velocity prior")


def analyze(output):
    """Check sensor separation, paired prefixes, and masked-interval equality."""
    table = pd.read_csv(output / "metrics.csv")
    masked = table[table.outage].copy()
    names = list(masked.sequence.unique())
    checks = {}
    prefix_differences, gap_differences, continuous_differences = [], [], []
    continuous_by_method = {m:[] for m in table.method.unique()}
    for name in names:
        stem = Path(name).stem
        sensor = np.load(output / "cache" / (stem+".npz"))
        assert not any("truth" in key or "scoring" in key for key in sensor.files)
        original = np.load(output / "histories" / (stem+"__profile_reset__1.npz"))
        unmasked = np.load(output / "histories" / (stem+"__profile_reset__0.npz"))
        stamps = original["time"]
        prefix, gap = (stamps>=2.1)&(stamps<15), (stamps>=15)&(stamps<25)
        for method in table.method.unique():
            history = np.load(output / "histories" / (stem+"__"+method+"__1.npz"))
            normal = np.load(output / "histories" / (stem+"__"+method+"__0.npz"))
            prefix_differences.append(np.max(np.abs(history["history"][prefix]-original["history"][prefix])))
            normal_difference = float(np.nanmax(np.abs(normal["history"]-unmasked["history"])))
            continuous_differences.append(normal_difference)
            continuous_by_method[method].append(normal_difference)
            if method != "damped_gap":
                gap_differences.append(np.max(np.abs(history["history"][gap]-original["history"][gap])))
    checks.update(sensor_cache_has_no_truth=True,
        maximum_prefix_difference=float(max(prefix_differences)),
        maximum_inertial_gap_difference=float(max(gap_differences)),
        maximum_continuous_ranging_difference=float(max(continuous_differences)),
        all_cases_finite=bool(table.finite_after_start.all()), case_count=len(table))
    checks["source_sha256"] = {p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in Path(__file__).parent.glob("*.py")}
    checks["continuous_ranging_maximum_difference_by_method"] = {
        method:max(values) for method, values in continuous_by_method.items()}
    assert checks["maximum_prefix_difference"] < 1e-9
    assert checks["maximum_inertial_gap_difference"] < 1e-9
    assert checks["all_cases_finite"]
    aggregate = masked.groupby("method").agg(
        common_median_rmse_m=("common_rmse_m","median"),
        gap_median_rmse_m=("gap_rmse_m","median"),
        post_median_rmse_m=("post_rmse_m","median"),
        post_handoff_median_rmse_m=("post_handoff_rmse_m","median"),
        maximum_error_m=("common_max_m","max"),
        stable_successes=("success","sum"), elapsed_median_s=("elapsed_s","median"))
    aggregate.to_csv(output / "summary.csv")
    normal_summary = table[~table.outage].groupby("method").agg(
        common_median_rmse_m=("common_rmse_m","median"),
        maximum_error_m=("common_max_m","max"), stable_successes=("success","sum"))
    normal_summary.to_csv(output / "continuous_summary.csv")
    continuous_paired = table[~table.outage].pivot(index="sequence",columns="method", values="common_rmse_m")
    continuous_paired.to_csv(output/"continuous_paired_rmse.csv")
    paired = masked.pivot(index="sequence", columns="method",
                          values=["gap_rmse_m","post_rmse_m","post_handoff_rmse_m","common_rmse_m","success"])
    paired.to_csv(output / "paired_metrics.csv")
    (output/"validation.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.8), layout="constrained")
    for axis, name in zip(axes.flat, names):
        for method, color, label in zip(PLOT_METHODS, COLORS, LABELS):
            history = np.load(output/"histories"/(Path(name).stem+"__"+method+"__1.npz"))
            axis.plot(history["time"], history["error"], color=color, label=label, linewidth=1.05)
        axis.axvspan(15, 25, color="gray", alpha=0.13)
        axis.axvline(27, color="#666666", linestyle=":", linewidth=0.8)
        axis.axhline(0.5, color="#777777", linestyle="--", linewidth=0.6)
        axis.set_xlim(8, 45)
        axis.set_ylim(bottom=0)
        axis.grid(alpha=0.18)
        axis.set_title(Path(name).stem.split("_twr_")[1].capitalize())
        axis.set_xlabel("Replay time (s)")
        axis.set_ylabel("Position error (m)")
    fig.legend(*axes.flat[0].get_legend_handles_labels(), loc="outside upper center", ncol=4)
    fig.savefig(output/"paired_errors.png", dpi=170, bbox_inches="tight")
    fig.savefig(output/"paired_errors.svg", bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    for axis, metric, title in zip(axes,
            ("gap_rmse_m","post_rmse_m"), ("No-range interval: 15-25 s", "After radio resumes: 25 s onward")):
        for offset, method, color, label in zip(np.linspace(-0.27,0.27,4), PLOT_METHODS, COLORS, LABELS):
            values = masked.set_index(["sequence","method"]).loc[[(n,method) for n in names],metric]
            axis.bar(np.arange(6)+offset, values, width=0.18, color=color, label=label)
        axis.set_xticks(np.arange(6), [Path(n).stem.split("_twr_")[1] for n in names], rotation=25)
        axis.set_title(title)
        axis.set_ylabel("Position RMSE (m)")
        axis.grid(axis="y", alpha=0.2)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", ncol=4)
    fig.savefig(output/"paired_segments.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(aggregate.round(4).to_string())
    print(json.dumps(checks, indent=2))


def main():
    """Analyze one complete matched result directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent/"results")
    analyze(parser.parse_args().output)


if __name__ == "__main__":
    main()
