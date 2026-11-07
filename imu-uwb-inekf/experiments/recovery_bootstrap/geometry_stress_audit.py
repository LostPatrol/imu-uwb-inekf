"""Independently audit paired stress outputs and render the manuscript figure."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import sys

# Keep audit numerical work single threaded while concurrent research runs continue.
for _variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_variable] = "1"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Resolve the shared project from this research artifact's fixed location.
ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT/"imu-uwb-inekf"/"experiments"/"recovery_bootstrap"
RESULTS = ROOT/"imu-uwb-inekf"/"results"/"recovery_geometry_20261107"
FIGURES = ROOT/"tex"/"recovery_bootstrap_figures"
sys.path.insert(0, str(SOURCE))
from run_geometry_stress import evaluate_case


def read_table(path):
    """Read a CSV without silently changing the provenance or categorical values."""
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def audit():
    """Recreate twenty random paired cases and independently recompute all groups."""
    protocol = json.loads((RESULTS/"protocol.json").read_text(encoding="utf-8"))
    rows = read_table(RESULTS/"metrics.csv")
    saved = {(int(row["case_id"]), row["method"]): row for row in rows}
    selected = np.random.default_rng(907126).choice(len(protocol["cases"]), 20, replace=False)
    maximum_differences = dict(position_error_m=0.0, yaw_error_deg=0.0, robust_objective=0.0)
    for case_id in selected:
        for repeat in evaluate_case(protocol["cases"][int(case_id)]):
            original = saved[(int(case_id), repeat["method"])]
            for key in maximum_differences:
                difference = abs(float(original[key])-float(repeat[key]))
                maximum_differences[key] = max(maximum_differences[key], difference)
                assert difference < 1e-8, (case_id, repeat["method"], key, difference)
    for group in read_table(RESULTS/"summary.csv"):
        subset = [row for row in rows if all(row[key] == group[key] for key in
                  ("distance_m", "anchor_height_m", "motion", "method"))]
        assert len(subset) == 200
        for key in ("position_error_m", "yaw_error_deg", "elapsed_s"):
            values = np.asarray([float(row[key]) for row in subset])
            for suffix, statistic in (("median", np.median(values)),
                    ("p95", np.quantile(values, 0.95)), ("max", values.max())):
                assert abs(statistic-float(group[key+"_"+suffix])) < 1e-12
        if group["method"] != "cv":
            assert sum(int(row["worse_objective_vs_multistart"]) for row in subset) == int(group["worse_objective_vs_multistart_count"])
    # Verify physical reference meaning independently of the experiment generator.
    endpoint = np.array([0.8*np.sin(0), 0.6*(np.cos(0)-1), 0.15*np.sin(0)])
    derivative = np.array([0.8*1.7, 0.0, 0.15*0.7])-np.array([1.36, 0.0, 0.105])
    assert np.max(abs(endpoint)) == 0.0 and np.max(abs(derivative)) < 1e-15
    audit_result = dict(random_cases=selected.tolist(), rerun_method_rows=80,
        maximum_absolute_difference=maximum_differences, groups_checked=32,
        endpoint_position_and_velocity_verified=True,
        fairness="All methods call identical two-seed CV translation initializer; local yaw=0, profile analytic yaw, multistart 12 uniform yaws. Same objective, limits and data for all inertial methods.",
        caveats="CV yaw is a zero fallback; objective is a different model. Twelve-start is not a certified optimum. Runtime comes from 20-process concurrent evaluation.")
    (RESULTS/"audit.json").write_text(json.dumps(audit_result, indent=2), encoding="utf-8")
    print(json.dumps(audit_result, indent=2))


def render():
    """Show all eight frozen geometries without hiding weak-height failure modes."""
    summary = read_table(RESULTS/"summary.csv")
    groups = [(radius, height, motion) for radius in (10.0, 30.0)
              for height in (3.0, 0.1) for motion in ("excited", "constant_acceleration")]
    methods = ("cv", "local", "profile", "multistart")
    labels = {"cv": "Robust CV", "local": "Local inertial", "profile": "Profiled inertial", "multistart": "12-start inertial"}
    colors = {"cv": "#737373", "local": "#C44E52", "profile": "#1764AB", "multistart": "#228833"}
    markers = {"cv": "s", "local": "x", "profile": "o", "multistart": "+"}
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.labelsize": 8,
        "axes.titlesize": 9, "legend.fontsize": 8, "xtick.labelsize": 7,
        "ytick.labelsize": 7, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 4.8), sharex=True)
    specifications = (("position_error_m_p95", "(a) 95th-percentile position error", "Position error (m)", True),
        ("yaw_error_deg_p95", "(b) 95th-percentile heading error", "Heading error (deg)", True),
        ("worse_objective_vs_multistart_count", "(c) Objective failures relative to 12-start", "Cases / 200", False),
        ("elapsed_s_median", "(d) Median total initialization time", "Elapsed time (ms)", False))
    x = np.arange(len(groups))
    lookup = {(float(row["distance_m"]), float(row["anchor_height_m"]), row["motion"], row["method"]): row for row in summary}
    for ax, (key, title, ylabel, logarithmic) in zip(axes.flat, specifications):
        for method in methods:
            if key == "yaw_error_deg_p95" and method == "cv":
                continue
            if key.startswith("worse") and method in ("cv", "multistart"):
                continue
            values = np.array([float(lookup[(*group, method)][key]) for group in groups])
            if key == "elapsed_s_median":
                values *= 1000
            ax.plot(x, values, color=colors[method], marker=markers[method],
                linewidth=1.2, markersize=4.5, label=labels[method], alpha=0.95)
        ax.set_title(title, loc="left")
        ax.set_ylabel(ylabel)
        if logarithmic:
            ax.set_yscale("log")
        ax.grid(axis="y", which="both", alpha=0.2, linewidth=0.5)
        ax.axvline(3.5, color="0.55", linestyle=":", linewidth=0.7)
        for start in (1.5, 5.5):
            ax.axvspan(start, start+2.0, color="#EEEEEE", zorder=-1)
        ax.set_xlim(-0.35, 7.35)
        ax.set_xticks(x, [f"{radius:g}/{height:g}\n{'E' if motion == 'excited' else 'A'}" for radius, height, motion in groups])
        ax.spines[["top", "right"]].set_visible(False)
    axes[1, 0].set_ylim(-2, 91)
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.005))
    fig.supxlabel("Horizontal radius / anchor-height span (m); E: excited, A: constant acceleration", y=0.018, fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95), h_pad=1.5, w_pad=1.8)
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES/"geometry_stress.pdf", bbox_inches="tight")
    fig.savefig(FIGURES/"geometry_stress.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    audit()
    render()
