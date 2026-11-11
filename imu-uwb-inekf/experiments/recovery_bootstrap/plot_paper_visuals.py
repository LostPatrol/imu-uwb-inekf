"""Draw vector method schematics and real-window summaries for the paper.

The schematics illustrate the model; measured points come only from saved CSV.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

# Resolve the manuscript assets relative to this experiment's main project.
PROJECT = Path(__file__).resolve().parents[2]
# Keep method colors consistent with the candidate-cost figure.
COLORS = {
    "local": "#D66A3A",
    "profile": "#007F83",
    "multistart": "#65748B",
    "ink": "#233B53",
    "nuisance": "#8A65AC",
    "grid": "#E0E6EC",
}
# The six exposed recordings retain their original chronological order.
LOGS = ("triangle", "octagon", "hourglass", "random", "star", "square")
# Names and marker shapes also distinguish policies in grayscale.
POLICIES = (
    ("local", "Local", "x"),
    ("profile", "Profile-1", "o"),
    ("multistart", "Uniform-12", "D"),
)


def arrow(ax, start, end, color, width=1.0, style="-|>"):
    """Draw a short arrow using the supplied axis coordinate system."""
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle=style,
                                mutation_scale=8, lw=width, color=color))


def box(ax, x, y, width, height, title, formula, color):
    """Draw one lightly colored algorithm stage."""
    patch = FancyBboxPatch((x, y), width, height,
                           boxstyle="round,pad=0.012,rounding_size=0.018",
                           facecolor=color + "10", edgecolor=color, lw=0.85)
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height * 0.70, title, ha="center",
            va="center", fontsize=8, weight="bold", color=COLORS["ink"])
    ax.text(x + width / 2, y + height * 0.28, formula, ha="center",
            va="center", fontsize=8, color=COLORS["ink"])


def method_figure(output):
    """Illustrate trajectory structure, candidate generation, and projection."""
    fig = plt.figure(figsize=(7.16, 2.55))
    geometry = fig.add_axes([0.018, 0.12, 0.267, 0.77])
    flow = fig.add_axes([0.318, 0.06, 0.385, 0.84])
    projection = fig.add_axes([0.748, 0.09, 0.235, 0.80])
    for ax in (flow, projection):
        ax.set(xlim=(0, 1), ylim=(0, 1))
        ax.axis("off")
    for x, title in ((0.018, "(a) Window model"),
                     (0.318, "(b) Candidate and refinement"),
                     (0.748, "(c) Heading information")):
        fig.text(x, 0.965, title, fontsize=9, weight="bold",
                 color=COLORS["ink"], va="top")

    # Use a deterministic model illustration, never a claimed measured path.
    tau = np.linspace(-3.0, 0.0, 151)
    displacement = np.column_stack((0.8 * np.sin(1.7 * tau) - 1.36 * tau,
                                    0.6 * (np.cos(1.1 * tau) - 1)))
    yaw = np.deg2rad(42)
    rotation = np.array([[np.cos(yaw), -np.sin(yaw)],
                         [np.sin(yaw), np.cos(yaw)]])
    translation = np.array([1.0, 0.35]) + tau[:, None] * np.array([0.25, 0.15])
    position = translation + displacement @ rotation.T
    geometry.plot(*translation.T, "--", lw=1.4, color=COLORS["local"])
    geometry.plot(*position.T, lw=1.9, color=COLORS["profile"])
    for index in (25, 65, 105):
        geometry.plot([translation[index, 0], position[index, 0]],
                      [translation[index, 1], position[index, 1]],
                      color=COLORS["profile"], alpha=0.35, lw=0.8)
    anchors = np.array([[-0.35, 2.4], [4.0, 2.65], [3.75, -0.45]])
    geometry.scatter(*anchors.T, marker="^", s=22, color=COLORS["ink"])
    geometry.plot([anchors[0, 0], position[65, 0]],
                  [anchors[0, 1], position[65, 1]], ":", lw=0.9,
                  color=COLORS["ink"], alpha=0.7)
    geometry.scatter(*position[-1], s=21, color=COLORS["profile"], zorder=4)
    geometry.annotate(r"$p_T=c_0$", position[-1], xytext=(6, -13),
                      textcoords="offset points", fontsize=8, color=COLORS["ink"])
    geometry.text(-0.48, 2.8, "surveyed anchors", fontsize=7.7,
                  color=COLORS["ink"])
    geometry.text(1.82, 1.72, r"$p(t)$", color=COLORS["profile"], fontsize=9)
    geometry.text(-0.12, -0.42, r"$c(t)$", color=COLORS["local"], fontsize=9)
    geometry.text(0.08, 1.07, r"$p(t)=c(t)+R_z(\psi)d(t)$",
                  transform=geometry.transAxes, fontsize=8.5, color=COLORS["ink"])
    geometry.set(xlim=(-0.6, 5.0), ylim=(-0.7, 3.0), aspect="equal")
    geometry.axis("off")
    fig.text(0.018, 0.055, r"Known IMU shape $d(t)$; unknown $c(t),\psi$",
             fontsize=7.8, color=COLORS["ink"])

    box(flow, 0.02, 0.78, 0.67, 0.18, "Squared-range lift",
        r"$y=B\nu+Dc+Cu$", COLORS["ink"])
    box(flow, 0.02, 0.53, 0.67, 0.18, "Nuisance projection",
        r"$\bar y=P_LWy,\quad\bar C=P_LWC$", COLORS["profile"])
    box(flow, 0.02, 0.28, 0.67, 0.18, "Circle candidate",
        r"$u=[\cos\psi,\sin\psi]^\mathsf{T}$", COLORS["profile"])
    box(flow, 0.02, 0.02, 0.96, 0.18, "Raw-range Huber refinement",
        r"endpoint $p_T,v_T,\psi$ for the InEKF", COLORS["ink"])
    for top, bottom in ((0.78, 0.71), (0.53, 0.46), (0.28, 0.20)):
        arrow(flow, (0.355, top - 0.012), (0.355, bottom + 0.012),
              COLORS["profile"])
    cv_patch = FancyBboxPatch((0.73, 0.30), 0.25, 0.37,
                             boxstyle="round,pad=0.012",
                             facecolor=COLORS["local"] + "10",
                             edgecolor=COLORS["local"], lw=0.85)
    flow.add_patch(cv_patch)
    flow.text(0.855, 0.485, "CV fit from\ntwo seeds\n$\\downarrow$\nretain best\ntranslation",
              ha="center", va="center", fontsize=7.6, color=COLORS["ink"])
    arrow(flow, (0.855, 0.285), (0.855, 0.212), COLORS["local"])

    # These Euclidean sketches depict the local information formula, not data.
    projection.text(0, 0.96, "Yaw component survives", fontsize=8,
                    color=COLORS["ink"], va="top")
    projection.plot([0.06, 0.92], [0.59, 0.59],
                    color=COLORS["nuisance"], lw=2.7, alpha=0.45)
    arrow(projection, (0.09, 0.59), (0.67, 0.84), COLORS["profile"], 1.35)
    arrow(projection, (0.67, 0.59), (0.67, 0.84), COLORS["local"], 1.35)
    projection.text(0.33, 0.83, r"$a_\psi$", fontsize=8.5, color=COLORS["profile"])
    projection.text(0.72, 0.77, r"$\mathcal{I}>0$", fontsize=8.5,
                    color=COLORS["local"])
    projection.text(0.13, 0.51, r"$\mathrm{col}(A_n)$", fontsize=8,
                    color=COLORS["nuisance"])
    projection.text(0, 0.40, "Yaw lies in nuisance span", fontsize=8,
                    color=COLORS["ink"])
    projection.plot([0.06, 0.92], [0.08, 0.31],
                    color=COLORS["nuisance"], lw=2.7, alpha=0.45)
    arrow(projection, (0.09, 0.088), (0.69, 0.248), COLORS["profile"], 1.35)
    projection.text(0.31, 0.27, r"$a_\psi$", fontsize=8.5, color=COLORS["profile"])
    projection.text(0.65, 0.095, r"$\mathcal{I}=0$", fontsize=8.5,
                    color=COLORS["nuisance"])
    projection.text(0.11, -0.02, r"$\mathrm{col}(A_n)$", fontsize=8,
                    color=COLORS["nuisance"])
    for suffix in ("pdf", "png"):
        fig.savefig(output / f"method_overview.{suffix}", dpi=300)
    plt.close(fig)


def real_results_figure(summary_path, output):
    """Plot saved per-log objective misses and endpoint position percentiles."""
    with summary_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    indexed = {}
    for row in rows:
        for log in LOGS:
            if f"_twr_{log}.csv" in row["sequence"]:
                indexed[log, row["method"]] = row
    selected = [indexed[log, method] for log in LOGS for method, _, _ in POLICIES]
    if any(int(row["configurations"]) != 88 for row in selected):
        raise ValueError("The plotted protocol requires 88 configurations per log.")

    fig, axes = plt.subplots(1, 2, figsize=(3.4, 3.0), sharey=True)
    y = np.arange(len(LOGS))
    for ax, field in zip(axes, ("objective_misses", "p95_position_error_m")):
        for index, log in enumerate(LOGS):
            start = float(indexed[log, "local"][field])
            end = float(indexed[log, "profile"][field])
            ax.plot([start, end], [index, index], color=COLORS["grid"],
                    lw=2.5, zorder=1)
        # Draw the larger reference marker first so coincident points remain visible.
        for method, _, marker in reversed(POLICIES):
            values = [float(indexed[log, method][field]) for log in LOGS]
            ax.scatter(values, y, marker=marker,
                       s=32 if method == "multistart" else 20,
                       facecolors="none" if method == "multistart" else COLORS[method],
                       edgecolors=COLORS[method] if marker != "x" else None,
                       color=COLORS[method], linewidths=1.1, zorder=3)
        ax.grid(axis="x", color=COLORS["grid"], linewidth=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", labelsize=8, length=2)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color(COLORS["grid"])
        ax.set_ylim(5.6, -0.6)
    axes[0].set(yticks=y, yticklabels=[name.title() for name in LOGS],
                xlim=(-2.5, 39), xticks=(0, 15, 30), xlabel="Objective misses")
    axes[1].set(xlim=(-0.4, 8.8), xticks=(0, 4, 8), xlabel=r"Position $p_{95}$ (m)")
    axes[0].set_title("(a) Optimization", loc="left", fontsize=8.5, color=COLORS["ink"])
    axes[1].set_title("(b) Endpoint error", loc="left", fontsize=8.5, color=COLORS["ink"])
    handles = [Line2D([], [], marker=marker, color=COLORS[method], ls="none",
                      markerfacecolor="none" if method == "multistart" else COLORS[method],
                      markersize=5, label=label) for method, label, marker in POLICIES]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.55, 0.995),
               ncol=3, frameon=False, fontsize=7.8, handletextpad=0.25,
               columnspacing=0.65)
    fig.subplots_adjust(left=0.235, right=0.985, bottom=0.16, top=0.82, wspace=0.28)
    for suffix in ("pdf", "png"):
        fig.savefig(output / f"real_window_outcomes.{suffix}", dpi=300)
    plt.close(fig)
    return selected


def main():
    """Rebuild the manuscript visuals without running an estimator fit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-summary", type=Path,
                        default=PROJECT / "results/recovery_real_windows_20261111/per_log_summary.csv")
    parser.add_argument("--output", type=Path,
                        default=PROJECT.parent / "tex/recovery_bootstrap_figures")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "pdf.fonttype": 42, "axes.labelsize": 8,
                         "text.color": COLORS["ink"], "axes.labelcolor": COLORS["ink"]})
    method_figure(args.output)
    selected = real_results_figure(args.real_summary, args.output)
    manifest = {
        "description": "Model schematics and saved real-window outcomes; no estimator rerun.",
        "schematic_panels": ["trajectory decomposition", "candidate pipeline",
                             "nuisance-projected yaw sensitivity"],
        "real_summary_sha256": hashlib.sha256(args.real_summary.read_bytes()).hexdigest(),
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "real_plotted_rows": selected,
        "figures": ["method_overview.pdf", "real_window_outcomes.pdf"],
    }
    (args.output / "paper_visuals_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
