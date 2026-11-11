"""Rebuild the endpoint panel and two illustrative causal handoff trajectories."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator, FormatStrFormatter

# Resolve the analysis inputs and the separate manuscript figure directory.
ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "imu-uwb-inekf/experiments/radio_inertial_models/evidence/paired_acquisition"
OUTPUT = ROOT / "tex/recovery_window_revision_figures"
# Keep the printed traces available even without the large local replay state arrays.
SAMPLES = Path(__file__).with_name("closed_loop_plot_samples.csv")
# Use both marker shapes and colors so the two fitted models also separate in print.
COLORS = ("#ac5326", "#126d76")


def rms(values):
    """Compute the pooled root mean squared horizontal error before any rounding."""
    return np.sqrt(np.mean(np.square(values)))


def draw_pair(ax, before, after, labels, title):
    """Place paired model values on one common logarithmic horizontal-error axis."""
    x = np.arange(len(labels))
    ax.vlines(x, np.minimum(before, after), np.maximum(before, after), color="#a3acaf", lw=1.4)
    ax.plot(x, before, "s", ms=5.0, color=COLORS[0], mfc="white", mew=1.4, label="Affine")
    ax.plot(x, after, "o", ms=5.1, color=COLORS[1], label="Body-bias fit")
    ax.set(yscale="log", ylim=(0.095, 5.0), xticks=x, xticklabels=labels, title=title)
    ax.set_xlim(-0.45, len(labels) - 0.55)
    ax.yaxis.set_major_locator(FixedLocator([.1, .2, .5, 1., 2., 4.]))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
    ax.minorticks_off()
    ax.grid(axis="y", color="#dde1e1", lw=.6)
    ax.set_axisbelow(True)
    ax.spines[["right", "top"]].set_visible(False)


def extract_trace_examples():
    """Extract the first Delft flight and the largest late QDrone loss on the scoring grid."""
    replay = ROOT / "imu-uwb-inekf/experiments/unified_window_recovery/results"
    cases = {"Delft": "delft__2021-02-12_09_43_34_twr_triangle__L20",
             "QDrone": "qdrone__building__raw_data7_site2_newmarket_building1__L2"}
    methods = {"continue": "profile_startup_native_recovery", "replace": "ieks_profile",
               "lifted": "old_lifted"}
    frames = []
    for site, case in cases.items():
        with np.load(replay / (case + ".npz")) as trace:
            clock = trace["score_time"]
            support = trace["score_truth_supported"]
            audit = json.loads((replay / (case + ".json")).read_text())
            active = audit["audits"]["ieks_profile"]["events"][-1]["handoff_s"]
            frame = pd.DataFrame({"site": site, "relative_s": clock-clock[0],
                                  "handoff_s": active-clock[0], "reference_supported": support})
            for label, key in methods.items():
                source = trace[key+"_position"] if key == "old_lifted" else trace[key+"_X"][:, :3, 4]
                idx = np.searchsorted(trace[key+"_time"], clock, side="right") - 1
                error = np.linalg.norm(source[idx, :2] - trace["score_truth"][:, :2], axis=1)
                frame[label] = np.where(support, error, np.nan)
            frames.append(frame)
    pd.concat(frames, ignore_index=True).to_csv(SAMPLES, index=False, float_format="%.12g")


def draw_traces():
    """Compare persistent same-history replacement and continuation with the old lifted backend."""
    data = pd.read_csv(SAMPLES)
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.40), sharey=True)
    for ax, site, title in zip(axes, ["Delft", "QDrone"],
                                ["(a) Delft triangle: 20 s outage", "(b) QDrone building: 2 s outage"]):
        group = data[data.site == site]
        ax.axvline(group.handoff_s.iloc[0], color="#8c9295", lw=.9, ls=":")
        for key, label, color, style, width in [
            ("lifted", "Old lifted (L)", "#747a80", "-.", 1.1),
            ("continue", "Continue (H)", COLORS[0], "--", 1.6),
            ("replace", "Replace (W)", COLORS[1], "-", 1.1)]:
            ax.plot(group.relative_s, group[key], color=color, ls=style, lw=width, label=label)
        ax.set(yscale="log", ylim=(.002, 400), xlim=(0, 11), xticks=[0, 2, 4, 6, 8, 10],
               title=title, xlabel="Time after radio return (s)")
        ax.yaxis.set_major_locator(FixedLocator([.01, .1, 1, 10, 100]))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
        ax.grid(axis="y", color="#dde1e1", lw=.6)
        ax.spines[["right", "top"]].set_visible(False)
    axes[0].set_ylabel("Instantaneous XY error (m)")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3,
               frameon=False, bbox_to_anchor=(.5, -.005), handlelength=2.0)
    fig.subplots_adjust(left=.091, right=.99, top=.87, bottom=.31, wspace=.14)
    fig.savefig(OUTPUT / "closed_loop_xy.pdf", metadata={"Title": "Two same-history handoff interventions"})
    plt.close(fig)


def main():
    """Plot each study separately, using precisely the same yaw policy on both sides."""
    delft = pd.read_csv(DATA / "delft_same_profile.csv")
    qdrone = pd.read_csv(DATA / "qdrone_window_controls.csv")
    delft_xy = delft.groupby("duration_s")[["xy_error_m_old", "xy_error_m_new"]].agg(rms)
    sites = ["bridge", "building", "field", "tunnel"]
    q_xy = qdrone[qdrone.method.isin(["Profile1_affine", "profile_1"])].groupby(
        ["site", "method"]).xy_error_m.agg(rms).unstack().reindex(sites)
    assert len(delft) == 528 and len(qdrone[qdrone.method == "profile_1"]) == 30
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8,
                         "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.3,
                         "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.17), sharey=True)
    draw_pair(axes[0], delft_xy.xy_error_m_old, delft_xy.xy_error_m_new, ["2 s", "4 s", "8 s"],
              "(a) Delft: window duration")
    draw_pair(axes[1], q_xy.Profile1_affine, q_xy.profile_1, ["Bridge", "Building", "Field", "Tunnel"],
              "(b) QDrone: recording site")
    axes[0].set_ylabel("Endpoint XY RMSE (m)")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(.5, .005), handlelength=1.1)
    fig.subplots_adjust(left=.091, right=.99, top=.87, bottom=.275, wspace=.14)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT / "paired_xy.pdf", metadata={"Title": "Paired horizontal acquisition error"})
    plt.close(fig)
    if not SAMPLES.exists():
        extract_trace_examples()
    draw_traces()


if __name__ == "__main__":
    main()
