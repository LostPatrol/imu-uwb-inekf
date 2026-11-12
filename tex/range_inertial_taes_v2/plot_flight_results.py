"""Plot within-recording endpoint accuracy and matched range-estimator comparisons.

Run with the project Python from any directory; no estimator replay is required.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

# Directory containing the journal manuscript.
HERE = Path(__file__).resolve().parent
# Root of the research workspace.
ROOT = HERE.parents[1]
# Versioned flight experiments used in both figures.
EXP = ROOT / "imu-uwb-inekf" / "experiments"
# Matched windows for the trajectory-model comparison.
ACQ = EXP / "radio_inertial_models" / "evidence" / "paired_acquisition"
# Vector assets compiled into the article.
OUT = HERE / "figures"
# Affine trajectory color, paired with an open marker.
OLD = "#b75b28"
# Inertial trajectory correction color, paired with a filled marker.
NEW = "#176a94"


def rms(values):
    """Return root mean square without rounding individual observations."""
    return np.sqrt(np.mean(np.square(values)))


def endpoint_data():
    """Aggregate within recordings, using only Delft's zero-degree coordinate basis."""
    delft = pd.read_csv(ACQ / "delft_same_profile.csv")
    delft = delft[delft.frame_deg == 0]
    delft = delft.groupby("sequence")[["xy_error_m_old", "xy_error_m_new"]].agg(rms)
    delft.columns = ["old", "new"]
    delft.index = [name.removesuffix(".csv").split("_twr_")[-1] for name in delft.index]
    qdrone = pd.read_csv(ACQ / "qdrone_window_controls.csv")
    qdrone = qdrone[qdrone.method.isin(["Profile1_affine", "profile_1"])]
    qdrone = qdrone.groupby(["site", "sequence", "method"]).xy_error_m.agg(rms).unstack()
    qdrone = qdrone.rename(columns={"Profile1_affine": "old", "profile_1": "new"})
    labels = []
    for site, group in qdrone.groupby(level="site", sort=False):
        labels.extend(f"{site} {number}" for number in range(1, len(group) + 1))
    qdrone.index = labels
    return delft, qdrone


def plot_endpoints():
    """Show each recording's paired old/new endpoint RMSE rather than pooled bars."""
    delft, qdrone = endpoint_data()
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.65), sharex=True)
    for ax, data, title in zip(axes, [delft, qdrone],
                               ["Delft (motion capture)", "QDrone (prism proxy)"]):
        y = np.arange(len(data))
        ax.hlines(y, data.old, data.new, color="0.70", linewidth=1.1, zorder=1)
        ax.scatter(data.old, y, marker="o", s=24, facecolors="white", edgecolors=OLD,
                   linewidths=1.1, zorder=3)
        ax.scatter(data.new, y, marker="D", s=20, color=NEW, linewidths=0.6, zorder=4)
        ax.set_yticks(y, data.index)
        ax.set_ylim(len(data) - .45, -.55)
        ax.set_xscale("log")
        ax.set_xlim(.065, 8)
        ax.set_xticks([.1, .3, 1, 3], ["0.1", "0.3", "1", "3"])
        ax.set_xlabel("Horizontal RMSE (m, log scale)")
        ax.set_title(title, fontsize=8.5, pad=6)
        ax.grid(axis="x", color="0.89", linewidth=.6)
        ax.tick_params(axis="y", length=0)
        ax.minorticks_off()
        ax.spines[["top", "right", "left"]].set_visible(False)
    legend = [Line2D([], [], marker="o", color=OLD, markerfacecolor="white", linestyle="none",
                     label="Affine fit"),
              Line2D([], [], marker="D", color=NEW, linestyle="none", label="Corrected inertial fit")]
    fig.legend(handles=legend, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(.54, 1.015), handletextpad=.45, columnspacing=2)
    fig.subplots_adjust(left=.12, right=.99, bottom=.20, top=.80, wspace=.47)
    fig.savefig(OUT / "flight_endpoints.pdf", metadata={"CreationDate": None})
    plt.close(fig)
    print("Endpoint recording RMSEs (Delft: 22 windows each; QDrone: 3 each):")
    print(delft.to_string())
    print(qdrone.to_string())


def window_length_data():
    """Check matched static-calibration windows before aggregating all five methods."""
    published = EXP / "published_window_comparison" / "evidence"
    range_only = EXP / "published_range_only" / "results"
    author = pd.read_csv(published / "profile1_author_windows.csv")
    author = author[author.calibration == "static"]
    ranges = pd.read_csv(range_only / "windows.csv")
    ranges = ranges[ranges.method.isin(["relax_recover_weighted_K2",
                                       "relax_recover_weighted_K3_selected_here"])]
    profile_protocol = json.loads((published / "profile1_protocol.json").read_text(encoding="utf8"))
    range_protocol = json.loads((range_only / "protocol.json").read_text(encoding="utf8"))
    author_protocol = json.loads((published / "protocol.json").read_text(encoding="utf8"))
    for name in ["static", "sigma"]:
        assert profile_protocol["calibration"][name] == author_protocol["static_calibration"][name]
        assert profile_protocol["calibration"][name] == range_protocol["calibration_" + name]
    frame = pd.concat([author, ranges], ignore_index=True)
    keys = ["sequence", "endpoint_s", "duration_s", "layout"]
    reference = author[author.method == "Profile1_BiasPrior"].set_index(keys).sort_index()
    for method, group in frame.groupby("method"):
        assert not group.duplicated(keys).any(), method
        paired = group.set_index(keys).sort_index()
        assert paired.index.equals(reference.index), method
        assert np.array_equal(paired.events, reference.events), method
        assert np.allclose(paired.actual_endpoint_s, reference.actual_endpoint_s, atol=1e-10, rtol=0), method
        assert np.isfinite(paired.xy_error_m).all(), method
        assert paired.groupby(["layout", "duration_s"]).size().tolist() == [24, 24, 18] * 2, method
    summary = frame.groupby(["layout", "method", "duration_s"]).xy_error_m.agg(rms)
    print("Matched five methods: 66 windows per layout, duration counts 24/24/18; identical static offsets.")
    print(summary.unstack("duration_s").to_string())
    return summary


def plot_window_lengths():
    """Show the effect of inertial shape at identical ranges, endpoint times and scale."""
    summary = window_length_data()
    # The quadratic range-only control and inertial fit share all radio packets.
    styles = [
        ("relax_recover_weighted_K3_selected_here", "Range-only quadratic", OLD, "s", "--"),
        ("Profile1_BiasPrior", "Proposed range–inertial fit", NEW, "o", "-"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 1.95), sharex=True, sharey=True)
    for ax, layout, title in zip(axes, ["all8", "lower4"], ["Eight anchors", "Four lowest anchors"]):
        for method, label, color, marker, line in styles:
            curve = summary.loc[(layout, method)].reindex([2., 4., 8.])
            ax.plot([2, 4, 8], curve, label=label, color=color, marker=marker,
                    linestyle=line, linewidth=1.5, markersize=4.4, markeredgewidth=.9,
                    markerfacecolor="white" if line == "--" else color, zorder=3)
            ax.annotate(f"{curve.iloc[-1]:.3f}", (8, curve.iloc[-1]),
                        xytext=(5, 0), textcoords="offset points", ha="left",
                        va="center", color=color, fontsize=8, fontweight="bold")
        ax.set_ylim(0, .76)
        ax.set_yticks([0, .2, .4, .6])
        ax.set_xticks([2, 4, 8])
        ax.set_xlim(1.6, 10.)
        ax.set_xlabel("Window length (s)", labelpad=2)
        ax.set_title(title, fontsize=8.5, pad=3)
        ax.grid(axis="y", color="0.88", linewidth=.6, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Horizontal RMSE (m)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(.52, 1.015), fontsize=8, handletextpad=.6, columnspacing=2)
    fig.subplots_adjust(left=.092, right=.99, bottom=.22, top=.73, wspace=.11)
    fig.savefig(OUT / "window_length_comparison.pdf", metadata={"CreationDate": None})
    plt.close(fig)


def main():
    """Regenerate selected embedded-font vector figures using only existing evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure", choices=["all", "endpoints", "window-length"], default="all")
    args = parser.parse_args()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 8,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": .6})
    OUT.mkdir(parents=True, exist_ok=True)
    if args.figure in ["all", "endpoints"]:
        plot_endpoints()
    if args.figure in ["all", "window-length"]:
        plot_window_lengths()


if __name__ == "__main__":
    main()
