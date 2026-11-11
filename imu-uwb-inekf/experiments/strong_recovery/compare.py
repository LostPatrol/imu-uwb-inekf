"""Build flight-paired XY evidence and figures without pooling native sensor domains.

Use --collect after the native replays. Without it, only the compact published
case table is needed to regenerate the site, duration and time-window summaries.
"""

import argparse
from pathlib import Path
from shutil import copyfile

import numpy as np
import pandas as pd

# Native replay inputs on this project; only compact case evidence is published.
HERE = Path(__file__).resolve().parent
PRIVATE = HERE.parents[2] / "agent/codex"
EVIDENCE = HERE / "evidence"
# Method and window names shared by all adapters in the final tables.
BRANCHES = ("K", "guarded", "always_C", "bounds")
WINDOWS = ("0-2", "2-5", "6-11")
# A tolerance for floating-point equality, not a practical significance threshold.
NUM_TIE = 1e-12


def standardize(frame, fields, domain, methods):
    """Map one native long table to one matched row per actual recording/length/window."""
    frame = frame.rename(columns=fields).copy()
    frame["domain"] = domain
    if "site" not in frame:
        frame["site"] = domain
    frame["branch"] = frame["branch"].map(methods)
    frame = frame[frame.branch.isin(BRANCHES)]
    keys = ["domain", "site", "sequence", "gap_s", "window"]
    wide = frame.pivot(index=keys, columns="branch", values="xy_m").reindex(columns=BRANCHES)
    wide["reference_coverage"] = frame[frame.branch == "K"].set_index(keys)["support"]
    return wide.reset_index()


def collect():
    """Read the preregistered native cohorts; keep unsupported external windows visible."""
    parts = []
    data = pd.read_csv(PRIVATE / "strong_twr/length_results/paired_metrics_complete.csv")
    for window, native in zip(WINDOWS, ("early0_2", "early2_5", "future6_11")):
        data["window"] = window
        parts.append(standardize(data, {"method":"branch", native+"_xy_rmse_m":"xy_m", native+"_coverage":"support"},
            "Delft TWR", {key:key for key in BRANCHES}))
    for source, label in [("independent", "Toronto TDoA"), ("feedback_extension", "Toronto supplementary TDoA")]:
        data = pd.read_csv(HERE / "oursland" / source / "paired.csv")
        for window, native in zip(WINDOWS, ("early", "middle", "later")):
            data["window"] = window
            parts.append(standardize(data, {"method":"branch", "name":"sequence", native+"_xy_rmse":"xy_m", native+"_coverage":"support"},
                label, {"K":"K", "always_C":"always_C", "guarded":"guarded", "radio_bounds":"bounds"}))
    data = pd.read_csv(HERE / "delft_tdoa/paired/metrics.csv")
    parts.append(standardize(data[data.queue == "first_per_stamp"],
        {"name":"sequence", "outage_s":"gap_s", "interval":"window", "policy":"branch", "xy_rmse":"xy_m", "coverage":"support"},
        "Delft TDoA", {"K":"K", "always_C":"always_C", "guarded_C":"guarded", "bound_xy":"bounds"}))
    for folder, label in [("qdrone_verified", "QDrone TWR"), ("qdrone_midpoint_verified", "QDrone midpoint")]:
        data = pd.read_csv(PRIVATE / "strong_external" / folder / "metrics.csv")
        data["window"] = data.window.map(dict(first2="0-2", two_to_five="2-5", six_to_eleven="6-11"))
        parts.append(standardize(data, {"method":"branch", "length_s":"gap_s", "xy_rmse_m":"xy_m", "truth_coverage":"support"},
            label, {"K":"K", "proposal":"guarded", "always_C":"always_C", "radio_bounds":"bounds"}))
    return pd.concat(parts, ignore_index=True).sort_values(["domain", "site", "sequence", "gap_s", "window"])


def summarize(cases, keys):
    """Count every eligible matched pair, including ties, and report absolute and paired medians."""
    rows = []
    for labels, group in cases.groupby(keys):
        eligible = group[group.reference_coverage >= .95]
        for branch in BRANCHES[1:]:
            pairs = eligible[["K", branch]].dropna()
            delta = pairs[branch] - pairs.K
            rows.append({**dict(zip(keys, labels)), "method":branch, "all_cases":len(group), "eligible":len(eligible),
                "pairs":len(pairs), "K_m":pairs.K.median(), "method_m":pairs[branch].median(),
                "paired_delta_m":delta.median(), "best_delta_m":delta.min(), "worst_delta_m":delta.max(),
                "wins":int((delta < -NUM_TIE).sum()), "ties":int((delta.abs() <= NUM_TIE).sum()),
                "losses":int((delta > NUM_TIE).sum())})
    return pd.DataFrame(rows)


def plot(cases, target):
    """Show native RMSE magnitudes on common per-flight terms; circles mark the fixed durations."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    domains = ["Delft TWR", "QDrone TWR", "QDrone midpoint", "Toronto TDoA", "Delft TDoA"]
    fig, axes = plt.subplots(1, len(domains), figsize=(13.0, 3.4), layout="constrained")
    styles = [("K", "#1d3048", "o"), ("guarded", "#166b63", "s"), ("always_C", "#9a889a", "."),
              ("bounds", "#c25923", "^")]
    for ax, domain in zip(axes, domains):
        group = cases[(cases.domain == domain) & (cases.window == "0-2") & (cases.reference_coverage >= .95)]
        for branch, colour, marker in styles:
            line = group.groupby("gap_s")[branch].median()
            ax.plot(line.index, line.values, color=colour, marker=marker, label=branch,
                    linestyle=":" if branch == "always_C" else "-", linewidth=1.5, markersize=5)
        ax.set(title=domain, xlabel="Outage (s)", xticks=[2,5,10,20], yscale="log")
        ax.grid(axis="y", alpha=.2, which="both")
    axes[0].set_ylabel("Median early XY RMSE (m)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=4, frameon=False)
    fig.savefig(target, dpi=175)
    plt.close(fig)


def collect_audits():
    """Retain small reproducible diagnostics underlying the reported failures and scope."""
    sources = {
        "delft_twr_packet_order.csv":PRIVATE/"strong_twr/packet_summary.csv",
        "delft_twr_mechanism_pairs.csv":PRIVATE/"strong_twr/paired_diagnostics.csv",
        "delft_twr_first_packets.png":PRIVATE/"strong_twr/first_packets.png",
        "delft_twr_display_jumps.csv":PRIVATE/"strong_twr/length_results/paired_summary.csv",
        "qdrone_compatibility.json":PRIVATE/"strong_external/compatibility.json",
    }
    for folder, name in [("qdrone_verified","qdrone"),("qdrone_midpoint_verified","qdrone_midpoint")]:
        for artifact in ["truth_motion_coverage", "fixed_disk_containment_audit", "jumps"]:
            sources[name+"_"+artifact+".csv"] = PRIVATE/"strong_external"/folder/(artifact+".csv")
    for target, path in sources.items():
        copyfile(path, EVIDENCE/target)


def main():
    """Publish compact paired data or regenerate summaries from that data alone."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--collect", action="store_true")
    args = parser.parse_args()
    EVIDENCE.mkdir(exist_ok=True)
    table = EVIDENCE / "paired_xy.csv"
    if args.collect:
        collect().to_csv(table, index=False)
        collect_audits()
    cases = pd.read_csv(table)
    summarize(cases, ["domain","gap_s","window"]).to_csv(EVIDENCE/"by_duration.csv",index=False)
    summarize(cases, ["domain","site","gap_s","window"]).to_csv(EVIDENCE/"by_site.csv",index=False)
    plot(cases, EVIDENCE / "early_xy.png")


if __name__ == "__main__":
    main()
