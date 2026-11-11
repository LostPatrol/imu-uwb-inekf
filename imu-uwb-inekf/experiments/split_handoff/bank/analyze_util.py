"""Summarize matched shared UTIL replay metrics without selecting bank rules."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter, ScalarFormatter
import numpy as np
import pandas as pd

# These are the matched shared handoff baselines, never outcome-selected subsets.
BASELINES = ["retain_normal", "forced_yaw_only", "full_attitude", "forced_evidence_yaw", "normal_unmodified"]
# Every paired comparison uses the same dataset, scenario and injected yaw.
PAIR_KEYS = ["name", "scenario", "yaw_offset_deg"]


def paired_metrics(frame):
    """Preserve every bank case and match baseline post-handoff errors by case key."""
    bank_rows = frame[frame.policy == "bank_yaw"].copy()
    for baseline in BASELINES:
        reference = frame[frame.policy == baseline][PAIR_KEYS+["post_xy_rmse", "post_yaw_rmse_deg", "failed"]]
        reference = reference.rename(columns={key: baseline+"_"+key for key in reference.columns if key not in PAIR_KEYS})
        bank_rows = bank_rows.merge(reference, on=PAIR_KEYS, how="left", validate="one_to_one")
        bank_rows["xy_change_vs_"+baseline] = bank_rows.post_xy_rmse-bank_rows[baseline+"_post_xy_rmse"]
    bank_rows["error_group"] = np.where(bank_rows.yaw_offset_deg == 0, "nominal_injection", "large_injection")
    bank_rows["geometry"] = bank_rows.name.str.extract(r"const(\d+)").astype(int)
    bank_rows["radio_protocol"] = bank_rows.name.str.extract(r"tdoa(\d+)").astype(int)
    bank_rows["trial"] = bank_rows.name.str.extract(r"trial(\d+)").astype(int)
    return bank_rows


def summarize(group):
    """Report both successful and failed trajectories and paired adverse changes."""
    change = group.xy_change_vs_retain_normal
    row = dict(cases=len(group), accepted=int(group.heading_accepted.fillna(False).astype(bool).sum()),
        failures=int(group.failed.sum()), bank_xy_median=float(group.post_xy_rmse.median()),
        finite_xy_cases=int(np.isfinite(group.post_xy_rmse).sum()),
        minimum_post_samples=int(group.post_samples.min()), minimum_last_time=float(group.post_last_time.min()),
        bank_xy_p95=float(group.post_xy_rmse.quantile(.95)), bank_xy_max=float(group.post_xy_rmse.max()),
        retained_xy_median=float(group.retain_normal_post_xy_rmse.median()),
        paired_xy_change_median=float(change.median()), paired_xy_change_mean=float(change.mean()),
        paired_xy_change_worst=float(change.max()), improved_cases=int((change < -1e-8).sum()),
        worsened_cases=int((change > 1e-8).sum()), tied_cases=int((change.abs() <= 1e-8).sum()))
    for label in ("decision_elapsed_s", "total_nfev", "nonconverged_prefix_fits", "blocks"):
        if label in group:
            row[label+"_median"] = float(group[label].median())
            row[label+"_max"] = float(group[label].max())
    for baseline in BASELINES[1:]:
        row[baseline+"_xy_median"] = float(group[baseline+"_post_xy_rmse"].median())
        row["paired_change_vs_"+baseline+"_median"] = float(group["xy_change_vs_"+baseline].median())
    return pd.Series(row)


def plot_pairs(pairs, output):
    """Plot every matched case against retention and forced full-attitude handoff."""
    figure, axes = plt.subplots(1, 2, figsize=(9, 4.2), layout="constrained")
    for axis, baseline in zip(axes, ("retain_normal", "full_attitude")):
        x = pairs[baseline+"_post_xy_rmse"].to_numpy()
        y = pairs.post_xy_rmse.to_numpy()
        positive = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
        limits = [float(min(x[positive].min(), y[positive].min())*.9),
                  float(max(x[positive].max(), y[positive].max())*1.1)]
        for group, color in (("nominal_injection", "#2868ac"), ("large_injection", "#ce6d20")):
            select = positive & (pairs.error_group.to_numpy() == group)
            axis.scatter(x[select], y[select], c=color, s=19, alpha=.6, label=group.replace("_", " "))
        switched = positive & pairs.heading_accepted.fillna(False).astype(bool).to_numpy()
        axis.scatter(x[switched], y[switched], color="#8b1e2d", marker="x", s=40, label="accepted correction")
        axis.plot(limits, limits, color=".4", linewidth=1)
        axis.set(xscale="log", yscale="log", xlim=limits, ylim=limits,
                 xlabel=baseline.replace("_", " ")+" post XY RMSE (m)", ylabel="bank post XY RMSE (m)")
        for coordinate in (axis.xaxis, axis.yaxis):
            coordinate.set_major_locator(LogLocator(base=10, subs=[1., 2., 5.]))
            coordinate.set_major_formatter(ScalarFormatter())
            coordinate.set_minor_formatter(NullFormatter())
        axis.grid(alpha=.2)
        axis.set_title(f"{int(positive.sum())} matched trajectories")
    axes[0].legend(fontsize=8)
    figure.savefig(output/"paired_post_xy.png", dpi=180)
    figure.savefig(output/"paired_post_xy.svg")
    plt.close(figure)


def main():
    """Read explicit current-run result directories and save bank-local summaries."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True, help="Current shared UTIL output directories")
    parser.add_argument("--output", default="results")
    args = parser.parse_args()
    output = Path(__file__).parent/args.output
    output.mkdir(exist_ok=True)
    frames = []
    for source in args.inputs:
        directory = Path(source)
        frame = pd.read_csv(directory/"metrics.csv")
        frame["split"] = directory.name
        records_file = directory/"records.json"
        if records_file.exists():
            records = json.loads(records_file.read_text(encoding="utf-8"))
            diagnostics = {record["key"]: record.get("decision") or {} for record in records}
            for column, field in (("decision_elapsed_s", "elapsed_s"), ("total_nfev", "total_nfev"),
                                  ("nonconverged_prefix_fits", "nonconverged_prefix_fits"), ("blocks", "blocks"),
                                  ("posterior_mass", "posterior_mass")):
                frame[column] = frame.key.map(lambda key: diagnostics.get(key, {}).get(field, np.nan))
        frames.append(frame)
    pairs = paired_metrics(pd.concat(frames, ignore_index=True))
    pairs.to_csv(output/"paired_cases.csv", index=False)
    plot_pairs(pairs, output)
    for label, keys in [("scenario", ["split", "scenario", "error_group"]),
                        ("geometry", ["split", "geometry", "radio_protocol", "scenario", "error_group"]),
                        ("injection", ["split", "scenario", "yaw_offset_deg"])]:
        summary = pairs.groupby(keys, dropna=False).apply(summarize, include_groups=False).reset_index()
        summary.to_csv(output/(label+"_summary.csv"), index=False)
        if label == "scenario":
            print(summary.to_string(index=False))
    print(f"Wrote all {len(pairs)} paired cases to {output}")


if __name__ == "__main__":
    main()
