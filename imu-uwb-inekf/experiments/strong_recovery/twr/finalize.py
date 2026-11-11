"""Offline paired summaries, display jumps, and plots for frozen TWR results."""

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# This script only summarizes completed outputs; it never calls an estimator.
ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT/"agent/codex/strong_twr/length_results"
# These four predeclared durations are stress tests on six shared development flights.
LENGTHS = (2, 5, 10, 20)
METHODS = ("K", "always_C", "guarded", "bounds")


def main():
    """Preserve every case while reporting paired changes and return-to-K jumps."""
    frame = pd.read_csv(OUT/"paired_metrics.csv")
    jumps, paired, feasibility = [], [], []
    for file in sorted(OUT.glob("*.npz")):
        data = np.load(file)
        times, original = data["time"], data["K_pos"]
        scope = (times >= 25) & (times <= 27.02)
        end = int(np.searchsorted(times, 27))
        for method in METHODS:
            offset = data[method+"_pos"]-original
            added = np.r_[0., np.linalg.norm(np.diff(offset[:, :2], axis=0), axis=1)]
            early = (times >= 25) & (times < 27)
            z_error = data[method+"_pos"][early, 2]-data["truth"][early, 2]
            jumps.append(dict(sequence=file.stem.split("__gap")[0],
                gap_s=int(file.stem.split("__gap")[1]), method=method,
                early0_2_z_rmse_m=float(np.sqrt(np.mean(z_error**2))),
                early0_2_z_peak_m=float(np.max(np.abs(z_error))),
                xy_added_offset_jump_max_m=float(np.max(added[scope])),
                return27_xy_added_offset_jump_m=float(added[end]),
                xy_offset_max_m=float(np.max(np.linalg.norm(offset[scope, :2], axis=1)))))
        diagnostic = json.loads((OUT/(file.stem+"_bounds_diagnostics.json")).read_text(encoding="utf-8"))
        feasibility.append(dict(sequence=file.stem.split("__gap")[0],
            gap_s=int(file.stem.split("__gap")[1]), **diagnostic["scoring_only"], **diagnostic["geometry"]))
    frame = frame.merge(pd.DataFrame(jumps), on=["sequence", "gap_s", "method"])
    frame.to_csv(OUT/"paired_metrics_complete.csv", index=False)
    pd.DataFrame(feasibility).to_csv(OUT/"bounds_feasibility.csv", index=False)
    for length, group in frame.groupby("gap_s"):
        baseline = group[group.method == "K"].set_index("sequence")
        for method in METHODS[1:]:
            selected = group[group.method == method].set_index("sequence")
            delta = selected.early0_2_xy_rmse_m-baseline.early0_2_xy_rmse_m
            paired.append(dict(gap_s=length, method=method,
                k_xy_rmse_median_m=float(baseline.early0_2_xy_rmse_m.median()),
                method_xy_rmse_median_m=float(selected.early0_2_xy_rmse_m.median()),
                paired_difference_median_m=float(delta.median()), improved=int(np.sum(delta < -1e-12)),
                unchanged=int(np.sum(np.abs(delta) <= 1e-12)), worse=int(np.sum(delta > 1e-12)),
                worst_deterioration_m=float(delta.max()),
                xy_peak_max_m=float(selected.early0_2_xy_peak_m.max()),
                xy_added_offset_jump_max_m=float(selected.xy_added_offset_jump_max_m.max()),
                return27_xy_added_offset_jump_max_m=float(selected.return27_xy_added_offset_jump_m.max())))
    table = pd.DataFrame(paired)
    table.to_csv(OUT/"paired_summary.csv", index=False)
    figure, axes = plt.subplots(1, 4, figsize=(12.5, 3.7), sharey=True)
    for axis, length in zip(axes, LENGTHS):
        group = frame[frame.gap_s == length]
        for sequence, cases in group.groupby("sequence"):
            values = cases.set_index("method").loc[list(METHODS), "early0_2_xy_rmse_m"]
            axis.plot(np.arange(4), values, color=".7", lw=.8)
            axis.scatter(np.arange(4), values, s=18)
        axis.set_xticks(np.arange(4), ["K", "Always C", "Guarded", "Bounds"], rotation=35)
        axis.set_title(f"{length} s gap")
        axis.set_yscale("log")
        axis.grid(alpha=.2)
    axes[0].set_ylabel("First 2 s XY RMSE (m)")
    figure.suptitle("Six Delft development flights; frozen independent output candidates")
    figure.tight_layout()
    figure.savefig(OUT/"paired_early_xy.png", dpi=170)
    plt.close(figure)
    # Longest-gap traces show all failures and instantaneous bounds effects.
    figure, axes = plt.subplots(2, 3, figsize=(11, 6), sharex=True)
    for axis, file in zip(axes.flat, sorted(OUT.glob("*__gap20.npz"))):
        data = np.load(file)
        selected = (data["time"] >= 25) & (data["time"] < 27)
        for method in ("K", "guarded", "bounds"):
            error = np.linalg.norm(data[method+"_pos"][:, :2]-data["truth"][:, :2], axis=1)
            axis.plot(data["time"][selected]-25, error[selected], label=method)
        axis.set_title(file.stem.split("_twr_")[1].split("__")[0])
        axis.set_yscale("log")
        axis.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    for axis in axes[-1]:
        axis.set_xlabel("Time since radio return (s)")
    for axis in axes[:, 0]:
        axis.set_ylabel("XY error (m)")
    figure.suptitle("20 s gaps: all six flights, including large initial peaks")
    figure.tight_layout()
    figure.savefig(OUT/"gap20_traces.png", dpi=170)
    plt.close(figure)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
