"""Summarize cold full-attitude acquisition and expose nonobservable regimes."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Keep seed policies and their plotting colors fixed across geometry panels.
METHODS = ("single_identity", "moving_gravity", "local_gravity", "radio_wahba", "cube24")
LABELS = ("Identity seed", "Body mean gravity", "Gyro-compensated gravity", "Radio path + Wahba", "24 cube rotations")
COLORS = ("#6c80ac", "#cb8e38", "#807247", "#2b9572", "#915d9e")


def analyze(source):
    """Write descriptive summaries and paired differences against the finite multi-start."""
    frame = pd.read_csv(source / "metrics.csv")
    aggregate = frame.groupby(["motion", "geometry", "method"]).agg(
        cases=("acquired", "size"), acquired=("acquired", "sum"),
        rotation_median_deg=("rotation_error_deg", "median"),
        rotation_p95_deg=("rotation_error_deg", lambda x:x.quantile(.95)),
        position_median_m=("position_error_m", "median"),
        position_p95_m=("position_error_m", lambda x:x.quantile(.95)),
        cost_misses=("cost_miss", "sum"), nonconverged=("success", lambda x:int((~x).sum())),
        nfev_median=("nfev", "median"), refine_median_s=("refine_seconds", "median"))
    aggregate.to_csv(source / "aggregate.csv")
    keys = ["seed", "motion", "geometry", "tilt_deg"]
    reference = frame[frame.method == "cube24"].set_index(keys)
    paired = []
    for method in METHODS[:-1]:
        part = frame[frame.method == method].set_index(keys).loc[reference.index]
        delta = part.cost-reference.cost
        tolerance = np.maximum(.001, 1e-5*np.abs(reference.cost))
        for motion, geometry in reference.reset_index()[["motion", "geometry"]].drop_duplicates().itertuples(index=False):
            mask = ((reference.index.get_level_values("motion") == motion)
                    & (reference.index.get_level_values("geometry") == geometry))
            paired.append(dict(method=method, motion=motion, geometry=geometry, cases=int(mask.sum()),
                objective_worse_than_cube24=int((delta[mask] > tolerance[mask]).sum()),
                objective_better_than_cube24=int((delta[mask] < -tolerance[mask]).sum()),
                rotation_difference_median_deg=float((part.rotation_error_deg-reference.rotation_error_deg)[mask].median()),
                position_difference_median_m=float((part.position_error_m-reference.position_error_m)[mask].median())))
    pd.DataFrame(paired).to_csv(source / "paired_vs_cube24.csv", index=False)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    tilts = sorted(frame.tilt_deg.unique())
    for axis, geometry, title in ((axes[0,0], "normal", "Rich excitation, 3 m anchor height span"),
                                   (axes[0,1], "weak_height", "Rich excitation, 0.1 m anchor height span")):
        for method, label, color in zip(METHODS, LABELS, COLORS):
            part = frame[(frame.method == method) & (frame.motion == "rich") & (frame.geometry == geometry)]
            values = part.groupby("tilt_deg").acquired.mean().reindex(tilts)
            axis.plot(tilts, values, "x--" if method == "cube24" else ".-", label=label, color=color)
        axis.set_xticks(tilts)
        axis.set_ylim(-.03, 1.05)
        axis.set_title(title)
        axis.set_xlabel("True initial tilt (deg); unknown to all methods")
        axis.set_ylabel("SO(3) error <= 10 deg, fraction")
        axis.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    for method, label, color in zip(METHODS, LABELS, COLORS):
        part = frame[(frame.method == method) & (frame.motion == "weak")]
        values = part.groupby("tilt_deg").rotation_error_deg.median().reindex(tilts)
        axes[1,0].plot(tilts, values, "x--" if method == "cube24" else ".-", label=label, color=color)
    axes[1,0].set_xticks(tilts)
    axes[1,0].set_xlabel("True initial tilt (deg)")
    axes[1,0].set_ylabel("Median full SO(3) error (deg)")
    axes[1,0].set_title("Weak excitation: both geometries, all failures retained")
    axes[1,0].grid(alpha=.2)
    truth_info = frame.drop_duplicates(keys)
    motions = ("rich", "weak", "constant_acceleration", "constant_velocity")
    for offset, geometry, color in ((-.15, "normal", "#2b9572"), (.15, "weak_height", "#cb8e38")):
        medians = [truth_info[(truth_info.motion == motion) & (truth_info.geometry == geometry)].rotation_info_min.median() for motion in motions]
        axes[1,1].bar(np.arange(4)+offset, np.maximum(medians, 1e-10), width=.3, color=color, label=geometry)
    axes[1,1].set_xticks(range(4), ["Rich", "Weak", "Constant acc.", "Constant vel."], rotation=15)
    axes[1,1].set_yscale("log")
    axes[1,1].set_title("Truth-point rotation information after eliminating p/v")
    axes[1,1].set_ylabel("Smallest eigenvalue (1/rad^2); floor 1e-10")
    axes[1,1].legend(fontsize=8)
    fig.suptitle(f"Moving-window full-attitude simulation: {len(truth_info)} paired attitude conditions, no static prefix")
    fig.tight_layout()
    fig.savefig(source / "cold_rotation.png", dpi=170)
    fig.savefig(source / "cold_rotation.svg")
    plt.close(fig)
    print(aggregate.to_string())


def main():
    """Read completed benchmark output and create reviewable figures and comparisons."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args()
    analyze(args.source)


if __name__ == "__main__":
    main()
