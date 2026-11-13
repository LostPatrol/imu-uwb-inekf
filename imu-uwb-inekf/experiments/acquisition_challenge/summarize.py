"""Produce paired recovery matrices and a flight-level cold-acquisition figure."""

import argparse
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Short display labels denote implementations, with provenance documented in the report.
LABELS = {"body_bias_prior": "Window (body bias)", "profile1": "Window (affine)",
          "SFUISE": "SFUISE", "Jiang_independent": "Jiang (indep.)",
          "Pfeiffer_EKF": "Pfeiffer EKF", "Pfeiffer_MHE": "Pfeiffer MHE"}
ORDER = ["SFUISE", "Jiang_independent", "Pfeiffer_EKF", "Pfeiffer_MHE", "body_bias_prior", "profile1"]
HERE = Path(__file__).resolve().parent


def recovery_matrix(scores, snapshots):
    """Join distinct treatment types while preserving duplicate-confidence and deadline meanings."""
    sensors = scores[(scores.clock == "sensor") & (scores.kind == "recovery")]
    conditioned = sensors[sensors.protocol.isin(["seed_trajectory", "seed_trajectory15",
                 "conditional_seed_or_prior", "recovery_MAP_trajectory", "graph_MAP_trajectory"])]
    conditioned = conditioned[conditioned.case_id != "native"].sort_values("coverage_s")
    conditioned = conditioned.drop_duplicates(["episode", "method", "case_id"], keep="last").copy()
    conditioned = conditioned.rename(columns={"xy_10s_m": "xy_m", "zabs_10s_m": "zabs_m",
                                                "yaw_10s_deg": "yaw_deg"})
    sub = snapshots[(snapshots.state_s == 10) & (snapshots.case_id != "native")].copy()
    sub["dataset"] = sub.episode.str.split("_").str[0]
    sub["protocol"] = "snapshot"
    sub["terminal_failure"] = False
    # The ten-second graph has one authentic endpoint even before a continuation is generated.
    matched = conditioned.set_index(["episode", "method", "case_id"]).index
    sub = sub[~sub.set_index(["episode", "method", "case_id"]).index.isin(matched)]
    frame = pd.concat([conditioned, sub], ignore_index=True)
    frame["role"] = frame.method.map({"SFUISE": "free_initial_iterate",
        "Pfeiffer_MHE": "position_seed_only", "Pfeiffer_EKF": "filter_covariance",
        "body_bias_prior": "Gaussian_MAP", "profile1": "Gaussian_MAP",
        "Jiang_independent": "Gaussian_MAP"})
    frame["finite10"] = np.isfinite(frame.xy_m)
    frame["within10"] = frame.xy_m <= .5
    frame["verified15"] = frame.confirm_050_s.notna() & (frame.coverage_s >= 15)
    frame["has15_protocol"] = frame.coverage_s >= 15
    frame["stayed15"] = frame.stayed_after_confirm_050.eq(True) & frame.has15_protocol
    return frame


def aggregate(frame, fields):
    """Summarize computed cases, threshold successes on all, and conditional finite-error medians."""
    return frame.groupby(fields, as_index=False, dropna=False).agg(cases=("case_id", "size"),
        independent_recordings=("episode", "nunique"), finite_at_10=("finite10", "sum"),
        within_05m_at_10=("within10", "sum"), terminal_failures=("terminal_failure", "sum"),
        median_finite_xy10_m=("xy_m", "median"), p90_finite_xy10_m=("xy_m", lambda s: s.quantile(.9)),
        median_finite_zabs10_m=("zabs_m", "median"), median_finite_yaw10_deg=("yaw_deg", "median"),
        trajectories_15s=("has15_protocol", "sum"), confirmations_by_15s=("verified15", "sum"),
        stayed_after_confirmation_to_15s=("stayed15", "sum"),
        median_confirm_if_observed_s=("confirm_050_s", "median"))


def pair_cold(scores):
    """Count flight-paired wins, losses, ties and median differences without treating seeds as flights."""
    records = []
    cold = scores[(scores.kind == "start") & (scores.protocol == "native")]
    for (dataset, clock), group in cold.groupby(["dataset", "clock"]):
        for left, right in combinations(group.method.unique(), 2):
            w = group[group.method == left].set_index("episode")
            v = group[group.method == right].set_index("episode")
            for variable in ("confirm_050_s", "xy_2s_m", "xy_10s_m", "xy_15s_m"):
                delta = (w[variable] - v[variable]).dropna()
                records.append(dict(dataset=dataset, clock=clock, method_a=left, method_b=right,
                    metric=variable, paired_flights=len(delta), lower_a=(delta < -1e-8).sum(),
                    lower_b=(delta > 1e-8).sum(), ties=(abs(delta) <= 1e-8).sum(),
                    median_a_minus_b=delta.median()))
    return pd.DataFrame(records)


def cold_figure(scores, destination):
    """Display individual acquisition times and ten-second XY, stratified by real site."""
    group = scores[(scores.kind == "start") & (scores.clock == "sensor") & (scores.protocol == "native")]
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 5.6), layout="constrained",
                             gridspec_kw={"width_ratios": [1, 1.05]})
    for r, (site, title) in enumerate((("delft", "Delft flights"), ("isas", "ISAS walks"))):
        data = group[group.dataset == site]
        methods = [m for m in ORDER if m in set(data.method)]
        for j, (key, heading) in enumerate((("confirm_050_s", "Time to confirm usable XY (s)"),
                                            ("xy_10s_m", "Horizontal error at 10 s (m)"))):
            ax = axes[r, j]
            for i, method in enumerate(methods):
                values = data[data.method == method].sort_values("episode")[key].dropna().values
                if not len(values):
                    continue
                offset = np.linspace(-.15, .15, len(values)) if len(values) > 1 else [0.]
                c = "#a63e24" if method == "body_bias_prior" else "#216185"
                ax.scatter(values, i + np.asarray(offset), s=15, alpha=.75, color=c, zorder=3)
                ax.plot([np.median(values), np.median(values)], [i-.25, i+.25], color=c, lw=2.4)
            ax.set_yticks(range(len(methods)), [LABELS[m] for m in methods] if j == 0 else [])
            ax.invert_yaxis()
            ax.set_axisbelow(True)
            ax.grid(axis="x", color="#dde2e6")
            ax.spines[["top", "right", "left"]].set_visible(False)
            ax.tick_params(axis="y", length=0)
            ax.set_title(f"{title}   |   {heading}", fontsize=10, loc="left")
            if j == 0:
                ax.set_xlim(0., 15.)
            else:
                ax.set_xscale("log")
                ax.axvline(.5, color="#777777", ls=":", lw=1)
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def main():
    """Build summaries without using numbers rounded for the manuscript or report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=HERE / "results")
    args = parser.parse_args()
    path = args.results
    scores = pd.read_csv(path / "acquisition_scores.csv")
    snapshots = pd.read_csv(path / "snapshot_scores.csv")
    frame = recovery_matrix(scores, snapshots)
    cols = ["dataset", "method", "role", "confidence"]
    aggregate(frame, cols).to_csv(path / "recovery_summary.csv", index=False, float_format="%.8g")
    aggregate(frame, cols + ["magnitude_m", "yaw_delta_deg"]).to_csv(
        path / "recovery_by_treatment.csv", index=False, float_format="%.8g")
    aggregate(frame, ["episode", "method", "role", "confidence"]).to_csv(
        path / "recovery_by_recording.csv", index=False, float_format="%.8g")
    pair_cold(scores).to_csv(path / "cold_paired.csv", index=False, float_format="%.8g")
    cold_figure(scores, path / "cold_acquisition.png")


if __name__ == "__main__":
    main()
