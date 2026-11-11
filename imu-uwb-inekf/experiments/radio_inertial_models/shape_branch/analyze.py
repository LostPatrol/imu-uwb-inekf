"""Plot paired physical-shape errors without treating correlated windows as IID."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import physical_shape as shape

# Keep the same method order in tables and plots.
METHODS = ("K1", "K2", "Bias_prior", "BiasTilt_prior", "BiasTilt")
LABELS = ("Profile K1", "Profile K2", "Body bias + prior", "Bias/tilt + prior", "Free bias/tilt")


def run(synthetic, raw, output):
    """Write complete paired comparisons and a compact four-panel visualization."""
    summary = pd.read_csv(synthetic / "synthetic_summary.csv")
    frame = pd.read_csv(synthetic / "synthetic_metrics.csv")
    real = pd.read_csv(raw / "raw_metrics.csv")
    output.mkdir(parents=True, exist_ok=True)
    comparison = []
    for (level, method), part in frame.groupby(["level", "method"]):
        baseline = frame[(frame.level == level) & (frame.method == "K2")].set_index("seed")
        aligned = part.set_index("seed").loc[baseline.index]
        comparison.append(dict(level=level, method=method,
            position_wins_vs_K2=int((aligned.position_error_m < baseline.position_error_m).sum()),
            heading_wins_vs_K2=int((aligned.yaw_error_deg < baseline.yaw_error_deg).sum()),
            position_difference_median_m=float((aligned.position_error_m-baseline.position_error_m).median()),
            heading_difference_median_deg=float((aligned.yaw_error_deg-baseline.yaw_error_deg).median())))
    pd.DataFrame(comparison).to_csv(output / "paired_vs_K2.csv", index=False)
    pooled = real.groupby("method").agg(cases=("position_error_m", "size"),
        position_median_m=("position_error_m", "median"),
        position_p95_m=("position_error_m", lambda x:x.quantile(.95)),
        position_rmse_m=("position_error_m", lambda x:np.sqrt(np.mean(x*x))),
        nonconverged=("success", lambda x:int((~x).sum())))
    pooled.to_csv(output / "raw_pooled_summary.csv")
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    colors = ("#687aa7", "#ce8e36", "#4d9a76", "#8657a0", "#a75757")
    for ax, metric, title in ((axes[0, 0], "position_error_m_p95", "Position p95 (m)"),
                              (axes[0, 1], "yaw_error_deg_p95", "Physical heading p95 (deg)")):
        levels = ("ideal", "mild", "moderate", "bias_only", "tilt_only")
        for offset, method, label, color in zip(np.linspace(-.32, .32, 5), METHODS, LABELS, colors):
            values = summary[summary.method == method].set_index("level").loc[list(levels), metric]
            ax.bar(np.arange(5)+offset, values, width=.15, label=label, color=color)
        ax.set_xticks(range(5), levels)
        ax.set_title(title)
        ax.grid(axis="y", alpha=.2)
    axes[0, 0].legend(fontsize=8)
    names = sorted(real.sequence.unique())
    for method, label, color in zip(METHODS, LABELS, colors):
        values = [real[(real.method == method) & (real.sequence == name)].position_error_m.quantile(.95) for name in names]
        axes[1, 0].plot(range(len(names)), values, ".-", label=label, color=color)
    axes[1, 0].set_xticks(range(len(names)), [name.split("_twr_")[-1].replace(".csv", "") for name in names], rotation=20)
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_title("Raw development logs: position p95 (m), 11 overlapping windows/log")
    axes[1, 0].grid(axis="y", alpha=.2)
    moderate = summary[summary.level == "moderate"].set_index("method").loc[list(METHODS)]
    axes[1, 1].bar(np.arange(5)-.15, moderate.information_median, width=.3, label="Range data only", color="#687aa7")
    axes[1, 1].bar(np.arange(5)+.15, moderate.prior_information_median, width=.3, label="Including calibration prior", color="#ce8e36")
    axes[1, 1].set_xticks(range(5), LABELS, rotation=20, ha="right")
    axes[1, 1].set_title("Moderate: median local yaw information (1/rad^2)")
    axes[1, 1].set_yscale("log")
    axes[1, 1].legend(fontsize=8)
    fig.suptitle(f"Physical IMU shape study: {int(summary.cases.iloc[0])} paired synthetic seeds; raw same-date development windows")
    fig.tight_layout()
    fig.savefig(output / "comparison.png", dpi=170)
    fig.savefig(output / "comparison.svg")
    plt.close(fig)
    (output / "verification.json").write_text(json.dumps(shape.verify(), indent=2), encoding="utf-8")
    metadata = json.loads((synthetic / "protocol.json").read_text(encoding="utf-8"))
    replay_seeds = metadata["case_seeds"][:3]
    replay = pd.DataFrame([row for seed in replay_seeds for row in shape.evaluate(seed, metadata["levels"])])
    saved = frame[frame.seed.isin(replay_seeds)].sort_values(["seed", "level", "method"]).reset_index(drop=True)
    replay = replay.sort_values(["seed", "level", "method"]).reset_index(drop=True)
    columns = [name for name in replay.select_dtypes(include=np.number) if name != "elapsed_s"]
    np.testing.assert_allclose(saved[columns], replay[columns], atol=1e-10, rtol=1e-10)
    checks = dict(replayed_seeds=replay_seeds, replayed_fits=len(replay),
                  maximum_numeric_difference=float(np.max(np.abs(saved[columns].to_numpy()-replay[columns].to_numpy()))),
                  synthetic_all_numeric_finite=bool(np.isfinite(frame.select_dtypes(include=np.number)).all().all()),
                  raw_all_numeric_finite=bool(np.isfinite(real.select_dtypes(include=np.number)).all().all()))
    (output / "replay_audit.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    print(pooled.to_string())


def main():
    """Select existing result folders and generate reviewable comparison artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic", type=Path, default=shape.HERE / "results_paper200")
    parser.add_argument("--raw", type=Path, default=shape.HERE / "results")
    parser.add_argument("--output", type=Path, default=shape.HERE / "analysis")
    args = parser.parse_args()
    run(args.synthetic, args.raw, args.output)


if __name__ == "__main__":
    main()
