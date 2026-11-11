"""Plot fixed-window capture accuracy and subsequent real-flight tracking."""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    """Visualize all matched policies for both gyro-calibrated model variants."""
    root = Path(__file__).parent
    policies = ["zero_yaw", "raw_local", "projected_single", "uniform8"]
    labels = ["Fixed zero yaw", "Raw local (zero start)", "Projected single candidate", "Uniform 8 starts"]
    for folder in ["gyro_calibrated", "full_attitude"]:
        frame = pd.read_csv(root / folder / "metrics.csv")
        names = frame.name.unique()
        fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
        fields = ["endpoint_yaw_error_deg", "yaw_rmse_deg", "position_rmse"]
        ylabels = ["Absolute endpoint yaw error (deg)", "Post-handoff yaw RMSE (deg)", "Post-handoff position RMSE (m)"]
        for axis, field, ylabel in zip(axes, fields, ylabels):
            for index, (policy, label) in enumerate(zip(policies, labels)):
                values = frame[frame.policy == policy].set_index("name").reindex(names)[field]
                axis.bar(np.arange(len(names)) + (index - 1.5) * .19, np.abs(values), width=.19, label=label)
            axis.set_ylabel(ylabel)
            axis.grid(axis="y", alpha=.25)
        axes[0].legend(ncol=2, fontsize=8)
        axes[-1].set_xticks(np.arange(len(names)), [name.replace("-trial", "\ntrial") for name in names], fontsize=9)
        fig.suptitle(f"Real fixed 10-16 s capture / 16-45 s tracking: {folder}")
        fig.tight_layout()
        fig.savefig(root / folder / "comparison.png", dpi=160)
        plt.close(fig)


if __name__ == "__main__":
    main()
