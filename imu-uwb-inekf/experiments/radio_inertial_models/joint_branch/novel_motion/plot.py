"""Render saved calibration-transfer endpoint errors with compact axis labels."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# Keep this rendering-only pass independent of fitting and calibration.
OUTPUT=Path(__file__).resolve().parent/"results"


def main():
    """Plot common-seed five-versus-six calibration effects from saved metrics."""
    frame=pd.read_csv(OUTPUT/"metrics.csv")
    paired=frame.pivot(index=["sequence","endpoint_s","duration_s","layout"],
        columns=["calibration_scope","radio_model"],values="position_error_m")
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout="constrained")
    for ax,(layout,part) in zip(axes,paired.groupby(level="layout")):
        for model in ("static","geometry"):
            ax.scatter(part[("six_all_motions",model)],part[("five_other_motions",model)],
                s=18,alpha=.7,label=model)
        ax.plot([.02,3],[.02,3],color="black",linewidth=.8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(layout)
        ax.grid(alpha=.2)
    axes[0].legend()
    fig.supxlabel("Six-motion calibration endpoint error (m)")
    fig.supylabel("Five-motion calibration endpoint error (m)")
    fig.savefig(OUTPUT/"five_vs_six.png",dpi=180,bbox_inches="tight")
    plt.close(fig)


if __name__=="__main__":
    main()
