"""Plot all six frozen solver policies using complete evaluation counts and misses."""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# Relative locations are resolved from the project rather than the shell directory.
PROJECT=Path(__file__).resolve().parents[2]
RESULT=PROJECT/"results/recovery_window_candidates_20261107"
FIGURES=PROJECT.parent/"tex/recovery_bootstrap_figures"
# Ordered complete method set; no unsuccessful or slower policy is omitted.
METHODS=("local","profile","profile_two","profile_all","uniform_four","multistart")
LABELS={"local":"Local","profile":"Profile","profile_two":"P2", "profile_all":"Pall",
        "uniform_four":"U4","multistart":"M12"}
COLORS={"local":"#697580","profile":"#0072B2","profile_two":"#56B4E9",
        "profile_all":"#9C6BB0","uniform_four":"#D98200","multistart":"#228B65"}


def main():
    """Render discrete algorithm comparisons without implying a continuous frontier."""
    frame=pd.read_csv(RESULT/"summary.csv")
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8,"axes.labelsize":8,
                         "axes.titlesize":9,"pdf.fonttype":42,"ps.fonttype":42})
    figure,axes=plt.subplots(1,2,figsize=(7.1,2.65),sharex=True,sharey=True)
    for axis,layout,title in zip(axes,("all8","lower4"),("All 8 anchors","Lowest 4 anchors")):
        table=frame[frame.layout==layout].set_index("method")
        for method in METHODS:
            row=table.loc[method]
            x=float(row.total_nfev/row.cases)
            y=float(100*row.misses/row.cases)
            axis.scatter(x,y,s=44 if method=="profile" else 29,c=COLORS[method],
                         marker="D" if method=="profile" else "o",edgecolor="white",linewidth=.6,zorder=3)
            offset=(5,5)
            horizontal="left"
            if method=="multistart":
                offset=(-5,6)
                horizontal="right"
            if method=="uniform_four" and layout=="all8":
                offset=(-5,6)
                horizontal="right"
            axis.annotate(f"{LABELS[method]}: {int(row.misses)}/{int(row.cases)}",(x,y),
                          xytext=offset,textcoords="offset points",ha=horizontal,va="bottom",
                          fontsize=7,fontweight="bold" if method=="profile" else "normal",
                          color=COLORS[method],annotation_clip=False)
        axis.set_title(title+" (264 cases)")
        axis.set_xlabel("Mean total function evaluations")
        axis.set_xlim(0,540)
        axis.set_ylim(-1.2,24)
        axis.set_xticks([0,100,200,300,400,500])
        axis.set_yticks([0,5,10,15,20])
        axis.grid(alpha=.20,zorder=0)
        axis.spines[["top","right"]].set_visible(False)
    axes[0].set_ylabel("Missed best-found objective (%)")
    figure.tight_layout(pad=.8,w_pad=1.3)
    FIGURES.mkdir(parents=True,exist_ok=True)
    figure.savefig(FIGURES/"window_solver_tradeoff.pdf",bbox_inches="tight")
    figure.savefig(FIGURES/"window_solver_tradeoff.png",dpi=240,bbox_inches="tight")
    plt.close(figure)


if __name__=="__main__":
    main()
