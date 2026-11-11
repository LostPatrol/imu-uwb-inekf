"""Report every exploratory state-projection trigger and sustained outcome."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Six flights, four counterfactual outage durations, and three fixed scoring periods.
HERE=Path(__file__).resolve().parent
RESULTS=HERE/"results"
DELFT=HERE.parent


def main():
    """Keep nontriggers, full-period failures and all detrimental pairs visible."""
    records=json.loads((RESULTS/"records.json").read_text(encoding="utf8"))
    frame=pd.read_csv(RESULTS/"metrics.csv")
    pairs=pd.read_csv(RESULTS/"paired_xy.csv")
    triggers=[row["trigger"] for row in records if row["triggered"]]
    trigger_table=[]
    for record in records:
        values={key:record[key] for key in ("name","outage_s","triggered","failed","pretrigger_or_nontrigger_max_difference")}
        if record["triggered"]:
            values.update(record["trigger"])
        trigger_table.append(values)
    pd.DataFrame(trigger_table).to_csv(RESULTS/"triggers.csv",index=False)
    records_frame=pd.DataFrame(records)
    no_trigger_flights=[name for name,group in records_frame.groupby("name") if not group.triggered.any()]
    outcomes=[]
    for (length,interval),group in pairs.groupby(["outage_s","interval"]):
        delta=group.delta_m.dropna()
        outcomes.append(dict(outage_s=int(length),interval=interval,cases=len(group),complete_pairs=len(delta),
            wins=int((delta< -1e-9).sum()),ties=int((np.abs(delta)<=1e-9).sum()),losses=int((delta>1e-9).sum()),
            median_paired_delta_m=float(delta.median()),worst_delta_m=float(delta.max())))
    pd.DataFrame(outcomes).to_csv(RESULTS/"outcomes_by_duration.csv",index=False)
    summary=dict(exploratory=True,traces=len(records),flights=records_frame.name.nunique(),
        triggered=len(triggers),nontriggered=len(records)-len(triggers),
        flights_with_no_trigger_at_any_duration=no_trigger_flights,
        failures=int(records_frame.failed.sum()),minimum_score_coverage=float(frame.coverage.min()),
        nontrigger_or_pretrigger_max_difference=float(records_frame.pretrigger_or_nontrigger_max_difference.max()),
        max_instant_attitude_height_velocity_bias_change=max([v["instant_attitude_height_velocity_bias_max_change"] for v in triggers],default=0.),
        max_physical_covariance_difference=max([v["physical_covariance_max_difference"] for v in triggers],default=0.),
        max_physical_covariance_relative_difference=max([v["physical_covariance_relative_difference"] for v in triggers],default=0.),
        max_snapshot_state_difference=max([v["snapshot_state_max_difference"] for v in triggers],default=0.),
        max_snapshot_covariance_difference=max([v["snapshot_covariance_max_difference"] for v in triggers],default=0.),
        max_shift_m=max([v["shift_m"] for v in triggers],default=0.))
    (RESULTS/"audit.json").write_text(json.dumps(summary,indent=2),encoding="utf8")
    fig,axes=plt.subplots(1,3,figsize=(10.8,3.8),sharex=True,sharey=True)
    finite=pairs[["K","single_state_projection"]].to_numpy(float)
    finite=finite[np.isfinite(finite)&(finite>0)]
    lower=max(.01,float(finite.min())*.7)
    upper=max(1.,float(finite.max())*1.3)
    for ax,interval in zip(axes,["0-2","2-5","6-11"]):
        group=pairs[pairs.interval==interval]
        for length,subset in group.groupby("outage_s"):
            ax.scatter(subset.K,subset.single_state_projection,label=f"L={length:g} s",s=25)
        ax.plot([lower,upper],[lower,upper],"k--",lw=.8)
        ax.set(xscale="log",yscale="log",xlabel="Original K XY RMSE (m)",title=interval+" s after resumption")
        ax.grid(alpha=.2)
    axes[0].set_ylabel("One state projection XY RMSE (m)")
    axes[-1].legend(fontsize=7)
    fig.suptitle("Delft exploratory: single feedback reset, then actual original recursion")
    fig.tight_layout()
    fig.savefig(RESULTS/"paired_xy.png",dpi=160)
    fig.savefig(RESULTS/"paired_xy.svg")
    plt.close(fig)
    print(json.dumps(summary,indent=2),flush=True)
    print(pd.read_csv(RESULTS/"paired_differences.csv").to_string(index=False),flush=True)


if __name__=="__main__":
    main()
