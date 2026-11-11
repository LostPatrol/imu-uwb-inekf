"""Consolidate all matched deferred handoff cases without selecting triggered runs."""

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import run

# Policy order is fixed across tables and plots.
POLICIES=run.POLICIES
# Every recording contributes to its scenario/pressure panel, including rejections.
COLORS=("#555555","#187f53","#c17b18","#ad3d4b")


def main():
    """Save paired errors, trigger counts, mechanism checks and complete-case plots."""
    output=run.HERE/"results"
    records=json.loads((output/"records.json").read_text(encoding="utf8"))
    metrics=pd.read_csv(output/"metrics.csv")
    assert len(records)==128 and len(metrics)==128 and metrics["key"].nunique()==128
    keys=["name","scenario","yaw_offset_deg"]
    paired=metrics.pivot(index=keys,columns="policy",values="future_xy_rmse").reset_index()
    first=metrics[metrics.policy=="first_pass"]
    paired=paired.merge(first[keys+["first_accept_s","late_accept","failed"]],on=keys)
    for policy in POLICIES[1:]:
        paired[policy+"_minus_retain_m"]=paired[policy]-paired["retain"]
    paired.to_csv(output/"paired_future.csv",index=False)
    event_scores=pd.read_csv(output/"event_scoring.csv")
    accepted=event_scores[(event_scores.policy=="first_pass")&event_scores.accepted]
    accepted.merge(paired,on=keys).to_csv(output/"accepted_cases.csv",index=False)
    counts=first.groupby(["scenario","yaw_offset_deg"]).agg(
        cases=("key","size"),accepted=("first_accept_s","count"),late_accepts=("late_accept","sum"))
    counts.to_csv(output/"acceptance.csv")
    events=[dict(key=record["key"],name=record["name"],scenario=record["scenario"],
        yaw_offset_deg=record["yaw_offset_deg"],policy=record["policy"],**event)
        for record in records for event in record["events"]]
    changes=[event["change"] for event in events if event["applied"]]
    clock_mismatches=[]
    for _,group in metrics.groupby(keys):
        reference=next(record for record in records if record["key"]==group[group.policy=="retain"].iloc[0]["key"])
        clock=np.load(reference["artifact"])["rows"][:,0]
        for _,metric in group.iterrows():
            if metric.failed:
                continue
            record=next(record for record in records if record["key"]==metric["key"])
            actual=np.load(record["artifact"])["rows"][:,0]
            if not np.array_equal(clock,actual):
                clock_mismatches.append(metric["key"])
    checks=dict(trajectories=len(records),base_cases=len(paired),
        unique_keys=metrics["key"].nunique(),failures=int(metrics.failed.sum()),
        incomplete_future=int(np.sum(metrics.future_output_fraction<.99)),
        first_pass_accepts=int(first.first_accept_s.notna().sum()),
        first_pass_late_accepts=int(first.late_accept.sum()),
        first_pass_future_xy_wins=int(np.sum(paired.first_pass_minus_retain_m < -1e-10)),
        first_pass_future_xy_losses=int(np.sum(paired.first_pass_minus_retain_m > 1e-10)),
        first_pass_future_xy_ties=int(np.sum(np.abs(paired.first_pass_minus_retain_m)<=1e-10)),
        failed_keys=metrics[metrics.failed]["key"].tolist(),
        nonfailed_output_clock_mismatches=clock_mismatches,
        maximum_handoff_changes={key:max(change[key] for change in changes) for key in changes[0]},
        candidate_packets_after_decision=sum(event["audit"]["candidate_latest_time"]>
            next(record["origin"] for record in records if record["key"]==event["key"])+event["time_s"]+1e-9
            for event in events),
        source_hash_matches={path:hashlib.sha256(Path(path).read_bytes()).hexdigest()==value
            for path,value in json.loads((output/"protocol.json").read_text())["source_sha256"].items()})
    provenance=output/"source_provenance.json"
    if provenance.exists():
        checks["imported_util_source_provenance"]=json.loads(provenance.read_text(encoding="utf8"))
    run.save_json(output/"mechanism_audit.json",checks)
    # Every baseline/candidate pair is displayed, without selecting acceptance cases.
    figure,axes=plt.subplots(2,2,figsize=(13,8),sharey=True)
    for row,scenario in enumerate(run.SCENARIOS):
        for column,yaw in enumerate(run.YAWS):
            axis=axes[row,column]
            subset=paired[(paired.scenario==scenario)&(paired.yaw_offset_deg==yaw)].sort_values("name")
            positions=np.arange(len(subset))
            for number,(policy,color) in enumerate(zip(POLICIES,COLORS)):
                axis.scatter(positions+(number-1.5)*.15,subset[policy],label=policy,color=color,s=28)
                failed=metrics[(metrics.scenario==scenario)&(metrics.yaw_offset_deg==yaw)&
                               (metrics.policy==policy)&metrics.failed]
                for _,failure in failed.iterrows():
                    index=int(np.flatnonzero(subset.name.to_numpy()==failure["name"])[0])
                    x=positions[index]+(number-1.5)*.15
                    axis.scatter([x],[failure.future_xy_rmse],marker="x",s=55,color="black")
                    axis.annotate(f"failed; {failure.future_output_fraction:.0%} future",(x,failure.future_xy_rmse),
                                  xytext=(0,-13),textcoords="offset points",ha="center",fontsize=7)
            labels=[name.replace("-trial6", "").replace("-traj1", "").replace("const","g")
                    for name in subset.name]
            axis.set_xticks(positions,labels,rotation=35,ha="right")
            axis.set_title(f"{scenario}, initial yaw stress {yaw:g} degrees")
            axis.grid(alpha=.2)
            axis.set_yscale("log")
            axis.set_ylabel("Common future XY RMSE (m)")
    axes[0,0].legend(fontsize=8)
    figure.suptitle("All 32 matched cases; fixed future begins at 14 s / 34 s")
    figure.tight_layout()
    figure.savefig(output/"paired_future.png",dpi=160)
    plt.close(figure)
    # Rows are predetermined recording/scenario/pressure cases, never sorted by performance.
    ordered=sorted((record for record in records if record["policy"]=="first_pass"),
        key=lambda record:(record["scenario"],record["yaw_offset_deg"],record["name"]))
    screen=np.full((len(ordered),4),np.nan)
    labels=[]
    for index,record in enumerate(ordered):
        labels.append(f"{record['scenario']} {record['yaw_offset_deg']:g} {record['name'].replace('-trial6','')}")
        for event in record["events"]:
            screen[index,event["attempt"]-1]=int(event["decision"]["accepted"])
    figure,axis=plt.subplots(figsize=(9,10))
    image=axis.imshow(screen,aspect="auto",vmin=0,vmax=1,cmap=matplotlib.colors.ListedColormap(["#ddd7d2","#238f55"]))
    axis.set_yticks(np.arange(len(labels)),labels,fontsize=7)
    axis.set_xticks(np.arange(4),["2 / 22 s","6 / 26 s","10 / 30 s","14 / 34 s"])
    axis.set_title("Frozen screen: gray reject, green accept, blank already handed off")
    figure.tight_layout()
    figure.savefig(output/"deferred_decisions.png",dpi=160)
    plt.close(figure)
    print(counts.to_string())
    print(pd.read_csv(output/"summary.csv").to_string(index=False))
    print(json.dumps(checks,indent=2))


if __name__=="__main__":
    main()
