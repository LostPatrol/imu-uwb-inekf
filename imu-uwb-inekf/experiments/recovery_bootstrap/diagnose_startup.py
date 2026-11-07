"""Controlled, truth-initialized diagnostics of early range-update transients.

Truth is deliberately used only to isolate initial error in this diagnostic;
these runs are not cold-start benchmarks or publication accuracy comparisons.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import numpy as np
import pandas as pd
from run_research import Invariant15, coordinate_map, PRIOR_STD, write_json

# Orthogonal policy changes distinguish unconditional updates, hard rejection, and Huber weighting.
POLICIES={"warmup256":(256,12.25,2.0),"hard_only":(0,12.25,None),
          "huber_only":(0,1e30,2.0),"hard_huber":(0,12.25,2.0)}
# Inspect a fixed early horizon without conflating this with stable-tail recovery.
HORIZON_S=1.0


def diagnose(job):
    """Replay one controlled initial position/prior/policy and retain each update."""
    cache,name,policy,offset_m,position_std=job
    data=dict(np.load(Path(cache)/(Path(name).stem+".npz")))
    model=json.loads((Path(cache)/"manifest.json").read_text(encoding="utf-8"))["model"]
    state=np.eye(5)
    state[:3,:3]=data["rotations"][0]
    state[:3,4]=data["truth_position"][0]+offset_m*np.array([2,-2,1])/3
    standard=PRIOR_STD.copy()
    standard[6:9]=position_std
    chart=coordinate_map(state)
    warmup,gate,huber=POLICIES[policy]
    tracker=Invariant15(state,chart@np.diag(standard**2)@chart.T,data["bias"],
        range_bias=model["range_bias"],range_sigma=model["range_sigma"],
        gate_threshold=gate,gate_warmup_updates=warmup,huber_threshold=huber)
    events=[]
    errors=[]
    for k,timestamp in enumerate(data["time"]):
        if timestamp>HORIZON_S: break
        if k: tracker.predict(data["imu"][k-1],timestamp-data["time"][k-1])
        indices=np.flatnonzero(data["changed"][k]&(data["ranges"][k]>0))
        for anchor in indices:
            before=tracker.X.copy()
            measured=data["ranges"][k,anchor]-model["range_bias"][anchor]
            true_distance=np.linalg.norm(data["truth_position"][k]-data["anchors"][anchor])
            covariance=tracker.physical_covariance()
            accepted=tracker.correct_twr(data["anchors"][anchor],data["ranges"][k,anchor],int(anchor))
            rotation_delta=tracker.X[:3,:3]@before[:3,:3].T
            angle=np.arccos(np.clip((np.trace(rotation_delta)-1)/2,-1,1))
            error=np.linalg.norm(tracker.X[:3,4]-data["truth_position"][k])
            events.append(dict(sequence=name,policy=policy,offset_m=offset_m,position_std=position_std,
                time_s=timestamp,anchor=int(anchor),raw_range_m=data["ranges"][k,anchor],
                calibrated_range_error_m=measured-true_distance,innovation_m=tracker.last_innovation,
                nis=tracker.last_nis,accepted=accepted,position_injection_m=np.linalg.norm(tracker.X[:3,4]-before[:3,4]),
                attitude_injection_rad=angle,position_error_m=error,prior_position_trace=float(np.trace(covariance[6:9,6:9]))))
        errors.append(np.linalg.norm(tracker.X[:3,4]-data["truth_position"][k]))
    return dict(sequence=name,policy=policy,offset_m=offset_m,position_std=position_std,
                max_first_second_m=max(errors),last_first_second_m=errors[-1]),events


def main():
    """Enumerate all six logs and report every diagnostic condition, including failures."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--workers",type=int,default=8)
    args=parser.parse_args()
    names=json.loads((args.cache/"manifest.json").read_text(encoding="utf-8"))["evaluation"]
    jobs=[(str(args.cache),name,policy,offset,prior) for name in names for policy in POLICIES
          for offset in (0.0,9.0) for prior in (0.2,8.0)]
    rows,events=[],[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for summary,detail in pool.map(diagnose,jobs):
            rows.append(summary); events.extend(detail)
    args.output.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output/"startup_metrics.csv",index=False)
    pd.DataFrame(events).to_csv(args.output/"startup_events.csv",index=False)
    write_json(args.output/"protocol.json",dict(horizon_s=HORIZON_S,policies=POLICIES,
        truth_initialization=True,scope="Causal update diagnostic only; not cold-start evidence",
        conditions=len(rows),events=len(events)))
    print(pd.DataFrame(rows).query("sequence.str.contains('random')",engine="python").to_string(index=False))


if __name__=="__main__":
    main()
