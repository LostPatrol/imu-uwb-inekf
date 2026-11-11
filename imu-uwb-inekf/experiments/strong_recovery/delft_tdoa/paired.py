"""Pair the frozen common translation selector with original Delft trajectories.

Radio outages are physically removed before the shared candidate sees packets.
"""

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import run as native
import adapter

# Only the common frozen selector is imported; no experiment output supplies states.
HERE = Path(__file__).resolve().parent
SHADOW_PATH = HERE.parent/"shadow.py"
BOUNDS_PATH = HERE.parent/"radio_bounds.py"


def shared_module():
    """Import the separately maintained common selector after its source exists."""
    spec = importlib.util.spec_from_file_location("delft_common_shadow",SHADOW_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def bounds_module():
    """Import the second independently frozen radio geometry candidate."""
    spec=importlib.util.spec_from_file_location("delft_radio_bounds",BOUNDS_PATH)
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    return module


def visible_packets(data,length):
    """Remove the blackout's actual packets rather than merely hiding their score."""
    radio=data["radio"]
    return radio[~((radio[:,0] >= 25-length)&(radio[:,0] < 25))]


def invoke(module,rows,data,length):
    """Provide matching native packet signs, antenna origin and radio noise budget."""
    radio=visible_packets(data,length)
    return module.recover(rows[:,0],rows[:,1:4],rows[:,4:7],np.zeros((len(rows),3)),
                          radio[:,0],data["anchors"][radio[:,2].astype(int)],
                          data["anchors"][radio[:,1].astype(int)],radio[:,3],np.sqrt(native.VARIANCE),25.)


def evaluate(output,baseline,calibration,records):
    """Score causal state holds, querying the unchanged K directly after T+2."""
    metrics=[]
    for record in records:
        rows=np.load(output/(record["key"]+".npz"))["rows"]
        base_key=record["key"].rsplit("__",1)[0]
        base=np.load(baseline/(base_key+".npz"))["rows"]
        reference=adapter.truth(record["name"],record["origin"],calibration["lag_s"])
        for left,right in adapter.PROTOCOL["post_score_intervals_s"]:
            # The display function is exactly K at and after 27, even between output timestamps.
            selected=base if left>=2 else rows
            metrics.append({**record,"interval":f"{left}-{right}",**native.score(selected,reference,25+left,25+right)})
    frame=pd.DataFrame(metrics)
    frame.to_csv(output/"metrics.csv",index=False)
    frame.groupby(["queue","outage_s","interval","policy"])[["xy_rmse","z_rmse","coverage"]].agg(["median","max"]).to_csv(output/"summary.csv")
    paired=frame.pivot(index=["name","queue","outage_s","interval"],columns="policy",values="xy_rmse")
    paired["guarded_delta_m"]=paired.guarded_C-paired.K
    paired["always_delta_m"]=paired.always_C-paired.K
    paired["bound_delta_m"]=paired.bound_xy-paired.K
    paired.to_csv(output/"paired_xy.csv")
    (output/"score_provenance.json").write_text(json.dumps(dict(
        state_sampling="100Hz causal ZOH: searchsorted(right)-1; GT interpolation only",
        output_scope="At t>=27 query original K directly, including between recorded samples",
        correction="Replaces initial offline state linear interpolation; no algorithm, calibration or gate changes",
        run_sha256=adapter.digest(HERE/"run.py"),paired_sha256=adapter.digest(HERE/"paired.py")),indent=2),encoding="utf8")


def main():
    """Generate every sensor-only paired output before loading scoring truth."""
    parser=argparse.ArgumentParser()
    parser.add_argument("--baseline",type=Path,default=HERE/"baseline")
    parser.add_argument("--output",type=Path,default=HERE/"paired")
    parser.add_argument("--score-only",action="store_true")
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    calibration=json.loads((HERE/"calibration.json").read_text(encoding="utf8"))
    if args.score_only:
        records=json.loads((args.output/"records.json").read_text(encoding="utf8"))
        evaluate(args.output,args.baseline,calibration,records)
        return
    module=shared_module()
    bound=bounds_module()
    baseline=json.loads((args.baseline/"records.json").read_text(encoding="utf8"))
    records=[]
    cache={}
    for record in baseline:
        if record["outage_s"] == 0:
            continue
        if record["name"] not in cache:
            cache[record["name"]]=adapter.load(record["name"],calibration)
        data=native.queued(cache[record["name"]],record["queue"])
        rows=np.load(args.baseline/(record["key"]+".npz"))["rows"]
        result=invoke(module,rows,data,record["outage_s"])
        radio=visible_packets(data,record["outage_s"])
        bounded=bound.bound_xy(rows[:,0],rows[:,1:4],0.,radio[:,0],
                               data["anchors"][radio[:,2].astype(int)],data["anchors"][radio[:,1].astype(int)],
                               radio[:,3],np.sqrt(native.VARIANCE),25.)
        for policy,p,v,active,diagnostics in [
            ("K",rows[:,1:4],rows[:,4:7],np.zeros(len(rows),dtype=bool),{}),
            ("guarded_C",result["pos"],result["vel"],result["use_c"],result["diagnostics"]),
            ("always_C",result["always_c_pos"],result["always_c_vel"],result["always_c_available"],result["diagnostics"]),
            ("bound_xy",bounded["pos"],rows[:,4:7],bounded["active"],bounded["diagnostics"])]:
            output=rows.copy()
            output[:,1:4],output[:,4:7]=p,v
            key=record["key"]+"__"+policy
            np.savez_compressed(args.output/(key+".npz"),rows=output,use_c=np.asarray(active),
                                diagnostics=json.dumps(diagnostics,default=lambda x:np.asarray(x).tolist()))
            records.append({**record,"key":key,"policy":policy,
                            "selected_samples":int(np.sum(active)),
                            "yaw_max_change":float(np.max(np.abs(output[:,7:]-rows[:,7:]))),
                            "outside_gate_position_max_change":float(np.max(np.abs(output[(rows[:,0]<25)|(rows[:,0]>=27),1:4]-rows[(rows[:,0]<25)|(rows[:,0]>=27),1:4])))})
        (args.output/"records.json").write_text(json.dumps(records,indent=2),encoding="utf8")
        print(record["key"],"selected",int(np.sum(result["use_c"])),flush=True)
    provenance=dict(shared_source_sha256=adapter.digest(SHADOW_PATH),bounds_source_sha256=adapter.digest(BOUNDS_PATH),adapter_sha256=adapter.digest(HERE/"adapter.py"),
                    paired_sha256=adapter.digest(HERE/"paired.py"),baseline_provenance=json.loads((args.baseline/"provenance.json").read_text()),
                    test_truth_used_during_generation=False)
    (args.output/"provenance.json").write_text(json.dumps(provenance,indent=2),encoding="utf8")
    evaluate(args.output,args.baseline,calibration,records)


if __name__ == "__main__":
    main()
