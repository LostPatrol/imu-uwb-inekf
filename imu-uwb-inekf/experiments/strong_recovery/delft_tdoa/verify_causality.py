"""Poison hidden and future real Delft packets to verify causal adaptation."""

import json
from pathlib import Path

import numpy as np
import adapter
import paired
import run as native

# Fixed first held-out recording and L=10 are audit cases, not gate tuning cases.
HERE=Path(__file__).resolve().parent


def run_bounds(module,rows,data):
    """Apply the shared geometric bound to the same physically visible packets."""
    radio=paired.visible_packets(data,10.)
    return module.bound_xy(rows[:,0],rows[:,1:4],0.,radio[:,0],
                           data["anchors"][radio[:,2].astype(int)],data["anchors"][radio[:,1].astype(int)],
                           radio[:,3],np.sqrt(.13),25.)


def main():
    """Verify past outputs do not depend on unavailable radio or future K states."""
    calibration=json.loads((HERE/"calibration.json").read_text(encoding="utf8"))
    name=adapter.PROTOCOL["test_names"][0]
    data=native.queued(adapter.load(name,calibration),"first_per_stamp")
    key=name+"__first_per_stamp__L10"
    rows=np.load(HERE/"baseline"/(key+".npz"))["rows"]
    shadow,bounds=paired.shared_module(),paired.bounds_module()
    original=paired.invoke(shadow,rows,data,10.)
    bound_original=run_bounds(bounds,rows,data)
    poisoned=dict(data)
    poisoned["radio"]=data["radio"].copy()
    dark=(poisoned["radio"][:,0]>=15)&(poisoned["radio"][:,0]<25)
    poisoned["radio"][dark,3]+=1e6
    replay,_=native.replay(poisoned,"first_per_stamp",(15.,25.))
    masked_state_error=float(np.max(np.abs(rows-replay)))
    dark_shadow=paired.invoke(shadow,rows,poisoned,10.)
    dark_bound=run_bounds(bounds,rows,poisoned)
    results=[dict(case="masked_radio",native_max_difference=masked_state_error,
                  shadow_max_difference=float(np.max(np.abs(dark_shadow["pos"]-original["pos"]))),
                  bound_max_difference=float(np.max(np.abs(dark_bound["pos"]-bound_original["pos"]))))]
    for cut in (26.,27.):
        changed=dict(data)
        changed["radio"]=data["radio"].copy()
        changed["radio"][changed["radio"][:,0]>=cut,3]+=1e6
        changed_rows=rows.copy()
        changed_rows[changed_rows[:,0]>=cut,1:7]+=1e5
        result=paired.invoke(shadow,changed_rows,changed,10.)
        bounded=run_bounds(bounds,changed_rows,changed)
        past=rows[:,0]<cut
        results.append(dict(case=f"future_at_{cut:g}",
                            shadow_max_difference=float(np.max(np.abs(result["pos"][past]-original["pos"][past]))),
                            bound_max_difference=float(np.max(np.abs(bounded["pos"][past]-bound_original["pos"][past])))))
    passed=all(value==0 for result in results for key,value in result.items() if key.endswith("difference"))
    # This synthetic scorer check is plumbing validation, not a physical performance experiment.
    score_rows=np.zeros((2,16))
    score_rows[:,0]=[.99,1.01]
    score_rows[1,1]=100.
    score_truth=np.array([[.9,0.,0.,0.],[1.1,0.,0.,0.]])
    scorer_error=native.score(score_rows,score_truth,1.,1.01)["xy_rmse"]
    passed &= scorer_error==0
    output=dict(passed=passed,record=name,uses_truth=False,results=results,
                future_state_scoring_sentinel_xy_rmse=scorer_error,
                shadow_sha256=adapter.digest(paired.SHADOW_PATH),bounds_sha256=adapter.digest(paired.BOUNDS_PATH))
    (HERE/"causality.json").write_text(json.dumps(output,indent=2),encoding="utf8")
    print(json.dumps(output,indent=2),flush=True)
    if not passed:
        raise RuntimeError("Hidden or future information changed a past output")


if __name__=="__main__":
    main()
