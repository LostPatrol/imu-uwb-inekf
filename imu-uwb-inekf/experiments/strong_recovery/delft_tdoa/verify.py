"""Verify the zero-lever author mirror on real calibration-record packets."""

import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np
import adapter
import run as native

# The shared mirror is read-only and is checked against actual original calls.
HERE=Path(__file__).resolve().parent
FAST_PATH=HERE.parent/"oursland"/"fast_original.py"
os.environ["NUMBA_CACHE_DIR"]=str(HERE/"numba_cache")
sys.path.insert(0,str(FAST_PATH.parent))
# Numba's persisted environment must resolve the mirror's canonical module name.
import fast_original as FAST


def compare(data,queue):
    """Compare every predict and correct state, covariance, bias and gate."""
    start,state,bias=adapter.initial_state(data)
    estimator=native.make_filter(state,native.P0,bias)
    X,P,b=state.copy(),native.P0.copy(),bias.copy()
    maximum=np.zeros(3)
    calls=0
    accepted_equal=True
    j=np.searchsorted(data["radio"][:,0],start,side="right")
    previous=start
    def inspect():
        """Measure numerical differences after the same real transaction."""
        maximum[:]=np.maximum(maximum,[np.max(np.abs(X-estimator.X)),np.max(np.abs(P-estimator.P)),np.max(np.abs(b-estimator.bias))])
    for k in range(np.searchsorted(data["times"],start,side="right"),len(data["times"])):
        stamp=data["times"][k]
        if stamp>5:
            break
        u=data["imu"][k-1]
        estimator.predict(u,stamp-previous)
        X,P=FAST.predict(X,P,b,u,stamp-previous)
        inspect()
        corrected=False
        while j<len(data["radio"]) and data["radio"][j,0]<=stamp+1e-10:
            if corrected:
                estimator.predict(u,0.)
                X,P=FAST.predict(X,P,b,u,0.)
                inspect()
            packet=data["radio"][j,1:]
            _,_,_,discard=estimator.correct(packet,data["anchors"])
            X,P,b,keep=FAST.correction(X,P,b,packet,data["anchors"],.13,0,0,np.zeros(3))
            accepted_equal &= bool(keep)==bool(not discard)
            inspect()
            corrected=True
            calls+=1
            j+=1
        previous=stamp
    return dict(queue=queue,real_correct_calls=calls,max_state_difference=float(maximum[0]),
                max_covariance_difference=float(maximum[1]),max_bias_difference=float(maximum[2]),
                all_gate_decisions_equal=accepted_equal,passed=bool(maximum.max()<1e-10 and accepted_equal))


def main():
    """Keep calibration-side physical replay validation separate from test scoring."""
    calibration=json.loads((HERE/"calibration.json").read_text(encoding="utf8"))
    data=adapter.load(adapter.PROTOCOL["calibration_names"][0],calibration)
    rows=[compare(native.queued(data,q),q) for q in adapter.PROTOCOL["queues"]]
    result=dict(record=data["name"],uses_test_data=False,rows=rows,
                original_source_sha256=adapter.digest(native.AUTHOR_PATH),mirror_source_sha256=adapter.digest(FAST_PATH))
    (HERE/"verification.json").write_text(json.dumps(result,indent=2),encoding="utf8")
    print(json.dumps(result,indent=2),flush=True)
    if not all(row["passed"] for row in rows):
        raise RuntimeError("Delft zero-lever native mirror validation failed")


if __name__=="__main__":
    main()
