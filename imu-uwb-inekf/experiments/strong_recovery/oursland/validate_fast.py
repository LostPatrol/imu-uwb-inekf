"""Check the accelerated author mirror against the unchanged original class."""

import argparse
import json
import time
import warnings

import audit
import fast_original as fast
import numpy as np


def inputs(data):
    """Create identical first-packet native event indices for the compiled mirror."""
    events=np.unique(np.r_[data["times"],data["radio"][:,0]])
    imu_at,radio_at={},{}
    for index,stamp in enumerate(data["times"]):
        imu_at.setdefault(float(stamp),index)
    for index,stamp in enumerate(data["radio"][:,0]):
        radio_at.setdefault(float(stamp),index)
    return events,np.array([imu_at.get(float(t),-1) for t in events]),np.array([radio_at.get(float(t),-1) for t in events])


def run(name,yaw,full=False):
    """Compare complete rows, final X/P/bias, and all accept/reject counts."""
    data,start,X,P,bias=audit.initial(name,yaw)
    events,ii,ri=inputs(data)
    if full:
        key=f"{name}__yaw{yaw:g}__author_original"
        path=audit.HERE/"diagnostic"/(key+".npz")
        oracle=np.load(path)
        info=json.loads(path.with_suffix(".json").read_text(encoding="utf8"))
        expected_X,expected_P,expected_bias,expected_rows=(oracle[key] for key in ("final_X","final_P","final_bias","rows"))
    else:
        stop=start+1.
        events=events[events<=stop]
        ii,ri=ii[:len(events)],ri[:len(events)]
        expected_X,expected_P,canonical_bias,expected_rows,accepted,rejected,failed=audit.native.replay(
            data,X,P,bias,start,"author",None,False,end=stop,return_final=True)
        expected_bias=np.r_[canonical_bias[3:],canonical_bias[:3]]
        info=dict(accepted_radio=accepted,rejected_radio=rejected,failed=failed)
    started=time.perf_counter()
    result=fast.replay(events,ii,ri,data["imu"],data["radio"],data["anchors"],X,P,np.r_[bias[3:],bias[:3]],start,start)
    FX,FP,Fb,rows,accepted,rejected,predicts,corrections,skipped,failed=result
    values=dict(name=name,yaw_offset_deg=yaw,full_45s=full,row_shape_equal=rows.shape==expected_rows.shape,
        rows_max_abs=None if rows.shape!=expected_rows.shape else float(np.max(np.abs(rows-expected_rows))),
        X_max_abs=float(np.max(np.abs(FX-expected_X))),P_max_abs=float(np.max(np.abs(FP-expected_P))),
        bias_max_abs=float(np.max(np.abs(Fb-expected_bias))),accepted_equal=accepted==info["accepted_radio"],
        rejected_equal=rejected==info["rejected_radio"],failure_equal=bool(failed)==info["failed"],
        fast_runtime_seconds=time.perf_counter()-started,predict_calls=int(predicts),correction_calls=int(corrections))
    values["validated"]=bool(values["row_shape_equal"] and values["rows_max_abs"]<1e-7 and values["P_max_abs"]<1e-7
        and values["bias_max_abs"]<1e-9 and values["accepted_equal"] and values["rejected_equal"] and values["failure_equal"])
    return values


def main():
    """Require reference equivalence before any accelerated mechanism replay."""
    parser=argparse.ArgumentParser()
    parser.add_argument("--full",action="store_true")
    args=parser.parse_args()
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    jobs=[(name,yaw) for name in audit.NAMES for yaw in audit.YAWS] if args.full else [(audit.NAMES[0],180.)]
    records=[run(name,yaw,args.full) for name,yaw in jobs]
    output=audit.HERE/"audit"
    output.mkdir(parents=True,exist_ok=True)
    (output/("fast_full_validation.json" if args.full else "fast_smoke_validation.json")).write_text(json.dumps(records,indent=2),encoding="utf8")
    print(json.dumps(records,indent=2))
    if not all(record["validated"] for record in records):
        raise RuntimeError("Author mirror failed numerical equivalence; cannot use as original")


if __name__=="__main__":
    main()
