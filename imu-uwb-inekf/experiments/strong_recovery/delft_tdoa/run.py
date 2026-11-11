"""Replay untouched author predict/correct on genuine Delft radio packets.

The adapter makes timestamp multiplicity explicit and scores truth only last.
"""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import warnings

# Limit BLAS overhead for the author's small dense matrix exponentials.
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"
sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import adapter

# The original implementation is imported directly and never edited.
HERE = Path(__file__).resolve().parent
AUTHOR_PATH = HERE.parents[3]/"imu-uwb-inekf - backup"/"scripts"/"estimation"/"inekf.py"
if not AUTHOR_PATH.exists():
    AUTHOR_PATH = Path("E:/0syncthing/Code/InEKF/imu-uwb-inekf - backup/scripts/estimation/inekf.py")
SPEC = importlib.util.spec_from_file_location("delft_untouched_author", AUTHOR_PATH)
AUTHOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUTHOR)
# Original main covariance and radio variance remain fixed for the new field.
P0 = np.diag(np.array([.01,.01,1.,.01,.01,.01,.1,.1,.1,.0001,.0001,.0001,.000001,.000001,.000001])**2)
VARIANCE = .13
THREAD_LIMIT = threadpool_limits(limits=1)


def queued(data, queue):
    """Fix the offered packet budget before initialization or any method sees it."""
    result = dict(data)
    if queue == "first_per_stamp":
        _, indices = np.unique(data["radio"][:,0],return_index=True)
        result["radio"] = data["radio"][indices]
        result["channel"] = data["channel"][indices]
    return result


def make_filter(state, covariance, bias):
    """Configure only external Delft calibration on the original author object."""
    estimator = AUTHOR.InEKF(state.copy(), covariance.copy(), bias.copy())
    estimator.t_uv = np.zeros(3)
    estimator.w_tdoa = np.array([np.sqrt(VARIANCE)])
    return estimator


def replay(data, queue, outage=None):
    """Use held IMU prediction and new radio packets at their actual CSV times."""
    start, state, bias = adapter.initial_state(data)
    estimator = make_filter(state,P0,bias)
    times, imu, radio = (data[key] for key in ("times","imu","radio"))
    output, accepted, rejected, skipped, zero_predict = [],0,0,0,0
    radio_index = np.searchsorted(radio[:,0],start,side="right")
    begin = np.searchsorted(times,start,side="right")
    previous = start
    failure = None
    started = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore",DeprecationWarning)
        try:
            for k in range(begin,len(times)):
                stamp = times[k]
                estimator.predict(imu[k-1],float(stamp-previous))
                corrected = False
                while radio_index < len(radio) and radio[radio_index,0] <= stamp+1e-10:
                    packet = radio[radio_index]
                    if outage is not None and outage[0] <= packet[0] < outage[1]:
                        skipped += 1
                    else:
                        # Native correct keeps stale prediction snapshots; dt=0 explicitly refreshes extra packets.
                        if corrected:
                            estimator.predict(imu[k-1],0.)
                            zero_predict += 1
                        _,_,_,discard = estimator.correct(packet[1:],data["anchors"])
                        accepted += int(not discard)
                        rejected += int(discard)
                        corrected = True
                    radio_index += 1
                output.append(np.r_[stamp,estimator.X[:3,4],estimator.X[:3,3],estimator.X[:3,:3].ravel()])
                previous = stamp
                if not np.isfinite(estimator.X).all() or not np.isfinite(estimator.P).all():
                    failure = "nonfinite_state_or_covariance"
                    break
                if np.max(np.abs(estimator.X[:3,4])) > 1000:
                    failure = "position_exceeds_1000m"
                    break
        except Exception as exception:
            failure = type(exception).__name__+": "+str(exception)
    info = dict(failed=failure is not None,failure=failure,accepted_radio=accepted,
                rejected_radio=rejected,outage_skipped=skipped,zero_dt_predict_calls=zero_predict,
                initial_wait_s=start,initial_position=state[:3,4].tolist(),initial_velocity=state[:3,3].tolist(),
                runtime_s=time.perf_counter()-started,last_time=float(previous),offered_radio=len(radio))
    return np.asarray(output).reshape(-1,16),info


def score(rows, reference, start, end):
    """Score absolute horizontal radial and height error on the fixed 100Hz grid."""
    grid = np.arange(start,end-1e-9,.01)
    valid = (grid >= reference[0,0]) & (grid <= reference[-1,0])
    valid &= (grid >= rows[0,0]) & (grid <= rows[-1,0]) if len(rows) else False
    grid = grid[valid]
    if not len(grid):
        return dict(xy_rmse=np.nan,z_rmse=np.nan,samples=0,coverage=0.)
    # State readout is causal: no future native update may interpolate backward.
    state_index=np.searchsorted(rows[:,0],grid,side="right")-1
    position=rows[state_index,1:4]
    target = np.column_stack([np.interp(grid,reference[:,0],reference[:,k]) for k in (1,2,3)])
    error = position-target
    return dict(xy_rmse=float(np.sqrt(np.mean(np.sum(error[:,:2]**2,axis=1)))),
                z_rmse=float(np.sqrt(np.mean(error[:,2]**2))),samples=len(grid),coverage=len(grid)/round((end-start)*100))


def main():
    """Generate all frozen sensor-only traces before opening holdout truth."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--names",nargs="+")
    parser.add_argument("--queues",nargs="+",default=["first_per_stamp"])
    parser.add_argument("--lengths",nargs="+",type=float,default=[0,2,5,10,20])
    parser.add_argument("--output",type=Path,default=HERE/"baseline")
    parser.add_argument("--score-only",action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    calibration_path = HERE/"calibration.json"
    if not calibration_path.exists():
        calibration_path.write_text(json.dumps(adapter.calibrate(),indent=2),encoding="utf8")
    calibration = json.loads(calibration_path.read_text(encoding="utf8"))
    records = []
    if args.score_only:
        records = json.loads((args.output/"records.json").read_text(encoding="utf8"))
    else:
        for name in args.names or adapter.PROTOCOL["test_names"]:
            raw = adapter.load(name,calibration)
            for queue in args.queues:
                data = queued(raw,queue)
                for length in args.lengths:
                    key = name+"__"+queue+f"__L{length:g}"
                    artifact = args.output/(key+".npz")
                    if artifact.exists():
                        cached = np.load(artifact)
                        info = json.loads(str(cached["info"]))
                    else:
                        rows,info = replay(data,queue,None if length == 0 else (25-length,25.))
                        np.savez_compressed(artifact,rows=rows,info=json.dumps(info))
                    record = dict(key=key,name=name,queue=queue,outage_s=length,origin=raw["origin"],**info)
                    records.append(record)
                    (args.output/"records.json").write_text(json.dumps(records,indent=2),encoding="utf8")
                    print(json.dumps(record),flush=True)
        provenance = dict(author_source=str(AUTHOR_PATH),author_sha256=adapter.digest(AUTHOR_PATH),
                          protocol_sha256=adapter.digest(HERE/"protocol.json"),
                          adapter_sha256=adapter.digest(HERE/"adapter.py"),run_sha256=adapter.digest(HERE/"run.py"),
                          calibration_sha256=adapter.digest(calibration_path),test_truth_used_during_generation=False)
        (args.output/"provenance.json").write_text(json.dumps(provenance,indent=2),encoding="utf8")
    print("Generation complete; reading test truth for scoring only.",flush=True)
    metrics=[]
    for record in records:
        rows=np.load(args.output/(record["key"]+".npz"))["rows"]
        reference=adapter.truth(record["name"],record["origin"],calibration["lag_s"])
        for left,right in adapter.PROTOCOL["post_score_intervals_s"]:
            metrics.append({**record,"interval":f"{left}-{right}",**score(rows,reference,25+left,25+right)})
    frame=pd.DataFrame(metrics)
    frame.to_csv(args.output/"metrics.csv",index=False)
    frame.groupby(["queue","outage_s","interval"])[["xy_rmse","z_rmse","coverage"]].agg(["median","max"]).to_csv(args.output/"summary.csv")


if __name__ == "__main__":
    main()
