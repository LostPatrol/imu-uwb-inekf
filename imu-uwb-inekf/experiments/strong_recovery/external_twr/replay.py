"""Replay a full original TWR reference and the common read-only recovery display.

The fixed cross-site cohort is selected before estimator scores are available.
Motion capture is loaded only by the separate downstream score function.
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation, Slerp

from adapter import (CODE, ROOT, ROBOT, RANGE_SIGMA_M, SETTINGS, active_core,
                     initial_reference, read_sensors)

# Resume time, gap lengths and scoring windows are frozen before external replay.
RESUME = 25.0
LENGTHS = (2., 5., 10., 20.)
# The preselected cohort needs 45 s after its static prefix, independent of scores.
REQUIRED_DURATION = 45.0
# Evaluation stops after every method has the complete six-to-eleven-second window.
END = 36.0
# Exact edges are common to every case, avoiding case-dependent IMU subdivisions.
EDGES = np.array([0., 5., 15., 20., 23., 25., 27., 30., 31., 36.])


def write_json(path, payload):
    """Store readable UTF-8 audit data, including full failures instead of dropping cases."""
    def convert(value):
        """Convert only NumPy containers/scalars used by the shared audit interface."""
        if isinstance(value,np.ndarray):
            return value.tolist()
        if isinstance(value,np.generic):
            return value.item()
        raise TypeError(type(value).__name__)
    path.write_text(json.dumps(payload,indent=2,ensure_ascii=False,default=convert),encoding="utf-8")


def reference_replay(record, length, resume=RESUME):
    """Execute the full native supervisor at every true sensor/radio arrival time."""
    begun = time.perf_counter()
    reference = initial_reference(record)
    end = resume+11.
    edges = np.unique(np.r_[0.,resume-np.asarray(LENGTHS),resume,resume+np.array([2.,5.,6.,11.])])
    timeline = np.unique(np.r_[record.imu_time[record.imu_time<=end],
        record.radio_time[record.radio_time<=end],edges])
    timeline = timeline[timeline>=record.imu_time[0]]
    n = len(timeline)
    states = np.full((n,5,5),np.nan)
    biases = np.full((n,6),np.nan)
    covariance_diagonal = np.full((n,15),np.nan)
    modes = np.zeros(n,dtype=np.uint8)
    events,snapshots = [],{}
    packet = 0
    resumed = False
    failure = None
    for k,stamp in enumerate(timeline):
        try:
            if k:
                sample = max(0,int(np.searchsorted(record.imu_time,timeline[k-1],side="right")-1))
                reference.predict(record.imu[sample],float(stamp-timeline[k-1]))
            while packet<len(record.radio_time) and record.radio_time[packet]<=stamp:
                radio_time = record.radio_time[packet]
                anchor_id = int(record.radio_id[packet])
                blocked = resume-length<=radio_time<resume
                if not blocked:
                    restarted = False
                    if radio_time>=resume and not resumed:
                        snapshots["pre_resume_X"] = reference.X.copy()
                        snapshots["pre_resume_P"] = reference.P.copy()
                        snapshots["pre_resume_bias"] = reference.bias.copy()
                        resumed = True
                        restarted = reference.on_ranges_resumed()
                    before = reference.X.copy()
                    mode = reference.mode
                    accepted = reference.correct_twr(record.anchors[anchor_id],record.radio_range[packet],anchor_id)
                    events.append(dict(time_s=float(radio_time),arrival_s=float(stamp),
                        anchor=anchor_id,range_m=float(record.radio_range[packet]),mode=mode,
                        accepted=bool(accepted),nis=float(reference.last_nis),
                        innovation=float(reference.last_innovation),restart=bool(restarted),
                        position_jump_m=float(np.linalg.norm(reference.X[:3,4]-before[:3,4])),
                        velocity_jump_mps=float(np.linalg.norm(reference.X[:3,3]-before[:3,3])),
                        cumulative_accepted=reference.stats.accepted,cumulative_rejected=reference.stats.rejected))
                packet += 1
            states[k],biases[k] = reference.X,reference.bias
            covariance_diagonal[k] = np.diag(reference.P)
            modes[k] = reference.mode=="track"
            if stamp in edges:
                label = str(float(stamp))
                snapshots[label+"_X"],snapshots[label+"_P"] = reference.X.copy(),reference.P.copy()
                snapshots[label+"_bias"] = reference.bias.copy()
                if reference.mode=="capture":
                    snapshots[label+"_lifted_P"] = reference.capture.lifted.P.copy()
                    snapshots[label+"_lifted_chi"] = reference.capture.lifted.chi.copy()
            if not (np.isfinite(states[k]).all() and np.isfinite(covariance_diagonal[k]).all()):
                raise FloatingPointError("Nonfinite native full reference state/covariance")
        except Exception:
            failure = dict(time_s=float(stamp),traceback=traceback.format_exc())
            break
    output = dict(time=timeline,X=states,bias=biases,P_diagonal=covariance_diagonal,
        mode_track=modes,K=states[:,:3,4],K_velocity=states[:,:3,3],
        lever_world=np.einsum("nij,j->ni",states[:,:3,:3],record.lever),**snapshots)
    audit = dict(length_s=length,resume_s=resume,elapsed_s=time.perf_counter()-begun,
        failure=failure,stats=vars(reference.stats),full_reference_preserved=True,
        display_does_not_change_reference=True)
    return output,events,audit


def apply_shadow(record, result, length, resume=RESUME):
    """Pass copies of only causal sensors/K history to the one shared shadow module."""
    spec = importlib.util.spec_from_file_location("strong_shadow",CODE/"experiments/strong_recovery/shadow.py")
    shared = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = shared
    spec.loader.exec_module(shared)
    received = ~((record.radio_time>=resume-length)&(record.radio_time<resume))
    received &= record.radio_time<=resume+11.
    shadow = shared.recover(result["time"].copy(),result["K"].copy(),result["K_velocity"].copy(),
        result["lever_world"].copy(),record.radio_time[received].copy(),
        record.anchors[record.radio_id[received]].copy(),None,
        record.radio_range[received].copy(),np.full(np.count_nonzero(received),RANGE_SIGMA_M),resume)
    result["proposal"],result["proposal_velocity"] = shadow["pos"],shadow["vel"]
    result["always_C"],result["always_C_velocity"] = shadow["always_c_pos"],shadow["always_c_vel"]
    result["use_C"] = shadow["use_c"]
    bounds_spec = importlib.util.spec_from_file_location("strong_bounds",CODE/"experiments/strong_recovery/radio_bounds.py")
    bounds = importlib.util.module_from_spec(bounds_spec)
    sys.modules[bounds_spec.name] = bounds
    bounds_spec.loader.exec_module(bounds)
    radius = float(record.metadata.get("bounds_lever_radius_m",np.linalg.norm(record.lever)))
    bounded = bounds.bound_xy(result["time"].copy(),result["K"].copy(),radius,
        record.radio_time[received].copy(),record.anchors[record.radio_id[received]].copy(),None,
        record.radio_range[received].copy(),np.full(np.count_nonzero(received),RANGE_SIGMA_M),resume)
    result["radio_bounds"],result["radio_bounds_active"] = bounded["pos"],bounded["active"]
    beyond = result["time"]>=resume+2.
    return dict(shadow=shadow["diagnostics"],radio_bounds=bounded["diagnostics"],
        shadow_equals_K_after_2s=bool(np.array_equal(result["proposal"][beyond],result["K"][beyond],equal_nan=True)),
        bounds_equals_K_after_2s=bool(np.array_equal(result["radio_bounds"][beyond],result["K"][beyond],equal_nan=True)))


def score(directory, record, result, length):
    """Read only downstream MoCap to score fixed windows without trajectory alignment."""
    columns = ["timestamp"]+[f"pose.position.{a}" for a in "xyz"]+[f"pose.orientation.{a}" for a in "xyzw"]
    truth = pd.read_csv(directory/record.name/ROBOT/"mocap.csv",usecols=columns)
    values = truth[columns].to_numpy(float)
    keep = np.isfinite(values).all(axis=1)&(np.linalg.norm(values[:,4:],axis=1)>1e-6)
    truth = truth.loc[keep].sort_values("timestamp").drop_duplicates("timestamp")
    stamps = truth.timestamp.to_numpy(float)-record.origin_s
    positions = truth[columns[1:4]].to_numpy(float)
    quaternions = truth[columns[4:]].to_numpy(float)
    clock = result["time"]
    indices = np.clip(np.searchsorted(stamps,clock),1,len(stamps)-1)
    supported = (clock>=stamps[0])&(clock<=stamps[-1])&((stamps[indices]-stamps[indices-1])<.20)
    reference = np.column_stack([np.interp(clock,stamps,positions[:,j]) for j in range(3)])
    rotation_truth = np.full((len(clock),3,3),np.nan)
    rotation_truth[supported] = Slerp(stamps,Rotation.from_quat(quaternions))(clock[supported]).as_matrix()
    rows=[]
    for method in ("K","proposal","always_C","radio_bounds"):
        error = result[method]-reference
        for label,begin,end in (("first2",0.,2.),("two_to_five",2.,5.),("six_to_eleven",6.,11.)):
            window = (clock>=RESUME+begin)&(clock<RESUME+end)
            available = supported&window&np.isfinite(error).all(axis=1)
            fraction = float(np.sum(available)/max(1,np.sum(window)))
            selected = error[available]
            rows.append(dict(sequence=record.name,length_s=length,method=method,window=label,
                evaluated_fraction=fraction,xy_rmse_m=float(np.sqrt(np.mean(np.sum(selected[:,:2]**2,axis=1)))) if len(selected) else np.nan,
                z_rmse_m=float(np.sqrt(np.mean(selected[:,2]**2))) if len(selected) else np.nan,
                xy_peak_m=float(np.max(np.linalg.norm(selected[:,:2],axis=1))) if len(selected) else np.nan))
    result["scoring_position_truth"] = reference
    result["scoring_rotation_truth"] = rotation_truth
    result["scoring_truth_supported"] = supported
    return rows


def main():
    """Preserve all frozen candidates, including unavailable or unsupported recordings."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",type=Path,required=True)
    parser.add_argument("--metadata",type=Path,default=ROOT/"agent/codex/strong_data")
    parser.add_argument("--output",type=Path,default=ROOT/"agent/codex/strong_external/results")
    parser.add_argument("--sequence",default="")
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    frozen=json.loads((args.metadata/"frozen_candidates.json").read_text(encoding="utf-8"))
    cohort=frozen["miluv_priority_frozen_experiments"]
    protocol=dict(dataset="MILUV",official_git_commit="399ad46f",cohort=cohort,
        resume_s=RESUME,lengths_s=LENGTHS,required_sensor_duration_s=REQUIRED_DURATION,
        yaw_perturbation_deg=0,range_sigma_m=RANGE_SIGMA_M,range_bias_m=0.,
        original_policy=SETTINGS,original_reference="Complete historical CaptureThenTrack reset+LiftedTWRInEKF+Invariant15",
        adaptation="SI decode and supplied nonzero tag lever only; body acceleration and native P/gates retained.",
        gt_in_estimator=False,gt_initialized=False,per_test_alignment=False,
        official_uwb_range_csv_rejected="Upstream PostProcess selects rows with abs(range-gt_range)<4; require uwb_range_unfiltered.csv reconstructed from raw transactions.",
        shadow="Same shared strong_recovery/shadow.py; display only, no posterior or covariance reset.")
    write_json(args.output/"protocol.json",protocol)
    eligibility,rows=[],[]
    for name in cohort:
        if args.sequence and args.sequence not in name:
            continue
        entry=dict(sequence=name,status="pending")
        try:
            record=read_sensors(args.dataset,name,args.metadata)
            entry.update(duration_s=float(record.imu_time[-1]),metadata=record.metadata)
            if record.imu_time[-1]<REQUIRED_DURATION:
                entry.update(status="unsupported",reason="Less than fixed required 45 s sensor duration")
                eligibility.append(entry)
                continue
            entry["status"]="eligible"
            completed=[]
            for length in LENGTHS:
                result,events,audit=reference_replay(record,length)
                identifier=name+"__L"+str(int(length))
                pd.DataFrame(events).to_csv(args.output/(identifier+"_events.csv"),index=False)
                if audit["failure"] is None:
                    try:
                        audit["shadow_diagnostics"] = apply_shadow(record,result,length)
                    except Exception:
                        audit["display_failure"] = traceback.format_exc()
                if audit["failure"] is not None or "display_failure" in audit:
                    for method in ("proposal","always_C","radio_bounds"):
                        result[method]=np.full_like(result["K"],np.nan)
                # Retain native K even if a display or downstream scoring tool fails.
                np.savez_compressed(args.output/(identifier+".npz"),**result)
                write_json(args.output/(identifier+".json"),audit)
                completed.append((length,identifier,result,audit))
                print(f"K DONE {identifier}: {audit['elapsed_s']:.2f}s failure={audit['failure'] is not None}",flush=True)
            # No scoring file is opened until all paired estimator outputs exist.
            for length,identifier,result,audit in completed:
                try:
                    rows.extend(score(args.dataset,record,result,length))
                except Exception:
                    audit["scoring_failure"] = traceback.format_exc()
                np.savez_compressed(args.output/(identifier+".npz"),**result)
                write_json(args.output/(identifier+".json"),audit)
        except FileNotFoundError as exc:
            entry.update(status="unavailable",reason=str(exc))
        except Exception:
            entry.update(status="failed",reason=traceback.format_exc())
        eligibility.append(entry)
        write_json(args.output/"eligibility.json",eligibility)
        pd.DataFrame(rows).to_csv(args.output/"metrics.csv",index=False)
    write_json(args.output/"eligibility.json",eligibility)
    write_json(args.output/"source_sha256.json",{p.name:hashlib.sha256(p.read_bytes()).hexdigest()
        for p in Path(__file__).parent.glob("*.py")})


if __name__=="__main__":
    main()
