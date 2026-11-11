"""Audit zero-lever Delft equivalence and the necessary external tag Jacobian."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation

from adapter import (CODE, ROOT, LEGACY, SETTINGS, PRIOR_STD, CaptureThenTrack,
    LiftedTWRInEKF, LeverInvariant15, Invariant15, active_core,
    coordinate_map, handoff_covariance, make_reference)
from invariant15 import group_exp


def lever_jacobian():
    """Differentiate the complete tag measurement under the native left injection."""
    state=np.eye(5)
    state[:3,:3]=Rotation.from_euler("xyz",[.2,-.3,.5]).as_matrix()
    state[:3,3]=[.4,-.6,.3]
    state[:3,4]=[1.,-.3,.8]
    core=LeverInvariant15(state,np.eye(15),np.zeros(6),lever=[.13189,-.17245,-.05249])
    anchor=np.array([-3.,2.,1.9])
    expected,row=core.range_model(anchor)
    numerical=np.zeros(15)
    for j in range(9):
        perturb=np.zeros(9)
        perturb[j]=1e-6
        plus,minus=group_exp(perturb)@state,group_exp(-perturb)@state
        p1=plus[:3,4]+plus[:3,:3]@core.external_lever
        p2=minus[:3,4]+minus[:3,:3]@core.external_lever
        numerical[j]=(np.linalg.norm(p1-anchor)-np.linalg.norm(p2-anchor))/(2e-6)
    effective=anchor-state[:3,:3]@core.external_lever
    return dict(tag_range_jacobian_max_error=float(np.max(np.abs(numerical-row[0]))),
        capture_effective_anchor_range_error=float(abs(expected-np.linalg.norm(state[:3,4]-effective))))


def delft_equivalence(cache, duration):
    """Compare native/adapted complete zero-lever filters, including both handoffs."""
    protocol=json.loads((cache/"protocol.json").read_text(encoding="utf-8"))
    name=protocol["evaluation"][0]
    data=dict(np.load(cache/(Path(name).stem+".npz")))
    model=protocol["model"]
    state=np.eye(5)
    state[:3,:3]=data["rotations"][0]
    state[:3,4]=data["anchors"].mean(axis=0)
    chart=coordinate_map(state)
    prior=chart@np.diag(PRIOR_STD**2)@chart.T
    native=CaptureThenTrack(state,prior,data["bias"],reset_on_resume=True,
        tracker_factory=Invariant15,imu_core_factory=Invariant15,
        capture_factory=LiftedTWRInEKF,handoff_covariance_factory=handoff_covariance,
        range_bias=model["range_bias"],range_sigma=model["range_sigma"],**SETTINGS)
    adapted=make_reference(state,prior,data["bias"],np.zeros(3),
        range_bias=model["range_bias"],range_sigma=model["range_sigma"])
    maxima=np.zeros(5)
    mismatched=0
    resumed=False
    started=time.perf_counter()
    for k,stamp in enumerate(data["time"]):
        if stamp>duration:
            break
        if k:
            dt=float(stamp-data["time"][k-1])
            for core in (native,adapted):
                core.predict(data["imu"][k-1],dt)
        valid=data["changed"][k]&(data["ranges"][k]>0)
        if 15.<=stamp<25.:
            valid[:]=False
        if np.any(valid) and stamp>=25. and not resumed:
            native.on_ranges_resumed(); adapted.on_ranges_resumed()
            resumed=True
        for anchor in np.flatnonzero(valid):
            yes=native.correct_twr(data["anchors"][anchor],data["ranges"][k,anchor],int(anchor))
            other=adapted.correct_twr(data["anchors"][anchor],data["ranges"][k,anchor],int(anchor))
            mismatched+=yes!=other
        values=[np.max(np.abs(native.X-adapted.X)),np.max(np.abs(native.P-adapted.P)),
            np.max(np.abs(native.bias-adapted.bias)),0.,0.]
        if native.mode!=adapted.mode:
            mismatched+=1
        if native.mode=="capture":
            values[3]=np.max(np.abs(native.capture.lifted.P-adapted.capture.lifted.P))
            values[4]=np.max(np.abs(native.capture.lifted.chi-adapted.capture.lifted.chi))
        maxima=np.maximum(maxima,values)
    return dict(sequence=name,duration_s=duration,elapsed_s=time.perf_counter()-started,
        max_X_difference=float(maxima[0]),max_P_difference=float(maxima[1]),
        max_bias_difference=float(maxima[2]),max_lifted_P_difference=float(maxima[3]),
        max_lifted_chi_difference=float(maxima[4]),mode_or_gate_decision_mismatches=int(mismatched),
        native_stats=vars(native.stats),adapted_stats=vars(adapted.stats),
        zero_lever_port_is_exact=bool(np.max(maxima)==0 and mismatched==0),
        checked_startup_and_post_gap_capture=bool(resumed and duration>=28.))


def main():
    """Run one small real compatibility check and retain its full audit."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache",type=Path,default=CODE/"experiments/split_handoff/delft/results/cache")
    parser.add_argument("--duration",type=float,default=30.)
    args=parser.parse_args()
    output=ROOT/"agent/codex/strong_external"
    output.mkdir(parents=True,exist_ok=True)
    result=dict(lever=lever_jacobian(),delft=delft_equivalence(args.cache,args.duration))
    (output/"compatibility.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result),flush=True)


if __name__=="__main__":
    main()
