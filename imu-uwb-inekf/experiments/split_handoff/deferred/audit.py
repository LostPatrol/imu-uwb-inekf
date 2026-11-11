"""Check deferred candidate causality without reading trajectory truth or scores."""

from pathlib import Path
import hashlib
import json

import numpy as np

import run


def candidate_inputs(data, endpoint, altered=False):
    """Build the same cold candidate prefix with optionally corrupted future sensors."""
    initial_time, state, bias=run.util.initial_state(data)
    _, evidence, _=run.partition_radio(data,initial_time)
    evidence=dict(evidence,radio=evidence["radio"].copy(),imu=evidence["imu"].copy())
    if altered:
        evidence["radio"][evidence["radio"][:,0]>endpoint,3]+=10000.
        evidence["imu"][evidence["times"]>endpoint]+=10000.
    path=run.util.radio_path(evidence)
    path=path[path[:,0]<=endpoint]
    return run.util.window_inputs(evidence,data["times"][0],endpoint,bias,path)


def outage_inputs(data, altered=False):
    """Verify masked samples cannot affect the recovered-window candidate model."""
    data=dict(data,radio=data["radio"].copy())
    origin=data["times"][0]
    hidden=(data["radio"][:,0]>=origin+10.)&(data["radio"][:,0]<origin+20.)
    if altered:
        data["radio"][hidden,3]+=10000.
    initial_time,_,bias=run.util.initial_state(data)
    _,evidence,_=run.partition_radio(data,initial_time)
    radio=evidence["radio"]
    evidence=dict(evidence,radio=radio[(radio[:,0]<origin+10.)|(radio[:,0]>=origin+20.)])
    endpoint=run.aligned_time(data,26.)
    path=run.util.radio_path(evidence)
    path=path[path[:,0]<=endpoint]
    return run.util.window_inputs(evidence,run.aligned_time(data,20.),endpoint,bias,path)


def main():
    """Persist a future-perturbation test and the sensor-only RF split audit."""
    data=run.util.load(run.util.record_name(2,6,3),duration=run.DURATION)
    data.pop("truth")
    endpoint=run.aligned_time(data,6.)
    original=candidate_inputs(data,endpoint)
    changed=candidate_inputs(data,endpoint,True)
    differences={key:float(np.max(np.abs(original[key]-changed[key]))) for key in
        ("tau","force","B","plus","minus","y","sigma","path","Qend")}
    assert max(differences.values())==0.
    original_outage,changed_outage=outage_inputs(data),outage_inputs(data,True)
    outage_differences={key:float(np.max(np.abs(original_outage[key]-changed_outage[key]))) for key in
        ("tau","force","B","plus","minus","y","sigma","path","Qend")}
    assert max(outage_differences.values())==0.
    initial_time,_,_=run.util.initial_state(data)
    navigation,evidence,split=run.partition_radio(data,initial_time)
    assert np.min(navigation["radio"][:,0])>initial_time
    assert np.min(evidence["radio"][:,0])>initial_time
    origin=data["times"][0]
    masked=evidence["radio"][(evidence["radio"][:,0]<origin+10.) |
                             (evidence["radio"][:,0]>=origin+20.)]
    assert not np.any((masked[:,0]>=origin+10.)&(masked[:,0]<origin+20.))
    output=run.HERE/"results"
    output.mkdir(exist_ok=True)
    result=dict(name=data["name"],future_perturbation_max_absolute_difference=differences,
        masked_packet_perturbation_max_absolute_difference=outage_differences,
        initial_and_mask_packets_excluded=True,radio_split=split,
        source_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in
            (Path(__file__),run.UTIL/"run.py",run.UTIL/"kernel.py")})
    run.save_json(output/"causality_audit.json",result)
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
