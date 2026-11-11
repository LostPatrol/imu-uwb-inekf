"""Check native frontend derivatives and decisions under unavailable radio changes."""

import json

import numpy as np

import run


def main():
    """Audit sensor-only derivatives and invariance to unavailable radio values."""
    data=run.util.load(run.util.record_name(2,1,3),duration=run.DURATION)
    data.pop("truth")
    origin=data["times"][0]
    start=origin+20.
    endpoint=float(data["times"][np.searchsorted(data["times"],origin+20.1)])
    selected=data["radio"][(data["radio"][:,0]>=start)&(data["radio"][:,0]<=endpoint)]
    tau=selected[:,0]-endpoint
    ids=selected[:,1:3].astype(int)
    sigma=np.full(len(selected),np.sqrt(.14))
    parameters=np.r_[np.mean(data["anchors"],axis=0)+[.7,-.4,.6],[.2,-.3,.1]]
    def model(value):
        """Return the same native residual used by recovery fitting."""
        return run.native_model(value,tau,data["anchors"][ids[:,0]],data["anchors"][ids[:,1]],selected[:,3],sigma)
    numeric=np.empty((len(selected),6))
    for index in range(6):
        step=np.zeros(6)
        step[index]=1e-6
        numeric[:,index]=(model(parameters+step)[0]-model(parameters-step)[0])/2e-6
    jacobian_error=float(np.max(np.abs(numeric-model(parameters)[1])))
    state=np.eye(5)
    covariance=run.transport(state,np.eye(15)*100.)
    original=run.horizontal_candidate(data,start,endpoint,state,covariance)
    changed=dict(data,radio=data["radio"].copy())
    unavailable=(changed["radio"][:,0]>endpoint)|((changed["radio"][:,0]>=origin+10.)&(
        changed["radio"][:,0]<start))
    changed["radio"][unavailable,3]+=10000.
    altered=run.horizontal_candidate(changed,start,endpoint,state,covariance)
    same=json.dumps(run.util.safe_json(original),sort_keys=True)==json.dumps(run.util.safe_json(altered),sort_keys=True)
    assert jacobian_error<1e-7 and same
    output=run.HERE/"results"
    run.save_json(output/"frontend_audit.json",dict(name=data["name"],native_jacobian_max_absolute_error=jacobian_error,
        future_and_masked_radio_perturbation_decision_identical=same,
        evaluation_truth_used=False,perturbation_m=10000.))
    print("Native Jacobian max error",jacobian_error,"unavailable-radio invariance",same)


if __name__=="__main__":
    main()
