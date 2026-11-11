"""Audit author, production class and standalone research kernel on real TDoA.

Only current source modules and raw data are read; no prior report is loaded.
The original author class is imported unchanged and is always the oracle.
"""

import importlib.util
import json
import os
from pathlib import Path
import sys

# Limit small dense linear algebra independently in every replay process.
for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parents[1]
CODE = EXPERIMENTS.parent
sys.path.insert(0, str(EXPERIMENTS/"split_handoff"/"oursland"))
import native_schedule as native
import numpy as np
from scipy.spatial.transform import Rotation
from threadpoolctl import threadpool_limits

# The actual production class is distinct from split_handoff/util/kernel.py.
SPEC = importlib.util.spec_from_file_location("strong_current_inekf", CODE/"scripts"/"estimation"/"inekf.py")
current = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = current
SPEC.loader.exec_module(current)
THREAD_LIMIT = threadpool_limits(1)
kernel = native.ours.util.kernel
NAMES = ("const1-trial4-tdoa2", "const2-trial4-tdoa3")
YAWS = (0., 180.)


def initial(name, yaw):
    """Build the identical sensor-only initial X, author P and canonical bias."""
    data = native.load_raw(name, False)
    data.pop("truth")
    common = native.ours.util.load(name, duration=45.)
    common.pop("truth")
    start, X, bias = native.ours.util.initial_state(common)
    X[:3,:3] = Rotation.from_euler("z",yaw,degrees=True).as_matrix()@X[:3,:3]
    return data, start, X, native.ORIGINAL_P.copy(), bias


def geometry(X, packet, anchors):
    """Return the literal author gate row, transformed row and complete row."""
    tag = X[:3,4]+X[:3,:3]@kernel.LEVER
    a, b = tag-anchors[int(packet[1])], tag-anchors[int(packet[2])]
    gradient = b/np.linalg.norm(b)-a/np.linalg.norm(a)
    left = np.zeros((1,15))
    left[0,6:9] = -gradient
    inverse = np.eye(15)
    inverse[:9,:9] = native.ours.author.SE3_2_adjoint(native.ours.author.SE3_2_inv(X))
    complete = left.copy()
    complete[0,:3] = gradient@current.SO3_wedge(tag)
    return left, left@inverse, complete


def single_step(name, yaw):
    """Measure actual propagation and observation differences at the first packet."""
    data,start,X,P,bias = initial(name,yaw)
    first = np.searchsorted(data["times"],start,side="right")
    dt = data["times"][first]-start
    sample = data["imu"][first-1]
    oracle = native.ours.make_filter(X,P,bias)
    production = current.InEKF(X,P,bias,current.FilterConfig.proposed())
    production.w_tdoa = np.array([np.sqrt(.13)])
    oracle.predict(sample,dt)
    production.predict(sample,dt)
    UX,UP = kernel.predict(X,P,bias,sample,dt)
    packet = data["radio"][np.searchsorted(data["radio"][:,0],start,side="right")]
    left,transformed,complete = geometry(oracle.X,packet,data["anchors"])
    finite = np.zeros((1,15))
    epsilon = 1e-6
    def observation(state):
        """Evaluate the native signed distance difference without truth."""
        tag = state[:3,4]+state[:3,:3]@kernel.LEVER
        return np.linalg.norm(tag-data["anchors"][int(packet[2])])-np.linalg.norm(tag-data["anchors"][int(packet[1])])
    for index in range(9):
        delta = np.zeros(9)
        delta[index] = epsilon
        finite[0,index] = (observation(current.SE3_2_exp(-delta)@oracle.X)-observation(current.SE3_2_exp(delta)@oracle.X))/(2*epsilon)
    prior_variance = float(left@oracle.P@left.T+.13)
    transformed_variance = float(transformed@oracle.P@transformed.T+.13)
    complete_variance = float(complete@oracle.P@complete.T+.13)
    residual = float(packet[3]-observation(oracle.X))
    propagate = dict(production_X_max=float(np.max(np.abs(production.X-oracle.X))),
        production_P_max=float(np.max(np.abs(production.P-oracle.P))), util_X_max=float(np.max(np.abs(UX-oracle.X))),
        util_P_max=float(np.max(np.abs(UP-oracle.P))))
    oracle.correct(packet[1:],data["anchors"])
    production.correct(packet[1:],data["anchors"])
    observation_packet=packet.copy()
    observation_packet[0]=start+dt
    UX,UP,Ub,keep = kernel.correct(UX,UP,bias,observation_packet,data["anchors"],start+dt,.13)
    return dict(name=name,yaw_offset_deg=yaw,dt=float(dt),propagation=propagate,
        transformed_jacobian_finite_difference_error=float(np.max(np.abs(transformed-finite))),
        complete_jacobian_finite_difference_error=float(np.max(np.abs(complete-finite))),
        residual=residual,original_gate_variance=prior_variance,transformed_variance=transformed_variance,
        complete_variance=complete_variance,author_transformed_H=transformed.tolist(),complete_H=complete.tolist(),
        production_post_X_max=float(np.max(np.abs(production.X-oracle.X))),
        util_post_X_max=float(np.max(np.abs(UX-oracle.X))),
        production_post_yaw_difference_deg=float(np.degrees((Rotation.from_matrix(oracle.X[:3,:3]).inv()*Rotation.from_matrix(production.X[:3,:3])).as_rotvec()[2])),
        util_post_yaw_difference_deg=float(np.degrees((Rotation.from_matrix(oracle.X[:3,:3]).inv()*Rotation.from_matrix(UX[:3,:3])).as_rotvec()[2])))


def main():
    """Save small mechanistic measurements before any larger matrix is authorized."""
    output = HERE/"audit"
    output.mkdir(parents=True,exist_ok=True)
    records = [single_step(name,yaw) for name in NAMES for yaw in YAWS]
    (output/"single_step.json").write_text(json.dumps(records,indent=2),encoding="utf8")
    print(json.dumps(records,indent=2))


if __name__=="__main__":
    main()
