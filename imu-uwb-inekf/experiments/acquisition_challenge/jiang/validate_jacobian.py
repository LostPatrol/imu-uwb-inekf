"""Compare analytic factor derivatives with independent central differences."""
import json
import os
for name in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[name]='1'
import numpy as np
from scipy.optimize._numdiff import approx_derivative
from scipy.sparse import coo_matrix
from scipy.spatial.transform import Rotation
from factor_graph import BatchGraph, graph_residual, graph_jacobian, retract_state,exp_so3,log_so3


def main():
    """Check nonzero lever/bias, both radio branches, and canonical zero/pi yaw priors."""
    imu_t=np.arange(0.,.401,.01)
    acc=np.tile([.15,.08,9.81],(len(imu_t),1))
    gyro=np.tile([.1,.2,.3],(len(imu_t),1))
    event_t=np.arange(.1,.401,.1)
    anchors=np.array([[-5.,-5.,0.],[5.,-5.,0.],[5.,5.,3.],[-5.,5.,3.]])
    graph=BatchGraph(imu_t,acc,gyro,event_t,anchors,np.array([3.,20.,4.,15.]),
                     np.full(4,.15),Rotation.from_euler('xyz',[.1,-.15,.7]).as_matrix(),
                     np.array([.001,.002,-.003,.01,-.02,.03]),lever=[.1,-.025,0.])
    sigma=np.r_[np.ones(6),np.full(3,.1),np.full(3,.01),np.full(3,.1)]
    sigma[2]=.087
    records=[]
    cost_invariance=[]
    rng=np.random.default_rng(240)
    axis=np.array([-.8,.5,.2])
    axis/=np.linalg.norm(axis)
    for angle in (np.pi-1e-6,np.pi,np.pi+1e-6,12.*np.pi+.2):
        rotation=exp_so3(axis*angle)
        assert np.linalg.norm(exp_so3(log_so3(rotation))-rotation)<1e-7
    for yaw in (0.,np.pi, np.pi+.2):
        initial=graph.initial_state([1.,-1.,.8],yaw)
        x=initial.reshape(-1)+rng.normal(0.,.02,initial.size)
        args=(graph.dt,graph.dr,graph.dv,graph.dp,graph.jac,graph.whiten,
              graph.bias_scale,graph.bias_decay,graph.event_node,graph.anchor,
              graph.observed,graph.sigma,graph.r0,graph.bias0,graph.lever,graph.gravity,
              initial[0].copy(),sigma,True)
        rows,cols,data=graph_jacobian(x,*args)
        analytic=coo_matrix((data,(rows,cols)),shape=graph.sparsity.shape).toarray()
        numerical=approx_derivative(lambda step:graph_residual(retract_state(x,step),*args),
                                   np.zeros_like(x),method='3-point')
        error=np.abs(analytic-numerical)
        scaled=error/np.maximum(1.,np.abs(numerical))
        records.append(dict(yaw_rad=yaw,max_abs=float(error.max()),max_scaled=float(scaled.max())))
        assert scaled.max()<2e-5,(np.unravel_index(scaled.argmax(),scaled.shape),records[-1])
        # No velocity or tilt prior rows can become independent observations.
        cursor=(len(graph.times)-1)*15+len(graph.observed)
        assert np.count_nonzero(analytic[cursor:cursor+2])==0
        assert np.count_nonzero(analytic[cursor+3:cursor+6])==0
        canonical=retract_state(x,np.zeros_like(x))
        assert np.linalg.norm(canonical.reshape((-1,15))[:,:3],axis=1).max()<=np.pi+1e-7
        before=graph_residual(x,*args)
        after=graph_residual(canonical,*args)
        relative_cost_delta=abs(before@before-after@after)/max(1.,before@before)
        cost_invariance.append(float(relative_cost_delta))
        assert relative_cost_delta<1e-10
    print(json.dumps(dict(derivative_checks=records,passed=True,
        retraction='left Exp(delta) R',canonical_cost_relative_deltas=cost_invariance,
        near_pi_and_many_turn_roundtrip_passed=True,no_tilt_or_velocity_prior=True),indent=2))


if __name__=='__main__':
    main()
