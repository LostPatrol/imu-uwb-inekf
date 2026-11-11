"""Numerical mirror of the untouched Oursland author equations, not the oracle.

Acceleration uses the frozen dynamics' A**4=0 polynomial exponential. Author
noise/bias/gate/ground-clamp/prior semantics remain literal and are validated
against the unchanged class before this mirror may support external replay.
"""

import numpy as np
from numba import njit

# Preserve the author's external antenna calibration and main observation noise.
LEVER=np.array([-.01245,.00127,.0908])
VARIANCE=.13
NOISE=np.array([.1**2]*3+[2.**2]*3+[0.]*3+[.0001**2]*3+[.000001**2]*3)


@njit(cache=True)
def wedge(phi):
    """Return the author's skew matrix."""
    return np.array([[0.,-phi[2],phi[1]],[phi[2],0.,-phi[0]],[-phi[1],phi[0],0.]])


@njit(cache=True)
def exp_rotation(phi):
    """Preserve the author's Rodrigues formula and np.isclose-to-zero branch."""
    angle=np.sqrt(phi@phi)
    if angle<=1e-8:
        return np.eye(3)+wedge(phi)
    axis=phi/angle
    return np.cos(angle)*np.eye(3)+(1-np.cos(angle))*np.outer(axis,axis)+np.sin(angle)*wedge(axis)


@njit(cache=True)
def left_jacobian(phi):
    """Preserve the literal original small-angle sign, including its discontinuity."""
    angle=np.sqrt(phi@phi)
    if angle<=1e-8:
        return np.eye(3)-.5*wedge(phi)
    axis=phi/angle
    return (np.sin(angle)/angle)*np.eye(3)+(1-np.sin(angle)/angle)*np.outer(axis,axis)+((1-np.cos(angle))/angle)*wedge(axis)


@njit(cache=True)
def adjoint(X):
    """Return the original SE_2(3) adjoint in the upper navigation block."""
    result=np.eye(15)
    R=X[:3,:3].copy()
    result[:3,:3]=R
    result[3:6,3:6]=R
    result[6:9,6:9]=R
    result[3:6,:3]=wedge(X[:3,3])@R
    result[6:9,:3]=wedge(X[:3,4])@R
    return result


@njit(cache=True)
def inverse(X):
    """Return the exact original nominal group inverse."""
    result=np.eye(5)
    result[:3,:3]=X[:3,:3].T
    result[:3,3]=-result[:3,:3].copy()@X[:3,3].copy()
    result[:3,4]=-result[:3,:3].copy()@X[:3,4].copy()
    return result


@njit(cache=True)
def predict(X,P,bias,u,dt,clamp=True):
    """Mirror original propagation, including accel-first nominal biases and dt² Q."""
    R=X[:3,:3].copy()
    a=u[:3]*9.81-bias[:3]
    omega=u[3:]*(np.pi/180)-bias[3:]
    acceleration=R@a+np.array([0.,0.,-9.81])
    result=X.copy()
    result[:3,:3]=R@exp_rotation(omega*dt)
    result[:3,3]=X[:3,3]+acceleration*dt
    result[:3,4]=X[:3,4]+X[:3,3]*dt+.5*acceleration*dt**2
    if clamp and result[2,4]<0:
        result[2,4]=0.
        result[:3,3]=0.
    mapped=adjoint(result)
    A=np.zeros((15,15))
    A[:3,9:12]=-result[:3,:3]
    A[3:6,:3]=wedge(np.array([0.,0.,-9.81]))
    A[3:6,9:12]=-mapped[3:6,:3]
    A[3:6,12:15]=-result[:3,:3]
    A[6:9,3:6]=np.eye(3)
    A[6:9,9:12]=-mapped[6:9,:3]
    scaled=A*dt
    squared=scaled@scaled
    Phi=np.eye(15)+scaled+.5*squared+(squared@scaled)/6.
    Q=np.diag(NOISE*dt**2)
    posterior=Phi@P@Phi.T+Phi@mapped@Q@mapped.T@Phi.T
    return result,posterior


@njit(cache=True)
def correction(Xhat,Phat,bias,packet,anchors,variance=VARIANCE,geometry=0,gate=0,lever=LEVER):
    """Mirror original correction; nonzero geometry/gate modes are labeled ablations."""
    tag=Xhat[:3,:3].copy()@lever+Xhat[:3,4]
    a=anchors[int(packet[0])]-tag
    b=anchors[int(packet[1])]-tag
    da,db=np.sqrt(a@a),np.sqrt(b@b)
    residual=packet[2]-(db-da)
    Hleft=np.zeros((1,15))
    Hleft[0,6:9]=b/db-a/da
    H=Hleft@adjoint(inverse(Xhat))
    if geometry==1:
        H=Hleft.copy()
        H[0,:3]=-Hleft[0,6:9]@wedge(tag)
    gate_H=Hleft if gate==0 else H
    gate_variance=(gate_H@Phat@gate_H.T)[0,0]+variance
    if residual**2/gate_variance>=25.:
        return Xhat.copy(),Phat.copy(),bias.copy(),False
    S=(H@Phat@H.T)[0,0]+variance
    K=(Phat@H.T)/S
    IKH=np.eye(15)-K@H
    posterior=IKH@Phat@IKH.T+(K*variance)@K.T
    delta=-(K[:,0]*residual)
    increment=np.eye(5)
    increment[:3,:3]=exp_rotation(delta[:3])
    increment[:3,3:5]=left_jacobian(delta[:3])@np.column_stack((delta[3:6],delta[6:9]))
    return increment@Xhat,posterior,bias+delta[9:15],True


@njit(cache=True)
def replay(events,imu_indices,radio_indices,imu,radio,anchors,X,P,bias,start,
           previous_imu_time,outage_start=-1.,outage_end=-1.,clamp=True,geometry=0,gate=0,refresh=False,zero_bias=False,lever=LEVER):
    """Preserve previous-event author calls and stale predicted-prior semantics."""
    X,P,bias=X.copy(),P.copy(),bias.copy()
    Xhat,Phat=X.copy(),P.copy()
    rows=np.zeros((len(events),19))
    count,accepted,rejected,predicts,corrections,skipped=0,0,0,0,0,0
    failed=False
    last_output=-np.inf
    for k in range(max(1,np.searchsorted(events,start,side="right")),len(events)):
        previous,stamp=events[k-1],events[k]
        if previous<=start:
            continue
        if imu_indices[k-1]>=0:
            dt=stamp-previous_imu_time
            previous_imu_time=stamp
            Xhat,Phat=predict(X,P,bias,imu[imu_indices[k-1]],dt,clamp)
            X,P=Xhat.copy(),Phat.copy()
            predicts+=1
        if radio_indices[k-1]>=0:
            if outage_start<=previous<outage_end:
                skipped+=1
            else:
                packet=radio[radio_indices[k-1],1:]
                X,P,bias,keep=correction(Xhat,Phat,bias,packet,anchors,VARIANCE,geometry,gate,lever)
                accepted+=int(keep)
                rejected+=int(not keep)
                corrections+=1
                if refresh:
                    Xhat,Phat=X.copy(),P.copy()
                if zero_bias:
                    bias[:]=0.
        if stamp-last_output>=.01:
            rows[count,0]=stamp
            rows[count,1:4]=X[:3,4]
            rows[count,4:7]=X[:3,3]
            rows[count,7:16]=X[:3,:3].flatten()
            count+=1
            last_output=stamp
        if not np.all(np.isfinite(X)) or not np.all(np.isfinite(P)) or np.max(np.abs(X[:3,4]))>1000:
            failed=True
            break
    return X,P,bias,rows[:count].copy(),accepted,rejected,predicts,corrections,skipped,failed
