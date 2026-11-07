"""Exact polynomial-nuisance extension and bias-mismatch Monte Carlo study.

Quadratic translation absorbs constant world acceleration bias. This matches
constant attitude with constant body accelerometer bias, not arbitrary rotation.
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from initializer import (project_out, circle_minimum, yaw_rotation, solve_window,
                         robust_cost, HUBER_SCALE, MAX_EVALUATIONS, RANK_RTOL)


def polynomial_lift(tau, displacement, anchors, ranges, sigmas, degree=2):
    """Profile all lifted nuisances for a degree-K additive translation polynomial."""
    t, d, a = tau, displacement, anchors
    powers = np.column_stack([t**j for j in range(degree+1)])
    # Norm polynomial has degree 2K; horizontal cross terms have K+1 coefficients.
    nuisance = np.column_stack([t**j for j in range(2*degree+1)] +
                                [t**j*d[:, axis] for j in range(degree+1) for axis in (0, 1)])
    translation = np.column_stack([-powers[:, j:j+1]*a for j in range(degree+1)])
    for j in range(degree+1):
        translation[:, 3*j+2] += powers[:, j]*d[:, 2]
    circular = np.column_stack((-(a[:, 0]*d[:, 0]+a[:, 1]*d[:, 1]),
                                a[:, 0]*d[:, 1]-a[:, 1]*d[:, 0]))
    y = 0.5*(ranges**2-sigmas**2-np.sum(a*a, axis=1)-np.sum(d*d, axis=1))+a[:, 2]*d[:, 2]
    weight = 1/np.sqrt(np.maximum(ranges,sigmas)**2*sigmas**2+0.5*sigmas**4)
    linear = np.column_stack((nuisance, translation))*weight[:, None]
    projected, rank = project_out(linear, np.column_stack((y,circular))*weight[:,None])
    c, response = projected[:,1:], projected[:,0]
    yaw, _, _ = circle_minimum(c.T@c, c.T@response)
    vector = np.array([np.cos(yaw),np.sin(yaw)])
    solution = np.linalg.lstsq(linear,(y-circular@vector)*weight,rcond=RANK_RTOL)[0]
    return np.r_[solution[nuisance.shape[1]:],yaw], dict(rank=rank, singular=np.linalg.svd(c,compute_uv=False))


def polynomial_residual_jacobian(x,t,d,a,r,sigma,degree=2):
    """Differentiate polynomial translation and yaw in the original range model."""
    powers = np.column_stack([t**j for j in range(degree+1)])
    rotated = d@yaw_rotation(x[-1]).T
    p = powers@x[:-1].reshape(-1,3)+rotated
    delta = p-a
    distance = np.maximum(np.linalg.norm(delta,axis=1),1e-12)
    direction = delta/distance[:,None]
    dyaw = np.column_stack((-rotated[:,1],rotated[:,0],np.zeros(t.size)))
    jacobian = np.column_stack([powers[:,j:j+1]*direction for j in range(degree+1)]+
                               [np.sum(direction*dyaw,axis=1)])/sigma[:,None]
    return (distance-r)/sigma,jacobian


def solve_polynomial(t,d,a,r,sigma,degree=2):
    """Refine a polynomial lift and report information after removing all translation."""
    started=time.perf_counter()
    seed,diagnostic=polynomial_lift(t,d,a,r,sigma,degree)
    # Polynomial localization provides an alternative translation seed if lifting is ill-conditioned.
    def fun(x):
        """Return Huber-normalized original range residuals."""
        return polynomial_residual_jacobian(x,t,d,a,r,sigma,degree)[0]
    def jac(x):
        """Return the analytic Jacobian of polynomial position plus rotated inertial motion."""
        return polynomial_residual_jacobian(x,t,d,a,r,sigma,degree)[1]
    cv=solve_window(t,d,a,r,sigma,method="cv")
    alternatives=[seed,np.r_[cv.parameters[:6],np.zeros(3*(degree-1)),seed[-1]]]
    fits=[least_squares(fun,x,jac=jac,loss="huber",f_scale=HUBER_SCALE,max_nfev=MAX_EVALUATIONS)
          for x in alternatives]
    fit=min(fits,key=lambda item:robust_cost(item.fun))
    weights=np.minimum(1,HUBER_SCALE/np.maximum(np.abs(fit.fun),1e-12))
    weighted=jac(fit.x)*np.sqrt(weights[:,None])
    projected,_=project_out(weighted[:,:-1],weighted[:,-1:])
    return fit.x,float(np.sum(projected**2)),robust_cost(fit.fun),time.perf_counter()-started,diagnostic


def trial(job):
    """Pair fixed-bias and bias-aware estimators on exactly the same random measurements."""
    seed,motion,bias_std=job
    rng=np.random.default_rng(seed)
    t=np.sort(rng.uniform(-4,0,240))
    if motion=="excited":
        d=np.column_stack((0.8*np.sin(1.7*t),0.6*(np.cos(1.1*t)-1),0.15*np.sin(0.7*t)))
    else:
        d=0.5*t[:,None]**2*np.array([0.4,-0.3,0.1])
    points=np.array([[-4,-4,0],[4,-4,0],[-4,4,0],[4,4,0],[-4,-4,3],[4,-4,3],[-4,4,3],[4,4,3]])
    a=points[rng.integers(0,8,t.size)]
    truth=np.r_[rng.uniform(-2,2,2),1.2,rng.uniform(-0.5,0.5,3),rng.uniform(-np.pi,np.pi)]
    p=truth[:3]+t[:,None]*truth[3:6]+d@yaw_rotation(truth[-1]).T
    sigma=np.full(t.size,0.1)
    ranges=np.linalg.norm(p-a,axis=1)+rng.normal(0,sigma)
    bias=rng.normal(0,bias_std,3)
    measured=d+0.5*t[:,None]**2*bias
    fixed=solve_window(t,measured,a,ranges,sigma,"profile")
    poly,information,objective,elapsed,diagnostic=solve_polynomial(t,measured,a,ranges,sigma)
    rows=[]
    for method,x,info,cost,seconds in (("fixed_bias",fixed.parameters,fixed.yaw_information,fixed.robust_objective,fixed.elapsed_s),
                                      ("free_bias",poly,information,objective,elapsed)):
        error=abs(np.arctan2(np.sin(x[-1]-truth[-1]),np.cos(x[-1]-truth[-1])))
        rows.append(dict(seed=seed,motion=motion,bias_std=bias_std,method=method,
            yaw_error_deg=float(np.rad2deg(error)),position_error_m=float(np.linalg.norm(x[:3]-truth[:3])),
            yaw_information=info,accepted=bool(info>=1/0.35**2),
            false_confident=bool(info>=1/0.35**2 and error>0.35),objective=cost,elapsed_s=seconds))
    return rows


def main():
    """Run the paired bias study with independent seeded Monte Carlo trials."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--seeds",type=int,default=200)
    parser.add_argument("--workers",type=int,default=20)
    args=parser.parse_args()
    jobs=[(seed,motion,bias_std) for seed in range(args.seeds)
          for motion in ("excited","constant_acceleration") for bias_std in (0.0,0.1,0.3)]
    rows=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for number,result in enumerate(pool.map(trial,jobs),1):
            rows.extend(result)
            if number%50==0:
                print(f"DONE {number}/{len(jobs)}",flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output,index=False)


if __name__=="__main__":
    main()
