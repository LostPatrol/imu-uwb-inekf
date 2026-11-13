"""Independent Jiang-style 10 s range/IMU batch initializer.

This implements a full pose/velocity/bias graph, not the unavailable author
D-iSAM2 implementation. Sparse batch continuation is explicitly an adaptation.
"""
from __future__ import annotations

from dataclasses import dataclass
import time
import numpy as np
from numba import njit
from scipy.linalg import solve_triangular
from scipy.optimize._numdiff import approx_derivative, group_columns
from scipy.sparse import lil_matrix, eye, coo_matrix
from scipy.sparse.linalg import spsolve
from scipy.spatial.transform import Rotation

# Physical gravity and numerical differentiation/damping settings.
GRAVITY = np.array([0., 0., -9.81])
BIAS_STEP = 1e-5
ROTATION_EPS = 1e-8


@njit(cache=True)
def skew(v):
    """Return the cross-product matrix."""
    return np.array([[0., -v[2], v[1]], [v[2], 0., -v[0]], [-v[1], v[0], 0.]])


@njit(cache=True)
def exp_so3(v):
    """Evaluate the rotation exponential, including its small-angle limit."""
    a = np.linalg.norm(v)
    k = skew(v)
    if a < ROTATION_EPS:
        return np.eye(3) + k + .5 * k @ k
    return np.eye(3) + np.sin(a)/a*k + (1.-np.cos(a))/(a*a)*k@k


@njit(cache=True)
def log_so3(r):
    """Evaluate SO(3) logarithm with an explicit near-pi branch."""
    angle = np.arccos(min(1., max(-1., (np.trace(r)-1.)*.5)))
    v = np.array([r[2, 1]-r[1, 2], r[0, 2]-r[2, 0], r[1, 0]-r[0, 1]])
    if angle < 1e-7:
        return .5*v
    if np.pi-angle < 1e-5:
        axis = np.sqrt(np.maximum((np.diag(r)+1.)*.5, 0.))
        j = np.argmax(axis)
        if axis[j] > 1e-8:
            for i in range(3):
                if i != j:
                    axis[i] = (r[i, j]+r[j, i])/(4.*axis[j])
        if axis@v < 0.:
            axis=-axis
        return angle*axis
    return angle/(2.*np.sin(angle))*v


@njit(cache=True)
def left_jacobian(v):
    """Map rotation-vector increments to left tangent increments."""
    angle=np.linalg.norm(v)
    k=skew(v)
    if angle < 1e-6:
        return np.eye(3)+.5*k+k@k/6.
    return np.eye(3)+(1.-np.cos(angle))/angle**2*k+(angle-np.sin(angle))/angle**3*k@k


@njit(cache=True)
def inverse_left_jacobian(v):
    """Differentiate SO(3) log with respect to a left tangent increment."""
    angle=np.linalg.norm(v)
    k=skew(v)
    if angle < 1e-6:
        return np.eye(3)-.5*k+k@k/12.
    coefficient=(1.-.5*angle/np.tan(.5*angle))/angle**2
    return np.eye(3)-.5*k+coefficient*k@k


@njit(cache=True)
def retract_state(x,step):
    """Apply a left SO(3) retraction and canonicalize its stored rotation vectors."""
    updated=x+step
    for i in range(len(x)//15):
        start=i*15
        updated[start:start+3]=log_so3(exp_so3(step[start:start+3])@exp_so3(x[start:start+3]))
    return updated


@njit(cache=True)
def integrate_segment(dt, acc, gyro, bias, sigma_acc, sigma_gyro):
    """Preintegrate held body samples and full rotation/velocity/position covariance."""
    r, v, p, cov = np.eye(3), np.zeros(3), np.zeros(3), np.zeros((9, 9))
    for k in range(len(dt)):
        h = dt[k]
        a, w = acc[k]-bias[3:], gyro[k]-bias[:3]
        f = np.eye(9)
        f[:3, :3] = exp_so3(-w*h)
        f[3:6, :3] = -r@skew(a)*h
        f[6:9, :3] = -.5*r@skew(a)*h*h
        f[6:9, 3:6] = np.eye(3)*h
        q = np.zeros((9, 9))
        q[:3, :3] = np.eye(3)*sigma_gyro**2*h
        q[3:6, 3:6] = np.eye(3)*sigma_acc**2*h
        q[6:9, 6:9] = np.eye(3)*sigma_acc**2*h**3/3.
        q[3:6, 6:9] = np.eye(3)*sigma_acc**2*h*h/2.
        q[6:9, 3:6] = q[3:6, 6:9]
        cov = f@cov@f.T+q
        p += v*h+.5*(r@a)*h*h
        v += (r@a)*h
        r = r@exp_so3(w*h)
    return r, v, p, cov


@njit(cache=True)
def graph_residual(x, delta_t, delta_r, delta_v, delta_p, bias_jac, whitening,
                   bias_scale, bias_decay, event_node, anchor, observed, sigma,
                   r0, bias0, lever, gravity, prior_state, prior_sigma, framework):
    """Whiten all preintegration, Gauss-Markov bias, radio and prior factors."""
    n = len(x)//15
    state = x.reshape((n, 15))
    out = np.zeros((n-1)*15+len(observed)+15)
    rotations = np.empty((n, 3, 3))
    for i in range(n):
        rotations[i] = exp_so3(state[i, :3])@r0
    for i in range(n-1):
        a, b, h = state[i], state[i+1], delta_t[i]
        db = a[9:15]-bias0
        correction = bias_jac[i]@db
        relative = (delta_r[i]@exp_so3(correction[:3])).T@rotations[i].T@rotations[i+1]
        e = np.empty(9)
        e[:3] = log_so3(relative)
        e[3:6] = rotations[i].T@(b[3:6]-a[3:6]-gravity*h)-delta_v[i]-correction[3:6]
        e[6:9] = rotations[i].T@(b[6:9]-a[6:9]-a[3:6]*h-.5*gravity*h*h)-delta_p[i]-correction[6:9]
        out[i*15:i*15+9] = whitening[i]@e
        out[i*15+9:i*15+15] = ((b[9:15]-bias0)-bias_decay[i]*(a[9:15]-bias0))/bias_scale[i]
    cursor = (n-1)*15
    for j in range(len(observed)):
        # Equal-prior max mixture: Gaussian LOS and positive half-Cauchy NLOS.
        node = event_node[j]
        e = observed[j]-np.linalg.norm(state[node, 6:9]+rotations[node]@lever-anchor[j])
        u = e/sigma[j]
        rho = u*u
        if e >= 0.:
            # gamma=2*sigma is an explicitly chosen, unpublished model parameter.
            # 2/(pi*gamma) divided by the Gaussian peak is sqrt(2*pi)/pi.
            cauchy_rho = 2.*np.log(1.+u*u/4.)+2.*np.log(np.pi/np.sqrt(2.*np.pi))
            rho = min(rho, cauchy_rho)
        out[cursor+j] = np.sign(e)*np.sqrt(max(rho, 0.))
    cursor += len(observed)
    first = state[0]
    # Factory-calibrated bias prior is the only native state prior in the paper.
    out[cursor+9:cursor+15] = (first[9:15]-bias0)/prior_sigma[9:15]
    if framework:
        prior_rotation = exp_so3(prior_state[:3])@r0
        # The shared challenge supplies position and physical yaw, not tilt/velocity.
        yaw = np.arctan2(rotations[0, 1, 0], rotations[0, 0, 0])
        prior_yaw = np.arctan2(prior_rotation[1, 0], prior_rotation[0, 0])
        yaw_error = np.arctan2(np.sin(yaw-prior_yaw), np.cos(yaw-prior_yaw))
        out[cursor+2] = yaw_error/prior_sigma[2]
        out[cursor+6:cursor+9] = (first[6:9]-prior_state[6:9])/prior_sigma[6:9]
    return out


@njit(cache=True)
def graph_jacobian(x, delta_t, delta_r, delta_v, delta_p, bias_jac, whitening,
                   bias_scale, bias_decay, event_node, anchor, observed, sigma,
                   r0, bias0, lever, gravity, prior_state, prior_sigma, framework):
    """Differentiate the full graph with respect to left SO(3) tangent increments."""
    n=len(x)//15
    state=x.reshape((n,15))
    rotations=np.empty((n,3,3))
    tangent=np.empty((n,3,3))
    for i in range(n):
        rotations[i]=exp_so3(state[i,:3])@r0
        tangent[i]=np.eye(3)
    # Dense local blocks keep assembly simple; sparse conversion drops exact zeros.
    count=(n-1)*450+len(observed)*6+225
    rows=np.empty(count,dtype=np.int64)
    cols=np.empty(count,dtype=np.int64)
    data=np.empty(count)
    at=0
    for i in range(n-1):
        a,b,h=state[i],state[i+1],delta_t[i]
        correction=bias_jac[i]@(a[9:15]-bias0)
        relative=(delta_r[i]@exp_so3(correction[:3])).T@rotations[i].T@rotations[i+1]
        error_rotation=log_so3(relative)
        local=np.zeros((15,30))
        # Relative-rotation residual, with exact first-order bias correction derivative.
        rotation_part=inverse_left_jacobian(-error_rotation)@rotations[i+1].T
        local[:3,:3]=-rotation_part@tangent[i]
        local[:3,15:18]=rotation_part@tangent[i+1]
        local[:3,9:15]=-inverse_left_jacobian(error_rotation)@left_jacobian(-correction[:3])@bias_jac[i,:3]
        velocity_difference=b[3:6]-a[3:6]-gravity*h
        position_difference=b[6:9]-a[6:9]-a[3:6]*h-.5*gravity*h*h
        rit=rotations[i].T
        local[3:6,:3]=rit@skew(velocity_difference)@tangent[i]
        local[3:6,3:6]=-rit
        local[3:6,18:21]=rit
        local[3:6,9:15]=-bias_jac[i,3:6]
        local[6:9,:3]=rit@skew(position_difference)@tangent[i]
        local[6:9,3:6]=-rit*h
        local[6:9,6:9]=-rit
        local[6:9,21:24]=rit
        local[6:9,9:15]=-bias_jac[i,6:9]
        local[:9]=whitening[i]@local[:9]
        for j in range(6):
            local[9+j,9+j]=-bias_decay[i]/bias_scale[i,j]
            local[9+j,24+j]=1./bias_scale[i,j]
        for row in range(15):
            for col in range(30):
                rows[at],cols[at],data[at]=i*15+row,i*15+col,local[row,col]
                at+=1
    cursor=(n-1)*15
    for j in range(len(observed)):
        node=event_node[j]
        lever_world=rotations[node]@lever
        difference=state[node,6:9]+lever_world-anchor[j]
        distance=np.linalg.norm(difference)
        unit=difference/max(distance,1e-12)
        error=observed[j]-distance
        u=error/sigma[j]
        derivative=1./sigma[j]
        if error >= 0.:
            cauchy_rho=2.*np.log(1.+u*u/4.)+2.*np.log(np.pi/np.sqrt(2.*np.pi))
            if cauchy_rho < u*u:
                derivative=2.*u/((4.+u*u)*np.sqrt(cauchy_rho)*sigma[j])
        rotation_derivative=unit@skew(lever_world)@tangent[node]*derivative
        for k in range(3):
            rows[at],cols[at],data[at]=cursor+j,node*15+k,rotation_derivative[k]
            at+=1
            rows[at],cols[at],data[at]=cursor+j,node*15+6+k,-unit[k]*derivative
            at+=1
    cursor+=len(observed)
    local=np.zeros((15,15))
    for k in range(6):
        local[9+k,9+k]=1./prior_sigma[9+k]
    if framework:
        first_column=rotations[0,:,0]
        px,py,pz=first_column[0],first_column[1],first_column[2]
        denominator=max(px*px+py*py,1e-12)
        yaw_derivative=np.array([-px*pz,-py*pz,denominator])/denominator
        local[2,:3]=yaw_derivative@tangent[0]/prior_sigma[2]
        for k in range(3):
            local[6+k,6+k]=1./prior_sigma[6+k]
    for row in range(15):
        for col in range(15):
            rows[at],cols[at],data[at]=cursor+row,col,local[row,col]
            at+=1
    return rows,cols,data


@dataclass
class GraphConfig:
    """Frozen independent-reproduction parameters absent from the article."""
    sigma_acc: float = .10
    sigma_gyro: float = .01
    sigma_bg_rw: float = .0001
    sigma_ba_rw: float = .01
    bias_tau_s: float = 1000.
    max_iterations: int = 80
    relative_cost_tolerance: float = 1e-7
    jacobian: str = 'analytic'


class BatchGraph:
    """Prepare and solve a range-keyframed full inertial factor graph."""

    def __init__(self, imu_t, acc, gyro, event_t, event_anchor, event_range,
                 event_sigma, r0, bias0, config=None, horizon=None, lever=None, gravity=None):
        """Consume sensor arrays only; no truth arrays or altitude input are accepted."""
        self.config = config or GraphConfig()
        self.r0, self.bias0 = np.asarray(r0), np.asarray(bias0)
        self.lever = np.zeros(3) if lever is None else np.asarray(lever)
        self.gravity = GRAVITY.copy() if gravity is None else np.asarray(gravity)
        self.times = np.unique(np.r_[0., event_t, [] if horizon is None else [horizon]])
        self.event_node = np.searchsorted(self.times, event_t)
        self.anchor, self.observed, self.sigma = map(np.asarray, (event_anchor, event_range, event_sigma))
        self.dt = np.diff(self.times)
        n = len(self.dt)
        self.dr, self.dv, self.dp = np.empty((n, 3, 3)), np.empty((n, 3)), np.empty((n, 3))
        self.jac, self.whiten = np.empty((n, 9, 6)), np.empty((n, 9, 9))
        cfg = self.config
        for i, (start, end) in enumerate(zip(self.times[:-1], self.times[1:])):
            bounds = np.r_[start, imu_t[(imu_t > start) & (imu_t < end)], end]
            index = np.clip(np.searchsorted(imu_t, bounds[:-1], side='right')-1, 0, len(imu_t)-1)
            h = np.diff(bounds)
            args = (h, acc[index], gyro[index], self.bias0, cfg.sigma_acc, cfg.sigma_gyro)
            r, v, p, cov = integrate_segment(*args)
            self.dr[i], self.dv[i], self.dp[i] = r, v, p
            self.whiten[i] = solve_triangular(np.linalg.cholesky(cov+np.eye(9)*1e-15), np.eye(9), lower=True)
            for j in range(6):
                perturbed = self.bias0.copy()
                perturbed[j] += BIAS_STEP
                rr, vv, pp, _ = integrate_segment(h, acc[index], gyro[index], perturbed, cfg.sigma_acc, cfg.sigma_gyro)
                self.jac[i, :, j] = np.r_[log_so3(r.T@rr), vv-v, pp-p]/BIAS_STEP
        self.bias_decay = np.exp(-self.dt/cfg.bias_tau_s)
        # Stationary exact Gauss-Markov transition covariance from white drive.
        rw = np.r_[np.full(3, cfg.sigma_bg_rw), np.full(3, cfg.sigma_ba_rw)]
        self.bias_scale = np.sqrt(cfg.bias_tau_s/2.*(1.-self.bias_decay**2))[:, None]*rw
        self.sparsity = self._sparsity()
        self.groups = group_columns(self.sparsity)

    def _sparsity(self):
        """Declare consecutive-state factor dependencies for colored differentiation."""
        n = len(self.times)
        s = lil_matrix(((n-1)*15+len(self.observed)+15, n*15), dtype=int)
        for i in range(n-1):
            s[i*15:(i+1)*15, i*15:(i+2)*15] = 1
        cursor = (n-1)*15
        for j, node in enumerate(self.event_node):
            if np.linalg.norm(self.lever) > 0.:
                s[cursor+j, node*15:node*15+3] = 1
            s[cursor+j, node*15+6:node*15+9] = 1
        s[cursor+len(self.observed):, :15] = 1
        return s.tocsr()

    def initial_state(self, position, yaw_rad):
        """Generate a single IMU-propagated seed from the supplied pose guess."""
        x = np.zeros((len(self.times), 15))
        nominal_yaw = Rotation.from_matrix(self.r0).as_euler('ZYX')[0]
        tilt = Rotation.from_euler('z', -nominal_yaw).as_matrix()@self.r0
        r = Rotation.from_euler('z', yaw_rad).as_matrix()@tilt
        x[0, :3] = Rotation.from_matrix(r@self.r0.T).as_rotvec()
        x[0, 6:9], x[:, 9:15] = position, self.bias0
        for i, h in enumerate(self.dt):
            x[i+1, 6:9] = x[i, 6:9]+x[i, 3:6]*h+.5*self.gravity*h*h+r@self.dp[i]
            x[i+1, 3:6] = x[i, 3:6]+self.gravity*h+r@self.dv[i]
            r = r@self.dr[i]
            x[i+1, :3] = Rotation.from_matrix(r@self.r0.T).as_rotvec()
        return x

    def warm_state(self, old_times, old_state):
        """Copy only equal-time nodes, then propagate the latest causal optimized state."""
        initial=np.zeros((len(self.times),15))
        indices=np.searchsorted(self.times,old_times)
        valid=indices<len(self.times)
        valid[valid]=np.abs(self.times[indices[valid]]-old_times[valid])<1e-9
        initial[indices[valid]]=old_state[valid]
        last=int(indices[valid][-1])
        r=Rotation.from_rotvec(initial[last,:3]).as_matrix()@self.r0
        for i in range(last,len(self.dt)):
            h=self.dt[i]
            db=initial[i,9:15]-self.bias0
            correction=self.jac[i]@db
            initial[i+1,6:9]=initial[i,6:9]+initial[i,3:6]*h+.5*self.gravity*h*h+r@(self.dp[i]+correction[6:9])
            initial[i+1,3:6]=initial[i,3:6]+self.gravity*h+r@(self.dv[i]+correction[3:6])
            r=r@self.dr[i]@exp_so3(correction[:3])
            initial[i+1,:3]=Rotation.from_matrix(r@self.r0.T).as_rotvec()
            initial[i+1,9:15]=self.bias0+self.bias_decay[i]*db
        return initial

    def solve(self, initial, prior_sigma, framework=False, prior_state=None):
        """Run damped sparse batch optimization with accepted/rejected steps recorded."""
        start = time.perf_counter()
        prior = initial[0].copy() if prior_state is None else np.asarray(prior_state).copy()
        args = (self.dt, self.dr, self.dv, self.dp, self.jac, self.whiten,
                self.bias_scale, self.bias_decay, self.event_node, self.anchor,
                self.observed, self.sigma, self.r0, self.bias0, self.lever, self.gravity, prior, prior_sigma, framework)
        fun = lambda x: graph_residual(x, *args)
        x = initial.reshape(-1).copy()
        x = retract_state(x,np.zeros_like(x))
        residual = fun(x)
        initial_cost = cost = .5*float(residual@residual)
        damping, accepted, rejected, calls = 1e-3, 0, 0, 1
        reason = 'iteration_limit'
        for iteration in range(self.config.max_iterations):
            if self.config.jacobian == 'analytic':
                rows,cols,data=graph_jacobian(x,*args)
                j=coo_matrix((data,(rows,cols)),shape=self.sparsity.shape).tocsr()
                j.eliminate_zeros()
            else:
                local_fun=lambda step:fun(retract_state(x,step))
                j = approx_derivative(local_fun, np.zeros_like(x), method='2-point', f0=residual,
                                      sparsity=(self.sparsity, self.groups))
                calls += int(self.groups.max()+1)
            hessian, gradient = (j.T@j).tocsc(), j.T@residual
            diagonal = np.maximum(hessian.diagonal(), 1e-6)
            improved = False
            for attempt in range(14):
                step = spsolve(hessian+damping*eye(len(x), format='csc').multiply(diagonal), -gradient)
                trial = retract_state(x,step)
                rr = fun(trial)
                calls += 1
                new_cost = .5*float(rr@rr)
                if np.isfinite(new_cost) and new_cost < cost:
                    relative = (cost-new_cost)/max(cost, 1.)
                    x, residual, cost = trial, rr, new_cost
                    damping = max(damping/3., 1e-12)
                    accepted += 1
                    improved = True
                    break
                damping *= 10.
                rejected += 1
            if not improved:
                reason = 'no_decreasing_step'
                break
            if relative < self.config.relative_cost_tolerance:
                reason = 'relative_cost'
                break
        return dict(state=x.reshape((-1, 15)), times=self.times.copy(), wall_s=time.perf_counter()-start,
                    initial_cost=initial_cost, final_cost=cost, accepted_steps=accepted,
                    rejected_steps=rejected, residual_calls=calls, termination=reason,
                    iterations=iteration+1, finite=bool(np.isfinite(x).all()))
