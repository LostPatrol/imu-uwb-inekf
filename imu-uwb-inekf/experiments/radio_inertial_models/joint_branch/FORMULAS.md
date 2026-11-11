<!-- Joint endpoint model definitions and the fixed comparison protocol. -->

Let $\tau_k=t_k-T\le0$, and let $d_k$ be the measured-IMU displacement reanchored at the endpoint, so $d(T)=0$. The unknown endpoint variables are $p_T,v_T,\psi$.

The unconstrained K2 control is
\[
 p_k=p_T+\tau_kv_T+R_z(\psi)d_k+\tfrac12\tau_k^2c_W.
\]
The three-dimensional world acceleration defect $c_W$ has no prior.

The rotating-body accelerometer-bias model is
\[
 p_k=p_T+\tau_kv_T+R_z(\psi)[d_k-B_k\delta b],
 \qquad \delta b\sim\mathcal N(0,0.10^2I)\ \mathrm{m/s^2}.
\]
Here $B_k=J_2(t_k)-J_2(T)-\tau_kJ_1(T)$, $J_1(t)=\int_0^tQ(s)\,ds$, and $J_2(t)=\int_0^t(t-s)Q(s)\,ds$. The original held-rotation integration and prior are reused. No initial-tilt correction is fitted in this comparison.

The original range is $y_k=\|p_k-a_k\|+\mu_{a_k}(D_k,u_k)+e_k$, with $D_k=\|p_k-a_k\|$ and $u_k=(p_k-a_k)/D_k$. The static model uses the training per-anchor median. The geometric model uses
\[
 \mu_a(D,u)=b_a+c_a(D-\bar D_a)+\kappa_a^Tu,
\]
with robust ridge coefficients frozen from the earlier six recordings. The six-training-log leave-one-out choice of ridge is not repeated on evaluation data. Every joint model uses the same static training MAD scale $\sigma_a$.

For the two AR variants, define $r_k=[D_k+\mu_{a_k}(D_k,u_k)-y_k]/\sigma_{a_k}$ and the preceding event $k^-$ for the same anchor. Then
\[
 \phi_k=\rho_{a_k}^{(t_k-t_{k^-})/\widetilde{\Delta t}_{a_k}},\qquad
 w_k=\frac{r_k-\phi_kr_{k^-}}{\sqrt{\max(1-\phi_k^2,0.05)}}.
\]
The first event of each anchor uses $w_k=r_k$. Static-AR and geometry-AR retain their separately trained correlations. The objective is $\sum_k\operatorname{Huber}_2(w_k)$ plus $\|\delta b/0.10\|^2$ only for BiasPrior. Without AR, $w_k=r_k$.

All joint variants share the static-calibrated CV position/velocity seed, zero defect or bias, and four yaw starts $(-\pi,-\pi/2,0,\pi/2)$. Each start has at most 120 function evaluations. The Gaussian prior is quadratic; Huber applies only to radio factors. No test truth affects initialization, weights, parameters, starts, or acceptance. The same 0.20 s common IMU/TWR correction and 1.5 s static prefix are retained.

The physical endpoint error is $\|\widehat p_T-p_T^{\mathrm{OptiTrack}}\|$. Six later logs each contribute eleven overlapping windows, in all8 and lower4 layouts. These are descriptive development comparisons in the same room; the windows are not independent statistical replicates.
