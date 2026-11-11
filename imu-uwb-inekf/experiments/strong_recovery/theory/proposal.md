# Preserve the native backend; test a translation-only recovery shadow

This is a theoretical proposal, not a claim of measured improvement. No real recording, old report or MEMORY was read, and no threshold was fitted. The motivating observations supplied by the main task are that the old lift yields earlier position and that Oursland's original heading is already strong.

## Minimal architecture and the precise non-harm claim

Keep the original native InEKF K completely unchanged: same initialization, IMU/radio packets, update order, outlier gates and covariance. Run a separate candidate C from a short radio prefix, changing position or observable position/velocity only. Reuse K's already-available attitude and bias conditionally; do not replace its heading merely because a different initialization is available.

The exact testable invariant is K_new(k)=K_baseline(k) for every sample, including X, b, P and accepted/rejected packet identities. It follows by induction from the same deterministic recursion and inputs. It proves that the native backend is preserved. It does not prove that a selector's published output is never worse: with ambiguous or biased radio, no causal selector has that universal guarantee.

The lift can publish an explicitly provisional early position while the native backend continues. Before two branches can be distinguished, do not describe this provisional output as an accepted complete navigation state. Candidate readiness and the strong backend's readiness are separate quantities.

## Causal evidence and packet accounting

Form C on a prefix ending at t0, before reading a subsequent radio block. Each new packet i is predicted by both branches using only earlier measurements and currently arrived IMU. Record both pre-update innovations, then let each branch consume that raw packet once. Select at a later already-arrived block boundary. An early decision can be made at 2 s by reserving an earlier part of that same 2 s as the fitting prefix; the final candidate state includes the scored suffix only after its predictions were recorded.

Parallel mutually exclusive hypotheses may both use a packet. That is not double counting within either hypothesis. Do not starve K of its original first-two-second/recovery radio merely to make C independent: doing so removes the original backend's range-time and lever-arm orientation information and invalidates the non-harm comparison. The prohibited operation is feeding C's same-radio p/v posterior back into an already raw-radio-updated K as a new independent measurement, or scoring the same packet again as fresh validation evidence.

In particular, do not build a prefix predictor using a terminal R that has already incorporated the validation radio. Use the pre-update K attitude/bias at each prediction time. Shared historic information creates correlated errors but does not leak a future validation value.

For each block b, retain paired raw prediction losses

    D_b = mean_i [rho((y_i-h(x_K,i^-))/sigma_i)
                  -rho((y_i-h(x_C,i^-))/sigma_i)].

Use a common fixed radio scale, all arrived packets and the same robust clipping. Do not score only the packets accepted by a branch's own gate. D_b>0 favors C, but a few correlated blocks do not make it a calibrated confidence bound. If Gaussian predictive likelihood is used instead, include both innovation quadratic form and log(det S): NIS alone rewards an arbitrarily inflated covariance. Compare position-only and position/velocity candidates separately rather than assuming one dominates.

## What translation is actually identifiable?

For an endpoint-reanchored known inertial shape s_i, model

    p_i = p_T + tau_i*v_T + s_i;
    J_i = [n_i^T, tau_i*n_i^T]/sigma_i.

For TDoA replace n_i by n_plus,i-n_minus,i; include the known rotating tag lever arm in the raw measurement. With surveyed anchors and a trustworthy conditional R/b, no fresh heading solve is necessary to estimate p/v. This remains conditional on the inertial input and calibration being adequate.

Partition I=J^T J into p/v blocks. Velocity information after free position is

    I_v|p = I_vv - I_vp*I_pp^dagger*I_pv.

Weak z or bias/tilt directions require corresponding nuisance elimination, not simply deleting their columns and declaring the remaining inverse a marginal covariance. Near the reference time and stable geometry, velocity information scales as O(f*T^3), whereas position information is typically O(f*T). Thus early good position need not imply a safe new velocity. To compare condition numbers, scale state units as [p,T*v]; a raw p/v Hessian condition number depends on arbitrary metre/second units.

A position reset with the old velocity gives approximately

    e_p(t0+h) = e_p(t0) + h*e_v(t0) + .5*h^2*e_a.

An old 5 m/s velocity error recreates 5 m position error in one second even after a perfect position reset. Conversely, replacing a good native velocity by a weak short-window fit can make recovery worse. The candidate set should therefore include retain, p-only and observable-p/v, using predeclared rules. If a direction is unobservable, retain it or give it an explicit prior; a pseudoinverse's zero eigenvalue is not zero uncertainty.

## Covariance and the cost of discarding relative geometry

Replacing p/v while keeping R/b does not preserve the original cross-covariance. Work in the physical error chart before converting back to the InEKF group. Let a=(attitude,bias) and d=(position,velocity), with conservative marginal MSE bounds P_a,P_d but unknown cross-covariance. A simple bound valid for every compatible cross-covariance is

    P_joint <= diag((1+eps)*P_a, (1+1/eps)*P_d), eps>0.

This follows from covariance Cauchy-Schwarz; eps=1 gives 2*diag(P_a,P_d). Merely setting cross blocks to zero and keeping the old marginal blocks does not give a joint bound. Inflate only a disposable candidate branch, leaving K intact. This bound assumes conservative input marginal MSE bounds; it cannot repair an overconfident Hessian, unmodeled bias or selection-dependent miscalibration. CI/SCI are alternatives if their correlation assumptions are actually satisfied, not a reason to pretend shared IMU noise is independent.

Forcing a recovery branch's current p to a single anchor-relative range basin can lose tangential position/velocity or choose a mirror solution, even when its residual is small. Collinear vertical anchors admit an exact common-yaw gauge: rotating p,v,R about that line preserves all ranges and body IMU. Far-field nearly parallel range directions give a weak version, with little tangent information. Conversely, a grossly wrong native p can cause its valid reacquired ranges to fail the NIS gate, making a new p useful to restore the correct range directions and updates. High NIS alone does not distinguish this from NLOS bias.

## Minimal falsifiable evaluation

Before looking at a new recording's errors, fix prefix lengths, decision block boundaries and candidate types. Keep healthy continuous ranging as well as outages and startup. Record:

- Exact unchanged-K invariant, including packet/update identities; any mismatch falsifies the non-harm implementation.
- First available position, first complete state, and decision latency separately; include the provisional interval and no-output durations.
- Paired pre-update D_b, native/candidate accepted-radio counts and maximum innovation; do not censor NIS rejects or failed candidates.
- Candidate delta-p/delta-v, information eigenspaces and predicted next-1/5/15-second displacement differences.
- Fixed-clock future XY/Z error, peak error and velocity error; report healthy-case degradation and worst tails, not only recovery median.
- Original attitude SO(3)/yaw drift, including indirect covariance-driven changes if a repair is ever committed.
- Conditional accepted errors and covariance coverage separately; selection rates and block scores are not physical success probabilities.

Small counterexamples are implemented in `counterexamples.py`: unknown cross-covariance, old velocity undoing a position repair, exact range/IMU yaw ambiguity, NIS rejection after position loss, and covariance-inflation gaming of NIS scores. No real data are required.

## Recent primary references

1. C. Cros, P.-O. Amblard, C. Prieur, J.-F. Da Rocha, “Revisiting Split Covariance Intersection: Correlated Components and Optimality,” IEEE Transactions on Automatic Control 70(7), 4593–4607, 2025. DOI `10.1109/TAC.2025.3530854`; author manuscript `arXiv:2501.07915`. It treats conservative fusion when shared process/measurement information makes correlations partially known. It does not establish that this project's fitted covariances are conservative.
2. F. Jiang et al., “Robust Indoor Localization with Ranging-IMU Fusion,” ICRA 2024, DOI `10.1109/ICRA57147.2024.10611274`. It supplies a strong range/IMU baseline and an asymmetric radio-noise treatment; the authors explicitly discuss difficulty identifying NLOS causally and recovery with accumulated nonlinear information. It is not a theorem that short-block switching always helps.

Both were checked against author-hosted primary manuscripts and publication metadata. No paper is being invoked as a proof of the proposed selector's performance.
