"""Frozen translation-only radio shadow with strictly past block prediction.

The native strong tracker remains untouched, including its heading and covariance.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

# All domains share these rules; none are selected from evaluation outcomes.
RULES = dict(block_s=.25, window_s=1., horizon_s=2., warmup_blocks=2,
             winning_blocks=2, huber_sigma=2., min_fit_rows=12, max_nfev=40,
             velocity_floor_mps=1., velocity_scale_factor=2., rank_rtol=1e-7,
             ambiguous_cost_gap=4., ambiguity_sigma=2.)


def _radio(position, plus, minus):
    """Evaluate native TWR or directed plus-minus TDoA and its position gradient."""
    difference = position-plus
    distance = np.maximum(np.linalg.norm(difference, axis=1), 1e-10)
    gradient = difference/distance[:, None]
    if minus is not None:
        difference = position-minus
        other = np.maximum(np.linalg.norm(difference, axis=1), 1e-10)
        gradient -= difference/other[:, None]
        distance -= other
    return distance, gradient


def _huber(residual):
    """Return twice the common fixed-sigma Huber loss per raw observation."""
    absolute = np.abs(residual)
    threshold = RULES["huber_sigma"]
    return np.where(absolute <= threshold, residual**2, 2*threshold*absolute-threshold**2)


def _loss(squared, radio_count):
    """Robustify raw radio while retaining a Gaussian velocity-scale prior."""
    output = np.vstack((squared.copy(), np.ones_like(squared), np.zeros_like(squared)))
    threshold = RULES["huber_sigma"]
    index = np.flatnonzero((np.arange(len(squared)) < radio_count) & (squared > threshold**2))
    root = np.sqrt(squared[index])
    output[0, index] = 2*threshold*root-threshold**2
    output[1, index] = threshold/root
    output[2, index] = -.5*threshold/root**3
    return output


def _fit(stamps, plus, minus, y, sigma, lever, left, seed, speed_scale):
    """Jointly fit radio-only 3-D p/v and reject singular or competing mirror modes."""
    if len(stamps) < RULES["min_fit_rows"]:
        return None, dict(reason="insufficient_prefix", rows=len(stamps))
    tau = stamps-left
    span = float(np.ptp(stamps))
    if span <= 1e-8:
        return None, dict(reason="single_time_prefix", rows=len(stamps))
    anchors = plus if minus is None else np.vstack((plus, minus))
    center = anchors.mean(axis=0)
    _, anchor_singular, axes = np.linalg.svd(anchors-center, full_matrices=False)
    velocity_std = max(RULES["velocity_floor_mps"], RULES["velocity_scale_factor"]*speed_scale)

    def residual(parameters):
        """Use raw radio and a zero-centered velocity scale, without a position prior."""
        predicted, _ = _radio(parameters[:3]+tau[:, None]*parameters[3:]+lever, plus, minus)
        return np.r_[(predicted-y)/sigma, parameters[3:]/velocity_std]

    def jacobian(parameters):
        """Differentiate native radio coordinates and the weak velocity prior."""
        _, gradient = _radio(parameters[:3]+tau[:, None]*parameters[3:]+lever, plus, minus)
        native = np.column_stack((gradient, tau[:, None]*gradient))/sigma[:, None]
        return np.vstack((native, np.column_stack((np.zeros((3, 3)), np.eye(3)/velocity_std))))

    # Reflection across the weakest anchor plane exposes common range mirror modes.
    reflected = seed.copy()
    normal = axes[-1]
    antenna = seed[:3]+lever[-1]
    reflected[:3] -= 2*normal*np.dot(antenna-center, normal)
    reflected[3:] -= 2*normal*np.dot(seed[3:], normal)
    if np.linalg.norm(reflected[:3]-seed[:3]) < 1e-6:
        reflected[:3] += max(1., speed_scale)*normal
    fits = [least_squares(residual, initial, jac=jacobian,
            loss=lambda values:_loss(values, len(stamps)), max_nfev=RULES["max_nfev"])
            for initial in (seed, reflected)]
    fits.sort(key=lambda fit:fit.cost)
    best = fits[0]
    info = dict(rows=len(stamps), fit_success=bool(best.success),
                nfev=sum(int(fit.nfev) for fit in fits), cost=float(best.cost),
                alternate_fit_success=bool(fits[1].success), alternate_cost=float(fits[1].cost),
                velocity_prior_std_mps=float(velocity_std))
    if not best.success or not np.all(np.isfinite(best.x)):
        return None, dict(info, reason="fit_not_converged")
    native = jacobian(best.x)[:len(stamps)]
    # Compare p and v as displacements over the observed prefix duration.
    scale = np.r_[np.ones(3), np.ones(3)/span]
    scaled = native*scale[None, :]
    nuisance = scaled[:, [2, 5]]
    u, nuisance_singular, _ = np.linalg.svd(nuisance, full_matrices=False)
    nuisance_rank = int(np.sum(nuisance_singular > max(nuisance_singular[0], 1e-12)*RULES["rank_rtol"]))
    horizontal = scaled[:, [0, 1, 3, 4]]
    horizontal -= u[:, :nuisance_rank]@(u[:, :nuisance_rank].T@horizontal)
    _, singular, vt = np.linalg.svd(horizontal, full_matrices=False)
    if singular[-1] <= max(singular[0], 1e-12)*RULES["rank_rtol"]:
        return None, dict(info, reason="horizontal_pv_rank_deficient")
    horizontal_scale = scale[[0, 1, 3, 4]]
    covariance = (horizontal_scale[:, None]*vt.T/singular**2)@vt*horizontal_scale[None, :]
    endpoint_map = np.column_stack((np.eye(2), RULES["block_s"]*np.eye(2)))
    position_std = float(np.sqrt(max(0., np.linalg.eigvalsh(covariance[:2, :2])[-1],
        np.linalg.eigvalsh(endpoint_map@covariance@endpoint_map.T)[-1])))
    info["horizontal_position_std_m"] = position_std
    info["anchor_geometry_rank"] = int(np.sum(anchor_singular > max(anchor_singular[0], 1e-12)*RULES["rank_rtol"]))
    other = fits[1]
    difference = best.x-other.x
    separation = float(max(np.linalg.norm(difference[:2]),
                           np.linalg.norm(difference[:2]+RULES["block_s"]*difference[3:5])))
    if (other.success and 2*(other.cost-best.cost) <= RULES["ambiguous_cost_gap"]
            and separation > RULES["ambiguity_sigma"]*position_std):
        return None, dict(info, reason="competing_mirror", separation_m=separation)
    return best.x.copy(), dict(info, reason="valid")


def _jumps(time, k_pos, k_vel, position, velocity, active, resume):
    """Report sampled block-boundary display steps and added K-relative offset jumps."""
    block = np.floor((time-resume)/RULES["block_s"]+1e-9).astype(int)
    indices = np.flatnonzero((block[1:] != block[:-1]) & (active[1:] | active[:-1]))+1
    output = []
    for index in indices:
        shift = position[index]-k_pos[index]
        old_shift = position[index-1]-k_pos[index-1]
        output.append(dict(time=float(time[index]), from_c=bool(active[index-1]), to_c=bool(active[index]),
            display_step_m=float(np.linalg.norm(position[index]-position[index-1])),
            added_offset_jump_m=float(np.linalg.norm(shift-old_shift)),
            added_velocity_jump_mps=float(np.linalg.norm((velocity[index]-k_vel[index])-
                                                         (velocity[index-1]-k_vel[index-1])))))
    return output


def recover(time, k_pos, k_vel, lever_world, obs_time, plus_xyz, minus_xyz, observations, sigma, resume):
    """Return causal short-horizon translation overlays and serializable block evidence."""
    time, k_pos, k_vel = (np.asarray(value, dtype=float) for value in (time, k_pos, k_vel))
    if len(time) and np.any(np.diff(time) < 0):
        raise ValueError("K sample times must be nondecreasing")
    position, velocity = k_pos.copy(), k_vel.copy()
    always_position, always_velocity = k_pos.copy(), k_vel.copy()
    use_c, available = np.zeros(len(time), dtype=bool), np.zeros(len(time), dtype=bool)
    lever = np.zeros_like(k_pos) if lever_world is None else np.broadcast_to(np.asarray(lever_world, dtype=float), k_pos.shape)
    stamps, plus, y = (np.asarray(value, dtype=float) for value in (obs_time, plus_xyz, observations))
    minus = None if minus_xyz is None else np.asarray(minus_xyz, dtype=float)
    noise = np.broadcast_to(np.asarray(sigma, dtype=float), y.shape)
    scoped = np.flatnonzero((stamps >= resume) & (stamps < resume+RULES["horizon_s"]))
    order = scoped[np.argsort(stamps[scoped], kind="stable")]
    stamps, plus, y, noise = (value[order] for value in (stamps, plus, y, noise))
    minus = None if minus is None else minus[order]
    valid = (np.isfinite(stamps) & np.isfinite(y) & np.isfinite(noise) & (noise > 0)
             & np.all(np.isfinite(plus), axis=1))
    if minus is not None:
        valid &= np.all(np.isfinite(minus), axis=1)
    excluded = int((~valid).sum())
    stamps, plus, y, noise = (value[valid] for value in (stamps, plus, y, noise))
    minus = None if minus is None else minus[valid]
    # Known lever directions use strictly earlier K samples, including inside prefixes.
    lever_index = np.searchsorted(time, stamps, side="left")-1
    radio_lever = np.zeros((len(stamps), 3))
    if len(time):
        radio_lever[lever_index >= 0] = lever[np.maximum(lever_index[lever_index >= 0], 0)]
    blocks, streak, previous = [], 0, None
    for block in range(round(RULES["horizon_s"]/RULES["block_s"])):
        left = float(resume+block*RULES["block_s"])
        right = float(resume+(block+1)*RULES["block_s"])
        past = int(np.searchsorted(time, left, side="left")-1)
        predicted_k = None
        if past >= 0:
            predicted_k = np.r_[k_pos[past]+(left-time[past])*k_vel[past], k_vel[past]]
        train = (stamps >= max(resume, left-RULES["window_s"])) & (stamps < left)
        if lever_world is not None:
            train &= lever_index >= 0
        hold = (stamps >= left) & (stamps < right)
        candidate, fit_info = None, dict(reason="no_strictly_past_K_seed", rows=int(train.sum()))
        if predicted_k is not None:
            history = (time < left) & (time >= left-RULES["window_s"])
            speed_scale = float(np.median(np.linalg.norm(k_vel[history], axis=1))) if history.any() else float(np.linalg.norm(k_vel[past]))
            seed = predicted_k.copy() if previous is None else np.r_[previous[1][:3]+(left-previous[0])*previous[1][3:], previous[1][3:]]
            candidate, fit_info = _fit(stamps[train], plus[train], None if minus is None else minus[train],
                y[train], noise[train], radio_lever[train], left, seed, speed_scale)
        selected = bool(candidate is not None and streak >= RULES["winning_blocks"])
        samples = (time >= left) & (time < right)
        if candidate is not None:
            predicted_position = candidate[:3]+(time[samples]-left)[:, None]*candidate[3:]
            always_position[samples], always_velocity[samples] = predicted_position, candidate[3:]
            # Only horizontal translation is displayed; native height/vertical velocity stay exact.
            always_position[samples, 2], always_velocity[samples, 2] = k_pos[samples, 2], k_vel[samples, 2]
            available[samples] = True
            if selected:
                position[samples], velocity[samples], use_c[samples] = predicted_position, candidate[3:], True
                position[samples, 2], velocity[samples, 2] = k_pos[samples, 2], k_vel[samples, 2]
            previous = (left, candidate.copy())
        info = dict(block=block, left=left, right=right, train_rows=int(train.sum()), validation_rows=int(hold.sum()),
                    fit=fit_info, selected_c=selected, streak_before=streak,
                    loss_k=None, loss_c=None, gain=None)
        if candidate is not None and predicted_k is not None and hold.any():
            offset = (stamps[hold]-left)[:, None]
            frozen_lever = lever[past]
            forecast_c = candidate[:3]+offset*candidate[3:]+frozen_lever
            forecast_k = predicted_k[:3]+offset*predicted_k[3:]+frozen_lever
            forecast_c[:, 2] = forecast_k[:, 2]
            pred_c, _ = _radio(forecast_c, plus[hold], None if minus is None else minus[hold])
            pred_k, _ = _radio(forecast_k, plus[hold], None if minus is None else minus[hold])
            loss_c = float(np.mean(_huber((pred_c-y[hold])/noise[hold])))
            loss_k = float(np.mean(_huber((pred_k-y[hold])/noise[hold])))
            gain = loss_k-loss_c
            info.update(loss_k=loss_k, loss_c=loss_c, gain=gain)
            # Same-block evidence can only change the following block's display choice.
            streak = streak+1 if block >= RULES["warmup_blocks"] and gain > 1e-10 else 0
        else:
            streak = 0
        info["streak_after"] = streak
        blocks.append(info)
    diagnostics = dict(rules=RULES.copy(), resume=float(resume), return_to_k=float(resume+RULES["horizon_s"]),
        excluded_radio_rows=excluded, blocks=blocks, switches=_jumps(time, k_pos, k_vel, position, velocity, use_c, resume),
        always_c_switches=_jumps(time, k_pos, k_vel, always_position, always_velocity, available, resume),
        always_c_scope="same first 2 seconds; valid C without prediction gate, then exact K",
        translation_components="XY only; z and vz always equal native K",
        modifies_heading=False, modifies_covariance=False, shared_posterior_fusion=False)
    return dict(pos=position, vel=velocity, always_c_pos=always_position, always_c_vel=always_velocity,
                use_c=use_c, always_c_available=available, diagnostics=diagnostics)
