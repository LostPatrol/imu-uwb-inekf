"""Protect horizontal outputs with causal, height-independent native radio bounds.

The estimator state is immutable. The metric projection claim is conditional on
the measured, motion-dilated bounds containing the physical horizontal truth.
"""

from math import hypot, sqrt

import numpy as np

# Only the immediate recovery output is under test.
RECOVERY_SECONDS = 2.0
# Older measurements have weaker displacement bounds; keep the most recent per channel.
MAX_AGE_SECONDS = 0.25
# Assumed upper speed of the indoor vehicle body, in metres per second.
BODY_SPEED_BOUND = 10.0
# Fixed one-sided radio margin in native standard deviations, not a simultaneous confidence guarantee.
SIGMA_MARGIN = 3.0


def bound_xy(time, k_pos, lever_radius, obs_time, plus_xyz, minus_xyz, observations, sigma, resume):
    """Return protected positions using only native packets already received at each output time."""
    time = np.asarray(time, float)
    k_pos = np.asarray(k_pos, float)
    obs_time = np.asarray(obs_time, float)
    plus = np.asarray(plus_xyz, float)
    minus = None if minus_xyz is None else np.asarray(minus_xyz, float)
    y = np.asarray(observations, float)
    sigma = np.broadcast_to(np.asarray(sigma, float), y.shape)
    lever_radius = float(lever_radius)
    if (time.ndim != 1 or k_pos.shape != (len(time), 3) or y.ndim != 1 or
            obs_time.shape != y.shape or plus.shape != (len(y), 3) or
            (minus is not None and minus.shape != plus.shape) or
            not np.isfinite(lever_radius) or lever_radius < 0 or not np.isfinite(resume)):
        raise ValueError("Expected aligned position times, native XYZ anchors and a finite lever radius.")
    if np.any(~np.isfinite(time)) or np.any(np.diff(time) < 0) or np.any(~np.isfinite(obs_time)) or np.any(np.diff(obs_time) < 0):
        raise ValueError("State and radio clocks must be finite and nondecreasing in arrival order.")

    pos = k_pos.copy()
    active = np.zeros(len(time), dtype=bool)
    displacement = np.zeros(len(time))
    first = int(np.searchsorted(obs_time, resume))
    last = int(np.searchsorted(obs_time, resume + RECOVERY_SECONDS))
    packets = []
    for j in range(first, last):
        if not (np.isfinite(y[j]) and np.isfinite(sigma[j]) and sigma[j] > 0 and np.isfinite(plus[j]).all()):
            continue
        margin = SIGMA_MARGIN * sigma[j]
        if minus is None:
            if y[j] <= 0:
                continue
            anchor = tuple(plus[j])
            packets.append((obs_time[j], anchor, anchor, None, y[j] + margin + lever_radius))
        else:
            if not np.isfinite(minus[j]).all():
                continue
            # A positive range difference makes the minus anchor nearer; a negative one reverses it.
            far, near = (plus[j], minus[j]) if y[j] >= 0 else (minus[j], plus[j])
            delta = abs(y[j]) - margin - 2.0 * lever_radius
            if delta <= 0 or delta >= np.linalg.norm(far - near):
                continue
            channel = (tuple(plus[j]), tuple(minus[j]))
            packets.append((obs_time[j], channel, tuple(near), tuple(far), delta))

    latest, cursor = {}, 0
    steps = 0
    for i in range(np.searchsorted(time, resume), np.searchsorted(time, resume + RECOVERY_SECONDS)):
        now = time[i]
        while cursor < len(packets) and packets[cursor][0] <= now:
            latest[packets[cursor][1]] = cursor
            cursor += 1
        x, h = k_pos[i, :2]
        if not np.isfinite(x) or not np.isfinite(h):
            continue
        for j in sorted(latest.values()):
            stamp, _, near, far, delta = packets[j]
            age = now - stamp
            if age > MAX_AGE_SECONDS:
                continue
            dx, dy = x - near[0], h - near[1]
            radial = hypot(dx, dy)
            if far is None:
                # An absolute range bounds horizontal distance even if height and yaw are unknown.
                radius = delta + BODY_SPEED_BOUND * age
                if radial > radius:
                    fraction = radius / radial
                    x, h = near[0] + fraction * dx, near[1] + fraction * dy
                    steps += 1
            else:
                # Eliminate the unbounded height first; a weak pair yields no horizontal constraint.
                separation = delta - 2.0 * BODY_SPEED_BOUND * age
                vertical = abs(far[2] - near[2])
                if separation <= vertical:
                    continue
                horizontal = sqrt((separation - vertical) * (separation + vertical))
                ax, ay = near[0] - far[0], near[1] - far[1]
                violation = horizontal * radial - ax * dx - ay * dy - (ax * ax + ay * ay - horizontal * horizontal) / 2.0
                if violation <= 0 or radial == 0:
                    continue
                # A supporting half-plane contains the entire convex branch; one projection suffices for contraction.
                gx, gy = horizontal * dx / radial - ax, horizontal * dy / radial - ay
                norm2 = gx * gx + gy * gy
                if norm2 > np.finfo(float).tiny:
                    step = violation / norm2
                    x, h = x - step * gx, h - step * gy
                    steps += 1
        pos[i, :2] = x, h
        displacement[i] = hypot(x - k_pos[i, 0], h - k_pos[i, 1])
        active[i] = displacement[i] > 0
    return {"pos": pos, "active": active, "diagnostics": {
        "valid_packet_count": len(packets), "projection_steps": steps,
        "active_samples": int(np.sum(active)), "max_shift_m": float(np.max(displacement, initial=0.0)),
    }}
