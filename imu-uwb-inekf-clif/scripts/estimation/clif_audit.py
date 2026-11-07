"""Numerical audit of the body-frame lifted TWR structure behind CLIF.

The audit needs no dataset: it builds an analytic, bounded truth trajectory and
checks the claims that decide whether a single unified filter is worth building.

1. closed-form quadratic dynamics match finite differences of the defining
   products (s00 = 1/2 z_i^T z_i and so on);
2. the angular-rate contributions that the product rule produces are large but
   cancel, so the quadratic subsystem has constant coefficients and is driven
   only by z and the specific force;
3. under identical physical perturbations the body-frame lifting predicts
   half-squared ranges far better than the world-frame lifting that consumes
   R_hat f + g;
4. the s states drift off their defining quadratics under Euler
   discretisation and under state-only updates;
5. the single-anchor lifted system leaves directions unobservable, which fixes
   what multi-anchor and IMU coupling must supply.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Gravity in the world frame, m/s^2 (z up).
GRAVITY_WORLD = np.array([0.0, 0.0, -9.81])

# Eight anchor positions of a 10 m x 10 m x 3 m room, metres.
ANCHORS = np.array(
    [
        [-5.0, -5.0, 0.0],
        [5.0, -5.0, 0.0],
        [-5.0, 5.0, 0.0],
        [5.0, 5.0, 0.0],
        [-5.0, -5.0, 3.0],
        [5.0, -5.0, 3.0],
        [-5.0, 5.0, 3.0],
        [5.0, 5.0, 3.0],
    ],
    dtype=np.float64,
)

# Number of lifted entries for one anchor: z_i, z_v, z_g, s00, s01, s02, s11, s12, s22.
LIFTED_DIM = 15

# Index of s00 inside the lifted vector.
S00_INDEX = 9

# Quadratic state names in the order used by every report entry.
QUADRATIC_NAMES = ("s00", "s01", "s02", "s11", "s12", "s22")

# Typical residual IMU errors used to perturb both liftings.
GYRO_BIAS_ERROR = np.array([0.01, -0.008, 0.006])  # rad/s
ACCEL_BIAS_ERROR = np.array([0.10, -0.08, 0.06])  # m/s^2
INITIAL_ATTITUDE_ERROR = np.array([0.0, 0.0, 0.30])  # rad, ~17 deg heading

# Sampling step used when comparing the two liftings, seconds.
COMPARISON_DT = 0.01


def skew(vector: np.ndarray) -> np.ndarray:
    """Return the 3x3 skew-symmetric cross-product matrix of a 3-vector."""
    x, y, z = float(vector[0]), float(vector[1]), float(vector[2])
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def exp_so3(phi: np.ndarray) -> np.ndarray:
    """Rodrigues formula for SO(3), with a second-order branch near zero."""
    angle = float(np.linalg.norm(phi))
    cross = skew(phi)
    if angle < 1e-8:
        return np.eye(3) + cross + 0.5 * cross @ cross
    return np.eye(3) + (math.sin(angle) / angle) * cross + ((1.0 - math.cos(angle)) / angle**2) * cross @ cross


@dataclass
class Scenario:
    """Bounded truth trajectory: analytic position, integrated attitude."""

    duration_s: float = 10.0
    truth_dt: float = 1e-3
    lever_arm: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def position(self, t: float) -> np.ndarray:
        """Bounded flight path staying several metres inside the anchor box."""
        return np.array([2.0 * math.sin(0.6 * t), 1.8 * math.sin(0.45 * t + 0.4), 1.4 + 0.5 * math.sin(0.8 * t)])

    def velocity(self, t: float) -> np.ndarray:
        """Analytic time derivative of position."""
        return np.array([1.2 * math.cos(0.6 * t), 0.81 * math.cos(0.45 * t + 0.4), 0.4 * math.cos(0.8 * t)])

    def acceleration(self, t: float) -> np.ndarray:
        """Analytic second time derivative of position."""
        return np.array([-0.72 * math.sin(0.6 * t), -0.3645 * math.sin(0.45 * t + 0.4), -0.32 * math.sin(0.8 * t)])

    def omega(self, t: float) -> np.ndarray:
        """Body angular rate in rad/s, multi-axis so Omega changes sign."""
        return np.array([0.6 * math.sin(0.7 * t), 0.45 * math.cos(0.5 * t), 0.9 + 0.25 * math.sin(0.3 * t)])

    def start_attitude(self) -> np.ndarray:
        """Initial attitude: yaw 60 deg, pitch 5 deg."""
        yaw, pitch = math.radians(60.0), math.radians(5.0)
        rz = np.array([[math.cos(yaw), -math.sin(yaw), 0.0], [math.sin(yaw), math.cos(yaw), 0.0], [0.0, 0.0, 1.0]])
        ry = np.array([[math.cos(pitch), 0.0, math.sin(pitch)], [0.0, 1.0, 0.0], [-math.sin(pitch), 0.0, math.cos(pitch)]])
        return rz @ ry


def build_truth(scenario: Scenario) -> dict:
    """Evaluate the truth trajectory and every lifted quantity on it."""
    steps = int(round(scenario.duration_s / scenario.truth_dt)) + 1
    times = np.arange(steps) * scenario.truth_dt
    rotation = scenario.start_attitude()

    rotations = np.empty((steps, 3, 3))
    positions = np.empty((steps, 3))
    velocities = np.empty((steps, 3))
    accelerations = np.empty((steps, 3))
    forces = np.empty((steps, 3))
    omegas = np.empty((steps, 3))

    for k in range(steps):
        t = float(times[k])
        position = scenario.position(t)
        acceleration = scenario.acceleration(t)
        rotations[k] = rotation
        positions[k] = position
        velocities[k] = scenario.velocity(t)
        accelerations[k] = acceleration
        omegas[k] = scenario.omega(t)
        # Specific force is the world acceleration rotated into the body frame.
        forces[k] = rotation.T @ (acceleration - GRAVITY_WORLD)
        if k + 1 < steps:
            rotation = rotation @ exp_so3(scenario.omega(t + 0.5 * scenario.truth_dt) * scenario.truth_dt)

    displacement = ANCHORS[None, :, :] - positions[:, None, :]
    return {
        "t": times,
        "R": rotations,
        "v": velocities,
        "p": positions,
        "a": accelerations,
        "omega": omegas,
        "f": forces,
        "z_anchor": np.einsum("kji,kaj->kai", rotations, displacement),
        "z_velocity": np.einsum("kji,kj->ki", rotations, velocities),
        "z_gravity": -np.einsum("kji,j->ki", rotations, GRAVITY_WORLD),
    }


def quadratic_states(z_anchor, z_velocity, z_gravity) -> dict:
    """Evaluate the six quadratic forms at every sample; s11..s22 are global."""
    return {
        "s00": 0.5 * np.einsum("kai,kai->ka", z_anchor, z_anchor),
        "s01": 0.5 * np.einsum("kai,ki->ka", z_anchor, z_velocity),
        "s02": 0.5 * np.einsum("kai,ki->ka", z_anchor, z_gravity),
        "s11": 0.5 * np.einsum("ki,ki->k", z_velocity, z_velocity),
        "s12": 0.5 * np.einsum("ki,ki->k", z_velocity, z_gravity),
        "s22": 0.5 * np.einsum("ki,ki->k", z_gravity, z_gravity),
    }


def quadratic_rates(z_anchor, z_velocity, z_gravity, force, quad) -> dict:
    """Closed-form rates of the quadratic states; contains no angular rate."""
    # s11/s12/s22 are global per sample, so widen them for the per-anchor rows.
    s11 = np.asarray(quad["s11"]).reshape(-1, 1)
    s12 = np.asarray(quad["s12"]).reshape(-1, 1)
    s22 = np.asarray(quad["s22"]).reshape(-1)
    return {
        "s00": -2.0 * quad["s01"],
        "s01": -s11 - np.asarray(quad["s02"]) + 0.5 * np.einsum("kai,ki->ka", z_anchor, force),
        "s02": -s12 + np.zeros_like(np.asarray(quad["s02"])),
        "s11": -2.0 * s12.reshape(-1) + np.einsum("ki,ki->k", force, z_velocity),
        "s12": -s22 + 0.5 * np.einsum("ki,ki->k", force, z_gravity),
        "s22": np.zeros_like(s22),
    }


def check_closed_form(truth: dict, quad: dict) -> dict:
    """Compare analytic rates against central differences of the products."""
    dt = float(truth["t"][1] - truth["t"][0])
    analytic = quadratic_rates(truth["z_anchor"], truth["z_velocity"], truth["z_gravity"], truth["f"], quad)
    report = {}
    for name in QUADRATIC_NAMES:
        values = quad[name]
        numeric = (values[2:] - values[:-2]) / (2.0 * dt)
        error = np.abs(np.asarray(analytic[name])[1:-1] - numeric)
        scale = max(float(np.max(np.abs(numeric))), float(np.max(np.abs(analytic[name]))))
        if scale < 1e-9:
            # s22 has an identically zero rate, so only round-off is measured.
            report[name] = {"max_abs_error": float(np.max(error)), "scaled_error": 0.0, "trivially_constant": True}
            continue
        report[name] = {"max_abs_error": float(np.max(error)), "scaled_error": float(np.max(error) / scale)}
    report["pass"] = all(item["scaled_error"] < 1e-6 for item in report.values() if isinstance(item, dict))
    return report


def check_omega_cancellation(truth: dict, quad: dict) -> dict:
    """Size the angular-rate terms the product rule produces and would retain."""
    cross = np.array([skew(omega) for omega in truth["omega"]])
    z_i = truth["z_anchor"]
    z_v = truth["z_velocity"]
    z_g = truth["z_gravity"]

    # Each entry is one half of a cancelling pair, or an identically zero
    # quadratic form for the diagonal states.
    terms = {
        "s00": np.einsum("kai,kij,kaj->ka", z_i, cross, z_i),
        "s01": np.einsum("kai,kij,kj->ka", z_i, cross, z_v),
        "s02": np.einsum("kai,kij,kj->ka", z_i, cross, z_g),
        "s11": np.einsum("ki,kij,kj->k", z_v, cross, z_v),
        "s12": np.einsum("ki,kij,kj->k", z_v, cross, z_g),
        "s22": np.einsum("ki,kij,kj->k", z_g, cross, z_g),
    }
    dt = float(truth["t"][1] - truth["t"][0])
    analytic = quadratic_rates(z_i, z_v, z_g, truth["f"], quad)

    report = {}
    for name in QUADRATIC_NAMES:
        values = quad[name]
        numeric = (values[2:] - values[:-2]) / (2.0 * dt)
        fd_error = float(np.max(np.abs(np.asarray(analytic[name])[1:-1] - numeric)))
        term = float(np.max(np.abs(terms[name])))
        if term > 1e-6:
            # A retained angular-rate term of this size would break the match.
            report[name] = {
                "omega_term_magnitude": term,
                "fd_abs_error": fd_error,
                "detectability_ratio": fd_error / term,
                "verdict": "cancelled_in_pair",
            }
        else:
            # Diagonal states vanish identically because z^T Omega z = 0.
            report[name] = {"omega_term_magnitude": term, "fd_abs_error": fd_error, "verdict": "identically_zero"}
    report["note"] = (
        "diagonal states vanish because z^T Omega z = 0; cross states cancel in pairs. "
        "The finite-difference residual stays far below the term size, so a retained term would be visible."
    )
    report["pass"] = all(
        item.get("detectability_ratio", 0.0) < 1e-3 for item in report.values() if isinstance(item, dict)
    )
    return report


def check_constant_subsystem() -> dict:
    """Verify the s block keeps constant coefficients across Omega and force."""
    rows = {name: [] for name in QUADRATIC_NAMES}
    for trial in range(8):
        rng = np.random.default_rng(1000 + trial)
        transition = lifted_transition(float(trial), rng.normal(0.0, 1.0, 3), rng.normal(0.0, 5.0, 3))
        for name, row in zip(QUADRATIC_NAMES, range(S00_INDEX, LIFTED_DIM)):
            rows[name].append(transition[row, S00_INDEX:].copy())
    spread = {name: float(np.max(np.ptp(np.vstack(values), axis=0))) for name, values in rows.items()}
    return {"s_block_is_constant": bool(max(spread.values()) < 1e-12), "max_variation": spread, "pass": bool(max(spread.values()) < 1e-12)}


def _rk4_step(function, state, t, dt):
    """One classical Runge-Kutta step for a callable f(t, state)."""
    k1 = function(t, state)
    k2 = function(t + 0.5 * dt, state + 0.5 * dt * k1)
    k3 = function(t + 0.5 * dt, state + 0.5 * dt * k2)
    k4 = function(t + dt, state + dt * k3)
    return state + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _sample_index(t: float, dt: float, steps: int) -> int:
    """Nearest sample index for a time inside the truth arrays."""
    return min(max(int(round(t / dt)), 0), steps - 1)


def _interp(values: np.ndarray, t: float, dt: float) -> np.ndarray:
    """Linear interpolation of a sampled 3-vector with edge clamping."""
    position = t / dt
    index = min(max(int(position), 0), len(values) - 2)
    weight = min(max(position - index, 0.0), 1.0)
    return values[index] * (1.0 - weight) + values[index + 1] * weight


def propagate_body(truth: dict, gyro_error, accel_error, dt: float) -> dict:
    """Run the body-frame lifting from perturbed IMU only, with no attitude state."""
    steps = len(truth["t"])
    stride = int(round(dt / (truth["t"][1] - truth["t"][0])))
    count = steps // stride
    truth_dt = float(truth["t"][1] - truth["t"][0])

    z_anchor = truth["z_anchor"][0].copy()
    z_velocity = truth["z_velocity"][0].copy()
    z_gravity = truth["z_gravity"][0].copy()
    quad = quadratic_states(z_anchor[None], z_velocity[None], z_gravity[None])
    s00 = np.array([quad["s00"][0, i] for i in range(len(ANCHORS))])
    s01 = np.array([quad["s01"][0, i] for i in range(len(ANCHORS))])
    s02 = np.array([quad["s02"][0, i] for i in range(len(ANCHORS))])
    s11, s12, s22 = float(quad["s11"][0]), float(quad["s12"][0]), float(quad["s22"][0])
    state = np.concatenate([z_anchor.reshape(-1), z_velocity, z_gravity, s00, s01, s02, [s11, s12, s22]])

    def derivative(t, y):
        z_a = y[0 : 3 * len(ANCHORS)].reshape(-1, 3)
        z_v = y[3 * len(ANCHORS) : 3 * len(ANCHORS) + 3]
        z_g = y[3 * len(ANCHORS) + 3 : 3 * len(ANCHORS) + 6]
        q00 = y[3 * len(ANCHORS) + 6 : 4 * len(ANCHORS) + 6]
        q01 = y[4 * len(ANCHORS) + 6 : 5 * len(ANCHORS) + 6]
        q02 = y[5 * len(ANCHORS) + 6 : 6 * len(ANCHORS) + 6]
        q11, q12, q22 = y[-3], y[-2], y[-1]

        cross = skew(_interp(truth["omega"], t, truth_dt) + gyro_error)
        force = _interp(truth["f"], t, truth_dt) + accel_error
        derivatives = np.empty_like(y)
        derivatives[0 : 3 * len(ANCHORS)] = (-z_a @ cross.T - z_v).reshape(-1)
        derivatives[3 * len(ANCHORS) : 3 * len(ANCHORS) + 3] = -cross @ z_v - z_g + force
        derivatives[3 * len(ANCHORS) + 3 : 3 * len(ANCHORS) + 6] = -cross @ z_g
        derivatives[3 * len(ANCHORS) + 6 : 4 * len(ANCHORS) + 6] = -2.0 * q01
        derivatives[4 * len(ANCHORS) + 6 : 5 * len(ANCHORS) + 6] = -q11 - q02 + 0.5 * (z_a @ force)
        derivatives[5 * len(ANCHORS) + 6 : 6 * len(ANCHORS) + 6] = -q12
        derivatives[-3] = -2.0 * q12 + float(force @ z_v)
        derivatives[-2] = -q22 + 0.5 * float(force @ z_g)
        derivatives[-1] = 0.0
        return derivatives

    history = np.empty((count, len(ANCHORS)))
    for k in range(count):
        history[k] = state[3 * len(ANCHORS) + 6 : 4 * len(ANCHORS) + 6]
        if k + 1 == count:
            break
        state = _rk4_step(derivative, state, k * dt, dt)
    return {"t": np.arange(count) * dt, "s00": history}


def propagate_world(truth: dict, attitude_error, gyro_error, accel_error, dt: float) -> dict:
    """Run the world-frame lifting with a perturbed attitude and IMU."""
    steps = len(truth["t"])
    stride = int(round(dt / (truth["t"][1] - truth["t"][0])))
    count = steps // stride
    truth_dt = float(truth["t"][1] - truth["t"][0])

    rotation = truth["R"][0] @ exp_so3(np.asarray(attitude_error, dtype=float))
    position = truth["p"][0].copy()
    velocity = truth["v"][0].copy()
    q = 0.5 * float(position @ position)
    s = float(position @ velocity)
    h = float(velocity @ velocity)
    # The attitude rides inside the integrator so that a zero-perturbation run
    # is limited by RK4 truncation instead of by a frozen attitude per step.
    state = np.concatenate([position, velocity, [q, s, h], rotation.reshape(-1)])

    def derivative(t, y):
        pos, vel, quads = y[0:3], y[3:6], y[6:9]
        rot = y[9:18].reshape(3, 3)
        force = _interp(truth["f"], t, truth_dt) + accel_error
        world_accel = rot @ force + GRAVITY_WORLD
        rotation_rate = rot @ skew(_interp(truth["omega"], t, truth_dt) + gyro_error)
        return np.concatenate(
            [
                vel,
                world_accel,
                [quads[1], quads[2] + float(world_accel @ pos), 2.0 * float(world_accel @ vel)],
                rotation_rate.reshape(-1),
            ]
        )

    history = {"q": np.empty(count), "p": np.empty((count, 3))}
    for k in range(count):
        history["q"][k] = state[6]
        history["p"][k] = state[0:3]
        if k + 1 == count:
            break
        state = _rk4_step(derivative, state, k * dt, dt)
    return history


def half_squared_range_truth(truth: dict, dt: float) -> np.ndarray:
    """Truth half-squared ranges sampled at the comparison rate."""
    stride = int(round(dt / float(truth["t"][1] - truth["t"][0])))
    count = len(truth["t"]) // stride
    return quadratic_states(truth["z_anchor"], truth["z_velocity"], truth["z_gravity"])["s00"][::stride][:count]


def world_half_squared_range(history: dict, anchors=ANCHORS) -> np.ndarray:
    """Half-squared range implied by the world-frame lifting."""
    return history["q"][:, None] - history["p"] @ anchors.T + 0.5 * np.sum(anchors**2, axis=1)[None, :]


def check_zero_perturbation(truth: dict, dt: float) -> dict:
    """Propagators must reproduce truth when no error is injected."""
    truth_s00 = half_squared_range_truth(truth, dt)
    body = propagate_body(truth, np.zeros(3), np.zeros(3), dt)["s00"]
    world = world_half_squared_range(propagate_world(truth, np.zeros(3), np.zeros(3), np.zeros(3), dt))
    body_error = float(np.max(np.abs(body - truth_s00)))
    world_error = float(np.max(np.abs(world - truth_s00)))
    return {
        "body_max_abs_error_m2": body_error,
        "world_max_abs_error_m2": world_error,
        # The residual is integrator truncation, not a structural mismatch.
        "pass": bool(body_error < 0.05 and world_error < 0.05),
    }


def check_attitude_immunity(truth: dict, dt: float = COMPARISON_DT) -> dict:
    """Compare predicted range accuracy of both liftings under equal errors."""
    cases = {
        "attitude_error_only": (INITIAL_ATTITUDE_ERROR, np.zeros(3), np.zeros(3)),
        "gyro_bias_only": (np.zeros(3), GYRO_BIAS_ERROR, np.zeros(3)),
        "accel_bias_only": (np.zeros(3), np.zeros(3), ACCEL_BIAS_ERROR),
        "combined": (INITIAL_ATTITUDE_ERROR, GYRO_BIAS_ERROR, ACCEL_BIAS_ERROR),
    }
    truth_s00 = half_squared_range_truth(truth, dt)
    ranges = np.sqrt(2.0 * truth_s00)
    scale = float(np.median(ranges))
    report = {}
    for name, (attitude_error, gyro_error, accel_error) in cases.items():
        world = world_half_squared_range(propagate_world(truth, attitude_error, gyro_error, accel_error, dt))
        body = propagate_body(truth, gyro_error, accel_error, dt)["s00"]
        entry = {}
        for label, predicted in (("world", world), ("body", body)):
            error = predicted - truth_s00
            entry[label] = {
                "rmse_half_squared_m2": float(np.sqrt(np.mean(error**2))),
                "max_abs_half_squared_m2": float(np.max(np.abs(error))),
                "range_equivalent_rmse_m": float(np.sqrt(np.mean(error**2)) / scale),
            }
        entry["body_over_world_rmse"] = entry["body"]["rmse_half_squared_m2"] / max(entry["world"]["rmse_half_squared_m2"], 1e-12)
        report[name] = entry
    report["perturbations"] = {
        "gyro_bias_rad_s": GYRO_BIAS_ERROR.tolist(),
        "accel_bias_m_s2": ACCEL_BIAS_ERROR.tolist(),
        "initial_attitude_rad": INITIAL_ATTITUDE_ERROR.tolist(),
    }
    # Structural claim: an initial attitude error cannot reach the body lifting,
    # while bias-induced errors stay comparable between the two liftings.
    report["pass"] = bool(
        report["attitude_error_only"]["body_over_world_rmse"] < 1e-3
        and report["combined"]["body_over_world_rmse"] <= 1.0
    )
    return report


def _integrate_lifted_chain(truth: dict, dt: float, integrator: str) -> np.ndarray:
    """Integrate the lifted chain and return the manifold residual over time."""
    stride = int(round(dt / float(truth["t"][1] - truth["t"][0])))
    count = len(truth["t"]) // stride
    truth_dt = float(truth["t"][1] - truth["t"][0])
    anchors = len(ANCHORS)

    z_anchor = truth["z_anchor"][0].copy()
    z_velocity = truth["z_velocity"][0].copy()
    z_gravity = truth["z_gravity"][0].copy()
    quad = quadratic_states(z_anchor[None], z_velocity[None], z_gravity[None])
    s00 = np.array([quad["s00"][0, i] for i in range(anchors)])
    s01 = np.array([quad["s01"][0, i] for i in range(anchors)])
    s02 = np.array([quad["s02"][0, i] for i in range(anchors)])
    state = np.concatenate(
        [z_anchor.reshape(-1), z_velocity, z_gravity, s00, s01, s02, [float(quad["s11"][0]), float(quad["s12"][0]), float(quad["s22"][0])]]
    )

    def derivative(t, y):
        z_a = y[0 : 3 * anchors].reshape(-1, 3)
        z_v = y[3 * anchors : 3 * anchors + 3]
        z_g = y[3 * anchors + 3 : 3 * anchors + 6]
        q00 = y[3 * anchors + 6 : 4 * anchors + 6]
        q01 = y[4 * anchors + 6 : 5 * anchors + 6]
        q02 = y[5 * anchors + 6 : 6 * anchors + 6]
        q11, q12, q22 = y[-3], y[-2], y[-1]
        index = _sample_index(t, truth_dt, len(truth["t"]))
        cross = skew(truth["omega"][index])
        force = truth["f"][index]
        out = np.empty_like(y)
        out[0 : 3 * anchors] = (-z_a @ cross.T - z_v).reshape(-1)
        out[3 * anchors : 3 * anchors + 3] = -cross @ z_v - z_g + force
        out[3 * anchors + 3 : 3 * anchors + 6] = -cross @ z_g
        out[3 * anchors + 6 : 4 * anchors + 6] = -2.0 * q01
        out[4 * anchors + 6 : 5 * anchors + 6] = -q11 - q02 + 0.5 * (z_a @ force)
        out[5 * anchors + 6 : 6 * anchors + 6] = -q12
        out[-3] = -2.0 * q12 + float(force @ z_v)
        out[-2] = -q22 + 0.5 * float(force @ z_g)
        out[-1] = 0.0
        return out

    residuals = np.empty(count)
    for k in range(count):
        z_a = state[0 : 3 * anchors].reshape(-1, 3)
        q00 = state[3 * anchors + 6 : 4 * anchors + 6]
        residuals[k] = float(np.max(np.abs(q00 - 0.5 * np.sum(z_a**2, axis=1))))
        if k + 1 == count:
            break
        if integrator == "rk4":
            state = _rk4_step(derivative, state, k * dt, dt)
        else:
            state = state + derivative(k * dt, state) * dt
    return residuals


def check_redundancy_drift(truth: dict, euler_dt: float = 1e-3, update_sigma_m2: float = 0.35) -> dict:
    """Compare manifold drift of Euler and RK4 propagation with one update."""
    euler = _integrate_lifted_chain(truth, euler_dt, "euler")
    rk4 = _integrate_lifted_chain(truth, euler_dt, "rk4")
    return {
        "dt_s": euler_dt,
        "euler_peak_residual_m2": float(np.max(euler)),
        "euler_final_residual_m2": float(euler[-1]),
        "rk4_final_residual_m2": float(rk4[-1]),
        "update_sigma_m2": float(update_sigma_m2),
        # The lifted chain is a fifth-order integrator chain, so the residual
        # a projection must repair is compared with what one update carries.
        "euler_to_update_ratio": float(np.max(euler) / update_sigma_m2),
        "rk4_to_update_ratio": float(rk4[-1] / update_sigma_m2),
        "note": "s carries half-squared ranges, hence m^2",
        "pass": bool(rk4[-1] < 0.01 * update_sigma_m2),
    }


def lifted_transition(t: float, omega: np.ndarray, force: np.ndarray) -> np.ndarray:
    """Jacobian F(t) of the 15-dim single-anchor lifted system."""
    del t  # F depends only on the inputs at that time
    cross = skew(omega)
    matrix = np.zeros((LIFTED_DIM, LIFTED_DIM))
    matrix[0:3, 0:3] = -cross
    matrix[0:3, 3:6] = -np.eye(3)
    matrix[3:6, 3:6] = -cross
    matrix[3:6, 6:9] = -np.eye(3)
    matrix[6:9, 6:9] = -cross
    matrix[S00_INDEX, S00_INDEX + 1] = -2.0
    matrix[S00_INDEX + 1, 0:3] = 0.5 * force
    matrix[S00_INDEX + 1, S00_INDEX + 2] = -1.0
    matrix[S00_INDEX + 1, S00_INDEX + 3] = -1.0
    matrix[S00_INDEX + 2, S00_INDEX + 4] = -1.0
    matrix[S00_INDEX + 3, 3:6] = force
    matrix[S00_INDEX + 3, S00_INDEX + 4] = -2.0
    matrix[S00_INDEX + 4, 6:9] = 0.5 * force
    matrix[S00_INDEX + 4, S00_INDEX + 5] = -1.0
    return matrix


def check_observability(truth: dict, dt: float = COMPARISON_DT) -> dict:
    """Observability Gramian of one anchor's half-squared range output."""
    truth_dt = float(truth["t"][1] - truth["t"][0])
    stride = int(round(dt / truth_dt))
    count = len(truth["t"]) // stride
    output_row = np.zeros(LIFTED_DIM)
    output_row[S00_INDEX] = 1.0

    phi = np.eye(LIFTED_DIM)
    gramian = np.zeros((LIFTED_DIM, LIFTED_DIM))
    for k in range(count):
        index = _sample_index(k * dt, truth_dt, len(truth["t"]))
        transition = lifted_transition(0.0, truth["omega"][index], truth["f"][index])
        phi = phi + transition @ phi * dt
        observed = output_row @ phi
        gramian += np.outer(observed, observed) * dt

    eigenvalues = np.linalg.eigvalsh(0.5 * (gramian + gramian.T))
    threshold = eigenvalues[-1] * 1e-10
    positive = eigenvalues[eigenvalues > threshold]
    return {
        "dimension": LIFTED_DIM,
        "largest_eigenvalue": float(eigenvalues[-1]),
        "smallest_positive_eigenvalue": float(positive[0]) if positive.size else 0.0,
        "condition_number": float(eigenvalues[-1] / positive[0]) if positive.size else float("inf"),
        "rank_at_1e-10_relative": int(positive.size),
        "note": "single-anchor range only; z_g and s22 need IMU coupling to be observable",
    }


def build_figure(report: dict, output: Path) -> None:
    """Plot the decisive audit panels."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(15.0, 4.2))
    cases = ["attitude_error_only", "gyro_bias_only", "accel_bias_only", "combined"]
    world = [max(report["attitude_immunity"][case]["world"]["max_abs_half_squared_m2"], 1e-9) for case in cases]
    body = [max(report["attitude_immunity"][case]["body"]["max_abs_half_squared_m2"], 1e-9) for case in cases]
    index = np.arange(len(cases))
    axes[0].bar(index - 0.2, world, width=0.4, label="world-frame lifting")
    axes[0].bar(index + 0.2, body, width=0.4, label="body-frame lifting")
    axes[0].set_yscale("log")
    axes[0].set_xticks(index)
    axes[0].set_xticklabels([name.replace("_", "\n") for name in cases], fontsize=8)
    axes[0].set_ylabel("max |half-squared range error| (m$^2$)")
    axes[0].set_title("Predicted TWR error under identical perturbations")
    axes[0].legend(fontsize=8)

    drift = report["redundancy_drift"]
    labels = ["Euler\n(peak)", "Euler\n(final)", "RK4\n(final)", "1 update\n(1$\\sigma$)"]
    values = [
        max(drift["euler_peak_residual_m2"], 1e-12),
        max(drift["euler_final_residual_m2"], 1e-12),
        max(drift["rk4_final_residual_m2"], 1e-12),
        drift["update_sigma_m2"],
    ]
    axes[1].bar(labels, values, color=["tab:red", "tab:orange", "tab:green", "tab:gray"])
    axes[1].set_yscale("log")
    axes[1].set_ylabel("|s$_{00}$ - 0.5 z$^T$z| (m$^2$)")
    axes[1].set_title("Manifold residual vs one update")

    observability = report["observability"]
    axes[2].axis("off")
    axes[2].text(
        0.02,
        0.98,
        (
            f"lifted dim            : {observability['dimension']}\n"
            f"rank @1e-10 relative  : {observability['rank_at_1e-10_relative']}\n"
            f"log10 condition       : {math.log10(max(observability['condition_number'], 1.0)):.2f}\n"
            f"smallest positive eig : {observability['smallest_positive_eigenvalue']:.3e}\n\n"
            "single-anchor range output only"
        ),
        va="top",
        family="monospace",
        fontsize=9,
    )

    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def main() -> int:
    """Run every audit check, persist JSON plus a figure, and report pass/fail."""
    parser = argparse.ArgumentParser(description="Audit the body-frame lifted TWR structure")
    parser.add_argument("--duration", type=float, default=10.0, help="truth duration in seconds")
    parser.add_argument("--output-dir", type=Path, default=Path("results/clif_audit"))
    parser.add_argument("--no-figure", action="store_true", help="skip figure generation")
    args = parser.parse_args()

    truth = build_truth(Scenario(duration_s=args.duration))
    quad = quadratic_states(truth["z_anchor"], truth["z_velocity"], truth["z_gravity"])

    report = {
        "zero_perturbation": check_zero_perturbation(truth, COMPARISON_DT),
        "closed_form": check_closed_form(truth, quad),
        "omega_cancellation": check_omega_cancellation(truth, quad),
        "constant_subsystem": check_constant_subsystem(),
        "attitude_immunity": check_attitude_immunity(truth),
        "redundancy_drift": check_redundancy_drift(truth),
        "observability": check_observability(truth),
    }
    report["pass_all"] = bool(
        report["zero_perturbation"]["pass"]
        and report["closed_form"]["pass"]
        and report["omega_cancellation"]["pass"]
        and report["constant_subsystem"]["pass"]
        and report["attitude_immunity"]["pass"]
        and report["redundancy_drift"]["pass"]
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "clif_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    if not args.no_figure:
        build_figure(report, args.output_dir / "clif_audit.png")

    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["pass_all"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
