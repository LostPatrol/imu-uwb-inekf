"""Core of the single-covariance body-frame lifted TWR filter (CLIF).

Four pieces are kept separate so each can be verified on its own:

* :func:`lifted_quantities` maps a nominal state to z_i = R^T (a_i - p), z_v,
  z_g and the bias-corrected specific force;
* :func:`exact_lifted_step` advances the quadratic states with the exact
  nilpotent series, because the chain matrix is nilpotent of index five;
* :func:`quadratic_from_quantities` builds the initial lifted states from the
  defining products;
* :func:`anchor_driving_jacobian` couples the lifted block to the 15-dim
  invariant error, so ranges can correct attitude, position and bias inside a
  single covariance.

Error convention: the truth state is ``X_true = Exp(xi) X_hat`` with
``xi = [theta, v, p, b_g, b_a]``.  Every Jacobian is written for that
convention and is checked against central differences in the tests.
"""

from __future__ import annotations

import math

import numpy as np

# Gravity vector in the world frame, m/s^2.
GRAVITY_WORLD = np.array([0.0, 0.0, -9.81])

# Dimension of the invariant error state: theta, v, p, b_g, b_a.
ERROR_DIM = 15

# Number of quadratic states carried for one anchor block.
QUADRATIC_COUNT = 6

# Fixed lifted-chain order used by every array in this module.
CHAIN_INDEX = {"s00": 0, "s01": 1, "s02": 2, "s11": 3, "s12": 4, "s22": 5}

# Chain matrix is nilpotent of index five: the longest path is four hops.
_NILPOTENT_ORDER = 5


def skew(vector: np.ndarray) -> np.ndarray:
    """Return the 3x3 skew-symmetric cross-product matrix of a 3-vector."""
    x, y, z = float(vector[0]), float(vector[1]), float(vector[2])
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def so3_exp(phi: np.ndarray) -> np.ndarray:
    """Rodrigues formula with a second-order branch near zero."""
    angle = float(np.linalg.norm(phi))
    cross = skew(phi)
    if angle < 1e-8:
        return np.eye(3) + cross + 0.5 * cross @ cross
    return np.eye(3) + (math.sin(angle) / angle) * cross + ((1.0 - math.cos(angle)) / angle**2) * cross @ cross


def chain_matrix() -> np.ndarray:
    """Constant 6x6 system matrix of the quadratic chain."""
    matrix = np.zeros((QUADRATIC_COUNT, QUADRATIC_COUNT))
    matrix[CHAIN_INDEX["s00"], CHAIN_INDEX["s01"]] = -2.0
    matrix[CHAIN_INDEX["s01"], CHAIN_INDEX["s02"]] = -1.0
    matrix[CHAIN_INDEX["s01"], CHAIN_INDEX["s11"]] = -1.0
    matrix[CHAIN_INDEX["s02"], CHAIN_INDEX["s12"]] = -1.0
    matrix[CHAIN_INDEX["s11"], CHAIN_INDEX["s12"]] = -2.0
    matrix[CHAIN_INDEX["s12"], CHAIN_INDEX["s22"]] = -1.0
    return matrix


CHAIN = chain_matrix()


def nilpotent_transition(dt: float) -> np.ndarray:
    """Exact exp(CHAIN * dt) from the nilpotent series, without expm."""
    result = np.eye(QUADRATIC_COUNT)
    term = np.eye(QUADRATIC_COUNT)
    for order in range(1, _NILPOTENT_ORDER):
        term = term @ (CHAIN * dt) / order
        result = result + term
    return result


def nilpotent_input_map(dt: float) -> np.ndarray:
    """Exact integral over one step of exp(CHAIN * (dt - tau)) dtau."""
    result = np.eye(QUADRATIC_COUNT) * dt
    term = np.eye(QUADRATIC_COUNT) * dt
    for order in range(1, _NILPOTENT_ORDER):
        term = term @ (CHAIN * dt) / (order + 1)
        result = result + term
    return result


def lifted_quantities(X: np.ndarray, specific_force_body: np.ndarray, anchors: np.ndarray) -> dict:
    """Map a nominal state and bias-corrected specific force to lifted blocks."""
    rotation = X[:3, :3]
    velocity = X[:3, 3]
    position = X[:3, 4]
    return {
        "z_anchor": (anchors - position[None, :]) @ rotation,
        "z_velocity": rotation.T @ velocity,
        # Guide convention z_g = -R^T g, which is what the chain equations assume.
        "z_gravity": -(rotation.T @ GRAVITY_WORLD),
        "z_force": np.asarray(specific_force_body, dtype=np.float64).reshape(3),
    }


def quadratic_from_quantities(quantities: dict) -> dict:
    """Initialise quadratic states from their defining products."""
    z_anchor = quantities["z_anchor"]
    z_velocity = quantities["z_velocity"]
    z_gravity = quantities["z_gravity"]
    return {
        "s00": 0.5 * np.sum(z_anchor**2, axis=1),
        "s01": 0.5 * (z_anchor @ z_velocity),
        "s02": 0.5 * (z_anchor @ z_gravity),
        # s11, s12 and s22 are global states, not per anchor.
        "s11": 0.5 * float(z_velocity @ z_velocity),
        "s12": 0.5 * float(z_velocity @ z_gravity),
        "s22": 0.5 * float(z_gravity @ z_gravity),
    }


def stacked_quadratic(quadratic: dict) -> np.ndarray:
    """Stack the chain per anchor, broadcasting the three global states."""
    anchors = len(np.atleast_1d(quadratic["s00"]))
    stacked = np.empty((anchors, QUADRATIC_COUNT))
    stacked[:, CHAIN_INDEX["s00"]] = np.atleast_1d(quadratic["s00"])
    stacked[:, CHAIN_INDEX["s01"]] = np.atleast_1d(quadratic["s01"])
    stacked[:, CHAIN_INDEX["s02"]] = np.atleast_1d(quadratic["s02"])
    stacked[:, CHAIN_INDEX["s11"]] = float(quadratic["s11"])
    stacked[:, CHAIN_INDEX["s12"]] = float(quadratic["s12"])
    stacked[:, CHAIN_INDEX["s22"]] = float(quadratic["s22"])
    return stacked


def unstack_quadratic(stacked: np.ndarray) -> dict:
    """Inverse of :func:`stacked_quadratic`; global states are taken once."""
    return {
        "s00": np.array(stacked[:, CHAIN_INDEX["s00"]], copy=True),
        "s01": np.array(stacked[:, CHAIN_INDEX["s01"]], copy=True),
        "s02": np.array(stacked[:, CHAIN_INDEX["s02"]], copy=True),
        "s11": float(stacked[0, CHAIN_INDEX["s11"]]),
        "s12": float(stacked[0, CHAIN_INDEX["s12"]]),
        "s22": float(stacked[0, CHAIN_INDEX["s22"]]),
    }


def quadratic_inputs(quantities: dict) -> dict:
    """Driving terms u01, u11, u12 of the quadratic chain."""
    force = quantities["z_force"]
    return {
        "u01": 0.5 * (quantities["z_anchor"] @ force),
        "u11": float(force @ quantities["z_velocity"]),
        "u12": float(0.5 * (force @ quantities["z_gravity"])),
    }


def exact_lifted_step(quadratic: dict, quantities: dict, dt: float) -> dict:
    """Advance every quadratic state one exact step under frozen inputs.

    The input enters through the same transition integral, so an input on s01
    also moves s00 inside the step; that is why the full input map is used
    instead of only the diagonal entry.
    """
    transition = nilpotent_transition(dt)
    input_map = nilpotent_input_map(dt)
    inputs = quadratic_inputs(quantities)

    stacked = stacked_quadratic(quadratic)
    driving = np.zeros_like(stacked)
    driving[:, CHAIN_INDEX["s01"]] = np.atleast_1d(inputs["u01"])
    driving[:, CHAIN_INDEX["s11"]] = inputs["u11"]
    driving[:, CHAIN_INDEX["s12"]] = inputs["u12"]
    advanced = stacked @ transition.T + driving @ input_map.T
    return unstack_quadratic(advanced)


def anchor_driving_jacobian(quantities: dict, rotation: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    """Return d u / d xi for every anchor with shape (anchors, 6, 15).

    Under ``X_true = Exp(xi) X_hat`` the first-order perturbations are
    ``dz_i = R^T a_i^x theta - R^T p``, ``dz_v = R^T v``,
    ``dz_g = -R^T g^x theta`` and ``da_c = -b_a``.  With ``u01 = 0.5 a_c^T z_i``,
    ``u11 = a_c^T z_v`` and ``u12 = 0.5 a_c^T z_g`` this gives the three
    non-trivial rows below; the remaining rows are zero because no other
    quadratic state is driven directly.
    """
    force = np.asarray(quantities["z_force"], dtype=np.float64).reshape(3)
    force_rotated = force @ rotation.T  # a_c^T R^T, shape (3,)
    anchor_skew = np.array([skew(anchor) for anchor in anchors])  # (anchors, 3, 3)

    per_anchor = np.zeros((anchors.shape[0], QUADRATIC_COUNT, ERROR_DIM))
    # d s01 / d theta = 0.5 a_c^T R^T a_i^x, d s01 / d p = -0.5 a_c^T R^T,
    # d s01 / d b_a = -0.5 z_i^T.
    per_anchor[:, CHAIN_INDEX["s01"], 0:3] = 0.5 * np.einsum("j,ija->ia", force_rotated, anchor_skew)
    per_anchor[:, CHAIN_INDEX["s01"], 6:9] = -0.5 * force_rotated[None, :]
    per_anchor[:, CHAIN_INDEX["s01"], 12:15] = -0.5 * quantities["z_anchor"]
    # d s11 / d v = a_c^T R^T, d s11 / d b_a = -z_v^T.
    per_anchor[:, CHAIN_INDEX["s11"], 3:6] = force_rotated[None, :]
    per_anchor[:, CHAIN_INDEX["s11"], 12:15] = -quantities["z_velocity"][None, :]
    # d s12 / d theta = -0.5 a_c^T R^T g^x, d s12 / d b_a = -0.5 z_g^T.
    # The sign is negative because this module defines z_g = -R^T g.
    per_anchor[:, CHAIN_INDEX["s12"], 0:3] = -0.5 * (force_rotated @ skew(GRAVITY_WORLD))[None, :]
    per_anchor[:, CHAIN_INDEX["s12"], 12:15] = -0.5 * quantities["z_gravity"][None, :]
    return per_anchor


def half_squared_range(quadratic: dict, anchor_index: int) -> float:
    """Predicted half-squared range from the lifted output row (zero lever arm)."""
    return float(quadratic["s00"][anchor_index])
