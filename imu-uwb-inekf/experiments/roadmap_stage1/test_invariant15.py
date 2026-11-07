"""Check the shared model and invariant range/reset coordinate derivatives."""

import numpy as np
from scipy.linalg import logm

from eskf15 import ESKF15, rotation_terms
from invariant15 import Invariant15, coordinate_map, inverse_coordinate_map, group_exp


def run_checks():
    """Check covariance equivalence, group derivatives, and origin independence."""
    rng = np.random.default_rng(47)
    state = np.eye(5)
    state[:3, :3] = rotation_terms(np.array([0.2, -0.4, 1.1]))[0]
    state[:3, 3:] = rng.normal(size=(3, 2))
    factor = rng.normal(size=(15, 15)) * 0.03
    physical = factor @ factor.T + np.eye(15) * 0.01
    mapping = coordinate_map(state)
    np.testing.assert_allclose(inverse_coordinate_map(state) @ mapping, np.eye(15), atol=1e-14)
    conventional = ESKF15(state, physical, np.zeros(6))
    invariant = Invariant15(state, mapping @ physical @ mapping.T, np.zeros(6))
    conventional.predict([0.1, -0.2, 1.0, 5.0, 6.0, -7.0], 0.02)
    invariant.predict([0.1, -0.2, 1.0, 5.0, 6.0, -7.0], 0.02)
    np.testing.assert_allclose(invariant.X, conventional.X, atol=1e-14)
    np.testing.assert_allclose(invariant.physical_covariance(), conventional.P, atol=1e-13)
    anchor = np.array([4.0, -1.0, 2.0])
    predicted, row = invariant.range_model(anchor)
    epsilon = 1e-6
    derivative = np.zeros(9)
    for column in range(9):
        direction = np.eye(9)[column] * epsilon
        plus = (group_exp(direction) @ invariant.X)[:3, 4]
        minus = (group_exp(-direction) @ invariant.X)[:3, 4]
        derivative[column] = (np.linalg.norm(plus - anchor) - np.linalg.norm(minus - anchor)) / (2 * epsilon)
    np.testing.assert_allclose(row[0, :9], derivative, atol=2e-9)
    delta = rng.normal(size=15) * 0.2
    reset = Invariant15.reset_jacobian(delta)
    numeric = np.empty((9, 9))
    for column in range(9):
        direction = np.eye(9)[column] * epsilon
        values = []
        for sign in (1, -1):
            algebra = logm(group_exp(delta[:9] + sign * direction) @ group_exp(-delta[:9])).real
            values.append(np.r_[algebra[2, 1], algebra[0, 2], algebra[1, 0], algebra[:3, 3], algebra[:3, 4]])
        numeric[:, column] = (values[0] - values[1]) / (2 * epsilon)
    np.testing.assert_allclose(reset[:9, :9], numeric, atol=2e-9)
    for angle in (np.pi, 15.0, 70.0):
        # Early recovery may generate corrections beyond the principal rotation.
        delta[:3] = np.array([angle, 0.0, 0.0])
        reset = Invariant15.reset_jacobian(delta)
        for column in range(9):
            direction = np.eye(9)[column] * epsilon
            values = []
            for sign in (1, -1):
                algebra = logm(group_exp(delta[:9] + sign * direction) @ group_exp(-delta[:9])).real
                values.append(np.r_[algebra[2, 1], algebra[0, 2], algebra[1, 0], algebra[:3, 3], algebra[:3, 4]])
            numeric[:, column] = (values[0] - values[1]) / (2 * epsilon)
        np.testing.assert_allclose(reset[:9, :9], numeric, atol=1e-8)
    shift = np.array([7.0, -9.0, 3.0])
    translated = invariant.X.copy()
    translated[:3, 4] += shift
    moved_mapping = coordinate_map(translated)
    moved = Invariant15(translated, moved_mapping @ invariant.physical_covariance() @ moved_mapping.T, invariant.bias)
    invariant.correct_twr(anchor, predicted + 0.05)
    moved.correct_twr(anchor + shift, predicted + 0.05)
    np.testing.assert_allclose(moved.X[:3, 4] - shift, invariant.X[:3, 4], atol=1e-13)
    np.testing.assert_allclose(moved.physical_covariance(), invariant.physical_covariance(), atol=1e-12)
    assert np.min(np.linalg.eigvalsh(invariant.P)) > -1e-12
    print("Invariant15: shared prediction, covariance maps, range/reset derivatives and translation checks passed")


if __name__ == "__main__":
    run_checks()
