"""Check single-use information and full-state model derivatives before replay."""

import json
from pathlib import Path

import numpy as np
from scipy.optimize._numdiff import approx_derivative
from scipy.spatial.transform import Rotation

from smoother import conditional_replace, difference, inject, nominal_step, discrete_error_matrices


def main():
    """Verify scalar Gaussian replacement and the exact shared IMU tangent model."""
    state = np.eye(5)
    state[:3, 4] = [1., 0., 0.]
    covariance = np.eye(15)
    prior = state.copy(), covariance.copy(), np.zeros(6)
    result, posterior, bias, detail = conditional_replace(np.array([0.]), np.zeros((1, 6)),
        prior, [(0, 0, 2., 1.)], np.zeros((1, 3)), False, 0.)
    assert abs(result[0, 4]-1.5) < 1e-10
    # Invariant covariance is converted back to compare its physical p component.
    from invariant15 import inverse_coordinate_map
    inverse = inverse_coordinate_map(result)
    physical = inverse@posterior@inverse.T
    assert abs(physical[6, 6]-.5) < 1e-10
    assert all(row["packet_factors"] == 1 for row in detail["iterations"])
    sample, dt = np.array([.2, -.1, 1.1, 3., 2., 10.]), .01
    next_state = nominal_step(state, prior[2], sample, dt)
    transition, noise = discrete_error_matrices(state, prior[2], sample, dt)
    def fun(delta):
        """Evaluate exact physical dynamics errors for finite differences."""
        perturbed, shifted_bias = inject(state, prior[2], delta)
        target = nominal_step(perturbed, shifted_bias, sample, dt)
        return difference(next_state, prior[2], target, shifted_bias)
    error = float(np.max(np.abs(approx_derivative(fun, np.zeros(15))-transition)))
    assert error < 1e-7
    angle = np.array([.1, -.2, 1.7])
    reference_rotation = Rotation.from_rotvec(angle).as_matrix()
    def prior_residual(delta):
        """Differentiate the unmodified Gaussian prior around a rotated seed."""
        return Rotation.from_matrix(reference_rotation@Rotation.from_rotvec(delta).as_matrix()).as_rotvec()
    from eskf15 import rotation_terms
    prior_jacobian_error = float(np.max(np.abs(approx_derivative(prior_residual, np.zeros(3))-
        np.linalg.inv(rotation_terms(angle)[1]))))
    assert prior_jacobian_error < 1e-7
    payload = dict(single_packet_after_four_passes_position=1.5,
        single_packet_posterior_variance=.5, duplicate_information_avoided=True,
        shared_native_F_finite_difference_max_error=error,
        large_angle_original_prior_residual_jacobian_max_error=prior_jacobian_error,
        prior_not_mutated=bool(np.array_equal(prior[0], state)),
        covariance_finite=bool(np.isfinite(physical).all()))
    destination = Path(__file__).resolve().parent/"verification.json"
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
