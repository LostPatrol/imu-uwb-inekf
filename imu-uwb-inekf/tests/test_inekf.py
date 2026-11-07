from pathlib import Path
import sys
import traceback

import numpy as np
from scipy import linalg
from scipy.spatial.transform import Rotation


ESTIMATION_DIR = Path(__file__).resolve().parents[1] / "scripts" / "estimation"
sys.path.insert(0, str(ESTIMATION_DIR))

from inekf import (  # noqa: E402
    BA,
    BG,
    FilterConfig,
    InEKF,
    SE3_2_adjoint,
    SE3_2_exp,
    SE3_2_inv,
    SE3_2_log,
    SE3_2_right_jacobian,
    SO3_exp,
    SO3_gamma2,
    SO3_left_jacobian,
    SO3_log,
)
from pipeline import Dataset, run_dataset  # noqa: E402
from pipeline import _initial_filter, _stationary_flags  # noqa: E402
from compiled_replay import (  # noqa: E402
    execute_compiled_replay,
    prepare_compiled_replay,
)


def make_filter():
    xi = np.array([0.2, -0.1, 0.3, 0.4, -0.2, 0.1, 1.1, -0.7, 0.8])
    return InEKF(SE3_2_exp(xi), np.eye(15) * 0.05, np.zeros(6), FilterConfig.verified())


def test_so3_round_trip_small_and_near_pi():
    rotations = [
        np.array([1e-10, -2e-10, 3e-10]),
        np.array([0.3, -0.4, 0.2]),
        np.array([np.pi - 1e-8, 0.0, 0.0]),
    ]
    for phi in rotations:
        reconstructed = SO3_exp(SO3_log(SO3_exp(phi)))
        np.testing.assert_allclose(reconstructed, SO3_exp(phi), atol=2e-8)


def test_rotation_products_do_not_require_per_step_svd_projection():
    increment = SO3_exp(np.array([0.0007, -0.0004, 0.0011]))
    rotation = np.eye(3)
    for _ in range(20000):
        rotation = rotation @ increment
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=3e-12)
    assert abs(np.linalg.det(rotation) - 1.0) < 3e-12


def test_left_jacobian_matches_exponential_perturbation():
    phi = np.array([0.4, -0.2, 0.3])
    delta = np.array([2e-7, -3e-7, 1e-7])
    lhs = SO3_exp(phi + delta)
    rhs = SO3_exp(SO3_left_jacobian(phi) @ delta) @ SO3_exp(phi)
    np.testing.assert_allclose(lhs, rhs, atol=2e-13)


def test_se23_exp_log_and_adjoint_identity():
    xi = np.array([0.3, -0.2, 0.1, 0.4, 0.2, -0.1, 1.0, -0.5, 0.7])
    T = SE3_2_exp(xi)
    np.testing.assert_allclose(SE3_2_exp(SE3_2_log(T)), T, atol=1e-12)

    eta = np.array([-0.2, 0.1, 0.15, 0.3, -0.4, 0.2, -0.5, 0.6, 0.1])
    lhs = T @ SE3_2_exp(eta) @ SE3_2_inv(T)
    rhs = SE3_2_exp(SE3_2_adjoint(T) @ eta)
    np.testing.assert_allclose(lhs, rhs, atol=1e-12)


def test_complete_uwb_jacobian_matches_central_difference():
    estimator = make_filter()
    anchors = np.array(
        [[-2.0, -1.0, 0.2], [2.5, 1.2, 2.1], [0.0, 2.0, 1.4]]
    )
    uwb = np.array([0.0, 1.0, 0.0])
    analytic = estimator.measurement_jacobian(uwb, anchors)[0, :9]
    numeric = np.zeros(9)
    epsilon = 1e-6
    for index in range(9):
        delta = np.zeros(9)
        delta[index] = epsilon
        plus = SE3_2_exp(-delta) @ estimator.X_hat
        minus = SE3_2_exp(delta) @ estimator.X_hat
        numeric[index] = (
            estimator.measurement(uwb, anchors, plus)
            - estimator.measurement(uwb, anchors, minus)
        ) / (2.0 * epsilon)
    np.testing.assert_allclose(analytic, numeric, atol=2e-8, rtol=2e-7)


def test_se23_reset_jacobian_matches_injected_error_coordinates():
    injected = np.array([0.4, -0.2, 0.1, 0.3, 0.1, -0.2, -0.5, 0.4, 0.2])
    analytic = SE3_2_right_jacobian(injected)
    numeric = np.zeros((9, 9))
    epsilon = 1e-7
    for index in range(9):
        perturbation = np.zeros(9)
        perturbation[index] = epsilon
        plus = SE3_2_log(SE3_2_exp(-injected) @ SE3_2_exp(injected + perturbation))
        minus = SE3_2_log(SE3_2_exp(-injected) @ SE3_2_exp(injected - perturbation))
        numeric[:, index] = (plus - minus) / (2.0 * epsilon)
    np.testing.assert_allclose(analytic, numeric, atol=2e-8, rtol=2e-7)


def test_verified_bias_order_cancels_known_stationary_biases():
    gyro_bias = np.array([0.01, -0.02, 0.03])
    accel_bias = np.array([0.1, -0.2, 0.05])
    # The nominal six-vector follows the same [b_g, b_a] order as BG/BA.
    bias = np.concatenate((gyro_bias, accel_bias))
    estimator = InEKF(np.eye(5), np.eye(15) * 1e-4, bias, FilterConfig.verified())
    imu = np.concatenate(
        ((np.array([0.0, 0.0, 9.81]) + accel_bias) / 9.81, np.degrees(gyro_bias))
    )
    estimator.predict(imu, 0.01, u_next=imu)
    np.testing.assert_allclose(estimator.X[:3, :3], np.eye(3), atol=1e-12)
    np.testing.assert_allclose(estimator.X[:3, 3:5], 0.0, atol=1e-12)


def test_van_loan_covariance_is_symmetric_positive_semidefinite():
    estimator = make_filter()
    imu = np.array([0.1, -0.05, 1.02, 2.0, -1.0, 4.0])
    estimator.predict(imu, 0.01, u_next=imu + 0.01)
    np.testing.assert_allclose(estimator.P, estimator.P.T, atol=1e-13)
    assert np.linalg.eigvalsh(estimator.P).min() > -1e-12


def test_structured_transition_is_exact_for_current_error_dynamics():
    estimator = make_filter()
    A = estimator._error_dynamics(estimator.X)
    np.testing.assert_allclose(
        np.linalg.matrix_power(A, 4),
        np.zeros((15, 15)),
        atol=2e-13,
    )
    dt = 0.013
    np.testing.assert_allclose(
        estimator.structured_transition(A, dt),
        linalg.expm(A * dt),
        atol=2e-15,
        rtol=2e-14,
    )


def test_structured_process_noise_matches_dense_formula():
    estimator = make_filter()
    dt = 0.007
    A = estimator._error_dynamics(estimator.X)
    Phi = estimator.structured_transition(A, dt)
    adjoint = estimator._augmented_adjoint(estimator.X)
    variances = estimator._sample_noise_variances() * dt**2
    noise_map = Phi @ adjoint
    structured = (noise_map * variances[np.newaxis, :]) @ noise_map.T
    dense = (
        noise_map
        @ (estimator._sample_noise_covariance() * dt**2)
        @ noise_map.T
    )
    np.testing.assert_allclose(structured, dense, atol=2e-17, rtol=2e-14)


def test_gamma2_matches_numerical_constant_rate_integration():
    phi = np.array([0.6, -0.35, 0.2])
    acceleration = np.array([0.3, -0.4, 9.7])
    samples = 20001
    normalized_time = np.linspace(0.0, 1.0, samples)
    integrand = np.array(
        [
            (1.0 - timestamp) * (SO3_exp(phi * timestamp) @ acceleration)
            for timestamp in normalized_time
        ]
    )
    numeric = np.trapz(integrand, normalized_time, axis=0)
    np.testing.assert_allclose(
        SO3_gamma2(phi) @ acceleration,
        numeric,
        atol=3e-9,
        rtol=2e-9,
    )


def test_fast_van_loan_matches_reference_block_exponential():
    rng = np.random.default_rng(4)
    A = rng.normal(scale=0.3, size=(15, 15))
    covariance = rng.normal(size=(15, 6))
    covariance = covariance @ covariance.T * 1e-3
    dt = 0.002
    Phi, Qd = InEKF.van_loan_discretization(A, covariance, dt)
    block = np.zeros((30, 30))
    block[:15, :15] = A
    block[:15, 15:] = covariance
    block[15:, 15:] = -A.T
    reference = linalg.expm(block * dt)
    reference_phi = reference[:15, :15]
    reference_qd = reference[:15, 15:] @ reference_phi.T
    np.testing.assert_allclose(Phi, reference_phi, atol=2e-15, rtol=2e-14)
    np.testing.assert_allclose(Qd, reference_qd, atol=2e-15, rtol=2e-14)


def test_direct_transformed_jacobian_matches_dense_adjoint_path():
    estimator = make_filter()
    anchors = np.array(
        [[-2.0, -1.0, 0.2], [2.5, 1.2, 2.1], [0.0, 2.0, 1.4]]
    )
    uwb = np.array([0.0, 1.0, 0.0])
    H_left = estimator._legacy_measurement_jacobian(uwb, anchors, estimator.X)
    augmented = np.eye(15)
    augmented[:9, :9] = SE3_2_adjoint(SE3_2_inv(estimator.X))
    dense = H_left @ augmented
    direct = estimator._direct_transformed_measurement_jacobian(
        uwb,
        anchors,
        estimator.X,
    )
    np.testing.assert_allclose(direct, dense, atol=2e-15, rtol=2e-14)


def test_rank_one_scalar_update_matches_joseph_update():
    base = FilterConfig.pre_optimization()
    rank_one_config = FilterConfig(
        **{
            **base.__dict__,
            "name": "rank_one_test",
            "rank_one_scalar_update": True,
        }
    )
    joseph = InEKF(make_filter().X, make_filter().P, np.zeros(6), base)
    rank_one = InEKF(make_filter().X, make_filter().P, np.zeros(6), rank_one_config)
    anchors = np.array(
        [[-2.0, -1.0, 0.2], [2.5, 1.2, 2.1], [0.0, 2.0, 1.4]]
    )
    measurement = np.array([0.0, 1.0, joseph.measurement([0.0, 1.0, 0.0], anchors) + 0.08])
    joseph.correct(measurement, anchors)
    rank_one.correct(measurement, anchors)
    np.testing.assert_allclose(rank_one.X, joseph.X, atol=2e-15, rtol=2e-14)
    np.testing.assert_allclose(rank_one.P, joseph.P, atol=2e-15, rtol=2e-14)


def test_stationary_update_reduces_gyro_bias_error_without_changing_velocity():
    config = FilterConfig.proposed()
    X = np.eye(5)
    X[:3, 3] = np.array([0.2, -0.1, 0.05])
    estimator = InEKF(X, np.eye(15) * 0.1, np.zeros(6), config)
    gyro_bias = np.array([0.01, -0.02, 0.03])
    before_velocity = estimator.X[:3, 3].copy()
    before_bias_error = np.linalg.norm(estimator.bias[:3] - gyro_bias)
    estimator.correct_stationary(np.degrees(gyro_bias))
    np.testing.assert_allclose(estimator.X[:3, 3], before_velocity, atol=1e-14)
    assert np.linalg.norm(estimator.bias[:3] - gyro_bias) < before_bias_error
    assert estimator.stationary_update_count == 1


def test_coordinate_consistent_gate_is_invariant_to_world_translation():
    physical_covariance = np.diag(
        [0.02**2, 0.02**2, 1.0**2]
        + [0.05**2] * 3
        + [0.1**2] * 3
        + [0.01**2] * 6
    )
    anchors = np.array(
        [
            [-3.0, -2.0, 2.5],
            [3.2, -1.8, 2.4],
            [2.8, 2.3, 2.6],
            [-3.1, 2.1, 2.5],
        ]
    )
    translation = np.array([4.7, -3.2, 1.1])
    base_position = np.array([1.2, -0.4, 0.8])
    rotation = SO3_exp(np.array([0.1, -0.05, 1.7]))
    measurement = np.array([0.0, 1.0, 0.0])

    outcomes = []
    for offset in (np.zeros(3), translation):
        X0 = np.eye(5)
        X0[:3, :3] = rotation
        X0[:3, 4] = base_position + offset
        estimator = InEKF(X0, physical_covariance, np.zeros(6), FilterConfig.proposed())
        estimator.initialize_right_invariant_covariance(physical_covariance)
        shifted_anchors = anchors + offset
        measurement[2] = estimator.measurement(measurement, shifted_anchors) + 0.12
        _, _, _, rejected = estimator.correct(measurement, shifted_anchors)
        outcomes.append((estimator.last_nis, rejected))

    assert np.isclose(outcomes[0][0], outcomes[1][0], rtol=1e-10, atol=1e-12)
    assert outcomes[0][1] == outcomes[1][1]


def test_compiled_replay_matches_proposed_equations():
    t_imu = np.arange(0.0, 0.31, 0.01)
    t_uwb = np.arange(0.005, 0.305, 0.01)
    anchors = np.array(
        [
            [-2.0, -2.0, 2.0],
            [2.0, -2.0, 2.0],
            [2.0, 2.0, 2.0],
            [-2.0, 2.0, 2.0],
        ]
    )
    imu = np.zeros((len(t_imu), 6))
    imu[:, 2] = 1.0
    imu[:, 3] = 0.02
    tag_offset = np.array([-0.01245, 0.00127, 0.0908])
    distance_a = np.linalg.norm(tag_offset - anchors[0])
    distance_b = np.linalg.norm(tag_offset - anchors[1])
    measurement = distance_b - distance_a
    uwb = np.column_stack(
        (
            np.zeros(len(t_uwb)),
            np.ones(len(t_uwb)),
            measurement + 0.01 * np.sin(20.0 * t_uwb),
        )
    )
    t_gt = t_imu.copy()
    dataset = Dataset(
        name="synthetic_const1",
        anchors=anchors,
        t_imu=t_imu,
        imu=imu,
        t_uwb=t_uwb,
        uwb=uwb,
        t_gt=t_gt,
        p_gt=np.zeros((len(t_gt), 3)),
        q_gt_xyzw=np.tile(
            Rotation.identity().as_quat(),
            (len(t_gt), 1),
        ),
    )

    proposed = run_dataset(dataset, variant="proposed")
    compiled = run_dataset(dataset, variant="task03_compiled")
    np.testing.assert_array_equal(compiled.times, proposed.times)
    np.testing.assert_allclose(compiled.states, proposed.states, atol=2e-10, rtol=2e-10)
    np.testing.assert_allclose(
        compiled.rotations,
        proposed.rotations,
        atol=2e-10,
        rtol=2e-10,
    )
    np.testing.assert_allclose(
        compiled.covariances,
        proposed.covariances,
        atol=2e-10,
        rtol=2e-10,
    )
    np.testing.assert_allclose(compiled.biases, proposed.biases, atol=2e-12, rtol=2e-10)
    np.testing.assert_allclose(compiled.nis, proposed.nis, atol=2e-10, rtol=2e-10)
    np.testing.assert_array_equal(compiled.rejected, proposed.rejected)

    timing_estimator = _initial_filter(
        dataset,
        0.0,
        FilterConfig.task03_compiled(),
    )
    timing_output = execute_compiled_replay(
        prepare_compiled_replay(
            dataset,
            timing_estimator,
            _stationary_flags(dataset),
        ),
        record_history=False,
    )
    assert timing_output[0].shape == (0, 9)
    assert timing_output[2].shape == (0, 15, 15)
    assert timing_output[5].size == 0
    assert timing_output[-2] == len(dataset.uwb)
    assert timing_output[-1] == int(np.sum(compiled.rejected))


if __name__ == "__main__":
    failures = 0
    tests = sorted(
        (name, value)
        for name, value in globals().items()
        if name.startswith("test_") and callable(value)
    )
    for name, test in tests:
        try:
            test()
            print(f"PASS {name}")
        except Exception:
            failures += 1
            print(f"FAIL {name}")
            traceback.print_exc()
    print(f"{len(tests) - failures}/{len(tests)} tests passed")
    raise SystemExit(1 if failures else 0)
