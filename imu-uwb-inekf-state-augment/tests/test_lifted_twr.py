from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.linalg import expm


PROJECT = Path(__file__).resolve().parents[1]
ESTIMATION = PROJECT / "scripts" / "estimation"
sys.path.insert(0, str(ESTIMATION))

from inekf import (  # noqa: E402
    GRAVITY_MAGNITUDE,
    InEKF,
    SO3_left_jacobian,
    SO3_wedge,
)
from lifted_twr import (  # noqa: E402
    DirectRangeEKF,
    LiftedRangeKF,
    LiftedTWRInEKF,
    SquaredRangeEKF,
    TWRInEKF,
    paper_consistent_initial_covariance,
)
from twr_experiment import (  # noqa: E402
    DEFAULT_DATASET,
    RealLog,
    build_loso_range_calibrations,
    load_real_log,
    run_synthetic_single_range,
    trajectory_metrics,
)


class PaperConsistencyTests(unittest.TestCase):
    def test_so3_left_jacobian_small_angle_has_positive_sign(self) -> None:
        phi = np.array([1e-10, -2e-10, 3e-10])
        expected = np.eye(3) + 0.5 * SO3_wedge(phi)
        np.testing.assert_allclose(
            SO3_left_jacobian(phi), expected, rtol=0.0, atol=1e-18
        )

    def test_bias_order_is_gyro_then_accelerometer(self) -> None:
        state = np.eye(5)
        state[:3, 4] = [0.0, 0.0, 1.0]
        covariance = np.eye(15) * 1e-8
        gyro_bias = np.array([0.01, -0.02, 0.03])
        accel_bias = np.array([0.10, -0.20, 0.30])
        bias = np.concatenate((gyro_bias, accel_bias))
        sample = np.concatenate(
            (
                (np.array([0.0, 0.0, GRAVITY_MAGNITUDE]) + accel_bias)
                / GRAVITY_MAGNITUDE,
                np.rad2deg(gyro_bias),
            )
        )
        filter_instance = InEKF(state, covariance, bias)
        predicted, _, _ = filter_instance.predict(sample, 0.01)
        np.testing.assert_allclose(predicted[:3, :3], np.eye(3), atol=1e-12)
        np.testing.assert_allclose(predicted[:3, 3], np.zeros(3), atol=1e-12)

    def test_initial_covariance_uses_paper_bias_blocks(self) -> None:
        covariance = paper_consistent_initial_covariance(
            gyro_bias_std=0.02, accel_bias_std=0.30
        )
        np.testing.assert_allclose(np.diag(covariance)[9:12], 0.02**2)
        np.testing.assert_allclose(np.diag(covariance)[12:15], 0.30**2)


class LiftedModelTests(unittest.TestCase):
    def test_closed_form_transition_matches_matrix_exponential(self) -> None:
        acceleration = np.array([1.2, -0.7, 0.4])
        dt = 0.037
        actual = LiftedRangeKF.transition_matrix(acceleration, dt)
        expected = expm(LiftedRangeKF.continuous_matrix(acceleration) * dt)
        np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=2e-15)

    def test_exact_transition_preserves_lift_manifold(self) -> None:
        filter_instance = LiftedRangeKF(
            [2.0, -1.0, 3.0], [0.4, 0.5, -0.2]
        )
        filter_instance.predict([1.2, -0.7, 0.4], 0.037)
        np.testing.assert_allclose(
            filter_instance.constraint_residual(), np.zeros(3), atol=3e-14
        )
        self.assertEqual(filter_instance.chi[-1], 1.0)

    def test_half_squared_range_is_constant_linear_output(self) -> None:
        position = np.array([1.2, -0.7, 2.3])
        velocity = np.array([0.4, 0.1, -0.2])
        anchor = np.array([-2.0, 4.0, 1.5])
        filter_instance = LiftedRangeKF(position, velocity)
        row_before = filter_instance.observation_row(anchor)
        expected = 0.5 * np.linalg.norm(position - anchor) ** 2
        self.assertAlmostEqual(float((row_before @ filter_instance.chi).item()), expected)

        filter_instance.predict([0.3, -0.4, 0.5], 0.1)
        row_after = filter_instance.observation_row(anchor)
        np.testing.assert_array_equal(row_before, row_after)

    def test_squared_noise_debiasing_is_unbiased_in_monte_carlo(self) -> None:
        generator = np.random.default_rng(42)
        true_range = 7.0
        sigma = 0.2
        noisy = true_range + generator.normal(0.0, sigma, size=200_000)
        processed = 0.5 * (noisy**2 - sigma**2)
        self.assertAlmostEqual(
            float(np.mean(processed)), 0.5 * true_range**2, delta=0.01
        )

    def test_multianchor_noiseless_updates_recover_position(self) -> None:
        anchors = np.array(
            [
                [-4.0, -4.0, 0.0],
                [-4.0, 4.0, 0.0],
                [4.0, 4.0, 0.0],
                [4.0, -4.0, 0.0],
                [-4.0, -4.0, 3.0],
                [-4.0, 4.0, 3.0],
                [4.0, 4.0, 3.0],
                [4.0, -4.0, 3.0],
            ]
        )
        truth = np.array([0.6, -1.1, 1.4])
        ranges = np.linalg.norm(anchors - truth, axis=1)
        filter_instance = LiftedRangeKF(
            [8.0, -7.0, 5.0],
            position_std=20.0,
            q_std=200.0,
            range_sigma=1e-3,
            range_bias=0.0,
            gate_warmup_updates=100,
        )
        for _ in range(3):
            filter_instance.correct_batch(anchors, ranges)
        self.assertLess(np.linalg.norm(filter_instance.position - truth), 0.01)
        self.assertGreaterEqual(
            np.min(np.linalg.eigvalsh(filter_instance.P)), -1e-10
        )

    def test_optional_projection_enforces_physical_lift_manifold(self) -> None:
        filter_instance = LiftedRangeKF(
            [2.0, -1.0, 3.0],
            [0.4, 0.5, -0.2],
            project_constraints=True,
            range_sigma=0.1,
            range_bias=0.0,
        )
        filter_instance.correct([4.0, -3.0, 2.0], 3.0)
        np.testing.assert_allclose(
            filter_instance.constraint_residual(), np.zeros(3), atol=1e-12
        )
        self.assertGreaterEqual(
            np.min(np.linalg.eigvalsh(filter_instance.P)), -1e-9
        )

    def test_six_state_squared_range_update_moves_toward_truth(self) -> None:
        filter_instance = SquaredRangeEKF(
            [0.0, 0.0, 0.0],
            position_std=2.0,
            range_sigma=0.05,
            range_bias=0.0,
        )
        accepted = filter_instance.correct([10.0, 0.0, 0.0], 9.0)
        self.assertTrue(accepted)
        self.assertGreater(filter_instance.position[0], 0.0)

    def test_hover_cascade_has_zero_world_acceleration(self) -> None:
        state = np.eye(5)
        state[:3, 4] = [0.0, 0.0, 1.0]
        cascade = LiftedTWRInEKF(
            state,
            paper_consistent_initial_covariance(),
            np.zeros(6),
        )
        world_acceleration = cascade.predict([0, 0, 1, 0, 0, 0], 0.01)
        np.testing.assert_allclose(world_acceleration, np.zeros(3), atol=1e-12)
        np.testing.assert_allclose(cascade.X[:3, 4], [0.0, 0.0, 1.0], atol=1e-12)

    def test_per_anchor_bias_and_sigma_require_and_use_anchor_index(self) -> None:
        filter_instance = DirectRangeEKF(
            [0.0, 0.0, 0.0],
            position_std=0.1,
            range_sigma=[0.1, 0.2],
            range_bias=[-1.0, 0.0],
        )
        accepted = filter_instance.correct(
            [10.0, 0.0, 0.0], 9.0, anchor_index=0
        )
        self.assertTrue(accepted)
        np.testing.assert_allclose(filter_instance.position, np.zeros(3), atol=1e-12)

        missing_index = DirectRangeEKF(
            [0.0, 0.0, 0.0],
            range_sigma=[0.1, 0.2],
            range_bias=[-1.0, 0.0],
        )
        self.assertFalse(missing_index.correct([10.0, 0.0, 0.0], 9.0))
        self.assertEqual(missing_index.stats.invalid, 1)

    def test_robust_policy_downweights_moderate_and_rejects_large_nis(self) -> None:
        moderate = DirectRangeEKF(
            [0.0, 0.0, 0.0],
            position_std=1e-3,
            velocity_std=1e-3,
            range_sigma=1.0,
            range_bias=0.0,
            gate_threshold=3.5**2,
            gate_warmup_updates=0,
            huber_threshold=2.0,
        )
        self.assertTrue(moderate.correct([10.0, 0.0, 0.0], 12.5))
        self.assertEqual(moderate.stats.downweighted, 1)
        self.assertLess(moderate.stats.mean_accepted_weight, 1.0)

        large = DirectRangeEKF(
            [0.0, 0.0, 0.0],
            position_std=1e-3,
            velocity_std=1e-3,
            range_sigma=1.0,
            range_bias=0.0,
            gate_threshold=3.5**2,
            gate_warmup_updates=0,
            huber_threshold=2.0,
        )
        self.assertFalse(large.correct([10.0, 0.0, 0.0], 14.0))
        self.assertEqual(large.stats.rejected, 1)


class CalibrationAndMetricTests(unittest.TestCase):
    @staticmethod
    def _constant_log(name: str, residual: float) -> RealLog:
        time_values = np.arange(20, dtype=np.float64)
        truth = np.zeros((20, 3), dtype=np.float64)
        ranges = np.full((20, 2), 10.0 + residual)
        return RealLog(
            name=name,
            time=time_values,
            imu=np.zeros((20, 6)),
            ranges=ranges,
            changed=np.ones((20, 2), dtype=bool),
            ground_truth_position=truth,
            ground_truth_velocity=np.zeros((20, 3)),
        )

    def test_loso_calibration_excludes_held_out_truth(self) -> None:
        anchors = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]])
        logs = {
            "held.csv": self._constant_log("held.csv", 100.0),
            "train_a.csv": self._constant_log("train_a.csv", 1.0),
            "train_b.csv": self._constant_log("train_b.csv", 3.0),
        }
        calibrations, table = build_loso_range_calibrations(
            logs, anchors, sigma_floor_m=0.08, sigma_ceiling_m=5.0
        )
        np.testing.assert_allclose(
            calibrations["held.csv"]["range_bias_m"], [2.0, 2.0]
        )
        held_rows = table.loc[table["held_out_sequence"] == "held.csv"]
        self.assertTrue(np.all(held_rows["training_sequence_count"] == 2))

    def test_xy_metrics_remove_only_vertical_component(self) -> None:
        time_values = np.array([0.0, 1.0])
        truth = np.zeros((2, 3))
        position = np.array([[3.0, 4.0, 12.0], [3.0, 4.0, 12.0]])
        velocity = np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 2.0]])
        metrics = trajectory_metrics(
            time_values, position, velocity, truth, np.zeros((2, 3))
        )
        self.assertAlmostEqual(metrics["position_rmse_m"], 13.0)
        self.assertAlmostEqual(metrics["position_xy_rmse_m"], 5.0)
        self.assertAlmostEqual(metrics["velocity_rmse_mps"], 2.0)
        self.assertAlmostEqual(metrics["velocity_xy_rmse_mps"], 0.0)


class BaselineAndConvergenceTests(unittest.TestCase):
    def test_twr_inekf_range_residual_moves_position_toward_truth(self) -> None:
        state = np.eye(5)
        covariance = np.eye(15)
        filter_instance = TWRInEKF(
            state,
            covariance,
            np.zeros(6),
            range_sigma=0.05,
            range_bias=0.0,
        )
        # Anchor at x=10; measured range 9 corresponds locally to p_x=1.
        accepted = filter_instance.correct_twr([10.0, 0.0, 0.0], 9.0)
        self.assertTrue(accepted)
        self.assertGreater(filter_instance.X[0, 4], 0.0)

    def test_large_error_single_range_case_favors_lifted_observer(self) -> None:
        result = run_synthetic_single_range(
            seed=7,
            initial_position=(35.0, -25.0, 20.0),
            initial_velocity=(0.0, 0.0, 0.0),
        )
        direct = result["metrics"]["methods"]["direct_twr"]
        lifted = result["metrics"]["methods"]["lifted_twr"]
        ordinary = result["metrics"]["methods"]["twr_inekf"]
        self.assertGreater(direct["position_tail5_rmse_m"], 50.0)
        self.assertLess(lifted["position_tail5_rmse_m"], 1.0)
        self.assertTrue(lifted["success_tail5_below_1m"])
        self.assertTrue(ordinary["finite"])
        self.assertFalse(ordinary["success_tail5_below_1m"])

    @unittest.skipUnless(DEFAULT_DATASET.is_dir(), "TU Delft dataset is not installed")
    def test_real_twr_loader_finds_asynchronous_events(self) -> None:
        log = load_real_log(
            DEFAULT_DATASET,
            "2021-02-12_09_41_16_twr_triangle.csv",
            duration=2.0,
        )
        self.assertGreater(log.time.size, 150)
        self.assertGreater(np.count_nonzero(log.changed), log.time.size)
        self.assertTrue(np.all(np.isfinite(log.ranges)))
        self.assertTrue(np.all(np.diff(log.time) > 0))


if __name__ == "__main__":
    unittest.main()
