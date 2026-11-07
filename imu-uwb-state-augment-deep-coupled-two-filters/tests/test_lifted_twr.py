"""Verify TWR models, robust calibration, and scalar protocol compatibility."""

from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.linalg import expm


PROJECT = Path(__file__).resolve().parents[1]  # Deep-coupled project root.
ESTIMATION = PROJECT / "scripts" / "estimation"  # Local filter modules.
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
    TU_DELFT_COMMON_SENSOR_DELAY_S,
    load_anchors,
    load_real_log,
    run_synthetic_single_range,
)
from time_alignment_diagnostic import scan_sequence  # noqa: E402


def protocol_filter(factory, **parameters):
    """Create each range model with the same small physical uncertainty."""

    position = np.array([0.0, 0.0, 1.0])
    if factory is TWRInEKF:
        state = np.eye(5)
        state[:3, 4] = position
        return factory(state, np.eye(15) * 1e-6, np.zeros(6), **parameters)
    if factory is LiftedRangeKF:
        parameters = {"q_std": 1e-3, "s_std": 1e-3, "h_std": 1e-3, **parameters}
    return factory(
        position, position_std=1e-3, velocity_std=1e-3, **parameters
    )


class RangeProtocolTests(unittest.TestCase):
    def test_per_anchor_calibration_matches_selected_scalar_model(self) -> None:
        """Check each model selects both sigma and bias for the given anchor."""

        anchor = np.array([4.0, -3.0, 2.0])
        measured_range = float(np.linalg.norm(anchor - [0.0, 0.0, 1.0])) + 0.75
        for factory in (LiftedRangeKF, DirectRangeEKF, SquaredRangeEKF, TWRInEKF):
            with self.subTest(model=factory.__name__):
                indexed = protocol_filter(
                    factory, range_sigma=[0.1, 0.35], range_bias=[-0.2, 0.6]
                )
                scalar = protocol_filter(factory, range_sigma=0.35, range_bias=0.6)
                correct_indexed = (
                    indexed.correct_twr if factory is TWRInEKF else indexed.correct
                )
                correct_scalar = (
                    scalar.correct_twr if factory is TWRInEKF else scalar.correct
                )
                self.assertFalse(correct_indexed(anchor, measured_range))
                self.assertEqual(indexed.stats.invalid, 1)
                self.assertTrue(correct_indexed(anchor, measured_range, anchor_index=1))
                self.assertTrue(correct_scalar(anchor, measured_range))
                for attribute in ("chi", "x", "X", "bias", "P"):
                    if hasattr(indexed, attribute):
                        np.testing.assert_array_equal(
                            getattr(indexed, attribute), getattr(scalar, attribute)
                        )
                self.assertEqual(indexed.stats.per_anchor[1]["accepted"], 1)

    def test_huber_and_hard_gate_activate_after_common_warmup(self) -> None:
        """Verify warmup, information reduction, and large-residual rejection."""

        anchor = [10.0, 0.0, 1.0]
        parameters = {
            "range_sigma": [0.2, 1.0],
            "range_bias": [-0.3, 0.0],
            "gate_threshold": 3.5**2,
            "gate_warmup_updates": 1,
        }
        for factory in (LiftedRangeKF, DirectRangeEKF, SquaredRangeEKF, TWRInEKF):
            with self.subTest(model=factory.__name__):
                robust = protocol_filter(factory, huber_threshold=2.0, **parameters)
                plain = protocol_filter(factory, huber_threshold=None, **parameters)
                correct_robust = (
                    robust.correct_twr if factory is TWRInEKF else robust.correct
                )
                correct_plain = (
                    plain.correct_twr if factory is TWRInEKF else plain.correct
                )
                self.assertTrue(correct_robust(anchor, 15.0, anchor_index=1))
                self.assertTrue(correct_plain(anchor, 15.0, anchor_index=1))
                self.assertGreater(robust.last_nis, parameters["gate_threshold"])
                self.assertEqual(robust.stats.downweighted, 0)
                self.assertEqual(robust.stats.mean_accepted_weight, 1.0)

                self.assertTrue(correct_robust(anchor, 12.5, anchor_index=1))
                self.assertTrue(correct_plain(anchor, 12.5, anchor_index=1))
                self.assertEqual(robust.stats.downweighted, 1)
                self.assertLess(robust.stats.mean_accepted_weight, 1.0)
                self.assertGreater(float(np.trace(robust.P)), float(np.trace(plain.P)))
                self.assertFalse(correct_robust(anchor, 15.0, anchor_index=1))
                self.assertEqual(
                    robust.stats.per_anchor[1],
                    {"accepted": 2, "rejected": 1, "downweighted": 1},
                )
                np.testing.assert_allclose(robust.P, robust.P.T, atol=1e-14)
                self.assertGreaterEqual(float(np.min(np.linalg.eigvalsh(robust.P))), -1e-10)


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

    def test_negative_common_sensor_delay_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-negative"):
            load_real_log(
                DEFAULT_DATASET,
                "unused.csv",
                sensor_delay_s=-0.01,
            )

    @unittest.skipUnless(DEFAULT_DATASET.is_dir(), "TU Delft dataset is not installed")
    def test_common_delay_keeps_imu_and_twr_paired_while_aligning_truth(
        self,
    ) -> None:
        sequence = "2021-02-12_09_41_16_twr_triangle.csv"
        raw = load_real_log(DEFAULT_DATASET, sequence, duration=2.3)
        aligned = load_real_log(
            DEFAULT_DATASET,
            sequence,
            duration=2.0,
            sensor_delay_s=TU_DELFT_COMMON_SENSOR_DELAY_S,
        )
        source_index = int(
            np.argmin(
                np.abs(raw.time - TU_DELFT_COMMON_SENSOR_DELAY_S)
            )
        )
        self.assertAlmostEqual(raw.time[source_index], 0.20, places=12)
        self.assertAlmostEqual(aligned.time[0], 0.0, places=12)
        self.assertAlmostEqual(aligned.sensor_delay_s, 0.20, places=12)
        np.testing.assert_array_equal(aligned.imu[0], raw.imu[source_index])
        np.testing.assert_array_equal(
            aligned.ranges[0], raw.ranges[source_index]
        )
        np.testing.assert_array_equal(
            aligned.changed[0], raw.changed[source_index]
        )
        # The packet in the raw +0.20 s row was acquired at raw t=0, so it
        # must be compared with the first truth epoch, not its arrival row.
        np.testing.assert_allclose(
            aligned.ground_truth_position[0],
            raw.ground_truth_position[0],
            rtol=0.0,
            atol=1e-12,
        )

    @unittest.skipUnless(DEFAULT_DATASET.is_dir(), "TU Delft dataset is not installed")
    def test_calibrated_common_delay_reduces_centered_range_residual(
        self,
    ) -> None:
        sequence = "2021-02-12_09_41_16_twr_triangle.csv"
        anchors = load_anchors(DEFAULT_DATASET)
        raw = load_real_log(DEFAULT_DATASET, sequence)
        aligned = load_real_log(
            DEFAULT_DATASET,
            sequence,
            sensor_delay_s=TU_DELFT_COMMON_SENSOR_DELAY_S,
        )

        def centered_rmse(log) -> float:
            centered = []
            for anchor_index, anchor in enumerate(anchors):
                updates = np.flatnonzero(log.changed[:, anchor_index])
                residual = (
                    log.ranges[updates, anchor_index]
                    - np.linalg.norm(
                        log.ground_truth_position[updates] - anchor,
                        axis=1,
                    )
                )
                centered.append(residual - np.mean(residual))
            values = np.concatenate(centered)
            return float(np.sqrt(np.mean(values * values)))

        self.assertLess(
            centered_rmse(aligned),
            centered_rmse(raw) - 0.01,
        )

    @unittest.skipUnless(DEFAULT_DATASET.is_dir(), "TU Delft dataset is not installed")
    def test_delay_diagnostic_has_independent_twr_and_imu_evidence(
        self,
    ) -> None:
        scan = scan_sequence(
            DEFAULT_DATASET,
            "2021-02-12_09_41_16_twr_triangle.csv",
            np.array([0.0, TU_DELFT_COMMON_SENSOR_DELAY_S]),
            load_anchors(DEFAULT_DATASET),
        )
        self.assertLess(
            scan.iloc[1]["range_centered_rmse_m"],
            scan.iloc[0]["range_centered_rmse_m"] - 0.01,
        )
        self.assertLess(
            scan.iloc[1]["imu_standardized_rmse"],
            0.5 * scan.iloc[0]["imu_standardized_rmse"],
        )


if __name__ == "__main__":
    unittest.main()
