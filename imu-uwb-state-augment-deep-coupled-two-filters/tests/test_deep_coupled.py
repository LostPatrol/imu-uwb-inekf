"""Check capture supervision, indexed calibration, and injectable deep cores."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]  # Deep-coupled project root.
ESTIMATION = PROJECT / "scripts" / "estimation"  # Local filter modules.
sys.path.insert(0, str(ESTIMATION))

from deep_coupled import DeepCoupledTwoFilter  # noqa: E402
from deep_coupled_experiment import (  # noqa: E402
    SYNTHETIC_END_TO_END_DURATION_S,
    run_synthetic_deep,
    run_synthetic_monte_carlo,
)
from lifted_twr import (  # noqa: E402
    InEKF,
    LiftedTWRInEKF,
    TWRInEKF,
    paper_consistent_initial_covariance,
)
from twr_experiment import (  # noqa: E402
    DEFAULT_DATASET,
    TU_DELFT_COMMON_SENSOR_DELAY_S,
    _initial_state,
    load_anchors,
    load_real_log,
    run_real_sequence,
    trajectory_metrics,
)


def initial_filter_state(
    position: tuple[float, float, float] = (0.5, -0.2, 1.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a small, finite navigation prior for structural checks."""

    state = np.eye(5)
    state[:3, 4] = position
    covariance = paper_consistent_initial_covariance(
        position_std=1.0, velocity_std=0.5
    )
    return state, covariance, np.zeros(6)


class DeepCoupledStructureTests(unittest.TestCase):
    def test_indexed_policy_and_factory_hooks_survive_tracker_reseed(self) -> None:
        """Verify calibration, core synchronization, and reseed covariance hooks."""

        class TranslationAwareCore(InEKF):
            """Expose the translation hook required by coordinate-aware cores."""

            def __init__(self, state, covariance, bias) -> None:
                """Initialize the real IMU core and count translation changes."""

                super().__init__(state, covariance, bias)
                self.translation_updates = 0

            def set_translation(self, velocity, position) -> None:
                """Synchronize the state through the injected core interface."""

                self.translation_updates += 1
                self.X[:3, 3] = velocity
                self.X[:3, 4] = position
                self.X_hat = self.X.copy()

        tracker_settings = []
        handoffs = []

        def tracker_factory(state, covariance, bias, **settings):
            """Build the actual tracker while retaining its protocol arguments."""

            tracker_settings.append(settings)
            return TWRInEKF(state, covariance, bias, **settings)

        def handoff_factory(capture, position_floor, velocity_floor):
            """Return a distinctive covariance to verify the handoff hook."""

            handoffs.append((capture, position_floor, velocity_floor))
            return np.eye(15) * 0.7

        state, covariance, bias = initial_filter_state()
        sigma = np.array([0.1, 0.3])
        range_bias = np.array([-0.2, 0.6])
        estimator = DeepCoupledTwoFilter(
            state,
            covariance,
            bias,
            range_sigma=sigma,
            range_bias=range_bias,
            huber_threshold=1.5,
            capture_gate_warmup_updates=1,
            tracker_gate_warmup_updates=1,
            imu_core_factory=TranslationAwareCore,
            tracker_factory=tracker_factory,
            handoff_covariance_factory=handoff_factory,
            handoff_position_std_floor_m=0.8,
            handoff_velocity_std_floor_mps=0.9,
            first_reseed_time_s=100.0,
        )
        core = estimator.capture.imu_core
        self.assertIsInstance(core, TranslationAwareCore)
        self.assertEqual(core.translation_updates, 1)
        estimator.predict([0, 0, 1, 0, 0, 0], 0.01)
        self.assertEqual(core.translation_updates, 2)

        # External calibration changes must not alter either branch or a reseed.
        sigma[:] = 9.0
        range_bias[:] = 9.0
        anchor = np.array([4.0, -3.0, 2.0])
        measured_range = float(np.linalg.norm(anchor - state[:3, 4])) + 0.6
        self.assertEqual(
            estimator.correct_twr(anchor, measured_range, anchor_index=1),
            (True, True),
        )
        self.assertEqual(core.translation_updates, 3)
        self.assertAlmostEqual(estimator._capture_prefit_residuals[-1], 0.0)
        self.assertAlmostEqual(estimator._tracker_prefit_residuals[-1], 0.0)
        self.assertEqual(estimator.capture.lifted.stats.per_anchor[1]["accepted"], 1)
        self.assertEqual(estimator.tracker.stats.per_anchor[1]["accepted"], 1)

        previous_tracker = estimator.tracker
        estimator._reseed_tracker()
        self.assertIsNot(estimator.tracker, previous_tracker)
        self.assertIs(estimator.capture.imu_core, core)
        self.assertEqual(handoffs, [(estimator.capture, 0.8, 0.9)])
        np.testing.assert_array_equal(estimator.tracker.P, np.eye(15) * 0.7)
        self.assertEqual(len(tracker_settings), 2)
        for settings in tracker_settings:
            np.testing.assert_array_equal(settings["range_sigma"], [0.1, 0.3])
            np.testing.assert_array_equal(settings["range_bias"], [-0.2, 0.6])
            self.assertEqual(settings["huber_threshold"], 1.5)
            self.assertEqual(settings["gate_warmup_updates"], 1)
        self.assertEqual(estimator.capture.lifted.huber_threshold, 1.5)
        self.assertEqual(estimator.capture.lifted.gate_warmup_updates, 1)
        self.assertEqual(estimator.stats.tracker_reseeds, 1)

        measured_range = float(
            np.linalg.norm(anchor - estimator.capture.lifted.position)
        ) - 0.2
        self.assertEqual(
            estimator.correct_twr(anchor, measured_range, anchor_index=0),
            (True, True),
        )
        self.assertEqual(estimator.tracker.stats.per_anchor[0]["accepted"], 1)

    def test_missing_anchor_index_does_not_create_health_evidence(self) -> None:
        """Treat a missing calibration index as invalid in both branches."""

        state, covariance, bias = initial_filter_state()
        estimator = DeepCoupledTwoFilter(
            state,
            covariance,
            bias,
            range_sigma=[0.1, 0.2],
            range_bias=[-0.2, 0.3],
            first_reseed_time_s=100.0,
        )
        self.assertEqual(estimator.correct_twr([4, -3, 2], 5.0), (False, False))
        self.assertEqual(estimator.capture.lifted.stats.invalid, 1)
        self.assertEqual(estimator.tracker.stats.invalid, 1)
        self.assertEqual(estimator.stats.tracker_invalid, 1)
        self.assertEqual(len(estimator._capture_prefit_residuals), 0)
        self.assertEqual(len(estimator._tracker_prefit_residuals), 0)
        self.assertEqual(estimator.tracker_recent_distinct_anchor_count, 0)

    def test_capture_branch_is_identical_to_original_loose_cascade(self) -> None:
        state, covariance, bias = initial_filter_state()
        parameters = {
            "position_std": 1.0,
            "velocity_std": 0.5,
            "q_std": 10.0,
            "s_std": 5.0,
            "h_std": 5.0,
            "accel_noise_std": 0.5,
            "range_sigma": 0.1,
            "range_bias": 0.0,
            "gate_threshold": 25.0,
            "gate_warmup_updates": 16,
        }
        loose = LiftedTWRInEKF(
            state.copy(), covariance.copy(), bias.copy(), **parameters
        )
        deep = DeepCoupledTwoFilter(
            state.copy(),
            covariance.copy(),
            bias.copy(),
            position_std=parameters["position_std"],
            velocity_std=parameters["velocity_std"],
            q_std=parameters["q_std"],
            s_std=parameters["s_std"],
            h_std=parameters["h_std"],
            accel_noise_std=parameters["accel_noise_std"],
            range_sigma=parameters["range_sigma"],
            range_bias=parameters["range_bias"],
            gate_threshold=parameters["gate_threshold"],
            capture_gate_warmup_updates=parameters["gate_warmup_updates"],
            first_reseed_time_s=100.0,
        )
        anchors = np.array(
            [
                [-4.0, -4.0, 0.0],
                [4.0, -4.0, 0.0],
                [4.0, 4.0, 3.0],
                [-4.0, 4.0, 3.0],
            ]
        )
        truth = state[:3, 4]
        for _ in range(8):
            loose.predict([0, 0, 1, 0, 0, 0], 0.01)
            deep.predict([0, 0, 1, 0, 0, 0], 0.01)
            for anchor in anchors:
                measured_range = float(np.linalg.norm(anchor - truth))
                self.assertEqual(
                    loose.correct_twr(anchor, measured_range),
                    deep.correct_twr(anchor, measured_range)[0],
                )
        np.testing.assert_array_equal(
            loose.lifted.chi, deep.capture.lifted.chi
        )
        np.testing.assert_array_equal(
            loose.lifted.P, deep.capture.lifted.P
        )
        np.testing.assert_array_equal(loose.X, deep.capture.X)
        np.testing.assert_array_equal(loose.bias, deep.capture.bias)

    def test_healthy_tracker_weight_ramps_smoothly_to_one(self) -> None:
        state, covariance, bias = initial_filter_state()
        estimator = DeepCoupledTwoFilter(
            state,
            covariance,
            bias,
            track_output_weight=0.30,
            maximum_track_output_weight=1.0,
            track_weight_rise_time_s=5.0,
            track_weight_fall_time_s=1.0,
            tracker_health_smoothing_time_s=0.1,
            full_weight_health_enter=0.5,
            full_weight_health_exit=0.25,
            full_weight_health_hold_s=0.5,
            health_window=16,
            min_track_updates=8,
        )
        estimator.capture.lifted.chi[3:6] = [1.0, 2.0, 3.0]
        estimator.capture.lifted.chi[6:9] = [0.1, 0.2, 0.3]
        estimator.tracker.X[:3, 4] = [1.3, 2.3, 3.3]
        estimator.tracker.X[:3, 3] = [0.4, 0.5, 0.6]
        for _ in range(8):
            estimator._capture_accepts.append(True)
            estimator._tracker_accepts.append(True)
            estimator._capture_prefit_residuals.append(0.20)
            estimator._tracker_prefit_residuals.append(0.05)
        estimator._tracker_updates_since_seed = 8

        estimator.mode = "capture"
        np.testing.assert_allclose(estimator.position, [1.0, 2.0, 3.0])
        estimator._enter_track_mode()
        self.assertAlmostEqual(estimator.tracker_health_score, 1.0)
        self.assertAlmostEqual(estimator.tracker_output_weight, 0.30)
        np.testing.assert_allclose(estimator.position, [1.09, 2.09, 3.09])
        np.testing.assert_allclose(estimator.velocity, [0.19, 0.29, 0.39])

        estimator._advance_smoothed_health(1.0)
        self.assertTrue(estimator.full_tracker_output_authorized)
        estimator._advance_output_weight(2.5)
        self.assertAlmostEqual(estimator.tracker_output_weight, 0.65)
        np.testing.assert_allclose(
            estimator.position, [1.195, 2.195, 3.195]
        )
        estimator._advance_output_weight(2.5)
        self.assertAlmostEqual(estimator.tracker_output_weight, 1.0)
        np.testing.assert_allclose(estimator.position, [1.3, 2.3, 3.3])

        # A causal health loss reduces the contribution before confirmed
        # fallback switches the public output fully back to capture.
        estimator.tracker.X[:3, 4] = [4.0, 5.0, 6.0]
        self.assertAlmostEqual(estimator.tracker_health_score, 0.0)
        estimator._advance_smoothed_health(1.0)
        self.assertFalse(estimator.full_tracker_output_authorized)
        estimator._advance_output_weight(0.5)
        self.assertAlmostEqual(estimator.tracker_output_weight, 0.5)
        estimator._advance_output_weight(0.5)
        self.assertAlmostEqual(estimator.tracker_output_weight, 0.3)

    def test_handoff_covariance_is_symmetric_positive_semidefinite(self) -> None:
        state, covariance, bias = initial_filter_state()
        estimator = DeepCoupledTwoFilter(state, covariance, bias)
        handoff = estimator._handoff_covariance()
        np.testing.assert_allclose(handoff, handoff.T, atol=1e-14)
        self.assertGreaterEqual(
            float(np.min(np.linalg.eigvalsh(handoff))), -1e-12
        )
        self.assertGreater(
            float(np.min(np.diag(handoff)[3:9])), 0.5**2
        )

    def test_persistently_weak_health_removes_partial_tracker_output(
        self,
    ) -> None:
        state, covariance, bias = initial_filter_state()
        estimator = DeepCoupledTwoFilter(
            state,
            covariance,
            bias,
            track_output_weight=0.30,
            minimum_track_output_health=0.27,
            track_output_health_grace_s=5.0,
            track_weight_rise_time_s=5.0,
            track_weight_fall_time_s=1.0,
        )
        capture_position = estimator.capture.lifted.position.copy()
        estimator.tracker.X[:3, 4] = capture_position + 1.0
        estimator._enter_track_mode()

        # The hand-off begins at 30% even though the slow health state starts
        # from zero.
        self.assertAlmostEqual(estimator.tracker_output_weight, 0.30)
        estimator.elapsed_s = 4.99
        estimator._advance_output_weight(0.30)
        self.assertAlmostEqual(estimator.tracker_output_weight, 0.30)

        # Once the grace period has expired, persistently weak evidence
        # slews the public output fully back to capture without resetting the
        # still-running tracker.
        estimator.elapsed_s = 6.0
        estimator._advance_output_weight(0.30)
        self.assertAlmostEqual(estimator.tracker_output_weight, 0.0)
        self.assertEqual(estimator.mode, "track")
        np.testing.assert_allclose(estimator.position, capture_position)

        # Recovery uses only the smoothed causal health state and the same
        # bounded slew limiter.
        estimator._smoothed_tracker_health = 0.28
        estimator._advance_output_weight(1.0)
        self.assertAlmostEqual(estimator.tracker_output_weight, 0.14)
        estimator._advance_output_weight(2.0)
        self.assertAlmostEqual(estimator.tracker_output_weight, 0.30)

    def test_healthy_tracker_switches_and_nonfinite_tracker_falls_back(
        self,
    ) -> None:
        state, covariance, bias = initial_filter_state()
        estimator = DeepCoupledTwoFilter(
            state,
            covariance,
            bias,
            range_sigma=0.05,
            range_bias=0.0,
            health_window=16,
            min_track_updates=8,
            minimum_capture_dwell_s=0.0,
            switch_hold_updates=2,
            fallback_hold_updates=2,
        )
        anchors = np.array(
            [
                [-4.0, -4.0, 0.0],
                [4.0, -4.0, 0.0],
                [4.0, 4.0, 0.0],
                [-4.0, 4.0, 0.0],
                [-4.0, -4.0, 3.0],
                [4.0, -4.0, 3.0],
                [4.0, 4.0, 3.0],
                [-4.0, 4.0, 3.0],
            ]
        )
        truth = state[:3, 4]
        estimator.predict([0, 0, 1, 0, 0, 0], 0.01)
        for _ in range(3):
            for anchor in anchors:
                estimator.correct_twr(
                    anchor, float(np.linalg.norm(anchor - truth))
                )
        self.assertEqual(estimator.mode, "track")
        self.assertEqual(estimator.stats.mode_switches_to_track, 1)

        estimator.tracker.X[0, 0] = np.nan
        estimator._update_mode()
        estimator._update_mode()
        self.assertEqual(estimator.mode, "capture")
        self.assertEqual(estimator.stats.fallbacks_to_capture, 1)
        self.assertEqual(estimator.stats.tracker_reseeds, 1)
        self.assertAlmostEqual(estimator.tracker_output_weight, 0.0)
        np.testing.assert_allclose(
            estimator.position, estimator.capture.lifted.position
        )

    def test_decisive_range_override_requires_multiple_anchors(self) -> None:
        state, covariance, bias = initial_filter_state()
        estimator = DeepCoupledTwoFilter(
            state,
            covariance,
            bias,
            health_window=16,
            min_track_updates=8,
            decisive_range_advantage_m=0.40,
            minimum_decisive_anchor_count=4,
        )
        for _ in range(8):
            estimator._capture_prefit_residuals.append(1.0)
            estimator._tracker_prefit_residuals.append(0.1)
            estimator._tracker_accepts.append(True)
            estimator._tracker_anchor_history.append((0.0, 0.0, 0.0))
        estimator._tracker_updates_since_seed = 8
        estimator._refresh_decisive_range_advantage()
        self.assertFalse(estimator.tracker_has_decisive_range_advantage)

        estimator._tracker_anchor_history.clear()
        anchors = (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        )
        for index in range(8):
            estimator._tracker_anchor_history.append(anchors[index % 4])
        estimator._refresh_decisive_range_advantage()
        self.assertTrue(estimator.tracker_has_decisive_range_advantage)

        estimator._enter_track_mode()
        estimator.tracker.X[:3, 4] = (
            estimator.capture.lifted.position
            + np.array([10.0, 0.0, 0.0])
        )
        self.assertFalse(estimator._tracker_is_lost())

    def test_synthetic_end_to_end_rejects_capture_only_duration(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 30.0 s"):
            run_synthetic_monte_carlo(
                PROJECT / "results" / "invalid_short_synthetic_test",
                trials=1,
                duration=20.0,
            )

    def test_synthetic_end_to_end_exercises_adaptive_output(self) -> None:
        generator = np.random.default_rng(20260731)
        initial_position = generator.uniform(-50.0, 50.0, size=3)
        initial_velocity = generator.uniform(-10.0, 10.0, size=3)
        result = run_synthetic_deep(
            seed=0,
            initial_position=initial_position,
            initial_velocity=initial_velocity,
            duration=SYNTHETIC_END_TO_END_DURATION_S,
        )
        coupling = result["metrics"]["coupling"]
        self.assertGreaterEqual(coupling["mode_switches_to_track"], 1)
        self.assertAlmostEqual(
            coupling["maximum_observed_tracker_output_weight"], 1.0
        )
        self.assertGreater(
            coupling["maximum_deep_capture_position_difference_m"],
            1e-3,
        )
        self.assertGreater(
            coupling[
                "tail5_maximum_deep_capture_position_difference_m"
            ],
            1e-3,
        )
        methods = result["metrics"]["methods"]
        self.assertLess(
            methods["deep_coupled"]["position_tail5_rmse_m"],
            methods["loose_lifted"]["position_tail5_rmse_m"],
        )


class DeepCoupledRealRegressionTests(unittest.TestCase):
    @unittest.skipUnless(
        DEFAULT_DATASET.is_dir(), "TU Delft dataset is not installed"
    )
    def test_representative_real_run_does_not_regress_loose_output(
        self,
    ) -> None:
        from deep_coupled_experiment import run_real_deep_sequence

        log = load_real_log(
            DEFAULT_DATASET,
            "2021-02-12_09_56_09_twr_random.csv",
            duration=30.0,
            sensor_delay_s=TU_DELFT_COMMON_SENSOR_DELAY_S,
        )
        result = run_real_deep_sequence(
            log,
            load_anchors(DEFAULT_DATASET),
            initial_offset=(6.0, -6.0, 3.0),
        )
        self.assertAlmostEqual(
            result["metrics"]["sensor_delay_s"],
            TU_DELFT_COMMON_SENSOR_DELAY_S,
        )
        loose = result["metrics"]["methods"]["loose_lifted"]
        deep = result["metrics"]["methods"]["deep_coupled"]
        for metric in (
            "position_rmse_m",
            "position_tail5_rmse_m",
            "velocity_rmse_mps",
            "velocity_tail5_rmse_mps",
        ):
            self.assertLessEqual(deep[metric], loose[metric] + 1e-9)

        # Recompute one metric independently so the test is not merely
        # comparing two values emitted by the same metrics dictionary.
        deep_history = result["histories"]["deep_coupled"]
        recomputed = trajectory_metrics(
            result["time"],
            deep_history["position"],
            deep_history["velocity"],
            result["ground_truth_position"],
            result["ground_truth_velocity"],
        )
        self.assertAlmostEqual(
            recomputed["position_rmse_m"],
            deep["position_rmse_m"],
            places=14,
        )

    @unittest.skipUnless(
        DEFAULT_DATASET.is_dir(), "TU Delft dataset is not installed"
    )
    def test_added_ekf_and_inekf_match_original_task06_baselines(
        self,
    ) -> None:
        from deep_coupled_experiment import run_real_deep_sequence

        log = load_real_log(
            DEFAULT_DATASET,
            "2021-02-12_09_41_16_twr_triangle.csv",
            duration=2.0,
            sensor_delay_s=TU_DELFT_COMMON_SENSOR_DELAY_S,
        )
        anchors = load_anchors(DEFAULT_DATASET)
        original = run_real_sequence(
            log,
            anchors,
            initial_offset=(6.0, -6.0, 3.0),
            include_ordinary_inekf=True,
        )
        expanded = run_real_deep_sequence(
            log,
            anchors,
            initial_offset=(6.0, -6.0, 3.0),
        )
        np.testing.assert_array_equal(
            expanded["histories"]["ordinary_ekf"]["position"],
            original["histories"]["direct_twr"]["position"],
        )
        np.testing.assert_array_equal(
            expanded["histories"]["ordinary_ekf"]["velocity"],
            original["histories"]["direct_twr"]["velocity"],
        )
        np.testing.assert_array_equal(
            expanded["histories"]["ordinary_inekf"]["position"],
            original["histories"]["twr_inekf"]["position"],
        )
        np.testing.assert_array_equal(
            expanded["histories"]["ordinary_inekf"]["velocity"],
            original["histories"]["twr_inekf"]["velocity"],
        )


if __name__ == "__main__":
    unittest.main()
