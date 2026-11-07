"""Factory-based checks of capture initialization and explicit outage reset.

The mocks expose branch execution and covariance handoff without dataset inputs.
"""

from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import numpy as np

# Keep direct unittest discovery independent of unrelated project modules.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from capture_baseline import CaptureThenTrack
from eskf15 import UpdateStats


class FakeFilter:
    """Minimal active-branch mock with genuine persistent measurement counters."""

    def __init__(self, state, covariance, bias, **parameters):
        """Retain construction inputs and record later executions."""
        self.X = state.copy()
        self.P = covariance.copy()
        self.bias = bias.copy()
        self.parameters = parameters
        self.stats = UpdateStats()
        self.predicted_dt = []
        self.measurements = []
        self.last_nis = 0.0
        self.last_innovation = 0.0

    def predict(self, imu, dt):
        """Advance position so split propagation can be checked independently."""
        self.predicted_dt.append(dt)
        self.X[:3, 4] += self.X[:3, 3] * dt

    def correct_twr(self, anchor, measurement, anchor_index=None):
        """Use deterministic accepted/rejected/invalid inputs for count aggregation."""
        self.measurements.append(measurement)
        if measurement < 0.0:
            self.stats.invalid += 1
            return False
        accepted = measurement < 50.0
        self.last_nis = measurement
        self.last_innovation = measurement
        self.stats.record(measurement, accepted, 1.0 if accepted else 0.0, anchor_index)
        return accepted


class FakeCapture(FakeFilter):
    """Capture mock with the lifted/IMU nesting of the real capture class."""

    def __init__(self, state, covariance, bias, **parameters):
        """Expose the nested objects used for diagnostics and handoff."""
        super().__init__(state, covariance, bias, **parameters)
        self.imu_core = self
        self.lifted = SimpleNamespace(stats=self.stats, last_nis=0.0, last_innovation=0.0)

    def correct_twr(self, anchor, measurement, anchor_index=None):
        """Mirror the latest innovation into lifted diagnostics."""
        accepted = super().correct_twr(anchor, measurement, anchor_index)
        self.lifted.last_nis = self.last_nis
        self.lifted.last_innovation = self.last_innovation
        return accepted


class CaptureBaselineTests(unittest.TestCase):
    """Verify fixed-time one-shot handoff and optional causal outage reset."""

    def make_estimator(self, reset_on_resume=False):
        """Build a supervisor and retain every branch instance for assertions."""
        captures = []
        trackers = []
        handoffs = []

        def capture_factory(state, covariance, bias, **parameters):
            """Record each fresh capture instance."""
            capture = FakeCapture(state, covariance, bias, **parameters)
            captures.append(capture)
            return capture

        def tracker_factory(state, covariance, bias, **parameters):
            """Record each single tracker created at handoff."""
            tracker = FakeFilter(state, covariance, bias, **parameters)
            trackers.append(tracker)
            return tracker

        def handoff_covariance_factory(capture, position_floor, velocity_floor):
            """Record fixed uncertainty floors without using any truth information."""
            handoffs.append((capture.X.copy(), position_floor, velocity_floor))
            return np.eye(15) * 0.75**2

        state = np.eye(5)
        state[:3, 3] = [1.0, 0.0, 0.0]
        estimator = CaptureThenTrack(
            state, np.eye(15), np.arange(6) * 0.01,
            tracker_factory=tracker_factory, imu_core_factory=FakeFilter,
            handoff_covariance_factory=handoff_covariance_factory,
            capture_factory=capture_factory,
            range_sigma=0.3, range_bias=-0.1, gate_threshold=12.25,
            gate_warmup_updates=256, huber_threshold=2.0,
            reset_on_resume=reset_on_resume,
        )
        return estimator, captures, trackers, handoffs

    def test_one_shot_handoff_splits_at_two_seconds_and_stops_capture(self):
        """The capture is discarded at exactly two seconds even inside an IMU step."""
        estimator, captures, trackers, handoffs = self.make_estimator()
        capture = captures[0]
        estimator.predict(np.zeros(6), 1.4)
        self.assertEqual(estimator.mode, "capture")
        self.assertEqual(estimator.tracker_output_weight, 0.0)
        estimator.correct_twr([0, 0, 0], 1.0, anchor_index=0)
        estimator.predict(np.zeros(6), 0.9)
        self.assertEqual(estimator.mode, "track")
        self.assertIsNone(estimator.capture)
        self.assertEqual(estimator.tracker_output_weight, 1.0)
        np.testing.assert_allclose(capture.predicted_dt, [1.4, 0.6])
        np.testing.assert_allclose(trackers[0].predicted_dt, [0.3])
        np.testing.assert_allclose(estimator.position, [2.3, 0.0, 0.0])
        self.assertEqual(handoffs[0][1:], (0.75, 0.75))
        estimator.predict(np.zeros(6), 0.1)
        estimator.correct_twr([0, 0, 0], 2.0, anchor_index=0)
        self.assertEqual(len(capture.predicted_dt), 2)
        self.assertEqual(capture.measurements, [1.0])
        self.assertEqual(trackers[0].measurements, [2.0])
        self.assertEqual(estimator.stats.tracker_reseeds, 1)
        self.assertEqual(estimator.stats.accepted, 2)
        self.assertEqual(estimator.stats.capture_accepted, 1)
        self.assertEqual(estimator.stats.tracker_accepted, 1)
        self.assertEqual(capture.parameters["position_std"], 8.0)
        self.assertEqual(capture.parameters["velocity_std"], 2.0)
        self.assertEqual(capture.parameters["q_std"], 80.0)
        self.assertEqual(capture.parameters["accel_noise_std"], 10.0)
        self.assertEqual(trackers[0].parameters["gate_warmup_updates"], 256)

    def test_resumption_reset_is_optional_and_uses_current_tracker(self):
        """Only a configured reset resumes capture, seeded from the current estimate."""
        simple, _, simple_trackers, _ = self.make_estimator()
        simple.predict(np.zeros(6), 2.0)
        self.assertFalse(simple.on_ranges_resumed())
        self.assertIs(simple.tracker, simple_trackers[0])
        estimator, captures, trackers, handoffs = self.make_estimator(reset_on_resume=True)
        self.assertFalse(estimator.on_ranges_resumed())
        estimator.predict(np.zeros(6), 2.0)
        estimator.predict(np.zeros(6), 0.7)
        trackers[0].bias[:] = 0.2
        current_state = estimator.X
        self.assertTrue(estimator.on_ranges_resumed())
        self.assertEqual(len(captures), 2)
        self.assertIsNone(estimator.tracker)
        np.testing.assert_array_equal(captures[1].X, current_state)
        np.testing.assert_array_equal(captures[1].bias, np.full(6, 0.2))
        np.testing.assert_array_equal(captures[1].P, trackers[0].P)
        self.assertFalse(estimator.on_ranges_resumed())
        estimator.predict(np.zeros(6), 1.5)
        self.assertEqual(estimator.mode, "capture")
        estimator.predict(np.zeros(6), 0.5)
        self.assertEqual(estimator.mode, "track")
        self.assertEqual(len(trackers), 2)
        self.assertEqual(len(handoffs), 2)
        self.assertEqual(estimator.stats.tracker_reseeds, 2)
        self.assertEqual(estimator.stats.capture_restarts, 1)
        np.testing.assert_allclose(estimator.position, [4.7, 0.0, 0.0])

    def test_measurement_diagnostics_survive_all_branch_replacements(self):
        """Valid/rejected/invalid counts and anchor totals persist over outage reset."""
        estimator, _, _, _ = self.make_estimator(reset_on_resume=True)
        estimator.correct_twr([0, 0, 0], 1.0, anchor_index=0)
        estimator.predict(np.zeros(6), 2.0)
        estimator.correct_twr([0, 0, 0], 2.0, anchor_index=0)
        estimator.correct_twr([0, 0, 0], 100.0, anchor_index=1)
        estimator.on_ranges_resumed()
        estimator.correct_twr([0, 0, 0], -1.0, anchor_index=1)
        estimator.correct_twr([0, 0, 0], 3.0, anchor_index=1)
        estimator.predict(np.zeros(6), 2.0)
        self.assertEqual((estimator.stats.accepted, estimator.stats.rejected, estimator.stats.invalid), (3, 1, 1))
        self.assertEqual(estimator.stats.nis_sum, 106.0)
        self.assertEqual(estimator.stats.nis_max, 100.0)
        self.assertEqual(estimator.stats.per_anchor[0]["accepted"], 2)
        self.assertEqual(estimator.stats.per_anchor[1]["accepted"], 1)
        self.assertEqual(estimator.stats.per_anchor[1]["rejected"], 1)
        self.assertEqual(estimator.stats.capture_invalid, 1)
        self.assertEqual(estimator.stats.tracker_rejected, 1)


if __name__ == "__main__":
    unittest.main()
