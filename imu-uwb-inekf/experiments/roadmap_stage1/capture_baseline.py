"""Fixed-duration lifted initialization and optional outage-reset baselines.

Capture runs alone for exactly two seconds, then its state seeds one tracker.
The supervisor uses elapsed propagation time and explicit range-resumption events.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Iterable

import numpy as np

from eskf15 import UpdateStats


# Fixed initialization/reset duration; no trajectory-specific health threshold.
CAPTURE_DURATION_S = 2.0
# Match the full deep model's lifted position prior in metres.
CAPTURE_POSITION_STD = 8.0
# Match the full deep model's lifted velocity prior in metres per second.
CAPTURE_VELOCITY_STD = 2.0
# Match the full deep model's independent quadratic-state priors.
CAPTURE_Q_STD = 80.0
# Prior for the lifted position-velocity inner product.
CAPTURE_S_STD = 40.0
# Prior for squared velocity.
CAPTURE_H_STD = 20.0
# Match the full deep model's world-acceleration sample uncertainty.
CAPTURE_ACCEL_NOISE_STD = 10.0
# Minimum handoff position standard deviation in metres.
HANDOFF_POSITION_STD_FLOOR = 0.75
# Minimum handoff velocity standard deviation in metres per second.
HANDOFF_VELOCITY_STD_FLOOR = 0.75
# Fields summed across filter instances; NIS maximum is aggregated separately.
ADDITIVE_STATS = (
    "accepted", "rejected", "invalid", "nis_sum", "downweighted", "weight_sum"
)


@dataclass
class CaptureStats(UpdateStats):
    """Persistent range counts and one-shot supervision events."""

    tracker_reseeds: int = 0
    capture_restarts: int = 0
    mode_switches_to_track: int = 0
    fallbacks_to_capture: int = 0
    tracker_accepted: int = 0
    tracker_rejected: int = 0
    tracker_invalid: int = 0
    capture_accepted: int = 0
    capture_rejected: int = 0
    capture_invalid: int = 0
    output_track_updates: int = 0
    output_capture_updates: int = 0
    output_full_tracker_updates: int = 0


class CaptureThenTrack:
    """One capture startup followed by one tracker, with optional outage reset."""

    def __init__(
        self,
        state: np.ndarray,
        invariantP: np.ndarray,
        bias: np.ndarray,
        *,
        tracker_factory: Callable,
        imu_core_factory: Callable,
        handoff_covariance_factory: Callable,
        capture_factory: Callable,
        range_sigma: float | Iterable[float] = 0.20,
        range_bias: float | Iterable[float] = -0.18,
        gate_threshold: float = 25.0,
        gate_warmup_updates: int = 0,
        huber_threshold: float | None = None,
        reset_on_resume: bool = False,
    ) -> None:
        """Construct startup capture using the supplied common physical-model factories."""
        self._tracker_factory = tracker_factory
        self._imu_core_factory = imu_core_factory
        self._capture_factory = capture_factory
        self._handoff_covariance_factory = handoff_covariance_factory
        self._range_parameters = {
            "range_sigma": range_sigma,
            "range_bias": range_bias,
            "gate_threshold": gate_threshold,
            "gate_warmup_updates": gate_warmup_updates,
            "huber_threshold": huber_threshold,
        }
        self.reset_on_resume = bool(reset_on_resume)
        self.elapsed_s = 0.0
        self._capture_elapsed_s = 0.0
        self.mode = "capture"
        self.stats = CaptureStats()
        self.tracker = None
        self.capture = self._new_capture(state, invariantP, bias)

    def _new_capture(self, state: np.ndarray, covariance: np.ndarray, bias: np.ndarray):
        """Build an independent lifted capture with the full deep model's priors."""
        return self._capture_factory(
            np.asarray(state).copy(), np.asarray(covariance).copy(), np.asarray(bias).copy(),
            imu_core_factory=self._imu_core_factory,
            position_std=CAPTURE_POSITION_STD,
            velocity_std=CAPTURE_VELOCITY_STD,
            q_std=CAPTURE_Q_STD,
            s_std=CAPTURE_S_STD,
            h_std=CAPTURE_H_STD,
            accel_noise_std=CAPTURE_ACCEL_NOISE_STD,
            **self._range_parameters,
        )

    def _handoff(self) -> None:
        """Replace capture with one tracker using its state and handoff covariance."""
        covariance = self._handoff_covariance_factory(
            self.capture, HANDOFF_POSITION_STD_FLOOR, HANDOFF_VELOCITY_STD_FLOOR
        )
        self.tracker = self._tracker_factory(
            self.capture.X.copy(), np.asarray(covariance).copy(), self.capture.bias.copy(),
            **self._range_parameters,
        )
        self.capture = None
        self.mode = "track"
        self.stats.tracker_reseeds += 1
        self.stats.mode_switches_to_track += 1

    @property
    def X(self) -> np.ndarray:
        """Return the currently active estimator's nominal matrix."""
        return (self.capture if self.mode == "capture" else self.tracker).X.copy()

    @property
    def bias(self) -> np.ndarray:
        """Return the currently active estimator's bias estimate."""
        return (self.capture if self.mode == "capture" else self.tracker).bias.copy()

    @property
    def P(self) -> np.ndarray:
        """Return the active IMU/tracker invariant covariance for reseeding."""
        source = self.capture.imu_core if self.mode == "capture" else self.tracker
        return source.P.copy()

    @property
    def position(self) -> np.ndarray:
        """Return the active world position."""
        return self.X[:3, 4]

    @property
    def velocity(self) -> np.ndarray:
        """Return the active world velocity."""
        return self.X[:3, 3]

    @property
    def tracker_output_weight(self) -> float:
        """Return the discrete output selection without shared-information blending."""
        return 0.0 if self.mode == "capture" else 1.0

    @property
    def last_nis(self) -> float:
        """Forward the currently active branch's latest nominal NIS."""
        branch = self.capture.lifted if self.mode == "capture" else self.tracker
        return getattr(branch, "last_nis", math.nan)

    @property
    def last_innovation(self) -> float:
        """Forward the active branch's latest native range innovation."""
        branch = self.capture.lifted if self.mode == "capture" else self.tracker
        return getattr(branch, "last_innovation", math.nan)

    def predict(self, imu: Iterable[float], dt: float):
        """Split a crossing propagation exactly at the two-second handoff boundary."""
        dt = float(dt)
        if not np.isfinite(dt) or dt < 0.0:
            raise ValueError("dt must be finite and non-negative")
        if self.mode == "track":
            self.tracker.predict(imu, dt)
        else:
            capture_dt = min(dt, CAPTURE_DURATION_S - self._capture_elapsed_s)
            self.capture.predict(imu, capture_dt)
            self._capture_elapsed_s += capture_dt
            remaining = dt - capture_dt
            if self._capture_elapsed_s >= CAPTURE_DURATION_S:
                self._handoff()
                if remaining > 0.0:
                    self.tracker.predict(imu, remaining)
        self.elapsed_s += dt
        return self.X, self.P, self.bias

    @staticmethod
    def _branch_stats(branch):
        """Locate measurement statistics in a tracker or lifted capture."""
        return branch.lifted.stats if hasattr(branch, "lifted") else branch.stats

    def correct_twr(
        self, anchor: Iterable[float], measurement: float, anchor_index: int | None = None
    ) -> bool:
        """Update only the active branch and aggregate its counts across replacements."""
        branch = self.capture if self.mode == "capture" else self.tracker
        source = self._branch_stats(branch)
        before = {name: getattr(source, name) for name in ADDITIVE_STATS}
        anchor_before = {
            index: values.copy() for index, values in source.per_anchor.items()
        }
        accepted = branch.correct_twr(anchor, measurement, anchor_index=anchor_index)
        for name in ADDITIVE_STATS:
            change = getattr(source, name) - before[name]
            setattr(self.stats, name, getattr(self.stats, name) + change)
        self.stats.nis_max = max(self.stats.nis_max, source.nis_max)
        for index, values in source.per_anchor.items():
            target = self.stats.per_anchor.setdefault(
                index, {"accepted": 0, "rejected": 0, "downweighted": 0}
            )
            for key, value in values.items():
                target[key] += value - anchor_before.get(index, {}).get(key, 0)
        for name in ("accepted", "rejected", "invalid"):
            field = f"{self.mode if self.mode == 'capture' else 'tracker'}_{name}"
            setattr(self.stats, field, getattr(self.stats, field) + getattr(source, name) - before[name])
        if self.mode == "capture":
            self.stats.output_capture_updates += 1
        else:
            self.stats.output_track_updates += 1
            self.stats.output_full_tracker_updates += 1
        return accepted

    def on_ranges_resumed(self) -> bool:
        """Start a fresh two-second capture only for the configured reset baseline."""
        if not self.reset_on_resume or self.mode != "track":
            return False
        self.capture = self._new_capture(self.tracker.X, self.tracker.P, self.tracker.bias)
        self.tracker = None
        self.mode = "capture"
        self._capture_elapsed_s = 0.0
        self.stats.capture_restarts += 1
        self.stats.fallbacks_to_capture += 1
        return True
