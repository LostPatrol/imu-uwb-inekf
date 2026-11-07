"""Guarded lifted capture and raw-range InEKF tracking for IMU/TWR navigation.

The independent capture branch provides discrete reseeds and monitors health;
causal supervision gradually transfers the output to the joint tracker.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Callable, Iterable

import numpy as np

from lifted_twr import (
    P_SLICE,
    V_SLICE,
    LiftedTWRInEKF,
    TWRInEKF,
    _anchor_parameter,
    _as_vector3,
    _for_anchor,
    _symmetrize,
)


@dataclass
class CouplingStats:
    """Supervisory statistics independent of filter-instance resets."""

    mode_switches_to_track: int = 0
    fallbacks_to_capture: int = 0
    tracker_reseeds: int = 0
    tracker_accepted: int = 0
    tracker_rejected: int = 0
    tracker_invalid: int = 0
    output_track_updates: int = 0
    output_capture_updates: int = 0
    output_health_guard_updates: int = 0
    output_full_tracker_updates: int = 0
    output_transition_updates: int = 0
    decisive_range_override_updates: int = 0


class DeepCoupledTwoFilter:
    """Large-error lifted capture plus tightly-coupled TWR-InEKF tracking.

    ``mode`` is either ``"capture"`` or ``"track"``.  The capture estimator
    is never overwritten by the tracker, so a bad tightly-coupled correction
    cannot destroy the recovery path.
    """

    def __init__(
        self,
        X0: np.ndarray,
        P0: np.ndarray,
        bias: np.ndarray,
        *,
        position_std: float = 8.0,
        velocity_std: float = 2.0,
        q_std: float = 80.0,
        s_std: float = 40.0,
        h_std: float = 20.0,
        accel_noise_std: float = 10.0,
        range_sigma: float | Iterable[float] = 0.20,
        range_bias: float | Iterable[float] = -0.18,
        gate_threshold: float = 25.0,
        capture_gate_warmup_updates: int = 64,
        tracker_gate_warmup_updates: int = 128,
        huber_threshold: float | None = None,
        lever_arm: Iterable[float] = (0.0, 0.0, 0.0),
        tracker_factory: Callable[..., TWRInEKF] | None = None,
        imu_core_factory: Callable | None = None,
        handoff_covariance_factory: Callable[
            [LiftedTWRInEKF, float, float], np.ndarray
        ] | None = None,
        health_window: int = 96,
        min_track_updates: int = 64,
        agreement_threshold_m: float = 0.80,
        fallback_disagreement_m: float = 2.50,
        max_track_reject_fraction: float = 0.20,
        fallback_reject_fraction: float = 0.45,
        max_prefit_range_rmse_m: float = 0.80,
        range_score_margin_m: float = 0.08,
        full_weight_agreement_m: float = 0.10,
        range_evidence_ratio_floor: float = 0.03,
        range_evidence_ratio_full: float = 0.12,
        decisive_range_advantage_m: float = 0.40,
        minimum_decisive_anchor_count: int = 4,
        first_reseed_time_s: float = 2.0,
        reseed_cooldown_s: float = 2.0,
        capture_position_std_threshold_m: float = 2.5,
        handoff_position_std_floor_m: float = 0.75,
        handoff_velocity_std_floor_mps: float = 0.75,
        track_output_weight: float = 0.30,
        maximum_track_output_weight: float = 1.0,
        track_weight_rise_time_s: float = 5.0,
        track_weight_fall_time_s: float = 1.0,
        tracker_health_smoothing_time_s: float = 10.0,
        full_weight_health_enter: float = 0.38,
        full_weight_health_exit: float = 0.25,
        full_weight_health_hold_s: float = 1.0,
        minimum_track_output_health: float = 0.27,
        track_output_health_grace_s: float = 5.0,
        minimum_capture_dwell_s: float = 20.0,
        switch_hold_updates: int = 12,
        fallback_hold_updates: int = 12,
    ) -> None:
        """Initialize both branches with a shared range and robust policy."""

        state = np.asarray(X0, dtype=np.float64)
        covariance = np.asarray(P0, dtype=np.float64)
        bias_vector = np.asarray(bias, dtype=np.float64).reshape(-1)
        if state.shape != (5, 5) or covariance.shape != (15, 15):
            raise ValueError("X0 and P0 must have shapes (5,5) and (15,15)")
        if bias_vector.shape != (6,):
            raise ValueError("bias must contain [gyro bias, accelerometer bias]")
        if health_window < 16 or min_track_updates < 8:
            raise ValueError("health windows are too short for guarded switching")

        self._tracker_factory = (
            TWRInEKF if tracker_factory is None else tracker_factory
        )
        self._handoff_covariance_factory = handoff_covariance_factory
        self._tracker_parameters = {
            "range_sigma": _anchor_parameter(
                range_sigma, "range_sigma", positive=True
            ),
            "range_bias": _anchor_parameter(range_bias, "range_bias"),
            "gate_threshold": float(gate_threshold),
            "gate_warmup_updates": int(tracker_gate_warmup_updates),
            "huber_threshold": (
                None if huber_threshold is None else float(huber_threshold)
            ),
            "lever_arm": _as_vector3(lever_arm, "lever_arm"),
        }
        self.capture = LiftedTWRInEKF(
            state.copy(),
            covariance.copy(),
            bias_vector.copy(),
            position_std=position_std,
            velocity_std=velocity_std,
            q_std=q_std,
            s_std=s_std,
            h_std=h_std,
            accel_noise_std=accel_noise_std,
            range_sigma=range_sigma,
            range_bias=range_bias,
            gate_threshold=gate_threshold,
            gate_warmup_updates=capture_gate_warmup_updates,
            huber_threshold=huber_threshold,
            imu_core_factory=imu_core_factory,
        )
        self.tracker = self._new_tracker(
            state.copy(), covariance.copy(), bias_vector.copy()
        )

        self.mode = "capture"
        self.elapsed_s = 0.0
        self.last_reseed_s = -math.inf
        self.stats = CouplingStats()

        self.health_window = int(health_window)
        self.min_track_updates = int(min_track_updates)
        self.agreement_threshold_m = float(agreement_threshold_m)
        self.fallback_disagreement_m = float(fallback_disagreement_m)
        self.max_track_reject_fraction = float(max_track_reject_fraction)
        self.fallback_reject_fraction = float(fallback_reject_fraction)
        self.max_prefit_range_rmse_m = float(max_prefit_range_rmse_m)
        self.range_score_margin_m = float(range_score_margin_m)
        self.full_weight_agreement_m = float(full_weight_agreement_m)
        self.range_evidence_ratio_floor = float(
            range_evidence_ratio_floor
        )
        self.range_evidence_ratio_full = float(
            range_evidence_ratio_full
        )
        self.decisive_range_advantage_m = float(
            decisive_range_advantage_m
        )
        self.minimum_decisive_anchor_count = int(
            minimum_decisive_anchor_count
        )
        if (
            self.full_weight_agreement_m <= 0.0
            or self.range_evidence_ratio_floor < 0.0
            or self.range_evidence_ratio_full
            <= self.range_evidence_ratio_floor
            or self.decisive_range_advantage_m <= 0.0
            or self.minimum_decisive_anchor_count < 2
        ):
            raise ValueError("adaptive range-evidence thresholds are invalid")
        self.first_reseed_time_s = float(first_reseed_time_s)
        self.reseed_cooldown_s = float(reseed_cooldown_s)
        self.capture_position_std_threshold_m = float(
            capture_position_std_threshold_m
        )
        self.handoff_position_std_floor_m = float(
            handoff_position_std_floor_m
        )
        self.handoff_velocity_std_floor_mps = float(
            handoff_velocity_std_floor_mps
        )
        # ``track_output_weight`` is retained as a backward-compatible name
        # for the initial hand-off weight.  It is no longer the fixed
        # steady-state weight.
        self.track_output_weight = float(track_output_weight)
        self.maximum_track_output_weight = float(
            maximum_track_output_weight
        )
        if not (
            0.0
            < self.track_output_weight
            <= self.maximum_track_output_weight
            <= 1.0
        ):
            raise ValueError(
                "output weights must satisfy "
                "0 < initial <= maximum <= 1"
            )
        self.track_weight_rise_time_s = float(track_weight_rise_time_s)
        self.track_weight_fall_time_s = float(track_weight_fall_time_s)
        if (
            self.track_weight_rise_time_s <= 0.0
            or self.track_weight_fall_time_s <= 0.0
        ):
            raise ValueError("weight rise/fall times must be positive")
        self.tracker_health_smoothing_time_s = float(
            tracker_health_smoothing_time_s
        )
        self.full_weight_health_enter = float(full_weight_health_enter)
        self.full_weight_health_exit = float(full_weight_health_exit)
        self.full_weight_health_hold_s = float(full_weight_health_hold_s)
        self.minimum_track_output_health = float(
            minimum_track_output_health
        )
        self.track_output_health_grace_s = float(
            track_output_health_grace_s
        )
        if (
            self.tracker_health_smoothing_time_s <= 0.0
            or not 0.0
            <= self.full_weight_health_exit
            < self.full_weight_health_enter
            <= 1.0
            or self.full_weight_health_hold_s < 0.0
            or not 0.0 <= self.minimum_track_output_health <= 1.0
            or self.track_output_health_grace_s < 0.0
        ):
            raise ValueError("smoothed-health hysteresis is invalid")
        self._tracker_output_weight = 0.0
        self._smoothed_tracker_health = 0.0
        self._full_weight_health_hold_s = 0.0
        self._full_tracker_output_authorized = False
        self._track_mode_started_s = -math.inf
        self.minimum_capture_dwell_s = float(minimum_capture_dwell_s)
        if self.minimum_capture_dwell_s < 0.0:
            raise ValueError("minimum_capture_dwell_s must be non-negative")
        self.switch_hold_updates = int(switch_hold_updates)
        self.fallback_hold_updates = int(fallback_hold_updates)

        self._capture_accepts: deque[bool] = deque(maxlen=self.health_window)
        self._tracker_accepts: deque[bool] = deque(maxlen=self.health_window)
        self._capture_prefit_residuals: deque[float] = deque(
            maxlen=self.health_window
        )
        self._tracker_prefit_residuals: deque[float] = deque(
            maxlen=self.health_window
        )
        self._tracker_anchor_history: deque[
            tuple[float, float, float]
        ] = deque(maxlen=self.health_window)
        self._tracker_decisive_range_advantage = False
        self._tracker_updates_since_seed = 0
        self._switch_ready_count = 0
        self._fallback_count = 0

    def _new_tracker(
        self, state: np.ndarray, covariance: np.ndarray, bias: np.ndarray
    ) -> TWRInEKF:
        """Retain the tracker implementation and range policy across reseeds."""

        return self._tracker_factory(
            state, covariance, bias, **self._tracker_parameters
        )

    @property
    def position(self) -> np.ndarray:
        if self.mode == "track":
            weight = self.tracker_output_weight
            return (
                (1.0 - weight) * self.capture.lifted.position
                + weight * self.tracker.X[:3, 4]
            )
        return self.capture.lifted.position

    @property
    def velocity(self) -> np.ndarray:
        if self.mode == "track":
            weight = self.tracker_output_weight
            return (
                (1.0 - weight) * self.capture.lifted.velocity
                + weight * self.tracker.X[:3, 3]
            )
        return self.capture.lifted.velocity

    @property
    def X(self) -> np.ndarray:
        if self.mode == "track":
            result = self.tracker.X.copy()
            result[:3, 3] = self.velocity
            result[:3, 4] = self.position
            return result
        return self.capture.X

    @property
    def bias(self) -> np.ndarray:
        if self.mode == "track":
            return self.tracker.bias.copy()
        return self.capture.bias

    @property
    def disagreement_m(self) -> float:
        return float(
            np.linalg.norm(
                self.tracker.X[:3, 4] - self.capture.lifted.position
            )
        )

    @staticmethod
    def _reject_fraction(window: deque[bool]) -> float:
        if not window:
            return 1.0
        return 1.0 - float(np.mean(np.asarray(window, dtype=np.float64)))

    @staticmethod
    def _range_rmse(window: deque[float]) -> float:
        if not window:
            return math.inf
        values = np.asarray(window, dtype=np.float64)
        return float(np.sqrt(np.mean(values * values)))

    @property
    def capture_prefit_range_rmse_m(self) -> float:
        return self._range_rmse(self._capture_prefit_residuals)

    @property
    def tracker_prefit_range_rmse_m(self) -> float:
        return self._range_rmse(self._tracker_prefit_residuals)

    @property
    def tracker_recent_reject_fraction(self) -> float:
        return self._reject_fraction(self._tracker_accepts)

    @property
    def tracker_range_evidence_ratio(self) -> float:
        """Range-fit gain per metre of capture/tracker state separation."""

        capture_score = self.capture_prefit_range_rmse_m
        tracker_score = self.tracker_prefit_range_rmse_m
        disagreement = self.disagreement_m
        if not (
            np.isfinite(capture_score)
            and np.isfinite(tracker_score)
            and np.isfinite(disagreement)
        ):
            return 0.0
        return float(
            max(capture_score - tracker_score, 0.0)
            / max(disagreement, self.full_weight_agreement_m)
        )

    @property
    def tracker_range_evidence_score(self) -> float:
        if self.disagreement_m <= self.full_weight_agreement_m:
            return 1.0
        return float(
            np.clip(
                (
                    self.tracker_range_evidence_ratio
                    - self.range_evidence_ratio_floor
                )
                / (
                    self.range_evidence_ratio_full
                    - self.range_evidence_ratio_floor
                ),
                0.0,
                1.0,
            )
        )

    @property
    def tracker_recent_distinct_anchor_count(self) -> int:
        return len(set(self._tracker_anchor_history))

    @property
    def tracker_has_decisive_range_advantage(self) -> bool:
        """Return true only for a large prefit gain under diverse geometry.

        Ordinary branch agreement remains the primary safety check.  This
        override handles the complementary case where an uncorrected IMU bias
        makes capture drift while the joint tracker is still supported by at
        least four distinct recent anchors.  A single anchor can never enable
        it because fitting one sphere does not identify a 3-D position.
        """

        return bool(self._tracker_decisive_range_advantage)

    def _refresh_decisive_range_advantage(self) -> None:
        """Cache the range-only decision at UWB update rate."""

        if (
            self._tracker_updates_since_seed < self.min_track_updates
            or len(self._tracker_prefit_residuals) < self.min_track_updates
            or self.tracker_recent_distinct_anchor_count
            < self.minimum_decisive_anchor_count
            or self.tracker_recent_reject_fraction
            > self.max_track_reject_fraction
        ):
            self._tracker_decisive_range_advantage = False
            return
        tracker_score = self.tracker_prefit_range_rmse_m
        capture_score = self.capture_prefit_range_rmse_m
        self._tracker_decisive_range_advantage = bool(
            np.isfinite(tracker_score)
            and np.isfinite(capture_score)
            and tracker_score <= self.max_prefit_range_rmse_m
            and capture_score - tracker_score
            >= self.decisive_range_advantage_m
        )

    @staticmethod
    def _descending_health_score(
        value: float, healthy_limit: float, failed_limit: float
    ) -> float:
        """Return one at/below the healthy limit and zero at/above failure."""

        if (
            not np.isfinite(value)
            or not np.isfinite(healthy_limit)
            or not np.isfinite(failed_limit)
            or failed_limit <= healthy_limit
        ):
            return 0.0
        return float(
            np.clip(
                (failed_limit - value)
                / (failed_limit - healthy_limit),
                0.0,
                1.0,
            )
        )

    @property
    def tracker_health_score(self) -> float:
        """Continuous causal health score used only for output supervision."""

        if (
            not np.all(np.isfinite(self.tracker.X))
            or self._tracker_updates_since_seed < self.min_track_updates
            or len(self._tracker_accepts) < self.min_track_updates
            or len(self._tracker_prefit_residuals) < self.min_track_updates
        ):
            return 0.0

        reject_score = self._descending_health_score(
            self.tracker_recent_reject_fraction,
            self.max_track_reject_fraction,
            self.fallback_reject_fraction,
        )
        disagreement_score = (
            1.0
            if self.tracker_has_decisive_range_advantage
            else self._descending_health_score(
                self.disagreement_m,
                self.agreement_threshold_m,
                self.fallback_disagreement_m,
            )
        )
        tracker_score = self.tracker_prefit_range_rmse_m
        capture_score = self.capture_prefit_range_rmse_m
        absolute_range_score = self._descending_health_score(
            tracker_score,
            self.max_prefit_range_rmse_m,
            self.max_prefit_range_rmse_m
            + 3.0 * self.range_score_margin_m,
        )
        relative_range_score = self._descending_health_score(
            tracker_score - capture_score,
            self.range_score_margin_m,
            3.0 * self.range_score_margin_m,
        )
        range_evidence_score = self.tracker_range_evidence_score
        return float(
            min(
                reject_score,
                disagreement_score,
                absolute_range_score,
                relative_range_score,
                range_evidence_score,
            )
        )

    @property
    def tracker_output_weight(self) -> float:
        """Current bounded tracker contribution to the reported p/v state."""

        if self.mode != "track":
            return 0.0
        return float(self._tracker_output_weight)

    @property
    def smoothed_tracker_health(self) -> float:
        return float(self._smoothed_tracker_health)

    @property
    def full_tracker_output_authorized(self) -> bool:
        return bool(
            self.mode == "track"
            and self._full_tracker_output_authorized
        )

    @property
    def tracker_target_weight(self) -> float:
        if self.mode != "track":
            return 0.0
        if self.full_tracker_output_authorized:
            return float(self.maximum_track_output_weight)

        # A newly accepted tracker gets the requested conservative 30%
        # hand-off while its deliberately slow health low-pass catches up.
        # After that grace interval, weak persistent evidence removes even
        # the partial contribution.  The tracker continues to run and can
        # regain the 30% output causally; only the public p/v output returns
        # to the independent capture branch.
        track_age_s = self.elapsed_s - self._track_mode_started_s
        if (
            track_age_s > self.track_output_health_grace_s
            and self._smoothed_tracker_health
            < self.minimum_track_output_health
        ):
            return 0.0
        return float(self.track_output_weight)

    def _advance_smoothed_health(self, dt: float) -> None:
        """Update persistent health evidence and full-weight authorization."""

        if self.mode != "track":
            self._smoothed_tracker_health = 0.0
            self._full_weight_health_hold_s = 0.0
            self._full_tracker_output_authorized = False
            return

        smoothing = 1.0 - math.exp(
            -dt / self.tracker_health_smoothing_time_s
        )
        self._smoothed_tracker_health += smoothing * (
            self.tracker_health_score - self._smoothed_tracker_health
        )

        # Confirmed loss bypasses the low-pass delay.  Ordinary evidence
        # fluctuations use hysteresis so that a healthy tracker is not
        # repeatedly switched in and out.
        if self._tracker_is_lost():
            self._full_tracker_output_authorized = False
            self._full_weight_health_hold_s = 0.0
            return

        if self._full_tracker_output_authorized:
            if (
                self._smoothed_tracker_health
                < self.full_weight_health_exit
            ):
                self._full_tracker_output_authorized = False
                self._full_weight_health_hold_s = 0.0
            return

        if (
            self._smoothed_tracker_health
            >= self.full_weight_health_enter
        ):
            self._full_weight_health_hold_s += dt
            if (
                self._full_weight_health_hold_s
                >= self.full_weight_health_hold_s
            ):
                self._full_tracker_output_authorized = True
        else:
            self._full_weight_health_hold_s = 0.0

    def _advance_output_weight(self, dt: float) -> None:
        """Slew the causal output weight toward its health-derived target."""

        if self.mode != "track":
            self._tracker_output_weight = 0.0
            return

        target = self.tracker_target_weight
        current = self._tracker_output_weight
        if target >= current:
            rise_span = (
                self.maximum_track_output_weight
                - self.track_output_weight
            )
            if rise_span <= 1e-15:
                updated = target
            else:
                maximum_step = (
                    rise_span / self.track_weight_rise_time_s * dt
                )
                updated = min(target, current + maximum_step)
        else:
            maximum_step = (
                self.maximum_track_output_weight
                / self.track_weight_fall_time_s
                * dt
            )
            updated = max(target, current - maximum_step)

        self._tracker_output_weight = float(
            np.clip(updated, 0.0, self.maximum_track_output_weight)
        )
        if (
            self.maximum_track_output_weight
            - self._tracker_output_weight
            <= 1e-12
        ):
            self._tracker_output_weight = self.maximum_track_output_weight

    def predict(self, imu_sample: Iterable[float], dt: float) -> np.ndarray:
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        world_acceleration = self.capture.predict(imu_sample, dt)
        self.tracker.predict(np.asarray(imu_sample, dtype=np.float64), dt)
        self.elapsed_s += float(dt)
        self._advance_smoothed_health(float(dt))
        self._advance_output_weight(float(dt))
        return world_acceleration

    def _prefit_range_residuals(
        self, anchor: np.ndarray, corrected_range: float
    ) -> tuple[float, float]:
        """Compute causal residuals before either branch uses the sample."""

        capture_predicted = float(
            np.linalg.norm(anchor - self.capture.lifted.position)
        )
        tracker_tag = (
            self.tracker.X_hat[:3, 4]
            + self.tracker.X_hat[:3, :3] @ self.tracker.t_uv
        )
        tracker_predicted = float(np.linalg.norm(anchor - tracker_tag))
        return (
            corrected_range - capture_predicted,
            corrected_range - tracker_predicted,
        )

    def _capture_is_confident(self) -> bool:
        if len(self._capture_accepts) < self.min_track_updates:
            return False
        if self._reject_fraction(self._capture_accepts) > 0.20:
            return False
        covariance = self.capture.lifted.P[P_SLICE, P_SLICE]
        maximum_variance = float(np.max(np.linalg.eigvalsh(covariance)))
        return (
            np.all(np.isfinite(self.capture.lifted.position))
            and maximum_variance
            <= self.capture_position_std_threshold_m**2
        )

    def _tracker_is_ready(self) -> bool:
        # A raw-range InEKF needs time to build useful attitude/velocity/bias
        # cross-covariances.  Range-fit agreement alone becomes optimistic
        # within a few anchor cycles, so it is not sufficient for hand-off.
        if self.elapsed_s <= self.minimum_capture_dwell_s + 1e-9:
            return False
        if self._tracker_updates_since_seed < self.min_track_updates:
            return False
        if len(self._tracker_accepts) < self.min_track_updates:
            return False
        if (
            self.tracker_recent_reject_fraction
            > self.max_track_reject_fraction
        ):
            return False
        if (
            self.disagreement_m > self.agreement_threshold_m
            and not self.tracker_has_decisive_range_advantage
        ):
            return False
        tracker_score = self.tracker_prefit_range_rmse_m
        capture_score = self.capture_prefit_range_rmse_m
        if tracker_score > self.max_prefit_range_rmse_m:
            return False
        return tracker_score <= capture_score + self.range_score_margin_m

    def _tracker_is_lost(self) -> bool:
        if not np.all(np.isfinite(self.tracker.X)):
            return True
        if (
            self.disagreement_m > self.fallback_disagreement_m
            and not self.tracker_has_decisive_range_advantage
        ):
            return True
        if (
            len(self._tracker_accepts) >= self.min_track_updates
            and self.tracker_recent_reject_fraction
            > self.fallback_reject_fraction
        ):
            return True
        return (
            len(self._tracker_prefit_residuals) >= self.min_track_updates
            and self.tracker_prefit_range_rmse_m
            > self.capture_prefit_range_rmse_m
            + 3.0 * self.range_score_margin_m
        )

    def _handoff_covariance(self) -> np.ndarray:
        """Build the reseed prior in the tracker's covariance convention."""

        if self._handoff_covariance_factory is not None:
            return self._handoff_covariance_factory(
                self.capture,
                self.handoff_position_std_floor_m,
                self.handoff_velocity_std_floor_mps,
            )

        # The legacy prior inserts independent world translation blocks into
        # [theta, v, p, b_g, b_a]. Other coordinates require the factory hook.
        source = _symmetrize(self.capture.imu_core.P.copy())
        covariance = np.zeros((15, 15), dtype=np.float64)
        retained = np.r_[0:3, 9:15]
        covariance[np.ix_(retained, retained)] = source[
            np.ix_(retained, retained)
        ]

        lifted_covariance = self.capture.lifted.P
        position_covariance = _symmetrize(
            lifted_covariance[P_SLICE, P_SLICE]
        )
        velocity_covariance = _symmetrize(
            lifted_covariance[V_SLICE, V_SLICE]
        )
        velocity_position = lifted_covariance[V_SLICE, P_SLICE]

        position_covariance += (
            self.handoff_position_std_floor_m**2 * np.eye(3)
        )
        velocity_covariance += (
            self.handoff_velocity_std_floor_mps**2 * np.eye(3)
        )
        covariance[3:6, 3:6] = velocity_covariance
        covariance[6:9, 6:9] = position_covariance
        covariance[3:6, 6:9] = velocity_position
        covariance[6:9, 3:6] = velocity_position.T

        covariance = _symmetrize(covariance)
        minimum_eigenvalue = float(np.min(np.linalg.eigvalsh(covariance)))
        if minimum_eigenvalue < 1e-10:
            covariance += (1e-10 - minimum_eigenvalue) * np.eye(15)
        return covariance

    def _reseed_tracker(self) -> None:
        """Rebuild a lost tracker from the independent capture branch."""

        state = self.capture.X
        covariance = self._handoff_covariance()
        self.tracker = self._new_tracker(
            state.copy(), covariance, self.capture.bias.copy()
        )
        self._tracker_accepts.clear()
        self._tracker_prefit_residuals.clear()
        self._tracker_anchor_history.clear()
        self._tracker_decisive_range_advantage = False
        self._tracker_updates_since_seed = 0
        self._switch_ready_count = 0
        self.last_reseed_s = self.elapsed_s
        self.stats.tracker_reseeds += 1

    def _enter_track_mode(self) -> None:
        self.mode = "track"
        self._tracker_output_weight = self.track_output_weight
        self._track_mode_started_s = self.elapsed_s
        self._smoothed_tracker_health = 0.0
        self._full_weight_health_hold_s = 0.0
        self._full_tracker_output_authorized = False
        self.stats.mode_switches_to_track += 1
        self._switch_ready_count = 0
        self._fallback_count = 0

    def _enter_capture_mode(self) -> None:
        self.mode = "capture"
        self._tracker_output_weight = 0.0
        self._track_mode_started_s = -math.inf
        self._smoothed_tracker_health = 0.0
        self._full_weight_health_hold_s = 0.0
        self._full_tracker_output_authorized = False
        self.stats.fallbacks_to_capture += 1
        self._fallback_count = 0
        self._switch_ready_count = 0

    def _update_mode(self) -> None:
        if self.mode == "capture":
            if self._tracker_is_ready():
                self._switch_ready_count += 1
            else:
                self._switch_ready_count = 0
            if self._switch_ready_count >= self.switch_hold_updates:
                self._enter_track_mode()
                return

            reseed_due = (
                self.elapsed_s >= self.first_reseed_time_s
                and self.elapsed_s - self.last_reseed_s
                >= self.reseed_cooldown_s
            )
            tracker_needs_help = (
                self.disagreement_m > self.agreement_threshold_m
                or self.tracker_recent_reject_fraction
                > self.max_track_reject_fraction
            )
            if (
                reseed_due
                and tracker_needs_help
                and self._capture_is_confident()
            ):
                self._reseed_tracker()
            return

        if self._tracker_is_lost():
            self._fallback_count += 1
        else:
            self._fallback_count = 0
        if self._fallback_count >= self.fallback_hold_updates:
            self._enter_capture_mode()
            if self._capture_is_confident():
                self._reseed_tracker()

    def correct_twr(
        self,
        anchor: Iterable[float],
        measured_range: float,
        anchor_index: int | None = None,
    ) -> tuple[bool, bool]:
        """Update both branches using the same indexed range calibration."""

        anchor_vector = _as_vector3(anchor, "anchor")
        try:
            range_bias = _for_anchor(
                self._tracker_parameters["range_bias"],
                anchor_index,
                "range_bias",
            )
            # Validate both array indices before recording health evidence.
            _for_anchor(
                self._tracker_parameters["range_sigma"],
                anchor_index,
                "range_sigma",
            )
            corrected_range = float(measured_range) - range_bias
        except ValueError:
            corrected_range = math.nan
        if np.isfinite(corrected_range) and corrected_range > 0:
            capture_residual, tracker_residual = self._prefit_range_residuals(
                anchor_vector, corrected_range
            )
            self._capture_prefit_residuals.append(capture_residual)
            self._tracker_prefit_residuals.append(tracker_residual)
            self._tracker_anchor_history.append(
                tuple(float(value) for value in anchor_vector)
            )

        capture_accepted = self.capture.correct_twr(
            anchor_vector, measured_range, anchor_index=anchor_index
        )
        tracker_accepted = self.tracker.correct_twr(
            anchor_vector, measured_range, anchor_index=anchor_index
        )
        self._capture_accepts.append(bool(capture_accepted))
        self._tracker_accepts.append(bool(tracker_accepted))
        self._tracker_updates_since_seed += 1
        self._refresh_decisive_range_advantage()

        if tracker_accepted:
            self.stats.tracker_accepted += 1
        elif np.isfinite(corrected_range) and corrected_range > 0:
            self.stats.tracker_rejected += 1
        else:
            self.stats.tracker_invalid += 1

        self._update_mode()
        if self.mode == "track":
            self.stats.output_track_updates += 1
            if self.tracker_has_decisive_range_advantage:
                self.stats.decisive_range_override_updates += 1
            if (
                self.tracker_output_weight
                >= self.maximum_track_output_weight - 1e-12
            ):
                self.stats.output_full_tracker_updates += 1
            elif self.tracker_output_weight <= 1e-12:
                self.stats.output_health_guard_updates += 1
            else:
                self.stats.output_transition_updates += 1
        else:
            self.stats.output_capture_updates += 1
        return capture_accepted, tracker_accepted
