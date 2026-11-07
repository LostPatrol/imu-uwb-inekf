#!/usr/bin/env python3
"""Evaluate the guarded two-filter IMU/TWR estimator.

The primary comparison is deliberately paired: ``loose_lifted`` is the
unchanged capture branch inside the same estimator instance, while
``deep_coupled`` is the guarded output that becomes available after the
raw-range joint InEKF has accumulated useful cross-covariances.  This removes
run-to-run and input differences from the comparison.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import time
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from deep_coupled import DeepCoupledTwoFilter
from inekf import GRAVITY_MAGNITUDE
from lifted_twr import (
    DirectRangeEKF,
    TWRInEKF,
    paper_consistent_initial_covariance,
)
from twr_experiment import (
    DEFAULT_DATASET,
    TU_DELFT_COMMON_SENSOR_DELAY_S,
    _initial_state,
    load_anchors,
    load_real_log,
    real_sequence_files,
    run_synthetic_single_range,
    theodosis_trajectory,
    trajectory_metrics,
)


CORE_METRICS = (
    "position_rmse_m",
    "position_tail5_rmse_m",
    "velocity_rmse_mps",
    "velocity_tail5_rmse_mps",
)
METHOD_COLORS = {
    "ordinary_ekf": "#7f7f7f",
    "ordinary_inekf": "#9467bd",
    "loose_lifted": "#1b9e77",
    "tight_tracker": "#4c78a8",
    "deep_coupled": "#d95f02",
}
METHOD_LABELS = {
    "ordinary_ekf": "ordinary TWR EKF",
    "ordinary_inekf": "ordinary TWR-InEKF",
    "loose_lifted": "loose lifted cascade",
    "tight_tracker": "raw-TWR joint InEKF",
    "deep_coupled": "adaptive capture-to-track output",
}
METHOD_SHORT_LABELS = {
    "ordinary_ekf": "EKF",
    "ordinary_inekf": "InEKF",
    "loose_lifted": "loose",
    "tight_tracker": "tracker",
    "deep_coupled": "deep",
}
METHOD_LINE_STYLES = {
    "ordinary_ekf": {"linestyle": "--", "linewidth": 0.95},
    "ordinary_inekf": {"linestyle": ":", "linewidth": 1.0},
    "loose_lifted": {"linestyle": "-", "linewidth": 1.0},
    "tight_tracker": {"linestyle": "-.", "linewidth": 0.95},
    "deep_coupled": {"linestyle": "-", "linewidth": 1.35},
}
METHOD_ORDER = tuple(METHOD_COLORS)

# The original 20 s single-anchor Monte Carlo ended exactly when the real-data
# supervisor's mandatory capture dwell expired.  Its adaptive output therefore
# could not switch even once.  The end-to-end benchmark keeps that full 20 s
# single-anchor challenge, then exposes the estimator to a fixed eight-anchor
# room geometry for 25 s so the joint tracker can be exercised and evaluated.
SYNTHETIC_CAPTURE_DWELL_S = 20.0
SYNTHETIC_END_TO_END_DURATION_S = 45.0
SYNTHETIC_MINIMUM_TRACK_EVALUATION_S = 10.0
SYNTHETIC_TRUE_ACCEL_BIAS_MPS2 = (0.080, -0.056, 0.040)
SYNTHETIC_INITIAL_ACCEL_BIAS_STD_MPS2 = 0.20
SYNTHETIC_TRACKING_ANCHOR_OFFSETS_M = (
    (-45.0, 5.0, 0.0),
    (5.0, 5.0, 0.0),
    (-45.0, 35.0, 0.0),
    (5.0, 35.0, 0.0),
    (-45.0, 5.0, 8.0),
    (5.0, 5.0, 8.0),
    (-45.0, 35.0, 8.0),
    (5.0, 35.0, 8.0),
)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(_json_ready(value), stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def run_real_deep_sequence(
    log: Any,
    anchors: np.ndarray,
    *,
    initial_offset: Iterable[float] = (0.0, 0.0, 0.0),
) -> dict[str, Any]:
    """Run all five methods on one shared, already loaded real log."""

    offset = np.asarray(initial_offset, dtype=np.float64)
    if offset.shape != (3,):
        raise ValueError("initial_offset must contain three values")
    initial_state, covariance, bias = _initial_state(log, offset)
    estimator = DeepCoupledTwoFilter(
        initial_state,
        covariance,
        bias,
        range_sigma=0.20,
        range_bias=-0.18,
        accel_noise_std=10.0,
        gate_threshold=25.0,
        capture_gate_warmup_updates=64,
        tracker_gate_warmup_updates=128,
        track_output_weight=0.30,
        maximum_track_output_weight=1.0,
        track_weight_rise_time_s=5.0,
        track_weight_fall_time_s=1.0,
        minimum_capture_dwell_s=20.0,
    )
    ordinary_ekf = DirectRangeEKF(
        initial_state[:3, 4],
        initial_state[:3, 3],
        position_std=8.0,
        velocity_std=2.0,
        accel_noise_std=10.0,
        range_sigma=0.20,
        range_bias=-0.18,
        gate_threshold=25.0,
        gate_warmup_updates=64,
    )
    ordinary_inekf = TWRInEKF(
        initial_state.copy(),
        covariance.copy(),
        bias.copy(),
        range_sigma=0.20,
        range_bias=-0.18,
        gate_threshold=25.0,
        # Preserve the original ordinary-InEKF baseline exactly.  It has no
        # capture reseed and activates gating immediately.
        gate_warmup_updates=0,
        lever_arm=(0.0, 0.0, 0.0),
    )

    sample_count = log.time.size
    histories = {
        method: {
            "position": np.empty((sample_count, 3), dtype=np.float64),
            "velocity": np.empty((sample_count, 3), dtype=np.float64),
        }
        for method in METHOD_COLORS
    }
    diagnostic = {
        "mode_track": np.zeros(sample_count, dtype=bool),
        "disagreement_m": np.empty(sample_count, dtype=np.float64),
        "capture_prefit_range_rmse_m": np.empty(
            sample_count, dtype=np.float64
        ),
        "tracker_prefit_range_rmse_m": np.empty(
            sample_count, dtype=np.float64
        ),
        "tracker_health_score": np.empty(
            sample_count, dtype=np.float64
        ),
        "smoothed_tracker_health": np.empty(
            sample_count, dtype=np.float64
        ),
        "full_tracker_output_authorized": np.zeros(
            sample_count, dtype=bool
        ),
        "tracker_range_evidence_ratio": np.empty(
            sample_count, dtype=np.float64
        ),
        "tracker_range_evidence_score": np.empty(
            sample_count, dtype=np.float64
        ),
        "tracker_output_weight": np.empty(
            sample_count, dtype=np.float64
        ),
        "lift_constraint": np.empty((sample_count, 3), dtype=np.float64),
    }

    start_clock = time.perf_counter()
    for index in range(sample_count):
        if index > 0:
            world_acceleration = estimator.predict(
                log.imu[index - 1],
                float(log.time[index] - log.time[index - 1]),
            )
            ordinary_ekf.predict(
                world_acceleration,
                float(log.time[index] - log.time[index - 1]),
            )
            ordinary_inekf.predict(
                log.imu[index - 1],
                float(log.time[index] - log.time[index - 1]),
            )
        for anchor_index in np.flatnonzero(log.changed[index]):
            anchor = anchors[anchor_index]
            measured_range = float(log.ranges[index, anchor_index])
            estimator.correct_twr(
                anchor,
                measured_range,
            )
            ordinary_ekf.correct(anchor, measured_range)
            ordinary_inekf.correct_twr(anchor, measured_range)

        histories["ordinary_ekf"]["position"][index] = (
            ordinary_ekf.position
        )
        histories["ordinary_ekf"]["velocity"][index] = (
            ordinary_ekf.velocity
        )
        histories["ordinary_inekf"]["position"][index] = (
            ordinary_inekf.X[:3, 4]
        )
        histories["ordinary_inekf"]["velocity"][index] = (
            ordinary_inekf.X[:3, 3]
        )
        histories["loose_lifted"]["position"][
            index
        ] = estimator.capture.lifted.position
        histories["loose_lifted"]["velocity"][
            index
        ] = estimator.capture.lifted.velocity
        histories["tight_tracker"]["position"][
            index
        ] = estimator.tracker.X[:3, 4]
        histories["tight_tracker"]["velocity"][
            index
        ] = estimator.tracker.X[:3, 3]
        histories["deep_coupled"]["position"][index] = estimator.position
        histories["deep_coupled"]["velocity"][index] = estimator.velocity
        diagnostic["mode_track"][index] = estimator.mode == "track"
        diagnostic["disagreement_m"][index] = estimator.disagreement_m
        diagnostic["capture_prefit_range_rmse_m"][
            index
        ] = estimator.capture_prefit_range_rmse_m
        diagnostic["tracker_prefit_range_rmse_m"][
            index
        ] = estimator.tracker_prefit_range_rmse_m
        diagnostic["tracker_health_score"][
            index
        ] = estimator.tracker_health_score
        diagnostic["smoothed_tracker_health"][
            index
        ] = estimator.smoothed_tracker_health
        diagnostic["full_tracker_output_authorized"][
            index
        ] = estimator.full_tracker_output_authorized
        diagnostic["tracker_range_evidence_ratio"][
            index
        ] = estimator.tracker_range_evidence_ratio
        diagnostic["tracker_range_evidence_score"][
            index
        ] = estimator.tracker_range_evidence_score
        diagnostic["tracker_output_weight"][
            index
        ] = estimator.tracker_output_weight
        diagnostic["lift_constraint"][
            index
        ] = estimator.capture.lifted.constraint_residual()
    elapsed = time.perf_counter() - start_clock

    methods: dict[str, Any] = {}
    for method, history in histories.items():
        methods[method] = trajectory_metrics(
            log.time,
            history["position"],
            history["velocity"],
            log.ground_truth_position,
            log.ground_truth_velocity,
        )
    methods["loose_lifted"]["accepted"] = int(
        estimator.capture.lifted.stats.accepted
    )
    methods["loose_lifted"]["rejected"] = int(
        estimator.capture.lifted.stats.rejected
    )
    methods["tight_tracker"]["accepted"] = int(
        estimator.stats.tracker_accepted
    )
    methods["tight_tracker"]["rejected"] = int(
        estimator.stats.tracker_rejected
    )
    methods["deep_coupled"]["accepted"] = int(
        estimator.stats.tracker_accepted
    )
    methods["deep_coupled"]["rejected"] = int(
        estimator.stats.tracker_rejected
    )
    methods["ordinary_ekf"]["accepted"] = int(
        ordinary_ekf.stats.accepted
    )
    methods["ordinary_ekf"]["rejected"] = int(
        ordinary_ekf.stats.rejected
    )
    methods["ordinary_inekf"]["accepted"] = int(
        ordinary_inekf.stats.accepted
    )
    methods["ordinary_inekf"]["rejected"] = int(
        ordinary_inekf.stats.rejected
    )

    return {
        "metrics": {
            "sequence": log.name,
            "duration_s": float(log.time[-1]),
            "samples": int(sample_count),
            "twr_events": int(np.count_nonzero(log.changed)),
            "sensor_delay_s": float(log.sensor_delay_s),
            "initial_offset_m": offset.tolist(),
            "runtime_s": float(elapsed),
            "methods": methods,
            "coupling": {
                **asdict(estimator.stats),
                "final_mode": estimator.mode,
                "track_sample_fraction": float(
                    np.mean(diagnostic["mode_track"])
                ),
                "minimum_capture_dwell_s": estimator.minimum_capture_dwell_s,
                "initial_track_output_weight": (
                    estimator.track_output_weight
                ),
                "maximum_track_output_weight": (
                    estimator.maximum_track_output_weight
                ),
                "track_weight_rise_time_s": (
                    estimator.track_weight_rise_time_s
                ),
                "track_weight_fall_time_s": (
                    estimator.track_weight_fall_time_s
                ),
                "tracker_health_smoothing_time_s": (
                    estimator.tracker_health_smoothing_time_s
                ),
                "full_weight_health_enter": (
                    estimator.full_weight_health_enter
                ),
                "full_weight_health_exit": (
                    estimator.full_weight_health_exit
                ),
                "full_weight_health_hold_s": (
                    estimator.full_weight_health_hold_s
                ),
                "minimum_track_output_health": (
                    estimator.minimum_track_output_health
                ),
                "track_output_health_grace_s": (
                    estimator.track_output_health_grace_s
                ),
                "full_weight_agreement_m": (
                    estimator.full_weight_agreement_m
                ),
                "range_evidence_ratio_floor": (
                    estimator.range_evidence_ratio_floor
                ),
                "range_evidence_ratio_full": (
                    estimator.range_evidence_ratio_full
                ),
                "decisive_range_advantage_m": (
                    estimator.decisive_range_advantage_m
                ),
                "minimum_decisive_anchor_count": (
                    estimator.minimum_decisive_anchor_count
                ),
                "final_tracker_output_weight": (
                    estimator.tracker_output_weight
                ),
                "final_tracker_health_score": (
                    estimator.tracker_health_score
                ),
                "final_smoothed_tracker_health": (
                    estimator.smoothed_tracker_health
                ),
                "final_full_tracker_output_authorized": (
                    estimator.full_tracker_output_authorized
                ),
                "full_tracker_authorized_sample_fraction": float(
                    np.mean(
                        diagnostic["full_tracker_output_authorized"]
                    )
                ),
                "final_tracker_range_evidence_ratio": (
                    estimator.tracker_range_evidence_ratio
                ),
                "mean_tracker_output_weight": float(
                    np.mean(diagnostic["tracker_output_weight"])
                ),
                "full_tracker_sample_fraction": float(
                    np.mean(
                        diagnostic["tracker_output_weight"]
                        >= estimator.maximum_track_output_weight - 1e-12
                    )
                ),
                "final_disagreement_m": estimator.disagreement_m,
                "final_capture_prefit_range_rmse_m": (
                    estimator.capture_prefit_range_rmse_m
                ),
                "final_tracker_prefit_range_rmse_m": (
                    estimator.tracker_prefit_range_rmse_m
                ),
            },
        },
        "time": log.time,
        "ground_truth_position": log.ground_truth_position,
        "ground_truth_velocity": log.ground_truth_velocity,
        "histories": histories,
        "diagnostic": diagnostic,
    }


def save_sequence_result(result: dict[str, Any], output_directory: Path) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    _write_json(output_directory / "metrics.json", result["metrics"])
    arrays: dict[str, np.ndarray] = {
        "time": result["time"],
        "ground_truth_position": result["ground_truth_position"],
        "ground_truth_velocity": result["ground_truth_velocity"],
    }
    for method, history in result["histories"].items():
        for name, values in history.items():
            arrays[f"{method}_{name}"] = values
    arrays.update(result["diagnostic"])
    np.savez_compressed(output_directory / "timeseries.npz", **arrays)


def plot_real_result(result: dict[str, Any], output_path: Path) -> None:
    truth = result["ground_truth_position"]
    truth_velocity = result["ground_truth_velocity"]
    time_values = result["time"]
    histories = result["histories"]
    diagnostic = result["diagnostic"]

    figure, axes = plt.subplots(
        2, 2, figsize=(13.0, 9.0), constrained_layout=True
    )
    trajectory_axis, position_axis, velocity_axis, supervisor_axis = axes.ravel()
    trajectory_axis.plot(
        truth[:, 0], truth[:, 1], "k-", lw=2.0, label="OptiTrack"
    )
    for method, history in histories.items():
        trajectory_axis.plot(
            history["position"][:, 0],
            history["position"][:, 1],
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
            **METHOD_LINE_STYLES[method],
        )
    trajectory_axis.set_title("XY trajectory")
    trajectory_axis.set_xlabel("x [m]")
    trajectory_axis.set_ylabel("y [m]")
    trajectory_axis.axis("equal")
    trajectory_axis.grid(alpha=0.25)
    trajectory_axis.legend(fontsize=8)

    for method, history in histories.items():
        error = np.linalg.norm(history["position"] - truth, axis=1)
        tail_rmse = result["metrics"]["methods"][method][
            "position_tail5_rmse_m"
        ]
        position_axis.plot(
            time_values,
            error,
            color=METHOD_COLORS[method],
            label=f"{METHOD_LABELS[method]} (tail {tail_rmse:.3f} m)",
            **METHOD_LINE_STYLES[method],
        )
    position_axis.set_title("Position error")
    position_axis.set_xlabel("time [s]")
    position_axis.set_ylabel("error norm [m]")
    position_axis.grid(alpha=0.25)
    position_axis.legend(fontsize=8)

    for method, history in histories.items():
        error = np.linalg.norm(history["velocity"] - truth_velocity, axis=1)
        velocity_axis.plot(
            time_values,
            error,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
            **METHOD_LINE_STYLES[method],
        )
    velocity_axis.set_title("Velocity error (OptiTrack derivative)")
    velocity_axis.set_xlabel("time [s]")
    velocity_axis.set_ylabel("error norm [m/s]")
    velocity_axis.grid(alpha=0.25)
    velocity_axis.legend(fontsize=8)

    supervisor_axis.plot(
        time_values,
        diagnostic["disagreement_m"],
        color="#4c78a8",
        lw=1.1,
        label="branch disagreement",
    )
    supervisor_axis.plot(
        time_values,
        diagnostic["capture_prefit_range_rmse_m"],
        color=METHOD_COLORS["loose_lifted"],
        lw=0.9,
        label="loose range score",
    )
    supervisor_axis.plot(
        time_values,
        diagnostic["tracker_prefit_range_rmse_m"],
        color=METHOD_COLORS["tight_tracker"],
        lw=0.9,
        label="tracker range score",
    )
    finite_values = np.concatenate(
        (
            diagnostic["disagreement_m"],
            diagnostic["capture_prefit_range_rmse_m"],
            diagnostic["tracker_prefit_range_rmse_m"],
        )
    )
    finite_values = finite_values[np.isfinite(finite_values)]
    upper = max(1.0, float(np.percentile(finite_values, 99)) * 1.2)
    supervisor_axis.fill_between(
        time_values,
        0.0,
        upper,
        where=diagnostic["mode_track"],
        color="#f2cf5b",
        alpha=0.18,
        label="tracking output active",
    )
    weight_axis = supervisor_axis.twinx()
    weight_axis.plot(
        time_values,
        diagnostic["smoothed_tracker_health"],
        color="#666666",
        lw=1.0,
        ls="-.",
        label="persistent tracker health",
    )
    weight_axis.plot(
        time_values,
        diagnostic["tracker_output_weight"],
        color=METHOD_COLORS["deep_coupled"],
        lw=1.2,
        ls="--",
        label="tracker output weight",
    )
    weight_axis.set_ylim(-0.02, 1.02)
    weight_axis.set_ylabel("health / tracker weight")
    supervisor_axis.set_ylim(0.0, upper)
    supervisor_axis.set_title("Causal supervisor diagnostics")
    supervisor_axis.set_xlabel("time [s]")
    supervisor_axis.set_ylabel("distance / range RMSE [m]")
    supervisor_axis.grid(alpha=0.25)
    supervisor_handles, supervisor_labels = (
        supervisor_axis.get_legend_handles_labels()
    )
    weight_handles, weight_labels = weight_axis.get_legend_handles_labels()
    supervisor_axis.legend(
        supervisor_handles + weight_handles,
        supervisor_labels + weight_labels,
        fontsize=7,
        loc="upper right",
    )

    metrics = result["metrics"]
    figure.suptitle(
        f"Deep-coupled two-filter IMU/TWR | {metrics['sequence']} | "
        f"offset={metrics['initial_offset_m']} m | "
        f"common sensor delay={metrics['sensor_delay_s']:.2f} s",
        fontsize=12,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _paired_summary(table: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "runs": int(table.shape[0]),
        "comparison": "deep_coupled minus loose_lifted; negative is better",
        "metrics": {},
    }
    all_non_regressing = True
    for metric in CORE_METRICS:
        delta = table[f"delta_{metric}"].to_numpy(dtype=np.float64)
        tolerance = 1e-9
        metric_summary = {
            "loose_median": float(table[f"loose_{metric}"].median()),
            "deep_median": float(table[f"deep_{metric}"].median()),
            "median_delta": float(np.median(delta)),
            "mean_delta": float(np.mean(delta)),
            "better_count": int(np.count_nonzero(delta < -tolerance)),
            "equal_count": int(np.count_nonzero(np.abs(delta) <= tolerance)),
            "worse_count": int(np.count_nonzero(delta > tolerance)),
            "worst_delta": float(np.max(delta)),
        }
        metric_summary["non_regression_pass"] = (
            metric_summary["worse_count"] == 0
        )
        all_non_regressing &= metric_summary["non_regression_pass"]
        summary["metrics"][metric] = metric_summary
    summary["all_four_metrics_non_regression_pass"] = bool(
        all_non_regressing
    )
    return summary


def _all_method_summary(
    table: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Summarize every baseline without hiding ordinary-InEKF failures."""

    summary: dict[str, Any] = {
        "baseline_definitions": {
            "ordinary_ekf": (
                "six-state nonlinear raw-range EKF; same world acceleration "
                "as the lifted capture branch; 64-update gate warmup"
            ),
            "ordinary_inekf": (
                "original raw-range TWR-InEKF; no reseed and immediate NIS "
                "gating, matching the task-06 ordinary baseline"
            ),
            "loose_lifted": "unchanged task-06 lifted loose cascade",
            "tight_tracker": (
                "raw-range joint InEKF inside the new estimator, including "
                "capture-based discrete reseeds"
            ),
            "deep_coupled": (
                "guarded output with a health-driven smooth hand-off from "
                "30% to 100% raw-range joint InEKF"
            ),
        },
        "by_offset": {},
        "deep_pairwise": {},
    }
    rows: list[dict[str, Any]] = []
    groups: list[tuple[str, pd.DataFrame]] = [
        ("all", table),
        *[
            (str(offset_name), group)
            for offset_name, group in table.groupby(
                "offset_name", sort=True
            )
        ],
    ]
    for offset_name, group in groups:
        offset_summary: dict[str, Any] = {}
        for method in METHOD_ORDER:
            method_summary: dict[str, Any] = {
                "runs": int(group.shape[0]),
                "finite_count": int(
                    group[f"{method}_finite"].astype(bool).sum()
                ),
                "converged_0p5m_count": int(
                    group[f"{method}_converged_0p5m"].astype(bool).sum()
                ),
                "accepted_median": float(
                    group[f"{method}_accepted"].median()
                ),
                "accepted_minimum": int(
                    group[f"{method}_accepted"].min()
                ),
                "rejected_median": float(
                    group[f"{method}_rejected"].median()
                ),
                "rejected_maximum": int(
                    group[f"{method}_rejected"].max()
                ),
                "metrics": {},
            }
            row: dict[str, Any] = {
                "offset_name": offset_name,
                "method": method,
                "runs": int(group.shape[0]),
                "finite_count": method_summary["finite_count"],
                "converged_0p5m_count": method_summary[
                    "converged_0p5m_count"
                ],
                "accepted_median": method_summary["accepted_median"],
                "accepted_minimum": method_summary["accepted_minimum"],
                "rejected_median": method_summary["rejected_median"],
                "rejected_maximum": method_summary["rejected_maximum"],
            }
            for metric in CORE_METRICS:
                values = group[f"{method}_{metric}"].to_numpy(
                    dtype=np.float64
                )
                metric_summary = {
                    "median": float(np.median(values)),
                    "mean": float(np.mean(values)),
                    "maximum": float(np.max(values)),
                }
                method_summary["metrics"][metric] = metric_summary
                row[f"{metric}_median"] = metric_summary["median"]
                row[f"{metric}_mean"] = metric_summary["mean"]
                row[f"{metric}_maximum"] = metric_summary["maximum"]
            offset_summary[method] = method_summary
            rows.append(row)
        summary["by_offset"][offset_name] = offset_summary

    for baseline in METHOD_ORDER:
        if baseline == "deep_coupled":
            continue
        comparison: dict[str, Any] = {}
        for metric in CORE_METRICS:
            delta = (
                table[f"deep_coupled_{metric}"].to_numpy(dtype=np.float64)
                - table[f"{baseline}_{metric}"].to_numpy(dtype=np.float64)
            )
            tolerance = 1e-9
            comparison[metric] = {
                "median_delta": float(np.median(delta)),
                "mean_delta": float(np.mean(delta)),
                "better_count": int(np.count_nonzero(delta < -tolerance)),
                "equal_count": int(
                    np.count_nonzero(np.abs(delta) <= tolerance)
                ),
                "worse_count": int(np.count_nonzero(delta > tolerance)),
            }
        summary["deep_pairwise"][baseline] = comparison
    return summary, pd.DataFrame(rows)


def plot_batch_deltas(table: pd.DataFrame, output_path: Path) -> None:
    figure, axes = plt.subplots(
        2, 2, figsize=(12.5, 8.0), constrained_layout=True
    )
    titles = (
        "Full position RMSE",
        "Last-5-s position RMSE",
        "Full velocity RMSE",
        "Last-5-s velocity RMSE",
    )
    units = ("m", "m", "m/s", "m/s")
    labels = (
        "position_rmse_m",
        "position_tail5_rmse_m",
        "velocity_rmse_mps",
        "velocity_tail5_rmse_mps",
    )
    run_labels = [
        f"{Path(sequence).stem.replace('2021-02-12_', '')}:{offset[0]}"
        for sequence, offset in zip(table["sequence"], table["offset_name"])
    ]
    for axis, metric, title, unit in zip(axes.ravel(), labels, titles, units):
        delta = table[f"delta_{metric}"].to_numpy()
        colors = np.where(delta <= 1e-9, "#1b9e77", "#d62728")
        axis.bar(np.arange(delta.size), delta, color=colors, width=0.82)
        axis.axhline(0.0, color="0.2", lw=0.8)
        axis.set_title(title)
        axis.set_ylabel(f"deep - loose [{unit}]")
        axis.grid(axis="y", alpha=0.25)
        axis.set_xticks(np.arange(delta.size))
        axis.set_xticklabels(run_labels, rotation=72, ha="right", fontsize=6)
    figure.suptitle(
        "Paired real-data deltas (green/non-positive means no regression)",
        fontsize=12,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_all_method_comparison(
    table: pd.DataFrame, output_path: Path
) -> None:
    """Show every run on a log axis so catastrophic failures remain visible."""

    figure, axes = plt.subplots(
        2, 2, figsize=(12.5, 8.0), constrained_layout=True
    )
    titles = (
        "Full position RMSE",
        "Last-5-s position RMSE",
        "Full velocity RMSE",
        "Last-5-s velocity RMSE",
    )
    units = ("m", "m", "m/s", "m/s")
    markers = {"nominal": "o", "large": "^"}
    offset_shift = {"nominal": -0.12, "large": 0.12}
    for axis, metric, title, unit in zip(
        axes.ravel(), CORE_METRICS, titles, units
    ):
        for method_index, method in enumerate(METHOD_ORDER):
            for offset_name in ("nominal", "large"):
                values = table.loc[
                    table["offset_name"] == offset_name,
                    f"{method}_{metric}",
                ].to_numpy(dtype=np.float64)
                jitter = np.linspace(-0.045, 0.045, values.size)
                x_values = (
                    method_index + offset_shift[offset_name] + jitter
                )
                axis.scatter(
                    x_values,
                    values,
                    marker=markers[offset_name],
                    s=20,
                    color=METHOD_COLORS[method],
                    edgecolor="white",
                    linewidth=0.25,
                    alpha=0.78,
                    zorder=2,
                )
                median = float(np.median(values))
                center = method_index + offset_shift[offset_name]
                axis.plot(
                    [center - 0.08, center + 0.08],
                    [median, median],
                    color="black",
                    lw=1.2,
                    zorder=3,
                )
        axis.set_yscale("log")
        axis.set_title(title)
        axis.set_ylabel(f"RMSE [{unit}] (log scale)")
        axis.set_xticks(np.arange(len(METHOD_ORDER)))
        axis.set_xticklabels(
            [METHOD_SHORT_LABELS[method] for method in METHOD_ORDER],
            rotation=18,
            ha="right",
        )
        axis.grid(axis="y", which="both", alpha=0.22)

    axes[0, 0].legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker=markers["nominal"],
                color="none",
                markerfacecolor="0.35",
                markeredgecolor="white",
                markersize=6,
                label="nominal offset",
            ),
            Line2D(
                [0],
                [0],
                marker=markers["large"],
                color="none",
                markerfacecolor="0.35",
                markeredgecolor="white",
                markersize=6,
                label="large offset",
            ),
            Line2D(
                [0], [0], color="black", lw=1.2, label="offset median"
            ),
        ],
        fontsize=8,
    )
    figure.suptitle(
        "All five methods and all 24 real runs "
        "(points are individual runs)",
        fontsize=12,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def run_real_batch(
    dataset: Path,
    output_directory: Path,
    *,
    sensor_delay_s: float = TU_DELFT_COMMON_SENSOR_DELAY_S,
) -> dict[str, Any]:
    anchors = load_anchors(dataset)
    offsets = {
        "nominal": np.zeros(3),
        "large": np.array([6.0, -6.0, 3.0]),
    }
    rows: list[dict[str, Any]] = []
    runtime_start = time.perf_counter()
    for sequence in real_sequence_files(dataset):
        log = load_real_log(
            dataset,
            sequence,
            sensor_delay_s=sensor_delay_s,
        )
        for offset_name, offset in offsets.items():
            result = run_real_deep_sequence(
                log, anchors, initial_offset=offset
            )
            loose = result["metrics"]["methods"]["loose_lifted"]
            deep = result["metrics"]["methods"]["deep_coupled"]
            tracker = result["metrics"]["methods"]["tight_tracker"]
            row: dict[str, Any] = {
                "sequence": sequence,
                "offset_name": offset_name,
                "runtime_s": result["metrics"]["runtime_s"],
                **{
                    f"coupling_{key}": value
                    for key, value in result["metrics"]["coupling"].items()
                    if isinstance(value, (int, float, bool))
                },
            }
            for method in METHOD_ORDER:
                method_metrics = result["metrics"]["methods"][method]
                row[f"{method}_finite"] = bool(method_metrics["finite"])
                row[f"{method}_converged_0p5m"] = bool(
                    np.isfinite(method_metrics["convergence_0p5m_s"])
                )
                row[f"{method}_accepted"] = int(
                    method_metrics["accepted"]
                )
                row[f"{method}_rejected"] = int(
                    method_metrics["rejected"]
                )
                for metric in CORE_METRICS:
                    row[f"{method}_{metric}"] = method_metrics[metric]

            # Backward-compatible aliases retained for existing task reports.
            for metric in CORE_METRICS:
                row[f"loose_{metric}"] = loose[metric]
                row[f"tracker_{metric}"] = tracker[metric]
                row[f"deep_{metric}"] = deep[metric]
                row[f"delta_{metric}"] = deep[metric] - loose[metric]
            rows.append(row)

            if (
                sequence == "2021-02-12_09_56_09_twr_random.csv"
                and offset_name == "large"
            ):
                representative = output_directory / "representative"
                save_sequence_result(result, representative)
                plot_real_result(
                    result, representative / "trajectory_and_errors.png"
                )

    table = pd.DataFrame(rows)
    output_directory.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_directory / "paired_metrics.csv", index=False)
    summary = _paired_summary(table)
    all_methods, method_summary_table = _all_method_summary(table)
    summary["all_methods"] = all_methods
    summary["dataset"] = str(dataset)
    summary["time_alignment"] = {
        "model": "common IMU+TWR acquisition-to-log delay",
        "sensor_delay_s": float(sensor_delay_s),
        "uwb_only_shift": False,
    }
    summary["sequences"] = len(real_sequence_files(dataset))
    summary["offsets"] = {
        name: value.tolist() for name, value in offsets.items()
    }
    summary["wall_runtime_s"] = time.perf_counter() - runtime_start
    _write_json(output_directory / "summary.json", summary)
    method_summary_table.to_csv(
        output_directory / "all_methods_summary.csv", index=False
    )
    plot_batch_deltas(table, output_directory / "paired_deltas.png")
    plot_all_method_comparison(
        table, output_directory / "all_methods_comparison.png"
    )
    return summary


def run_synthetic_deep(
    *,
    seed: int,
    initial_position: Iterable[float],
    initial_velocity: Iterable[float] = (0.0, 0.0, 0.0),
    duration: float = SYNTHETIC_END_TO_END_DURATION_S,
    dt: float = 0.01,
    range_rate_hz: float = 20.0,
    range_sigma: float = 0.10,
    true_accel_bias_mps2: Iterable[float] = (
        SYNTHETIC_TRUE_ACCEL_BIAS_MPS2
    ),
    initial_accel_bias_std_mps2: float = (
        SYNTHETIC_INITIAL_ACCEL_BIAS_STD_MPS2
    ),
) -> dict[str, Any]:
    """Add the two-filter estimator to the task-06 single-range experiment."""

    true_accel_bias = np.asarray(
        tuple(true_accel_bias_mps2), dtype=np.float64
    )
    accel_bias_std = float(initial_accel_bias_std_mps2)
    if true_accel_bias.shape != (3,) or not np.all(
        np.isfinite(true_accel_bias)
    ):
        raise ValueError("true_accel_bias_mps2 must contain three finite values")
    if not np.isfinite(accel_bias_std) or accel_bias_std <= 0.0:
        raise ValueError(
            "initial_accel_bias_std_mps2 must be finite and positive"
        )

    base = run_synthetic_single_range(
        seed=seed,
        initial_position=initial_position,
        initial_velocity=initial_velocity,
        duration=duration,
        dt=dt,
        range_rate_hz=range_rate_hz,
        range_sigma=range_sigma,
        tracking_anchor_offsets=SYNTHETIC_TRACKING_ANCHOR_OFFSETS_M,
        tracking_anchor_start_s=SYNTHETIC_CAPTURE_DWELL_S,
        true_accel_bias_mps2=true_accel_bias,
        inekf_accel_bias_std_mps2=accel_bias_std,
    )
    time_values = base["time"]
    truth_position = base["ground_truth_position"]
    truth_velocity = base["ground_truth_velocity"]
    _, _, truth_acceleration = theodosis_trajectory(time_values)
    coordinate_offset = np.asarray(
        base["metrics"]["coordinate_offset_m"], dtype=np.float64
    )
    initial_position_vector = (
        np.asarray(initial_position, dtype=np.float64) + coordinate_offset
    )
    initial_velocity_vector = np.asarray(
        initial_velocity, dtype=np.float64
    )

    state = np.eye(5)
    state[:3, 3] = initial_velocity_vector
    state[:3, 4] = initial_position_vector
    covariance = paper_consistent_initial_covariance(
        attitude_std=(1e-3, 1e-3, 1e-3),
        velocity_std=30.0,
        position_std=60.0,
        gyro_bias_std=1e-6,
        accel_bias_std=accel_bias_std,
    )
    estimator = DeepCoupledTwoFilter(
        state,
        covariance,
        np.zeros(6),
        position_std=60.0,
        velocity_std=30.0,
        q_std=3000.0,
        s_std=3000.0,
        h_std=3000.0,
        accel_noise_std=0.25,
        range_sigma=range_sigma,
        range_bias=0.0,
        gate_threshold=25.0,
        capture_gate_warmup_updates=80,
        tracker_gate_warmup_updates=128,
        agreement_threshold_m=2.0,
        fallback_disagreement_m=8.0,
        max_prefit_range_rmse_m=1.0,
        capture_position_std_threshold_m=8.0,
        minimum_capture_dwell_s=SYNTHETIC_CAPTURE_DWELL_S,
        track_output_weight=0.30,
        maximum_track_output_weight=1.0,
        track_weight_rise_time_s=5.0,
        track_weight_fall_time_s=1.0,
    )
    for filter_instance in (estimator.capture.imu_core, estimator.tracker):
        filter_instance.w_gyr[:] = 1e-8
        filter_instance.w_acc[:] = 0.25**2
        filter_instance.w_gyr_b[:] = 1e-12
        filter_instance.w_acc_b[:] = 1e-12

    generator = np.random.default_rng(seed)
    update_stride = max(1, int(round(1.0 / (range_rate_hz * dt))))
    capture_anchor = coordinate_offset.copy()
    tracking_anchor_offsets = np.asarray(
        SYNTHETIC_TRACKING_ANCHOR_OFFSETS_M, dtype=np.float64
    )
    tracking_anchor_event = 0
    deep_position = np.empty_like(truth_position)
    deep_velocity = np.empty_like(truth_velocity)
    capture_position = np.empty_like(truth_position)
    capture_velocity = np.empty_like(truth_velocity)
    tracker_position = np.empty_like(truth_position)
    tracker_velocity = np.empty_like(truth_velocity)
    mode = np.zeros(time_values.size, dtype=bool)
    tracker_health = np.zeros(time_values.size, dtype=np.float64)
    smoothed_tracker_health = np.zeros(
        time_values.size, dtype=np.float64
    )
    full_tracker_output_authorized = np.zeros(
        time_values.size, dtype=bool
    )
    tracker_range_evidence_ratio = np.zeros(
        time_values.size, dtype=np.float64
    )
    tracker_range_evidence_score = np.zeros(
        time_values.size, dtype=np.float64
    )
    branch_position_disagreement = np.zeros(
        time_values.size, dtype=np.float64
    )
    capture_prefit_range_rmse = np.zeros(
        time_values.size, dtype=np.float64
    )
    tracker_prefit_range_rmse = np.zeros(
        time_values.size, dtype=np.float64
    )
    decisive_range_advantage = np.zeros(time_values.size, dtype=bool)
    tracker_weight = np.zeros(time_values.size, dtype=np.float64)
    for index in range(time_values.size):
        if index > 0:
            noisy_acceleration = (
                truth_acceleration[index - 1]
                + generator.normal(0.0, 0.10, size=3)
            )
            measured_acceleration = (
                noisy_acceleration + true_accel_bias
            )
            imu_sample = np.concatenate(
                (
                    (
                        measured_acceleration
                        + np.array([0.0, 0.0, GRAVITY_MAGNITUDE])
                    )
                    / GRAVITY_MAGNITUDE,
                    np.zeros(3),
                )
            )
            estimator.predict(imu_sample, dt)
        if index % update_stride == 0:
            if (
                time_values[index] + 1e-12
                >= SYNTHETIC_CAPTURE_DWELL_S
            ):
                anchor = (
                    coordinate_offset
                    + tracking_anchor_offsets[
                        tracking_anchor_event
                        % tracking_anchor_offsets.shape[0]
                    ]
                )
                tracking_anchor_event += 1
            else:
                anchor = capture_anchor
            measured_range = (
                np.linalg.norm(truth_position[index] - anchor)
                + generator.normal(0.0, range_sigma)
            )
            estimator.correct_twr(anchor, measured_range)
        deep_position[index] = estimator.position
        deep_velocity[index] = estimator.velocity
        capture_position[index] = estimator.capture.lifted.position
        capture_velocity[index] = estimator.capture.lifted.velocity
        tracker_position[index] = estimator.tracker.X[:3, 4]
        tracker_velocity[index] = estimator.tracker.X[:3, 3]
        mode[index] = estimator.mode == "track"
        tracker_health[index] = estimator.tracker_health_score
        smoothed_tracker_health[
            index
        ] = estimator.smoothed_tracker_health
        full_tracker_output_authorized[
            index
        ] = estimator.full_tracker_output_authorized
        tracker_range_evidence_ratio[
            index
        ] = estimator.tracker_range_evidence_ratio
        tracker_range_evidence_score[
            index
        ] = estimator.tracker_range_evidence_score
        branch_position_disagreement[index] = estimator.disagreement_m
        capture_prefit_range_rmse[
            index
        ] = estimator.capture_prefit_range_rmse_m
        tracker_prefit_range_rmse[
            index
        ] = estimator.tracker_prefit_range_rmse_m
        decisive_range_advantage[
            index
        ] = estimator.tracker_has_decisive_range_advantage
        tracker_weight[index] = estimator.tracker_output_weight

    base["histories"]["ordinary_ekf"] = base["histories"].pop(
        "direct_twr"
    )
    base["metrics"]["methods"]["ordinary_ekf"] = base["metrics"][
        "methods"
    ].pop("direct_twr")
    base["histories"].pop("lifted_twr")
    base["metrics"]["methods"].pop("lifted_twr")
    base["histories"]["loose_lifted"] = {
        "position": capture_position,
        "velocity": capture_velocity,
    }
    base["histories"]["ordinary_inekf"] = base["histories"].pop("twr_inekf")
    base["metrics"]["methods"]["ordinary_inekf"] = base["metrics"][
        "methods"
    ].pop("twr_inekf")
    base["histories"]["tight_tracker"] = {
        "position": tracker_position,
        "velocity": tracker_velocity,
    }
    base["histories"]["deep_coupled"] = {
        "position": deep_position,
        "velocity": deep_velocity,
    }
    for method in ("loose_lifted", "tight_tracker", "deep_coupled"):
        history = base["histories"][method]
        method_metrics = trajectory_metrics(
            time_values,
            history["position"],
            history["velocity"],
            truth_position,
            truth_velocity,
        )
        method_metrics["success_tail5_below_1m"] = (
            method_metrics["position_tail5_rmse_m"] < 1.0
        )
        base["metrics"]["methods"][method] = method_metrics
    base["metrics"]["coupling"] = {
        **asdict(estimator.stats),
        "track_sample_fraction": float(np.mean(mode)),
        "initial_track_output_weight": estimator.track_output_weight,
        "maximum_track_output_weight": (
            estimator.maximum_track_output_weight
        ),
        "decisive_range_advantage_m": (
            estimator.decisive_range_advantage_m
        ),
        "minimum_decisive_anchor_count": (
            estimator.minimum_decisive_anchor_count
        ),
        "final_tracker_output_weight": estimator.tracker_output_weight,
        "final_tracker_health_score": estimator.tracker_health_score,
        "final_smoothed_tracker_health": (
            estimator.smoothed_tracker_health
        ),
        "final_full_tracker_output_authorized": (
            estimator.full_tracker_output_authorized
        ),
        "full_tracker_authorized_sample_fraction": float(
            np.mean(full_tracker_output_authorized)
        ),
        "final_tracker_range_evidence_ratio": (
            estimator.tracker_range_evidence_ratio
        ),
        "mean_tracker_output_weight": float(np.mean(tracker_weight)),
        "full_tracker_sample_fraction": float(
            np.mean(
                tracker_weight
                >= estimator.maximum_track_output_weight - 1e-12
            )
        ),
        "maximum_observed_tracker_output_weight": float(
            np.max(tracker_weight)
        ),
        "maximum_deep_capture_position_difference_m": float(
            np.max(np.linalg.norm(deep_position - capture_position, axis=1))
        ),
        "maximum_deep_capture_velocity_difference_mps": float(
            np.max(np.linalg.norm(deep_velocity - capture_velocity, axis=1))
        ),
        "tail5_maximum_deep_capture_position_difference_m": float(
            np.max(
                np.linalg.norm(
                    deep_position[time_values >= time_values[-1] - 5.0]
                    - capture_position[
                        time_values >= time_values[-1] - 5.0
                    ],
                    axis=1,
                )
            )
        ),
        "first_track_time_s": (
            float(time_values[np.flatnonzero(mode)[0]])
            if np.any(mode)
            else None
        ),
        "final_mode": estimator.mode,
    }
    base["diagnostic"] = {
        "mode_track": mode,
        "tracker_health_score": tracker_health,
        "smoothed_tracker_health": smoothed_tracker_health,
        "full_tracker_output_authorized": (
            full_tracker_output_authorized
        ),
        "tracker_range_evidence_ratio": tracker_range_evidence_ratio,
        "tracker_range_evidence_score": tracker_range_evidence_score,
        "branch_position_disagreement_m": branch_position_disagreement,
        "capture_prefit_range_rmse_m": capture_prefit_range_rmse,
        "tracker_prefit_range_rmse_m": tracker_prefit_range_rmse,
        "decisive_range_advantage": decisive_range_advantage,
        "tracker_output_weight": tracker_weight,
    }
    return base


def plot_synthetic_deep(result: dict[str, Any], output_path: Path) -> None:
    time_values = result["time"]
    truth = result["ground_truth_position"]
    histories = result["histories"]
    methods = (
        "ordinary_ekf",
        "ordinary_inekf",
        "loose_lifted",
        "tight_tracker",
        "deep_coupled",
    )
    diagnostic = result["diagnostic"]
    figure, axes = plt.subplots(
        1, 3, figsize=(16.0, 4.8), constrained_layout=True
    )
    axes[0].plot(truth[:, 0], truth[:, 1], "k-", lw=2.0, label="truth")
    for method in ("loose_lifted", "tight_tracker", "deep_coupled"):
        axes[0].plot(
            histories[method]["position"][:, 0],
            histories[method]["position"][:, 1],
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
            **METHOD_LINE_STYLES[method],
        )
    axes[0].scatter(
        [0.0], [0.0], marker="^", color="black", s=45, label="capture anchor"
    )
    anchor_xy = np.unique(
        np.asarray(SYNTHETIC_TRACKING_ANCHOR_OFFSETS_M)[:, :2], axis=0
    )
    axes[0].scatter(
        anchor_xy[:, 0],
        anchor_xy[:, 1],
        marker="s",
        facecolors="none",
        edgecolors="#555555",
        s=38,
        label="tracking anchors (XY)",
    )
    local_xy = np.vstack((truth[:, :2], anchor_xy, np.array([[0.0, 0.0]])))
    local_min = np.min(local_xy, axis=0) - 5.0
    local_max = np.max(local_xy, axis=0) + 5.0
    axes[0].set_xlim(local_min[0], local_max[0])
    axes[0].set_ylim(local_min[1], local_max[1])
    axes[0].set_title("Local XY; failed baselines remain in error panel")
    axes[0].set_xlabel("x [m]")
    axes[0].set_ylabel("y [m]")
    axes[0].axis("equal")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)

    for method in methods:
        error = np.linalg.norm(
            histories[method]["position"] - truth, axis=1
        )
        axes[1].plot(
            time_values,
            error,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
            **METHOD_LINE_STYLES[method],
        )
    axes[1].axhline(1.0, color="0.3", ls="--", lw=0.8)
    axes[1].axvline(
        SYNTHETIC_CAPTURE_DWELL_S,
        color="0.45",
        ls=":",
        lw=1.0,
        label="multi-anchor phase",
    )
    axes[1].set_yscale("log")
    axes[1].set_title("Large-initial-error convergence")
    axes[1].set_xlabel("time [s]")
    axes[1].set_ylabel("position error norm [m]")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)

    axes[2].plot(
        time_values,
        diagnostic["tracker_health_score"],
        color="#4c78a8",
        lw=0.8,
        alpha=0.65,
        label="instant health",
    )
    axes[2].plot(
        time_values,
        diagnostic["smoothed_tracker_health"],
        color="#1b9e77",
        lw=1.1,
        label="smoothed health",
    )
    axes[2].plot(
        time_values,
        diagnostic["tracker_output_weight"],
        color=METHOD_COLORS["deep_coupled"],
        lw=1.3,
        label="tracker output weight",
    )
    axes[2].fill_between(
        time_values,
        0.0,
        1.0,
        where=diagnostic["mode_track"],
        color="#f2cf5b",
        alpha=0.18,
        label="track mode",
    )
    axes[2].axvline(
        SYNTHETIC_CAPTURE_DWELL_S,
        color="0.45",
        ls=":",
        lw=1.0,
    )
    axes[2].set_ylim(-0.02, 1.04)
    axes[2].set_title("Causal hand-off is exercised")
    axes[2].set_xlabel("time [s]")
    axes[2].set_ylabel("score / weight")
    axes[2].grid(alpha=0.25)
    axes[2].legend(fontsize=8, loc="lower right")
    figure.suptitle(
        f"{SYNTHETIC_CAPTURE_DWELL_S:.0f} s single-anchor large-error "
        f"capture + {time_values[-1] - SYNTHETIC_CAPTURE_DWELL_S:.0f} s "
        "eight-anchor "
        f"tracking | accel bias={result['metrics']['true_accel_bias_mps2']} m/s²",
        fontsize=11,
        y=1.02,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def run_synthetic_monte_carlo(
    output_directory: Path,
    *,
    trials: int = 20,
    seed: int = 20260731,
    duration: float = SYNTHETIC_END_TO_END_DURATION_S,
) -> dict[str, Any]:
    if trials < 1:
        raise ValueError("trials must be positive")
    minimum_duration = (
        SYNTHETIC_CAPTURE_DWELL_S
        + SYNTHETIC_MINIMUM_TRACK_EVALUATION_S
    )
    if not np.isfinite(duration) or duration < minimum_duration:
        raise ValueError(
            "synthetic Monte Carlo duration must be at least "
            f"{minimum_duration:.1f} s so adaptive tracking is exercised"
        )

    generator = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    activation_rows: list[dict[str, Any]] = []
    representative: dict[str, Any] | None = None
    for trial in range(trials):
        initial_position = generator.uniform(-50.0, 50.0, size=3)
        initial_velocity = generator.uniform(-10.0, 10.0, size=3)
        result = run_synthetic_deep(
            seed=trial,
            initial_position=initial_position,
            initial_velocity=initial_velocity,
            duration=duration,
        )
        if trial == 0:
            representative = result
        for method, metrics in result["metrics"]["methods"].items():
            rows.append(
                {
                    "trial": trial,
                    "method": method,
                    "initial_px_m": initial_position[0],
                    "initial_py_m": initial_position[1],
                    "initial_pz_m": initial_position[2],
                    "initial_vx_mps": initial_velocity[0],
                    "initial_vy_mps": initial_velocity[1],
                    "initial_vz_mps": initial_velocity[2],
                    **{
                        key: value
                        for key, value in metrics.items()
                        if isinstance(value, (int, float, bool))
                    },
                }
            )

        coupling = result["metrics"]["coupling"]
        maximum_weight = float(
            coupling["maximum_observed_tracker_output_weight"]
        )
        maximum_position_difference = float(
            coupling["maximum_deep_capture_position_difference_m"]
        )
        tail5_maximum_position_difference = float(
            coupling[
                "tail5_maximum_deep_capture_position_difference_m"
            ]
        )
        activated = bool(
            coupling["mode_switches_to_track"] > 0
            and maximum_weight > 1e-12
            and maximum_position_difference > 1e-9
        )
        activation_rows.append(
            {
                "trial": trial,
                "mode_switches_to_track": coupling[
                    "mode_switches_to_track"
                ],
                "first_track_time_s": coupling["first_track_time_s"],
                "track_sample_fraction": coupling[
                    "track_sample_fraction"
                ],
                "mean_tracker_output_weight": coupling[
                    "mean_tracker_output_weight"
                ],
                "maximum_tracker_output_weight": maximum_weight,
                "full_tracker_sample_fraction": coupling[
                    "full_tracker_sample_fraction"
                ],
                "maximum_deep_capture_position_difference_m": (
                    maximum_position_difference
                ),
                "tail5_maximum_deep_capture_position_difference_m": (
                    tail5_maximum_position_difference
                ),
                "adaptive_output_exercised": activated,
                "tail5_output_distinct": bool(
                    tail5_maximum_position_difference > 1e-9
                ),
            }
        )

    table = pd.DataFrame(rows)
    activation_table = pd.DataFrame(activation_rows)
    output_directory.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_directory / "monte_carlo_metrics.csv", index=False)
    activation_table.to_csv(
        output_directory / "adaptive_activation.csv", index=False
    )
    summary: dict[str, Any] = {
        "trials": int(trials),
        "seed": int(seed),
        "duration_s": float(duration),
        "scenario": (
            f"{SYNTHETIC_CAPTURE_DWELL_S:.0f} s single-anchor large-error "
            f"capture followed by {duration - SYNTHETIC_CAPTURE_DWELL_S:.0f} "
            "s cyclic eight-anchor tracking"
        ),
        "capture_phase_s": float(SYNTHETIC_CAPTURE_DWELL_S),
        "true_accel_bias_mps2": list(SYNTHETIC_TRUE_ACCEL_BIAS_MPS2),
        "initial_accel_bias_std_mps2": float(
            SYNTHETIC_INITIAL_ACCEL_BIAS_STD_MPS2
        ),
        "tracking_anchor_offsets_m": [
            list(anchor) for anchor in SYNTHETIC_TRACKING_ANCHOR_OFFSETS_M
        ],
        "total_range_rate_hz": 20.0,
        "initial_position_distribution_m": "independent uniform[-50, 50]",
        "initial_velocity_distribution_mps": "independent uniform[-10, 10]",
        "success_definition": "last-5-s position RMSE < 1 m",
        "methods": {},
    }
    for method, group in table.groupby("method"):
        summary["methods"][method] = {
            "success_count": int(group["success_tail5_below_1m"].sum()),
            "success_rate": float(group["success_tail5_below_1m"].mean()),
            "tail5_position_rmse_median_m": float(
                group["position_tail5_rmse_m"].median()
            ),
            "full_position_rmse_median_m": float(
                group["position_rmse_m"].median()
            ),
            "tail5_velocity_rmse_median_mps": float(
                group["velocity_tail5_rmse_mps"].median()
            ),
            "full_velocity_rmse_median_mps": float(
                group["velocity_rmse_mps"].median()
            ),
        }
    loose = table[table["method"] == "loose_lifted"].sort_values("trial")
    deep = table[table["method"] == "deep_coupled"].sort_values("trial")
    delta = (
        deep["position_tail5_rmse_m"].to_numpy()
        - loose["position_tail5_rmse_m"].to_numpy()
    )
    summary["deep_vs_loose_tail5"] = {
        "better_count": int(np.count_nonzero(delta < -1e-9)),
        "equal_count": int(np.count_nonzero(np.abs(delta) <= 1e-9)),
        "worse_count": int(np.count_nonzero(delta > 1e-9)),
        "worst_delta_m": float(np.max(delta)),
        "non_regression_pass": bool(np.all(delta <= 1e-9)),
    }
    summary["deep_vs_loose_all_core_metrics"] = {}
    all_core_median_non_regression = True
    for metric in CORE_METRICS:
        loose_values = loose[metric].to_numpy(dtype=np.float64)
        deep_values = deep[metric].to_numpy(dtype=np.float64)
        metric_delta = deep_values - loose_values
        loose_median = float(np.median(loose_values))
        deep_median = float(np.median(deep_values))
        median_non_regression = deep_median <= loose_median + 1e-9
        all_core_median_non_regression &= median_non_regression
        summary["deep_vs_loose_all_core_metrics"][metric] = {
            "loose_median": loose_median,
            "deep_median": deep_median,
            "median_delta": deep_median - loose_median,
            "better_count": int(np.count_nonzero(metric_delta < -1e-9)),
            "equal_count": int(
                np.count_nonzero(np.abs(metric_delta) <= 1e-9)
            ),
            "worse_count": int(np.count_nonzero(metric_delta > 1e-9)),
            "worst_delta": float(np.max(metric_delta)),
            "median_non_regression_pass": bool(median_non_regression),
        }
    summary["all_core_medians_non_regression_pass"] = bool(
        all_core_median_non_regression
    )
    summary["success_count_non_regression_pass"] = bool(
        summary["methods"]["deep_coupled"]["success_count"]
        >= summary["methods"]["loose_lifted"]["success_count"]
    )
    exercised = activation_table["adaptive_output_exercised"].astype(bool)
    tail5_distinct = activation_table["tail5_output_distinct"].astype(bool)
    full_weight_reached_count = int(
        np.count_nonzero(
            activation_table["maximum_tracker_output_weight"]
            >= 1.0 - 1e-12
        )
    )
    summary["adaptive_activation"] = {
        "definition": (
            "at least one track-mode switch, positive tracker weight, and "
            "nonzero deep-versus-capture trajectory difference"
        ),
        "exercised_count": int(exercised.sum()),
        "exercised_rate": float(exercised.mean()),
        "tail5_distinct_count": int(tail5_distinct.sum()),
        "median_first_track_time_s": float(
            activation_table["first_track_time_s"].median()
        ),
        "median_track_sample_fraction": float(
            activation_table["track_sample_fraction"].median()
        ),
        "median_mean_tracker_output_weight": float(
            activation_table["mean_tracker_output_weight"].median()
        ),
        "full_weight_reached_count": full_weight_reached_count,
        "nontrivial_activation_pass": bool(
            exercised.all()
            and tail5_distinct.all()
            and full_weight_reached_count == trials
        ),
    }
    _write_json(output_directory / "summary.json", summary)
    if representative is not None:
        plot_synthetic_deep(
            representative, output_directory / "representative.png"
        )
    return summary


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("real", "synthetic", "quick", "full"),
        default="quick",
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--sequence",
        default="2021-02-12_09_56_09_twr_random.csv",
    )
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument(
        "--synthetic-duration",
        type=float,
        default=SYNTHETIC_END_TO_END_DURATION_S,
        help=(
            "end-to-end synthetic duration; must leave at least 10 s after "
            "the mandatory 20 s capture phase"
        ),
    )
    parser.add_argument(
        "--sensor-delay",
        type=float,
        default=TU_DELFT_COMMON_SENSOR_DELAY_S,
        help=(
            "common IMU+TWR acquisition-to-log delay in seconds; "
            "TU Delft calibration default is 0.20"
        ),
    )
    parser.add_argument(
        "--offset", type=float, nargs=3, default=(6.0, -6.0, 3.0)
    )
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument(
        "--output", type=Path, default=Path("results/deep_coupled_quick")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    if args.mode in ("real", "quick"):
        anchors = load_anchors(args.dataset)
        log = load_real_log(
            args.dataset,
            args.sequence,
            args.duration if args.mode == "quick" else None,
            sensor_delay_s=args.sensor_delay,
        )
        result = run_real_deep_sequence(
            log, anchors, initial_offset=args.offset
        )
        real_output = (
            args.output / "real" if args.mode == "quick" else args.output
        )
        save_sequence_result(result, real_output)
        plot_real_result(
            result, real_output / "trajectory_and_errors.png"
        )
        _write_json(
            real_output / "non_regression.json",
            {
                metric: (
                    result["metrics"]["methods"]["deep_coupled"][metric]
                    - result["metrics"]["methods"]["loose_lifted"][metric]
                )
                for metric in CORE_METRICS
            },
        )
    if args.mode in ("synthetic", "quick"):
        synthetic_output = (
            args.output / "synthetic"
            if args.mode == "quick"
            else args.output
        )
        run_synthetic_monte_carlo(
            synthetic_output,
            trials=1 if args.mode == "quick" else args.trials,
            duration=args.synthetic_duration,
        )
    if args.mode == "full":
        real_summary = run_real_batch(
            args.dataset,
            args.output / "real",
            sensor_delay_s=args.sensor_delay,
        )
        synthetic_summary = run_synthetic_monte_carlo(
            args.output / "synthetic",
            trials=args.trials,
            duration=args.synthetic_duration,
        )
        combined = {
            "real": real_summary,
            "synthetic": synthetic_summary,
            "acceptance": {
                "real_all_core_metrics_non_regression": real_summary[
                    "all_four_metrics_non_regression_pass"
                ],
                "synthetic_tail5_non_regression": synthetic_summary[
                    "deep_vs_loose_tail5"
                ]["non_regression_pass"],
                "synthetic_all_core_medians_non_regression": (
                    synthetic_summary[
                        "all_core_medians_non_regression_pass"
                    ]
                ),
                "synthetic_success_count_non_regression": (
                    synthetic_summary[
                        "success_count_non_regression_pass"
                    ]
                ),
                "synthetic_adaptive_output_exercised": synthetic_summary[
                    "adaptive_activation"
                ]["nontrivial_activation_pass"],
            },
        }
        combined["acceptance"]["overall_pass"] = bool(
            all(combined["acceptance"].values())
        )
        _write_json(args.output / "acceptance_summary.json", combined)


if __name__ == "__main__":
    main()
