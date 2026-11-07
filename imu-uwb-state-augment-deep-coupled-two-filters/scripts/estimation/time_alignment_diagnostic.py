#!/usr/bin/env python3
"""Calibrate and validate the common TU Delft IMU/TWR logging delay.

The compact CSV places IMU, held TWR ranges, and OptiTrack columns on one
logger time axis.  This diagnostic tests the physically consistent model

    t_acquisition = t_log - tau

for the complete on-board packet (IMU and TWR together).  It never shifts UWB
relative to IMU.  Two independent, rotation-insensitive scores are reported:

* per-anchor-centered TWR residual RMSE against interpolated OptiTrack;
* standardized IMU specific-force-norm mismatch against differentiated
  OptiTrack position.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from inekf import GRAVITY_MAGNITUDE
from twr_experiment import (
    DEFAULT_DATASET,
    TU_DELFT_COMMON_SENSOR_DELAY_S,
    _smooth_velocity,
    load_anchors,
    real_sequence_files,
)


DEFAULT_CALIBRATION_SEQUENCE = (
    "2021-02-12_09_41_16_twr_triangle.csv"
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


def _standardize(values: np.ndarray) -> np.ndarray:
    centered = values - float(np.mean(values))
    scale = float(np.std(centered))
    if not np.isfinite(scale) or scale <= 1e-12:
        raise ValueError("time-alignment signal has no usable excitation")
    return centered / scale


def _savgol_window(time_values: np.ndarray, duration_s: float = 0.31) -> int:
    median_dt = float(np.median(np.diff(time_values)))
    requested = max(9, int(round(duration_s / median_dt)))
    if requested % 2 == 0:
        requested += 1
    maximum = time_values.size - (1 - time_values.size % 2)
    return max(5, min(requested, maximum))


def scan_sequence(
    dataset: Path,
    sequence: str,
    delays_s: np.ndarray,
    anchors: np.ndarray,
) -> pd.DataFrame:
    """Return range and IMU alignment scores for one sequence."""

    frame = pd.read_csv(dataset / "twr" / sequence, skipinitialspace=True)
    raw_time = frame["timeTick"].to_numpy(dtype=np.float64) / 1000.0
    truth = frame[["otX", "otY", "otZ"]].to_numpy(dtype=np.float64)
    truth_valid = np.all(np.isfinite(truth), axis=1)
    truth_valid &= np.linalg.norm(truth, axis=1) > 1e-9
    truth_time = raw_time[truth_valid]
    truth_position = truth[truth_valid]
    if truth_time.size < 9:
        raise ValueError(f"{sequence} has insufficient valid OptiTrack data")

    imu = frame[["accX", "accY", "accZ"]].to_numpy(dtype=np.float64)
    ranges = frame[[f"twr{index}" for index in range(8)]].to_numpy(
        dtype=np.float64
    )
    sensor_finite = np.all(np.isfinite(imu), axis=1)
    sensor_finite &= np.all(np.isfinite(ranges), axis=1)

    changed = np.ones_like(ranges, dtype=bool)
    changed[1:] = np.abs(np.diff(ranges, axis=0)) > 1e-12

    # Use the same sensor rows for every candidate, otherwise an apparent
    # score change could come only from adding/removing boundary samples.
    maximum_delay = float(np.max(delays_s))
    common = sensor_finite
    common &= raw_time - maximum_delay >= truth_time[0]
    common &= raw_time <= truth_time[-1]
    common_indices = np.flatnonzero(common)
    if common_indices.size < 9:
        raise ValueError(f"{sequence} has no common delay-scan interval")

    truth_velocity = _smooth_velocity(truth_time, truth_position)
    truth_acceleration = _smooth_velocity(truth_time, truth_velocity)
    gravity = np.array([0.0, 0.0, -GRAVITY_MAGNITUDE])
    expected_specific_force_norm = (
        np.linalg.norm(truth_acceleration - gravity, axis=1)
        / GRAVITY_MAGNITUDE
    )
    measured_specific_force_norm = np.linalg.norm(
        imu[common_indices], axis=1
    )
    measured_specific_force_norm = savgol_filter(
        measured_specific_force_norm,
        window_length=_savgol_window(raw_time[common_indices]),
        polyorder=3,
        mode="interp",
    )
    measured_specific_force_norm = _standardize(
        measured_specific_force_norm
    )

    rows: list[dict[str, float | str | int]] = []
    for delay in delays_s:
        centered_residuals: list[np.ndarray] = []
        event_count = 0
        for anchor_index, anchor in enumerate(anchors):
            event_indices = np.flatnonzero(
                common & changed[:, anchor_index]
            )
            event_time = raw_time[event_indices] - float(delay)
            aligned_position = np.column_stack(
                [
                    np.interp(
                        event_time,
                        truth_time,
                        truth_position[:, axis],
                    )
                    for axis in range(3)
                ]
            )
            residual = (
                ranges[event_indices, anchor_index]
                - np.linalg.norm(aligned_position - anchor, axis=1)
            )
            centered_residuals.append(residual - float(np.mean(residual)))
            event_count += int(residual.size)
        range_residual = np.concatenate(centered_residuals)

        acquisition_time = raw_time[common_indices] - float(delay)
        expected = np.interp(
            acquisition_time,
            truth_time,
            expected_specific_force_norm,
        )
        expected = _standardize(expected)
        rows.append(
            {
                "sequence": sequence,
                "delay_s": float(delay),
                "range_centered_rmse_m": float(
                    np.sqrt(np.mean(range_residual**2))
                ),
                "imu_standardized_rmse": float(
                    np.sqrt(
                        np.mean(
                            (
                                measured_specific_force_norm
                                - expected
                            )
                            ** 2
                        )
                    )
                ),
                "range_events": event_count,
                "imu_samples": int(common_indices.size),
            }
        )
    return pd.DataFrame(rows)


def _score_at(
    scan: pd.DataFrame, delay_s: float, metric: str
) -> float:
    index = int(
        np.argmin(
            np.abs(
                scan["delay_s"].to_numpy(dtype=np.float64)
                - float(delay_s)
            )
        )
    )
    return float(scan.iloc[index][metric])


def _sequence_summary(
    scan: pd.DataFrame, selected_delay_s: float
) -> dict[str, Any]:
    range_best = scan.loc[scan["range_centered_rmse_m"].idxmin()]
    imu_best = scan.loc[scan["imu_standardized_rmse"].idxmin()]
    return {
        "sequence": str(scan.iloc[0]["sequence"]),
        "range_optimal_delay_s": float(range_best["delay_s"]),
        "imu_optimal_delay_s": float(imu_best["delay_s"]),
        "range_rmse_zero_delay_m": _score_at(
            scan, 0.0, "range_centered_rmse_m"
        ),
        "range_rmse_selected_delay_m": _score_at(
            scan, selected_delay_s, "range_centered_rmse_m"
        ),
        "imu_rmse_zero_delay": _score_at(
            scan, 0.0, "imu_standardized_rmse"
        ),
        "imu_rmse_selected_delay": _score_at(
            scan, selected_delay_s, "imu_standardized_rmse"
        ),
    }


def _plot(
    scans: pd.DataFrame,
    sequence_summary: pd.DataFrame,
    calibration_sequence: str,
    selected_delay_s: float,
    output_path: Path,
) -> None:
    calibration = scans.loc[
        scans["sequence"] == calibration_sequence
    ]
    aggregate = scans.groupby("delay_s", as_index=False)[
        ["range_centered_rmse_m", "imu_standardized_rmse"]
    ].mean()

    figure, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))
    axes[0].plot(
        calibration["delay_s"],
        calibration["range_centered_rmse_m"],
        label="calibration sequence",
    )
    axes[0].plot(
        aggregate["delay_s"],
        aggregate["range_centered_rmse_m"],
        label="12-sequence mean",
    )
    axes[0].set_ylabel("centered TWR residual RMSE [m]")
    axes[0].legend(fontsize=8)

    axes[1].plot(
        calibration["delay_s"],
        calibration["imu_standardized_rmse"],
        label="calibration sequence",
    )
    axes[1].plot(
        aggregate["delay_s"],
        aggregate["imu_standardized_rmse"],
        label="12-sequence mean",
    )
    axes[1].set_ylabel("standardized IMU/OptiTrack mismatch")
    axes[1].legend(fontsize=8)

    sequence_axis = np.arange(sequence_summary.shape[0])
    axes[2].scatter(
        sequence_summary["range_optimal_delay_s"],
        sequence_axis - 0.12,
        s=22,
        label="TWR optimum",
    )
    axes[2].scatter(
        sequence_summary["imu_optimal_delay_s"],
        sequence_axis + 0.12,
        s=22,
        label="IMU optimum",
    )
    axes[2].set_yticks(sequence_axis)
    axes[2].set_yticklabels(
        [
            Path(name).stem.replace("2021-02-12_", "")
            for name in sequence_summary["sequence"]
        ],
        fontsize=7,
    )
    axes[2].set_ylabel("sequence")
    axes[2].legend(fontsize=8)

    for axis in axes:
        axis.axvline(
            selected_delay_s,
            color="#d95f02",
            linestyle="--",
            linewidth=1.2,
            label=None,
        )
        axis.set_xlabel(r"common IMU+TWR delay $\tau$ [s]")
        axis.grid(True, alpha=0.25)
    figure.suptitle(
        "Common on-board sensor delay calibration "
        f"(selected {selected_delay_s:.2f} s)"
    )
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def run_diagnostic(
    dataset: Path,
    output_directory: Path,
    *,
    calibration_sequence: str = DEFAULT_CALIBRATION_SEQUENCE,
    selected_delay_s: float = TU_DELFT_COMMON_SENSOR_DELAY_S,
    maximum_delay_s: float = 0.40,
    delay_step_s: float = 0.01,
) -> dict[str, Any]:
    if (
        not np.isfinite(selected_delay_s)
        or selected_delay_s < 0.0
        or not np.isfinite(maximum_delay_s)
        or maximum_delay_s <= 0.0
        or not np.isfinite(delay_step_s)
        or delay_step_s <= 0.0
    ):
        raise ValueError("delay scan parameters must be finite and positive")
    delays = np.arange(
        0.0,
        maximum_delay_s + 0.5 * delay_step_s,
        delay_step_s,
        dtype=np.float64,
    )
    sequences = real_sequence_files(dataset)
    if calibration_sequence not in sequences:
        raise ValueError("calibration sequence is not in the real batch")
    anchors = load_anchors(dataset)
    scans = pd.concat(
        [
            scan_sequence(dataset, sequence, delays, anchors)
            for sequence in sequences
        ],
        ignore_index=True,
    )
    summaries = pd.DataFrame(
        [
            _sequence_summary(group, selected_delay_s)
            for _, group in scans.groupby("sequence", sort=False)
        ]
    )
    calibration = summaries.loc[
        summaries["sequence"] == calibration_sequence
    ].iloc[0]
    validation = summaries.loc[
        summaries["sequence"] != calibration_sequence
    ]
    coarse_joint_calibration = 0.05 * round(
        (
            float(calibration["range_optimal_delay_s"])
            + float(calibration["imu_optimal_delay_s"])
        )
        / (2.0 * 0.05)
    )
    summary = {
        "model": "t_acquisition = t_log - tau for paired IMU+TWR",
        "uwb_only_shift": False,
        "calibration_sequence": calibration_sequence,
        "selected_delay_s": float(selected_delay_s),
        "calibration_joint_coarse_delay_s": float(
            coarse_joint_calibration
        ),
        "calibration": calibration.to_dict(),
        "validation_sequences": int(validation.shape[0]),
        "validation_range_improved_count": int(
            np.count_nonzero(
                validation["range_rmse_selected_delay_m"]
                < validation["range_rmse_zero_delay_m"]
            )
        ),
        "validation_imu_improved_count": int(
            np.count_nonzero(
                validation["imu_rmse_selected_delay"]
                < validation["imu_rmse_zero_delay"]
            )
        ),
        "all_sequence_range_rmse_zero_median_m": float(
            summaries["range_rmse_zero_delay_m"].median()
        ),
        "all_sequence_range_rmse_selected_median_m": float(
            summaries["range_rmse_selected_delay_m"].median()
        ),
        "all_sequence_imu_rmse_zero_median": float(
            summaries["imu_rmse_zero_delay"].median()
        ),
        "all_sequence_imu_rmse_selected_median": float(
            summaries["imu_rmse_selected_delay"].median()
        ),
        "limitations": (
            "offline calibration uses OptiTrack; live operation must obtain "
            "hardware acquisition timestamps or apply an equivalent "
            "fixed-lag state-buffer update"
        ),
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    scans.to_csv(output_directory / "delay_scan.csv", index=False)
    summaries.to_csv(output_directory / "per_sequence_summary.csv", index=False)
    _write_json(output_directory / "summary.json", summary)
    _plot(
        scans,
        summaries,
        calibration_sequence,
        selected_delay_s,
        output_directory / "time_alignment_diagnostic.png",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--calibration-sequence",
        default=DEFAULT_CALIBRATION_SEQUENCE,
    )
    parser.add_argument(
        "--selected-delay",
        type=float,
        default=TU_DELFT_COMMON_SENSOR_DELAY_S,
    )
    parser.add_argument("--maximum-delay", type=float, default=0.40)
    parser.add_argument("--delay-step", type=float, default=0.01)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results") / "time_alignment_calibration",
    )
    args = parser.parse_args()
    summary = run_diagnostic(
        args.dataset,
        args.output,
        calibration_sequence=args.calibration_sequence,
        selected_delay_s=args.selected_delay,
        maximum_delay_s=args.maximum_delay,
        delay_step_s=args.delay_step,
    )
    print(json.dumps(_json_ready(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
