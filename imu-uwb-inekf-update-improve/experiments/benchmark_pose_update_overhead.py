"""Measure the computational cost of the task-04 J_l/Q_l propagation.

The benchmark has two scopes:

1. A fixed-input microbenchmark of only the nominal R/v/p formulas.
2. Alternating AB/BA end-to-end replays with the complete author filter.

Filter construction, data loading, metric calculation, plotting, and source
imports are outside the recorded replay interval.  Both variants retain the
same history and process the same IMU/UWB event plan.
"""

import argparse
import csv
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import time

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy

from compare_pose_update import (
    DEFAULT_AUTHOR_REPOSITORY,
    IMPROVED_INEKF,
    load_dataset,
    load_module,
    replay,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = Path(r"E:\datasets\dataset\flight-dataset")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Estimate and validate the computational overhead of the "
            "task-04 J_l/Q_l pose propagation."
        )
    )
    parser.add_argument(
        "--case",
        nargs=4,
        action="append",
        metavar=("NAME", "ANCHORS", "CSV", "DURATION_S"),
        help=(
            "repeatable benchmark case; defaults to 5 s const1 and "
            "5 s const4-manual1 excerpts"
        ),
    )
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--micro-repeats", type=int, default=9)
    parser.add_argument(
        "--micro-target-seconds",
        type=float,
        default=0.15,
        help="target duration of one author-formula microbenchmark batch",
    )
    parser.add_argument(
        "--micro-warmup-seconds",
        type=float,
        default=3.0,
        help="alternating formula warmup before microbenchmark calibration",
    )
    parser.add_argument(
        "--author-repo",
        type=Path,
        default=DEFAULT_AUTHOR_REPOSITORY,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "results" / "task04_compute_overhead",
    )
    parser.add_argument("--yaw-deg", type=float, default=0.0)
    parser.add_argument("--tdoa-variance", type=float, default=0.13)
    return parser.parse_args()


def default_cases():
    survey = DEFAULT_DATASET_ROOT / "survey-results"
    csv_root = DEFAULT_DATASET_ROOT / "csv-data"
    return [
        (
            "const1_5s",
            survey / "anchor_const1.npz",
            csv_root / "const1" / "const1-trial1-tdoa2.csv",
            5.0,
        ),
        (
            "const4_manual1_5s",
            survey / "anchor_const4.npz",
            csv_root / "const4" / "const4-trial7-tdoa2-manual1.csv",
            5.0,
        ),
    ]


def requested_cases(raw_cases):
    if not raw_cases:
        return default_cases()
    parsed = []
    for name, anchor, csv_path, duration_text in raw_cases:
        duration = float(duration_text)
        if duration <= 0.0:
            raise ValueError(f"Duration must be positive for case {name}")
        parsed.append(
            (name, Path(anchor).resolve(), Path(csv_path).resolve(), duration)
        )
    return parsed


def source_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def author_nominal_step(module, rotation, velocity, position, acceleration, omega, dt):
    """Execute the author's nominal R/v/p expressions exactly as written."""
    phi = omega * dt
    gravity = np.array([0.0, 0.0, -module.GRAVITY_MAGNITUDE])
    rotation_new = rotation @ module.SO3_exp(phi)
    velocity_new = velocity + (rotation @ acceleration + gravity) * dt
    position_new = (
        position
        + velocity * dt
        + 0.5
        * (rotation @ acceleration + gravity)
        * dt**2
    )
    return rotation_new, velocity_new, position_new


def improved_nominal_step(
    module, rotation, velocity, position, acceleration, omega, dt
):
    """Execute the implemented task-04 nominal R/v/p expressions."""
    phi = omega * dt
    jacobian, second_jacobian = module.SO3_motion_jacobians(phi)
    gravity = np.array([0.0, 0.0, -module.GRAVITY_MAGNITUDE])
    rotation_new = rotation @ module.SO3_exp(phi)
    velocity_new = velocity + (
        rotation @ jacobian @ acceleration + gravity
    ) * dt
    position_new = (
        position
        + velocity * dt
        + rotation @ second_jacobian @ acceleration * dt**2
        + 0.5 * gravity * dt**2
    )
    return rotation_new, velocity_new, position_new


def time_nominal_batch(step, module, inputs, iterations):
    start = time.perf_counter()
    output = None
    for _ in range(iterations):
        output = step(module, *inputs)
    elapsed = time.perf_counter() - start
    if output is None or not np.all(np.isfinite(output[2])):
        raise AssertionError("Invalid nominal microbenchmark output")
    return elapsed


def calibrate_iterations(step, module, inputs, target_seconds):
    iterations = 256
    while True:
        elapsed = time_nominal_batch(step, module, inputs, iterations)
        if elapsed >= target_seconds or iterations >= 1_048_576:
            return iterations
        iterations *= 2


def distribution(values):
    median = statistics.median(values)
    deviations = [abs(value - median) for value in values]
    return {
        "median": float(median),
        "mad": float(statistics.median(deviations)),
        "minimum": float(min(values)),
        "maximum": float(max(values)),
        "values": [float(value) for value in values],
    }


def run_microbenchmark(
    author_module,
    improved_module,
    repeats,
    target_seconds,
    warmup_seconds,
    input_name,
    omega_deg_per_s,
):
    rotation = improved_module.SO3_exp(np.array([0.2, -0.1, 0.3]))
    velocity = np.array([0.4, -0.3, 0.2])
    position = np.array([1.0, -2.0, 10.0])
    acceleration = np.array([3.5, -1.7, 2.2])
    omega = np.radians(np.asarray(omega_deg_per_s, dtype=np.float64))
    dt = 0.001
    inputs = (rotation, velocity, position, acceleration, omega, dt)

    warmup_deadline = time.perf_counter() + warmup_seconds
    while time.perf_counter() < warmup_deadline:
        time_nominal_batch(
            author_nominal_step, author_module, inputs, 256
        )
        time_nominal_batch(
            improved_nominal_step, improved_module, inputs, 256
        )
    iterations = calibrate_iterations(
        author_nominal_step,
        author_module,
        inputs,
        target_seconds,
    )

    raw_rows = []
    timings = {"author_euler": [], "improved_jl_ql": []}
    variants = {
        "author_euler": (author_nominal_step, author_module),
        "improved_jl_ql": (improved_nominal_step, improved_module),
    }
    for round_index in range(repeats):
        order = (
            ("author_euler", "improved_jl_ql")
            if round_index % 2 == 0
            else ("improved_jl_ql", "author_euler")
        )
        for order_position, label in enumerate(order):
            step, module = variants[label]
            gc.collect()
            gc.disable()
            try:
                elapsed = time_nominal_batch(
                    step, module, inputs, iterations
                )
            finally:
                gc.enable()
            seconds_per_step = elapsed / iterations
            timings[label].append(seconds_per_step)
            raw_rows.append(
                {
                    "scope": "nominal_microbenchmark",
                    "case": input_name,
                    "round": round_index,
                    "order_position": order_position,
                    "variant": label,
                    "elapsed_seconds": elapsed,
                    "iterations": iterations,
                    "seconds_per_step": seconds_per_step,
                }
            )

    author_stats = distribution(timings["author_euler"])
    improved_stats = distribution(timings["improved_jl_ql"])
    paired_overheads = [
        100.0 * (improved - author) / author
        for author, improved in zip(
            timings["author_euler"], timings["improved_jl_ql"]
        )
    ]
    return (
        {
            "input": {
                "name": input_name,
                "dt_seconds": dt,
                "omega_deg_per_s": np.degrees(omega).tolist(),
                "phi_norm_rad": float(np.linalg.norm(omega * dt)),
                "iterations_per_batch": iterations,
                "repeats": repeats,
                "warmup_seconds": warmup_seconds,
            },
            "author_euler": author_stats,
            "improved_jl_ql": improved_stats,
            "ratio_of_medians": (
                improved_stats["median"] / author_stats["median"]
            ),
            "overhead_percent_ratio_of_medians": 100.0
            * (
                improved_stats["median"] / author_stats["median"]
                - 1.0
            ),
            "paired_overhead_percent": distribution(paired_overheads),
            "added_seconds_per_prediction": (
                improved_stats["median"] - author_stats["median"]
            ),
        },
        raw_rows,
    )


def motion_angle_statistics(dataset):
    events = np.unique(np.concatenate((dataset.t_imu, dataset.t_uwb)))
    imu_lookup = {
        float(timestamp): index
        for index, timestamp in enumerate(dataset.t_imu)
    }
    angles = []
    propagation_dts = []
    last_imu_propagation_time = 0.0
    for event_index in range(1, len(events)):
        previous_time = float(events[event_index - 1])
        current_time = float(events[event_index])
        imu_index = imu_lookup.get(previous_time)
        if imu_index is None:
            continue
        dt = current_time - last_imu_propagation_time
        last_imu_propagation_time = current_time
        omega = np.radians(dataset.imu[imu_index, 3:6])
        angles.append(float(np.linalg.norm(omega * dt)))
        propagation_dts.append(dt)

    angle_array = np.asarray(angles)
    dt_array = np.asarray(propagation_dts)
    return {
        "count": int(len(angle_array)),
        "small_angle_threshold_rad": 1e-4,
        "small_angle_count": int(np.count_nonzero(angle_array < 1e-4)),
        "small_angle_fraction": float(np.mean(angle_array < 1e-4)),
        "phi_norm_rad": {
            "median": float(np.median(angle_array)),
            "p95": float(np.percentile(angle_array, 95)),
            "maximum": float(np.max(angle_array)),
        },
        "propagation_dt_seconds": {
            "median": float(np.median(dt_array)),
            "p95": float(np.percentile(dt_array, 95)),
            "maximum": float(np.max(dt_array)),
        },
    }


def validate_replay_pair(author_result, improved_result):
    np.testing.assert_array_equal(author_result.times, improved_result.times)
    if author_result.prediction_count != improved_result.prediction_count:
        raise AssertionError("Prediction count changed between variants")
    if author_result.uwb_count != improved_result.uwb_count:
        raise AssertionError("UWB count changed between variants")
    if author_result.rejection_count != improved_result.rejection_count:
        raise AssertionError(
            "UWB rejection path changed; timing would not be paired fairly"
        )
    arrays = (
        author_result.rotations,
        author_result.velocities,
        author_result.positions,
        author_result.biases,
        improved_result.rotations,
        improved_result.velocities,
        improved_result.positions,
        improved_result.biases,
    )
    if not all(np.all(np.isfinite(array)) for array in arrays):
        raise AssertionError("Non-finite replay state")


def order_statistics(raw_rounds, label):
    first = [
        row["elapsed_seconds"]
        for row in raw_rounds
        if row["variant"] == label and row["order_position"] == 0
    ]
    second = [
        row["elapsed_seconds"]
        for row in raw_rounds
        if row["variant"] == label and row["order_position"] == 1
    ]
    return {
        "first_median_seconds": float(statistics.median(first)),
        "second_median_seconds": float(statistics.median(second)),
    }


def benchmark_replay_case(
    name,
    anchor_path,
    csv_path,
    duration,
    repeats,
    author_module,
    improved_module,
    yaw_degrees,
    tdoa_variance,
    microbenchmarks,
):
    dataset = load_dataset(anchor_path, csv_path, duration)
    warmup_dataset = load_dataset(
        anchor_path, csv_path, min(1.0, duration)
    )
    replay(
        "author_warmup",
        author_module,
        warmup_dataset,
        yaw_degrees,
        tdoa_variance,
    )
    replay(
        "improved_warmup",
        improved_module,
        warmup_dataset,
        yaw_degrees,
        tdoa_variance,
    )

    timings = {"author_euler": [], "improved_jl_ql": []}
    raw_rows = []
    state_differences = []
    deterministic_reference = {}
    variants = {
        "author_euler": author_module,
        "improved_jl_ql": improved_module,
    }

    for round_index in range(repeats):
        order = (
            ("author_euler", "improved_jl_ql")
            if round_index % 2 == 0
            else ("improved_jl_ql", "author_euler")
        )
        round_results = {}
        for order_position, label in enumerate(order):
            gc.collect()
            gc.disable()
            try:
                result = replay(
                    label,
                    variants[label],
                    dataset,
                    yaw_degrees,
                    tdoa_variance,
                )
            finally:
                gc.enable()
            round_results[label] = result
            timings[label].append(result.elapsed_seconds)
            raw_rows.append(
                {
                    "scope": "complete_filter_replay",
                    "case": name,
                    "round": round_index,
                    "order_position": order_position,
                    "variant": label,
                    "elapsed_seconds": result.elapsed_seconds,
                    "iterations": 1,
                    "seconds_per_step": "",
                }
            )
            final_vector = np.concatenate(
                (
                    result.rotations[-1].ravel(),
                    result.velocities[-1],
                    result.positions[-1],
                    result.biases[-1],
                )
            )
            if label not in deterministic_reference:
                deterministic_reference[label] = final_vector
            else:
                np.testing.assert_allclose(
                    final_vector,
                    deterministic_reference[label],
                    rtol=0.0,
                    atol=0.0,
                )

        author_result = round_results["author_euler"]
        improved_result = round_results["improved_jl_ql"]
        validate_replay_pair(author_result, improved_result)
        state_differences.append(
            {
                "max_position_difference_m": float(
                    np.max(
                        np.linalg.norm(
                            improved_result.positions
                            - author_result.positions,
                            axis=1,
                        )
                    )
                ),
                "max_velocity_difference_mps": float(
                    np.max(
                        np.linalg.norm(
                            improved_result.velocities
                            - author_result.velocities,
                            axis=1,
                        )
                    )
                ),
            }
        )
        print(
            f"[{name}] round {round_index + 1}/{repeats}: "
            f"author={author_result.elapsed_seconds:.6f}s, "
            f"improved={improved_result.elapsed_seconds:.6f}s"
        )

    author_stats = distribution(timings["author_euler"])
    improved_stats = distribution(timings["improved_jl_ql"])
    paired_overheads = [
        100.0 * (improved - author) / author
        for author, improved in zip(
            timings["author_euler"], timings["improved_jl_ql"]
        )
    ]
    prediction_count = author_result.prediction_count
    angle_statistics = motion_angle_statistics(dataset)
    small_angle_fraction = angle_statistics["small_angle_fraction"]
    weighted_added_seconds_per_prediction = (
        small_angle_fraction
        * microbenchmarks["small_angle"]["added_seconds_per_prediction"]
        + (1.0 - small_angle_fraction)
        * microbenchmarks["general_angle"]["added_seconds_per_prediction"]
    )
    projected_extra = (
        weighted_added_seconds_per_prediction * prediction_count
    )
    observed_extra = (
        improved_stats["median"] - author_stats["median"]
    )
    case_result = {
        "name": name,
        "inputs": {
            "anchors": str(Path(anchor_path).resolve()),
            "csv": str(Path(csv_path).resolve()),
            "duration_seconds": duration,
        },
        "counts": {
            "prediction_count": prediction_count,
            "uwb_count": author_result.uwb_count,
            "rejection_count_both": author_result.rejection_count,
            "event_state_count": int(len(author_result.times)),
        },
        "motion_angles": angle_statistics,
        "author_euler": author_stats,
        "improved_jl_ql": improved_stats,
        "ratio_of_medians": (
            improved_stats["median"] / author_stats["median"]
        ),
        "overhead_percent_ratio_of_medians": 100.0
        * (
            improved_stats["median"] / author_stats["median"]
            - 1.0
        ),
        "paired_overhead_percent": distribution(paired_overheads),
        "order_effects": {
            "author_euler": order_statistics(
                raw_rows, "author_euler"
            ),
            "improved_jl_ql": order_statistics(
                raw_rows, "improved_jl_ql"
            ),
        },
        "microbenchmark_projection": {
            "weighted_added_seconds_per_prediction": (
                weighted_added_seconds_per_prediction
            ),
            "projected_extra_seconds": projected_extra,
            "projected_overhead_percent_of_author_median": (
                100.0 * projected_extra / author_stats["median"]
            ),
            "observed_extra_seconds_ratio_of_medians": observed_extra,
        },
        "state_difference": state_differences[0],
        "deterministic_across_rounds": True,
        "rejection_decisions_unchanged": True,
    }
    return case_result, raw_rows


def create_figure(microbenchmarks, cases, output_path):
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "figure.dpi": 140,
        }
    )
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    axis = axes[0]
    micro_names = ("general_angle", "small_angle")
    micro_x = np.arange(len(micro_names))
    micro_width = 0.36
    author_micro_values = [
        1e6 * microbenchmarks[name]["author_euler"]["median"]
        for name in micro_names
    ]
    improved_micro_values = [
        1e6 * microbenchmarks[name]["improved_jl_ql"]["median"]
        for name in micro_names
    ]
    author_micro_errors = [
        1e6 * microbenchmarks[name]["author_euler"]["mad"]
        for name in micro_names
    ]
    improved_micro_errors = [
        1e6 * microbenchmarks[name]["improved_jl_ql"]["mad"]
        for name in micro_names
    ]
    axis.bar(
        micro_x - micro_width / 2,
        author_micro_values,
        micro_width,
        yerr=author_micro_errors,
        color="#d95f02",
        label="Author Euler",
        capsize=4,
    )
    axis.bar(
        micro_x + micro_width / 2,
        improved_micro_values,
        micro_width,
        yerr=improved_micro_errors,
        color="#1b9e77",
        label="Improved J/Q",
        capsize=4,
    )
    axis.set_xticks(micro_x, ("general angle", "small angle"))
    axis.set_ylabel("median time per nominal step [us]")
    axis.set_title("Nominal pose formula only")
    axis.legend(loc="best")

    axis = axes[1]
    x = np.arange(len(cases))
    width = 0.36
    author_values = [case["author_euler"]["median"] for case in cases]
    improved_values = [
        case["improved_jl_ql"]["median"] for case in cases
    ]
    author_errors = [case["author_euler"]["mad"] for case in cases]
    improved_errors = [case["improved_jl_ql"]["mad"] for case in cases]
    axis.bar(
        x - width / 2,
        author_values,
        width,
        yerr=author_errors,
        label="Author Euler",
        color="#d95f02",
        capsize=4,
    )
    axis.bar(
        x + width / 2,
        improved_values,
        width,
        yerr=improved_errors,
        label="Improved J/Q",
        color="#1b9e77",
        capsize=4,
    )
    axis.set_xticks(x, [case["name"] for case in cases], rotation=12)
    axis.set_ylabel("median complete replay time [s]")
    axis.set_title("Complete InEKF replay, AB/BA")
    axis.legend(loc="best")
    for index, case in enumerate(cases):
        height = max(author_values[index], improved_values[index])
        axis.text(
            index,
            height * 1.03,
            f"{case['overhead_percent_ratio_of_medians']:+.2f}%",
            ha="center",
            va="bottom",
        )

    axis = axes[2]
    projected = [
        case["microbenchmark_projection"][
            "projected_overhead_percent_of_author_median"
        ]
        for case in cases
    ]
    observed = [
        case["overhead_percent_ratio_of_medians"] for case in cases
    ]
    axis.bar(
        x - width / 2,
        projected,
        width,
        label="micro projected",
        color="#7570b3",
    )
    axis.bar(
        x + width / 2,
        observed,
        width,
        label="replay observed",
        color="#e7298a",
    )
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xticks(x, [case["name"] for case in cases], rotation=12)
    axis.set_ylabel("overhead relative to author [%]")
    axis.set_title("Local-cost estimate vs measured replay")
    axis.legend(loc="best")

    figure.suptitle(
        "Task-04 J_l/Q_l computational overhead validation",
        fontsize=14,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def main():
    args = parse_args()
    if args.repeats < 4:
        raise ValueError("--repeats must be at least 4 for AB/BA timing")
    if args.micro_repeats < 5:
        raise ValueError("--micro-repeats must be at least 5")

    author_inekf = (
        args.author_repo.resolve()
        / "scripts"
        / "estimation"
        / "inekf.py"
    )
    if not author_inekf.exists():
        raise FileNotFoundError(f"Author source not found: {author_inekf}")
    author_module = load_module("overhead_author_inekf", author_inekf)
    improved_module = load_module(
        "overhead_improved_inekf", IMPROVED_INEKF
    )

    print("[micro] general-angle nominal formulas")
    general_microbenchmark, raw_rows = run_microbenchmark(
        author_module,
        improved_module,
        args.micro_repeats,
        args.micro_target_seconds,
        args.micro_warmup_seconds,
        "general_angle",
        (20.0, -15.0, 30.0),
    )
    print(
        "[micro general] "
        f"author={1e6 * general_microbenchmark['author_euler']['median']:.3f}us, "
        f"improved={1e6 * general_microbenchmark['improved_jl_ql']['median']:.3f}us, "
        f"overhead={general_microbenchmark['overhead_percent_ratio_of_medians']:+.3f}%"
    )
    print("[micro] small-angle nominal formulas")
    small_microbenchmark, small_raw_rows = run_microbenchmark(
        author_module,
        improved_module,
        args.micro_repeats,
        args.micro_target_seconds,
        args.micro_warmup_seconds,
        "small_angle",
        (1.0, -1.0, 1.0),
    )
    raw_rows.extend(small_raw_rows)
    print(
        "[micro small] "
        f"author={1e6 * small_microbenchmark['author_euler']['median']:.3f}us, "
        f"improved={1e6 * small_microbenchmark['improved_jl_ql']['median']:.3f}us, "
        f"overhead={small_microbenchmark['overhead_percent_ratio_of_medians']:+.3f}%"
    )
    microbenchmarks = {
        "general_angle": general_microbenchmark,
        "small_angle": small_microbenchmark,
    }

    cases = []
    for name, anchor_path, csv_path, duration in requested_cases(args.case):
        print(f"[case] {name}: loading {duration:.3f}s")
        case_result, case_rows = benchmark_replay_case(
            name,
            anchor_path,
            csv_path,
            duration,
            args.repeats,
            author_module,
            improved_module,
            args.yaw_deg,
            args.tdoa_variance,
            microbenchmarks,
        )
        cases.append(case_result)
        raw_rows.extend(case_rows)

    theoretical_cost = {
        "comparison_scope": (
            "additional work beyond the shared SO3_exp rotation update"
        ),
        "general_angle_branch": {
            "additional_3x3_matrix_multiplications": 3,
            "estimated_additional_scalar_multiplications": 129,
            "estimated_additional_scalar_additions": 92,
            "additional_transcendentals_in_current_code": {
                "sqrt": 1,
                "sin": 1,
                "cos": 2,
            },
            "notes": (
                "Counts include Phi@Phi, R@J, R@Q, construction of J/Q, "
                "and phi dot phi; scalar coefficient divisions are not "
                "included."
            ),
        },
        "small_angle_branch": {
            "additional_3x3_matrix_multiplications": 3,
            "additional_transcendentals_in_motion_jacobians": 0,
            "notes": "Taylor coefficients replace helper sin/cos/sqrt.",
        },
        "memory": {
            "persistent_state_growth_bytes": 0,
            "temporary_3x3_matrices": ["Phi", "Phi_squared", "J_left", "Q_left"],
            "notes": (
                "All temporary arrays are per prediction and are released; "
                "filter state and history dimensions are unchanged."
            ),
        },
    }

    payload = {
        "task": "task04 pose update computational overhead validation",
        "command": [sys.executable, *sys.argv],
        "methodology": {
            "blas_threads": 1,
            "replay_order": "AB/BA alternating",
            "replay_repeats": args.repeats,
            "micro_repeats": args.micro_repeats,
            "micro_warmup_seconds_per_regime": args.micro_warmup_seconds,
            "warmup_duration_seconds_per_variant": 1.0,
            "timed_replay_includes": (
                "prediction, covariance, UWB correction, and identical "
                "history copies"
            ),
            "timed_replay_excludes": (
                "data loading, filter construction, warmup, metrics, JSON, "
                "CSV, and plotting"
            ),
            "summary_statistic": "median with median absolute deviation",
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "source_sha256": {
            "author_inekf": source_sha256(author_inekf),
            "improved_inekf": source_sha256(IMPROVED_INEKF),
        },
        "theoretical_cost_estimate": theoretical_cost,
        "nominal_microbenchmarks": microbenchmarks,
        "complete_replay_cases": cases,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "benchmark.json"
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    timing_path = args.output_dir / "raw_timings.csv"
    with timing_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "scope",
                "case",
                "round",
                "order_position",
                "variant",
                "elapsed_seconds",
                "iterations",
                "seconds_per_step",
            ),
        )
        writer.writeheader()
        writer.writerows(raw_rows)

    summary_path = args.output_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "case",
                "author_median_seconds",
                "improved_median_seconds",
                "overhead_percent",
                "paired_overhead_median_percent",
                "micro_projected_overhead_percent",
                "prediction_count",
                "uwb_count",
                "rejection_count_both",
                "small_angle_fraction",
            ),
        )
        writer.writeheader()
        for case in cases:
            writer.writerow(
                {
                    "case": case["name"],
                    "author_median_seconds": case["author_euler"]["median"],
                    "improved_median_seconds": case["improved_jl_ql"]["median"],
                    "overhead_percent": case[
                        "overhead_percent_ratio_of_medians"
                    ],
                    "paired_overhead_median_percent": case[
                        "paired_overhead_percent"
                    ]["median"],
                    "micro_projected_overhead_percent": case[
                        "microbenchmark_projection"
                    ]["projected_overhead_percent_of_author_median"],
                    "prediction_count": case["counts"]["prediction_count"],
                    "uwb_count": case["counts"]["uwb_count"],
                    "rejection_count_both": case["counts"][
                        "rejection_count_both"
                    ],
                    "small_angle_fraction": case["motion_angles"][
                        "small_angle_fraction"
                    ],
                }
            )

    figure_path = args.output_dir / "overhead.png"
    create_figure(microbenchmarks, cases, figure_path)
    print(f"[saved] {json_path}")
    print(f"[saved] {timing_path}")
    print(f"[saved] {summary_path}")
    print(f"[saved] {figure_path}")


if __name__ == "__main__":
    main()
