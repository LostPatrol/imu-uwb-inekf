"""Fair benchmark of the untouched author code and its compute-only acceleration.

Both sides use one precomputed event plan and receive identical IMU/UWB input
calls. Dataset loading, estimator construction, JIT warmup, history recording,
metric evaluation, and plotting are excluded from measured replay time.
The copied ``inekf.py`` must remain byte-identical to the backup; only event
planning, fixed buffers, and native execution are accelerated.
"""

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

# The current Windows conda-forge stack combines MKL 2026.x with LLVM OpenMP
# (loaded by Numba).  Its threaded MKL dispatch can terminate the interpreter
# with native exception 0xC06D007F during repeated small matrix products.
# The benchmark is intentionally single-core, so the MKL sequential layer is
# both fair and avoids loading the conflicting OpenMP runtime.  This variable
# is ignored by non-MKL BLAS implementations.
os.environ["MKL_THREADING_LAYER"] = "SEQUENTIAL"

# Import the established benchmark first: it fixes numerical-library thread
# counts before NumPy/SciPy are loaded and provides the read-only author adapter.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPOSITORY_ROOT.parent
BASE_EXPERIMENTS = WORKSPACE_ROOT / "imu-uwb-inekf" / "experiments"
sys.path.insert(0, str(BASE_EXPERIMENTS))
import benchmark_author_original_vs_proposed as benchmark_base

import matplotlib.pyplot as plt
import numpy as np


ESTIMATION_DIR = REPOSITORY_ROOT / "scripts" / "estimation"
sys.path.insert(0, str(ESTIMATION_DIR))

from accelerated_replay import (  # noqa: E402
    ensure_accelerated_kernel,
    execute_accelerated_replay,
    prepare_accelerated_replay,
)
from pipeline import EstimationResult, compute_metrics, load_dataset  # noqa: E402


AUTHOR = "author_original"
ACCELERATED = "author_accelerated"
VARIANTS = (AUTHOR, ACCELERATED)
ACC_AUTHOR_INEKF = ESTIMATION_DIR / "inekf.py"
ACCELERATED_CORE = ESTIMATION_DIR / "accelerated_replay.py"


def set_process_affinity(cpu_index):
    """Pin this benchmark process to one logical CPU when requested."""
    if cpu_index is None:
        return
    cpu_count = os.cpu_count() or 1
    if cpu_index < 0 or cpu_index >= cpu_count:
        raise ValueError(
            f"--cpu-affinity must be in [0, {cpu_count - 1}]"
        )
    if sys.platform == "win32":
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.SetProcessAffinityMask.argtypes = (
            ctypes.c_void_p,
            ctypes.c_size_t,
        )
        kernel32.SetProcessAffinityMask.restype = ctypes.c_int
        process = kernel32.GetCurrentProcess()
        if not kernel32.SetProcessAffinityMask(
            process,
            ctypes.c_size_t(1 << cpu_index),
        ):
            raise OSError(
                ctypes.get_last_error(),
                "SetProcessAffinityMask failed",
            )
    elif hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {cpu_index})
    else:
        raise RuntimeError(
            "CPU affinity is unsupported on this platform"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Fair common-schedule comparison of the untouched paper-author "
            "implementation and a compute-only native acceleration of exactly "
            "the same equations and settings."
        )
    )
    parser.add_argument(
        "-i",
        nargs=2,
        required=True,
        metavar=("ANCHORS", "CSV"),
    )
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--warmup-duration", type=float, default=30.0)
    parser.add_argument("--max-duration", type=float)
    parser.add_argument("--yaw-deg", type=float, default=0.0)
    parser.add_argument(
        "--cpu-affinity",
        type=int,
        help=(
            "Pin this process to one logical CPU. Used by the parallel "
            "all-file runner so author and accelerated replays share a core."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT
        / "results"
        / "fair_author_original_vs_acc",
    )
    parser.add_argument(
        "--author-tdoa-variance",
        type=float,
        default=0.13,
        help="The author's main.py uses 0.13 for every constellation.",
    )
    return parser.parse_args()


def _fixed_plan(plan):
    return (
        np.ascontiguousarray(plan.events, dtype=np.float64),
        np.ascontiguousarray(plan.imu_index_at_event, dtype=np.int64),
        np.ascontiguousarray(plan.uwb_start_at_event, dtype=np.int64),
        np.ascontiguousarray(plan.uwb_stop_at_event, dtype=np.int64),
    )


def load_acc_author_module():
    """Load the copied, byte-identical author class under a unique module name."""
    spec = importlib.util.spec_from_file_location(
        "acc_copied_author_inekf",
        ACC_AUTHOR_INEKF,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _prepare_acc(
    dataset,
    plan,
    acc_author_module,
    yaw_degrees,
    tdoa_variance,
):
    adapter = benchmark_base.make_author_adapter(
        dataset,
        acc_author_module,
        yaw_degrees,
        tdoa_variance,
    )
    prepared = prepare_accelerated_replay(
        dataset,
        adapter.estimator,
        plan=_fixed_plan(plan),
    )
    return adapter, prepared


def _acc_timing_run(
    dataset,
    plan,
    acc_author_module,
    yaw_degrees,
    tdoa_variance,
):
    start_initialization = time.perf_counter_ns()
    _, prepared = _prepare_acc(
        dataset,
        plan,
        acc_author_module,
        yaw_degrees,
        tdoa_variance,
    )
    initialization_seconds = (
        time.perf_counter_ns() - start_initialization
    ) * 1e-9
    gc.collect()
    gc_enabled = gc.isenabled()
    gc.disable()
    try:
        start_replay = time.perf_counter_ns()
        output = execute_accelerated_replay(
            prepared,
            record_history=False,
        )
        replay_seconds = (time.perf_counter_ns() - start_replay) * 1e-9
    finally:
        if gc_enabled:
            gc.enable()

    predict_calls = int(output[-3])
    processed_measurements = int(output[-2])
    rejected_measurements = int(output[-1])
    if predict_calls != len(dataset.t_imu) - 1:
        raise AssertionError(
            "Accelerated replay did not process every IMU interval: "
            f"{predict_calls} vs {len(dataset.t_imu) - 1}"
        )
    if processed_measurements != len(dataset.t_uwb):
        raise AssertionError(
            "Accelerated replay did not process every UWB measurement: "
            f"{processed_measurements} vs {len(dataset.t_uwb)}"
        )
    return {
        "initialization_seconds": initialization_seconds,
        "replay_seconds": replay_seconds,
        "predict_calls": predict_calls,
        "uwb_calls": processed_measurements,
        "stationary_calls": 0,
        "rejected_measurements": rejected_measurements,
    }


def _author_timing_run(
    dataset,
    plan,
    author_module,
    yaw_degrees,
    tdoa_variance,
):
    timing = benchmark_base.timing_run(
        AUTHOR,
        dataset,
        plan,
        author_module,
        yaw_degrees,
        tdoa_variance,
    )
    return {
        "initialization_seconds": timing["initialization_seconds"],
        "replay_seconds": timing["replay_seconds"],
        "predict_calls": timing["predict_calls"],
        "uwb_calls": timing["uwb_calls"],
        "stationary_calls": timing["stationary_calls"],
    }


def _acc_evaluated_run(
    dataset,
    plan,
    acc_author_module,
    yaw_degrees,
    tdoa_variance,
):
    _, prepared = _prepare_acc(
        dataset,
        plan,
        acc_author_module,
        yaw_degrees,
        tdoa_variance,
    )
    start_replay = time.perf_counter_ns()
    output = execute_accelerated_replay(
        prepared,
        record_history=True,
        record_covariances=False,
    )
    replay_seconds = (time.perf_counter_ns() - start_replay) * 1e-9
    (
        states,
        rotations,
        covariances,
        biases,
        nis_times,
        nis,
        rejected,
        predict_calls,
        processed_measurements,
        rejected_measurements,
    ) = output
    if processed_measurements != len(nis):
        raise AssertionError("Recorded accelerated NIS count is inconsistent")
    if rejected_measurements != int(np.sum(rejected)):
        raise AssertionError(
            "Recorded accelerated rejection count is inconsistent"
        )
    if predict_calls != len(dataset.t_imu) - 1:
        raise AssertionError("Recorded accelerated predict count is inconsistent")
    result = EstimationResult(
        variant=ACCELERATED,
        times=plan.events.copy(),
        states=states,
        rotations=rotations,
        covariances=covariances,
        biases=biases,
        nis_times=nis_times,
        nis=nis,
        rejected=rejected,
        runtime_seconds=replay_seconds,
        stationary_updates=0,
        metrics={},
    )
    diagnostics = compute_metrics(dataset, result)
    return result, diagnostics, {
        "replay_seconds": replay_seconds,
        "predict_calls": predict_calls,
        "uwb_calls": processed_measurements,
        "stationary_calls": 0,
    }


def _author_evaluated_run(
    dataset,
    plan,
    author_module,
    yaw_degrees,
    tdoa_variance,
):
    return benchmark_base.evaluated_run(
        AUTHOR,
        dataset,
        plan,
        author_module,
        yaw_degrees,
        tdoa_variance,
    )


def _sample_step(length, maximum=6000):
    return max(1, int(math.ceil(length / maximum)))


def create_effect_figure(
    dataset,
    results,
    diagnostics,
    timing_rows,
    output_path,
):
    figure, axes = plt.subplots(2, 3, figsize=(17.0, 9.2))
    colors = {AUTHOR: "tab:blue", ACCELERATED: "tab:orange"}
    gt_step = _sample_step(len(dataset.t_gt))
    axes[0, 0].plot(
        dataset.p_gt[::gt_step, 0],
        dataset.p_gt[::gt_step, 1],
        "k--",
        lw=1.0,
        label="Vicon",
    )

    for name in VARIANTS:
        result = results[name]
        diagnostic = diagnostics[name]
        state_step = _sample_step(len(result.times))
        diagnostic_step = _sample_step(len(diagnostic["times"]))
        axes[0, 0].plot(
            result.states[::state_step, 6],
            result.states[::state_step, 7],
            color=colors[name],
            lw=0.8,
            label=name,
        )
        axes[0, 1].plot(
            diagnostic["times"][::diagnostic_step],
            np.linalg.norm(
                diagnostic["position_error"][::diagnostic_step],
                axis=1,
            ),
            color=colors[name],
            lw=0.7,
            label=name,
        )
        axes[0, 2].plot(
            diagnostic["times"][::diagnostic_step],
            np.linalg.norm(
                diagnostic["velocity_error"][::diagnostic_step],
                axis=1,
            ),
            color=colors[name],
            lw=0.7,
            label=name,
        )
        axes[1, 0].plot(
            diagnostic["times"][::diagnostic_step],
            np.degrees(diagnostic["geodesic_error"][::diagnostic_step]),
            color=colors[name],
            lw=0.7,
            label=name,
        )
        axes[1, 1].plot(
            diagnostic["times"][::diagnostic_step],
            np.degrees(diagnostic["yaw_error"][::diagnostic_step]),
            color=colors[name],
            lw=0.7,
            label=name,
        )

    replay_medians = [
        np.median(timing_rows[name]["replay_seconds"])
        for name in VARIANTS
    ]
    bars = axes[1, 2].bar(
        VARIANTS,
        replay_medians,
        color=[colors[name] for name in VARIANTS],
    )
    for bar, value in zip(bars, replay_medians):
        axes[1, 2].text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.3f} s",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    axes[1, 2].set_yscale("log")
    axes[1, 2].set_ylabel("median replay seconds (log scale)")
    axes[1, 2].set_title(
        "Common-schedule speed\n"
        f"{replay_medians[0] / replay_medians[1]:.2f}x"
    )

    axes[0, 0].set(xlabel="x [m]", ylabel="y [m]", title="XY trajectory")
    axes[0, 0].axis("equal")
    axes[0, 1].set(
        xlabel="time [s]",
        ylabel="position error [m]",
        title="Position error",
    )
    axes[0, 2].set(
        xlabel="time [s]",
        ylabel="velocity error [m/s]",
        title="Velocity error",
    )
    axes[1, 0].set(
        xlabel="time [s]",
        ylabel="geodesic error [deg]",
        title="Orientation error",
    )
    axes[1, 1].set(
        xlabel="time [s]",
        ylabel="signed yaw error [deg]",
        title="Yaw error",
    )
    for axis in axes.flat[:5]:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    axes[1, 2].grid(axis="y", alpha=0.25)
    figure.suptitle(
        "Untouched paper-author InEKF vs compute-only native acceleration",
        fontsize=14,
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def create_nis_figure(results, output_path):
    figure, axes = plt.subplots(2, 1, figsize=(13.0, 8.0), sharex=True)
    colors = {AUTHOR: "tab:blue", ACCELERATED: "tab:orange"}
    maximum_rejected = 0
    for name in VARIANTS:
        result = results[name]
        step = _sample_step(len(result.nis_times), maximum=10000)
        axes[0].plot(
            result.nis_times[::step],
            np.maximum(result.nis[::step], 1e-8),
            color=colors[name],
            lw=0.65,
            label=name,
        )
        axes[1].step(
            result.nis_times,
            np.cumsum(result.rejected, dtype=np.int64),
            where="post",
            color=colors[name],
            lw=0.8,
            label=f"{name}: {int(np.sum(result.rejected))}",
        )
        maximum_rejected = max(
            maximum_rejected,
            int(np.sum(result.rejected)),
        )
    axes[0].axhline(25.0, color="tab:red", ls="--", lw=1.0, label="NIS gate 25")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("NIS (log scale)")
    axes[0].set_title("UWB innovation consistency")
    axes[1].set(xlabel="time [s]", ylabel="cumulative rejected UWB")
    axes[1].set_ylim(0.0, max(1.0, maximum_rejected * 1.05))
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _hash_arrays(arrays):
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(contiguous.dtype.str.encode("ascii"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _accuracy_payload(results):
    return {
        name: {
            **results[name].metrics,
            "rejected_measurements": int(np.sum(results[name].rejected)),
            "total_measurements": int(len(results[name].rejected)),
        }
        for name in VARIANTS
    }


def _equivalence_payload(author_result, accelerated_result):
    """Report direct agreement while excluding unused rotvec history fields."""
    return {
        "max_abs_rotation_matrix": float(
            np.max(
                np.abs(
                    author_result.rotations
                    - accelerated_result.rotations
                )
            )
        ),
        "max_abs_velocity_mps": float(
            np.max(
                np.abs(
                    author_result.states[:, 3:6]
                    - accelerated_result.states[:, 3:6]
                )
            )
        ),
        "max_abs_position_m": float(
            np.max(
                np.abs(
                    author_result.states[:, 6:9]
                    - accelerated_result.states[:, 6:9]
                )
            )
        ),
        "max_abs_bias": float(
            np.max(
                np.abs(
                    author_result.biases
                    - accelerated_result.biases
                )
            )
        ),
        "max_abs_nis": float(
            np.max(np.abs(author_result.nis - accelerated_result.nis))
        ),
        "rejection_decision_changes": int(
            np.sum(
                author_result.rejected
                != accelerated_result.rejected
            )
        ),
    }


def main():
    args = parse_args()
    set_process_affinity(args.cpu_affinity)
    if args.repeats < 2:
        raise ValueError("--repeats must be at least 2")
    if args.warmup_duration <= 0.0:
        raise ValueError("--warmup-duration must be positive")
    print(
        json.dumps(
            {
                "phase": "setup",
                "message": (
                    "Loading data and preparing the single-core benchmark. "
                    "A result line is printed after each replay."
                ),
                "max_duration_seconds": args.max_duration,
                "warmup_duration_seconds": args.warmup_duration,
                "repeats": args.repeats,
                "mkl_threading_layer": os.environ["MKL_THREADING_LAYER"],
                "cpu_affinity": args.cpu_affinity,
                "parallel_batch_jobs": int(
                    os.environ.get("INEKF_PARALLEL_BATCH_JOBS", "1")
                ),
                "output_dir": str(args.output_dir.resolve()),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    anchor_path = Path(args.i[0])
    csv_path = Path(args.i[1])
    author_module = benchmark_base.load_author_module()
    acc_author_module = load_acc_author_module()
    author_source_hash = benchmark_base.source_sha256(
        benchmark_base.AUTHOR_INEKF
    )
    copied_source_hash = benchmark_base.source_sha256(ACC_AUTHOR_INEKF)
    if copied_source_hash != author_source_hash:
        raise AssertionError(
            "The accelerated copy's inekf.py is not byte-identical to "
            "the backup author source. Algorithm/tuning isolation failed."
        )
    dataset = load_dataset(
        anchor_path,
        csv_path,
        max_duration=args.max_duration,
    )
    warmup_limit = min(
        args.warmup_duration,
        float(max(dataset.t_imu[-1], dataset.t_uwb[-1], dataset.t_gt[-1])),
    )
    warmup_dataset = load_dataset(
        anchor_path,
        csv_path,
        max_duration=warmup_limit,
    )
    plan = benchmark_base.build_event_plan(dataset)
    warmup_plan = benchmark_base.build_event_plan(warmup_dataset)
    jit_warmup_seconds = ensure_accelerated_kernel()

    timing_rows = {
        name: {
            "initialization_seconds": [],
            "replay_seconds": [],
        }
        for name in VARIANTS
    }

    warmups = {
        AUTHOR: _author_timing_run(
            warmup_dataset,
            warmup_plan,
            author_module,
            args.yaw_deg,
            args.author_tdoa_variance,
        ),
        ACCELERATED: _acc_timing_run(
            warmup_dataset,
            warmup_plan,
            acc_author_module,
            args.yaw_deg,
            args.author_tdoa_variance,
        ),
    }
    for name in VARIANTS:
        print(
            json.dumps(
                {
                    "phase": "warmup",
                    "variant": name,
                    "duration_seconds": warmup_limit,
                    **warmups[name],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    for repeat in range(args.repeats):
        order = VARIANTS if repeat % 2 == 0 else tuple(reversed(VARIANTS))
        for name in order:
            if name == AUTHOR:
                timing = _author_timing_run(
                    dataset,
                    plan,
                    author_module,
                    args.yaw_deg,
                    args.author_tdoa_variance,
                )
            else:
                timing = _acc_timing_run(
                    dataset,
                    plan,
                    acc_author_module,
                    args.yaw_deg,
                    args.author_tdoa_variance,
                )
            for field in timing_rows[name]:
                timing_rows[name][field].append(timing[field])
            print(
                json.dumps(
                    {
                        "phase": "measured",
                        "repeat": repeat,
                        "variant": name,
                        **timing,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    author_result, author_diagnostics, author_evaluated_timing = (
        _author_evaluated_run(
            dataset,
            plan,
            author_module,
            args.yaw_deg,
            args.author_tdoa_variance,
        )
    )
    acc_result, acc_diagnostics, acc_evaluated_timing = (
        _acc_evaluated_run(
            dataset,
            plan,
            acc_author_module,
            args.yaw_deg,
            args.author_tdoa_variance,
        )
    )
    results = {AUTHOR: author_result, ACCELERATED: acc_result}
    diagnostics = {
        AUTHOR: author_diagnostics,
        ACCELERATED: acc_diagnostics,
    }

    author_replay = np.asarray(timing_rows[AUTHOR]["replay_seconds"])
    accelerated_replay = np.asarray(
        timing_rows[ACCELERATED]["replay_seconds"]
    )
    paired_speedups = author_replay / accelerated_replay
    timing_statistics = {
        name: {
            field: benchmark_base.distribution(values)
            for field, values in fields.items()
        }
        for name, fields in timing_rows.items()
    }

    create_effect_figure(
        dataset,
        results,
        diagnostics,
        timing_rows,
        args.output_dir / "comparison.png",
    )
    create_nis_figure(
        results,
        args.output_dir / "nis_and_rejections.png",
    )
    equivalence = _equivalence_payload(author_result, acc_result)
    if (
        equivalence["max_abs_rotation_matrix"] > 1e-8
        or equivalence["max_abs_velocity_mps"] > 1e-8
        or equivalence["max_abs_position_m"] > 1e-8
        or equivalence["max_abs_bias"] > 1e-8
        or equivalence["max_abs_nis"] > 1e-6
        or equivalence["rejection_decision_changes"] != 0
    ):
        raise AssertionError(
            "Compute-only acceleration changed the author algorithm output: "
            f"{equivalence}"
        )

    payload = {
        "scope": {
            "description": (
                "Untouched paper-author source versus a byte-identical copy "
                "executed by a fixed-size native loop. Initial state, bias, "
                "covariance, noise, gate, prediction, and correction semantics "
                "are unchanged. Loading, construction, JIT warmup, history, "
                "metrics, and plots are outside measured replay."
            ),
            "dataset": str(csv_path.resolve()),
            "anchors": str(anchor_path.resolve()),
            "max_duration": args.max_duration,
            "yaw_degrees": args.yaw_deg,
            "warmup_duration_seconds": warmup_limit,
            "jit_warmup_or_cache_load_seconds": float(jit_warmup_seconds),
            "repeats": args.repeats,
            "thread_limit": 1,
            "cpu_affinity": args.cpu_affinity,
            "parallel_batch_jobs": int(
                os.environ.get("INEKF_PARALLEL_BATCH_JOBS", "1")
            ),
            "history_recorded_in_timing": False,
            "covariance_history_recorded_for_plots": False,
            "predict_calls_per_run": len(dataset.t_imu) - 1,
            "uwb_calls_per_run": len(dataset.t_uwb),
            "events_per_run": len(plan.events),
            "event_plan_sha256": _hash_arrays(_fixed_plan(plan)),
            "tdoa_variance_both_variants": args.author_tdoa_variance,
        },
        "algorithm_settings": {
            "identical_for_both_variants": True,
            "initial_yaw_std_rad": 1.0,
            "initial_roll_pitch_std_rad": 0.01,
            "initial_velocity_std_mps": 0.01,
            "initial_position_std_m": 0.1,
            "initial_accelerometer_bias_std": 0.0001,
            "initial_gyroscope_bias_std": 0.000001,
            "initial_bias": [0.0] * 6,
            "tdoa_variance": args.author_tdoa_variance,
            "gate_mahalanobis_threshold": 5.0,
            "ground_clamp": True,
            "author_bias_layout_preserved": True,
            "author_pre_gate_coordinate_semantics_preserved": True,
            "execution_only_changes": [
                "precomputed integer event lookups",
                "fixed contiguous float64 buffers",
                "Numba nopython event/predict/update loop",
                "exact cubic evaluation of expm(A*dt) because A^4=0",
                "direct expansion of H_left @ Ad(X_hat^-1)",
            ],
        },
        "provenance": {
            "author_repository": str(benchmark_base.AUTHOR_REPOSITORY),
            "author_commit": benchmark_base.AUTHOR_COMMIT,
            "author_source": str(benchmark_base.AUTHOR_INEKF),
            "author_source_sha256": author_source_hash,
            "author_source_modified": False,
            "accelerated_repository": str(REPOSITORY_ROOT),
            "accelerated_copied_author_source": str(ACC_AUTHOR_INEKF),
            "accelerated_copied_author_source_sha256": copied_source_hash,
            "copied_author_source_byte_identical": True,
            "accelerated_variant": ACCELERATED,
            "accelerated_core_source": str(ACCELERATED_CORE),
            "accelerated_core_source_sha256": benchmark_base.source_sha256(
                ACCELERATED_CORE
            ),
            "python": sys.version,
            "platform": benchmark_base.platform.platform(),
            "thread_environment": {
                key: benchmark_base.os.environ[key]
                for key in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                )
            },
        },
        "timing": timing_statistics,
        "speedup": {
            "ratio_of_medians": (
                timing_statistics[AUTHOR]["replay_seconds"]["median"]
                / timing_statistics[ACCELERATED]["replay_seconds"]["median"]
            ),
            "paired": benchmark_base.distribution(paired_speedups),
        },
        "accuracy": _accuracy_payload(results),
        "numerical_equivalence": equivalence,
        "evaluated_runs": {
            AUTHOR: author_evaluated_timing,
            ACCELERATED: acc_evaluated_timing,
        },
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "speedup": payload["speedup"],
                "accuracy": payload["accuracy"],
                "comparison": str(
                    (args.output_dir / "comparison.png").resolve()
                ),
                "nis_and_rejections": str(
                    (args.output_dir / "nis_and_rejections.png").resolve()
                ),
                "metrics": str(
                    (args.output_dir / "metrics.json").resolve()
                ),
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
