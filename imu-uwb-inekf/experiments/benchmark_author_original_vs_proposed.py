"""Fair, single-filter benchmark of the author's InEKF and ``proposed``.

The author's backup is imported read-only.  Both filters receive the same
precomputed event plan and exactly the same IMU/UWB calls.  Dataset loading,
filter construction, metric evaluation, and plotting are outside the timed
replay.  Timing runs do not retain histories; one separate evaluated run per
variant records a common, minimal history for accuracy plots and metrics.
"""

import argparse
from dataclasses import dataclass
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import time

# Force deterministic BLAS defaults before importing NumPy/SciPy.  Dynamic
# threadpool limiting is intentionally avoided because this environment loads
# both Intel and LLVM OpenMP runtimes.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPOSITORY_ROOT.parent
ESTIMATION_DIR = REPOSITORY_ROOT / "scripts" / "estimation"
AUTHOR_REPOSITORY = WORKSPACE_ROOT / "imu-uwb-inekf - backup"
AUTHOR_INEKF = AUTHOR_REPOSITORY / "scripts" / "estimation" / "inekf.py"
AUTHOR_COMMIT = "f3b664e0e8c86b62c6ed2ea2a112261810228e1c"

sys.path.insert(0, str(ESTIMATION_DIR))

from inekf import FilterConfig, SO3_exp  # noqa: E402
from pipeline import (  # noqa: E402
    EstimationResult,
    _initial_filter,
    _stationary_flags,
    compute_metrics,
    load_dataset,
)


@dataclass(frozen=True)
class EventPlan:
    events: np.ndarray
    imu_index_at_event: np.ndarray
    uwb_start_at_event: np.ndarray
    uwb_stop_at_event: np.ndarray
    stationary: np.ndarray


class AuthorAdapter:
    name = "author_original"
    supports_stationary = False

    def __init__(self, estimator, author_module, collect_diagnostics=False):
        self.estimator = estimator
        self.author_module = author_module
        self.collect_diagnostics = collect_diagnostics
        self.last_nis = math.nan
        self.stationary_update_count = 0

    @property
    def X(self):
        return self.estimator.X

    @property
    def P(self):
        return self.estimator.P

    @property
    def bias(self):
        return self.estimator.bias

    def predict(self, imu, dt):
        self.estimator.predict(imu, dt)

    def correct(self, uwb, anchors):
        if self.collect_diagnostics:
            self.last_nis = self._pre_gate_nis(uwb, anchors)
        _, _, _, rejected = self.estimator.correct(uwb, anchors)
        return bool(rejected)

    def _pre_gate_nis(self, uwb, anchors):
        X = self.estimator.X_hat
        anchor_a = anchors[int(uwb[0])]
        anchor_b = anchors[int(uwb[1])]
        tag = X[:3, 4] + X[:3, :3] @ self.estimator.t_uv
        distance_a = np.linalg.norm(anchor_a - tag)
        distance_b = np.linalg.norm(anchor_b - tag)
        prediction = distance_b - distance_a
        residual = float(uwb[2]) - prediction
        H = np.zeros((1, 15))
        H[0, 6:9] = (
            (anchor_b - tag) / distance_b
            - (anchor_a - tag) / distance_a
        )
        variance = float(
            (H @ self.estimator.P_hat @ H.T)[0, 0]
            + self.estimator.w_tdoa[0] ** 2
        )
        return residual * residual / variance


class ProposedAdapter:
    name = "proposed"
    supports_stationary = True

    def __init__(self, estimator):
        self.estimator = estimator

    @property
    def X(self):
        return self.estimator.X

    @property
    def P(self):
        return self.estimator.P

    @property
    def bias(self):
        return self.estimator.bias

    @property
    def last_nis(self):
        return self.estimator.last_nis

    @property
    def stationary_update_count(self):
        return self.estimator.stationary_update_count

    def predict(self, imu, dt):
        self.estimator.predict(imu, dt)

    def correct(self, uwb, anchors):
        _, _, _, rejected = self.estimator.correct(uwb, anchors)
        return bool(rejected)

    def correct_stationary(self, gyroscope):
        self.estimator.correct_stationary(gyroscope)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        nargs=2,
        required=True,
        metavar=("ANCHORS", "CSV"),
    )
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--warmup-duration", type=float, default=30.0)
    parser.add_argument("--max-duration", type=float)
    parser.add_argument("--yaw-deg", type=float, default=0.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT
        / "results"
        / "fair_author_original_vs_proposed",
    )
    parser.add_argument(
        "--author-tdoa-variance",
        type=float,
        default=0.13,
        help="author main.py uses 0.13 for every constellation",
    )
    return parser.parse_args()


def load_author_module():
    if not AUTHOR_INEKF.exists():
        raise FileNotFoundError(f"Author source not found: {AUTHOR_INEKF}")
    spec = importlib.util.spec_from_file_location(
        "author_original_inekf",
        AUTHOR_INEKF,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_event_plan(dataset):
    events = np.unique(np.concatenate((dataset.t_imu, dataset.t_uwb)))
    event_lookup = {float(timestamp): index for index, timestamp in enumerate(events)}
    imu_index = np.full(len(events), -1, dtype=np.int64)
    for index, timestamp in enumerate(dataset.t_imu):
        imu_index[event_lookup[float(timestamp)]] = index

    grouped_uwb = {}
    for index, timestamp in enumerate(dataset.t_uwb):
        grouped_uwb.setdefault(float(timestamp), []).append(index)
    ordered_uwb = []
    uwb_start = np.zeros(len(events), dtype=np.int64)
    uwb_stop = np.zeros(len(events), dtype=np.int64)
    for event_index, timestamp in enumerate(events):
        uwb_start[event_index] = len(ordered_uwb)
        ordered_uwb.extend(grouped_uwb.get(float(timestamp), ()))
        uwb_stop[event_index] = len(ordered_uwb)

    # Reorder the dataset's UWB stream once so each event owns a contiguous
    # interval.  load_dataset already sorts it, and this assertion protects
    # against accidentally timing different calls.
    ordered_uwb = np.asarray(ordered_uwb, dtype=np.int64)
    np.testing.assert_array_equal(ordered_uwb, np.arange(len(dataset.t_uwb)))
    return EventPlan(
        events=events,
        imu_index_at_event=imu_index,
        uwb_start_at_event=uwb_start,
        uwb_stop_at_event=uwb_stop,
        stationary=_stationary_flags(dataset),
    )


def make_author_adapter(
    dataset,
    author_module,
    yaw_degrees,
    tdoa_variance,
    collect_diagnostics=False,
):
    rotation_gt = Rotation.from_quat(dataset.q_gt_xyzw[0]).as_matrix()
    orientation = author_module.SO3_log(rotation_gt)
    orientation = orientation + np.array(
        [0.0, 0.0, math.radians(float(yaw_degrees))]
    )
    X0 = np.eye(5)
    X0[:3, :3] = author_module.SO3_exp(orientation)
    X0[:3, 4] = dataset.p_gt[0]
    P0 = np.diag(
        [0.01**2, 0.01**2, 1.0**2]
        + [0.01**2] * 3
        + [0.1**2] * 3
        + [0.0001**2] * 3
        + [0.000001**2] * 3
    )
    estimator = author_module.InEKF(X0, P0, np.zeros(6))
    estimator.w_tdoa = np.array([math.sqrt(float(tdoa_variance))])
    return AuthorAdapter(
        estimator,
        author_module,
        collect_diagnostics=collect_diagnostics,
    )


def make_proposed_adapter(dataset, yaw_degrees):
    estimator = _initial_filter(
        dataset,
        yaw_degrees,
        FilterConfig.proposed(),
    )
    return ProposedAdapter(estimator)


def make_adapter(
    name,
    dataset,
    author_module,
    yaw_degrees,
    tdoa_variance,
    collect_diagnostics=False,
):
    if name == "author_original":
        return make_author_adapter(
            dataset,
            author_module,
            yaw_degrees,
            tdoa_variance,
            collect_diagnostics=collect_diagnostics,
        )
    if name == "proposed":
        return make_proposed_adapter(dataset, yaw_degrees)
    raise ValueError(f"Unknown benchmark variant: {name}")


def _append_common_history(adapter, timestamp, output):
    state = np.zeros(9)
    state[3:6] = adapter.X[:3, 3]
    state[6:9] = adapter.X[:3, 4]
    output["times"].append(float(timestamp))
    output["states"].append(state)
    output["rotations"].append(adapter.X[:3, :3].copy())
    output["biases"].append(adapter.bias.copy())


def replay_common(dataset, plan, adapter, record_history=False):
    """Replay a precomputed common schedule and return disjoint timing scopes."""
    output = {
        "times": [],
        "states": [],
        "rotations": [],
        "biases": [],
        "nis_times": [],
        "nis": [],
        "rejected": [],
    }
    predict_calls = 0
    uwb_calls = 0
    stationary_calls = 0
    kernel_ns = 0
    last_imu_index = 0
    last_stationary_update = -math.inf
    initial_rest_seen = False
    initial_rest_complete = False

    start_replay = time.perf_counter_ns()
    initial_event_time = float(plan.events[0])
    initial_start = int(plan.uwb_start_at_event[0])
    initial_stop = int(plan.uwb_stop_at_event[0])
    for uwb_index in range(initial_start, initial_stop):
        start_kernel = time.perf_counter_ns()
        rejected = adapter.correct(dataset.uwb[uwb_index], dataset.anchors)
        kernel_ns += time.perf_counter_ns() - start_kernel
        uwb_calls += 1
        if record_history:
            output["nis_times"].append(initial_event_time)
            output["nis"].append(adapter.last_nis)
            output["rejected"].append(rejected)

    if record_history:
        _append_common_history(adapter, initial_event_time, output)

    for event_index in range(1, len(plan.events)):
        event_time = float(plan.events[event_index])
        imu_index = int(plan.imu_index_at_event[event_index])
        if imu_index >= 0:
            if imu_index > last_imu_index:
                dt = event_time - float(dataset.t_imu[last_imu_index])
                start_kernel = time.perf_counter_ns()
                adapter.predict(dataset.imu[last_imu_index], dt)
                kernel_ns += time.perf_counter_ns() - start_kernel
                predict_calls += 1
                last_imu_index = imu_index
            if adapter.supports_stationary:
                if (
                    plan.stationary[imu_index]
                    and not initial_rest_complete
                    and event_time - last_stationary_update >= 0.10
                ):
                    start_kernel = time.perf_counter_ns()
                    adapter.correct_stationary(dataset.imu[imu_index, 3:6])
                    kernel_ns += time.perf_counter_ns() - start_kernel
                    stationary_calls += 1
                    last_stationary_update = event_time
                    initial_rest_seen = True
                elif initial_rest_seen and not plan.stationary[imu_index]:
                    initial_rest_complete = True

        start = int(plan.uwb_start_at_event[event_index])
        stop = int(plan.uwb_stop_at_event[event_index])
        for uwb_index in range(start, stop):
            start_kernel = time.perf_counter_ns()
            rejected = adapter.correct(dataset.uwb[uwb_index], dataset.anchors)
            kernel_ns += time.perf_counter_ns() - start_kernel
            uwb_calls += 1
            if record_history:
                output["nis_times"].append(event_time)
                output["nis"].append(adapter.last_nis)
                output["rejected"].append(rejected)

        if record_history:
            _append_common_history(adapter, event_time, output)
    replay_ns = time.perf_counter_ns() - start_replay

    expected_predict = len(dataset.t_imu) - 1
    if predict_calls != expected_predict:
        raise AssertionError(
            f"Expected {expected_predict} predicts, got {predict_calls}"
        )
    if uwb_calls != len(dataset.t_uwb):
        raise AssertionError(f"Expected {len(dataset.t_uwb)} UWB calls, got {uwb_calls}")
    return {
        "replay_seconds": replay_ns * 1e-9,
        "kernel_seconds": kernel_ns * 1e-9,
        "scheduler_seconds": (replay_ns - kernel_ns) * 1e-9,
        "predict_calls": predict_calls,
        "uwb_calls": uwb_calls,
        "stationary_calls": stationary_calls,
        "output": output,
    }


def timing_run(
    name,
    dataset,
    plan,
    author_module,
    yaw_degrees,
    tdoa_variance,
):
    start_initialization = time.perf_counter_ns()
    adapter = make_adapter(
        name,
        dataset,
        author_module,
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
        timing = replay_common(dataset, plan, adapter, record_history=False)
    finally:
        if gc_enabled:
            gc.enable()
    timing["initialization_seconds"] = initialization_seconds
    return timing


def evaluated_run(
    name,
    dataset,
    plan,
    author_module,
    yaw_degrees,
    tdoa_variance,
):
    adapter = make_adapter(
        name,
        dataset,
        author_module,
        yaw_degrees,
        tdoa_variance,
        collect_diagnostics=True,
    )
    timing = replay_common(dataset, plan, adapter, record_history=True)
    output = timing.pop("output")
    result = EstimationResult(
        variant=name,
        times=np.asarray(output["times"]),
        states=np.asarray(output["states"]),
        rotations=np.asarray(output["rotations"]),
        covariances=np.empty((0, 15, 15)),
        biases=np.asarray(output["biases"]),
        nis_times=np.asarray(output["nis_times"]),
        nis=np.asarray(output["nis"]),
        rejected=np.asarray(output["rejected"], dtype=bool),
        runtime_seconds=timing["replay_seconds"],
        stationary_updates=adapter.stationary_update_count,
        metrics={},
    )
    diagnostics = compute_metrics(dataset, result)
    return result, diagnostics, timing


def distribution(values):
    values = np.asarray(values, dtype=float)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return {
        "runs": [float(value) for value in values],
        "median": median,
        "mad": mad,
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "p95": float(np.percentile(values, 95)),
        "mean": float(np.mean(values)),
        "sample_stdev": (
            float(statistics.stdev(values)) if len(values) > 1 else 0.0
        ),
    }


def source_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def create_figure(dataset, results, diagnostics, timing_rows, output_path):
    figure, axes = plt.subplots(2, 2, figsize=(12.5, 9.0))
    colors = {"author_original": "tab:blue", "proposed": "tab:orange"}
    gt_sample = slice(None, None, max(1, len(dataset.t_gt) // 6000))
    axes[0, 0].plot(
        dataset.p_gt[gt_sample, 0],
        dataset.p_gt[gt_sample, 1],
        "k--",
        lw=1.1,
        label="Vicon",
    )
    for name, result in results.items():
        sample = slice(None, None, max(1, len(result.times) // 6000))
        axes[0, 0].plot(
            result.states[sample, 6],
            result.states[sample, 7],
            color=colors[name],
            lw=0.8,
            label=name,
        )
        diagnostic = diagnostics[name]
        diagnostic_sample = slice(
            None,
            None,
            max(1, len(diagnostic["times"]) // 6000),
        )
        axes[0, 1].plot(
            diagnostic["times"][diagnostic_sample],
            np.linalg.norm(
                diagnostic["position_error"][diagnostic_sample],
                axis=1,
            ),
            color=colors[name],
            lw=0.75,
            label=name,
        )
        axes[1, 0].plot(
            diagnostic["times"][diagnostic_sample],
            np.degrees(
                np.abs(diagnostic["yaw_error"][diagnostic_sample])
            ),
            color=colors[name],
            lw=0.75,
            label=name,
        )

    names = ("author_original", "proposed")
    x = np.arange(len(names))
    replay_medians = [
        np.median(timing_rows[name]["replay_seconds"]) for name in names
    ]
    kernel_medians = [
        np.median(timing_rows[name]["kernel_seconds"]) for name in names
    ]
    width = 0.34
    axes[1, 1].bar(
        x - width / 2,
        replay_medians,
        width,
        label="common replay",
    )
    axes[1, 1].bar(
        x + width / 2,
        kernel_medians,
        width,
        label="predict+correct kernel",
    )
    axes[1, 1].set_xticks(x, names)
    axes[1, 1].set_ylabel("median seconds")
    axes[1, 1].set_title("Fair full-file timing")
    axes[1, 1].legend()
    axes[1, 1].grid(axis="y", alpha=0.25)

    axes[0, 0].set(xlabel="x [m]", ylabel="y [m]", title="XY trajectory")
    axes[0, 0].axis("equal")
    axes[0, 1].set(
        xlabel="time [s]",
        ylabel="position error [m]",
        title="Position error",
    )
    axes[1, 0].set(
        xlabel="time [s]",
        ylabel="absolute yaw error [deg]",
        title="Yaw error",
    )
    for axis in axes.flat[:3]:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main():
    args = parse_args()
    if args.repeats < 2:
        raise ValueError("--repeats must be at least 2")
    if args.warmup_duration <= 0.0:
        raise ValueError("--warmup-duration must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    anchor_path = Path(args.i[0])
    csv_path = Path(args.i[1])
    author_module = load_author_module()
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
    plan = build_event_plan(dataset)
    warmup_plan = build_event_plan(warmup_dataset)
    variants = ("author_original", "proposed")
    timing_rows = {
        name: {
            "initialization_seconds": [],
            "replay_seconds": [],
            "kernel_seconds": [],
            "scheduler_seconds": [],
        }
        for name in variants
    }

    for name in variants:
        warmup = timing_run(
            name,
            warmup_dataset,
            warmup_plan,
            author_module,
            args.yaw_deg,
            args.author_tdoa_variance,
        )
        print(
            json.dumps(
                {
                    "phase": "warmup",
                    "variant": name,
                    "duration_seconds": warmup_limit,
                    **{key: value for key, value in warmup.items() if key != "output"},
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    for repeat in range(args.repeats):
        order = variants if repeat % 2 == 0 else tuple(reversed(variants))
        for name in order:
            timing = timing_run(
                name,
                dataset,
                plan,
                author_module,
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
                        **{key: value for key, value in timing.items() if key != "output"},
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    results = {}
    diagnostics = {}
    evaluated_timing = {}
    for name in variants:
        result, diagnostic, timing = evaluated_run(
            name,
            dataset,
            plan,
            author_module,
            args.yaw_deg,
            args.author_tdoa_variance,
        )
        results[name] = result
        diagnostics[name] = diagnostic
        evaluated_timing[name] = timing

    statistics_payload = {
        name: {
            field: distribution(values)
            for field, values in fields.items()
        }
        for name, fields in timing_rows.items()
    }
    paired_replay_speedups = np.asarray(
        timing_rows["author_original"]["replay_seconds"]
    ) / np.asarray(timing_rows["proposed"]["replay_seconds"])
    paired_kernel_speedups = np.asarray(
        timing_rows["author_original"]["kernel_seconds"]
    ) / np.asarray(timing_rows["proposed"]["kernel_seconds"])

    create_figure(
        dataset,
        results,
        diagnostics,
        timing_rows,
        args.output_dir / "comparison.png",
    )
    payload = {
        "scope": {
            "description": (
                "Single-filter, common-schedule comparison. Loading, filter "
                "construction, history recording, metrics, and plotting are "
                "outside measured replay timings."
            ),
            "dataset": str(csv_path.resolve()),
            "anchors": str(anchor_path.resolve()),
            "max_duration": args.max_duration,
            "yaw_degrees": args.yaw_deg,
            "warmup_duration_seconds": warmup_limit,
            "repeats": args.repeats,
            "thread_limit": 1,
            "predict_calls_per_run": len(dataset.t_imu) - 1,
            "uwb_calls_per_run": len(dataset.t_uwb),
            "events_per_run": len(plan.events),
        },
        "provenance": {
            "author_repository": str(AUTHOR_REPOSITORY),
            "author_commit": AUTHOR_COMMIT,
            "author_source": str(AUTHOR_INEKF),
            "author_source_sha256": source_sha256(AUTHOR_INEKF),
            "author_source_modified": False,
            "python": sys.version,
            "platform": platform.platform(),
            "thread_environment": {
                key: os.environ[key]
                for key in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                )
            },
        },
        "timing": statistics_payload,
        "paired_speedup": {
            "replay": distribution(paired_replay_speedups),
            "kernel": distribution(paired_kernel_speedups),
        },
        "accuracy": {
            name: {
                **result.metrics,
                "evaluated_common_replay_seconds": evaluated_timing[name][
                    "replay_seconds"
                ],
            }
            for name, result in results.items()
        },
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "paired_speedup": payload["paired_speedup"],
                "accuracy": payload["accuracy"],
                "output_dir": str(args.output_dir.resolve()),
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
