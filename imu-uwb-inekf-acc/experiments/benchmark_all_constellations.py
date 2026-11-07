"""Batch full-file author-original versus compute-only acceleration benchmark.

The default representative scope selects the first TDoA2 trial from each of
the four anchor constellations.  ``--all-files`` expands the run to every CSV
under const1--const4.  Each case is executed in a fresh subprocess through
``benchmark_author_original_vs_acc.py`` and therefore inherits its source-hash
guard, single-core numerical-library setup, fair timing, plots, and numerical
equivalence checks.
"""

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import csv
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import time

# Match the single-case benchmark before importing NumPy/Matplotlib.  The
# current Windows MKL/LLVM OpenMP combination can otherwise terminate during
# the final aggregate plot even though every child benchmark has completed.
os.environ["MKL_THREADING_LAYER"] = "SEQUENTIAL"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SINGLE_BENCHMARK = (
    REPOSITORY_ROOT / "experiments" / "benchmark_author_original_vs_acc.py"
)
DEFAULT_DATASET_ROOT = Path(
    r"E:\datasets\dataset\flight-dataset"
)
REPRESENTATIVE_FILES = {
    1: "const1-trial1-tdoa2.csv",
    2: "const2-trial1-tdoa2.csv",
    3: "const3-trial1-tdoa2.csv",
    4: "const4-trial1-tdoa2-traj1.csv",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run complete-file author-original versus compute-only "
            "acceleration comparisons across all four constellations."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
    )
    parser.add_argument(
        "--all-files",
        action="store_true",
        help=(
            "Run every CSV in const1--const4 (79 files) instead of one "
            "consistent representative file per constellation."
        ),
    )
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--warmup-duration", type=float, default=10.0)
    parser.add_argument(
        "--max-duration",
        type=float,
        help=(
            "Optional per-file truncation for parallel smoke tests. "
            "Omit this option for complete-file results."
        ),
    )
    parser.add_argument(
        "--limit-cases",
        type=int,
        help=(
            "Optional deterministic prefix limit for smoke tests. "
            "Omit this option to run the full selected scope."
        ),
    )
    parser.add_argument("--yaw-deg", type=float, default=0.0)
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help=(
            "Number of CSV files to benchmark concurrently. Each child "
            "remains single-threaded."
        ),
    )
    parser.add_argument(
        "--cpu-list",
        help=(
            "Logical CPUs available to workers, e.g. 0-19 or 0,2,4-7. "
            "Defaults to every logical CPU."
        ),
    )
    parser.add_argument(
        "--no-affinity",
        action="store_true",
        help=(
            "Let the OS migrate workers instead of pinning each child to "
            "one logical CPU. Pinning is preferable for paired timing."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT
        / "results"
        / "full_all_constellations",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List resolved cases and commands without executing them.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Reuse a case only when its existing metrics.json parses "
            "successfully; otherwise run or rerun that case."
        ),
    )
    return parser.parse_args()


def discover_cases(dataset_root, all_files):
    survey_root = dataset_root / "survey-results"
    csv_root = dataset_root / "csv-data"
    cases = []
    for constellation in range(1, 5):
        anchor = survey_root / f"anchor_const{constellation}.npz"
        const_dir = csv_root / f"const{constellation}"
        if all_files:
            csv_paths = sorted(const_dir.glob("*.csv"))
        else:
            csv_paths = [
                const_dir / REPRESENTATIVE_FILES[constellation]
            ]
        if not anchor.is_file():
            raise FileNotFoundError(anchor)
        if not csv_paths:
            raise FileNotFoundError(f"No CSV files found under {const_dir}")
        for csv_path in csv_paths:
            if not csv_path.is_file():
                raise FileNotFoundError(csv_path)
            if all_files:
                case_id = (
                    f"const{constellation}__{csv_path.stem}"
                )
            else:
                case_id = f"const{constellation}"
            cases.append(
                {
                    "case_id": case_id,
                    "constellation": constellation,
                    "anchor": anchor.resolve(),
                    "csv": csv_path.resolve(),
                }
            )
    return cases


def parse_cpu_list(specification):
    cpu_count = os.cpu_count() or 1
    if not specification:
        return list(range(cpu_count))
    cpus = []
    for item in specification.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            first_text, last_text = item.split("-", 1)
            first = int(first_text)
            last = int(last_text)
            if last < first:
                raise ValueError(
                    f"Invalid descending CPU range: {item}"
                )
            cpus.extend(range(first, last + 1))
        else:
            cpus.append(int(item))
    if not cpus:
        raise ValueError("--cpu-list resolved to no CPUs")
    if len(set(cpus)) != len(cpus):
        raise ValueError("--cpu-list contains duplicate CPUs")
    invalid = [cpu for cpu in cpus if cpu < 0 or cpu >= cpu_count]
    if invalid:
        raise ValueError(
            f"CPU indices {invalid} are outside [0, {cpu_count - 1}]"
        )
    return cpus


def command_for_case(args, case, cpu_affinity=None):
    case_output = args.output_dir / case["case_id"]
    command = [
        sys.executable,
        "-u",
        str(SINGLE_BENCHMARK),
        "-i",
        str(case["anchor"]),
        str(case["csv"]),
        "--warmup-duration",
        str(args.warmup_duration),
        "--repeats",
        str(args.repeats),
        "--yaw-deg",
        str(args.yaw_deg),
        "--output-dir",
        str(case_output),
    ]
    if cpu_affinity is not None:
        command.extend(["--cpu-affinity", str(cpu_affinity)])
    if args.max_duration is not None:
        command.extend(
            ["--max-duration", str(args.max_duration)]
        )
    return command


def nested(payload, *keys):
    value = payload
    for key in keys:
        value = value[key]
    return value


def row_from_metrics(case, metrics_path):
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    author_accuracy = payload["accuracy"]["author_original"]
    accelerated_accuracy = payload["accuracy"]["author_accelerated"]
    equivalence = payload["numerical_equivalence"]
    return {
        "case_id": case["case_id"],
        "constellation": case["constellation"],
        "csv": str(case["csv"]),
        "events": payload["scope"]["events_per_run"],
        "predict_calls": payload["scope"]["predict_calls_per_run"],
        "uwb_calls": payload["scope"]["uwb_calls_per_run"],
        "author_median_s": nested(
            payload,
            "timing",
            "author_original",
            "replay_seconds",
            "median",
        ),
        "accelerated_median_s": nested(
            payload,
            "timing",
            "author_accelerated",
            "replay_seconds",
            "median",
        ),
        "speedup": payload["speedup"]["ratio_of_medians"],
        "author_position_rmse_m": author_accuracy["position_rmse_m"],
        "accelerated_position_rmse_m": accelerated_accuracy[
            "position_rmse_m"
        ],
        "author_velocity_rmse_mps": author_accuracy[
            "velocity_rmse_mps"
        ],
        "accelerated_velocity_rmse_mps": accelerated_accuracy[
            "velocity_rmse_mps"
        ],
        "author_orientation_rmse_rad": author_accuracy[
            "orientation_geodesic_rmse_rad"
        ],
        "accelerated_orientation_rmse_rad": accelerated_accuracy[
            "orientation_geodesic_rmse_rad"
        ],
        "author_yaw_rmse_rad": author_accuracy["yaw_rmse_rad"],
        "accelerated_yaw_rmse_rad": accelerated_accuracy["yaw_rmse_rad"],
        "author_rejected": author_accuracy["rejected_measurements"],
        "accelerated_rejected": accelerated_accuracy[
            "rejected_measurements"
        ],
        "total_measurements": author_accuracy["total_measurements"],
        "max_rotation_matrix_difference": equivalence[
            "max_abs_rotation_matrix"
        ],
        "max_velocity_difference": equivalence["max_abs_velocity_mps"],
        "max_position_difference": equivalence["max_abs_position_m"],
        "max_bias_difference": equivalence["max_abs_bias"],
        "max_nis_difference": equivalence["max_abs_nis"],
        "rejection_decision_changes": equivalence[
            "rejection_decision_changes"
        ],
        "cpu_affinity": payload["scope"].get("cpu_affinity"),
        "parallel_batch_jobs": payload["scope"].get(
            "parallel_batch_jobs",
            1,
        ),
        "case_wall_seconds": None,
    }


def write_csv(rows, destination):
    fieldnames = list(rows[0])
    with destination.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(rows, destination):
    if len(rows) > 20:
        plot_large_summary(rows, destination)
        return
    labels = [row["case_id"] for row in rows]
    x = np.arange(len(rows), dtype=float)
    width = 0.38
    author = np.array([row["author_median_s"] for row in rows])
    accelerated = np.array(
        [row["accelerated_median_s"] for row in rows]
    )
    speedup = np.array([row["speedup"] for row in rows])
    position = np.array(
        [row["author_position_rmse_m"] for row in rows]
    )
    yaw_deg = np.degrees(
        np.array([row["author_yaw_rmse_rad"] for row in rows])
    )

    figure, axes = plt.subplots(2, 2, figsize=(14, 9))
    runtime_axis, speed_axis, position_axis, yaw_axis = axes.ravel()

    runtime_axis.bar(
        x - width / 2,
        author,
        width,
        label="author_original",
    )
    runtime_axis.bar(
        x + width / 2,
        accelerated,
        width,
        label="author_accelerated",
    )
    runtime_axis.set_yscale("log")
    runtime_axis.set_ylabel("median replay seconds (log)")
    runtime_axis.set_title("Complete-file replay time")
    runtime_axis.legend()

    speed_axis.bar(x, speedup, color="tab:green")
    speed_axis.set_ylabel("speedup [x]")
    speed_axis.set_title("Ratio of replay medians")
    for index, value in enumerate(speedup):
        speed_axis.text(
            index,
            value,
            f"{value:.1f}x",
            ha="center",
            va="bottom",
        )

    position_axis.bar(x, position, color="tab:blue")
    position_axis.set_ylabel("position RMSE [m]")
    position_axis.set_title(
        "Position RMSE (both implementations overlap)"
    )

    yaw_axis.bar(x, yaw_deg, color="tab:orange")
    yaw_axis.set_ylabel("yaw RMSE [deg]")
    yaw_axis.set_title("Yaw RMSE (both implementations overlap)")

    for axis in axes.ravel():
        axis.set_xticks(x)
        axis.set_xticklabels(labels, rotation=30, ha="right")
        axis.grid(True, axis="y", alpha=0.3)

    figure.suptitle(
        "Untouched author InEKF vs compute-only acceleration: "
        "complete files",
        fontsize=15,
    )
    figure.tight_layout()
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def plot_large_summary(rows, destination):
    """Create a readable distribution summary for the 79-file scope."""
    author = np.array([row["author_median_s"] for row in rows])
    accelerated = np.array(
        [row["accelerated_median_s"] for row in rows]
    )
    speedup = np.array([row["speedup"] for row in rows])
    constellations = sorted(
        {int(row["constellation"]) for row in rows}
    )
    colors = {
        1: "tab:blue",
        2: "tab:orange",
        3: "tab:green",
        4: "tab:red",
    }

    figure, axes = plt.subplots(2, 2, figsize=(14, 9))
    runtime_axis, speed_axis, position_axis, yaw_axis = axes.ravel()

    for constellation in constellations:
        selected = np.array(
            [
                int(row["constellation"]) == constellation
                for row in rows
            ]
        )
        runtime_axis.scatter(
            author[selected],
            accelerated[selected],
            s=28,
            alpha=0.75,
            color=colors[constellation],
            label=f"const{constellation}",
        )
    runtime_axis.set_xscale("log")
    runtime_axis.set_yscale("log")
    runtime_axis.set_xlabel("author median replay [s]")
    runtime_axis.set_ylabel("accelerated median replay [s]")
    runtime_axis.set_title("Per-file replay time")
    runtime_axis.legend()

    speed_axis.hist(
        speedup,
        bins=min(20, max(6, len(rows) // 4)),
        color="tab:green",
        edgecolor="white",
    )
    speed_axis.axvline(
        float(np.median(speedup)),
        color="black",
        linestyle="--",
        label=f"median {np.median(speedup):.1f}x",
    )
    speed_axis.set_xlabel("speedup [x]")
    speed_axis.set_ylabel("file count")
    speed_axis.set_title("Speedup distribution")
    speed_axis.legend()

    position_groups = [
        [
            row["author_position_rmse_m"]
            for row in rows
            if int(row["constellation"]) == constellation
        ]
        for constellation in constellations
    ]
    yaw_groups = [
        [
            np.degrees(row["author_yaw_rmse_rad"])
            for row in rows
            if int(row["constellation"]) == constellation
        ]
        for constellation in constellations
    ]
    labels = [f"const{value}" for value in constellations]
    position_axis.boxplot(position_groups, tick_labels=labels)
    position_axis.set_ylabel("position RMSE [m]")
    position_axis.set_title("Position RMSE distribution")
    yaw_axis.boxplot(yaw_groups, tick_labels=labels)
    yaw_axis.set_ylabel("yaw RMSE [deg]")
    yaw_axis.set_title("Yaw RMSE distribution")

    for axis in axes.ravel():
        axis.grid(True, alpha=0.3)
    figure.suptitle(
        "Untouched author InEKF vs compute-only acceleration: "
        f"{len(rows)} complete files",
        fontsize=15,
    )
    figure.tight_layout()
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def prewarm_accelerated_cache():
    """Compile/load the Numba signature once before parallel child startup."""
    estimation_dir = REPOSITORY_ROOT / "scripts" / "estimation"
    code = (
        "import sys; "
        f"sys.path.insert(0, {str(estimation_dir)!r}); "
        "from accelerated_replay import ensure_accelerated_kernel; "
        "print(ensure_accelerated_kernel())"
    )
    started = time.perf_counter()
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPOSITORY_ROOT,
        env=os.environ.copy(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    elapsed = time.perf_counter() - started
    if completed.returncode != 0:
        raise RuntimeError(
            "Parallel Numba cache prewarm failed: "
            + completed.stderr.strip()
        )
    return elapsed


def run_case(args, index, total, case, cpu_queue):
    cpu_affinity = cpu_queue.get()
    started = time.perf_counter()
    case_output = args.output_dir / case["case_id"]
    case_output.mkdir(parents=True, exist_ok=True)
    stdout_path = case_output / "runner_stdout.log"
    stderr_path = case_output / "runner_stderr.log"
    print(
        json.dumps(
            {
                "phase": "case_start",
                "index": index,
                "total": total,
                "case": case["case_id"],
                "cpu_affinity": cpu_affinity,
                "csv": str(case["csv"]),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    environment = os.environ.copy()
    environment["INEKF_PARALLEL_BATCH_JOBS"] = str(args.jobs)
    try:
        with (
            stdout_path.open("w", encoding="utf-8") as stdout_handle,
            stderr_path.open("w", encoding="utf-8") as stderr_handle,
        ):
            completed = subprocess.run(
                command_for_case(args, case, cpu_affinity),
                cwd=REPOSITORY_ROOT,
                env=environment,
                stdout=stdout_handle,
                stderr=stderr_handle,
                check=False,
            )
        elapsed = time.perf_counter() - started
        metrics_path = case_output / "metrics.json"
        if completed.returncode != 0 or not metrics_path.is_file():
            return {
                "success": False,
                "case_id": case["case_id"],
                "returncode": completed.returncode,
                "cpu_affinity": cpu_affinity,
                "case_wall_seconds": elapsed,
                "stderr": str(stderr_path),
            }
        row = row_from_metrics(case, metrics_path)
        row["cpu_affinity"] = cpu_affinity
        row["parallel_batch_jobs"] = args.jobs
        row["case_wall_seconds"] = elapsed
        return {
            "success": True,
            "row": row,
        }
    except Exception as error:
        return {
            "success": False,
            "case_id": case["case_id"],
            "returncode": None,
            "cpu_affinity": cpu_affinity,
            "case_wall_seconds": time.perf_counter() - started,
            "error": repr(error),
            "stderr": str(stderr_path),
        }
    finally:
        cpu_queue.put(cpu_affinity)


def main():
    args = parse_args()
    if args.repeats < 2:
        raise ValueError("--repeats must be at least 2")
    if args.warmup_duration <= 0.0:
        raise ValueError("--warmup-duration must be positive")
    if args.jobs < 1:
        raise ValueError("--jobs must be positive")
    if args.max_duration is not None and args.max_duration <= 0.0:
        raise ValueError("--max-duration must be positive")
    if args.limit_cases is not None and args.limit_cases < 1:
        raise ValueError("--limit-cases must be positive")
    available_cpus = parse_cpu_list(args.cpu_list)
    if args.jobs > len(available_cpus):
        raise ValueError(
            f"--jobs {args.jobs} exceeds the {len(available_cpus)} "
            "CPUs selected by --cpu-list"
        )
    worker_cpus = (
        [None] * args.jobs
        if args.no_affinity
        else available_cpus[: args.jobs]
    )
    cases = discover_cases(args.dataset_root.resolve(), args.all_files)
    if args.limit_cases is not None:
        cases = cases[: args.limit_cases]
    args.output_dir = args.output_dir.resolve()
    batch_started = time.perf_counter()

    print(
        json.dumps(
            {
                "phase": "batch_setup",
                "scope": "all_files" if args.all_files else "representative",
                "case_count": len(cases),
                "complete_file_replay": args.max_duration is None,
                "max_duration_seconds": args.max_duration,
                "repeats": args.repeats,
                "warmup_duration_seconds": args.warmup_duration,
                "jobs": args.jobs,
                "cpu_affinity_enabled": not args.no_affinity,
                "worker_cpus": worker_cpus,
                "timing_advisory": (
                    "Parallel runs measure loaded-system throughput. "
                    "Use --jobs 1 for publication-grade isolated timing."
                    if args.jobs > 1
                    else "Isolated single-process timing."
                ),
                "output_dir": str(args.output_dir),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    if args.dry_run:
        for index, case in enumerate(cases):
            print(
                json.dumps(
                    {
                        "case": case["case_id"],
                        "anchor": str(case["anchor"]),
                        "csv": str(case["csv"]),
                        "command": command_for_case(
                            args,
                            case,
                            worker_cpus[index % len(worker_cpus)],
                        ),
                    },
                    ensure_ascii=False,
                )
            )
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    failures = []
    pending_cases = []
    reused_cases = 0
    for index, case in enumerate(cases, start=1):
        metrics_path = (
            args.output_dir / case["case_id"] / "metrics.json"
        )
        if args.resume and metrics_path.is_file():
            try:
                row = row_from_metrics(case, metrics_path)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                print(
                    json.dumps(
                        {
                            "phase": "resume_invalid",
                            "case": case["case_id"],
                            "metrics": str(metrics_path),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            else:
                rows.append(row)
                reused_cases += 1
                print(
                    json.dumps(
                        {
                            "phase": "case_reused",
                            "index": index,
                            "total": len(cases),
                            "case": case["case_id"],
                            "speedup": row["speedup"],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                continue
        pending_cases.append((index, case))

    cpu_queue = queue.Queue()
    for cpu in worker_cpus:
        cpu_queue.put(cpu)
    cache_prewarm_seconds = 0.0
    if pending_cases:
        cache_prewarm_seconds = prewarm_accelerated_cache()
        print(
            json.dumps(
                {
                    "phase": "cache_prewarm",
                    "seconds": cache_prewarm_seconds,
                }
            ),
            flush=True,
        )
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        outstanding = {
            executor.submit(
                run_case,
                args,
                index,
                len(cases),
                case,
                cpu_queue,
            )
            for index, case in pending_cases
        }
        while outstanding:
            completed_futures, outstanding = wait(
                outstanding,
                timeout=30.0,
                return_when=FIRST_COMPLETED,
            )
            if not completed_futures:
                print(
                    json.dumps(
                        {
                            "phase": "heartbeat",
                            "completed": len(rows),
                            "failed": len(failures),
                            "remaining": len(outstanding),
                            "elapsed_seconds": (
                                time.perf_counter() - batch_started
                            ),
                        }
                    ),
                    flush=True,
                )
                continue
            for future in completed_futures:
                outcome = future.result()
                if not outcome["success"]:
                    failure = {
                        key: value
                        for key, value in outcome.items()
                        if key != "success"
                    }
                    failures.append(failure)
                    print(
                        json.dumps(
                            {
                                "phase": "case_failed",
                                **failure,
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                    continue
                row = outcome["row"]
                rows.append(row)
                print(
                    json.dumps(
                        {
                            "phase": "case_complete",
                            "case": row["case_id"],
                            "cpu_affinity": row["cpu_affinity"],
                            "case_wall_seconds": row[
                                "case_wall_seconds"
                            ],
                            "author_median_s": row[
                                "author_median_s"
                            ],
                            "accelerated_median_s": row[
                                "accelerated_median_s"
                            ],
                            "speedup": row["speedup"],
                            "rejection_decision_changes": row[
                                "rejection_decision_changes"
                            ],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

    case_order = {
        case["case_id"]: index for index, case in enumerate(cases)
    }
    rows.sort(key=lambda row: case_order[row["case_id"]])
    batch_wall_seconds = time.perf_counter() - batch_started
    executed_cases_completed = len(rows) - reused_cases
    summary = {
        "scope": "all_files" if args.all_files else "representative",
        "complete_file_replay": args.max_duration is None,
        "max_duration_seconds": args.max_duration,
        "repeats": args.repeats,
        "warmup_duration_seconds": args.warmup_duration,
        "parallel_jobs": args.jobs,
        "cpu_affinity_enabled": not args.no_affinity,
        "worker_cpus": worker_cpus,
        "cache_prewarm_seconds": cache_prewarm_seconds,
        "batch_wall_seconds": batch_wall_seconds,
        "throughput_cases_per_hour": (
            executed_cases_completed * 3600.0 / batch_wall_seconds
            if batch_wall_seconds > 0.0 and executed_cases_completed > 0
            else None
        ),
        "timing_advisory": (
            "Parallel loaded-system timing; use --jobs 1 for isolated "
            "publication-grade absolute timing."
            if args.jobs > 1
            else "Isolated single-process timing."
        ),
        "cases_requested": len(cases),
        "cases_completed": len(rows),
        "cases_reused": reused_cases,
        "cases_executed_completed": executed_cases_completed,
        "failures": failures,
        "rows": rows,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if rows:
        write_csv(rows, args.output_dir / "summary.csv")
        plot_summary(rows, args.output_dir / "summary.png")
    print(
        json.dumps(
            {
                "phase": "batch_complete",
                "completed": len(rows),
                "failed": len(failures),
                "batch_wall_seconds": batch_wall_seconds,
                "summary": str(args.output_dir / "summary.json"),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
