"""Paired validation of the task03 fixed-size compiled replay kernel."""

import argparse
import gc
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ESTIMATION_DIR = REPOSITORY_ROOT / "scripts" / "estimation"
sys.path.insert(0, str(ESTIMATION_DIR))

from compiled_replay import ensure_compiled_kernel  # noqa: E402
from pipeline import compute_metrics, load_dataset, run_dataset  # noqa: E402


CASES = (
    {
        "name": "const1_auto_yaw0_30s",
        "constellation": "const1",
        "csv": "const1-trial1-tdoa2.csv",
        "duration": 30.0,
        "yaw": 0.0,
    },
    {
        "name": "const4_manual_yaw90_60s",
        "constellation": "const4",
        "csv": "const4-trial7-tdoa2-manual1.csv",
        "duration": 60.0,
        "yaw": 90.0,
    },
    {
        "name": "const3_fast_yaw180_60s",
        "constellation": "const3",
        "csv": "const3-trial7-tdoa3-manual3.csv",
        "duration": 60.0,
        "yaw": 180.0,
    },
)
VARIANTS = ("proposed", "task03_compiled")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(r"E:\datasets\dataset\flight-dataset"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "results" / "task03_compiled_optimization",
    )
    parser.add_argument("--repeats", type=int, default=4)
    return parser.parse_args()


def _paths(dataset_root, case):
    return (
        dataset_root
        / "survey-results"
        / f"anchor_{case['constellation']}.npz",
        dataset_root
        / "csv-data"
        / case["constellation"]
        / case["csv"],
    )


def _relative_change(before, after):
    return 100.0 * (after / before - 1.0)


def _maximum_deltas(proposed, compiled):
    fields = ("times", "states", "rotations", "covariances", "biases", "nis")
    deltas = {}
    for field in fields:
        before = np.asarray(getattr(proposed, field))
        after = np.asarray(getattr(compiled, field))
        if before.shape != after.shape:
            raise AssertionError(
                f"{field} shape mismatch: {before.shape} vs {after.shape}"
            )
        deltas[f"max_abs_{field}_delta"] = (
            float(np.max(np.abs(after - before))) if before.size else 0.0
        )
    if not np.array_equal(proposed.rejected, compiled.rejected):
        raise AssertionError("Compiled kernel changed UWB rejection decisions")
    deltas["rejection_decision_changes"] = int(
        np.count_nonzero(proposed.rejected != compiled.rejected)
    )
    return deltas


def _plot_case(axes, column, case, dataset, retained, runtimes):
    colors = {"proposed": "tab:blue", "task03_compiled": "tab:orange"}
    axes[0, column].plot(
        dataset.p_gt[:, 0],
        dataset.p_gt[:, 1],
        "k--",
        lw=1.0,
        label="Vicon",
    )
    for variant in VARIANTS:
        result = retained[variant]
        diagnostics = compute_metrics(dataset, result)
        axes[0, column].plot(
            result.states[:, 6],
            result.states[:, 7],
            color=colors[variant],
            lw=0.8,
            label=variant,
        )
        axes[1, column].plot(
            diagnostics["times"],
            np.linalg.norm(diagnostics["position_error"], axis=1),
            color=colors[variant],
            lw=0.7,
            label=variant,
        )
        axes[2, column].plot(
            diagnostics["times"],
            np.degrees(diagnostics["yaw_error"]),
            color=colors[variant],
            lw=0.7,
            label=variant,
        )

    baseline = float(np.median(runtimes["proposed"]))
    compiled = float(np.median(runtimes["task03_compiled"]))
    axes[0, column].set_title(
        f"{case['name']}\nsteady-state speedup {baseline / compiled:.2f}x"
    )
    axes[0, column].set(xlabel="x [m]", ylabel="y [m]")
    axes[0, column].axis("equal")
    axes[1, column].set(xlabel="time [s]", ylabel="position error [m]")
    axes[2, column].set(xlabel="time [s]", ylabel="yaw error [deg]")
    for row in range(3):
        axes[row, column].grid(alpha=0.25)


def main():
    args = parse_args()
    if args.repeats < 2:
        raise ValueError("--repeats must be at least two for AB/BA pairing")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    warmup_seconds = ensure_compiled_kernel()

    summaries = []
    rows = []
    figure, axes = plt.subplots(3, len(CASES), figsize=(16.0, 11.5))
    for column, case in enumerate(CASES):
        anchor_path, csv_path = _paths(args.dataset_root, case)
        dataset = load_dataset(
            anchor_path,
            csv_path,
            max_duration=case["duration"],
        )
        runtimes = {variant: [] for variant in VARIANTS}
        retained = {}
        for repeat in range(args.repeats):
            order = (
                VARIANTS
                if repeat % 2 == 0
                else tuple(reversed(VARIANTS))
            )
            for variant in order:
                result = run_dataset(
                    dataset,
                    variant=variant,
                    yaw_error_degrees=case["yaw"],
                )
                runtimes[variant].append(result.runtime_seconds)
                retained[variant] = result
                print(
                    json.dumps(
                        {
                            "case": case["name"],
                            "repeat": repeat,
                            "variant": variant,
                            **result.metrics,
                        },
                        ensure_ascii=False,
                        allow_nan=True,
                    ),
                    flush=True,
                )
                gc.collect()

        proposed = retained["proposed"]
        compiled = retained["task03_compiled"]
        deltas = _maximum_deltas(proposed, compiled)
        before_runtime = float(np.median(runtimes["proposed"]))
        after_runtime = float(np.median(runtimes["task03_compiled"]))
        paired_speedups = (
            np.asarray(runtimes["proposed"])
            / np.asarray(runtimes["task03_compiled"])
        )
        before = proposed.metrics
        after = compiled.metrics
        event_duration = float(dataset.t_imu[-1] - dataset.t_imu[0])
        summary = {
            "case": case["name"],
            "imu_samples": int(len(dataset.t_imu)),
            "uwb_measurements": int(len(dataset.t_uwb)),
            "imu_rate_hz": float(len(dataset.t_imu) / event_duration),
            "uwb_rate_hz": float(len(dataset.t_uwb) / event_duration),
            "proposed_runtime_median_seconds": before_runtime,
            "compiled_runtime_median_seconds": after_runtime,
            "steady_state_speedup": before_runtime / after_runtime,
            "paired_speedup_median": float(np.median(paired_speedups)),
            "paired_speedups": [
                float(value) for value in paired_speedups
            ],
            "runtime_reduction_percent": 100.0 * (
                1.0 - after_runtime / before_runtime
            ),
            "proposed_runtime_runs_seconds": [
                float(value) for value in runtimes["proposed"]
            ],
            "compiled_runtime_runs_seconds": [
                float(value) for value in runtimes["task03_compiled"]
            ],
            "position_rmse_change_percent": _relative_change(
                before["position_rmse_m"],
                after["position_rmse_m"],
            ),
            "velocity_rmse_change_percent": _relative_change(
                before["velocity_rmse_mps"],
                after["velocity_rmse_mps"],
            ),
            "orientation_rmse_change_percent": _relative_change(
                before["orientation_geodesic_rmse_rad"],
                after["orientation_geodesic_rmse_rad"],
            ),
            "yaw_rmse_change_percent": _relative_change(
                before["yaw_rmse_rad"],
                after["yaw_rmse_rad"],
            ),
            **deltas,
        }
        summaries.append(summary)
        for variant in VARIANTS:
            rows.append(
                {
                    "case": case["name"],
                    "variant": variant,
                    **retained[variant].metrics,
                    "runtime_runs_seconds": runtimes[variant],
                }
            )
        _plot_case(axes, column, case, dataset, retained, runtimes)

    axes[0, 0].legend(fontsize=8)
    axes[1, 0].legend(fontsize=8)
    axes[2, 0].legend(fontsize=8)
    figure.suptitle(
        "Task03: Python proposed baseline vs equation-equivalent compiled core",
        fontsize=14,
    )
    figure.tight_layout()
    figure.savefig(args.output_dir / "comparison.png", dpi=170)
    plt.close(figure)

    payload = {
        "description": (
            "AB/BA paired steady-state benchmark. JIT compilation/cache load "
            "is measured separately and excluded from replay latency."
        ),
        "repeats": args.repeats,
        "jit_warmup_or_cache_load_seconds": float(warmup_seconds),
        "cases": list(CASES),
        "rows": rows,
        "summaries": summaries,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    print(json.dumps(summaries, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
