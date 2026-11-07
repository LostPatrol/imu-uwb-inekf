"""Paired runtime and accuracy validation for the task02 compute kernels."""

import argparse
import csv
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

VARIANTS = ("pre_optimization", "proposed")


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
        default=REPOSITORY_ROOT / "results" / "task02_optimization",
    )
    parser.add_argument("--repeats", type=int, default=2)
    return parser.parse_args()


def _paths(dataset_root, case):
    anchor = (
        dataset_root
        / "survey-results"
        / f"anchor_{case['constellation']}.npz"
    )
    csv_path = (
        dataset_root
        / "csv-data"
        / case["constellation"]
        / case["csv"]
    )
    return anchor, csv_path


def _metric_rows(case, retained, runtimes):
    rows = []
    for variant in VARIANTS:
        metrics = retained[variant].metrics
        row = {
            "case": case["name"],
            "variant": variant,
            **metrics,
            "runtime_seconds": float(np.median(runtimes[variant])),
            "runtime_min_seconds": float(np.min(runtimes[variant])),
            "runtime_max_seconds": float(np.max(runtimes[variant])),
            "runtime_runs_seconds": [float(value) for value in runtimes[variant]],
        }
        rows.append(row)
    return rows


def _plot_case(axes, column, case, dataset, retained, runtimes):
    gt = dataset.p_gt
    axes[0, column].plot(gt[:, 0], gt[:, 1], "k--", lw=1.0, label="Vicon")
    colors = {"pre_optimization": "tab:blue", "proposed": "tab:orange"}
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

    before = float(np.median(runtimes["pre_optimization"]))
    after = float(np.median(runtimes["proposed"]))
    speedup = before / after
    axes[0, column].set_title(f"{case['name']}\nmedian speedup {speedup:.2f}x")
    axes[0, column].set(xlabel="x [m]", ylabel="y [m]")
    axes[0, column].axis("equal")
    axes[0, column].grid(alpha=0.25)
    axes[1, column].set(xlabel="time [s]", ylabel="position error [m]")
    axes[1, column].grid(alpha=0.25)


def main():
    args = parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    summaries = []
    figure, axes = plt.subplots(2, len(CASES), figsize=(16.0, 8.2))
    for column, case in enumerate(CASES):
        anchor, csv_path = _paths(args.dataset_root, case)
        dataset = load_dataset(anchor, csv_path, max_duration=case["duration"])
        runtimes = {variant: [] for variant in VARIANTS}
        retained = {}
        for repeat in range(args.repeats):
            order = VARIANTS if repeat % 2 == 0 else tuple(reversed(VARIANTS))
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

        rows = _metric_rows(case, retained, runtimes)
        all_rows.extend(rows)
        before = next(row for row in rows if row["variant"] == "pre_optimization")
        after = next(row for row in rows if row["variant"] == "proposed")
        summaries.append(
            {
                "case": case["name"],
                "speedup": before["runtime_seconds"] / after["runtime_seconds"],
                "position_rmse_change_percent": 100.0
                * (
                    after["position_rmse_m"] / before["position_rmse_m"]
                    - 1.0
                ),
                "velocity_rmse_change_percent": 100.0
                * (
                    after["velocity_rmse_mps"] / before["velocity_rmse_mps"]
                    - 1.0
                ),
                "orientation_rmse_change_percent": 100.0
                * (
                    after["orientation_geodesic_rmse_rad"]
                    / before["orientation_geodesic_rmse_rad"]
                    - 1.0
                ),
                "yaw_rmse_change_percent": 100.0
                * (
                    after["yaw_rmse_rad"] / before["yaw_rmse_rad"]
                    - 1.0
                ),
                "rejection_change": (
                    after["rejected_measurements"]
                    - before["rejected_measurements"]
                ),
            }
        )
        _plot_case(axes, column, case, dataset, retained, runtimes)

    axes[0, 0].legend(fontsize=8)
    axes[1, 0].legend(fontsize=8)
    figure.suptitle(
        "Task02 structured InEKF: current improved baseline vs optimized",
        fontsize=14,
    )
    figure.tight_layout()
    figure.savefig(args.output_dir / "comparison.png", dpi=170)
    plt.close(figure)

    payload = {
        "description": (
            "Paired comparison against the coordinate-correct task01 path. "
            "Runtime is the median of alternating-order runs."
        ),
        "repeats": args.repeats,
        "cases": list(CASES),
        "rows": all_rows,
        "summaries": summaries,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True),
        encoding="utf-8",
    )

    csv_rows = []
    for row in all_rows:
        csv_row = dict(row)
        csv_row["runtime_runs_seconds"] = json.dumps(
            csv_row["runtime_runs_seconds"]
        )
        csv_rows.append(csv_row)
    with (args.output_dir / "metrics.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(json.dumps(summaries, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
