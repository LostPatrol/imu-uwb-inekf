"""Full 60 s, 180 degree before/after validation for all manual test cases."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ESTIMATION_DIR = Path(__file__).resolve().parents[1] / "scripts" / "estimation"
sys.path.insert(0, str(ESTIMATION_DIR))

from pipeline import load_dataset, run_dataset


CASES = (
    ("c3m1", 3, "const3/const3-trial7-tdoa2-manual1.csv"),
    ("c3m2", 3, "const3/const3-trial7-tdoa2-manual2.csv"),
    ("c3m3", 3, "const3/const3-trial7-tdoa3-manual3.csv"),
    ("c3m4", 3, "const3/const3-trial7-tdoa3-manual4.csv"),
    ("c4m1", 4, "const4/const4-trial7-tdoa2-manual1.csv"),
    ("c4m2", 4, "const4/const4-trial7-tdoa2-manual2.csv"),
)
VARIANTS = ("pre_coordinate_fix", "proposed")


def run_job(job):
    root, tag, constellation, relative_csv, variant = job
    root = Path(root)
    dataset = load_dataset(
        root / f"survey-results/anchor_const{constellation}.npz",
        root / "csv-data" / relative_csv,
        max_duration=60.0,
    )
    result = run_dataset(dataset, variant=variant, yaw_error_degrees=180.0)
    return {"dataset": tag, "variant": variant, **result.metrics}


def plot(frame, output):
    datasets = [case[0] for case in CASES]
    x = np.arange(len(datasets))
    width = 0.36
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.5))
    fields = (
        ("position_rmse_m", "Position RMSE [m]", True),
        ("yaw_rmse_rad", "Yaw RMSE [rad]", False),
        ("rejection_rate", "UWB rejection rate", False),
    )
    for axis, (field, label, log_scale) in zip(axes, fields):
        for offset, variant, color in (
            (-width / 2, "pre_coordinate_fix", "tab:red"),
            (width / 2, "proposed", "tab:green"),
        ):
            subset = frame[frame["variant"] == variant].set_index("dataset")
            values = [float(subset.loc[tag, field]) for tag in datasets]
            axis.bar(x + offset, values, width, label=variant, color=color)
        if log_scale:
            axis.set_yscale("log")
        axis.set_xticks(x, datasets)
        axis.set_ylabel(label)
        axis.grid(axis="y", alpha=0.25, which="both")
    axes[0].legend()
    fig.suptitle("Full 60 s replay at 180° initial yaw error")
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)


def main():
    root = Path(r"E:\datasets\dataset\flight-dataset")
    output = Path(__file__).resolve().parents[1] / "results" / "coordinate_fix_full60"
    output.mkdir(parents=True, exist_ok=True)
    jobs = [(root, *case, variant) for case in CASES for variant in VARIANTS]
    rows = []
    with ProcessPoolExecutor(max_workers=min(4, os.cpu_count() or 1)) as pool:
        futures = [pool.submit(run_job, job) for job in jobs]
        for index, future in enumerate(as_completed(futures), 1):
            row = future.result()
            rows.append(row)
            print(
                f"[{index:02d}/{len(jobs)}] {row['dataset']} {row['variant']}",
                flush=True,
            )
    frame = pd.DataFrame(rows).sort_values(["dataset", "variant"])
    frame.to_csv(output / "metrics.csv", index=False)
    (output / "metrics.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    plot(frame, output / "full60_before_after.png")
    print(f"Saved to {output}", flush=True)


if __name__ == "__main__":
    main()
