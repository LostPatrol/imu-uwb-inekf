"""Scan physically meaningful initial covariance scales at the 180 degree case."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import json
import os
import sys

import pandas as pd


ESTIMATION_DIR = Path(__file__).resolve().parents[1] / "scripts" / "estimation"
sys.path.insert(0, str(ESTIMATION_DIR))

from pipeline import load_dataset, run_dataset


CASES = (
    ("c3m3", 3, "const3/const3-trial7-tdoa3-manual3.csv"),
    ("c4m1", 4, "const4/const4-trial7-tdoa2-manual1.csv"),
)


def run_job(job):
    root, tag, constellation, relative_csv, yaw_std, position_std = job
    root = Path(root)
    dataset = load_dataset(
        root / f"survey-results/anchor_const{constellation}.npz",
        root / "csv-data" / relative_csv,
        max_duration=15.0,
    )
    result = run_dataset(
        dataset,
        variant="proposed",
        yaw_error_degrees=180.0,
        initial_std_overrides={"yaw": yaw_std, "position": position_std},
    )
    return {
        "dataset": tag,
        "yaw_std": yaw_std,
        "position_std": position_std,
        **result.metrics,
    }


def main():
    root = Path(r"E:\datasets\dataset\flight-dataset")
    output = Path(__file__).resolve().parents[1] / "results" / "covariance_scale_scan"
    output.mkdir(parents=True, exist_ok=True)
    jobs = [
        (root, *case, yaw_std, position_std)
        for case in CASES
        for yaw_std in (0.5, 0.75, 1.0, 1.5, 2.0, 3.2)
        for position_std in (0.1, 0.25, 0.5, 1.0)
    ]
    rows = []
    with ProcessPoolExecutor(max_workers=min(4, os.cpu_count() or 1)) as pool:
        futures = [pool.submit(run_job, job) for job in jobs]
        for index, future in enumerate(as_completed(futures), 1):
            row = future.result()
            rows.append(row)
            print(
                f"[{index:02d}/{len(jobs)}] {row['dataset']} "
                f"yaw_std={row['yaw_std']} pos_std={row['position_std']}",
                flush=True,
            )
    frame = pd.DataFrame(rows).sort_values(["dataset", "yaw_std", "position_std"])
    frame.to_csv(output / "metrics.csv", index=False)
    (output / "metrics.json").write_text(
        json.dumps(rows, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    print(f"Saved to {output}", flush=True)


if __name__ == "__main__":
    main()
