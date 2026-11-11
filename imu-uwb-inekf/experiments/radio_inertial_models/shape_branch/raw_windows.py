"""Rebuild Delft development windows from raw CSVs with separate-log calibration.

These recordings share prior development exposure and are not independent tests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

import physical_shape as shape

# Existing source loaders define the paper's static prefix and common time delay.
sys.path.insert(0, str(shape.BASE.parent / "roadmap_stage1"))
import run as protocol
from run_research import preintegrate

# Match the eleven physical time windows in the paper, all eight anchors, one frame.
WINDOWS = ((5., 2.), (5., 4.), (10., 2.), (10., 4.), (10., 8.),
           (20., 2.), (20., 4.), (20., 8.), (30., 2.), (30., 4.), (30., 8.))


def run(output, limit):
    """Load raw recordings, calibrate earlier pairs, and score later recordings."""
    dataset = protocol.DATASET
    training, evaluation = [], []
    for motion in protocol.MOTIONS:
        names = sorted(path.name for path in (dataset / "twr").glob(f"*twr_{motion}.csv"))
        training.append(names[0])
        evaluation.append(names[1])
    logs = {name: protocol.load_real_log(dataset, name, sensor_delay_s=.20)
            for name in training + evaluation[:limit]}
    anchors = protocol.load_anchors(dataset)
    calibration = protocol.calibration(logs, anchors, training)
    rows = []
    for name in evaluation[:limit]:
        log = logs[name]
        rotation, bias = protocol.estimate_initial_rotation_and_bias(log)
        selected = log.time >= 1.5
        times = log.time[selected] - log.time[selected][0]
        imu, ranges, changed = log.imu[selected], log.ranges[selected], log.changed[selected]
        truth = log.ground_truth_position[selected]
        rotations, displacement, velocity = preintegrate(times, imu, rotation, bias)
        for endpoint, duration in WINDOWS:
            end = np.searchsorted(times, endpoint, side="right") - 1
            start = np.searchsorted(times, times[end] - duration, side="left")
            local_rows, anchor_ids = np.nonzero(changed[start:end+1] & (ranges[start:end+1] > 0))
            event_rows = local_rows + start
            tau = times[event_rows] - times[end]
            d = displacement[event_rows] - displacement[end] - tau[:, None] * velocity[end]
            sensitivity = shape.reanchor_integral(rotations[start:end], times[start:end+1], times[event_rows])
            values = (tau, d, anchors[anchor_ids],
                      ranges[event_rows, anchor_ids] - calibration["range_bias"][anchor_ids],
                      calibration["range_sigma"][anchor_ids])
            for model, prior in (("K1", False), ("K2", False), ("Bias", True),
                                 ("BiasTilt", True), ("BiasTilt", False)):
                fit = shape.solve(values, sensitivity, model, prior)
                x = fit.pop("x")
                fit.pop("yaw_index")
                rows.append(dict(sequence=name, endpoint_requested_s=endpoint, endpoint_actual_s=times[end],
                    duration_s=duration, events=len(tau), method=model + ("_prior" if prior else ""),
                    position_error_m=float(np.linalg.norm(x[:3] - truth[end])),
                    endpoint_x_m=x[0], endpoint_y_m=x[1], endpoint_z_m=x[2],
                    independent_validation=False, **fit))
        print(f"Completed raw log {name}", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "raw_metrics.csv", index=False)
    summary = frame.groupby(["sequence", "method"]).agg(cases=("position_error_m", "size"),
        position_median_m=("position_error_m", "median"), position_p95_m=("position_error_m", lambda x:x.quantile(.95)),
        information_median=("information", "median"), prior_information_median=("prior_information", "median"),
        nonconverged=("success", lambda x:int((~x).sum())))
    summary.to_csv(output / "raw_summary.csv")
    metadata = dict(dataset=str(dataset), training=training, evaluation=evaluation[:limit],
        common_sensor_delay_s=.20, static_prefix_s=1.5, windows=WINDOWS, anchor_layout="all8",
        calibration={key:value.tolist() if isinstance(value, np.ndarray) else value for key,value in calibration.items()},
        independent_blind_test=False, prior_development_exposure=True,
        scope="Same-date raw CSV development replication, not independent generalization. No historical cache read.",
        truth_usage="Training range calibration, valid interval cropping, fixed time alignment, and offline endpoint scoring",
        raw_sha256={name:hashlib.sha256((dataset / "twr" / name).read_bytes()).hexdigest()
                    for name in training + evaluation[:limit]})
    (output / "raw_protocol.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(summary.to_string())


def main():
    """Run the paper's raw time-window panel without truth in optimization."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=shape.HERE / "results")
    parser.add_argument("--logs", type=int, default=6)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    run(args.output, args.logs)


if __name__ == "__main__":
    main()
