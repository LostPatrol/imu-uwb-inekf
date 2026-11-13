"""Apply a conservative causal gyro-interpolation buffer without changing core states."""

import json
import numpy as np
import pandas as pd
import run


def buffer_for(name):
    """Bound the wait for every later gyro knot referenced during both fixed episodes."""
    source = run.DATA/"csv-data"/name.split("-")[0]/(name+".csv")
    frame = pd.read_csv(source, usecols=["t_acc", "t_gyro"])
    acc = frame.t_acc.dropna().to_numpy(float)
    gyro = frame.t_gyro.dropna().to_numpy(float)
    # Include the entire first prefix, native startup episode and recovery prefix/episode.
    acc = acc[acc <= 15.+run.DURATION+1e-9]
    upper = np.clip(np.searchsorted(gyro, acc, side="left"), 0, len(gyro)-1)
    waits = np.maximum(0., gyro[upper]-acc)
    return dict(applied_constant_buffer_s=float(np.max(waits)),
                referenced_knot_wait_max_s=float(np.max(waits)),
                referenced_knot_wait_median_s=float(np.median(waits)),
                imu_samples_audited=int(len(acc)), raw_gyro_samples=int(len(gyro)))


def main():
    """Delay every publication conservatively; keep state times and values unchanged."""
    buffers = {name:buffer_for(name) for name in run.NAMES}
    records = []
    for checkpoint in (run.HERE/"results").glob("*.json"):
        record = json.loads(checkpoint.read_text(encoding="utf8"))
        arrays = dict(np.load(checkpoint.with_suffix(".npz")))
        original = arrays.get("outputs_unbuffered", arrays["outputs"]).copy()
        delayed = original.copy()
        buffer = buffers[record["name"]]["applied_constant_buffer_s"]
        # Uniformly shifting the complete service queue by its maximum input lookahead
        # upper-bounds any schedule waiting on those gyro packets at individual events.
        delayed[:, 1] += buffer
        metrics, grid = run.score(record["name"], record["start_absolute_s"], delayed)
        arrays.update(outputs=delayed, outputs_unbuffered=original, evaluation_grid=grid)
        np.savez_compressed(checkpoint.with_suffix(".npz"), **arrays)
        record.update(metrics, gyro_interpolation_buffer_s=buffer,
                      applied_buffer_max_s=buffer, applied_buffer_median_s=buffer)
        run.dump(checkpoint, record)
        records.append(record)
    run.dump(run.HERE/"causal_clock.json", dict(
        scheme="Per-record fixed maximum forward-gyro-knot buffer added to whole publication service queue; no state backfill",
        state_values_unchanged=True, state_timestamps_unchanged=True,
        recordings=buffers, all_cases_recomputed=len(records),
        data_prefix="1.5s IMU+TDoA2 CV, using only radio timestamps <=prefix end; differs from TWR IMU-only prefix"))
    run.summarize(records)
    print(json.dumps(buffers, indent=2))


if __name__ == "__main__":
    main()
