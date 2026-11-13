"""Verify completed physical interventions, covariance mapping and immutable author core."""

from pathlib import Path
import json
import itertools
import numpy as np
import pandas as pd
import run


def main():
    """Check saved inputs and outputs without excluding any unsuccessful trial."""
    maxima = dict(xy_prior_m=0., yaw_prior_rad=0., z_sensor_m=0., velocity_sensor_mps=0., tilt_sensor=0., bias_sensor=0., physical_covariance=0.)
    count, invalid_outputs, failures, buffered_cases = 0, 0, 0, 0
    clock_errors = []
    for checkpoint in (run.HERE/"results").glob("*.json"):
        record = json.loads(checkpoint.read_text(encoding="utf8"))
        payload = np.load(checkpoint.with_suffix(".npz"))
        output, state = payload["outputs"], payload["initial_state"]
        base = np.load(run.HERE/"data"/record["name"]/"sensor_base.npz")
        mode = record["mode"]
        baseline = base["startup_state"] if mode == "sensor_bootstrap" else base["recovery_state"]
        bias = base["startup_bias"] if mode == "sensor_bootstrap" else base["recovery_bias"]
        maxima["z_sensor_m"] = max(maxima["z_sensor_m"], float(abs(state[2, 4]-baseline[2, 4])))
        maxima["velocity_sensor_mps"] = max(maxima["velocity_sensor_mps"], float(np.max(np.abs(state[:3, 3]-baseline[:3, 3]))))
        maxima["bias_sensor"] = max(maxima["bias_sensor"], float(np.max(np.abs(payload["initial_bias"]-bias))))
        maxima["tilt_sensor"] = max(maxima["tilt_sensor"], float(np.max(np.abs(state[2, :3]-baseline[2, :3]))))
        invariant, physical = run.covariance(state, "wide" if mode == "sensor_bootstrap" else mode)
        scale = max(1., float(np.max(np.abs(invariant))))
        maxima["physical_covariance"] = max(maxima["physical_covariance"], float(np.max(np.abs(invariant-payload["initial_invariant_covariance"])))/scale)
        if mode != "sensor_bootstrap":
            gt_position, gt_yaw = run.truth_at(record["name"], np.array([record["start_absolute_s"]]))
            directions = {"px":(1., 0.), "nx":(-1., 0.), "py":(0., 1.), "ny":(0., -1.)}
            expected_xy = gt_position[0, :2]+record["magnitude_m"]*np.array(directions[record["direction"]])
            maxima["xy_prior_m"] = max(maxima["xy_prior_m"], float(np.max(np.abs(state[:2, 4]-expected_xy))))
            estimate_yaw = np.arctan2(state[1, 0], state[0, 0])
            delta = estimate_yaw-gt_yaw[0]-np.radians(record["yaw_offset_deg"])
            maxima["yaw_prior_rad"] = max(maxima["yaw_prior_rad"], float(abs(np.arctan2(np.sin(delta), np.cos(delta)))))
        if len(output):
            invalid_outputs += int(not (np.all(np.diff(output[:, 0]) > 0) and np.all(np.diff(output[:, 1]) > 0) and np.all(output[:, 1] >= output[:, 0])))
        if "outputs_unbuffered" in payload:
            original = payload["outputs_unbuffered"]
            difference = output-original
            expected = np.zeros_like(difference)
            expected[:, 1] = record["gyro_interpolation_buffer_s"]
            clock_errors.append(float(np.max(np.abs(difference-expected))) if len(output) else 0.)
            buffered_cases += 1
        failures += int(record["failure"] is not None)
        count += 1
    protocol = json.loads((run.HERE/"protocol.json").read_text(encoding="utf8"))
    core_unchanged = protocol["author_sha256"] == run.digest(run.AUTHOR)
    frame = pd.DataFrame([json.loads(path.read_text(encoding="utf8")) for path in (run.HERE/"results").glob("*.json")])
    counts = frame.groupby(["name", "mode"]).size()
    expected = set(itertools.product(run.NAMES, (1., 5., 10., 20.), ("px", "nx", "py", "ny"), (0., 90., 180.), ("wide", "tight")))
    expected.update((name, 0., "none", 0., "sensor_bootstrap") for name in run.NAMES)
    observed = set(frame[["name", "magnitude_m", "direction", "yaw_offset_deg", "mode"]].itertuples(index=False, name=None))
    count_rows = [{"name":name, "mode":mode, "count":int(value)} for (name, mode), value in counts.items()]
    result = dict(completed_cases=count, planned_cases=388, core_unchanged=core_unchanged,
                  invalid_output_timelines=invalid_outputs, estimator_failures=failures,
                  maximum_errors=maxima, counts=count_rows, causally_buffered_cases=buffered_cases,
                  exact_case_schedule=observed == expected,
                  max_clock_transform_error_s=max(clock_errors, default=None))
    result["passed"] = (core_unchanged and count == 388 and observed == expected and buffered_cases == count and invalid_outputs == 0
                        and max(clock_errors, default=1.) < 1e-9 and all(value < 1e-9 for value in maxima.values()))
    run.dump(run.HERE/"audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
