"""Audit sampling-phase sensitivity without changing the frozen primary scores."""

import json
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

# All analysis uses saved estimates, including the originally scored boundary.
HERE = Path(__file__).resolve().parent
# Avoid shadowing the legacy roadmap module that also has the name run.
_REPLAY_SPEC = importlib.util.spec_from_file_location("unified_score_reader", HERE/"run.py")
_REPLAY = importlib.util.module_from_spec(_REPLAY_SPEC)
_REPLAY_SPEC.loader.exec_module(_REPLAY)
load_sensor, truth_support = _REPLAY.load_sensor, _REPLAY.truth_support
# All sensitivity artifacts remain separate from the frozen primary metrics.
RESULTS = HERE/"results"
METHODS = ("native", "ieks_local", "ieks_profile", "profile_startup_native_recovery", "old_lifted")
# A half-period offset exposes sensitivity to the original twenty-Hz phase.
OFFSET_S = .025


def main():
    """Rescore original emissions and a shifted grid, preserving primary evidence."""
    rows, boundary = [], []
    for path in sorted(RESULTS.glob("*__L*.json")):
        case = json.loads(path.read_text(encoding="utf-8"))
        first = case["metrics"][0]
        domain, identifier, length = first["domain"], first["sequence"], first["gap_s"]
        data = load_sensor(domain, identifier, length)
        estimates = np.load(path.with_suffix(".npz"))
        source_clock = estimates["native_time"]
        clocks = {
            "frozen20Hz": estimates["score_time"],
            "offset25ms20Hz": np.arange(25.+OFFSET_S, 36., .05),
            "native_emission_density": source_clock[(source_clock >= 25.) & (source_clock < 36.)],
        }
        for sampling, clock in clocks.items():
            ground, supported = truth_support(domain, identifier, data, clock)
            for method in METHODS:
                times = estimates[method+"_time"]
                position = estimates[method+"_position"] if method == "old_lifted" else estimates[method+"_X"][:, :3, 4]
                index = np.clip(np.searchsorted(times, clock, side="right")-1, 0, len(times)-1)
                error = position[index]-ground
                for window, left, right in (("first2", 25., 27.), ("two_to_five", 27., 30.), ("six_to_eleven", 31., 36.)):
                    selection = (clock >= left) & (clock < right-1e-8)
                    valid = selection & supported & np.isfinite(error).all(axis=1)
                    coverage = float(np.sum(valid)/max(1, np.sum(selection)))
                    rows.append(dict(domain=domain, sequence=identifier, gap_s=length, method=method,
                        sampling=sampling, window=window, samples=int(np.sum(valid)), coverage=coverage,
                        complete_window=coverage >= .95,
                        xy_rmse_m=float(np.sqrt(np.mean(np.sum(error[valid, :2]**2, axis=1)))) if np.any(valid) else None))
                boundary.append(dict(domain=domain, sequence=identifier, gap_s=length, method=method,
                    sampling=sampling, first_score_s=float(clock[0]),
                    first_source_s=float(times[index[0]]), first_source_repr=repr(float(times[index[0]])),
                    past_hold_age_s=float(clock[0]-times[index[0]]),
                    first_xy_error_m=float(np.linalg.norm(error[0, :2])) if supported[0] else None))
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS/"phase_sensitivity.csv", index=False)
    frame[frame.complete_window].groupby(["domain", "gap_s", "method", "sampling", "window"]).agg(
        recordings=("sequence", "size"), median_per_flight_xy_rmse_m=("xy_rmse_m", "median"),
        worst_flight_xy_rmse_m=("xy_rmse_m", "max")).to_csv(RESULTS/"phase_sensitivity_summary.csv")
    pd.DataFrame(boundary).to_csv(RESULTS/"phase_boundary_audit.csv", index=False)
    selected = frame[(frame.domain == "delft") & (frame.gap_s == 20) & (frame.window == "first2")]
    print(selected.groupby(["method", "sampling"]).xy_rmse_m.median().unstack().to_string())


if __name__ == "__main__":
    main()
