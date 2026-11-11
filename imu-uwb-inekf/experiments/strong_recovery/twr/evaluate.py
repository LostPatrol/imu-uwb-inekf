"""Score the frozen shared output overlay against intact original TWR capture."""

from pathlib import Path
import argparse
import hashlib
import json
import sys

import numpy as np
import pandas as pd

# The common shadow is owned by the shared algorithm task, never patched here.
CODE = Path(__file__).resolve().parents[3]
ROOT = CODE.parent
HERE = Path(__file__).resolve().parent
OUT = ROOT/"agent/codex/strong_twr/length_results"
CACHE = ROOT/"agent/codex/strong_twr/length_cache"
SCORING = CODE/"experiments/split_handoff/delft/results/cache"
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(CODE/"experiments/split_handoff"))
from cpu_replay import replay_pool

# Fixed windows relative to the common restoration clock; all six logs cover them.
WINDOWS = {"early0_2": (25.0, 27.0), "early2_5": (27.0, 30.0),
           "future6_11": (31.0, 36.0), "post": (25.0, np.inf),
           "common": (8.1, np.inf)}


def scores(times, position, truth):
    """Expose XY primary and 3D secondary errors including every displayed output."""
    delta = position-truth
    xy, total = np.linalg.norm(delta[:, :2], axis=1), np.linalg.norm(delta, axis=1)
    jumps = np.r_[0.0, np.linalg.norm(np.diff(position[:, :2], axis=0), axis=1)]
    result = {}
    intervals = dict(WINDOWS, tail5=(float(times[-1]-5), np.inf))
    for key, (left, right) in intervals.items():
        selected = (times >= left) & (times < right)
        finite = selected & np.isfinite(total)
        result[key+"_coverage"] = float(np.mean(np.isfinite(total[selected])))
        for label, error in (("xy", xy), ("3d", total)):
            result[key+"_"+label+"_rmse_m"] = float(np.sqrt(np.mean(error[finite]**2)))
            result[key+"_"+label+"_peak_m"] = float(np.max(error[finite]))
        result[key+"_xy_jump_max_m"] = float(np.max(jumps[selected]))
    return result


def evaluate_case(file):
    """Call one frozen API without altering K, then load truth only for scoring."""
    from shadow import recover
    from radio_bounds import bound_xy
    data = np.load(file)
    times, kp, kv = data["time"], data["k_pos"], data["k_vel"]
    result = recover(times, kp, kv, None, data["obs_time"], data["plus_xyz"], None,
                     data["observations"], data["sigma"], float(data["resume"]))
    bounds = bound_xy(times, kp, 0.0, data["obs_time"], data["plus_xyz"], None,
                     data["observations"], data["sigma"], float(data["resume"]))
    outputs = {"K": (kp, kv), "always_C": (result["always_c_pos"], result["always_c_vel"]),
               "guarded": (result["pos"], result["vel"]), "bounds": (bounds["pos"], kv)}
    prefix, suffix = times < 25, times >= 27
    for method in ("always_C", "guarded", "bounds"):
        assert np.array_equal(outputs[method][0][prefix], kp[prefix])
        assert np.array_equal(outputs[method][1][prefix], kv[prefix])
        assert np.array_equal(outputs[method][0][suffix], kp[suffix])
        assert np.array_equal(outputs[method][1][suffix], kv[suffix])
    stem = file.stem.split("__gap")[0]
    truth = np.load(SCORING/(stem+"_scoring.npz"))["position"]
    payload = dict(time=times, truth=truth, use_c=result["use_c"],
        always_c_available=result["always_c_available"], bounds_active=bounds["active"])
    feasibility = bound_feasibility(data, truth, bounds["pos"])
    early = (times >= 25) & (times < 27)
    support = early & result["always_c_available"]
    availability = dict(c_available_fraction=float(np.mean(result["always_c_available"][early])),
        guard_used_fraction=float(np.mean(result["use_c"][early])),
        bounds_active_fraction=float(np.mean(bounds["active"][early])),
        c_available_only_xy_rmse_m=float(np.sqrt(np.mean(np.sum(
            (result["always_c_pos"][support, :2]-truth[support, :2])**2, axis=1)))) if support.any() else None,
        k_on_c_available_xy_rmse_m=float(np.sqrt(np.mean(np.sum(
            (kp[support, :2]-truth[support, :2])**2, axis=1)))) if support.any() else None)
    rows = []
    for method, (position, velocity) in outputs.items():
        payload[method+"_pos"], payload[method+"_vel"] = position, velocity
        rows.append(dict(sequence=stem, gap_s=float(data["gap_s"]), method=method,
                         bounds_feasible_fraction=feasibility["all_contain_fraction"],
                         **availability,
                         **scores(times, position, truth)))
    np.savez_compressed(OUT/(file.stem+".npz"), **payload)
    # The shared module returns JSON-compatible frozen block decisions.
    (OUT/(file.stem+"_diagnostics.json")).write_text(json.dumps(result["diagnostics"],
        indent=2, default=lambda value: value.tolist() if hasattr(value, "tolist") else str(value)), encoding="utf-8")
    (OUT/(file.stem+"_bounds_diagnostics.json")).write_text(json.dumps(dict(
        geometry=bounds["diagnostics"], scoring_only=feasibility), indent=2,
        default=lambda value: value.tolist() if hasattr(value, "tolist") else str(value)), encoding="utf-8")
    return rows


def bound_feasibility(data, truth, bound_position):
    """Audit the registered disk-containment condition downstream of the candidate."""
    from radio_bounds import BODY_SPEED_BOUND, MAX_AGE_SECONDS, SIGMA_MARGIN
    times, stamps = data["time"], data["obs_time"]
    latest, cursor, contained, violations = {}, 0, [], []
    selected = np.flatnonzero((times >= 25) & (times < 27))
    for index in selected:
        now = times[index]
        while cursor < len(stamps) and stamps[cursor] <= now:
            if data["observations"][cursor] > 0:
                latest[int(data["obs_anchor"][cursor])] = cursor
            cursor += 1
        local = []
        for row in latest.values():
            age = now-stamps[row]
            if age > MAX_AGE_SECONDS:
                continue
            radius = data["observations"][row]+SIGMA_MARGIN*data["sigma"][row]+BODY_SPEED_BOUND*age
            local.append(float(np.linalg.norm(truth[index, :2]-data["plus_xyz"][row, :2])-radius))
        violation = max(local, default=-np.inf)
        contained.append(violation <= 1e-10)
        violations.append(max(0., violation))
    k_error = np.linalg.norm(data["k_pos"][selected, :2]-truth[selected, :2], axis=1)
    b_error = np.linalg.norm(bound_position[selected, :2]-truth[selected, :2], axis=1)
    contained = np.asarray(contained)
    excess = b_error-k_error
    assert np.all(excess[contained] <= 1e-8)
    return dict(all_contain_fraction=float(np.mean(contained)),
        max_constraint_violation_m=float(max(violations)),
        improved_samples=int(np.sum(excess < -1e-12)), worse_samples=int(np.sum(excess > 1e-12)),
        equal_samples=int(np.sum(np.abs(excess) <= 1e-12)),
        max_xy_error_increase_m=float(max(excess)),
        condition_checked_after_estimator=True)


def main():
    """Evaluate all fixed cached cases with no more than four numerical workers."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted(CACHE.glob("*.npz"))
    with replay_pool(len(files), workers=min(4, args.workers)) as pool:
        results = list(pool.map(evaluate_case, files))
    frame = pd.DataFrame([row for case in results for row in case])
    frame.to_csv(OUT/"paired_metrics.csv", index=False)
    aggregates = {key: (key, "median") for key in frame if key.endswith("_rmse_m")}
    aggregates.update({key: (key, "max") for key in frame if key.endswith("_peak_m") or key.endswith("_jump_max_m")})
    summary = frame.groupby(["gap_s", "method"]).agg(**aggregates)
    summary.to_csv(OUT/"summary.csv")
    counts = []
    for length, group in frame.groupby("gap_s"):
        pivot = group.pivot(index="sequence", columns="method")
        for method in ("always_C", "guarded", "bounds"):
            for column in (key for key in frame if key.endswith("_rmse_m") or key.endswith("_peak_m")):
                difference = pivot[column][method]-pivot[column]["K"]
                counts.append(dict(gap_s=length, method=method, metric=column,
                    improved=int(np.sum(difference < -1e-12)), unchanged=int(np.sum(np.abs(difference) <= 1e-12)),
                    worse=int(np.sum(difference > 1e-12)), worst_increase_m=float(difference.max())))
    pd.DataFrame(counts).to_csv(OUT/"improved_counts.csv", index=False)
    provenance = dict(domain="Delft same-date six-flight development domain", cases=len(files),
        branches="Independent shadow and bounds; neither feeds K or each other",
        always_c="C whenever available; otherwise K, with candidate availability and common-support scores separate",
        bounds_velocity="K velocity unchanged; horizontal display projection only",
        ground_truth="Loaded only after both candidates return; never selects a candidate",
        identity="All output p/v equals K for t<25 and t>=27; later equality is by design",
        source_sha256={name:hashlib.sha256((HERE.parent/name).read_bytes()).hexdigest()
                       for name in ("shadow.py", "radio_bounds.py", "protocol.json")})
    (OUT/"provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(summary[["early0_2_xy_rmse_m", "early0_2_xy_peak_m", "future6_11_xy_rmse_m"]].to_string())


if __name__ == "__main__":
    main()
