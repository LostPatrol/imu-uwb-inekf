"""Time independent yaw policies on fixed Delft BODY-bias real windows.

Reuse the existing directional-radio target; rebuild window preprocessing and
both CV fits for every standalone timed policy without shared candidate fits.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import platform
import sys
import time

# Numerical libraries must see these limits before their first import.
for _name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
sys.dont_write_bytecode = True

# Source and generated evidence live separately; existing results are read only.
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
ROOT = PROJECT.parent
JOINT_PATH = HERE.parent / "joint_branch/study.py"
CAL_PATH = HERE / "calibration.json"
INPUTS_PATH = HERE / "fixed_inputs.json"
ARCHIVE = HERE.parent / "evidence/paired_acquisition/delft_new_objective_seeds.csv"

_spec = importlib.util.spec_from_file_location("flight_timing_joint", JOINT_PATH)
joint = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(joint)
init = joint.shape.init

import numpy as np
import pandas as pd
import scipy
from threadpoolctl import threadpool_info, threadpool_limits

# Fix all policies and physical windows before evaluating fitted results.
METHODS = ("local", "profile_1", "uniform_2", "uniform_4")
DURATIONS = (2.0, 4.0, 8.0)
ENDPOINT = 10.0
REPEATS = 3
KEYS = ["sequence", "endpoint_s", "duration_s", "layout", "frame_deg"]


def sha(path):
    """Fingerprint immutable source, calibration, and input files."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(output, name, value):
    """Write task-local JSON with the existing NumPy-aware encoder."""
    joint.radio.dump(output / name, value)


def build_records(names, dataset, archive_cache):
    """Rebuild physics from raw CSVs; optionally cross-check an existing archive cache."""
    records, checks = {}, []
    anchors = joint.radio.load_anchors(dataset)
    for name in names:
        started = time.perf_counter()
        log = joint.radio.load_real_log(dataset, name, sensor_delay_s=.20)
        rotation, bias = joint.radio.estimate_initial_rotation_and_bias(log)
        keep = log.time >= 1.5
        times = log.time[keep] - log.time[keep][0]
        rotations, displacement, velocity = joint.radio.preintegrate(times, log.imu[keep], rotation, bias)
        data = dict(time=times, ranges=log.ranges[keep], changed=log.changed[keep],
                    rotations=rotations, displacement=displacement, integrated_velocity=velocity,
                    anchors=anchors, truth_position=log.ground_truth_position[keep])
        errors = {}
        cache_path = None if archive_cache is None else archive_cache / (Path(name).stem + ".npz")
        cache_status = "not_requested" if cache_path is None else "missing"
        if cache_path is not None and cache_path.is_file():
            try:
                with np.load(cache_path) as saved:
                    for key, value in data.items():
                        if key not in saved or value.shape != saved[key].shape:
                            raise ValueError(f"Archive shape/key mismatch: {name}/{key}")
                        errors[key] = float(np.max(np.abs(value.astype(float)-saved[key].astype(float))))
                cache_status = "matched" if max(errors.values()) <= 1e-12 else "different"
            except (OSError, KeyError, ValueError) as exc:
                cache_status = repr(exc)
        assert times[-1] >= ENDPOINT and times[0] <= ENDPOINT-max(DURATIONS)
        records[name] = data
        checks.append(dict(sequence=name, available_end_s=float(times[-1]),
                           available_start_s=float(times[0]), cache_max_abs_errors=errors,
                           cache_status=cache_status, cache_path=None if cache_path is None else str(cache_path),
                           cache_sha256=sha(cache_path) if cache_path is not None and cache_path.is_file() else None,
                           record_preprocessing_s=time.perf_counter()-started))
        print(f"INPUT {name}: end={times[-1]:.3f}s archive_cache={cache_status}", flush=True)
    return records, checks


def window(data, duration):
    """Select changed positive all-eight-anchor events and integrate body-bias sensitivity."""
    times, ranges = data["time"], data["ranges"]
    end = np.searchsorted(times, ENDPOINT, side="right")-1
    start = np.searchsorted(times, times[end]-duration, side="left")
    mask = data["changed"][start:end+1] & (ranges[start:end+1] > 0)
    index, ids = np.nonzero(mask)
    index += start
    tau = times[index]-times[end]
    displacement = data["displacement"][index]-data["displacement"][end]-tau[:, None]*data["integrated_velocity"][end]
    sensitivity = joint.shape.reanchor_integral(data["rotations"][start:end], times[start:end+1], times[index])
    values = (tau, displacement, data["anchors"][ids], ranges[index, ids], ids)
    return values, sensitivity, end, start, index


def cv_fit(values, calibration):
    """Run and count the unchanged two common robust CV translation fits."""
    tau, _, anchors, ranges, ids = values
    fits = []
    original = init.least_squares
    def counted(*args, **kwargs):
        """Keep actual CV optimizer outputs without altering its arguments."""
        fit = original(*args, **kwargs)
        fits.append(fit)
        return fit
    init.least_squares = counted
    try:
        cv = joint.radio.cv_candidate(tau, anchors, ranges-calibration["static"][ids], calibration["sigma"][ids])[0]
    finally:
        init.least_squares = original
    assert len(fits) == 2
    return cv, sum(fit.nfev for fit in fits)


def solve(data, duration, method, calibration):
    """Time fresh preprocessing, both CV fits, candidates, and exact existing BODY refinement."""
    started = time.perf_counter()
    values, sensitivity, end, start, index = window(data, duration)
    preprocessed = time.perf_counter()
    cv, cv_nfev = cv_fit(values, calibration)
    cv_finished = time.perf_counter()
    if method == "profile_1":
        candidate, _ = init.lifted_candidate(values[0], values[1], values[2],
            values[3]-calibration["static"][values[4]], calibration["sigma"][values[4]])
        angles = [float(candidate[6])]
    elif method == "local":
        angles = [0.0]
    else:
        angles = np.linspace(-np.pi, np.pi, int(method.rsplit("_", 1)[1]), endpoint=False)
    seeds = [np.r_[cv, angle, np.zeros(3)] for angle in angles]
    candidates_finished = time.perf_counter()
    x, fit = joint.solve(values, sensitivity, "BiasPrior", "geometry", calibration, seeds)
    finished = time.perf_counter()
    # Scoring and packet fingerprints are outside the measured estimator interval.
    error = x[:3]-data["truth_position"][end]
    return dict(elapsed_s=finished-started, preprocessing_s=preprocessed-started,
        cv_s=cv_finished-preprocessed, candidate_s=candidates_finished-cv_finished,
        raw_s=finished-candidates_finished, objective=fit["objective"], success=fit["success"],
        cv_nfev=cv_nfev, raw_nfev=fit["nfev"], total_nfev=cv_nfev+fit["nfev"], cv_calls=2,
        raw_calls=len(seeds), finite=bool(np.isfinite(x).all() and np.isfinite(fit["objective"])),
        xy_error_m=float(np.linalg.norm(error[:2])), z_error_m=float(abs(error[2])),
        position_error_m=float(np.linalg.norm(error)), px_m=x[0], py_m=x[1], pz_m=x[2],
        truth_x=data["truth_position"][end, 0], truth_y=data["truth_position"][end, 1],
        truth_z=data["truth_position"][end, 2], actual_endpoint_s=float(data["time"][end]),
        actual_start_s=float(data["time"][start]), events=len(values[0]),
        packet_sha256=hashlib.sha256(np.column_stack((index, values[4], values[3])).tobytes()).hexdigest(),
        seed_yaws_rad=json.dumps([float(a) for a in angles]))


def summarize(frame, output):
    """Use each paired window's three-run median, with descriptive flight-level coverage."""
    numeric = ["elapsed_s", "preprocessing_s", "cv_s", "candidate_s", "raw_s", "objective",
               "xy_error_m", "z_error_m", "position_error_m", "px_m", "py_m", "pz_m", "total_nfev"]
    medians = frame.groupby(KEYS+["method"], as_index=False)[numeric].median()
    best = medians.groupby(KEYS).objective.transform("min")
    medians["missed_best_found"] = medians.objective-best > 1e-4*(1+best)
    medians["elapsed_ms"] = 1000*medians.elapsed_s
    medians.to_csv(output/"window_medians.csv", index=False)
    pivot = medians.pivot(index=KEYS, columns="method", values="elapsed_ms")
    pivot.to_csv(output/"paired_window_ms.csv")
    rows = []
    for method in METHODS:
        part = medians[medians.method == method]
        ratios = pivot[method]/pivot.local
        rows.append(dict(method=method, windows=len(part), median_ms=float(part.elapsed_ms.median()),
            p95_ms=float(part.elapsed_ms.quantile(.95)), median_paired_ratio_vs_local=float(ratios.median()),
            median_paired_delta_vs_local_ms=float((pivot[method]-pivot.local).median()),
            median_paired_ratio_vs_profile=float((pivot[method]/pivot.profile_1).median()),
            objective_misses=int(part.missed_best_found.sum()),
            xy_rmse_m=float(np.sqrt(np.mean(part.xy_error_m**2))),
            xy_median_m=float(part.xy_error_m.median()), xy_p95_m=float(part.xy_error_m.quantile(.95)),
            median_cv_ms=float(1000*part.cv_s.median()), median_candidate_ms=float(1000*part.candidate_s.median()),
            median_raw_ms=float(1000*part.raw_s.median()), median_preprocessing_ms=float(1000*part.preprocessing_s.median())))
    pd.DataFrame(rows).to_csv(output/"summary.csv", index=False)
    archive_checks = dict(path=str(ARCHIVE), status="missing", matched_rows=0, expected_rows=54,
                          covered_methods=["local", "profile_1", "uniform_4"], unreferenced_methods=["uniform_2"])
    if ARCHIVE.is_file():
        old = pd.read_csv(ARCHIVE)
        old = old[(old.frame_deg == 0) & (old.layout == "all8") & (old.endpoint_s == ENDPOINT)]
        parity = medians.merge(old, on=KEYS+["method"], suffixes=("_new", "_archive"), validate="one_to_one")
        for key in ("objective", "xy_error_m", "z_error_m", "position_error_m", "total_nfev"):
            parity[key+"_abs_delta"] = abs(parity[key+"_new"]-parity[key+"_archive"])
        parity.to_csv(output/"archive_parity.csv", index=False)
        archive_checks.update(status="complete" if len(parity) == 54 else "partial", matched_rows=len(parity),
            sha256=sha(ARCHIVE), max_abs_deltas={key:float(parity[key+"_abs_delta"].max())
            for key in ("objective", "xy_error_m", "z_error_m", "position_error_m", "total_nfev")})
    checks = dict(archive_parity=archive_checks,
        max_repeated_objective_spread=float(frame.groupby(KEYS+["method"]).objective.agg(lambda x:x.max()-x.min()).max()),
        all_cv_calls_two=bool((frame.cv_calls == 2).all()),
        all_raw_calls_expected=bool((frame.raw_calls == frame.method.map(dict(local=1, profile_1=1, uniform_2=2, uniform_4=4))).all()),
        all_finite=bool(frame.finite.all()), best_solver_failures=int((~frame.success).sum()),
        paired_packets_per_window=frame.groupby(KEYS).packet_sha256.nunique().tolist(),
        paired_actual_endpoints_per_window=frame.groupby(KEYS).actual_endpoint_s.nunique().tolist())
    dump(output, "checks.json", checks)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)
    print(json.dumps(checks), flush=True)
    return rows


def main():
    """Predeclare 18 windows, warm all methods, then run three serial interleaved rounds."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Empty output directory; default: a new timestamped agent/codex run")
    parser.add_argument("--dataset", type=Path, default=joint.radio.DATASET,
                        help="Public Delft UWB dataset directory containing twr/ and anchor survey files")
    parser.add_argument("--archive-cache", type=Path,
                        help="Optional old NPZ directory for diagnostics only; absent files never replace raw inputs")
    args = parser.parse_args()
    output = args.output or ROOT / "agent/codex/flight_seed_timing_runs" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Preserve existing files: choose an empty --output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    calibration = {k:np.asarray(v) if isinstance(v,list) else v for k,v in json.loads(CAL_PATH.read_text()).items()}
    original = json.loads(INPUTS_PATH.read_text())
    names = original["evaluation"]
    jobs = [(name, duration) for name in names for duration in DURATIONS]
    sources = [Path(__file__), JOINT_PATH, Path(inspect.getfile(joint.radio)), Path(inspect.getfile(joint.shape)),
               Path(inspect.getfile(init)), PROJECT/"experiments/recovery_bootstrap/run_research.py", CAL_PATH, INPUTS_PATH]
    if ARCHIVE.is_file():
        sources.append(ARCHIVE)
    protocol = dict(created_utc=datetime.now(timezone.utc).isoformat(), windows=[dict(sequence=n, endpoint_s=ENDPOINT,
        duration_s=d, layout="all8", frame_deg=0) for n,d in jobs], methods=METHODS, repeats=REPEATS,
        warmup="one complete untimed call per method on first prespecified window",
        order="serial: outer rounds, inner fixed windows, cyclic method rotation by round+window index",
        objective="unchanged joint BiasPrior+geometry, mixed Huber range and quadratic 0.10 m/s^2 body-bias prior, max_nfev=120 per raw fit",
        candidates="local yaw 0; Profile-1 exact original static-corrected K1 lifted_candidate; uniform_K linspace(-pi,pi,K,endpoint=False), K=2,4",
        uniform2="two-start restriction of the existing uniform-angle formula; newly timed against this real BODY target",
        timed_inclusions="window slicing/event extraction, endpoint displacement reanchoring, exact body-bias sensitivity construction, fresh two CV solves, fresh candidates/seeds, all raw refinements and best selection",
        timed_exclusions="imports, raw CSV I/O, prefix attitude/bias estimate and record-level IMU preintegration (common streaming state), frozen training calibration, truth scoring, hashes, output I/O; no curvature computation because existing BODY solve does none",
        cached_work="only common record-level integrated physics, always rebuilt from raw CSV; optional archive cache is diagnostic only; no CV, profile, sensitivity or refined result caching",
        statistics="median across three rounds per window/method, then median over 18 paired window medians; paired ratios computed within windows; overlaps and six reused flights are not iid trials",
        objective_miss="J-Jbest > 1e-4*(1+Jbest) against best found among these four arms, not global optimality",
        training=original["training"], evaluation=names, blas_threads=1, process_id=os.getpid(),
        dataset=str(args.dataset.resolve()), output=str(output),
        fixed_inputs_provenance=original["provenance"],
        archive_cache=None if args.archive_cache is None else str(args.archive_cache.resolve()),
        environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                         platform=platform.platform(), processor=platform.processor(), threadpools=threadpool_info()),
        source_sha256={str(p):sha(p) for p in sources})
    dump(output, "protocol.json", protocol)
    with threadpool_limits(limits=1):
        records, checks = build_records(names, args.dataset, args.archive_cache)
        dump(output, "input_checks.json", dict(records=checks, cache_checked_records=sum(c["cache_status"] == "matched" for c in checks),
            total_records=len(names), actual_input="raw CSV only; archive cache does not supply solver inputs"))
        coverage = []
        for name, duration in jobs:
            values, _, end, start, _ = window(records[name], duration)
            coverage.append(dict(sequence=name, endpoint_s=ENDPOINT, duration_s=duration, actual_endpoint_s=records[name]["time"][end],
                actual_start_s=records[name]["time"][start], events=len(values[0]), observed_anchor_ids=json.dumps(np.unique(values[4]).tolist())))
        pd.DataFrame(coverage).to_csv(output/"coverage.csv", index=False)
        for method in METHODS:
            solve(records[names[0]], DURATIONS[0], method, calibration)
        print("WARMUP COMPLETE; serial timing begins", flush=True)
        rows = []
        for repeat in range(REPEATS):
            for number, (name, duration) in enumerate(jobs):
                for order, method in enumerate(np.roll(np.asarray(METHODS), repeat+number)):
                    result = solve(records[name], duration, str(method), calibration)
                    rows.append(dict(sequence=name, endpoint_s=ENDPOINT, duration_s=duration, layout="all8", frame_deg=0,
                        repeat=repeat, order=order, method=str(method), **result))
                print(f"ROUND {repeat+1}/3 WINDOW {number+1}/18 {Path(name).stem} {duration:g}s", flush=True)
        frame = pd.DataFrame(rows)
        frame.to_csv(output/"timing_runs.csv", index=False)
        summary = summarize(frame, output)
    protocol.update(completed_utc=datetime.now(timezone.utc).isoformat(),
        measured_s=float(frame.elapsed_s.sum()), completed_timed_calls=len(frame),
        immutable_sources_unchanged={str(p):sha(p)==protocol["source_sha256"][str(p)] for p in sources},
        summary=summary)
    dump(output, "protocol.json", protocol)
    print(f"OUTPUT {output}", flush=True)


if __name__ == "__main__":
    main()
