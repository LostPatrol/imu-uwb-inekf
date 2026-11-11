"""Replay causal single-InEKF conditional replacements on actual TWR recordings.

Sensor-only decoding precedes estimation; truth opens only after all three arms.
Historical lifted output is read strictly as a separately scored reference.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

# Keep independent recording workers from multiplying BLAS threads.
for _variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_variable] = "1"

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

# Experiment ownership includes all new implementations and derived artifacts.
HERE = Path(__file__).resolve().parent
CODE = HERE.parents[1]
ROOT = CODE.parent
OUTPUT = HERE / "results"
DELF = CODE / "experiments/split_handoff/delft/results/cache"
QDRONE = ROOT / "agent/codex/strong_external/qdrone_verified"
sys.path.insert(0, str(CODE / "experiments/roadmap_stage1"))
from invariant15 import Invariant15, coordinate_map
from smoother import conditional_replace

# The original common physical prior and online scalar update policy stay fixed.
PRIOR_STD = np.r_[.05, .05, 2., np.full(3, 2.), np.full(3, 8.), np.full(3, .01), np.full(3, .1)]
SETTINGS = dict(gate_threshold=12.25, gate_warmup_updates=256, huber_threshold=2.)
# Paper-length acquisition and sensor gap detector were frozen before scores.
ACQUISITION = 2.
GAP_TRIGGER = 1.
RESUME = 25.
LENGTHS = (2, 5, 10, 20)
METHODS = ("native", "ieks_local", "ieks_profile")
# Score native emissions causally on the established twenty-Hz support.
SCORE_DT = .05
WINDOWS = (("first2", 0., 2.), ("two_to_five", 2., 5.), ("six_to_eleven", 6., 11.))


def write_json(path, value):
    """Write UTF-8 audit data without dropping failure information."""
    def convert(item):
        """Convert scientific containers into portable JSON."""
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(type(item).__name__)
    path.write_text(json.dumps(value, indent=2, default=convert, ensure_ascii=False), encoding="utf-8")


def load_sensor(domain, identifier, length):
    """Decode original sensor caches without accessing any scoring arrays."""
    if domain == "delft":
        data = dict(np.load(DELF / (identifier+".npz")))
        protocol = json.loads((CODE / "results/roadmap_stage1_20261104_final/protocol.json").read_text())
        model = protocol["range_models"]["holdout"]
        times = data["time"]
        valid = data["changed"] & (data["ranges"] > 0)
        valid[(times >= RESUME-length) & (times < RESUME)] = False
        rows, ids = np.nonzero(valid)
        keep = times <= 36.
        return dict(time=times[keep], imu=data["imu"][keep], anchors=data["anchors"],
            rotation=data["rotations"][0], bias=data["bias"],
            packet_row=rows[rows < np.sum(keep)], packet_id=ids[rows < np.sum(keep)],
            packet_range=data["ranges"][rows[rows < np.sum(keep)], ids[rows < np.sum(keep)]],
            range_bias=np.asarray(model["range_bias"]), range_sigma=np.asarray(model["range_sigma"]),
            origin=0.)
    data = dict(np.load(QDRONE / "cache" / (identifier+".npz")))
    it, rt = data["imu_time"], data["radio_time"]
    # These exact edges match the frozen external reference event loop.
    edges = np.array([0., 5., 15., 20., 23., 25., 27., 30., 31., 36.])
    times = np.unique(np.r_[it[it <= 36.], rt[rt <= 36.], edges])
    times = times[times >= it[0]]
    sample = np.maximum(0, np.searchsorted(it, times, side="right")-1)
    valid = (rt >= times[0]) & (rt <= times[-1]) & ~((rt >= RESUME-length) & (rt < RESUME))
    rows = np.searchsorted(times, rt[valid])
    return dict(time=times, imu=data["imu"][sample], anchors=data["anchors"],
        rotation=data["initial_rotation"], bias=data["initial_bias"], packet_row=rows,
        packet_id=data["radio_id"][valid], packet_range=data["radio_range"][valid],
        range_bias=np.zeros(4), range_sigma=np.full(4, .3), origin=float(data["origin_s"]))


def snapshot(tracker):
    """Preserve the whole checkpoint Gaussian in common physical coordinates."""
    return tracker.X.copy(), tracker.physical_covariance().copy(), tracker.bias.copy()


def replay(data, method):
    """Run one full 15-state posterior, replacing intervals only at causal handoffs."""
    times, imu, anchors = data["time"], data["imu"], data["anchors"]
    initial = np.eye(5)
    initial[:3, :3] = data["rotation"]
    initial[:3, 4] = anchors.mean(axis=0)
    chart = coordinate_map(initial)
    tracker = Invariant15(initial, chart@np.diag(PRIOR_STD**2)@chart.T, data["bias"],
        range_sigma=data["range_sigma"], range_bias=data["range_bias"], **SETTINGS)
    states = np.full((len(times), 5, 5), np.nan)
    biases = np.full((len(times), 6), np.nan)
    covariance_diagonal = np.full((len(times), 15), np.nan)
    packets = [[] for _ in times]
    for k, identifier, raw in zip(data["packet_row"], data["packet_id"], data["packet_range"]):
        packets[int(k)].append((int(k), int(identifier), float(raw-data["range_bias"][identifier]),
                                float(data["range_sigma"][identifier])))
    checkpoint_index, checkpoint_prior = 0, snapshot(tracker)
    acquisition_start, pending = float(times[0]), method != "native"
    checkpoint_before_packets = True
    last_radio_index, last_radio_prior, last_radio_time = None, None, None
    audits, failure = [], None
    begun = time.perf_counter()
    for k, stamp in enumerate(times):
        try:
            if k:
                tracker.predict(imu[k-1], float(stamp-times[k-1]))
            if packets[k] and last_radio_time is not None and stamp-last_radio_time > GAP_TRIGGER:
                if method in ("ieks_local", "ieks_profile"):
                    checkpoint_index, checkpoint_prior = last_radio_index, last_radio_prior
                    acquisition_start, pending = float(stamp), True
                    checkpoint_before_packets = False
            for _, identifier, measured, _ in packets[k]:
                tracker.correct_twr(anchors[identifier], measured+data["range_bias"][identifier], identifier)
            if pending and stamp-acquisition_start >= ACQUISITION-1e-10:
                interval_packets = [p for group in packets[checkpoint_index:k+1] for p in group
                    if checkpoint_before_packets or p[0] > checkpoint_index]
                seed_packets = [p for p in interval_packets if times[p[0]] >= acquisition_start]
                supported = len(seed_packets) >= 16 and len({p[1] for p in seed_packets}) >= 4
                audit = dict(trigger_s=acquisition_start, handoff_s=float(stamp),
                    delay_s=float(stamp-acquisition_start), checkpoint_index=checkpoint_index,
                    checkpoint_s=float(times[checkpoint_index]), packet_factors=len(interval_packets),
                    post_return_seed_packets=len(seed_packets), supported=supported)
                pending = False
                if supported:
                    local_packets = [(p[0]-checkpoint_index, *p[1:]) for p in interval_packets]
                    prior_current = snapshot(tracker)
                    state, covariance, bias, detail = conditional_replace(
                        times[checkpoint_index:k+1], imu[checkpoint_index:k+1], checkpoint_prior,
                        local_packets, anchors, method in ("ieks_profile", "profile_startup_native_recovery"), acquisition_start)
                    audit.update(detail=detail,
                        endpoint_position_change_m=float(np.linalg.norm(state[:3, 4]-tracker.X[:3, 4])),
                        endpoint_attitude_change_deg=float(np.rad2deg(np.linalg.norm(
                            Rotation.from_matrix(state[:3, :3]@tracker.X[:3, :3].T).as_rotvec()))),
                        endpoint_bias_change=float(np.linalg.norm(bias-tracker.bias)),
                        replaced_native_endpoint_X=prior_current[0], replaced_native_endpoint_P=prior_current[1])
                    # The old interval posterior is discarded, never fused with this one.
                    tracker.X, tracker.P, tracker.bias = state.copy(), covariance.copy(), bias.copy()
                    tracker.X_hat, tracker.P_hat = tracker.X.copy(), tracker.P.copy()
                audits.append(audit)
            states[k], biases[k] = tracker.X, tracker.bias
            covariance_diagonal[k] = np.diag(tracker.physical_covariance())
            if packets[k]:
                last_radio_index, last_radio_prior, last_radio_time = k, snapshot(tracker), float(stamp)
            if not np.isfinite(states[k]).all() or np.linalg.norm(states[k, :3, 4]) > 1e5:
                raise FloatingPointError("Divergent/nonfinite online state")
        except Exception:
            failure = dict(time_s=float(stamp), traceback=traceback.format_exc())
            break
    return dict(time=times, X=states, bias=biases, P_diagonal=covariance_diagonal), dict(
        method=method, events=audits, failure=failure, elapsed_s=time.perf_counter()-begun)


def truth_support(domain, identifier, data, clock):
    """Open held-out scoring data only after all recursive estimates finish."""
    if domain == "delft":
        truth = np.load(DELF / (identifier+"_scoring.npz"))["position"]
        return np.column_stack([np.interp(clock, data["time"], truth[:len(data["time"]), j])
                                for j in range(3)]), np.ones(len(clock), dtype=bool)
    sys.path.insert(0, str(CODE / "experiments/strong_recovery/external_twr"))
    from qdrone import truth_at
    truth = np.load(QDRONE / "cache" / (identifier+"_scoring.npz"))["truth"]
    return truth_at(clock+data["origin"], truth)


def historical(domain, identifier, length):
    """Load the intact original lifted strong baseline after new estimation only."""
    if domain == "delft":
        path = ROOT / "agent/codex/strong_twr/length_cache" / (identifier+f"__gap{length}.npz")
        cache = np.load(path)
        return dict(time=cache["time"], position=cache["k_pos"])
    cache = np.load(QDRONE / (identifier+f"__L{length}.npz"))
    return dict(time=cache["time"], position=cache["K"])


def run_case(job):
    """Retain complete paired trajectories, solver audits, support and every failure."""
    domain, identifier, length = job
    case = domain+"__"+identifier+f"__L{length}"
    target = OUTPUT / (case+".json")
    if target.exists():
        return json.loads(target.read_text())["metrics"]
    data = load_sensor(domain, identifier, length)
    results, audits = {}, {}
    for method in METHODS:
        results[method], audits[method] = replay(data, method)
    historical_result = historical(domain, identifier, length)
    clock = np.arange(RESUME, RESUME+11.-1e-8, SCORE_DT)
    ground, supported = truth_support(domain, identifier, data, clock)
    outputs = {method: dict(time=value["time"], position=value["X"][:, :3, 4])
               for method, value in results.items()}
    outputs["old_lifted"] = historical_result
    rows = []
    for method, result in outputs.items():
        index = np.clip(np.searchsorted(result["time"], clock, side="right")-1, 0, len(result["time"])-1)
        error = result["position"][index]-ground
        for window, left, right in WINDOWS:
            selection = (clock >= RESUME+left) & (clock < RESUME+right-1e-8)
            valid = selection & supported & np.isfinite(error).all(axis=1)
            coverage = float(np.sum(valid)/np.sum(selection))
            rows.append(dict(domain=domain, sequence=identifier, gap_s=length, method=method,
                window=window, coverage=coverage, complete_window=coverage >= .95,
                truth_coverage=float(np.mean(supported[selection])), samples=int(np.sum(valid)),
                xy_rmse_m=float(np.sqrt(np.mean(np.sum(error[valid, :2]**2, axis=1)))) if np.any(valid) else None,
                xy_peak_m=float(np.max(np.linalg.norm(error[valid, :2], axis=1))) if np.any(valid) else None,
                failed=audits.get(method, {}).get("failure") is not None,
                handoffs=len([a for a in audits.get(method, {}).get("events", []) if a["supported"]])))
    payload = {"score_time": clock, "score_truth": ground, "score_truth_supported": supported}
    for method, result in results.items():
        payload.update({method+"_"+key: value for key, value in result.items()})
    payload.update(old_lifted_time=historical_result["time"], old_lifted_position=historical_result["position"])
    np.savez_compressed(OUTPUT / (case+".npz"), **payload)
    write_json(target, dict(case=case, audits=audits, metrics=rows))
    print(case, {m: (round(a["elapsed_s"], 2), len(a["events"]), bool(a["failure"]))
                 for m, a in audits.items()}, flush=True)
    return rows


def run_ablation(job):
    """Disable only recovery replacement after exactly the same Profile startup."""
    domain, identifier, length = job
    case = domain+"__"+identifier+f"__L{length}"
    path = OUTPUT/(case+".json")
    saved = json.loads(path.read_text(encoding="utf-8"))
    method = "profile_startup_native_recovery"
    if method in saved["audits"]:
        return saved["metrics"]
    data = load_sensor(domain, identifier, length)
    result, audit = replay(data, method)
    payload = dict(np.load(path.with_suffix(".npz")))
    prefix = result["time"] < RESUME+2.-1e-8
    comparisons = {key: bool(np.array_equal(result[key][prefix], payload["ieks_profile_"+key][prefix], equal_nan=True))
                   for key in ("X", "bias", "P_diagonal")}
    assert all(comparisons.values()), comparisons
    audit["exact_prehistory_through_first2"] = comparisons
    audit["prefix_checked_until_s"] = RESUME+2.
    clock = payload["score_time"]
    ground, supported = truth_support(domain, identifier, data, clock)
    index = np.clip(np.searchsorted(result["time"], clock, side="right")-1, 0, len(result["time"])-1)
    error = result["X"][index, :3, 4]-ground
    rows = []
    for window, left, right in WINDOWS:
        selection = (clock >= RESUME+left) & (clock < RESUME+right-1e-8)
        valid = selection & supported & np.isfinite(error).all(axis=1)
        coverage = float(np.sum(valid)/np.sum(selection))
        rows.append(dict(domain=domain, sequence=identifier, gap_s=length, method=method,
            window=window, coverage=coverage, complete_window=coverage >= .95,
            truth_coverage=float(np.mean(supported[selection])), samples=int(np.sum(valid)),
            xy_rmse_m=float(np.sqrt(np.mean(np.sum(error[valid, :2]**2, axis=1)))) if np.any(valid) else None,
            xy_peak_m=float(np.max(np.linalg.norm(error[valid, :2], axis=1))) if np.any(valid) else None,
            failed=audit["failure"] is not None,
            handoffs=len([a for a in audit["events"] if a["supported"]])))
    saved["metrics"].extend(rows)
    saved["audits"][method] = audit
    payload.update({method+"_"+key: value for key, value in result.items()})
    np.savez_compressed(path.with_suffix(".npz"), **payload)
    write_json(path, saved)
    print(case, "identical-prehistory verified", round(audit["elapsed_s"], 2), flush=True)
    return saved["metrics"]


def main():
    """Run frozen representative or full matched matrix with bounded workers."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--representative", action="store_true")
    parser.add_argument("--ablation", action="store_true")
    parser.add_argument("--domain", choices=("all", "delft", "qdrone"), default="all")
    args = parser.parse_args()
    OUTPUT.mkdir(exist_ok=True)
    protocol = json.loads((CODE / "results/roadmap_stage1_20261104_final/protocol.json").read_text())
    delft = [("delft", Path(name).stem, gap) for name in protocol["evaluation"] for gap in LENGTHS]
    qdrone = [("qdrone", path.stem, gap) for path in sorted((QDRONE / "cache").glob("*.npz"))
              if not path.stem.endswith("_scoring") for gap in LENGTHS]
    jobs = delft+qdrone if args.domain == "all" else delft if args.domain == "delft" else qdrone
    if args.representative:
        jobs = [("delft", Path(protocol["evaluation"][0]).stem, 10)]
    write_json(OUTPUT / "source_hashes.json", {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
               for p in HERE.glob("*.py")})
    rows, failures = [], []
    with ProcessPoolExecutor(max_workers=min(8, args.workers)) as pool:
        futures = {pool.submit(run_ablation if args.ablation else run_case, job): job for job in jobs}
        for future in as_completed(futures):
            try:
                rows.extend(future.result())
            except Exception:
                failures.append(dict(job=futures[future], traceback=traceback.format_exc()))
            pd.DataFrame(rows).to_csv(OUTPUT / "metrics.csv", index=False)
            write_json(OUTPUT / "failures.json", failures)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
