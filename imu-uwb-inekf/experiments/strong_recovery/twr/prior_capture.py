"""Test one packet-bound prior before the unchanged native capture constructor.

Only the first recovered TWR packet may move the prior. A triggering packet is
not updated again. All subsequent states come from the original recursion.
"""

from pathlib import Path
import argparse
import hashlib
import json
import sys
from concurrent.futures import as_completed

import numpy as np
import pandas as pd

# Keep every artifact outside the shared estimator and independent score files.
HERE = Path(__file__).resolve().parent
CODE = HERE.parents[2]
ROOT = CODE.parent
OUT = ROOT / "agent/codex/strong_prior"
sys.path.insert(0, str(HERE))
import diagnose
sys.path.insert(0, str(HERE.parent))
from feedback import single_offset
sys.path.insert(0, str(HERE.parent / "external_twr"))
import adapter
import replay as native_replay
import qdrone_run
from qdrone import truth_at
from cpu_replay import replay_pool

# These clocks, gaps and windows are fixed before opening any score cache.
RESUME = 25.0
END = 36.0
LENGTHS = (2., 5., 10., 20.)
WINDOWS = (("first2", 0., 2.), ("two_to_five", 2., 5.), ("six_to_eleven", 6., 11.))
# QDrone's supplied tag extrinsic is unknown; retain the registered engineering allowance.
QDRONE_RADIUS = 1.0
# Official correct-anchor caches and original K traces, never their score tables.
QDRONE = ROOT / "agent/codex/strong_external/qdrone_verified"
DELFT_K = ROOT / "agent/codex/strong_twr/length_cache"


def write_json(path, value):
    """Write NumPy audit values without changing their meaning."""
    def convert(item):
        """Convert only NumPy arrays and scalars."""
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(type(item).__name__)
    path.write_text(json.dumps(value, indent=2, default=convert), encoding="utf-8")


def install_prior(reference, packet, lever_radius, start_time=0.):
    """Hook only a resume constructor and its immediately following first packet."""
    stamp, anchor, measurement, anchor_id, sigma, bias = packet
    original_resume = reference.on_ranges_resumed
    original_new = reference._new_capture
    original_correct = reference.correct_twr
    original_handoff = reference._handoff
    audit = dict(triggered=False, shift_m=0., gaussian_skips=0, offered_packets=0,
                 constructor_calls=0, first_packet_time_s=float(stamp))
    armed = False
    consumed = False
    skip_pending = False

    def new_capture(state, covariance, native_bias):
        """Pass the bounded physical mean to the original constructor with original P."""
        nonlocal skip_pending
        if not armed:
            return original_new(state, covariance, native_bias)
        before = state.copy()
        delta = single_offset(stamp, before[:3, 4], lever_radius, anchor, None,
                              measurement-bias, sigma)
        triggered = bool(np.any(delta))
        argument = state
        if triggered:
            argument = state.copy()
            argument[:3, 4] += delta
        supplied_covariance = covariance.copy()
        capture = original_new(argument, covariance, native_bias)
        p, v = argument[:3, 4], argument[:3, 3]
        expected = np.r_[.5*p@p, p@v, v@v, p, v, 1.]
        assert np.array_equal(capture.lifted.chi, expected)
        # The native constructor itself round-trips P through physical coordinates.
        assert np.array_equal(covariance, supplied_covariance)
        assert np.array_equal(capture.X[:3, 3], before[:3, 3])
        assert np.array_equal(capture.X[:3, :3], before[:3, :3])
        assert capture.X[2, 4] == before[2, 4]
        skip_pending = triggered
        audit.update(triggered=triggered, shift_m=float(np.linalg.norm(delta)),
            delta=delta, pre_position=before[:3, 4], initial_position=p,
            initial_velocity=v, initial_chi=capture.lifted.chi.copy(),
            initial_lifted_P_diagonal=np.diag(capture.lifted.P),
            physical_lift_defect=capture.lifted.chi[:3]-expected[:3],
            original_invariant_P_supplied=True,
            native_constructor_P_roundtrip_max_abs=float(np.max(np.abs(capture.imu_core.P-covariance))))
        audit["constructor_calls"] += 1
        return capture

    def on_resume():
        """Use the native reset, counter increments and unchanged two-second timer."""
        nonlocal armed, consumed
        if consumed or reference.elapsed_s+start_time < stamp-1e-8:
            return original_resume()
        armed = True
        restarted = original_resume()
        armed = False
        consumed = bool(restarted)
        if restarted:
            assert reference._capture_elapsed_s == 0.
            audit["capture_timer_after_restart_s"] = reference._capture_elapsed_s
            audit["native_elapsed_at_restart_s"] = reference.elapsed_s
        return restarted

    def correct(anchor_value, value, anchor_index=None):
        """Count a geometry-used packet separately from all ordinary Gaussian attempts."""
        nonlocal skip_pending
        audit["offered_packets"] += 1
        if skip_pending:
            assert anchor_index == anchor_id and value == measurement
            assert np.array_equal(anchor_value, anchor)
            skip_pending = False
            audit["gaussian_skips"] += 1
            return False
        return original_correct(anchor_value, value, anchor_index)

    def handoff():
        """Observe the native timer boundary without changing the handoff operation."""
        if consumed:
            assert reference._capture_elapsed_s == 2.
            audit["capture_elapsed_at_handoff_s"] = reference._capture_elapsed_s
            audit["handoff_count_after_resume"] = audit.get("handoff_count_after_resume", 0)+1
        return original_handoff()

    reference._new_capture = new_capture
    reference.on_ranges_resumed = on_resume
    reference.correct_twr = correct
    reference._handoff = handoff
    return audit


def delft_case(name, length, directory):
    """Replay original Delft sample arrivals with one constructor hook."""
    data = dict(np.load(diagnose.CACHE / (Path(name).stem+".npz")))
    protocol = json.loads((CODE/"results/roadmap_stage1_20261104_final/protocol.json").read_text(encoding="utf-8"))
    model = protocol["range_models"]["holdout"]
    times = data["time"]
    valid = data["changed"] & (data["ranges"] > 0)
    valid[(times >= RESUME-length) & (times < RESUME)] = False
    rows, ids = np.nonzero(valid & (times >= RESUME)[:, None])
    first, anchor_id = int(rows[0]), int(ids[0])
    packet = (times[first], data["anchors"][anchor_id], data["ranges"][first, anchor_id],
              anchor_id, model["range_sigma"][anchor_id], model["range_bias"][anchor_id])
    reference = diagnose.make_capture(data, model)
    # Delft's native TWR model treats the antenna as the position, with zero lever.
    audit = install_prior(reference, packet, 0.)
    selected = np.flatnonzero(times <= END)
    output = np.empty((len(selected), 6))
    rotations = np.empty((len(selected), 3, 3))
    last_event = None
    resumed = False
    for k in selected:
        stamp = times[k]
        if k:
            reference.predict(data["imu"][k-1], float(stamp-times[k-1]))
        events = np.flatnonzero(valid[k])
        if len(events):
            if last_event is not None and stamp-last_event > 1.:
                reference.on_ranges_resumed()
                resumed |= stamp >= RESUME
            last_event = stamp
        for aid in events:
            reference.correct_twr(data["anchors"][aid], data["ranges"][k, aid], int(aid))
        output[k] = np.r_[reference.X[:3, 4], reference.X[:3, 3]]
        rotations[k] = reference.X[:3, :3]
    assert resumed
    baseline = np.load(DELFT_K/(Path(name).stem+f"__gap{int(length)}.npz"))
    kp, kv = baseline["k_pos"][selected], baseline["k_vel"][selected]
    audit.update(native_stats=vars(reference.stats), failure=None)
    return finalize(directory, name, length, times[selected], output[:, :3], output[:, 3:],
                    kp, kv, audit, rotations=rotations)


def qdrone_case(identifier, length, directory):
    """Reuse the full asynchronous union runner without touching shared sources."""
    record = qdrone_run.load_record(QDRONE/"cache"/(identifier+".npz"))
    first = int(np.searchsorted(record.radio_time, RESUME))
    aid = int(record.radio_id[first])
    packet = (record.radio_time[first], record.anchors[aid], record.radio_range[first],
              aid, adapter.RANGE_SIGMA_M, 0.)
    original_factory = native_replay.initial_reference
    audits = []

    def factory(value):
        """Install the single hook on this worker's private reference object."""
        reference = original_factory(value)
        audits.append(install_prior(reference, packet, QDRONE_RADIUS, record.imu_time[0]))
        return reference

    native_replay.initial_reference = factory
    try:
        result, events, native_audit = native_replay.reference_replay(record, length)
    finally:
        native_replay.initial_reference = original_factory
    audit = audits[0]
    audit.update(native_stats=native_audit["stats"], failure=native_audit["failure"],
                 native_events=len(events), elapsed_s=native_audit["elapsed_s"])
    baseline = np.load(QDRONE/(identifier+f"__L{int(length)}.npz"))
    assert np.array_equal(result["time"], baseline["time"])
    return finalize(directory, identifier, length, result["time"], result["K"], result["K_velocity"],
                    baseline["K"], baseline["K_velocity"], audit,
                    X=result["X"], bias=result["bias"], P_diagonal=result["P_diagonal"],
                    mode_track=result["mode_track"],
                    baseline_X=baseline["X"], baseline_bias=baseline["bias"],
                    baseline_P_diagonal=baseline["P_diagonal"])


def finalize(directory, name, length, times, position, velocity, kp, kv, audit, **extra):
    """Audit immutable prefixes, exact nontriggered recursion and packet budgets."""
    before = times < audit["first_packet_time_s"]
    assert np.array_equal(position[before], kp[before])
    assert np.array_equal(velocity[before], kv[before])
    if not audit["triggered"]:
        assert np.array_equal(position, kp)
        assert np.array_equal(velocity, kv)
        if "X" in extra:
            for key in ("X", "bias", "P_diagonal"):
                assert np.array_equal(extra[key], extra["baseline_"+key])
    stats = audit["native_stats"]
    attempts = stats["accepted"]+stats["rejected"]+stats["invalid"]
    assert attempts+audit["gaussian_skips"] == audit["offered_packets"]
    assert audit["gaussian_skips"] == int(audit["triggered"])
    assert audit["constructor_calls"] == 1
    assert audit["handoff_count_after_resume"] == 1
    audit.update(sequence=name, length_s=length, native_attempts=attempts,
                 nontrigger_exact_K=not audit["triggered"],
                 prefix_exact_K=True, no_return_to_K=True)
    case = Path(name).stem+f"__L{int(length)}"
    np.savez_compressed(directory/(case+".npz"), time=times, prior=position,
                        prior_velocity=velocity, K=kp, K_velocity=kv, **extra)
    write_json(directory/(case+".json"), audit)
    return audit


def run_job(job):
    """Dispatch one predeclared native case in a one-thread numerical process."""
    domain, name, length, directory = job
    if domain == "delft":
        return delft_case(name, length, Path(directory))
    return qdrone_case(name, length, Path(directory))


def score(domain, directory):
    """Open truth only after the complete frozen domain matrix has been estimated."""
    rows = []
    for file in sorted(directory.glob("*__L*.npz")):
        result = np.load(file)
        identifier, length = file.stem.rsplit("__L", 1)
        if domain == "delft":
            truth = np.load(diagnose.CACHE/(identifier+"_scoring.npz"))["position"][:len(result["time"])]
            clock = result["time"]
            indices = np.arange(len(clock))
            supported = np.ones(len(clock), bool)
            ground = truth
        else:
            record = qdrone_run.load_record(QDRONE/"cache"/(identifier+".npz"))
            truth = np.load(QDRONE/"cache"/(identifier+"_scoring.npz"))["truth"]
            clock = RESUME+np.arange(0., 11., qdrone_run.SCORE_DT)
            ground, supported = truth_at(clock+record.origin_s, truth)
            indices = np.clip(np.searchsorted(result["time"], clock, side="right")-1, 0, len(result["time"])-1)
        for method in ("K", "prior"):
            error = result[method][indices]-ground
            for label, left, right in WINDOWS:
                window = (clock >= RESUME+left) & (clock < RESUME+right)
                valid = window & supported & np.isfinite(error).all(axis=1)
                values = error[valid]
                coverage = np.sum(valid)/max(1, np.sum(window))
                rows.append(dict(sequence=identifier, length_s=int(length), method=method,
                    window=label, coverage=float(coverage), complete_window=coverage >= .95,
                    xy_rmse_m=float(np.sqrt(np.mean(np.sum(values[:, :2]**2, axis=1)))) if len(values) else None,
                    xy_peak_m=float(np.max(np.linalg.norm(values[:, :2], axis=1))) if len(values) else None))
    frame = pd.DataFrame(rows)
    frame.to_csv(directory/"metrics.csv", index=False)
    frame.groupby(["length_s", "window", "method"]).xy_rmse_m.median().to_csv(directory/"summary.csv")
    frame[frame.complete_window].groupby(["length_s", "window", "method"]).xy_rmse_m.median().to_csv(directory/"summary_complete.csv")
    paired = frame.pivot(index=["sequence", "length_s", "window"], columns="method", values="xy_rmse_m").reset_index()
    paired["difference_m"] = paired["prior"]-paired["K"]
    paired["outcome"] = np.where(paired.difference_m < -1e-12, "better", np.where(paired.difference_m > 1e-12, "worse", "equal"))
    paired.loc[~np.isfinite(paired.difference_m), "outcome"] = "unavailable"
    audits = json.loads((directory/"audit.json").read_text(encoding="utf-8"))
    info = pd.DataFrame([{key: entry[key] for key in ("sequence", "length_s", "triggered", "shift_m", "gaussian_skips")} for entry in audits])
    info["sequence"] = info.sequence.map(lambda value: Path(value).stem)
    paired = paired.merge(info, on=["sequence", "length_s"], validate="many_to_one")
    coverage = frame.groupby(["sequence", "length_s", "window"]).coverage.min().reset_index()
    paired = paired.merge(coverage, on=["sequence", "length_s", "window"], validate="one_to_one")
    paired["complete_window"] = paired.coverage >= .95
    paired.to_csv(directory/"paired.csv", index=False)
    paired.groupby(["length_s", "window", "outcome"]).size().to_csv(directory/"win_tie_loss.csv")
    paired[paired.complete_window].groupby(["length_s", "window", "outcome"]).size().to_csv(directory/"win_tie_loss_complete.csv")
    print(frame.groupby(["length_s", "window", "method"]).xy_rmse_m.median().to_string(), flush=True)


def main():
    """Freeze one mechanism, estimate all cases, then optionally run downstream scores."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=("delft", "qdrone"), required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--score-only", action="store_true")
    args = parser.parse_args()
    directory = OUT/args.domain
    directory.mkdir(parents=True, exist_ok=True)
    if args.score_only:
        score(args.domain, directory)
        return
    if args.domain == "delft":
        protocol = json.loads((CODE/"results/roadmap_stage1_20261104_final/protocol.json").read_text(encoding="utf-8"))
        names = protocol["evaluation"]
    else:
        cohort = json.loads((QDRONE/"eligibility.json").read_text(encoding="utf-8"))
        names = [entry["site"]+"__"+entry["name"] for entry in cohort if entry["status"] == "sensor_eligible"]
        assert len(names) == 10
    manifest = dict(domain=args.domain, names=names, lengths_s=LENGTHS, resume_s=RESUME,
        mechanism="First recovered packet bounds pre-update p before native capture construction; trigger packet skips Gaussian update; no later reset or return",
        prior="Original invariant P, v, attitude, bias and fixed lifted P; constructor initializes physical q/s/h; timer remains two seconds",
        radius_m=0. if args.domain == "delft" else QDRONE_RADIUS,
        scores_read=False, parameter_search=False,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write_json(directory/"frozen_protocol.json", manifest)
    jobs = [(args.domain, name, length, str(directory)) for name in names for length in LENGTHS]
    audits = []
    with replay_pool(len(jobs), workers=min(3, args.workers)) as pool:
        futures = {pool.submit(run_job, job): job for job in jobs}
        for future in as_completed(futures):
            audit = future.result()
            audits.append(audit)
            write_json(directory/"audit.json", audits)
            print(json.dumps(dict(done=len(audits), total=len(jobs), name=audit["sequence"],
                length=audit["length_s"], trigger=audit["triggered"], shift=audit["shift_m"])), flush=True)


if __name__ == "__main__":
    main()
