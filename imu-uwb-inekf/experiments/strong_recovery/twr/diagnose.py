"""Small causal diagnostics of the manuscript's real TWR recovery capture.

Translation variants share each source's exact inertial input and arrived radio.
Truth is loaded only after all estimator trajectories have been produced.
"""

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

# All generated artifacts remain in this task's temporary directory.
CODE = Path(__file__).resolve().parents[3]
ROOT = CODE.parent
OUT = ROOT / "agent/codex/strong_twr"
CACHE = CODE / "experiments/split_handoff/delft/results/cache"
for folder in (CODE/"experiments/roadmap_stage1", CODE/"experiments/recovery_bootstrap",
               CODE/"experiments/split_handoff"):
    sys.path.insert(0, str(folder))
from run import PRIOR_STD, SETTINGS, CaptureThenTrack, LiftedTWRInEKF, handoff_covariance
from invariant15 import Invariant15, coordinate_map
from run_research import solve_window, window_data, tracker_from_estimate
from lifted_twr import LiftedRangeKF, DirectRangeEKF, SquaredRangeEKF
from cpu_replay import replay_pool

# Fixed diagnosis interval; this is not a new gap-length or method-selection sweep.
GAP = (15.0, 25.0)
END = 27.0
# Physical p/v order in the 15-state error vector.
PV = np.r_[6:9, 3:6]


def core_of(tracker):
    """Return the native covariance owner without changing estimator state."""
    if isinstance(tracker, CaptureThenTrack):
        return tracker.capture.imu_core if tracker.mode == "capture" else tracker.tracker
    return tracker


def snapshot(tracker):
    """Copy the available state and physical covariance for an auditable fork."""
    core = core_of(tracker)
    return dict(X=tracker.X.copy(), bias=tracker.bias.copy(), P=core.physical_covariance().copy())


def make_capture(data, model):
    """Construct precisely the manuscript's centroid-start capture baseline."""
    state = np.eye(5)
    state[:3, :3] = data["rotations"][0]
    state[:3, 4] = data["anchors"].mean(axis=0)
    chart = coordinate_map(state)
    return CaptureThenTrack(state, chart@np.diag(PRIOR_STD**2)@chart.T, data["bias"],
        reset_on_resume=True, tracker_factory=Invariant15, imu_core_factory=Invariant15,
        capture_factory=LiftedTWRInEKF, handoff_covariance_factory=handoff_covariance,
        range_bias=model["range_bias"], range_sigma=model["range_sigma"], **SETTINGS)


def make_variants(seed, model):
    """Freeze one source mean/input while varying one translation mechanism at a time."""
    p, v = seed["X"][:3, 4], seed["X"][:3, 3]
    variants = {}
    for label, factory in (("lifted", LiftedRangeKF), ("raw", DirectRangeEKF),
                           ("squared", SquaredRangeEKF)):
        for warmup in (0, 256):
            kwargs = dict(position_std=8.0, velocity_std=2.0, accel_noise_std=10.0,
                range_bias=model["range_bias"], range_sigma=model["range_sigma"],
                gate_threshold=12.25, gate_warmup_updates=warmup, huber_threshold=2.0)
            if label == "lifted":
                kwargs.update(q_std=80.0, s_std=40.0, h_std=20.0)
            variants[f"{label}_warm{warmup}"] = factory(p, v, **kwargs)
    # Learned-covariance controls retain the same mean and independent lift nuisances.
    for warmup in (0, 256):
        for label, factory in (("raw", DirectRangeEKF), ("lifted", LiftedRangeKF)):
            kwargs = dict(position_std=8.0, velocity_std=2.0, accel_noise_std=10.0,
                range_bias=model["range_bias"], range_sigma=model["range_sigma"],
                gate_threshold=12.25, gate_warmup_updates=warmup, huber_threshold=2.0)
            if label == "lifted":
                kwargs.update(q_std=80.0, s_std=40.0, h_std=20.0)
            learned = factory(p, v, **kwargs)
            physical = seed["P"][np.ix_(PV, PV)]
            if label == "raw":
                learned.P = physical.copy()
            else:
                jacobian = np.zeros((10, 6))
                jacobian[0, :3] = p
                jacobian[1, :3], jacobian[1, 3:] = v, p
                jacobian[2, 3:] = 2*v
                jacobian[3:6, :3], jacobian[6:9, 3:] = np.eye(3), np.eye(3)
                learned.P = jacobian@physical@jacobian.T
                learned.P[:3, :3] += np.diag([80.0**2, 40.0**2, 20.0**2])
            variants[f"{label}_learned_cov_warm{warmup}"] = learned
    # Prediction-only output isolates withholding early radio corrections.
    variants["prediction_only"] = DirectRangeEKF(p, v, accel_noise_std=10.0)
    return variants


def diagnostic_case(name):
    """Replay the two real source histories and fixed conditional translation controls."""
    data = dict(np.load(CACHE/(Path(name).stem+".npz")))
    protocol = json.loads((CODE/"results/roadmap_stage1_20261104_final/protocol.json").read_text(encoding="utf-8"))
    model = protocol["range_models"]["holdout"]
    times = data["time"]
    selected = np.flatnonzero((times >= GAP[1]) & (times < END))
    stop = int(selected[-1])
    actual = {"capture": make_capture(data, model), "profile": None}
    histories = {key: np.full((stop+1, 6), np.nan) for key in actual}
    snapshots, variants, branches, measurements = {}, {}, {}, []
    resumed = False
    trace = {}
    for k, stamp in enumerate(times[:stop+1]):
        for source, tracker in actual.items():
            if tracker is not None and k:
                tracker.predict(data["imu"][k-1], float(stamp-times[k-1]))
        if actual["profile"] is None and stamp >= 2.0:
            estimate = solve_window(*window_data(data, 0, k, model), method="profile")
            actual["profile"] = tracker_from_estimate(estimate, data["rotations"][k], data["bias"], model)
            new_profile = True
        else:
            new_profile = False
        for edge in (GAP[0], GAP[1]):
            if k and times[k-1] < edge <= stamp:
                for source, tracker in actual.items():
                    snapshots[f"{source}_pre{int(edge)}"] = snapshot(tracker)
        usable = data["changed"][k] & (data["ranges"][k] > 0)
        if GAP[0] <= stamp < GAP[1]:
            usable[:] = False
        ids = np.flatnonzero(usable)
        first = False
        if len(ids) and stamp >= GAP[1] and not resumed:
            resumed = True
            first = True
            for source, tracker in actual.items():
                seed = snapshot(tracker)
                snapshots[f"{source}_resume"] = seed
                variants[source] = make_variants(seed, model)
                chart = coordinate_map(seed["X"])
                branches[source] = Invariant15(seed["X"], chart@seed["P"]@chart.T, seed["bias"])
                for label in variants[source]:
                    trace[source+"__"+label] = np.full((stop+1, 6), np.nan)
            actual["capture"].on_ranges_resumed()
        if resumed:
            for source, branch in branches.items():
                if not first:
                    sample = data["imu"][k-1]
                    u = branch.X[:3, :3]@(sample[:3]*9.81-branch.bias[3:])+np.array([0., 0., -9.81])
                    dt = float(stamp-times[k-1])
                    branch.predict(sample, dt)
                    for variant in variants[source].values():
                        variant.predict(u, dt)
                for anchor in ids:
                    for label, variant in variants[source].items():
                        if label == "prediction_only":
                            continue
                        before = variant.position.copy()
                        accepted = variant.correct(data["anchors"][anchor], data["ranges"][k, anchor], int(anchor))
                        covariance = variant.P[3:6, 3:6] if label.startswith("lifted") else variant.P[:3, :3]
                        measurements.append(dict(sequence=name, source=source, method=label, sample=k,
                            time_s=float(stamp), anchor=int(anchor), accepted=accepted,
                            nis=float(variant.last_nis), position_jump_m=float(np.linalg.norm(variant.position-before)),
                            p_std_max_m=float(np.sqrt(max(np.linalg.eigvalsh(covariance)))),
                            accepted_total=variant.stats.accepted, rejected_total=variant.stats.rejected,
                            **dict(zip(("p_before_x", "p_before_y", "p_before_z"), before)),
                            **dict(zip(("p_after_x", "p_after_y", "p_after_z"), variant.position))))
                for label, variant in variants[source].items():
                    trace[source+"__"+label][k] = np.r_[variant.position, variant.velocity]
        for source, tracker in actual.items():
            if tracker is None:
                continue
            if not (source == "profile" and new_profile):
                for anchor in ids:
                    tracker.correct_twr(data["anchors"][anchor], data["ranges"][k, anchor], int(anchor))
            histories[source][k] = np.r_[tracker.X[:3, 4], tracker.X[:3, 3]]
    # Scoring is strictly downstream of every estimator and fixed control.
    truth = np.load(CACHE/(Path(name).stem+"_scoring.npz"))["position"][:stop+1]
    rows = []
    outputs = {**trace, **{"actual__"+key: value for key, value in histories.items()}}
    for label, values in outputs.items():
        error = np.linalg.norm(values[selected, :2]-truth[selected, :2], axis=1)
        rows.append(dict(sequence=name, method=label, early_xy_rmse_m=float(np.sqrt(np.mean(error**2))),
            early_xy_peak_m=float(np.max(error)), first_xy_error_m=float(error[0]),
            finite_coverage=float(np.mean(np.isfinite(error)))))
    original = np.load(ROOT/"agent/codex/paper_baseline_check/capture_histories"/(Path(name).stem+"__capture__outage1.npz"))
    audit = dict(sequence=name, original_capture_max_difference_m=float(np.nanmax(np.abs(
        histories["capture"]-original["output"][:stop+1]))),
        conditional_capture_max_difference_m=float(np.nanmax(np.abs(
        histories["capture"][selected]-trace["capture__lifted_warm256"][selected]))),
        resume_time_s=float(times[selected[0]]), early_samples=len(selected))
    assert audit["original_capture_max_difference_m"] < 1e-7, audit
    assert audit["conditional_capture_max_difference_m"] < 1e-7, audit
    payload = dict(time=times[:stop+1], truth=truth, **outputs)
    for key, state in snapshots.items():
        for field, value in state.items():
            payload[key+"__"+field] = value
    np.savez_compressed(OUT/(Path(name).stem+".npz"), **payload)
    pd.DataFrame(measurements).to_csv(OUT/(Path(name).stem+"_measurements.csv"), index=False)
    print(json.dumps(audit), flush=True)
    return rows, audit


def main():
    """Run the six fixed small diagnostics on preferred physical CPU cores."""
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((CODE/"results/roadmap_stage1_20261104_final/protocol.json").read_text(encoding="utf-8"))
    with replay_pool(6, workers=6) as pool:
        results = list(pool.map(diagnostic_case, protocol["evaluation"]))
    frame = pd.DataFrame([row for rows, _ in results for row in rows])
    frame.to_csv(OUT/"paired_diagnostics.csv", index=False)
    frame.groupby("method").agg(early_xy_rmse_median_m=("early_xy_rmse_m", "median"),
        early_xy_peak_max_m=("early_xy_peak_m", "max"),
        first_xy_error_median_m=("first_xy_error_m", "median")).to_csv(OUT/"summary.csv")
    (OUT/"audit.json").write_text(json.dumps([audit for _, audit in results], indent=2), encoding="utf-8")
    print(frame.groupby("method").early_xy_rmse_m.median().to_string())


if __name__ == "__main__":
    main()
