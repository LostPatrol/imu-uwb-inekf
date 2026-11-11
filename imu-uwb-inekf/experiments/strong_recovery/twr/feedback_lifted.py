"""Exploratory one-shot horizontal feedback into the original native TWR filter.

The first violated current-packet radio disk requests a lifted translation rebase.
Subsequent recursion is native, with no second correction or return to stored K.
"""

from pathlib import Path
import argparse
import json
import sys

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

# All exploratory artifacts are separate from the frozen display-only matrix.
CODE = Path(__file__).resolve().parents[3]
ROOT = CODE.parent
HERE = Path(__file__).resolve().parent
OUT = ROOT/"agent/codex/strong_twr/feedback_results"
SENSORS = CODE/"experiments/split_handoff/delft/results/cache"
K_CACHE = ROOT/"agent/codex/strong_twr/length_cache"
for folder in (HERE, HERE.parent):
    sys.path.insert(0, str(folder))
import diagnose
from cpu_replay import replay_pool
from radio_bounds import bound_xy
from feedback import single_offset
from evaluate import scores

# These repeat the registered lengths and restoration clock without new selection.
LENGTHS = (2, 5, 10, 20)
RESUME = 25.0
# Physical p/v order is relevant only for the native covariance audit.
PV = np.r_[6:9, 3:6]


def lifting_defect(lifted):
    """Measure existing independent q/s/h offsets from the physical p/v manifold."""
    state = lifted.chi
    p, v = state[3:6], state[6:9]
    return np.array([state[0]-.5*p@p, state[1]-p@v, state[2]-v@v])


def rebase_lifted(lifted, delta_xy):
    """Translate a ten-dimensional homogeneous lifted state and its covariance."""
    delta = np.r_[np.asarray(delta_xy, dtype=float).reshape(2), 0.0]
    mapping = np.eye(10)
    mapping[0, 3:6], mapping[0, 9] = delta, .5*delta@delta
    mapping[1, 6:9] = delta
    mapping[3:6, 9] = delta
    before = lifting_defect(lifted)
    old_pv_covariance = lifted.P[3:9, 3:9].copy()
    lifted.chi = mapping@lifted.chi
    lifted.P = mapping@lifted.P@mapping.T
    assert np.allclose(lifting_defect(lifted), before, rtol=1e-10, atol=1e-9)
    return dict(defect_before=before.tolist(), defect_after=lifting_defect(lifted).tolist(),
        defect_change_max=float(np.max(np.abs(lifting_defect(lifted)-before))),
        pv_covariance_change_max=float(np.max(np.abs(lifted.P[3:9, 3:9]-old_pv_covariance))))


def apply_xy_rebase(estimator, delta_xy):
    """Apply one XY mean rebase, retaining native R/v/bias/Z and physical covariance."""
    delta_xy = np.asarray(delta_xy, dtype=float).reshape(2)
    if not np.all(np.isfinite(delta_xy)):
        raise ValueError("XY translation must be finite")
    if hasattr(estimator, "mode") and hasattr(estimator, "capture"):
        branch = estimator.capture if estimator.mode == "capture" else estimator.tracker
    else:
        branch = estimator
    core = branch.imu_core if hasattr(branch, "lifted") else branch
    if not hasattr(core, "set_translation") or not hasattr(core, "physical_covariance"):
        raise TypeError("Rebase requires the native physical-covariance translation hook")
    old_state, old_bias = branch.X.copy(), branch.bias.copy()
    physical = core.physical_covariance().copy()
    if not np.any(delta_xy):
        return dict(applied=False, physical_covariance_change_max=0.0)
    if hasattr(branch, "lifted"):
        audit = rebase_lifted(branch.lifted, delta_xy)
        branch._synchronize_translation()
    else:
        position = old_state[:3, 4].copy()
        position[:2] += delta_xy
        branch.set_translation(old_state[:3, 3], position)
        audit = {}
    after = core.physical_covariance()
    assert np.array_equal(branch.X[:3, :3], old_state[:3, :3])
    assert np.array_equal(branch.X[:3, 3], old_state[:3, 3])
    assert branch.X[2, 4] == old_state[2, 4]
    assert np.array_equal(branch.bias, old_bias)
    relative = float(np.linalg.norm(after-physical)/(1+np.linalg.norm(physical)))
    assert relative < 1e-8
    audit.update(applied=True, delta_xy=delta_xy.tolist(),
        physical_covariance_change_max=float(np.max(np.abs(after-physical))),
        physical_covariance_relative_change=relative)
    return audit


def self_check(data, model):
    """Check nonmanifold rebase, inverse mapping, zero rebase, and native covariance."""
    tracker = diagnose.make_capture(data, model)
    lifted = tracker.capture.lifted
    # Nonzero defects ensure the test would reject a manifold projection workaround.
    lifted.chi[:3] += np.array([11.0, -7.0, 3.0])
    initial_state, initial_covariance = lifted.chi.copy(), lifted.P.copy()
    initial_core = tracker.capture.imu_core
    zero = apply_xy_rebase(tracker, [0., 0.])
    assert np.array_equal(lifted.chi, initial_state)
    assert np.array_equal(lifted.P, initial_covariance)
    first = apply_xy_rebase(tracker, [1.25, -.75])
    second = apply_xy_rebase(tracker, [-1.25, .75])
    assert np.allclose(lifted.chi, initial_state, rtol=1e-12, atol=1e-12)
    assert np.allclose(lifted.P, initial_covariance, rtol=1e-12, atol=1e-12)
    assert first["defect_change_max"] < 1e-10
    native = apply_xy_rebase(initial_core, [.5, -.25])
    # The public single-packet helper is exactly the originally specified bound call.
    point = np.array([15., -12., 4.])
    stamp, measured, sigma = 25., 5., .2
    direct = bound_xy(np.array([stamp]), point[None], 0., np.array([stamp]),
        data["anchors"][:1], None, np.array([measured]), np.array([sigma]), stamp)
    shared = single_offset(stamp, point, 0., data["anchors"][0], None, measured, sigma)
    assert np.array_equal(shared, direct["pos"][0]-point)
    return dict(zero=zero, forward=first, inverse=second, native=native,
        defect_preserved=True, inverse_restores_lifted_state_and_covariance=True,
        shared_single_offset_exactly_equals_direct_bound=True)


def replay_case(job):
    """Trigger once from the current packet, then continue the original filter forever."""
    name, length = job
    data = dict(np.load(SENSORS/(Path(name).stem+".npz")))
    protocol = json.loads((CODE/"results/roadmap_stage1_20261104_final/protocol.json").read_text(encoding="utf-8"))
    model = protocol["range_models"]["holdout"]
    baseline = np.load(K_CACHE/(Path(name).stem+f"__gap{length}.npz"))
    tracker = diagnose.make_capture(data, model)
    times = data["time"]
    history = np.empty((len(times), 6))
    rotations, biases = np.empty((len(times), 3, 3)), np.empty((len(times), 6))
    trigger, last_event = None, None
    packets = []
    for k, stamp in enumerate(times):
        if k:
            tracker.predict(data["imu"][k-1], float(stamp-times[k-1]))
        usable = data["changed"][k] & (data["ranges"][k] > 0)
        if RESUME-length <= stamp < RESUME:
            usable[:] = False
        events = np.flatnonzero(usable)
        if len(events):
            if last_event is not None and stamp-last_event > 1.0:
                tracker.on_ranges_resumed()
            last_event = stamp
        for anchor in events:
            record = RESUME <= stamp < RESUME+2 and len(packets) < 16
            before = tracker.X.copy() if record else None
            accepted = tracker.correct_twr(data["anchors"][anchor], data["ranges"][k, anchor], int(anchor))
            post_update = tracker.X.copy() if record else None
            wrote_this_packet = False
            if trigger is None and RESUME <= stamp < RESUME+2:
                point = tracker.X[:3, 4].copy()
                delta = single_offset(float(stamp), point, 0., data["anchors"][anchor], None,
                    data["ranges"][k, anchor]-model["range_bias"][anchor], model["range_sigma"][anchor])[:2]
                if np.any(delta):
                    audit = apply_xy_rebase(tracker, delta)
                    trigger = dict(time_s=float(stamp), delay_s=float(stamp-RESUME), sample=k,
                        anchor=int(anchor), update_accepted=bool(accepted), mode=tracker.mode,
                        point_before=point.tolist(), point_after=tracker.X[:3, 4].tolist(), audit=audit)
                    wrote_this_packet = True
            if record:
                branch = tracker.capture if tracker.mode == "capture" else tracker.tracker
                packets.append(dict(time_s=float(stamp), sample=k, anchor=int(anchor),
                    accepted=bool(accepted), nis=float(tracker.last_nis), wrote_xy=wrote_this_packet,
                    before_p=before[:3, 4].tolist(), post_update_p=post_update[:3, 4].tolist(),
                    final_p=tracker.X[:3, 4].tolist(), final_v=tracker.X[:3, 3].tolist(),
                    lifted_chi=branch.lifted.chi.tolist() if hasattr(branch, "lifted") else None,
                    lifting_defect=lifting_defect(branch.lifted).tolist() if hasattr(branch, "lifted") else None))
        history[k] = np.r_[tracker.X[:3, 4], tracker.X[:3, 3]]
        rotations[k], biases[k] = tracker.X[:3, :3], tracker.bias
    original = np.column_stack((baseline["k_pos"], baseline["k_vel"]))
    prior = times < (trigger["time_s"] if trigger is not None else np.inf)
    prefix_difference = float(np.max(np.abs(history[prior]-original[prior])))
    assert prefix_difference == 0.0, (name, length, prefix_difference)
    if trigger is None:
        assert np.array_equal(history, original)
    # Truth is used exclusively after the entire recursive feedback run ends.
    truth = np.load(SENSORS/(Path(name).stem+"_scoring.npz"))["position"]
    identifier = Path(name).stem+f"__gap{length}"
    np.savez_compressed(OUT/(identifier+".npz"), time=times, feedback=history,
        K=original, feedback_rotation=rotations, feedback_bias=biases, truth=truth)
    audit = dict(sequence=name, gap_s=length, trigger=trigger,
        pretrigger_max_difference=prefix_difference, no_trigger_exact_K=bool(trigger is None), first_packets=packets)
    (OUT/(identifier+".json")).write_text(json.dumps(audit, indent=2), encoding="utf-8")
    rows = []
    for method, output in (("K", original), ("one_shot_feedback", history)):
        row = dict(sequence=name, gap_s=length, method=method, triggered=trigger is not None,
            trigger_delay_s=trigger["delay_s"] if trigger else None,
            **scores(times, output[:, :3], truth))
        jumps = np.r_[0., np.linalg.norm(np.diff(output[:, :2], axis=0), axis=1)]
        row["all_post_xy_jump_max_m"] = float(np.max(jumps[times >= RESUME]))
        rows.append(row)
    print(identifier, "trigger", None if trigger is None else trigger["delay_s"], flush=True)
    return rows, audit


def main():
    """Run twenty-four exploratory feedback cases separately with at most four workers."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((CODE/"results/roadmap_stage1_20261104_final/protocol.json").read_text(encoding="utf-8"))
    data = dict(np.load(SENSORS/(Path(protocol["evaluation"][0]).stem+".npz")))
    check = self_check(data, protocol["range_models"]["holdout"])
    (OUT/"rebase_self_check.json").write_text(json.dumps(check, indent=2), encoding="utf-8")
    jobs = [(name, length) for name in protocol["evaluation"] for length in LENGTHS]
    with replay_pool(len(jobs), workers=min(4, args.workers)) as pool:
        results = list(pool.map(replay_case, jobs))
    frame = pd.DataFrame([row for rows, _ in results for row in rows])
    frame.to_csv(OUT/"paired_metrics.csv", index=False)
    (OUT/"audit.json").write_text(json.dumps([audit for _, audit in results], indent=2), encoding="utf-8")
    summary = frame.groupby(["gap_s", "method"]).agg(
        early_xy_median_m=("early0_2_xy_rmse_m", "median"),
        early_xy_peak_max_m=("early0_2_xy_peak_m", "max"),
        next_xy_median_m=("early2_5_xy_rmse_m", "median"),
        future_xy_median_m=("future6_11_xy_rmse_m", "median"),
        future_xy_worst_m=("future6_11_xy_rmse_m", "max"),
        all_post_xy_jump_max_m=("all_post_xy_jump_max_m", "max"))
    summary.to_csv(OUT/"summary.csv")
    paired = []
    for length, group in frame.groupby("gap_s"):
        for metric in ("early0_2_xy_rmse_m", "early2_5_xy_rmse_m", "future6_11_xy_rmse_m"):
            pivot = group.pivot(index="sequence", columns="method", values=metric)
            delta = pivot.one_shot_feedback-pivot.K
            paired.append(dict(gap_s=length, metric=metric, median_difference_m=float(delta.median()),
                wins=int(np.sum(delta < -1e-12)), ties=int(np.sum(np.abs(delta) <= 1e-12)),
                losses=int(np.sum(delta > 1e-12)), worst_deterioration_m=float(delta.max())))
    pd.DataFrame(paired).to_csv(OUT/"paired_changes.csv", index=False)
    print(summary.to_string())
    finalize_results()


def finalize_results():
    """Audit saved no-trigger outputs and expose the persistent feedback failures."""
    audits = json.loads((OUT/"audit.json").read_text(encoding="utf-8"))
    milestones, no_trigger_checks = [], []
    for case in audits:
        identifier = Path(case["sequence"]).stem+f"__gap{case['gap_s']}"
        data, baseline = np.load(OUT/(identifier+".npz")), np.load(K_CACHE/(identifier+".npz"))
        if case["trigger"] is None:
            check = dict(sequence=case["sequence"], gap_s=case["gap_s"],
                pv_bitwise_equal=bool(np.array_equal(data["feedback"], data["K"])),
                rotation_bias_baseline_available="k_rotation" in baseline.files)
            if "k_rotation" in baseline.files:
                check.update(rotation_bitwise_equal=bool(np.array_equal(data["feedback_rotation"], baseline["k_rotation"])),
                    bias_bitwise_equal=bool(np.array_equal(data["feedback_bias"], baseline["k_bias"])))
                assert check["rotation_bitwise_equal"] and check["bias_bitwise_equal"]
            assert check["pv_bitwise_equal"]
            no_trigger_checks.append(check)
        for target in (25., 27., 30., 31., 36.):
            index = int(np.searchsorted(data["time"], target))
            original, modified = data["K"][index], data["feedback"][index]
            row = dict(sequence=case["sequence"], gap_s=case["gap_s"], time_s=float(data["time"][index]),
                k_xy_error_m=float(np.linalg.norm(original[:2]-data["truth"][index, :2])),
                feedback_xy_error_m=float(np.linalg.norm(modified[:2]-data["truth"][index, :2])),
                feedback_minus_k_position_m=float(np.linalg.norm(modified[:3]-original[:3])),
                feedback_minus_k_velocity_mps=float(np.linalg.norm(modified[3:]-original[3:])))
            if "k_rotation" in baseline.files:
                relative = data["feedback_rotation"][index]@baseline["k_rotation"][index].T
                row.update(rotation_difference_deg=float(np.degrees(np.linalg.norm(Rotation.from_matrix(relative).as_rotvec()))),
                    gyro_bias_difference_rad_s=float(np.linalg.norm(data["feedback_bias"][index, :3]-baseline["k_bias"][index, :3])),
                    acc_bias_difference_mps2=float(np.linalg.norm(data["feedback_bias"][index, 3:]-baseline["k_bias"][index, 3:])))
            milestones.append(row)
    pd.DataFrame(milestones).to_csv(OUT/"milestones.csv", index=False)
    (OUT/"no_trigger_checks.json").write_text(json.dumps(no_trigger_checks, indent=2), encoding="utf-8")
    figure, axes = plt.subplots(2, 3, figsize=(11, 6), sharex=True)
    for axis, file in zip(axes.flat, sorted(OUT.glob("*__gap20.npz"))):
        data = np.load(file)
        selected = (data["time"] >= 25) & (data["time"] < 36)
        for method, label in (("K", "Original K"), ("feedback", "One-shot state feedback")):
            error = np.linalg.norm(data[method][:, :2]-data["truth"][:, :2], axis=1)
            axis.plot(data["time"][selected]-25, error[selected], label=label)
        axis.set_title(file.stem.split("_twr_")[1].split("__")[0])
        axis.set_yscale("log")
        axis.axvline(2, color=".7", ls="--", lw=.7)
        axis.grid(alpha=.2)
    axes[0, 0].legend(fontsize=7)
    for axis in axes[-1]:
        axis.set_xlabel("Time since recovery (s)")
    for axis in axes[:, 0]:
        axis.set_ylabel("XY error (m)")
    figure.suptitle("Exploratory 20 s gap feedback: native recursion continues after one rebase")
    figure.tight_layout()
    figure.savefig(OUT/"gap20_feedback_xy.png", dpi=170)
    plt.close(figure)


if __name__ == "__main__":
    main()
