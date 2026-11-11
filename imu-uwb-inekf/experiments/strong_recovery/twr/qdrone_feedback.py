"""Exploratory one-shot QDrone feedback through the unchanged native event loop.

Only the verified anchor cohort is accepted. The feedback wrapper delegates
every predict/update/supervision call to the original full capture reference.
"""

from pathlib import Path
import argparse
import hashlib
import json
import sys
import traceback

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Inputs are the corrected verified cohort, never the invalidated anchor results.
CODE = Path(__file__).resolve().parents[3]
ROOT = CODE.parent
HERE = Path(__file__).resolve().parent
EXTERNAL = HERE.parent/"external_twr"
VERIFIED = ROOT/"agent/codex/strong_external/qdrone_verified"
OUT = ROOT/"agent/codex/strong_external/qdrone_feedback_verified"
sys.path.insert(0, str(EXTERNAL))
import qdrone_run as original_run
import replay as native_replay
from adapter import RANGE_SIGMA_M
from qdrone import truth_at
sys.path.insert(0, str(HERE))
from feedback_lifted import apply_xy_rebase
from feedback import single_offset
from cpu_replay import replay_pool

# Exact verified surveyed geometry is mandatory; this is provenance, not an error gate.
ANCHOR_SHA256 = "9b7d90cf9b2d37098d73d6a8d08179bcf66d5efef5b0806855fdb9b0c73b2720"
RESUME = 25.0
LENGTHS = (2., 5., 10., 20.)


class FeedbackReference:
    """Delegate the real native loop while intercepting only completed radio updates."""

    def __init__(self, reference, record, resume):
        """Track exact native emission timestamps without accumulating a second clock."""
        self.reference, self.record, self.resume = reference, record, float(resume)
        end = resume+11.
        edges = np.unique(np.r_[0., resume-np.asarray(native_replay.LENGTHS),
                                resume, resume+np.array([2., 5., 6., 11.])])
        timeline = np.unique(np.r_[record.imu_time[record.imu_time <= end],
                                   record.radio_time[record.radio_time <= end], edges])
        self.timeline = timeline[timeline >= record.imu_time[0]]
        self.index, self.trigger = 0, None
        self.clock = float(self.timeline[0])

    def __getattr__(self, name):
        """Expose the unchanged supervisor's state, statistics and resume method."""
        return getattr(self.reference, name)

    def predict(self, sample, dt):
        """Use the original propagation and the original loop's exact next timestamp."""
        expected = float(self.timeline[self.index+1]-self.timeline[self.index])
        assert dt == expected
        result = self.reference.predict(sample, dt)
        self.index += 1
        self.clock = float(self.timeline[self.index])
        return result

    def correct_twr(self, anchor, measurement, anchor_index=None):
        """After the native update, apply at most one current-packet internal XY rebase."""
        accepted = self.reference.correct_twr(anchor, measurement, anchor_index)
        if self.trigger is None and self.resume <= self.clock < self.resume+2:
            point = self.reference.X[:3, 4].copy()
            radius = float(self.record.metadata.get("bounds_lever_radius_m", np.linalg.norm(self.record.lever)))
            delta = single_offset(self.clock, point, radius, anchor, None, measurement, RANGE_SIGMA_M)
            if np.any(delta[:2]):
                audit = apply_xy_rebase(self.reference, delta[:2])
                self.trigger = dict(time_s=self.clock, delay_s=self.clock-self.resume,
                    anchor=int(anchor_index), mode=self.reference.mode, update_accepted=bool(accepted),
                    point_before=point.tolist(), point_after=self.reference.X[:3, 4].tolist(), audit=audit)
        return accepted


def run_feedback(record, length):
    """Call the original reference_replay itself with one delegated factory wrapper."""
    factory = native_replay.initial_reference
    wrapped = []

    def feedback_factory(current):
        """Wrap the exact same original initializer without changing its sensor prior."""
        reference = FeedbackReference(factory(current), current, RESUME)
        wrapped.append(reference)
        return reference

    native_replay.initial_reference = feedback_factory
    try:
        result, events, audit = native_replay.reference_replay(record, length, resume=RESUME)
    finally:
        native_replay.initial_reference = factory
    assert len(wrapped) == 1
    assert np.array_equal(result["time"], wrapped[0].timeline)
    audit.update(trigger=wrapped[0].trigger, native_event_loop_called_directly=True,
        full_reference_preserved=False, display_does_not_change_reference=False,
        feedback="One internal strict lifted XY rebase after first violated current packet; no return to K")
    return result, events, audit


def score_pair(record, feedback, baseline, truth, entry, length):
    """Use the verified uniform scoring support without resetting feedback after two seconds."""
    clock = RESUME+np.arange(0., 11., original_run.SCORE_DT)
    ground, supported = truth_at(clock+record.origin_s, truth)
    rows = []
    for method, result in (("K", baseline), ("one_shot_feedback", feedback)):
        indices = np.clip(np.searchsorted(result["time"], clock, side="right")-1, 0, len(result["time"])-1)
        position = result["K"][indices]
        error = position-ground
        for label, begin, end in original_run.WINDOWS:
            window = (clock >= RESUME+begin) & (clock < RESUME+end)
            valid = supported & window & np.isfinite(error).all(axis=1)
            selected = error[valid]
            coverage = float(np.sum(valid)/max(1, np.sum(window)))
            rows.append(dict(site=entry["site"], sequence=entry["name"], length_s=length,
                method=method, window=label, truth_coverage=float(np.mean(supported[window])),
                evaluated_fraction=coverage, complete_window=bool(coverage >= original_run.MIN_SCORE_COVERAGE),
                samples=int(np.sum(valid)),
                xy_rmse_m=float(np.sqrt(np.mean(np.sum(selected[:, :2]**2, axis=1)))) if len(selected) else None,
                z_rmse_m=float(np.sqrt(np.mean(selected[:, 2]**2))) if len(selected) else None,
                xy_peak_m=float(np.max(np.linalg.norm(selected[:, :2], axis=1))) if len(selected) else None))
    feedback["score_time"], feedback["score_truth"], feedback["score_truth_supported"] = clock, ground, supported
    return rows


def run_case(job):
    """Preserve every eligible external record, including native failures and missing GT."""
    entry, length = job
    identifier = entry["site"]+"__"+entry["name"]
    record = original_run.load_record(VERIFIED/"cache"/(identifier+".npz"))
    result, events, audit = run_feedback(record, length)
    # Cached baseline artifacts also contain scores, so open them only after replay.
    baseline = dict(np.load(VERIFIED/(identifier+f"__L{int(length)}.npz")))
    assert np.array_equal(result["time"], baseline["time"])
    trigger = audit["trigger"]
    prefix = result["time"] < (trigger["time_s"] if trigger is not None else np.inf)
    differences = {}
    for field in ("X", "bias", "P_diagonal"):
        equal = np.array_equal(result[field][prefix], baseline[field][prefix], equal_nan=True)
        assert equal, (identifier, length, field)
        differences[field] = equal
    audit["pretrigger_bitwise_equal_to_verified_K"] = differences
    if trigger is None:
        audit["no_trigger_bitwise_equal_to_verified_K"] = True
        snapshot_keys = [key for key in baseline if key in result and
            (key.endswith("_X") or key.endswith("_P") or key.endswith("_bias") or key.endswith("_chi"))]
        audit["no_trigger_full_snapshot_arrays_equal"] = all(
            np.array_equal(result[key], baseline[key], equal_nan=True) for key in snapshot_keys)
        assert audit["no_trigger_full_snapshot_arrays_equal"]
    # Paired baseline arrays are retained alongside the true continued feedback output.
    result["original_K"] = baseline["K"]
    result["original_K_velocity"] = baseline["K_velocity"]
    jumps = np.r_[0., np.linalg.norm(np.diff(result["K"][:, :2], axis=0), axis=1)]
    original_jumps = np.r_[0., np.linalg.norm(np.diff(baseline["K"][:, :2], axis=0), axis=1)]
    scope = result["time"] >= RESUME
    audit["post_xy_jump_max_m"] = float(np.nanmax(jumps[scope]))
    audit["original_post_xy_jump_max_m"] = float(np.nanmax(original_jumps[scope]))
    # No truth is opened until the complete native feedback loop has finished.
    truth = np.load(VERIFIED/"cache"/(identifier+"_scoring.npz"))["truth"]
    rows = score_pair(record, result, baseline, truth, entry, length)
    case = identifier+f"__L{int(length)}"
    np.savez_compressed(OUT/(case+".npz"), **result)
    pd.DataFrame(events).to_csv(OUT/(case+"_events.csv"), index=False)
    native_replay.write_json(OUT/(case+".json"), audit)
    for row in rows:
        row.update(triggered=trigger is not None, trigger_delay_s=trigger["delay_s"] if trigger else None,
            native_failure=audit["failure"] is not None)
    print(case, "trigger", None if trigger is None else trigger["delay_s"], "failure", audit["failure"] is not None, flush=True)
    return rows, dict(site=entry["site"], sequence=entry["name"], length_s=length,
        trigger=trigger, failure=audit["failure"], post_xy_jump_max_m=audit["post_xy_jump_max_m"],
        original_post_xy_jump_max_m=audit["original_post_xy_jump_max_m"])


def main():
    """Replay all ten verified sensor-eligible recordings with at most four workers."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((VERIFIED/"protocol.json").read_text(encoding="utf-8"))
    assert protocol["anchor_file_sha256"] == ANCHOR_SHA256
    cohort = json.loads((VERIFIED/"eligibility.json").read_text(encoding="utf-8"))
    selected = [entry for entry in cohort if entry["status"] == "sensor_eligible"]
    jobs = [(entry, length) for entry in selected for length in LENGTHS]
    manifest = dict(exploratory=True, original_verified_input=str(VERIFIED), anchor_sha256=ANCHOR_SHA256,
        resume_s=RESUME, lengths_s=LENGTHS, recordings=len(selected), cases=len(jobs),
        no_score_based_cohort_selection=True, original_native_loop_called=True,
        source_sha256={str(path.relative_to(CODE)):hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (EXTERNAL/"replay.py", EXTERNAL/"adapter.py", HERE/"feedback_lifted.py", Path(__file__))})
    native_replay.write_json(OUT/"protocol.json", manifest)
    rows, audits, failures = [], [], []
    from concurrent.futures import as_completed
    with replay_pool(len(jobs), workers=min(4, args.workers)) as pool:
        futures = {pool.submit(run_case, job):job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            try:
                metrics, audit = future.result()
                rows.extend(metrics)
                audits.append(audit)
            except Exception:
                failures.append(dict(site=job[0]["site"], sequence=job[0]["name"], length_s=job[1],
                                     traceback=traceback.format_exc()))
            pd.DataFrame(rows).to_csv(OUT/"paired_metrics.csv", index=False)
            native_replay.write_json(OUT/"audit.json", audits)
            native_replay.write_json(OUT/"failures.json", failures)
    print("Completed", len(audits), "failures", len(failures), flush=True)
    finalize_results()


def finalize_results():
    """Separate supported pairs and preserve every loss, trigger, and missing-GT record."""
    frame = pd.read_csv(OUT/"paired_metrics.csv")
    complete = frame[frame.complete_window].copy()
    summary = complete.groupby(["length_s", "window", "method"]).agg(
        recordings=("xy_rmse_m", "size"), xy_rmse_median_m=("xy_rmse_m", "median"),
        xy_rmse_worst_m=("xy_rmse_m", "max"), xy_peak_worst_m=("xy_peak_m", "max"),
        z_rmse_median_m=("z_rmse_m", "median"))
    summary.to_csv(OUT/"supported_summary.csv")
    changes = []
    index = ["site", "sequence", "length_s", "window"]
    pivot = complete.pivot(index=index, columns="method", values="xy_rmse_m").dropna()
    pivot["difference_m"] = pivot.one_shot_feedback-pivot.K
    pivot.to_csv(OUT/"supported_paired_changes.csv")
    for (length, window), group in pivot.reset_index().groupby(["length_s", "window"]):
        delta = group.difference_m
        changes.append(dict(length_s=length, window=window, supported_pairs=len(group),
            median_difference_m=float(delta.median()), wins=int(np.sum(delta < -1e-12)),
            ties=int(np.sum(np.abs(delta) <= 1e-12)), losses=int(np.sum(delta > 1e-12)),
            worst_increase_m=float(delta.max()), best_difference_m=float(delta.min())))
    pd.DataFrame(changes).to_csv(OUT/"paired_summary.csv", index=False)
    raw_audits = json.loads((OUT/"audit.json").read_text(encoding="utf-8"))
    triggers, no_trigger_full, pretrigger = [], [], []
    for audit in raw_audits:
        identifier = audit["site"]+"__"+audit["sequence"]+f"__L{int(audit['length_s'])}"
        detail = json.loads((OUT/(identifier+".json")).read_text(encoding="utf-8"))
        trigger = audit["trigger"]
        triggers.append(dict(site=audit["site"], sequence=audit["sequence"], length_s=audit["length_s"],
            triggered=trigger is not None, delay_s=trigger["delay_s"] if trigger else None,
            delta_xy_m=float(np.linalg.norm(trigger["audit"]["delta_xy"])) if trigger else 0.,
            defect_change_max=trigger["audit"]["defect_change_max"] if trigger else 0.,
            physical_covariance_relative_change=trigger["audit"]["physical_covariance_relative_change"] if trigger else 0.,
            post_xy_jump_max_m=audit["post_xy_jump_max_m"],
            original_post_xy_jump_max_m=audit["original_post_xy_jump_max_m"]))
        pretrigger.append(all(detail["pretrigger_bitwise_equal_to_verified_K"].values()))
        if trigger is None:
            no_trigger_full.append(detail["no_trigger_full_snapshot_arrays_equal"])
    pd.DataFrame(triggers).to_csv(OUT/"triggers.csv", index=False)
    # Check the independently loaded official score support against the verified primary K rows.
    primary = pd.read_csv(VERIFIED/"metrics.csv")
    baseline = frame[frame.method == "K"]
    primary = primary[primary.method == "K"]
    merged = baseline.merge(primary, on=index, suffixes=("_new", "_verified"))
    max_difference = {}
    for column in ("xy_rmse_m", "z_rmse_m", "xy_peak_m", "evaluated_fraction"):
        a, b = merged[column+"_new"].to_numpy(float), merged[column+"_verified"].to_numpy(float)
        assert np.array_equal(np.isnan(a), np.isnan(b))
        max_difference[column] = float(np.max(np.abs(a[np.isfinite(a)]-b[np.isfinite(b)])))
        assert max_difference[column] < 1e-10
    verification = dict(cases=len(raw_audits), failed_cases=len(json.loads((OUT/"failures.json").read_text())),
        triggered=sum(row["triggered"] for row in triggers), no_trigger=len(no_trigger_full),
        all_pretrigger_X_bias_Pdiagonal_bitwise_equal=all(pretrigger),
        all_no_trigger_full_native_snapshots_bitwise_equal=all(no_trigger_full),
        baseline_score_rows_matched=len(merged), baseline_metric_max_difference=max_difference,
        native_loop_ends_s=36., feedback_does_not_return_to_K=True,
        missing_truth_never_filled=True)
    native_replay.write_json(OUT/"final_verification.json", verification)
    figure, axes = plt.subplots(1, 4, figsize=(12, 3.5), sharey=True)
    early = pivot.reset_index()
    early = early[early.window == "first2"]
    for axis, length in zip(axes, LENGTHS):
        group = early[early.length_s == length]
        for _, row in group.iterrows():
            axis.plot([0, 1], [row.K, row.one_shot_feedback], marker="o", ms=4, lw=.8)
        axis.set_xticks([0, 1], ["Original K", "State feedback"], rotation=30)
        axis.set_title(f"{int(length)} s gap; n={len(group)}")
        axis.set_yscale("log")
        axis.grid(alpha=.2)
    axes[0].set_ylabel("First 2 s XY RMSE (m)")
    figure.suptitle("Verified QDrone T=25: supported pairs, one-shot internal feedback")
    figure.tight_layout()
    figure.savefig(OUT/"supported_early_xy.png", dpi=170)
    plt.close(figure)
    print(pd.DataFrame(changes).to_string(index=False))


if __name__ == "__main__":
    main()
