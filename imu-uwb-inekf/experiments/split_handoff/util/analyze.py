"""Audit matched handoffs and visualize every retained real UTIL counterfactual."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Report policies in the same order as the frozen experimental protocol.
POLICIES = ["normal_unmodified", "retain_normal", "full_attitude", "forced_yaw_only", "evidence_yaw", "bank_yaw", "forced_evidence_yaw"]
# Labels describe estimators, not a selected successful subset.
LABELS = ["Normal, no reset", "Shared PV, normal R", "Full pose", "Forced yaw only", "Evidence yaw", "Heading bank", "Forced evidence"]


def audit(folder, records):
    """Check all p/v/bias/physical-P invariants and rejection counterfactual equality."""
    grouped = {}
    for record in records:
        grouped.setdefault((record["name"], record["scenario"], record["yaw_offset_deg"]), []).append(record)
    maxima = dict(position_velocity=0., bias=0., physical_covariance=0., rejected_trajectory=0., yaw_only_gravity=0.)
    rejection_count = 0
    for rows in grouped.values():
        paired = [row for row in rows if row["policy"] != "normal_unmodified"]
        base_record = next(row for row in paired if row["policy"] == "retain_normal")
        base = np.load(folder / (base_record["key"] + ".npz"))
        for record in paired:
            value = np.load(folder / (record["key"] + ".npz"))
            maxima["position_velocity"] = max(maxima["position_velocity"], float(np.max(np.abs(value["handoff"][:3, 3:5] - base["handoff"][:3, 3:5]))))
            maxima["bias"] = max(maxima["bias"], float(np.max(np.abs(value["bias"] - base["bias"]))))
            maxima["physical_covariance"] = max(maxima["physical_covariance"], float(np.max(np.abs(value["handoff_physical_covariance"] - base["handoff_physical_covariance"]))))
            if record["policy"] not in ["full_attitude"]:
                # A left world-yaw rotation leaves the body representation of gravity unchanged.
                maxima["yaw_only_gravity"] = max(maxima["yaw_only_gravity"], float(np.max(np.abs(value["handoff"][2, :3] - base["handoff"][2, :3]))))
            if record["decision"] is not None and not record["decision"].get("accepted", False):
                rejection_count += 1
                maxima["rejected_trajectory"] = max(maxima["rejected_trajectory"], float(np.max(np.abs(value["after"] - base["after"]))))
    assert max(maxima.values()) < 1e-10, maxima
    result = dict(matched_cases=len(grouped), rejected_cases=rejection_count, maximum_differences=maxima)
    (folder / "audit.json").write_text(json.dumps(result, indent=2), encoding="utf8")
    return result


def summaries(folder, frame):
    """Save paired differences and recording-level units without treating injected yaws as new logs."""
    keys = ["name", "scenario", "yaw_offset_deg"]
    baseline = frame[frame.policy == "retain_normal"].set_index(keys)
    rows = []
    for policy in POLICIES:
        values = frame[frame.policy == policy].set_index(keys)
        for metric in ["post_xy_rmse", "post_z_rmse", "post_yaw_rmse_deg", "whole_xy_rmse"]:
            difference = values[metric] - baseline[metric]
            rows.append(dict(policy=policy, metric=metric, median_difference=float(difference.median()),
                             max_worsening=float(difference.max()), best_improvement=float(difference.min()),
                             improved=int((difference < -1e-9).sum()), worsened=int((difference > 1e-9).sum()),
                             unchanged=int((difference.abs() <= 1e-9).sum()), cases=len(difference)))
    pd.DataFrame(rows).to_csv(folder / "paired_to_retain.csv", index=False)
    fields = ["post_xy_rmse", "post_z_rmse", "post_yaw_rmse_deg", "whole_xy_rmse"]
    frame.groupby(["name", "scenario", "policy"])[fields].mean().to_csv(folder / "recording_units.csv")
    heading = frame[frame.policy != "normal_unmodified"]
    record_policy = heading.groupby(["name", "policy"]).post_xy_rmse.mean().unstack("policy")
    record_policy.subtract(record_policy["retain_normal"], axis=0).to_csv(folder / "heading_recording_paired.csv")
    frame.groupby(["scenario", "yaw_offset_deg", "policy"])[fields].agg(["median", "max"]).to_csv(folder / "by_injection.csv")
    frame.groupby(["name", "policy"])[fields].agg(["median", "max"]).to_csv(folder / "by_recording.csv")
    frame.groupby(["scenario", "policy"])["failed"].agg(["sum", "count"]).to_csv(folder / "failures.csv")
    decisions = frame[frame.policy.isin(["evidence_yaw", "bank_yaw"])]
    decisions.groupby(["scenario", "policy"])["heading_accepted"].agg(["sum", "count"]).to_csv(folder / "acceptance.csv")
    decisions[decisions.heading_accepted == True].to_csv(folder / "accepted_examples.csv", index=False)
    # Keep the gate effect distinct from the quality of its underlying candidate.
    controls = frame[frame.policy.isin(["retain_normal", "evidence_yaw", "forced_evidence_yaw"])].pivot(
        index=keys, columns="policy", values=["post_xy_rmse", "post_yaw_rmse_deg"])
    controls.to_csv(folder / "gate_counterfactual.csv")
    decisions.groupby(["scenario", "policy"])[["arrival_yaw_error_deg", "retained_yaw_error_deg", "handoff_yaw_error_deg"]].agg(["median", "max"]).to_csv(folder / "actual_heading_errors.csv")


def complete_period(folder, frame):
    """Separate full-clock scores from partial failure segments without discarding failures."""
    keys = ["name", "scenario", "yaw_offset_deg"]
    final_tick = frame.groupby(keys).post_last_time.transform("max")
    complete = (~frame.failed) & (frame.post_last_time >= final_tick - 1e-9)
    coverage = frame[keys + ["policy", "failed", "handoff_time", "post_last_time", "post_samples"]].copy()
    coverage["evaluation_final_tick"] = final_tick
    coverage["complete_period"] = complete
    coverage.to_csv(folder / "coverage.csv", index=False)
    frame[~complete].to_csv(folder / "failure_examples.csv", index=False)
    masked = frame.copy()
    fields = ["post_xy_rmse", "post_z_rmse", "post_yaw_rmse_deg", "whole_xy_rmse"]
    for field in fields:
        masked["available_segment_" + field] = masked[field]
        masked.loc[~complete, field] = np.nan
    masked["complete_period"] = complete
    masked.to_csv(folder / "complete_period_metrics.csv", index=False)
    masked.groupby(["scenario", "policy"])[fields].agg(["median", "max", "count"]).to_csv(folder / "complete_period_summary.csv")
    base = masked[masked.policy == "retain_normal"].set_index(keys)
    comparisons = []
    for (scenario, policy), values in masked.groupby(["scenario", "policy"]):
        values = values.set_index(keys)
        reference = base.loc[values.index]
        valid = values.complete_period & reference.complete_period
        difference = values.loc[valid, "post_xy_rmse"] - reference.loc[valid, "post_xy_rmse"]
        comparisons.append(dict(scenario=scenario, policy=policy, requested=len(values),
            complete_pairs=int(valid.sum()), failed_or_incomplete=int((~values.complete_period).sum()),
            policy_xy_median=float(values.loc[valid, "post_xy_rmse"].median()),
            retain_xy_median=float(reference.loc[valid, "post_xy_rmse"].median()),
            median_paired_xy_difference=float(difference.median())))
    pd.DataFrame(comparisons).to_csv(folder / "complete_period_paired.csv", index=False)
    return masked


def plot(folder, frame):
    """Show all scenario counterfactuals and their nominal/large-yaw distributions."""
    scenarios = ["cold_weak", "dynamic", "outage"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for column, scenario in enumerate(scenarios):
        subset = frame[frame.scenario == scenario]
        for row, metric, ylabel in [(0, "post_xy_rmse", "Post-handoff XY RMSE (m)"),
                                     (1, "post_yaw_rmse_deg", "Post-handoff yaw RMSE (degrees)")]:
            values = [subset[subset.policy == policy][metric].dropna().to_numpy() for policy in POLICIES]
            axes[row, column].boxplot(values, showfliers=True)
            axes[row, column].set_xticks(np.arange(1, len(POLICIES) + 1))
            labels = []
            for policy, label in zip(POLICIES, LABELS):
                failed = int((~subset[subset.policy == policy].complete_period).sum())
                labels.append(label + (f" ({failed} failed)" if failed else ""))
            axes[row, column].set_xticklabels(labels)
            if row == 0:
                axes[row, column].set_yscale("log")
            axes[row, column].tick_params(axis="x", labelrotation=55, labelsize=8)
            axes[row, column].set_title(scenario)
            axes[row, column].set_ylabel(ylabel)
            axes[row, column].grid(axis="y", alpha=.25)
    fig.suptitle("Real UTIL complete-period scores; incomplete trajectories counted in labels")
    fig.tight_layout()
    fig.savefig(folder / "comparison.png", dpi=160)
    plt.close(fig)
    # Outage paths show actual propagation without radio, followed by recovery and tracking.
    nominal = frame[(frame.scenario == "outage") & (frame.yaw_offset_deg == 180)]
    if len(nominal):
        name = sorted(nominal.name.unique())[0]
        fig, axis = plt.subplots(figsize=(9, 6))
        import run
        truth = run.load(name, duration=run.PROTOCOL["duration_seconds"])["truth"]
        origin = float(nominal[nominal.name == name].origin.iloc[0])
        # The loader retains the full truth log while sensor replay is capped at 45 s.
        truth = truth[(truth[:, 0] >= origin) & (truth[:, 0] <= origin + run.PROTOCOL["duration_seconds"])]
        axis.plot(truth[:, 1], truth[:, 2], "k--", label="Ground truth (score only)", linewidth=1.5)
        for policy, label in zip(POLICIES, LABELS):
            record = nominal[(nominal.name == name) & (nominal.policy == policy)].iloc[0]
            value = np.load(folder / (record["key"] + ".npz"))
            rows = np.vstack([value["before"], value["after"]])
            axis.plot(rows[:, 1], rows[:, 2], label=label, linewidth=1)
        axis.set_aspect("equal", adjustable="datalim")
        axis.set_xlabel("World x (m)")
        axis.set_ylabel("World y (m)")
        axis.set_title(f"{name}: actual 10-second radio outage, initial 180-degree yaw perturbation")
        axis.legend(fontsize=8)
        axis.grid(alpha=.25)
        fig.tight_layout()
        fig.savefig(folder / "outage_trajectory.png", dpi=160)
        plt.close(fig)
        fig, axis = plt.subplots(figsize=(9, 6))
        endpoint = float(nominal[nominal.name == name].handoff_time.iloc[0])
        future_truth = truth[truth[:, 0] >= endpoint]
        axis.plot(future_truth[:, 1], future_truth[:, 2], "k--", label="Ground truth (score only)", linewidth=1.5)
        for policy, label in zip(POLICIES, LABELS):
            record = nominal[(nominal.name == name) & (nominal.policy == policy)].iloc[0]
            value = np.load(folder / (record["key"] + ".npz"))
            rows = value["after"]
            axis.plot(rows[:, 1], rows[:, 2], label=label, linewidth=1)
        axis.set_xlim(future_truth[:, 1].min() - .5, future_truth[:, 1].max() + .5)
        axis.set_ylim(future_truth[:, 2].min() - .5, future_truth[:, 2].max() + .5)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("World x (m)")
        axis.set_ylabel("World y (m)")
        axis.set_title("Post-handoff zoom at truth bounds + 0.5 m\nFull excursions are shown in outage_trajectory.png", fontsize=11)
        axis.legend(fontsize=8)
        axis.grid(alpha=.25)
        fig.tight_layout()
        fig.savefig(folder / "outage_recovery_zoom.png", dpi=160)
        plt.close(fig)


def early_position_accuracy(folder, records):
    """Score actual early independent outputs and the later native initialization."""
    import run
    rows = []
    for record in records:
        if record["scenario"] != "cold_weak" or record["yaw_offset_deg"] != 0 or record["policy"] != "retain_normal":
            continue
        truth = run.load(record["name"], duration=run.PROTOCOL["duration_seconds"])["truth"]
        artifact = np.load(folder / (record["key"] + ".npz"))
        initial_time = record["origin"] + .6
        independent = artifact["before"][artifact["before"][:, 0] <= initial_time + 1e-9]
        estimates = [("independent_radio", row[0], row[1:4]) for row in independent]
        estimates.append(("native_initialization", initial_time, artifact["window_start_state"][:3, 4]))
        for source, stamp, position in estimates:
            actual = np.array([np.interp(stamp, truth[:, 0], truth[:, k]) for k in (1, 2, 3)])
            error = position - actual
            rows.append(dict(name=record["name"], source=source, available_seconds=stamp - record["origin"],
                             xy_error=float(np.linalg.norm(error[:2])), z_error=float(abs(error[2]))))
    pd.DataFrame(rows).to_csv(folder / "early_position_accuracy.csv", index=False)


def main():
    """Audit artifacts and write deterministic tables, plots and source hashes."""
    parser = argparse.ArgumentParser()
    parser.add_argument("folder")
    args = parser.parse_args()
    folder = Path(__file__).parent / args.folder
    records = json.loads((folder / "records.json").read_text(encoding="utf8"))
    frame = pd.read_csv(folder / "metrics.csv")
    print(audit(folder, records))
    summaries(folder, frame)
    complete_frame = complete_period(folder, frame)
    early_position_accuracy(folder, records)
    if args.folder != "smoke":
        plot(folder, complete_frame)
    sources = [Path(__file__), Path(__file__).with_name("run.py"), Path(__file__).with_name("kernel.py"),
               Path(__file__).with_name("protocol.json"), Path(__file__).parents[1] / "yaw_evidence.py",
               Path(__file__).parents[1] / "bank" / "bank.py", Path(__file__).parents[1] / "bank" / "frozen_rules.json"]
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    (folder / "source_hashes.json").write_text(json.dumps(hashes, indent=2), encoding="utf8")


if __name__ == "__main__":
    main()
