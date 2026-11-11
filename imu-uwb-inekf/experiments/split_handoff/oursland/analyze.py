"""Pair native-author and recomputed split-policy outcomes without selecting successes."""

import argparse
import json
from pathlib import Path
import sys

# Scoring and figures write only inside this experiment.
sys.dont_write_bytecode = True
import run as ours
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Metrics use absolute world coordinates; negative deltas favor the current policy.
METRICS = ["post_xy_rmse", "post_z_rmse", "post_yaw_rmse_deg", "whole_xy_rmse"]
# These columns identify an independently replayable paired condition.
KEYS = ["name", "scenario", "yaw_offset_deg"]
# Display the actual retained/full/gated designs beside normal recursive filtering.
POLICIES = ["normal_unmodified", "retain_normal", "full_attitude", "evidence_yaw", "bank_yaw"]


def compare(author, current, output, label):
    """Retain all per-case pairs, failure flags, and component-wise changes."""
    baseline = author[KEYS + METRICS + ["failed", "post_samples"]]
    joined = current.merge(baseline, on=KEYS, how="left", validate="many_to_one", suffixes=("", "_author"))
    if joined[METRICS[0] + "_author"].isna().any():
        raise ValueError("Missing native-author counterpart; never silently drop cases")
    for metric in METRICS:
        joined[metric + "_delta"] = joined[metric] - joined[metric + "_author"]
    joined["pair_complete"] = (~joined.failed.astype(bool) & ~joined.failed_author.astype(bool)
                               & (joined.post_samples == joined.post_samples_author))
    joined.to_csv(output / (label + "_paired_cases.csv"), index=False)
    summary = []
    for (scenario, policy), group in joined.groupby(["scenario", "policy"]):
        row = dict(scenario=scenario, policy=policy, cases=len(group),
                   current_failures=int(group["failed"].sum()), author_failures=int(group["failed_author"].sum()),
                   complete_pairs=int(group.pair_complete.sum()))
        for metric in METRICS:
            row[metric + "_median"] = float(group[metric].median())
            row[metric + "_author_median"] = float(group[metric + "_author"].median())
            row[metric + "_delta_median"] = float(group[metric + "_delta"].median())
            row[metric + "_wins"] = int(((group[metric + "_delta"] < 0) & group.pair_complete).sum())
            row[metric + "_complete_median"] = float(group.loc[group.pair_complete, metric].median())
            row[metric + "_complete_delta_median"] = float(group.loc[group.pair_complete, metric + "_delta"].median())
        summary.append(row)
    pd.DataFrame(summary).to_csv(output / (label + "_paired_summary.csv"), index=False)
    return joined


def main():
    """Publish native paired tables and transparent original-main/stress diagnostics."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior", choices=["common", "original"], default="original")
    args = parser.parse_args()
    original_prior = args.prior == "original"
    native_dir = ours.HERE / ("native_original_prior" if original_prior else "native_paired")
    policies_dir = ours.HERE / ("native_policies_original_prior" if original_prior else "native_policies")
    initialization = "sensor_only_original_prior" if original_prior else "sensor_only"
    output = ours.HERE / ("analysis_original_prior" if original_prior else "analysis")
    output.mkdir(parents=True, exist_ok=True)
    native = pd.read_csv(native_dir / "metrics.csv")
    author = native[native.variant == "author"]
    current = pd.read_csv(policies_dir / "metrics.csv")
    joined = compare(author, current, output, "native")
    direct = native[native.variant == "current_kernel"].copy()
    direct["policy"] = "current_kernel_direct"
    compare(author, direct, output, "native_direct")
    if original_prior:
        checks = []
        for row in author[author.scenario == "cold_weak"].itertuples():
            original = np.load(native_dir / (row.key + ".npz"))
            policy_key = f"{row.name}__cold_weak__yaw{row.yaw_offset_deg:g}__retain_normal"
            policy = np.load(policies_dir / (policy_key + ".npz"))
            initial = policy["window_start_state"]
            change = ours.util.kernel.mapping(initial)
            reconstructed = change @ policy["handoff_physical_covariance"] @ change.T
            state_error = float(np.max(np.abs(initial - original["initial_state"])))
            covariance_error = float(np.max(np.abs(reconstructed - original["initial_covariance"])))
            np.testing.assert_allclose(initial, original["initial_state"], rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(reconstructed, original["initial_covariance"], rtol=1e-12, atol=1e-12)
            checks.append(dict(name=row.name, yaw=row.yaw_offset_deg, initial_X_max_abs=state_error,
                               original_P_roundtrip_max_abs=covariance_error))
        (output / "original_prior_initialization_checks.json").write_text(json.dumps(checks, indent=2), encoding="utf8")
    commands = []
    for name in sorted(author.name.unique()):
        prefix = "& '.\\.venv\\Scripts\\python.exe' '.\\imu-uwb-inekf\\experiments\\split_handoff\\oursland\\"
        commands.append(dict(name=name,
            author_and_direct=prefix + f"native_schedule.py' --initialization {initialization} --names {name} --yaws 0 180 --output rerun_native_{args.prior}_{name}",
            handoff_policies=prefix + f"native_policies.py' --initial-prior {args.prior} --names {name} --yaws 0 180 --output rerun_policies_{args.prior}_{name}"))
    pd.DataFrame(commands).to_csv(output / "recording_rerun_commands.csv", index=False)
    author.to_csv(output / "native_author_cases.csv", index=False)
    overview = []
    for scenario, group in author.groupby("scenario"):
        overview.append(dict(scenario=scenario, policy="author_original_continuous", cases=len(group),
            failures=int(group.failed.sum()), xy_median=float(group.post_xy_rmse.median()),
            z_median=float(group.post_z_rmse.median()), yaw_median_deg=float(group.post_yaw_rmse_deg.median())))
    for (scenario, policy), group in current.groupby(["scenario", "policy"]):
        accepted = group.heading_accepted.astype(str).str.lower().eq("true").sum() if "heading_accepted" in group else 0
        overview.append(dict(scenario=scenario, policy=policy, cases=len(group), failures=int(group.failed.sum()),
            heading_acceptances=int(accepted), xy_median=float(group.post_xy_rmse.median()),
            z_median=float(group.post_z_rmse.median()), yaw_median_deg=float(group.post_yaw_rmse_deg.median())))
    (output / "overview.json").write_text(json.dumps(dict(prior=args.prior, cases=overview), indent=2), encoding="utf8")
    # The full 16-recording 100Hz adapter is reported separately, never as paper superiority.
    stress_path = ours.HERE / "holdout" / "metrics.csv"
    if stress_path.exists():
        stress = pd.read_csv(stress_path)
        stress.groupby("scenario")[["post_xy_rmse", "post_z_rmse", "post_yaw_rmse_deg", "post_coverage"]].agg(["median", "max"]).to_csv(output / "stress_100hz_summary.csv")
    reference = pd.read_csv(ours.HERE / "native_smoke" / "metrics.csv")
    reference.to_csv(output / "original_main_health_reference.csv", index=False)
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for axis, metric, unit in zip(axes, METRICS[:3], ["m", "m", "deg"]):
        for index, policy in enumerate(POLICIES):
            group = joined[joined.policy == policy]
            offsets = np.linspace(-.14, .14, len(group))
            axis.scatter(index + offsets, group[metric + "_delta"], s=16, alpha=.7)
            invalid = ~group.pair_complete.to_numpy()
            axis.scatter(index + offsets[invalid], group.loc[~group.pair_complete, metric + "_delta"], marker="x", color="red", s=30)
            median = group.loc[group.pair_complete, metric + "_delta"].median()
            axis.plot([index - .25, index + .25], [median] * 2, color="black", lw=2)
        axis.axhline(0, color="gray", lw=.8)
        axis.set_xticks(range(len(POLICIES)), [p.replace("_", "\n") for p in POLICIES], fontsize=8)
        axis.set_ylabel(f"Current minus original author [{unit}]")
        axis.set_title(metric.replace("post_", "Post ").replace("_", " "))
        axis.grid(axis="y", alpha=.25)
    figure.suptitle(f"Native author queue, {args.prior} prior: 4 recordings × 2 headings × 3 conditions; red x = failed/incomplete")
    figure.tight_layout()
    figure.savefig(output / "native_paired_deltas.png", dpi=180)
    plt.close(figure)
    # A predetermined difficult const4 opposite-heading recovery case illustrates actual trajectories.
    name, scenario, yaw = "const4-trial4-tdoa3-traj1", "outage", 180
    truth = ours.util.load(name, duration=45.)["truth"]
    origin = float(author[author.name == name].iloc[0].origin)
    truth = truth[(truth[:, 0] >= origin + 26) & (truth[:, 0] <= origin + 45)]
    records = json.loads((policies_dir / "records.json").read_text(encoding="utf8"))
    selected = [r for r in records if r["name"] == name and r["scenario"] == scenario and r["yaw_offset_deg"] == yaw and r["policy"] in POLICIES]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].plot(truth[:, 1], truth[:, 2], "k--", lw=1, label="Vicon")
    axes[1].plot(truth[:, 0] - author[author.name == name].iloc[0].origin, truth[:, 3], "k--", lw=1)
    axes[2].plot(truth[:, 0] - author[author.name == name].iloc[0].origin, ours.Rotation.from_quat(truth[:, 4:8]).as_euler("xyz")[:, 2] * 180 / np.pi, "k--", lw=1)
    for record in selected:
        rows = np.load(policies_dir / (record["key"] + ".npz"))["after"]
        axes[0].plot(rows[:, 1], rows[:, 2], lw=.7, label=record["policy"])
        axes[1].plot(rows[:, 0] - record["origin"], rows[:, 3], lw=.7, label=record["policy"])
        rotation = ours.Rotation.from_matrix(rows[:, 7:16].reshape(-1, 3, 3))
        axes[2].plot(rows[:, 0] - record["origin"], rotation.as_euler("xyz")[:, 2] * 180 / np.pi, lw=.7)
    original_key = f"{name}__yaw180__outage__{initialization}__author"
    rows = np.load(native_dir / (original_key + ".npz"))["rows"]
    rows = rows[rows[:, 0] > origin + 26]
    axes[0].plot(rows[:, 1], rows[:, 2], lw=1.2, label="original author")
    axes[1].plot(rows[:, 0] - author[author.name == name].iloc[0].origin, rows[:, 3], lw=1.2, label="original author")
    axes[2].plot(rows[:, 0] - author[author.name == name].iloc[0].origin, ours.Rotation.from_matrix(rows[:, 7:16].reshape(-1, 3, 3)).as_euler("xyz")[:, 2] * 180 / np.pi, lw=1.2)
    axes[0].set(xlabel="X [m]", ylabel="Y [m]", title="XY after recovery handoff")
    axes[0].axis("equal")
    axes[1].set(xlabel="Time [s]", ylabel="Z [m]", title="Vertical trajectory")
    axes[2].set(xlabel="Time [s]", ylabel="Yaw [deg]", title="Physical yaw")
    axes[0].legend(fontsize=7)
    for axis in axes:
        axis.grid(alpha=.25)
    figure.tight_layout()
    figure.savefig(output / "const4_opposite_heading_recovery.png", dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
