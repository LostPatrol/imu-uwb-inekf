"""Summarize frozen conditional-replacement trials, failures and causal emissions."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# All evidence is generated within the owned research directory.
HERE = Path(__file__).resolve().parent
RESULTS = HERE/"results"
# These descriptive tolerances avoid counting floating-point jitter as a benefit.
TIE_M = 1e-6


def main():
    """Aggregate only common support while keeping cohort/failure counts visible."""
    paths = sorted(RESULTS.glob("*__L*.json"))
    metrics, diagnostics, failures = [], [], []
    for path in paths:
        case = json.loads(path.read_text(encoding="utf-8"))
        metrics.extend(case["metrics"])
        domain, sequence, gap = path.stem.split("__", 1)[0], case["metrics"][0]["sequence"], case["metrics"][0]["gap_s"]
        for method, audit in case["audits"].items():
            if audit["failure"] is not None:
                failures.append(dict(case=path.stem, method=method, **audit["failure"]))
            for event in audit["events"]:
                detail = event.get("detail", {})
                last = detail.get("iterations", [{}])[-1]
                diagnostics.append(dict(domain=domain, sequence=sequence, gap_s=gap, method=method,
                    trigger_s=event["trigger_s"], handoff_s=event["handoff_s"],
                    algorithmic_wait_s=event["delay_s"], supported=event["supported"],
                    checkpoint_s=event["checkpoint_s"], packet_factors=event["packet_factors"],
                    compute_wall_s=detail.get("elapsed_s"), profile_used=detail.get("profile_used"),
                    profile_yaw_rad=(detail.get("parameters") or [None]*7)[6],
                    final_endpoint_delta_norm=last.get("endpoint_delta_norm"),
                    final_endpoint_rotation_delta_deg=last.get("endpoint_rotation_delta_deg"),
                    covariance_min_eigenvalue=detail.get("covariance_min_eigenvalue"),
                    covariance_navigation_bias_cross_norm=detail.get("full_cross_block_norm"),
                    endpoint_position_change_m=event.get("endpoint_position_change_m"),
                    endpoint_attitude_change_deg=event.get("endpoint_attitude_change_deg")))
    frame = pd.DataFrame(metrics)
    frame.to_csv(RESULTS/"metrics.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(RESULTS/"solver_diagnostics.csv", index=False)
    (RESULTS/"estimator_failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
    summary = frame.groupby(["domain", "gap_s", "method", "window"]).agg(
        attempted_recordings=("sequence", "size"), supported_recordings=("complete_window", "sum"),
        failed_recordings=("failed", "sum"), median_available_xy_m=("xy_rmse_m", "median"), worst_available_xy_m=("xy_rmse_m", "max"))
    summary.to_csv(RESULTS/"all_support_summary.csv")
    supported = frame[frame.complete_window & ~frame.failed]
    supported.groupby(["domain", "gap_s", "method", "window"]).agg(
        recordings=("sequence", "size"), median_xy_m=("xy_rmse_m", "median"),
        worst_xy_m=("xy_rmse_m", "max")).to_csv(RESULTS/"supported_summary.csv")
    paired = []
    for (domain, gap, window), group in frame.groupby(["domain", "gap_s", "window"]):
        pivot = group[group.complete_window & ~group.failed].pivot(index="sequence", columns="method", values="xy_rmse_m")
        for method in ("ieks_local", "ieks_profile"):
            for baseline in ("native", "old_lifted", "ieks_local", "profile_startup_native_recovery"):
                if method == baseline or method not in pivot or baseline not in pivot:
                    continue
                selected = pivot[[method, baseline]].dropna()
                delta = selected[method]-selected[baseline]
                paired.append(dict(domain=domain, gap_s=gap, window=window, method=method, baseline=baseline,
                    paired_recordings=len(delta), missing_or_failed_recordings=group.sequence.nunique()-len(delta),
                    median_difference_m=float(delta.median()) if len(delta) else None,
                    wins=int((delta < -TIE_M).sum()), ties=int((abs(delta) <= TIE_M).sum()),
                    losses=int((delta > TIE_M).sum()), worst_deterioration_m=float(delta.max()) if len(delta) else None))
    pd.DataFrame(paired).to_csv(RESULTS/"paired_changes.csv", index=False)
    plot_delft(paths)
    plot_counterexample()
    print(summary.to_string())
    print("Estimator failures", len(failures), "recording-duration cases", len(paths))


def plot_delft(paths):
    """Display actual persistent filtered errors, with acquisition delay visible."""
    figure, axes = plt.subplots(2, 3, figsize=(12, 6.5), sharex=True)
    chosen = [p for p in paths if p.stem.startswith("delft__") and p.stem.endswith("__L20")]
    for axis, path in zip(axes.flat, chosen):
        data = np.load(path.with_suffix(".npz"))
        clock, ground, support = data["score_time"], data["score_truth"], data["score_truth_supported"]
        for method in ("native", "ieks_local", "ieks_profile", "old_lifted"):
            if method == "old_lifted":
                times, position = data[method+"_time"], data[method+"_position"]
            else:
                times, position = data[method+"_time"], data[method+"_X"][:, :3, 4]
            index = np.maximum(0, np.searchsorted(times, clock, side="right")-1)
            error = np.linalg.norm(position[index, :2]-ground[:, :2], axis=1)
            error[~support] = np.nan
            axis.plot(clock-25., error, label=method, lw=1.1)
        axis.set_title(path.stem.split("_twr_")[-1].split("__")[0])
        axis.set_yscale("log")
        axis.axvline(2., color=".5", ls="--", lw=.8)
        axis.grid(alpha=.25)
    axes[0, 0].legend(fontsize=7)
    for axis in axes[-1]:
        axis.set_xlabel("Time since radio return (s)")
    for axis in axes[:, 0]:
        axis.set_ylabel("XY error (m)")
    figure.suptitle("20 s outage: genuine full-state conditional replacement and continued InEKF")
    figure.tight_layout()
    figure.savefig(RESULTS/"delft_gap20_xy.png", dpi=170)
    plt.close(figure)


def plot_counterexample():
    """Show the external wrong-basin failure against identical startup history."""
    path = RESULTS/"qdrone__building__raw_data7_site2_newmarket_building1__L2.npz"
    if not path.exists():
        return
    data = np.load(path)
    clock, truth = data["score_time"], data["score_truth"]
    figure, axis = plt.subplots(figsize=(8, 4.5))
    for method, label in (("profile_startup_native_recovery", "Identical history; native recovery"),
                          ("ieks_profile", "Profile-seeded conditional replacement"),
                          ("old_lifted", "Original lifted capture/tracker")):
        times = data[method+"_time"]
        position = data[method+"_position"] if method == "old_lifted" else data[method+"_X"][:, :3, 4]
        index = np.maximum(0, np.searchsorted(times, clock, side="right")-1)
        error = np.linalg.norm(position[index, :2]-truth[:, :2], axis=1)
        error[~data["score_truth_supported"]] = np.nan
        axis.plot(clock-25., error, label=label)
    axis.axvline(2., color=".5", ls="--", label="2 s acquisition")
    axis.set_yscale("log")
    axis.set_xlabel("Time since radio return (s)")
    axis.set_ylabel("XY error (m)")
    axis.set_title("QDrone building1, 2 s outage: actual harmful posterior handoff")
    axis.grid(alpha=.25)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(RESULTS/"qdrone_counterexample_xy.png", dpi=170)
    plt.close(figure)


if __name__ == "__main__":
    main()
