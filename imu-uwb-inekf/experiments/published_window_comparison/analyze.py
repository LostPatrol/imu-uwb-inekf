"""Summarize matched author replays with explicit failure denominators.

Native continuous estimates are deduplicated before aggregation. Common-window
errors retain all requested windows and report finite-only accuracy separately.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import driver

# Keep names and endpoint identifiers aligned with the frozen driver protocol.
HERE = Path(__file__).resolve().parent
KEYS = ["sequence", "endpoint_s", "duration_s", "layout"]
GROUPS = ["method", "calibration", "layout"]


def metrics(part):
    """Separate completion and threshold coverage from finite-only RMSE values."""
    finite = part[np.isfinite(part.error_3d_m)]
    result = dict(requested=len(part), completed=int(part.completed.sum()), finite=len(finite),
        below_0p5m=int(part.endpoint_below_0p5m.sum()),
        success_fraction=float(part.endpoint_below_0p5m.mean()),
        completion_fraction=float(part.completed.mean()),
        median_wall_s=float(part.elapsed_s.median()), total_wall_s=float(part.elapsed_s.sum()))
    for name, column in (("xy", "xy_error_m"), ("z", "z_error_m"), ("3d", "error_3d_m")):
        result[f"{name}_rmse_m"] = float(np.sqrt(np.mean(finite[column]**2))) if len(finite) else np.nan
        result[f"{name}_median_m"] = float(finite[column].median())
        result[f"{name}_p95_m"] = float(finite[column].quantile(.95))
    if "nfev" in part:
        result["median_nfev"] = float(part.nfev.median()) if part.nfev.notna().any() else np.nan
    if "solver_success" in part:
        result["solver_reports"] = int(part.solver_success.notna().sum())
        result["solver_converged"] = int(part.solver_success.fillna(False).sum())
    return pd.Series(result)


def grouped(frame, groups, path):
    """Write a group table without pooling separate information conditions."""
    frame.groupby(groups, dropna=False).apply(metrics, include_groups=False).to_csv(path)


def plot_common(frame, output, control="BiasPrior", filename="common_errors.png"):
    """Show common-input horizontal and vertical errors by method and calibration."""
    methods = ("Pfeiffer_MHE", "Pfeiffer_EKF", control)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    for row, layout in enumerate(("all8", "lower4")):
        for col, axis in enumerate(("xy", "z")):
            ax = axes[row, col]
            positions, labels, values = [], [], []
            for method in methods:
                for calibration in ("raw", "static"):
                    subset = frame[(frame.method == method) & (frame.calibration == calibration) & (frame.layout == layout)]
                    values.append(subset[f"{axis}_error_m"].dropna().to_numpy())
                    positions.append(len(positions) + 1)
                    label = method.replace("Pfeiffer_", "").replace("Profile1_BiasPrior", "Profile-1")
                    labels.append(label + "\n" + calibration)
            ax.boxplot(values, positions=positions, showfliers=True, tick_labels=labels)
            ax.set_title(f"{layout}: {axis.upper()} endpoint errors")
            ax.set_ylabel("metres")
            ax.set_yscale("symlog", linthresh=.5)
            ax.grid(axis="y", alpha=.2)
    seed_count = "one Profile-1 start" if control == "Profile1_BiasPrior" else "four starts"
    fig.suptitle("Same Delft 2/4/8 s raw packets; native author horizon 0.4 s\nAuthor zero-heading prior; BiasPrior estimates yaw with " + seed_count, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(output / filename, dpi=180)
    plt.close(fig)


def main():
    """Check packet pairing and generate complete per-window and layered evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "evidence")
    args = parser.parse_args()
    output = args.output
    common_files = sorted(output.glob("*__B.csv"))
    native_files = sorted(output.glob("*__A.csv"))
    common = pd.concat([pd.read_csv(path) for path in common_files], ignore_index=True)
    native = pd.concat([pd.read_csv(path) for path in native_files], ignore_index=True)
    assert len(common_files) == len(native_files) == 6
    assert len(common) == 792 and len(native) == 264
    assert not common.duplicated(KEYS + ["method", "calibration"]).any()
    packet_pairing = common.groupby(KEYS)[["packet_sha256", "imu_sha256", "events", "actual_endpoint_s", "actual_start_s"]].nunique()
    assert (packet_pairing == 1).all().all()
    # A failed author run may consume fewer packets; its row remains unsuccessful.
    assert (common[common.completed].events == common[common.completed].events_consumed).all()
    assert np.allclose(common[common.completed].actual_endpoint_s,
                       common[common.completed].estimate_timestamp_s, atol=1e-10)
    reference = pd.read_csv(HERE.parent / "radio_inertial_models" / "evidence" / "paired_acquisition" / "delft_same_profile.csv")
    reference = reference[reference.frame_deg == 0].rename(columns={"actual_endpoint_s": "reference_endpoint_s"})
    paired = common.merge(reference[[*KEYS, "events", "xy_error_m_old", "z_error_m_old", "3d_error_m_old", "xy_error_m_new", "z_error_m_new", "3d_error_m_new"]],
                          on=KEYS, suffixes=("", "_reference"), validate="many_to_one")
    assert len(paired) == len(common) and (paired.events == paired.events_reference).all()
    paired.to_csv(output / "common_with_acquisition_reference.csv", index=False)
    common.to_csv(output / "common_windows.csv", index=False)
    native.to_csv(output / "native_endpoint_copies.csv", index=False)
    # Duration is only a join key for A, which runs continuously from native t=0.
    native_unique = native.drop_duplicates(["sequence", "endpoint_s", "layout", "method"])
    native_unique.to_csv(output / "native_unique_endpoints.csv", index=False)
    native_runs = native_unique.sort_values("endpoint_s").drop_duplicates(["sequence", "layout", "method"], keep="last")
    native_runs[["sequence", "layout", "method", "native_time_s", "native_elapsed_s", "elapsed_s", "steps"]].to_csv(output / "native_runs.csv", index=False)
    matched_native = []
    for sequence, part in native_unique.groupby("sequence"):
        log = driver.load_real_log(driver.DATASET, sequence, sensor_delay_s=driver.SENSOR_DELAY_S)
        selected = log.time >= driver.PREFIX_S
        times = log.time[selected] - log.time[selected][0]
        truth = log.ground_truth_position[selected]
        for _, row in part.iterrows():
            end = np.searchsorted(times, row.endpoint_s, side="right") - 1
            item = row.to_dict()
            item.update(driver.score(row[["px_m", "py_m", "pz_m"]].to_numpy(float), truth[end]))
            item["protocol"] = "A_native_estimate_common_clock_scoring"
            matched_native.append(item)
    native_common_clock = pd.DataFrame(matched_native)
    native_common_clock.to_csv(output / "native_estimates_common_clock.csv", index=False)
    for extra, name in (([], "overall"), (["duration_s"], "duration"),
                        (["sequence"], "per_log"), (["sequence", "duration_s"], "per_log_duration")):
        grouped(common, GROUPS + extra, output / ("common_" + name + ".csv"))
    grouped(native_unique, GROUPS, output / "native_overall.csv")
    grouped(native_common_clock, GROUPS, output / "native_common_clock_overall.csv")
    grouped(native_unique, GROUPS + ["sequence"], output / "native_per_log.csv")
    fair = []
    for calibration in ("raw", "static"):
        for method in ("Pfeiffer_MHE", "Pfeiffer_EKF"):
            author = common[(common.method == method) & (common.calibration == calibration)]
            ours = common[(common.method == "BiasPrior") & (common.calibration == calibration)]
            pair = author.merge(ours, on=KEYS, suffixes=("_author", "_bias"), validate="one_to_one")
            for layout, part in pair.groupby("layout"):
                finite = part[np.isfinite(part.error_3d_m_author) & np.isfinite(part.error_3d_m_bias)]
                fair.append(dict(calibration=calibration, author_method=method, layout=layout,
                    requested=len(part), common_finite=len(finite),
                    bias_xy_better=int((finite.xy_error_m_bias < finite.xy_error_m_author).sum()),
                    bias_3d_better=int((finite.error_3d_m_bias < finite.error_3d_m_author).sum()),
                    author_success=int(part.endpoint_below_0p5m_author.sum()),
                    bias_success=int(part.endpoint_below_0p5m_bias.sum()),
                    median_bias_minus_author_xy_m=float((finite.xy_error_m_bias-finite.xy_error_m_author).median()),
                    median_bias_minus_author_3d_m=float((finite.error_3d_m_bias-finite.error_3d_m_author).median())))
    pd.DataFrame(fair).to_csv(output / "paired_effects.csv", index=False)
    plot_common(common, output)
    checks = dict(common_rows=len(common), native_rows=len(native), native_distinct_endpoints=len(native_unique),
        packet_pairing_identical=bool((packet_pairing == 1).all().all()),
        common_packets_match_acquisition_reference=True,
        common_finite=int(common.finite.sum()), common_completed=int(common.completed.sum()),
        expected_windows_per_arm=132, expected_native_endpoints_per_arm_layout=24,
        native_maximum_clock_rounding_s=float(np.max(np.abs(native.actual_endpoint_s-native.endpoint_s))),
        native_maximum_estimate_hold_s=float(np.max(native.actual_endpoint_s-native.estimate_timestamp_s)),
        failure_rows_retained=common[~common.completed][KEYS+["method", "calibration", "failure"]].to_dict("records"),
        source_sha256={str(p.name):hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__), HERE/"driver.py")})
    (output / "checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    print(pd.read_csv(output / "common_overall.csv").to_string(index=False))
    print(pd.read_csv(output / "native_overall.csv").to_string(index=False))


if __name__ == "__main__":
    main()
