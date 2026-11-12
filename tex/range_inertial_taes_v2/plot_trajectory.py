"""Plot a real Delft recording and continuous closed-loop replay, without rerunning filters.

Default: use the compact local data bundle. --extract refreshes it from original evidence.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

# Use the first recording in the frozen six-flight protocol, without selecting by outcome.
SEQUENCE = "2021-02-12_09_43_34_twr_triangle"
# All figure artifacts belong to this isolated manuscript revision.
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXP = ROOT / "imu-uwb-inekf" / "experiments"
BUNDLE = HERE / "trajectory_data.npz"
# Frozen recovery times are in the post-prefix replay clock, not raw logger time.
GAP_START, RETURN, HANDOFF, STOP = 5., 25., 27.01, 36.
# Retain the original estimator labels and a consistent, accessible color mapping.
METHODS = ("ieks_profile", "profile_startup_native_recovery", "old_lifted")
LABELS = ("Window handoff (W)", "Continue InEKF (H)", "Lifted capture (L)")
COLORS = ("#176a94", "#b75b28", "#34764b")


def sha256(path):
    """Identify exact source files while keeping the compact bundle self-contained."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def interpolate(time, values, query):
    """Interpolate measured reference positions without extrapolating support."""
    assert query.min() >= time.min() - 1e-10 and query.max() <= time.max() + 1e-10
    return np.column_stack([np.interp(query, time, values[:, axis]) for axis in range(3)])


def extract(dataset):
    """Preserve actual emitted positions, complete valid reference and original scoring clock."""
    result = EXP / "unified_window_recovery" / "results"
    cache = EXP / "split_handoff" / "delft" / "results" / "cache"
    protocol_path = EXP / "unified_window_recovery" / "protocol.json"
    replay_path = result / ("delft__" + SEQUENCE + "__L20.npz")
    audit_path = replay_path.with_suffix(".json")
    sensor_path = cache / (SEQUENCE + ".npz")
    truth_path = cache / (SEQUENCE + "_scoring.npz")
    raw_path = dataset / "twr" / (SEQUENCE + ".csv")
    frame = pd.read_csv(raw_path, skipinitialspace=True)
    raw_time = frame.timeTick.to_numpy(dtype=float) / 1000.
    truth = frame[["otX", "otY", "otZ"]].to_numpy(dtype=float)
    valid = np.isfinite(truth).all(axis=1) & (np.linalg.norm(truth, axis=1) > 1e-9)
    sensor_columns = ["gyroX", "gyroY", "gyroZ", "accX", "accY", "accZ"] + [f"twr{i}" for i in range(8)]
    sensor_valid = np.isfinite(frame[sensor_columns].to_numpy()).all(axis=1)
    initial = raw_time[np.flatnonzero(valid & sensor_valid)[0]]
    # Reproduce load_real_log's shared 0.20 s sensor correction and 1.50 s IMU prefix.
    corrected = raw_time - .20
    eligible = sensor_valid & (corrected >= initial - 1e-12) & (corrected <= raw_time[valid][-1] + 1e-12)
    prefix_end = np.flatnonzero(eligible & (corrected - initial >= 1.50))[0]
    origin = corrected[prefix_end]
    reference_time, reference = raw_time[valid] - origin, truth[valid]
    with np.load(sensor_path) as data:
        sensor_time = data["time"]
    with np.load(truth_path) as data:
        cached_truth = data["position"]
    truth_delta = np.max(abs(interpolate(reference_time, reference, sensor_time) - cached_truth))
    assert truth_delta < 1e-10
    with np.load(replay_path) as data:
        # Preserve the exact frozen [25,36) floating-point grid, including boundary holds.
        score_time = np.r_[np.arange(0., RETURN, .05), data["score_time"]]
        score_truth = interpolate(sensor_time, cached_truth, score_time)
        assert np.max(abs(score_truth[score_time >= RETURN] - data["score_truth"])) < 1e-10
        payload = dict(reference_time=reference_time, reference_position=reference,
                       score_time=score_time, score_truth=score_truth)
        for method in METHODS:
            times = data[method + "_time"]
            position = data[method + "_position"] if method == "old_lifted" else data[method + "_X"][:, :3, 4]
            keep = (times >= 0) & (times <= STOP)
            payload[method + "_time"] = times[keep]
            payload[method + "_position"] = position[keep]
    same = payload[METHODS[0] + "_time"] < HANDOFF - 1e-8
    assert np.array_equal(payload[METHODS[0] + "_time"], payload[METHODS[1] + "_time"])
    assert np.array_equal(payload[METHODS[0] + "_position"][same], payload[METHODS[1] + "_position"][same])
    audit = json.loads(audit_path.read_text(encoding="utf8"))
    handoffs = [event["handoff_s"] for event in audit["audits"]["ieks_profile"]["events"] if event["supported"]]
    assert np.allclose(handoffs, [2., HANDOFF], atol=1e-10, rtol=0)
    metadata = dict(sequence=SEQUENCE, selection="First flight in the frozen evaluation protocol, not selected by error",
                    raw_record_duration_s=float(raw_time[-1] - raw_time[0]),
                    valid_reference_duration_s=float(reference_time[-1] - reference_time[0]),
                    logger_origin_s=float(origin), reference_cache_max_difference_m=float(truth_delta),
                    replay_interval_s=[0., STOP], outage_interval_s=[GAP_START, RETURN],
                    replacement_time_s=HANDOFF, wall_time_in_replay=False,
                    primary_phase_scores=[row for row in audit["metrics"] if row["method"] in METHODS],
                    sources={path.relative_to(ROOT).as_posix(): sha256(path)
                             for path in [replay_path, audit_path, sensor_path, truth_path, protocol_path]})
    metadata["sources"]["Delft dataset/" + raw_path.relative_to(dataset).as_posix()] = sha256(raw_path)
    payload["metadata"] = json.dumps(metadata, ensure_ascii=False)
    np.savez_compressed(BUNDLE, **payload)
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


def error_curves(data):
    """Score the last emitted state, preserving every pre-handoff output."""
    clock, truth = data["score_time"], data["score_truth"]
    metadata = json.loads(str(data["metadata"]))
    errors = {}
    for method in METHODS:
        time, position = data[method + "_time"], data[method + "_position"]
        indices = np.maximum(0, np.searchsorted(time, clock, side="right") - 1)
        errors[method] = np.linalg.norm(position[indices, :2] - truth[:, :2], axis=1)
        for name, left, right in [("first2", 25., 27.), ("two_to_five", 27., 30.), ("six_to_eleven", 31., 36.)]:
            selected = (clock >= left) & (clock < right - 1e-8)
            value = np.sqrt(np.mean(errors[method][selected]**2))
            original = [row["xy_rmse_m"] for row in metadata["primary_phase_scores"]
                        if row["method"] == method and row["window"] == name]
            assert len(original) == 1 and abs(value - original[0]) < 1e-9
            print(method, name, int(selected.sum()), "samples; XY RMSE", value)
    return errors


def plot(data):
    """Distinguish full measured reference geometry from the shorter closed-loop evaluation."""
    errors = error_curves(data)
    metadata = json.loads(str(data["metadata"]))
    fig, (ax, time_ax) = plt.subplots(1, 2, figsize=(6.8, 3.0), gridspec_kw={"width_ratios": [1., 1.55]})
    rt, reference = data["reference_time"], data["reference_position"]
    ax.plot(reference[:, 0], reference[:, 1], color="0.65", linewidth=1.1, label="Complete valid reference")
    selected = (rt >= HANDOFF - 1e-8) & (rt <= STOP)
    ax.plot(reference[selected, 0], reference[selected, 1], color="0.12", linewidth=2.4, zorder=3,
            label="Reference, 27.01–36 s")
    # Show post-handoff estimates on the map and all policies on the error axis.
    for method, color, line in [("old_lifted", COLORS[2], "--"), ("ieks_profile", COLORS[0], "-")]:
        time, position = data[method + "_time"], data[method + "_position"]
        keep = (time >= HANDOFF - 1e-8) & (time <= STOP)
        ax.plot(position[keep, 0], position[keep, 1], color=color, linestyle=line, linewidth=1.1, zorder=4)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("World X (m)")
    ax.set_ylabel("World Y (m)")
    ax.set_title(f"(a) Full reference ({metadata['valid_reference_duration_s']:.2f} s)", fontsize=8.5, pad=6)
    ax.set_xticks([-1, 0, 1, 2])
    ax.set_yticks([-1, 0, 1])
    ax.grid(color="0.91", linewidth=.55)
    time_ax.axvspan(GAP_START, RETURN, color="0.90", zorder=0)
    time_ax.axvspan(RETURN, HANDOFF, color="#f4dec3", alpha=.8, zorder=0)
    for method, color, line in zip(METHODS, COLORS, ["-", "-.", "--"]):
        time_ax.plot(data["score_time"], errors[method], color=color, linestyle=line,
                     linewidth=1.1, drawstyle="steps-post", zorder=3)
    time_ax.axvline(HANDOFF, color="0.25", linestyle=":", linewidth=.8)
    time_ax.text(15, 350, "Ranges withheld", fontsize=7.5, ha="center", va="top")
    time_ax.text(27.5, 1.6, "Handoff\n27.01 s", fontsize=7.5, ha="left")
    time_ax.set_yscale("log")
    time_ax.set_ylim(.003, 500)
    time_ax.set_yticks([.01, .1, 1, 10, 100], ["0.01", "0.1", "1", "10", "100"])
    time_ax.set_xlim(0, STOP)
    time_ax.set_xticks([0, 5, 15, 25, 36])
    time_ax.set_xlabel("Replay time (s)")
    time_ax.set_ylabel("XY error (m, log scale)")
    time_ax.set_title("(b) Filter replay (0–36 s)", fontsize=8.5, pad=6)
    time_ax.grid(axis="y", color="0.91", linewidth=.55)
    for axis in [ax, time_ax]:
        axis.spines[["top", "right"]].set_visible(False)
        axis.minorticks_off()
    handles = [Line2D([], [], color="0.65", linewidth=1.1), Line2D([], [], color="0.12", linewidth=2.4)]
    handles += [Line2D([], [], color=color, linestyle=line, linewidth=1.1)
                for color, line in zip(COLORS, ["-", "-.", "--"])]
    fig.legend(handles, ["Complete valid reference", "Reference, 27.01–36 s"] + list(LABELS),
               loc="upper center", bbox_to_anchor=(.51, 1.005), ncol=2, frameon=False,
               fontsize=7.5, handlelength=2.2, columnspacing=1.2)
    fig.subplots_adjust(left=.085, right=.985, bottom=.17, top=.72, wspace=.34)
    output = HERE / "figures" / "flight_trajectory.pdf"
    output.parent.mkdir(exist_ok=True)
    fig.savefig(output, metadata={"CreationDate": None})
    plt.close(fig)


def main():
    """Build the isolated figure from its portable bundle or refresh original evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extract", action="store_true", help="Refresh the compact data bundle from source caches and raw CSV")
    parser.add_argument("--dataset", type=Path,
                        default=Path(os.environ.get("INEKF_TWR_DATASET", "E:/datasets/tu-delft-uwb-mhe/data_v1/UWB dataset")))
    args = parser.parse_args()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 8,
                         "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "pdf.fonttype": 42,
                         "axes.linewidth": .6})
    if args.extract or not BUNDLE.exists():
        extract(args.dataset)
    with np.load(BUNDLE, allow_pickle=False) as data:
        plot(data)


if __name__ == "__main__":
    main()
