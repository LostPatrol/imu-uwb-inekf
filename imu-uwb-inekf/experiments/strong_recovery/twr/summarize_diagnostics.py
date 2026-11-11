"""Offline first-packet and covariance summaries for fixed TWR diagnostics."""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

# This summary reads only this task's estimator traces and downstream scoring truth.
ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "agent/codex/strong_twr"
# Transaction checkpoints are fixed and do not select a best filter or endpoint.
CHECKPOINTS = (1, 2, 4, 8, 16, 32, 64, 128, 256)


def main():
    """Describe learned source differences and causal controls without changing them."""
    snapshots, packets = [], []
    for file in sorted(OUT.glob("2021*.npz")):
        data = np.load(file)
        times, truth = data["time"], data["truth"]
        for edge in ("pre15", "pre25", "resume"):
            a, b = (data[f"{source}_{edge}__X"] for source in ("capture", "profile"))
            ba, bb = (data[f"{source}_{edge}__bias"] for source in ("capture", "profile"))
            for source in ("capture", "profile"):
                state = data[f"{source}_{edge}__X"]
                covariance = data[f"{source}_{edge}__P"]
                index = int(np.searchsorted(times, 15 if edge == "pre15" else 25))
                snapshots.append(dict(sequence=file.stem, source=source, edge=edge,
                    xy_error_m=float(np.linalg.norm(state[:2, 4]-truth[index, :2])),
                    p_std_max_m=float(np.sqrt(max(np.linalg.eigvalsh(covariance[6:9, 6:9])))),
                    v_std_max_mps=float(np.sqrt(max(np.linalg.eigvalsh(covariance[3:6, 3:6])))),
                    source_position_difference_m=float(np.linalg.norm(a[:3, 4]-b[:3, 4])),
                    source_velocity_difference_mps=float(np.linalg.norm(a[:3, 3]-b[:3, 3])),
                    source_rotation_difference_deg=float(np.degrees(np.linalg.norm(Rotation.from_matrix(a[:3, :3]@b[:3, :3].T).as_rotvec()))),
                    source_bg_difference_rad_s=float(np.linalg.norm(ba[:3]-bb[:3])),
                    source_ba_difference_mps2=float(np.linalg.norm(ba[3:]-bb[3:]))))
        frame = pd.read_csv(OUT/(file.stem+"_measurements.csv"))
        for (source, method), group in frame.groupby(["source", "method"], sort=False):
            for count in CHECKPOINTS:
                if count > len(group):
                    continue
                row = group.iloc[count-1]
                p = row[["p_after_x", "p_after_y", "p_after_z"]].to_numpy(dtype=float)
                packets.append(dict(sequence=file.stem, source=source, method=method,
                    transaction=count, time_s=row.time_s, delay_s=row.time_s-25,
                    xy_error_m=float(np.linalg.norm(p[:2]-truth[int(row["sample"]), :2])),
                    p_std_max_m=row.p_std_max_m, accepted_total=row.accepted_total,
                    rejected_total=row.rejected_total, nis=row.nis,
                    position_jump_m=row.position_jump_m))
    pd.DataFrame(snapshots).to_csv(OUT/"source_snapshots.csv", index=False)
    frame = pd.DataFrame(packets)
    frame.to_csv(OUT/"packet_checkpoints.csv", index=False)
    frame.groupby(["source", "method", "transaction"]).agg(delay_median_s=("delay_s", "median"),
        xy_error_median_m=("xy_error_m", "median"), xy_error_max_m=("xy_error_m", "max"),
        p_std_max_median_m=("p_std_max_m", "median"), rejections_median=("rejected_total", "median")).to_csv(OUT/"packet_summary.csv")
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for method in ("lifted_warm256", "lifted_warm0", "raw_warm256", "squared_warm256"):
        curve = frame[(frame.source == "profile") & (frame.method == method)].groupby("transaction").median(numeric_only=True)
        axes[0].plot(curve.index, curve.xy_error_m, marker="o", label=method)
        axes[1].plot(curve.index, curve.p_std_max_m, marker="o", label=method)
    for axis in axes:
        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        axis.set_xlabel("Received radio transaction")
        axis.grid(alpha=.25)
    axes[0].set_ylabel("Median XY error (m)")
    axes[1].set_ylabel("Median maximum position std (m)")
    axes[0].legend(fontsize=7)
    figure.suptitle("Same Profile recovery state/input; six Delft development flights")
    figure.tight_layout()
    figure.savefig(OUT/"first_packets.png", dpi=170)
    plt.close(figure)
    print(pd.DataFrame(snapshots).groupby(["source", "edge"]).median(numeric_only=True).to_string())
    print(frame[(frame.source == "profile") & frame.method.isin(("lifted_warm256", "raw_warm256", "squared_warm256"))].groupby(["method", "transaction"]).median(numeric_only=True).to_string())


if __name__ == "__main__":
    main()
