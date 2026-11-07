"""Compare one deep estimator under two common sensor-delay assumptions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CORE_METRICS = (
    "position_rmse_m",
    "position_tail5_rmse_m",
    "velocity_rmse_mps",
    "velocity_tail5_rmse_mps",
)
METRIC_TITLES = {
    "position_rmse_m": "Full position RMSE (m)",
    "position_tail5_rmse_m": "Tail-5-s position RMSE (m)",
    "velocity_rmse_mps": "Full velocity RMSE (m/s)",
    "velocity_tail5_rmse_mps": "Tail-5-s velocity RMSE (m/s)",
}
KEY_COLUMNS = ["sequence", "offset_name"]
CONFIG_COLUMNS = (
    "coupling_minimum_capture_dwell_s",
    "coupling_initial_track_output_weight",
    "coupling_maximum_track_output_weight",
    "coupling_track_weight_rise_time_s",
    "coupling_track_weight_fall_time_s",
    "coupling_tracker_health_smoothing_time_s",
    "coupling_full_weight_health_enter",
    "coupling_full_weight_health_exit",
    "coupling_full_weight_health_hold_s",
    "coupling_minimum_track_output_health",
    "coupling_track_output_health_grace_s",
    "coupling_full_weight_agreement_m",
    "coupling_range_evidence_ratio_floor",
    "coupling_range_evidence_ratio_full",
    "coupling_decisive_range_advantage_m",
    "coupling_minimum_decisive_anchor_count",
)


def _load_table(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path)
    if table.duplicated(KEY_COLUMNS).any():
        raise ValueError(f"duplicate sequence/offset keys in {path}")
    return table.sort_values(KEY_COLUMNS).reset_index(drop=True)


def _validate_matched_runs(
    zero_delay: pd.DataFrame, corrected: pd.DataFrame
) -> None:
    if list(zero_delay.columns) != list(corrected.columns):
        raise ValueError("the two result tables have different schemas")
    if not zero_delay[KEY_COLUMNS].equals(corrected[KEY_COLUMNS]):
        raise ValueError("the two result tables do not contain matched runs")
    for column in CONFIG_COLUMNS:
        if not np.allclose(
            zero_delay[column].to_numpy(dtype=float),
            corrected[column].to_numpy(dtype=float),
            rtol=0.0,
            atol=0.0,
        ):
            raise ValueError(f"estimator setting differs: {column}")


def compare_tables(
    zero_delay: pd.DataFrame, corrected: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _validate_matched_runs(zero_delay, corrected)
    paired = corrected[KEY_COLUMNS].copy()
    summary_rows: list[dict[str, float | int | str]] = []
    tolerance = 1.0e-12
    for metric in CORE_METRICS:
        loose = corrected[f"loose_{metric}"].to_numpy(dtype=float)
        zero = zero_delay[f"deep_{metric}"].to_numpy(dtype=float)
        fixed = corrected[f"deep_{metric}"].to_numpy(dtype=float)
        versus_loose = fixed - loose
        versus_zero = fixed - zero
        paired[f"zero_delay_deep_{metric}"] = zero
        paired[f"corrected_deep_{metric}"] = fixed
        paired[f"corrected_minus_zero_{metric}"] = versus_zero
        summary_rows.append(
            {
                "metric": metric,
                "corrected_loose_median": float(np.median(loose)),
                "zero_delay_deep_median": float(np.median(zero)),
                "corrected_deep_median": float(np.median(fixed)),
                "corrected_vs_loose_paired_median_delta": float(
                    np.median(versus_loose)
                ),
                "corrected_vs_zero_paired_median_delta": float(
                    np.median(versus_zero)
                ),
                "corrected_vs_zero_relative_median_change_pct": float(
                    100.0 * (np.median(fixed) / np.median(zero) - 1.0)
                ),
                "corrected_vs_zero_better_count": int(
                    np.count_nonzero(versus_zero < -tolerance)
                ),
                "corrected_vs_zero_equal_count": int(
                    np.count_nonzero(np.abs(versus_zero) <= tolerance)
                ),
                "corrected_vs_zero_worse_count": int(
                    np.count_nonzero(versus_zero > tolerance)
                ),
                "corrected_vs_zero_worst_delta": float(
                    np.max(versus_zero)
                ),
            }
        )
    return pd.DataFrame(summary_rows), paired


def plot_paired_deltas(paired: pd.DataFrame, output_path: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12.0, 7.2), sharex=True)
    run_index = np.arange(len(paired))
    colors = np.where(paired["offset_name"] == "nominal", "#1976d2", "#ef6c00")
    for axis, metric in zip(axes.flat, CORE_METRICS, strict=True):
        delta = paired[f"corrected_minus_zero_{metric}"].to_numpy()
        axis.bar(run_index, delta, color=colors, width=0.78)
        axis.axhline(0.0, color="black", linewidth=0.9)
        axis.set_title(METRIC_TITLES[metric])
        axis.set_ylabel("corrected - zero delay")
        axis.grid(axis="y", alpha=0.25)
    for axis in axes[-1, :]:
        axis.set_xlabel("matched real replay index")
    figure.suptitle(
        "Current deep estimator: common 0.20-s correction vs zero-delay assumption"
    )
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zero-delay", type=Path, required=True)
    parser.add_argument("--corrected", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    zero_delay = _load_table(args.zero_delay)
    corrected = _load_table(args.corrected)
    summary, paired = compare_tables(zero_delay, corrected)
    args.output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output / "summary.csv", index=False)
    paired.to_csv(args.output / "paired_metrics.csv", index=False)
    payload = {
        "comparison": "same current deep estimator; common delay only",
        "zero_delay_s": 0.0,
        "corrected_delay_s": 0.2,
        "runs": int(len(paired)),
        "metrics": summary.set_index("metric").to_dict(orient="index"),
    }
    with (args.output / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    plot_paired_deltas(paired, args.output / "paired_deltas.png")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
