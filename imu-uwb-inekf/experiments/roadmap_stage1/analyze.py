"""Describe the fair IMU-TWR recovery comparison without assigning a stage verdict.

Read a runner's metrics.csv, retain case pairing, and resample whole sequences.
The outputs describe retrospective tests on the same-date 2021 dataset.
"""

import argparse
import csv
import json
import math
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Method order and names are shared by the summary and every figure.
METHOD_LABELS = {
    "inekf15": "InEKF-15",
    "eskf15": "ESKF-15",
    "capture_init": "Capture init",
    "capture_reset": "Capture reset",
    "deep": "Deep coupled",
}
# Fixed colors let readers follow the same method across figures.
METHOD_COLORS = {
    "inekf15": "#4c78a8",
    "eskf15": "#f58518",
    "capture_init": "#54a24b",
    "capture_reset": "#b279a2",
    "deep": "#e45756",
}
# Distinct markers and dashes expose methods with identical success fractions.
METHOD_STYLES = {
    "inekf15": ("o", "-"),
    "eskf15": ("s", "--"),
    "capture_init": ("^", "-."),
    "capture_reset": ("D", ":"),
    "deep": ("x", "-"),
}
# These runner columns receive finite-value descriptive statistics.
NUMERIC_METRICS = (
    "position_rmse_m", "position_tail5_rmse_m", "position_p95_m",
    "position_max_m", "velocity_rmse_mps", "first_dwell_s",
    "stable_recovery_s", "elapsed_s", "range_updates", "accepted",
    "rejected", "nis_mean", "nis_p95", "mean_tracker_weight", "reseeds",
    "threshold_coverage", "last_exit_s", "resume_error_m",
)
# Pair identity must be unique so duplicated runs cannot silently inflate counts.
IDENTITY_COLUMNS = ("suite", "sequence", "case", "method")
# These fields are needed even when a partial runner output omits telemetry.
REQUIRED_COLUMNS = IDENTITY_COLUMNS + (
    "initial_error_m", "yaw_deg", "position_rmse_m",
    "position_tail5_rmse_m", "stable_recovery_s", "success", "finite",
)
# Use a reproducible sequence bootstrap rather than resampling individual cases.
BOOTSTRAP_SEED = 20261104
# A small table of sequence means makes this resample count inexpensive.
BOOTSTRAP_RESAMPLES = 5000
# Normal quantile for a two-sided 95 percent Wilson interval.
WILSON_Z = 1.959963984540054
# Allow rounding noise when selecting nominal initialization and zero yaw.
ZERO_TOLERANCE = 1e-9
# Merge norm roundoff from different directions of the same prescribed radius.
INITIAL_ERROR_ROUND_DECIMALS = 9
# Stable filenames are the interface used by the runner and report.
FIGURE_NAMES = (
    "nominal_tail_comparison.png", "holdout_capture_domain.png",
    "outage_recovery.png",
)


def _number(value):
    """Read missing numeric values as NaN while preserving finite failures."""
    if value is None or str(value).strip().lower() in ("", "nan", "none", "null"):
        return math.nan
    return float(value)


def _boolean(value):
    """Parse explicit CSV booleans without treating the string 'False' as true."""
    text = str(value).strip().lower()
    if text in ("true", "1", "1.0"):
        return True
    if text in ("false", "0", "0.0"):
        return False
    raise ValueError(f"Invalid boolean {value!r}")


def load_rows(path):
    """Read and validate the case table needed for unambiguous pairing."""
    rows = []
    identities = set()
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = sorted(set(REQUIRED_COLUMNS) - set(reader.fieldnames or ()))
        if missing:
            raise ValueError(f"metrics.csv is missing columns: {', '.join(missing)}")
        for line, record in enumerate(reader, start=2):
            try:
                row = {column: record[column].strip() for column in IDENTITY_COLUMNS}
                if not all(row.values()):
                    raise ValueError("Case identity fields cannot be empty")
                row.update({column: _number(record.get(column))
                            for column in NUMERIC_METRICS + ("initial_error_m", "yaw_deg")})
                row.update({column: _boolean(record[column])
                            for column in ("success", "finite")})
                identity = tuple(row[column] for column in IDENTITY_COLUMNS)
                if identity in identities:
                    raise ValueError(f"Duplicate case identity {identity}")
                identities.add(identity)
                rows.append(row)
            except (AttributeError, TypeError, ValueError) as error:
                raise ValueError(f"metrics.csv line {line}: {error}") from error
    return rows


def _finite_or_none(value):
    """Make every numeric summary compatible with strict JSON."""
    return float(value) if math.isfinite(value) else None


def _stats(values):
    """Describe all finite values and explicitly count unavailable metrics."""
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    result = {"n": int(finite.size), "n_unavailable": int(array.size - finite.size)}
    if not finite.size:
        result.update({name: None for name in ("mean", "median", "min", "max", "p95")})
        return result
    with np.errstate(over="ignore", invalid="ignore"):
        values = (np.mean(finite), np.median(finite), np.min(finite),
                  np.max(finite), np.percentile(finite, 95))
    result.update({name: _finite_or_none(value) for name, value in
                   zip(("mean", "median", "min", "max", "p95"), values)})
    return result


def _wilson(successes, total):
    """Return a descriptive case-level Wilson interval, not a cluster interval."""
    if not total:
        return [None, None]
    rate = successes / total
    z_squared = WILSON_Z ** 2
    denominator = 1.0 + z_squared / total
    center = (rate + z_squared / (2.0 * total)) / denominator
    radius = WILSON_Z * math.sqrt(
        rate * (1.0 - rate) / total + z_squared / (4.0 * total ** 2)
    ) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def _methods(rows):
    """Order known methods consistently and retain any extra runner method."""
    present = {row["method"] for row in rows}
    return [name for name in METHOD_LABELS if name in present] + sorted(
        present - METHOD_LABELS.keys()
    )


def _groups(rows, columns):
    """Partition records without depending on dataframe groupby conventions."""
    grouped = {}
    for row in rows:
        grouped.setdefault(tuple(row[column] for column in columns), []).append(row)
    return grouped


def _group_summary(suite, method, rows):
    """Aggregate cases while keeping successful recovery times conditional."""
    successes = sum(row["success"] for row in rows)
    metrics = {}
    for column in NUMERIC_METRICS:
        eligible = rows
        if column == "stable_recovery_s":
            eligible = [row for row in rows if row["success"]]
        metrics[column] = _stats([row[column] for row in eligible])
    return {
        "suite": suite, "method": method, "n_cases": len(rows),
        "n_sequences": len({row["sequence"] for row in rows}),
        "success": {"n": successes, "rate": successes / len(rows),
                    "case_wilson95": _wilson(successes, len(rows))},
        "finite": {"n": sum(row["finite"] for row in rows),
                   "n_success_but_not_finite": sum(row["success"] and not row["finite"]
                                                 for row in rows)},
        "metrics": metrics,
    }


def _pair_effects(pairs):
    """Compute deep-minus-comparator effects using the same sequence and case."""
    effects = {"success_delta_fraction": [], "stable_recovery_delta_s": [],
               "position_rmse_delta_pct": [], "tail5_rmse_delta_pct": []}
    counts = {"both_success": 0, "deep_only_success": 0,
              "comparator_only_success": 0, "neither_success": 0}
    for deep, comparator in pairs:
        effects["success_delta_fraction"].append(
            float(deep["success"]) - float(comparator["success"])
        )
        if deep["success"] and comparator["success"]:
            counts["both_success"] += 1
            effects["stable_recovery_delta_s"].append(
                deep["stable_recovery_s"] - comparator["stable_recovery_s"]
            )
        elif deep["success"]:
            counts["deep_only_success"] += 1
        elif comparator["success"]:
            counts["comparator_only_success"] += 1
        else:
            counts["neither_success"] += 1
        for column, name in (("position_rmse_m", "position_rmse_delta_pct"),
                             ("position_tail5_rmse_m", "tail5_rmse_delta_pct")):
            before, after = comparator[column], deep[column]
            relative = math.nan
            if math.isfinite(before) and before > 0.0 and math.isfinite(after):
                relative = 100.0 * (after / before - 1.0)
            effects[name].append(relative)
    return {"n_pairs": len(pairs), "success_counts": counts,
            "effects": {name: _stats(values) for name, values in effects.items()}}


def _sequence_bootstrap(per_sequence):
    """Resample paired whole-sequence means with equal weight per sequence."""
    result = {}
    effect_names = ("success_delta_fraction", "stable_recovery_delta_s",
                    "position_rmse_delta_pct", "tail5_rmse_delta_pct")
    for name in effect_names:
        means = [sequence["effects"][name]["mean"] for sequence in per_sequence]
        values = np.asarray([value for value in means if value is not None], dtype=float)
        entry = {"n_sequences": int(values.size), "estimate": None, "ci95": [None, None]}
        if values.size:
            entry["estimate"] = float(np.mean(values))
        # One sequence cannot estimate between-sequence uncertainty.
        if values.size > 1:
            rng = np.random.default_rng(BOOTSTRAP_SEED)
            indices = rng.integers(values.size, size=(BOOTSTRAP_RESAMPLES, values.size))
            draws = np.mean(values[indices], axis=1)
            entry["ci95"] = np.percentile(draws, [2.5, 97.5]).tolist()
        result[name] = entry
    return {"seed": BOOTSTRAP_SEED, "resamples": BOOTSTRAP_RESAMPLES,
            "weighting": "equal sequence means; resample sequences only", "effects": result}


def _paired_summary(suite, comparator, rows):
    """Match methods within a suite and summarize paired sequence clusters."""
    index = {(row["sequence"], row["case"], row["method"]): row for row in rows}
    deep_keys = {(row["sequence"], row["case"]) for row in rows if row["method"] == "deep"}
    comparator_keys = {(row["sequence"], row["case"]) for row in rows
                       if row["method"] == comparator}
    matched = sorted(deep_keys & comparator_keys)
    pairs = [(index[(sequence, case, "deep")], index[(sequence, case, comparator)])
             for sequence, case in matched]
    per_sequence = []
    for sequence in sorted({key[0] for key in matched}):
        subset = [pair for pair in pairs if pair[0]["sequence"] == sequence]
        per_sequence.append({"sequence": sequence, **_pair_effects(subset)})
    return {
        "suite": suite, "reference": "deep", "comparator": comparator,
        "n_unmatched_deep": len(deep_keys - comparator_keys),
        "n_unmatched_comparator": len(comparator_keys - deep_keys),
        **_pair_effects(pairs), "per_sequence": per_sequence,
        "sequence_bootstrap95": _sequence_bootstrap(per_sequence),
    }


def build_summary(rows, metrics_path):
    """Build descriptive aggregates with explicit pairing and dataset limitations."""
    suites = sorted({row["suite"] for row in rows})
    groups = _groups(rows, ("suite", "method"))
    paired = []
    for suite in suites:
        subset = [row for row in rows if row["suite"] == suite]
        if "deep" in _methods(subset):
            paired.extend(_paired_summary(suite, method, subset)
                          for method in _methods(subset) if method != "deep")
    return {
        "input_file": str(metrics_path.resolve()), "n_rows": len(rows),
        "n_unique_cases": len({(row["suite"], row["sequence"], row["case"]) for row in rows}),
        "suites": suites, "methods": _methods(rows),
        "scope": {
            "dataset": "The source contains 12 logs recorded on one date in 2021.",
            "historical_supervision": "All 12 logs were used in historical supervised development.",
            "generalization": "LOSO/holdout here is retrospective; it does not establish untouched-data or unseen-date generalization.",
            "verdict": "Descriptive comparison only; no automatic stage-pass decision.",
        },
        "aggregation": {
            "case_wilson95": "Descriptive case-level Wilson 95% intervals; cases sharing a sequence are clustered, so these are not cluster-adjusted confidence intervals.",
            "sequence_bootstrap95": "Deterministic percentile intervals of equally weighted paired sequence means; only whole sequences are resampled. This describes variation among available logs, not independent acquisition dates.",
            "pair_identity": ["suite", "sequence", "case"],
            "effect_sign": "deep minus comparator; negative RMSE/recovery changes favor deep, positive success changes favor deep.",
            "relative_error_delta": "100 * (deep / comparator - 1), paired finite values with comparator > 0; finite failed runs are included.",
            "stable_recovery": "Method medians use successful cases only; paired differences use jointly successful cases only. Outage recovery time is measured from range resumption by the runner.",
            "success_definition": "Use the runner's success flag: a terminal suffix of at least 5 seconds with position error at most 0.5 m throughout.",
            "timing": "Elapsed replay times are descriptive and may include concurrent CPU contention; no speed claim is computed.",
            "missing_values": "Unavailable/nonfinite numeric values become null and are counted; failures never receive a fabricated recovery time.",
        },
        "groups": [_group_summary(suite, method, subset)
                   for (suite, method), subset in sorted(groups.items())],
        "paired_deep_comparisons": paired,
        "figures": list(FIGURE_NAMES),
    }


def _is_zero(value):
    """Identify zero initialization values without accepting missing numbers."""
    return math.isfinite(value) and abs(value) <= ZERO_TOLERANCE


def _empty_axes(axes, message):
    """Make missing suites visible instead of inventing plotted data."""
    for axis in np.asarray(axes).reshape(-1):
        axis.text(0.5, 0.5, message, transform=axis.transAxes, ha="center", va="center")
        axis.set_axis_off()


def _style_axis(axis, ylabel):
    """Apply legible labels and a light grid without clipping any error values."""
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.25)
    axis.set_axisbelow(True)


def _save_figure(figure, path):
    """Save a reviewable raster figure and release its plotting resources."""
    figure.savefig(path, dpi=180, facecolor="white")
    plt.close(figure)


def plot_nominal(rows, path):
    """Compare nominal full-trajectory and tail RMSE on a single evaluation suite."""
    nominal = [row for row in rows if row["suite"] != "outage"
               and _is_zero(row["initial_error_m"]) and _is_zero(row["yaw_deg"])]
    preferred = next((suite for suite in ("loso", "holdout")
                      if any(row["suite"] == suite for row in nominal)), None)
    nominal = [row for row in nominal if row["suite"] == preferred]
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    if not nominal:
        _empty_axes(axes, "No nominal zero-yaw cases in metrics.csv")
    else:
        methods = _methods(nominal)
        for axis, column, title in zip(
            axes, ("position_rmse_m", "position_tail5_rmse_m"),
            ("Full trajectory", "Final 5 seconds"),
        ):
            arrays = [[row[column] for row in nominal if row["method"] == method
                       and math.isfinite(row[column])] for method in methods]
            boxes = axis.boxplot(arrays, patch_artist=True, showfliers=False)
            for position, (method, values, box) in enumerate(zip(methods, arrays, boxes["boxes"]), 1):
                color = METHOD_COLORS.get(method, "#777777")
                box.set_facecolor(color)
                box.set_alpha(0.35)
                offsets = np.linspace(-0.06, 0.06, len(values))
                axis.scatter(position + offsets, values, s=20, color=color, alpha=0.8)
            labels = [f"{METHOD_LABELS.get(method, method)}\n(n={len(values)})"
                      for method, values in zip(methods, arrays)]
            axis.set_xticks(range(1, len(methods) + 1), labels, rotation=15)
            _style_axis(axis, "Position RMSE (m)")
            all_values = [value for values in arrays for value in values]
            if all_values and min(all_values) > 0.0 and max(all_values) / min(all_values) > 20.0:
                axis.set_yscale("log")
                axis.set_ylabel("Position RMSE (m, log scale)")
            axis.set_title(title)
    figure.suptitle(f"Nominal initialization: {preferred or 'unavailable'}")
    figure.text(0.5, 0.02, "Each point is a sequence/case; all finite outcomes are retained.",
                ha="center", fontsize=9)
    figure.tight_layout(rect=(0, 0.06, 1, 0.94))
    _save_figure(figure, path)


def _plot_recovery(axes, rows, x_values, x_column):
    """Plot case success fractions and successful-only median recovery times."""
    for method in _methods(rows):
        success, recovery = [], []
        for value in x_values:
            subset = [row for row in rows if row["method"] == method and row[x_column] == value]
            success.append(sum(row["success"] for row in subset) / len(subset)
                           if subset else math.nan)
            times = [row["stable_recovery_s"] for row in subset
                     if row["success"] and math.isfinite(row["stable_recovery_s"])]
            recovery.append(float(np.median(times)) if times else math.nan)
        color = METHOD_COLORS.get(method, "#777777")
        label = METHOD_LABELS.get(method, method)
        positions = x_values if x_column == "initial_error_m" else np.arange(len(x_values))
        marker, line_style = METHOD_STYLES.get(method, ("o", "-"))
        for axis, values in zip(axes, (success, recovery)):
            axis.plot(positions, values, color=color, label=label, marker=marker,
                      linestyle=line_style, markersize=6, markerfacecolor="none")
    _style_axis(axes[0], "Stable recovery success fraction")
    _style_axis(axes[1], "Median stable recovery time (s)")
    axes[0].set_ylim(-0.03, 1.05)
    axes[1].set_ylim(bottom=0.0)


def plot_holdout(rows, path):
    """Show the retrospective holdout capture domain for zero initial yaw error."""
    subset = [row for row in rows if row["suite"] == "holdout" and _is_zero(row["yaw_deg"])
              and math.isfinite(row["initial_error_m"])]
    subset = [{**row, "initial_error_m": round(row["initial_error_m"], INITIAL_ERROR_ROUND_DECIMALS)}
              for row in subset]
    figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    if subset:
        positions = sorted({row["initial_error_m"] for row in subset})
        _plot_recovery(axes, subset, positions, "initial_error_m")
        axes[1].set_xlabel("Initial position error magnitude (m)")
        axes[1].set_xticks(positions)
        handles, labels = axes[0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.04))
    else:
        _empty_axes(axes, "No zero-yaw holdout cases in metrics.csv")
    figure.suptitle("Retrospective holdout: initial yaw error = 0 deg")
    figure.text(0.5, 0.015, "Recovery times exclude failures; gaps mean no successful finite recovery time.",
                ha="center", fontsize=9)
    figure.tight_layout(rect=(0, 0.14, 1, 0.95))
    _save_figure(figure, path)


def _case_order(case):
    """Sort runner case labels naturally so numbered outages remain in order."""
    return [(1, int(token)) if token.isdigit() else (0, token)
            for token in re.split(r"(\d+)", case)]


def plot_outage(rows, path):
    """Describe recovery after range resumption for each runner outage case."""
    subset = [row for row in rows if row["suite"] == "outage"]
    figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    if subset:
        cases = sorted({row["case"] for row in subset}, key=_case_order)
        _plot_recovery(axes, subset, cases, "case")
        axes[1].set_xticks(np.arange(len(cases)), [case.replace("_", " ") for case in cases],
                          rotation=15, ha="right")
        axes[1].set_xlabel("Outage case (timing and perturbation defined by runner)")
        handles, labels = axes[0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.04))
    else:
        _empty_axes(axes, "No outage cases in metrics.csv")
    figure.suptitle("UWB outage: recovery measured from range resumption")
    figure.text(0.5, 0.015, "Recovery times exclude failures; gaps mean no successful finite recovery time.",
                ha="center", fontsize=9)
    figure.tight_layout(rect=(0, 0.14, 1, 0.95))
    _save_figure(figure, path)


def main():
    """Create descriptive JSON and three figures from a runner output directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Directory containing metrics.csv and receiving analysis outputs")
    args = parser.parse_args()
    metrics_path = args.output_dir / "metrics.csv"
    rows = load_rows(metrics_path)
    summary = build_summary(rows, metrics_path)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    with plt.rc_context({"font.size": 10, "axes.titlesize": 12, "figure.titlesize": 14}):
        plot_nominal(rows, args.output_dir / FIGURE_NAMES[0])
        plot_holdout(rows, args.output_dir / FIGURE_NAMES[1])
        plot_outage(rows, args.output_dir / FIGURE_NAMES[2])
    print(json.dumps({"n_rows": len(rows), "summary": str((args.output_dir / "summary.json").resolve()),
                      "figures": [str((args.output_dir / name).resolve()) for name in FIGURE_NAMES]}))


if __name__ == "__main__":
    main()
