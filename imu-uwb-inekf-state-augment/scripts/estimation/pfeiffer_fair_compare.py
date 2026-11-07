#!/usr/bin/env python3
"""Reproduce Pfeiffer et al. (2021) and run truth-free fair comparisons.

The upstream MIT-licensed repository is kept pristine.  Runtime adapters in
this file disable the OptiTrack-derived altitude measurement and replace the
ground-truth-dependent range standard deviation without editing upstream code.
The 2023/2025 LS/Newton/SVOR variants are not represented as author-code
reproductions because their implementation and complete tuning are not public.
"""

from __future__ import annotations

import argparse
import contextlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import io
import json
import math
from pathlib import Path
import random
import sys
import time
import types
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


WORKSPACE = Path(__file__).resolve().parents[3]
PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = Path(r"E:\datasets\tu-delft-uwb-mhe\data_v1\UWB dataset")
DEFAULT_AUTHOR_REPO = WORKSPACE / "third_party" / "pfeiffer_uwb_simulator_2021"
DEFAULT_CALIBRATION = (
    PROJECT
    / "results"
    / "task06_per_anchor_robust"
    / "improved_comparison"
    / "loso_per_anchor_calibration.csv"
)
DEFAULT_CURRENT_METRICS = (
    PROJECT
    / "results"
    / "task06_per_anchor_robust"
    / "improved_comparison"
    / "improvement_batch_metrics.csv"
)
DEFAULT_OUTPUT = PROJECT / "results" / "task06_author_fair_compare"
EXPECTED_AUTHOR_SHA = "955360a8736cae3600a1fbed530a9ec19a2ad724"
REPRESENTATIVE_SEQUENCE = "2021-02-12_09_56_09_twr_random.csv"


@dataclass(frozen=True)
class Protocol:
    name: str
    altitude_enabled: bool
    truth_dependent_range_sigma: bool
    calibration: str
    initial_position: str
    score_start_s: float
    score_strictly_after_start: bool
    purpose: str


PROTOCOLS = {
    "paper_oracle": Protocol(
        name="paper_oracle",
        altitude_enabled=True,
        truth_dependent_range_sigma=True,
        calibration="none",
        initial_position="author_zero",
        score_start_s=1.0,
        score_strictly_after_start=True,
        purpose="exact 2021 publication script protocol",
    ),
    "no_oracle_raw": Protocol(
        name="no_oracle_raw",
        altitude_enabled=False,
        truth_dependent_range_sigma=False,
        calibration="global_fixed",
        initial_position="first_optitrack_position",
        score_start_s=0.0,
        score_strictly_after_start=False,
        purpose="remove all runtime truth aid without dataset calibration",
    ),
    "no_oracle_loso": Protocol(
        name="no_oracle_loso",
        altitude_enabled=False,
        truth_dependent_range_sigma=False,
        calibration="loso_per_anchor",
        initial_position="first_optitrack_position",
        score_start_s=0.0,
        score_strictly_after_start=False,
        purpose="same LOSO calibration and scoring window as current Task 06",
    ),
}


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def read_author_sha(author_repo: Path) -> str:
    head = (author_repo / ".git" / "HEAD").read_text(encoding="utf-8").strip()
    if head.startswith("ref:"):
        reference = head.split(":", 1)[1].strip()
        return (author_repo / ".git" / reference).read_text(encoding="utf-8").strip()
    return head


def validate_inputs(
    dataset: Path,
    author_repo: Path,
    calibration_path: Path,
    current_metrics_path: Path,
) -> list[str]:
    required = [
        dataset / "anchor_positions.yaml",
        dataset / "twr",
        author_repo / "UWBsim" / "estimators" / "mhe.py",
        author_repo / "publication" / "headless_Nanchors.py",
        calibration_path,
        current_metrics_path,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing fair-comparison inputs: " + "; ".join(missing))
    sequences = sorted(path.name for path in (dataset / "twr").glob("*.csv"))
    if len(sequences) != 12:
        raise ValueError(f"Expected 12 TWR logs, found {len(sequences)}")
    calibration = pd.read_csv(calibration_path)
    required_columns = {
        "held_out_sequence",
        "anchor_index",
        "range_bias_m",
        "range_sigma_m",
    }
    if not required_columns.issubset(calibration.columns):
        raise ValueError("LOSO calibration table has missing columns")
    counts = calibration.groupby("held_out_sequence")["anchor_index"].nunique()
    if set(counts.index) != set(sequences) or not np.all(counts.to_numpy() == 8):
        raise ValueError("LOSO table must contain all eight anchors for every TWR log")
    return sequences


def calibration_for_sequence(
    calibration_path: Path, sequence: str, protocol_name: str
) -> tuple[np.ndarray, np.ndarray]:
    if protocol_name == "paper_oracle":
        return np.zeros(8), np.full(8, np.nan)
    if protocol_name == "no_oracle_raw":
        return np.zeros(8), np.full(8, 0.20)
    table = pd.read_csv(calibration_path)
    fold = table.loc[table["held_out_sequence"] == sequence].sort_values(
        "anchor_index"
    )
    if fold.shape[0] != 8 or fold["anchor_index"].tolist() != list(range(8)):
        raise ValueError(f"Incomplete LOSO calibration for {sequence}")
    return (
        fold["range_bias_m"].to_numpy(dtype=np.float64),
        fold["range_sigma_m"].to_numpy(dtype=np.float64),
    )


def position_metrics(error: np.ndarray) -> dict[str, float | bool]:
    error = np.asarray(error, dtype=np.float64)
    if error.ndim != 2 or error.shape[1] != 3 or error.shape[0] == 0:
        raise ValueError("position error must have shape (N, 3)")
    error_3d = np.linalg.norm(error, axis=1)
    error_xy = np.linalg.norm(error[:, :2], axis=1)
    error_z = np.abs(error[:, 2])
    return {
        "position_rmse_m": float(np.sqrt(np.mean(error_3d**2))),
        "position_xy_rmse_m": float(np.sqrt(np.mean(error_xy**2))),
        "position_z_rmse_m": float(np.sqrt(np.mean(error_z**2))),
        "position_p95_m": float(np.percentile(error_3d, 95)),
        "position_xy_p95_m": float(np.percentile(error_xy, 95)),
        "position_z_p95_m": float(np.percentile(error_z, 95)),
        "position_max_m": float(np.max(error_3d)),
        "position_xy_max_m": float(np.max(error_xy)),
        "position_z_max_m": float(np.max(error_z)),
        "success_below_0p5m": bool(np.sqrt(np.mean(error_3d**2)) < 0.5),
        "finite": bool(np.all(np.isfinite(error))),
    }


def _configure_author_params(
    dataset: Path, sequence: str, protocol: Protocol
) -> tuple[Any, np.ndarray]:
    from UWBsim.simulation import SimulationParams
    from UWBsim.utils.uwb_ranging import RangingSource, RangingType

    with (dataset / "anchor_positions.yaml").open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    anchors = np.asarray(
        [[document[index][axis] for axis in ("x", "y", "z")] for index in range(8)],
        dtype=np.float64,
    )
    params = SimulationParams()
    params.name = f"{protocol.name}:{sequence}"
    params.estimators.mhe.enable = True
    params.estimators.mhe.rate = 50
    params.estimators.mhe.N_max = 20
    params.estimators.mhe.iterations = 1
    params.estimators.mhe.ransac_iterations = 10
    params.estimators.mhe.ransac_fraction = 0.4
    params.estimators.mhe.ransac_threshold = 1.7
    params.estimators.mhe.mu = 10
    params.estimators.mhe.alpha = 0.015
    params.estimators.ekf.enable = True
    params.estimators.ekf.rate = 100
    params.estimators.ekf.outlierThreshold = 1500
    params.drone.altitude_enable = protocol.altitude_enabled
    params.drone.logfile = str(dataset / "twr" / sequence)
    params.ranging.rtype = RangingType.TWR
    params.ranging.source = RangingSource.LOG
    params.ranging.anchor_positions = anchors.tolist()
    params.ranging.anchor_enable = [True] * 8
    return params, anchors


def _install_truth_free_adapter(
    drone: Any, range_bias: np.ndarray, range_sigma: np.ndarray
) -> None:
    from UWBsim.utils.dataTypes import TWR_meas

    def fixed_standard_deviation(self: Any, measurement: Any) -> float:
        if isinstance(measurement, TWR_meas):
            return float(range_sigma[measurement.anchor_id])
        return float(measurement.stdDev)

    drone._get_meas_stdev_real = types.MethodType(fixed_standard_deviation, drone)
    original_get_twr = drone._get_twr

    def corrected_get_twr(self: Any, anchor_id: int, log_data: dict[str, float]) -> Any:
        measurement = original_get_twr(anchor_id, log_data)
        if measurement is not None:
            measurement.distance -= float(range_bias[anchor_id])
            measurement.stdDev = float(range_sigma[anchor_id])
        return measurement

    drone._get_twr = types.MethodType(corrected_get_twr, drone)


def _initialize_at_first_truth(drone: Any) -> None:
    from UWBsim.utils.dataTypes import State_XVQW

    position = np.array(
        [drone.log_data[axis] for axis in ("otX", "otY", "otZ")],
        dtype=np.float64,
    )
    initial = State_XVQW(x=position)
    for method, estimator in drone.estimators.items():
        estimator.reset(initial)
        drone.state_estimate[method] = State_XVQW(x=position)


def _install_timers(drone: Any) -> dict[str, list[int]]:
    elapsed: dict[str, list[int]] = {}
    for method, estimator in drone.estimators.items():
        elapsed[method] = []
        original_step = estimator.step

        def timed_step(
            simulation_time: float,
            *,
            original: Any = original_step,
            samples: list[int] = elapsed[method],
        ) -> Any:
            start = time.perf_counter_ns()
            result = original(simulation_time)
            samples.append(time.perf_counter_ns() - start)
            return result

        estimator.step = timed_step
    return elapsed


def run_one(job: dict[str, Any]) -> dict[str, Any]:
    author_repo = Path(job["author_repo"])
    sys.path.insert(0, str(author_repo))
    from UWBsim.simulation import UWBSimulation

    dataset = Path(job["dataset"])
    sequence = job["sequence"]
    protocol = PROTOCOLS[job["protocol"]]
    seed = int(job["seed"])
    range_bias = np.asarray(job["range_bias"], dtype=np.float64)
    range_sigma = np.asarray(job["range_sigma"], dtype=np.float64)
    params, _ = _configure_author_params(dataset, sequence, protocol)
    histories: dict[str, list[np.ndarray] | list[float]] = {
        "time": [],
        "truth": [],
        "mhe": [],
        "ekf": [],
    }

    def callback(drone: Any) -> None:
        if protocol.score_strictly_after_start:
            include = drone.time > protocol.score_start_s
        else:
            include = drone.time >= protocol.score_start_s
        if include:
            histories["time"].append(float(drone.time))
            histories["truth"].append(drone.state_true.x.copy())
            histories["mhe"].append(drone.state_estimate["mhe"].x.copy())
            histories["ekf"].append(drone.state_estimate["ekf"].x.copy())

    np.random.seed(seed)
    random.seed(seed)
    simulation = UWBSimulation(params, NotImplemented, callback)
    drone = simulation.drone1
    if not protocol.truth_dependent_range_sigma:
        _install_truth_free_adapter(drone, range_bias, range_sigma)
    if protocol.initial_position == "first_optitrack_position":
        _initialize_at_first_truth(drone)
    estimator_times = _install_timers(drone)
    wall_start = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        simulation.start_sim()
    wall_time = time.perf_counter() - wall_start

    truth = np.asarray(histories["truth"], dtype=np.float64)
    output: dict[str, Any] = {
        "sequence": sequence,
        "protocol": protocol.name,
        "seed": seed,
        "wall_time_s": wall_time,
        "score_samples": int(truth.shape[0]),
        "altitude_enabled": protocol.altitude_enabled,
        "runtime_truth_used_by_measurements": bool(
            protocol.altitude_enabled or protocol.truth_dependent_range_sigma
        ),
        "range_bias_m": range_bias,
        "range_sigma_m": range_sigma,
        "methods": {},
    }
    for method in ("mhe", "ekf"):
        estimate = np.asarray(histories[method], dtype=np.float64)
        samples = estimator_times[method]
        output["methods"][method] = {
            **position_metrics(estimate - truth),
            "estimator_steps": len(samples),
            "estimator_compute_time_s": float(np.sum(samples) * 1e-9),
            "mean_estimator_step_ms": float(np.mean(samples) * 1e-6),
        }
    if job["save_trace"]:
        output["trace"] = {
            "time": np.asarray(histories["time"], dtype=np.float64),
            "truth": truth,
            "mhe": np.asarray(histories["mhe"], dtype=np.float64),
            "ekf": np.asarray(histories["ekf"], dtype=np.float64),
        }
    return output


def flatten_results(results: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for result in results:
        for method, metrics in result["methods"].items():
            rows.append(
                {
                    "sequence": result["sequence"],
                    "protocol": result["protocol"],
                    "seed": result["seed"],
                    "method": method,
                    "wall_time_s": result["wall_time_s"],
                    "score_samples": result["score_samples"],
                    "altitude_enabled": result["altitude_enabled"],
                    "runtime_truth_used_by_measurements": result[
                        "runtime_truth_used_by_measurements"
                    ],
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def common_fair_table(author: pd.DataFrame, current_path: Path) -> pd.DataFrame:
    author_fair = author.loc[author["protocol"] == "no_oracle_loso"]
    author_per_sequence = (
        author_fair.groupby(["sequence", "method"], as_index=False)[
            ["position_rmse_m", "position_xy_rmse_m", "position_z_rmse_m"]
        ]
        .mean()
        .rename(columns={"method": "source_method"})
    )
    author_per_sequence["family"] = "Pfeiffer 2021 author code"
    author_per_sequence["method"] = author_per_sequence["source_method"].map(
        {"mhe": "author_mhe", "ekf": "author_ekf"}
    )

    current = pd.read_csv(current_path)
    current = current.loc[
        (current["offset"] == "nominal")
        & (current["configuration"] == "loso_per_anchor_robust")
    ].copy()
    current["position_z_rmse_m"] = np.sqrt(
        np.maximum(
            current["position_rmse_m"] ** 2 - current["position_xy_rmse_m"] ** 2,
            0.0,
        )
    )
    current["family"] = "Current Task 06"
    current["source_method"] = current["method"]
    current["method"] = current["method"].map(
        {
            "direct_twr": "task06_direct",
            "twr_inekf": "task06_inekf",
            "lifted_twr": "task06_lifted",
        }
    )
    columns = [
        "sequence",
        "family",
        "method",
        "source_method",
        "position_rmse_m",
        "position_xy_rmse_m",
        "position_z_rmse_m",
    ]
    return pd.concat(
        [author_per_sequence[columns], current[columns]], ignore_index=True
    )


METHOD_LABELS = {
    "author_mhe": "Author MHE",
    "author_ekf": "Author EKF",
    "task06_direct": "Task06 direct",
    "task06_inekf": "Task06 InEKF",
    "task06_lifted": "Task06 lifted",
}
METHOD_COLORS = {
    "author_mhe": "#4c78a8",
    "author_ekf": "#9ecae9",
    "task06_direct": "#f58518",
    "task06_inekf": "#7a5195",
    "task06_lifted": "#2ca02c",
}


def _boxplot_metric(
    fair: pd.DataFrame, metric: str, ylabel: str, title: str, output: Path
) -> None:
    order = list(METHOD_LABELS)
    values = [fair.loc[fair["method"] == method, metric].to_numpy() for method in order]
    figure, axis = plt.subplots(figsize=(11.5, 5.8), constrained_layout=True)
    boxes = axis.boxplot(values, tick_labels=[METHOD_LABELS[m] for m in order], patch_artist=True)
    for patch, method in zip(boxes["boxes"], order):
        patch.set_facecolor(METHOD_COLORS[method])
        patch.set_alpha(0.78)
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(axis="y", alpha=0.28)
    figure.savefig(output, dpi=190)
    plt.close(figure)


def plot_protocol_ablation(table: pd.DataFrame, output: Path) -> None:
    summary = table.groupby(["protocol", "method"], as_index=False)[
        ["position_rmse_m", "position_xy_rmse_m", "position_z_rmse_m"]
    ].mean()
    protocols = list(PROTOCOLS)
    figure, axes = plt.subplots(1, 2, figsize=(13.5, 5.5), constrained_layout=True)
    x_values = np.arange(len(protocols))
    width = 0.36
    for axis, method, color in zip(axes, ("mhe", "ekf"), ("#4c78a8", "#f58518")):
        group = summary.loc[summary["method"] == method].set_index("protocol")
        for index, (metric, label, alpha) in enumerate(
            (
                ("position_rmse_m", "3D", 1.0),
                ("position_xy_rmse_m", "XY", 0.68),
                ("position_z_rmse_m", "z", 0.42),
            )
        ):
            axis.bar(
                x_values + (index - 1) * width / 1.7,
                [group.loc[p, metric] for p in protocols],
                width / 1.7,
                color=color,
                alpha=alpha,
                label=label,
            )
        axis.set_xticks(x_values, ["paper oracle", "raw no oracle", "LOSO no oracle"])
        axis.set_ylabel("mean position RMSE [m]")
        axis.set_title(f"Author {method.upper()}")
        axis.grid(axis="y", alpha=0.25)
        axis.legend()
    figure.suptitle("Effect of removing OptiTrack-derived measurement assistance")
    figure.savefig(output, dpi=190)
    plt.close(figure)


def plot_representative(traces: dict[str, dict[str, np.ndarray]], output: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13.2, 9.0), constrained_layout=True)
    axes = axes.flat
    colors = {
        "paper_oracle": "#4c78a8",
        "no_oracle_raw": "#e45756",
        "no_oracle_loso": "#2ca02c",
    }
    truth = traces["paper_oracle"]["truth"]
    axes[0].plot(truth[:, 0], truth[:, 1], "k-", lw=2.0, label="OptiTrack")
    for protocol, trace in traces.items():
        label = protocol.replace("_", " ")
        axes[0].plot(trace["mhe"][:, 0], trace["mhe"][:, 1], color=colors[protocol], lw=1.0, label=f"MHE {label}")
        error = trace["mhe"] - trace["truth"]
        axes[1].plot(trace["time"], np.linalg.norm(error, axis=1), color=colors[protocol], lw=0.9, label=label)
        axes[2].plot(trace["time"], np.linalg.norm(error[:, :2], axis=1), color=colors[protocol], lw=0.9, label=label)
        axes[3].plot(trace["time"], np.abs(error[:, 2]), color=colors[protocol], lw=0.9, label=label)
    axes[0].set_title("MHE XY trajectory")
    axes[0].set_xlabel("x [m]")
    axes[0].set_ylabel("y [m]")
    axes[0].axis("equal")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    for axis, title, ylabel in (
        (axes[1], "3D error", "3D error [m]"),
        (axes[2], "XY error", "XY error [m]"),
        (axes[3], "Vertical error", "|z error| [m]"),
    ):
        axis.set_title(title)
        axis.set_xlabel("time [s]")
        axis.set_ylabel(ylabel)
        axis.set_yscale("symlog", linthresh=0.03)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.suptitle(f"Author-code protocol audit: {REPRESENTATIVE_SEQUENCE}")
    figure.savefig(output, dpi=190)
    plt.close(figure)


def build_summary(
    table: pd.DataFrame,
    fair: pd.DataFrame,
    author_sha: str,
    seeds: list[int],
) -> dict[str, Any]:
    protocol_summary: dict[str, Any] = {}
    for (protocol, method), group in table.groupby(["protocol", "method"]):
        protocol_summary[f"{protocol}/{method}"] = {
            "runs": int(group.shape[0]),
            "success_count": int(group["success_below_0p5m"].sum()),
            "position_rmse_mean_m": float(group["position_rmse_m"].mean()),
            "position_rmse_median_m": float(group["position_rmse_m"].median()),
            "position_xy_rmse_mean_m": float(group["position_xy_rmse_m"].mean()),
            "position_z_rmse_mean_m": float(group["position_z_rmse_m"].mean()),
            "mean_estimator_step_ms": float(group["mean_estimator_step_ms"].mean()),
        }
    fair_summary: dict[str, Any] = {}
    for method, group in fair.groupby("method"):
        fair_summary[method] = {
            "sequences": int(group.shape[0]),
            "position_rmse_mean_m": float(group["position_rmse_m"].mean()),
            "position_rmse_median_m": float(group["position_rmse_m"].median()),
            "position_xy_rmse_mean_m": float(group["position_xy_rmse_m"].mean()),
            "position_xy_rmse_median_m": float(group["position_xy_rmse_m"].median()),
            "position_z_rmse_mean_m": float(group["position_z_rmse_m"].mean()),
        }
    return {
        "author_repository": "https://github.com/SUPfeiffer/uwb-simulator",
        "author_commit": author_sha,
        "expected_author_commit": EXPECTED_AUTHOR_SHA,
        "author_source_unchanged": author_sha == EXPECTED_AUTHOR_SHA,
        "seeds": seeds,
        "protocols": {key: asdict(value) for key, value in PROTOCOLS.items()},
        "protocol_summary": protocol_summary,
        "fair_common_protocol_summary": fair_summary,
        "scope_limit": (
            "2021 SG+RANSAC MHE/EKF are upstream-source reproductions; "
            "2023/2025 LS/Newton/SVOR code and full tuning are not public and "
            "are therefore analyzed from the papers but not impersonated here."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--author-repo", type=Path, default=DEFAULT_AUTHOR_REPO)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--current-metrics", type=Path, default=DEFAULT_CURRENT_METRICS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--sequence", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    all_sequences = validate_inputs(
        args.dataset, args.author_repo, args.calibration, args.current_metrics
    )
    sequences = args.sequence or all_sequences
    unknown = sorted(set(sequences) - set(all_sequences))
    if unknown:
        raise ValueError(f"Unknown TWR sequence(s): {unknown}")
    if args.seeds < 1 or args.workers < 1:
        raise ValueError("--seeds and --workers must be positive")
    seeds = list(range(args.seeds))
    author_sha = read_author_sha(args.author_repo)
    if author_sha != EXPECTED_AUTHOR_SHA:
        raise RuntimeError(
            f"Author repository must be at {EXPECTED_AUTHOR_SHA}, found {author_sha}"
        )
    jobs: list[dict[str, Any]] = []
    for sequence in sequences:
        for protocol_name in PROTOCOLS:
            bias, sigma = calibration_for_sequence(
                args.calibration, sequence, protocol_name
            )
            for seed in seeds:
                jobs.append(
                    {
                        "dataset": str(args.dataset),
                        "author_repo": str(args.author_repo),
                        "sequence": sequence,
                        "protocol": protocol_name,
                        "seed": seed,
                        "range_bias": bias.tolist(),
                        "range_sigma": sigma.tolist(),
                        "save_trace": bool(
                            sequence == REPRESENTATIVE_SEQUENCE and seed == 0
                        ),
                    }
                )
    print(
        f"Running {len(jobs)} author-code jobs "
        f"({len(sequences)} logs x {len(PROTOCOLS)} protocols x {len(seeds)} seeds)"
    )
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_one, job) for job in jobs]
        for index, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            if index == 1 or index % max(1, len(seeds)) == 0 or index == len(jobs):
                print(f"completed {index}/{len(jobs)}", flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    traces: dict[str, dict[str, np.ndarray]] = {}
    for result in results:
        trace = result.pop("trace", None)
        if trace is not None:
            traces[result["protocol"]] = trace
            np.savez_compressed(
                args.output / f"trace_{result['protocol']}.npz", **trace
            )
    table = flatten_results(results).sort_values(
        ["protocol", "sequence", "seed", "method"]
    )
    table.to_csv(args.output / "author_protocol_metrics.csv", index=False)
    fair = common_fair_table(table, args.current_metrics)
    fair.to_csv(args.output / "fair_common_protocol_metrics.csv", index=False)
    summary = build_summary(table, fair, author_sha, seeds)
    with (args.output / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(_json_ready(summary), stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    plot_protocol_ablation(table, args.output / "truth_assistance_ablation.png")
    _boxplot_metric(
        fair,
        "position_rmse_m",
        "full-trajectory 3D RMSE [m]",
        "Fair common protocol: 3D position accuracy across 12 logs",
        args.output / "fair_3d_comparison.png",
    )
    _boxplot_metric(
        fair,
        "position_xy_rmse_m",
        "full-trajectory XY RMSE [m]",
        "Fair common protocol: XY position accuracy across 12 logs",
        args.output / "fair_xy_comparison.png",
    )
    if set(traces) == set(PROTOCOLS):
        plot_representative(traces, args.output / "representative_protocols.png")
    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
