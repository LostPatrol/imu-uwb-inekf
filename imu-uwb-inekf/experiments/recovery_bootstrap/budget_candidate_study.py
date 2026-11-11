"""Compare fixed yaw-candidate budgets on two existing far-field simulations.

The two CV translation fits select one translation; each yaw then has one raw
seven-variable refinement. Timings include CV, profiling and final curvature.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import time

# Every process uses one numerical-library thread, including sequential timing.
for _variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_variable] = "1"
os.environ["MPLBACKEND"] = "Agg"

import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import numpy as np
import pandas as pd
import scipy
from scipy.optimize import least_squares

import initializer
from initializer import (CURVATURE_PRIOR_STD, HUBER_SCALE, MAX_EVALUATIONS,
                         lifted_candidate, project_out, raw_residual_jacobian,
                         robust_cost, yaw_rotation)
from run_geometry_stress import (OBJECTIVE_ABS_TOL, OBJECTIVE_REL_TOL,
                                 RANGE_SIGMA, TRANSACTIONS, WINDOW_SECONDS)
from window_candidate_study import complete_cv

# Resolve source and output relative to the checked-in experiment entry point.
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
# Uniform policies include zero, while twelve starts reproduce the old reference.
METHODS = ("local", "profile_1", "uniform_2", "uniform_4", "uniform_12")
# Display names explicitly count yaw candidates rather than CV translation fits.
LABELS = dict(zip(METHODS, ("Local", "Profile-1", "Uniform-2", "Uniform-4", "Uniform-12")))
# Match the original two excited far-field geometries without adding a new grid.
HEIGHTS = (3.0, 0.1)
# Use explicit SciPy defaults consistently for all raw refinements.
SOLVER_TOL = 1e-8


def write_json(path, value):
    """Write deterministic, readable UTF-8 metadata."""
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def case_arrays(specification):
    """Reproduce the frozen geometry generator with unchanged RNG call order."""
    case_id, seed, distance, height, motion = specification
    if distance != 30.0 or motion != "excited":
        raise ValueError("Only the two preregistered excited far-field groups are supported")
    rng = np.random.default_rng(seed)
    tau = np.sort(rng.uniform(-WINDOW_SECONDS, 0.0, TRANSACTIONS))
    anchor_set = np.array([[x, y, z] for x in (-4.0, 4.0)
                           for y in (-4.0, 4.0) for z in (0.0, height)])
    anchors = anchor_set[rng.integers(0, len(anchor_set), TRANSACTIONS)]
    bearing = rng.uniform(-np.pi, np.pi)
    position = np.array([distance*np.cos(bearing), distance*np.sin(bearing), 1.2])
    velocity = rng.uniform(-0.5, 0.5, 3)
    yaw = rng.uniform(-np.pi, np.pi)
    displacement = np.column_stack((0.8*np.sin(1.7*tau),
        0.6*(np.cos(1.1*tau)-1.0), 0.15*np.sin(0.7*tau)))
    displacement -= tau[:, None]*np.array([1.36, 0.0, 0.105])
    positions = position + tau[:, None]*velocity + displacement@yaw_rotation(yaw).T
    ranges = np.linalg.norm(positions-anchors, axis=1) + rng.normal(0.0, RANGE_SIGMA, TRANSACTIONS)
    sigmas = np.full(TRANSACTIONS, RANGE_SIGMA)
    return (tau, displacement, anchors, ranges, sigmas), position, yaw


def solve_policy(values, method):
    """Include both common CV fits, every yaw refinement and final curvature."""
    started = time.perf_counter()
    tau, displacement, anchors, ranges, sigmas = values
    (cv, _, _), cv_nfev, cv_calls = complete_cv(tau, anchors, ranges, sigmas)
    candidate_s = 0.0
    if method == "profile_1":
        profile_started = time.perf_counter()
        candidate, _ = lifted_candidate(*values)
        angles = [candidate[6]]
        candidate_s = time.perf_counter()-profile_started
    elif method == "local":
        angles = [0.0]
    else:
        count = int(method.rsplit("_", 1)[1])
        angles = np.linspace(-np.pi, np.pi, count, endpoint=False)

    def residual(parameters):
        """Evaluate the same normalized asynchronous raw-range residuals."""
        return raw_residual_jacobian(parameters, *values)[0]

    def jacobian(parameters):
        """Evaluate the same analytic seven-variable Jacobian."""
        return raw_residual_jacobian(parameters, *values)[1]

    fits = [least_squares(residual, np.r_[cv, angle], jac=jacobian,
            loss="huber", f_scale=HUBER_SCALE, max_nfev=MAX_EVALUATIONS,
            ftol=SOLVER_TOL, xtol=SOLVER_TOL, gtol=SOLVER_TOL) for angle in angles]
    best = min(fits, key=lambda fit: robust_cost(fit.fun))
    final_residual, final_jacobian = raw_residual_jacobian(best.x, *values)
    # Keep the same post-fit curvature work as the original initializer.
    weights = np.minimum(1.0, HUBER_SCALE/np.maximum(np.abs(final_residual), 1e-12))
    weighted = final_jacobian*np.sqrt(weights[:, None])
    projected, _ = project_out(weighted[:, :6], weighted[:, 6:7])
    covariance = np.linalg.inv(weighted.T@weighted + np.diag(1/CURVATURE_PRIOR_STD**2))
    elapsed_s = time.perf_counter()-started
    parameters = best.x.copy()
    parameters[6] = np.arctan2(np.sin(parameters[6]), np.cos(parameters[6]))
    raw_nfev = sum(fit.nfev for fit in fits)
    return dict(parameters=parameters, objective=robust_cost(best.fun),
        cv_nfev=cv_nfev, raw_nfev=raw_nfev, total_nfev=cv_nfev+raw_nfev,
        cv_calls=cv_calls, yaw_candidates=len(angles), raw_calls=len(fits),
        elapsed_s=elapsed_s, profile_s=candidate_s, solver_success=bool(best.success),
        capped_raw_calls=sum(fit.nfev == MAX_EVALUATIONS for fit in fits),
        finite=bool(np.isfinite(parameters).all() and np.isfinite(covariance).all()),
        yaw_information=float(np.sum(projected**2)))


def evaluate(specification):
    """Evaluate all five policies on identical draws with balanced execution order."""
    values, position, yaw = case_arrays(specification)
    rows = []
    for method in np.roll(np.asarray(METHODS), specification[0] % len(METHODS)):
        fit = solve_policy(values, str(method))
        parameters = fit.pop("parameters")
        error = parameters[:3]-position
        yaw_error = np.arctan2(np.sin(parameters[6]-yaw), np.cos(parameters[6]-yaw))
        rows.append(dict(case_id=specification[0], seed=specification[1],
            distance_m=specification[2], anchor_height_m=specification[3], method=str(method),
            position_error_m=float(np.linalg.norm(error)),
            horizontal_error_m=float(np.linalg.norm(error[:2])),
            height_error_m=float(abs(error[2])), yaw_error_deg=float(abs(np.rad2deg(yaw_error))),
            estimated_px=parameters[0], estimated_py=parameters[1], estimated_pz=parameters[2],
            estimated_yaw_rad=parameters[6], **fit))
    return rows


def summarize(frame, timing, output):
    """Preserve negative results and distinguish objective misses from truth errors."""
    frame = frame.sort_values(["case_id", "method"]).reset_index(drop=True)
    best = frame.groupby("case_id")["objective"].transform("min")
    tolerance = np.maximum(OBJECTIVE_ABS_TOL, OBJECTIVE_REL_TOL*np.abs(best))
    frame["objective_gap_best_found"] = frame.objective-best
    frame["relative_objective_gap"] = (frame.objective-best)/(1+best)
    frame["objective_miss"] = (frame.objective-best > tolerance) | ~frame.finite
    reference = frame[frame.method == "uniform_12"].set_index("case_id").objective
    reference = frame.case_id.map(reference)
    reference_tolerance = np.maximum(OBJECTIVE_ABS_TOL, OBJECTIVE_REL_TOL*np.abs(reference))
    frame["objective_gap_vs_uniform_12"] = frame.objective-reference
    frame["objective_miss_vs_uniform_12"] = (frame.objective-reference > reference_tolerance) | ~frame.finite
    frame["objective_better_than_uniform_12"] = frame.objective-reference < -reference_tolerance
    frame.to_csv(output/"metrics.csv", index=False)
    rows = []
    for height in HEIGHTS:
        for method in METHODS:
            part = frame[(frame.anchor_height_m == height) & (frame.method == method)]
            measured = timing[(timing.anchor_height_m == height) & (timing.method == method)]
            row = dict(anchor_height_m=height, method=method, cases=len(part),
                objective_misses=int(part.objective_miss.sum()),
                objective_misses_vs_uniform_12=int(part.objective_miss_vs_uniform_12.sum()),
                objective_better_than_uniform_12=int(part.objective_better_than_uniform_12.sum()),
                position_failures_gt_0_5m=int((part.position_error_m > 0.5).sum()),
                yaw_failures_gt_0_35rad=int((np.deg2rad(part.yaw_error_deg) > 0.35).sum()),
                solver_nonconverged=int((~part.solver_success).sum()),
                nonfinite=int((~part.finite).sum()), total_nfev=int(part.total_nfev.sum()),
                median_nfev=float(part.total_nfev.median()), p95_nfev=float(part.total_nfev.quantile(.95)),
                timing_fits=len(measured), median_ms=float(measured.elapsed_s.median()*1000),
                p95_ms=float(measured.elapsed_s.quantile(.95)*1000),
                median_profile_ms=float(measured.profile_s.median()*1000))
            for key in ("position_error_m", "horizontal_error_m", "height_error_m", "yaw_error_deg"):
                row.update({key+"_"+stat: float(value) for stat, value in
                    (("median", part[key].median()), ("p95", part[key].quantile(.95)), ("max", part[key].max()))})
            rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(output/"summary.csv", index=False)
    write_json(output/"summary.json", rows)
    sensitivity = []
    for tolerance in (1e-5, 1e-4, 1e-3):
        for (height, method), part in frame.groupby(["anchor_height_m", "method"]):
            sensitivity.append(dict(anchor_height_m=height, method=method,
                relative_tolerance=tolerance, misses=int((part.relative_objective_gap > tolerance).sum())))
    pd.DataFrame(sensitivity).to_csv(output/"tolerance_sensitivity.csv", index=False)
    latex = [r"% One yaw candidate means one 7D fit after two common 6D CV fits.",
        r"% NFEV and ms are medians; timing uses 8 cases/group x 5 repeats, including profile overhead.",
        r"% Miss compares all-five-policy joint best with max(1e-3,1e-5*abs(best)); not a global certificate.",
        r"\begin{tabular}{llrrrrr}", r"\hline",
        r"Height & Policy & Miss & $p_{95}$ (m) & $\psi_{95}$ ($^\circ$) & NFEV & ms \\",
        r"\hline"]
    for row in rows:
        latex.append(f"{row['anchor_height_m']:g} & {LABELS[row['method']]} & "
            f"{row['objective_misses']}/{row['cases']} & {row['position_error_m_p95']:.2f} & "
            f"{row['yaw_error_deg_p95']:.2f} & {row['median_nfev']:g} & {row['median_ms']:.1f} " + r"\\")
    latex.extend([r"\hline", r"\end{tabular}"])
    (output/"budget_table.tex").write_text("\n".join(latex)+"\n", encoding="utf-8")
    render(summary, output)
    print(summary.to_string(index=False), flush=True)
    return frame


def render(summary, output):
    """Plot actual measured cost against objective misses and truth-error tails."""
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 3, figsize=(10.2, 5.7))
    colors = ("#b23b3b", "#0067a1", "#b17700", "#5a7a00", "#555555")
    for row_index, height in enumerate(HEIGHTS):
        for method, color in zip(METHODS, colors):
            row = summary[(summary.anchor_height_m == height) & (summary.method == method)].iloc[0]
            for column, key in enumerate(("objective_misses", "position_error_m_p95", "yaw_error_deg_p95")):
                value = row[key]/row.cases*100 if column == 0 else row[key]
                ax = axes[row_index, column]
                ax.scatter(row.median_ms, value, color=color, label=LABELS[method], s=36)
                ax.annotate(LABELS[method], (row.median_ms, value), xytext=(4, 4),
                            textcoords="offset points", fontsize=7, color=color)
                ax.grid(alpha=.2)
                ax.set_xscale("log")
                ax.set_xlim(summary[summary.anchor_height_m == height].median_ms.min()*.9,
                            summary[summary.anchor_height_m == height].median_ms.max()*1.5)
                ax.set_xticks([50, 100, 200] if height == 3.0 else [100, 200, 400])
                ax.xaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
                ax.minorticks_off()
                ax.set_xlabel("Median complete wall clock (ms)")
                ax.set_title(f"Anchor height span {height:g} m")
                ax.set_ylabel(("Objective miss (%)", "Position error p95 (m)", "Yaw error p95 (deg)")[column])
    fig.suptitle("30 m far field; 200 shared cases per geometry", y=1.01)
    fig.tight_layout()
    fig.savefig(output/"candidate_budget.png", dpi=220, bbox_inches="tight")
    fig.savefig(output/"candidate_budget.pdf", bbox_inches="tight")
    plt.close(fig)


def render_compact(summary, output):
    """Render a legible single-column cost/miss figure from existing summaries."""
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9,
                         "axes.labelsize": 8, "xtick.labelsize": 8,
                         "ytick.labelsize": 8, "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 1, figsize=(3.4, 3.35), sharex=True)
    labels = ("Local", "P1", "U2", "U4", "U12")
    markers = ("x", "o", "s", "^", "D")
    # Orange local, teal profile, and distinct cool hues for uniform policies.
    colors = ("#D66A3A", "#007F83", "#3465A4", "#8A65AC", "#65748B")
    for index, (ax, height) in enumerate(zip(axes, HEIGHTS)):
        for method, label, marker, color in zip(METHODS, labels, markers, colors):
            row = summary[(summary.anchor_height_m == height) & (summary.method == method)].iloc[0]
            miss_percent = row.objective_misses/row.cases*100
            style = dict(color=color, s=32, marker=marker, linewidths=1.1)
            if marker != "x":
                style["edgecolors"] = "white"
            ax.scatter(row.median_ms, miss_percent, **style)
            ax.annotate(label, (row.median_ms, miss_percent), xytext=(0, 6),
                        textcoords="offset points", ha="center", fontsize=8,
                        color=color, weight="bold" if method == "profile_1" else "normal")
        ax.set_xscale("log")
        ax.set_xlim(40, 600)
        ax.set_xticks([50, 100, 200, 400])
        ax.xaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
        ax.minorticks_off()
        ax.set_ylim(-2.5, 40.5)
        ax.set_yticks([0, 10, 20, 30])
        ax.set_ylabel("Objective miss (%)")
        ax.set_title(f"({'ab'[index]}) Anchor height span {height:g} m", loc="left")
        ax.grid(color="#E0E6EC", linewidth=.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#9AA9B8")
        ax.set_axisbelow(True)
    axes[-1].set_xlabel("Median complete solve time (ms; log scale)")
    fig.subplots_adjust(left=.18, right=.97, top=.91, bottom=.15, hspace=.52)
    # Preserve the physical 3.4-inch width; avoid tight bounding-box rescaling.
    fig.savefig(output/"candidate_budget_compact.pdf")
    fig.savefig(output/"candidate_budget_compact.png", dpi=300)
    plt.close(fig)


def sequential_timing(cases, repeats, output):
    """Exclude one all-policy warmup, then interleave fixed-case timing repeats."""
    values, _, _ = case_arrays(cases[0])
    for method in METHODS:
        solve_policy(values, method)
    rows = []
    for index, specification in enumerate(cases):
        values, _, _ = case_arrays(specification)
        for repeat in range(repeats):
            for method in np.roll(np.asarray(METHODS), (index+repeat) % len(METHODS)):
                fit = solve_policy(values, str(method))
                fit.pop("parameters")
                rows.append(dict(case_id=specification[0], anchor_height_m=specification[3],
                                 repeat=repeat, method=str(method), **fit))
    frame = pd.DataFrame(rows)
    frame.to_csv(output/"sequential_timing.csv", index=False)
    return frame


def validate(frame, original, timing, output):
    """Check common-policy parity and deterministic repeated-fit objectives."""
    saved = pd.read_csv(original/"metrics.csv")
    mapping = {"local": "local", "profile_1": "profile", "uniform_12": "multistart"}
    differences = {}
    position_differences = {}
    for method, old_method in mapping.items():
        new = frame[frame.method == method].set_index("case_id")
        old = saved[saved.method == old_method].set_index("case_id").loc[new.index]
        delta = np.max(np.abs(new.objective.to_numpy()-old.robust_objective.to_numpy()))
        # Cross-host differences are checked at the original scientific objective scale.
        np.testing.assert_allclose(new.objective, old.robust_objective, rtol=0, atol=OBJECTIVE_ABS_TOL)
        differences[method] = float(delta)
        # The archived run used another host; tiny flat-basin state differences are reported.
        position_differences[method] = float(np.max(np.abs(new.position_error_m.to_numpy()-old.position_error_m.to_numpy())))
    paired = timing.merge(frame[["case_id", "method", "objective"]],
                           on=["case_id", "method"], suffixes=("_timing", "_main"))
    np.testing.assert_allclose(paired.objective_timing, paired.objective_main, rtol=0, atol=1e-10)
    assert (frame.cv_calls == 2).all() and (frame.yaw_candidates == frame.raw_calls).all()
    current_differences = {}
    parent = json.loads((original/"protocol.json").read_text(encoding="utf-8"))
    for height in HEIGHTS:
        case = next(case for case in parent["cases"] if case[2:] == [30.0, height, "excited"])
        values, _, _ = case_arrays(case)
        for method, old_method in mapping.items():
            direct = initializer.solve_window(*values, method=old_method)
            saved_objective = frame[(frame.case_id == case[0]) & (frame.method == method)].objective.iloc[0]
            delta = abs(direct.robust_objective-saved_objective)
            assert delta < 1e-10, (case[0], method, delta)
            current_differences[f"{case[0]}_{method}"] = float(delta)
    write_json(output/"validation.json", dict(original_objective_max_abs_difference=differences,
        original_position_error_max_abs_difference_m=position_differences,
        archived_cross_host_objective_tolerance=OBJECTIVE_ABS_TOL,
        direct_current_initializer_objective_differences=current_differences,
        timing_max_abs_objective_difference=float(np.max(np.abs(paired.objective_timing-paired.objective_main))),
        common_cv_calls=2, one_raw_fit_per_yaw_verified=True, all_finite=bool(frame.finite.all())))


def main():
    """Freeze a reduced existing-case panel before measuring candidate budgets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=200)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timing-per-group", type=int, default=8)
    parser.add_argument("--timing-repeats", type=int, default=5)
    parser.add_argument("--timing-only", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--main-only", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--output", type=Path, default=PROJECT/"results/recovery_candidate_budget_20261111")
    args = parser.parse_args()
    if args.plot_only:
        render_compact(pd.read_csv(args.output/"summary.csv"), args.output)
        protocol_path = args.output/"protocol.json"
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        protocol.update(publication_plot_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                        publication_plot_summary_sha256=hashlib.sha256((args.output/"summary.csv").read_bytes()).hexdigest(),
                        publication_plot="candidate_budget_compact.pdf/png; 3.4 inch width, minimum 8 pt text; saved summary only, no fitting or timing")
        write_json(protocol_path, protocol)
        return
    original = PROJECT/"results/recovery_geometry_20261107"
    parent = json.loads((original/"protocol.json").read_text(encoding="utf-8"))
    groups = [[case for case in parent["cases"] if case[2:] == [30.0, height, "excited"]]
              for height in HEIGHTS]
    if args.seeds < 1 or args.seeds > min(map(len, groups)):
        raise ValueError("Seed count must select an existing prefix, from 1 to 200")
    cases = [case for group in groups for case in group[:args.seeds]]
    timing_cases = [case for group in groups for case in group[:min(args.seeds, args.timing_per_group)]]
    args.output.mkdir(parents=True, exist_ok=True)
    if args.timing_only or args.summarize_only:
        timing_started = time.perf_counter()
        timing = (pd.read_csv(args.output/"sequential_timing.csv") if args.summarize_only else
                  sequential_timing(timing_cases, args.timing_repeats, args.output))
        frame = summarize(pd.read_csv(args.output/"metrics.csv"), timing, args.output)
        validate(frame, original, timing, args.output)
        protocol_path = args.output/"protocol.json"
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        protocol.update(completed_utc=datetime.now(timezone.utc).isoformat(),
            recorded_policy_elapsed_s=float(timing.elapsed_s.sum()),
            summary_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        if args.timing_only:
            protocol["timing_elapsed_s"] = time.perf_counter()-timing_started
            protocol["timing_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        else:
            protocol.pop("timing_elapsed_s", None)
        write_json(protocol_path, protocol)
        return
    if (args.output/"metrics.csv").exists():
        raise FileExistsError("Completed results will not be overwritten")
    protocol = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        parent_protocol_sha256=hashlib.sha256((original/"protocol.json").read_bytes()).hexdigest(),
        cases=cases, methods=METHODS, seeds_per_geometry=args.seeds, workers=args.workers,
        common_translation="Two 6D robust CV fits (squared-range seed and anchor-centroid seed), select best; one selected translation paired with every yaw",
        candidate_definition="One yaw candidate means one 7D refinement, not two; Profile-1 uses the best relaxed quartic/auxiliary angle",
        uniform_angles="linspace(-pi,pi,K,endpoint=False); Local yaw=0; K=2,4,12",
        model="Existing exact calibrated displacement, known gravity, no IMU mismatch; Gaussian range sigma=0.1 m, 240 events, 4 seconds",
        optimization=dict(loss="huber", f_scale=HUBER_SCALE, max_nfev_per_raw_fit=MAX_EVALUATIONS,
                          ftol=SOLVER_TOL, xtol=SOLVER_TOL, gtol=SOLVER_TOL),
        objective_miss="Cost exceeds best found across all five policies by max(1e-3,1e-5*abs(best)); no global certificate",
        nfev="Sum actual scipy nfev of both CV fits plus all 7D fits; profiling work is not nfev and is included in wall clock",
        budget_scope="Fixed candidate-count policies with measured actual costs, not equal elapsed-time or equal-total-nfev limits",
        sequential_timing_cases=timing_cases, timing_repeats=args.timing_repeats,
        timing_protocol="One unrecorded five-policy warmup on first case; one process, one BLAS thread, rotate method order by case/repeat; include CV, profile, all raw fits and post-fit curvature; exclude data generation and I/O",
        timing_environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                                platform=platform.platform(), processor=platform.processor(), blas_threads=1),
        source_sha256={path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in
            (Path(__file__), HERE/"initializer.py", HERE/"run_geometry_stress.py", HERE/"window_candidate_study.py")})
    write_json(args.output/"protocol.json", protocol)
    started = time.perf_counter()
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for index, result in enumerate(pool.map(evaluate, cases), 1):
            rows.extend(result)
            if index % 40 == 0 or index == len(cases):
                print(f"Completed {index}/{len(cases)} cases in {time.perf_counter()-started:.1f}s", flush=True)
    # Main-run timing is saved as a loaded diagnostic; publication timing is serial.
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output/"metrics.csv", index=False)
    if args.main_only:
        protocol.update(main_completed_utc=datetime.now(timezone.utc).isoformat(),
                        main_elapsed_s=time.perf_counter()-started)
        write_json(args.output/"protocol.json", protocol)
        print("MAIN COMPLETE; run --timing-only when other fitting work is idle", flush=True)
        return
    print("MAIN COMPLETE; sequential timing starts", flush=True)
    timing = sequential_timing(timing_cases, args.timing_repeats, args.output)
    frame = summarize(frame, timing, args.output)
    validate(frame, original, timing, args.output)
    protocol.update(completed_utc=datetime.now(timezone.utc).isoformat(), elapsed_s=time.perf_counter()-started)
    write_json(args.output/"protocol.json", protocol)


if __name__ == "__main__":
    main()
