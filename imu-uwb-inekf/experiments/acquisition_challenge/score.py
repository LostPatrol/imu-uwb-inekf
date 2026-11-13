"""Evaluate causal publications against isolated truth, retaining censored episodes.

Sensor-time and measured serial publication clocks are scored side by side.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# Paths follow the replay drivers; only the scorer opens the reference directory.
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
SCRATCH = PROJECT / "agent/codex"
DEFAULT_DATA = SCRATCH / "acquisition_challenge_20261113"
RULE = json.loads((HERE / "protocol.json").read_text(encoding="utf-8"))
GRID = np.arange(0., RULE["horizon_s"] + .01, RULE["evaluation_grid_s"])
XYZ = ["x", "y", "z"]


def truth_at(ref, time):
    """Interpolate only within observed truth support and unwrap finite yaw first."""
    t = ref.t.to_numpy(float)
    values = [np.interp(time, t, ref[k], left=np.nan, right=np.nan) for k in XYZ]
    if "yaw_rad" in ref:
        values.append(np.interp(time, t, np.unwrap(ref.yaw_rad), left=np.nan, right=np.nan))
    else:
        values.append(np.full(len(time), np.nan))
    return np.asarray(values).T


def serial_clock(arrival, work):
    """Schedule each measurement-driven update after both its data and its predecessor."""
    publish = np.empty_like(arrival)
    previous = 0.
    for i, (t, cost) in enumerate(zip(arrival, work)):
        previous = max(t, previous) + max(float(cost), 0.)
        publish[i] = previous
    return publish


def threshold_time(t, error, bound):
    """Return entry and confirmation after a complete one-second below-threshold dwell."""
    n = int(round(RULE["dwell_s"] / RULE["evaluation_grid_s"])) + 1
    if len(t) < n:
        return np.nan, np.nan
    meets = np.convolve((error <= bound).astype(int), np.ones(n), mode="valid")
    first = np.flatnonzero(meets == n)
    return (t[first[0]], t[first[0]+n-1]) if first.size else (np.nan, np.nan)


def track_score(frame, ref, clock, coverage_s, work_type, publication_name):
    """Score a single causal trajectory on its registered span, never extrapolating truth."""
    g = frame.sort_values(["arrival_s", "state_s"], kind="stable")
    arrival = g.arrival_s.to_numpy(float)
    state = g.state_s.to_numpy(float)
    if np.any(state - arrival > 1e-6):
        raise ValueError(f"Published future state in {publication_name}")
    cost = g.wall_s.to_numpy(float)
    delta = np.diff(cost, prepend=0.) if work_type == "cumulative" else cost
    delivery = (g.publish_s.to_numpy(float) if "publish_s" in g and g.publish_s.notna().all()
                else serial_clock(arrival, delta))
    available = arrival if clock == "sensor" else delivery
    pos = g[XYZ].to_numpy(float)
    yaw = g.yaw_rad.to_numpy(float) if "yaw_rad" in g else np.full(len(g), np.nan)
    finite = np.flatnonzero(np.isfinite(pos).all(axis=1))
    if finite.size:
        first = finite[0]
        first_ref = truth_at(ref, [state[first]])[0]
        first_xy = np.linalg.norm(pos[first, :2] - first_ref[:2])
    else:
        first, first_xy = None, np.nan
    grid = GRID[GRID <= coverage_s + 1e-7]
    error = np.full((len(grid), 3), np.nan)
    yaw_error = np.full(len(grid), np.nan)
    if len(finite):
        last = np.searchsorted(available[finite], grid + 1e-8, "right") - 1
        valid = last >= 0
        j = finite[last[valid]]
        reference = truth_at(ref, grid[valid])
        error[valid] = pos[j] - reference[:, :3]
        diff = yaw[j] - reference[:, 3]
        yaw_error[valid] = np.abs(np.rad2deg(np.arctan2(np.sin(diff), np.cos(diff))))
    e = np.linalg.norm(error[:, :2], axis=1)
    terminal = len(pos) and not np.isfinite(pos[-1]).all()
    if terminal:
        after_failure = grid >= available[-1] - 1e-8
        error[after_failure], yaw_error[after_failure], e[after_failure] = np.nan, np.nan, np.nan
    supported = np.flatnonzero(np.isfinite(e))
    metrics = dict(first_sensor_s=arrival[first] if first is not None else np.nan,
                   first_publish_s=delivery[first] if first is not None else np.nan,
                   first_state_s=state[first] if first is not None else np.nan, first_xy_m=first_xy,
                   first_work_s=delta[first] if first is not None else np.nan, coverage_s=coverage_s,
                   estimate_count=len(finite), ref_first_s=ref.t.iloc[0],
                   scoreable_points=int(np.isfinite(e).sum()), grid_points=len(grid), terminal_failure=bool(terminal),
                   first_evaluable_grid_s=grid[supported[0]] if len(supported) else np.nan,
                   first_evaluable_xy_m=e[supported[0]] if len(supported) else np.nan)
    for bound, label in ((.25, "025"), (.5, "050"), (1., "100")):
        enter, confirm = threshold_time(grid, e, bound)
        metrics.update({f"entry_{label}_s": enter, f"confirm_{label}_s": confirm})
        if label == "050" and np.isfinite(confirm):
            i = np.flatnonzero(grid == confirm)[0]
            stays = bool(((e[i:] <= bound) & np.isfinite(e[i:])).all())
            metrics.update(xy_at_confirm_m=e[i], yaw_at_confirm_deg=yaw_error[i],
                           stayed_after_confirm_050=stays)
    for at in (2, 10, 15):
        match = np.flatnonzero(np.isclose(grid, at, rtol=0., atol=1e-8))
        metrics[f"xy_{at}s_m"] = e[match[0]] if len(match) else np.nan
        metrics[f"zabs_{at}s_m"] = abs(error[match[0], 2]) if len(match) else np.nan
        metrics[f"yaw_{at}s_deg"] = yaw_error[match[0]] if len(match) else np.nan
    return metrics


def datasets():
    """Locate estimator outputs by evidence class; archived trial runs are excluded."""
    profile = SCRATCH / "profile_challenge/results"
    sf = SCRATCH / "sfuise_challenge"
    jiang = HERE / "jiang/evidence"
    sources = []
    for path in sorted(profile.glob("*__native.csv")):
        sources.append((path, "native", None, "increment", 15))
    for path in sorted(profile.glob("*__trajectory__*.csv")):
        sources.append((path, "recovery_MAP_trajectory", None, "increment", 15))
    for path in sorted(profile.glob("*__stress_*.csv")):
        sources.append((path, "snapshot", None, "increment", 0))
    for path in sorted((sf / "results").glob("*/native.csv")):
        sources.append((path, "native", "SFUISE", "cumulative", 15))
    for path in sorted((sf / "stress10").glob("*/stress.csv")):
        sources.append((path, "seed_trajectory", "SFUISE", "cumulative", 10))
    for path in sorted((sf / "stress15").glob("*/stress.csv")):
        sources.append((path, "seed_trajectory15", "SFUISE", "cumulative", 15))
    for path in sorted(jiang.glob("*__native.csv")):
        sources.append((path, "native", "Jiang_independent", "increment", 15))
    for path in sorted(jiang.glob("*__stress.csv")):
        sources.append((path, "snapshot", "Jiang_independent", "increment", 0))
    if (jiang / "stress_10to15.csv").exists():
        sources.append((jiang / "stress_10to15.csv", "graph_MAP_trajectory", "Jiang_independent", "increment", 15))
    archived = SCRATCH / "pfeiffer_acquisition/traces"
    native_root = archived / "native" if (archived / "native").exists() else HERE / "pfeiffer/native"
    for path in sorted(native_root.glob("*__*__native_zero.csv")):
        sources.append((path, "native", None, "cumulative", 15))
    for name in ("stress_ekf", "stress_mhe"):
        root = archived / name if (archived / name).exists() else HERE / "pfeiffer" / name
        for path in sorted(root.glob("*__*__*.csv")):
            sources.append((path, "conditional_seed_or_prior", None, "cumulative", 10))
    return sources


def snapshot(frame, ref):
    """Score each single sensor-deadline MAP solution at its represented epoch."""
    time = frame.state_s.to_numpy(float)
    true = truth_at(ref, time)
    p = frame[XYZ].to_numpy(float)
    yaw = frame.yaw_rad.to_numpy(float) - true[:, 3]
    out = frame[["episode", "method", "case_id", "arrival_s", "state_s", "wall_s", "status"]].copy()
    out["xy_m"] = np.linalg.norm(p[:, :2] - true[:, :2], axis=1)
    out["zabs_m"] = abs(p[:, 2] - true[:, 2])
    out["yaw_deg"] = abs(np.rad2deg(np.arctan2(np.sin(yaw), np.cos(yaw))))
    return out


def main():
    """Write auditable per-flight and per-treatment scores with confidence aliases intact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=HERE / "results")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    refs = {p.stem: pd.read_csv(p) for p in (args.data / "scoring").glob("*.csv")
            if p.stem.endswith(("_start", "_recovery"))}
    priors = {p.parent.name: pd.read_csv(p).set_index("case_id")
              for p in (args.data / "streams").glob("*/priors.csv")}
    radios = {ep: pd.read_csv(args.data / "streams" / ep / "radio.csv",
                              usecols=["t","anchor_id"]) for ep in refs}
    score_rows, snap_rows = [], []
    for path, protocol, model, work_type, span in datasets():
        frame = pd.read_csv(path)
        if model:
            frame["method"] = model
        for episode, group in frame.groupby("episode", sort=False):
            if episode not in refs:
                continue
            ref = refs[episode]
            if protocol == "snapshot":
                snap_rows.append(snapshot(group, ref))
                continue
            for (method, case), trajectory in group.groupby(["method", "case_id"], sort=False):
                extra = {}
                if episode in priors and case in priors[episode].index:
                    extra = priors[episode].loc[case, ["magnitude_m", "direction", "yaw_delta_deg", "confidence"]].to_dict()
                for clock in ("sensor", "measured_queue"):
                    scores = track_score(trajectory, ref, clock, span, work_type, f"{method}/{episode}/{case}")
                    if protocol == "native":
                        early = radios[episode][radios[episode].t <= scores["first_sensor_s"]+1e-8]
                        scores.update(raw_ranges_by_first_output=len(early),
                                      distinct_anchors_by_first_output=early.anchor_id.nunique())
                    score_rows.append(dict(episode=episode, dataset=episode.split("_")[0],
                        kind=episode.rsplit("_", 1)[-1], method=method, case_id=case, protocol=protocol,
                        clock=clock, **extra, **scores))
    if score_rows:
        scores = pd.DataFrame(score_rows)
        scores.to_csv(args.out / "acquisition_scores.csv", index=False, float_format="%.8g")
        cold = scores[(scores.kind == "start") & (scores.protocol == "native")].groupby(
            ["dataset", "method", "clock"], as_index=False).agg(
            flights=("episode", "nunique"), confirmed=("confirm_050_s", "count"),
            median_release_to_confirm_s=("confirm_050_s", "median"),
            median_first_sensor_s=("first_sensor_s", "median"), median_first_publish_s=("first_publish_s", "median"),
            first_xy_supported=("first_xy_m", "count"), median_first_xy_m=("first_xy_m", "median"),
            median_xy_at_confirmation_m=("xy_at_confirm_m", "median"),
            median_xy_2s_m=("xy_2s_m", "median"), median_xy_10s_m=("xy_10s_m", "median"),
            median_yaw_10s_deg=("yaw_10s_deg", "median"), median_xy_15s_m=("xy_15s_m", "median"))
        cold["median_full_input_to_confirm_s"] = cold.median_release_to_confirm_s + RULE["prefix_s"]
        cold.to_csv(args.out / "cold_summary.csv", index=False, float_format="%.8g")
        scores[(scores.protocol == "native") & (scores.kind == "start")].to_csv(
            args.out / "cold_recordings.csv", index=False, float_format="%.8g")
        print(cold[cold.clock == "sensor"].to_string(index=False))
    if snap_rows:
        raw = pd.concat(snap_rows, ignore_index=True)
        raw = raw.drop_duplicates(["episode", "method", "case_id", "state_s"], keep="last")
        index = pd.concat([v.reset_index().assign(episode=k) for k, v in priors.items()])
        raw.merge(index[["episode", "case_id", "magnitude_m", "direction", "yaw_delta_deg", "confidence"]],
                  on=["episode", "case_id"], how="left").to_csv(
                  args.out / "snapshot_scores.csv", index=False, float_format="%.8g")


if __name__ == "__main__":
    main()
