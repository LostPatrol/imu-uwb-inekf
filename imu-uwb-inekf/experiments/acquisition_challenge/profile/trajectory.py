"""Continuous wrong-prior MAP challenge with a fixed physical t=0 origin.

This explicitly separate protocol grows [0,t] through 15 seconds, never moving
the original prior to a later window or warm-starting from a previous estimate.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import time

from run import (np, pd, init, StartPrior, load_episode, make_window, sensor_seed,
                 attempt, to_row, digest)

# A one-second refresh is the requested bounded continuation of the snapshot panel.
ARRIVALS = tuple(float(value) for value in range(2, 16))
MODEL = "body_bias_prior"


def replay_batch(episode, output, confidence):
    """Execute independent per-prior timelines and retain every requested node."""
    data = load_episode(episode)
    prior_frame = pd.read_csv(episode/"priors.csv")
    prior_frame = prior_frame[prior_frame.confidence == confidence]
    windows = []
    previous = -1.
    for arrival in ARRIVALS:
        started = time.perf_counter()
        window, events, packet_hash, start = make_window(data, arrival, arrival)
        window_s = time.perf_counter()-started
        integration_s = float(data["integration_work"][(data["time"] > previous)
            & (data["time"] <= arrival)].sum())
        seed, candidate_s = sensor_seed(window)
        windows.append((arrival, window, events, packet_hash, start, seed,
            window_s+integration_s, candidate_s))
        previous = arrival
    cases = list(prior_frame.itertuples(index=False))
    # Only one batch computes native growing-window snapshots, then references reuse it.
    if confidence == pd.read_csv(episode/"priors.csv").confidence.iloc[0]:
        cases.insert(0, None)
    rows = []
    for case in cases:
        available_at = 0.
        for arrival, window, events, packet_hash, start, seed, preparation_s, candidate_s in windows:
            prior = None
            case_id = "native" if case is None else str(case.case_id)
            if case is not None:
                reference = window.start_rotation
                heading = np.arctan2(reference[1, 0], reference[0, 0])
                prior = StartPrior(np.array([case.px, case.py, case.pz]),
                    init.yaw_rotation(case.yaw_rad-heading) @ reference,
                    case.pos_std_m, case.yaw_std_rad)
            fit = attempt(window, MODEL, prior, seed)
            elapsed = preparation_s+candidate_s+fit["fit_elapsed_s"]
            compute_start = max(arrival, available_at)
            available_at = compute_start+elapsed
            row = to_row(episode.name, case_id, arrival, MODEL, fit, events, packet_hash, start, elapsed)
            row.update(compute_start_s=compute_start, publish_s=available_at,
                prior_time_s=0., protocol="growing_through_15_independently_sensor_seeded",
                candidate_s=candidate_s, incremental_preparation_s=preparation_s,
                confidence="none" if case is None else case.confidence,
                magnitude_m=np.nan if case is None else case.magnitude_m,
                direction="none" if case is None else case.direction,
                yaw_delta_deg=np.nan if case is None else case.yaw_delta_deg,
                independent_acquisition=episode.name)
            rows.append(row)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output/(episode.name+"__trajectory__"+confidence+".csv"), index=False)
    return dict(episode=episode.name, confidence=confidence, fits=len(rows))


def summarize(streams, output):
    """Consolidate complete timelines and preserve their distinct protocol label."""
    files = sorted(output.glob("*__trajectory__*.csv"))
    frame = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    frame.to_csv(output/"trajectory.csv", index=False)
    references = frame.loc[frame.case_id != "native", ["episode", "case_id", "method", "arrival_s"]].copy()
    references["source_csv"] = "trajectory.csv"
    references["native_case_id"] = "native"
    references["semantics"] = "native_unconditional_sensor_reset_ignores_prior"
    references["independent_repeated_run"] = False
    references.to_csv(output/"trajectory_reset_references.csv", index=False)
    protocol = dict(model=MODEL, arrivals_s=ARRIVALS,
        window="[0,t], growing independently through 15 seconds",
        external_prior_time_s=0., previous_state_or_covariance_consumed=False,
        seed="Original two-fit CV plus one lifted circle candidate; one raw refinement",
        shared_candidate_reused_across_prior_cases=True,
        nominal_per_case_latency_includes_full_candidate=True,
        per_case_publish_queue=True, same_recording_cases_not_independent_recordings=True,
        source_sha256={path.name:digest(path) for path in
            (Path(__file__), Path(__file__).with_name("run.py"), Path(__file__).with_name("solver.py"))},
        rows=len(frame), episodes=sorted(frame.episode.unique()))
    (output/"trajectory_protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")


def main():
    """Run the full fixed matrix with confidence batches and single-thread BLAS."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--streams", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--episodes", nargs="*")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    episodes = [path for path in sorted(args.streams.iterdir()) if (path/"priors.csv").exists()
                and (not args.episodes or path.name in args.episodes)]
    if not args.summarize_only:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            jobs = [pool.submit(replay_batch, episode, args.output, str(confidence))
                    for episode in episodes
                    for confidence in pd.read_csv(episode/"priors.csv").confidence.unique()]
            for job in as_completed(jobs):
                print("Completed", job.result(), flush=True)
    summarize(args.streams, args.output)


if __name__ == "__main__":
    main()
