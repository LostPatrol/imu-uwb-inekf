"""Replay literal position-only handoffs using the already frozen candidate artifacts.

The common terminal velocity is retained from the normally tracking shadow filter.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import analyze
import run


def finish_comparison(source, output):
    """Compare already scored panels without regenerating or rescoring trajectories."""
    keys = ["name", "scenario", "yaw_offset_deg", "policy"]
    source_metrics = pd.read_csv(source / "metrics.csv")
    current = pd.read_csv(output / "metrics.csv")
    analyze.complete_period(output, current)
    comparison = current.merge(source_metrics, on=keys, suffixes=("_position_only", "_pv_reset"))
    comparison.to_csv(output / "paired_to_pv_reset.csv", index=False)
    retain = current[current.policy == "retain_normal"].set_index(keys[:-1])
    paired = []
    for policy in ["retain_normal", "full_attitude", "evidence_yaw"]:
        values = current[current.policy == policy].set_index(keys[:-1])
        for metric in ["post_xy_rmse", "post_z_rmse", "post_yaw_rmse_deg"]:
            difference = values[metric] - retain[metric]
            paired.append(dict(policy=policy, metric=metric, median_difference=float(difference.median()),
                               improved=int((difference < -1e-9).sum()), worsened=int((difference > 1e-9).sum()),
                               unchanged=int((difference.abs() <= 1e-9).sum()), max_worsening=float(difference.max()),
                               best_improvement=float(difference.min())))
    pd.DataFrame(paired).to_csv(output / "paired_to_retain.csv", index=False)
    sources = [Path(__file__), Path(run.__file__), Path(run.kernel.__file__)]
    (output / "source_hashes.json").write_text(json.dumps(
        {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}, indent=2), encoding="utf8")


def main():
    """Reuse candidate rotations without fitting, deciding, or selecting with truth again."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="holdout_final")
    parser.add_argument("--output", default="holdout_position_only")
    parser.add_argument("--analysis-only", action="store_true")
    args = parser.parse_args()
    source = Path(__file__).parent / args.source
    output = Path(__file__).parent / args.output
    output.mkdir(exist_ok=True)
    if args.analysis_only:
        finish_comparison(source, output)
        return
    records = json.loads((source / "records.json").read_text(encoding="utf8"))
    selected = [row for row in records if row["policy"] in ["retain_normal", "full_attitude", "evidence_yaw"]
                and row["yaw_offset_deg"] in [0, 180]]
    completed = []
    data = None
    case_key, replay_cache = None, {}
    for original in selected:
        if data is None or data["name"] != original["name"]:
            data = run.load(original["name"], duration=run.PROTOCOL["duration_seconds"])
            data.pop("truth")
        cached = np.load(source / (original["key"] + ".npz"))
        handoff = cached["handoff"].copy()
        handoff[:3, 3] = cached["retained"][:3, 3]
        change = run.kernel.mapping(handoff)
        covariance = change @ run.PHYSICAL @ change.T
        current = (original["name"], original["scenario"], original["yaw_offset_deg"])
        if current != case_key:
            case_key, replay_cache = current, {}
        # Rejected yaw and retain have identical deterministic initial value problems.
        cache_key = handoff.tobytes()
        if cache_key not in replay_cache:
            replay_cache[cache_key] = run.bounded_replay(
                data, handoff, covariance, cached["bias"], original["handoff_time"], data["times"][-1], None)
        final, _, _, after, accepted, rejected, failed = replay_cache[cache_key]
        record = dict(original)
        record.update(panel="position_only", velocity_source="same normal shadow terminal velocity",
                      failed=bool(original["failed"] or failed), accepted_radio=accepted, rejected_radio=rejected)
        np.savez_compressed(output / (record["key"] + ".npz"),
                            before=cached["before"], after=after, handoff=handoff,
                            handoff_physical_covariance=run.PHYSICAL, bias=cached["bias"],
                            window_start_state=cached["window_start_state"], retained=cached["retained"], final=final)
        completed.append(record)
    (output / "records.json").write_text(json.dumps(run.safe_json(completed), indent=2), encoding="utf8")
    (output / "protocol.json").write_text(json.dumps(dict(source=args.source, panel="position_only",
        yaws=[0, 180], policies=["retain_normal", "full_attitude", "evidence_yaw"],
        information_policy="Same cached candidate R and shared antenna-derived p, same shadow v/bias and physical P; no new radio fit or heading decision.",
        primary_metric="Same post-handoff XY trajectory clock as the source PV-reset panel.",
        truth_usage="Score only after all position-only trajectories have been generated."), indent=2), encoding="utf8")
    print(f"Sensor-only position-only generation complete: {len(completed)} trajectories.", flush=True)
    run.score_split(output, completed)
    print(analyze.audit(output, completed), flush=True)
    finish_comparison(source, output)


if __name__ == "__main__":
    main()
