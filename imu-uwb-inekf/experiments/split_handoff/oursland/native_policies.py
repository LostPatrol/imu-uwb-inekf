"""Recompute split handoffs with native-author-queue filtering on four fixed flights."""

import argparse
import importlib
import json
from pathlib import Path
import sys
import time

# Keep all reused implementation directories read-only.
sys.dont_write_bytecode = True
import native_schedule as native
import numpy as np
import pandas as pd

# Four cases cover every constellation and both radio protocols, fixed before scoring.
NAMES = ["const1-trial4-tdoa2", "const2-trial4-tdoa3", "const3-trial4-tdoa2", "const4-trial4-tdoa3-traj1"]
# Nominal and opposite initial headings keep the native-rate author cost bounded.
YAWS = [0., 180.]
# The same algorithm modules are reused; none is copied or weakened.
util = native.ours.util
# Native event sources are cached without their truth fields.
RAW = {}


def bounded_native(data, state, covariance, bias, start, end, outage):
    """Use identical author-main call times and .13 measurement variance downstream."""
    raw = RAW[data["name"]]
    origin = data["times"][0]
    interval = None if outage is None else (origin + outage[0], origin + outage[1])
    return native.replay(raw, state, covariance, bias, start, "current_kernel", interval, False,
                         end=end, return_final=True)


def main():
    """Generate all native split handoffs before scoring against unseen truth."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--names", nargs="+", default=NAMES)
    parser.add_argument("--yaws", nargs="+", type=float, default=YAWS)
    parser.add_argument("--scenarios", nargs="+", default=list(util.PROTOCOL["scenarios"]))
    parser.add_argument("--output", default="native_policies")
    parser.add_argument("--initial-prior", choices=["common", "original"], default="common")
    args = parser.parse_args()
    output = native.ours.HERE / args.output
    output.mkdir(parents=True, exist_ok=True)
    util.bounded_replay = bounded_native
    decisions = dict(evidence_yaw=importlib.import_module("yaw_evidence"),
                     bank_yaw=importlib.import_module("bank.bank"))
    records = []
    start_clock = time.perf_counter()
    for name in args.names:
        data = util.load(name, duration=45.)
        data.pop("truth")
        RAW[name] = native.load_raw(name, False)
        RAW[name].pop("truth")
        for scenario in args.scenarios:
            visible = util.visible_data(data, scenario)
            requested = data["times"][0] + util.PROTOCOL["scenarios"][scenario]["capture_end"]
            endpoint = data["times"][np.searchsorted(data["times"], requested)]
            path = util.radio_path(util.visible_data(data, scenario, endpoint))
            for yaw in args.yaws:
                if args.initial_prior == "original":
                    _, initial, _ = util.initial_state(visible)
                    initial[:3, :3] = native.Rotation.from_euler("z", yaw, degrees=True).as_matrix() @ initial[:3, :3]
                    inverse = util.kernel.inverse_mapping(initial)
                    # The original invariant P is exactly shared initially; its physical
                    # pullback is then the common covariance used by every handoff policy.
                    util.PHYSICAL = inverse @ native.ORIGINAL_P @ inverse.T
                checkpoint = output / f"case__{name}__{scenario}__yaw{yaw:g}.json"
                if checkpoint.exists():
                    records.extend(json.loads(checkpoint.read_text(encoding="utf8")))
                    continue
                case = util.run_case(visible, path, scenario, yaw, decisions, output)
                records.extend(case)
                checkpoint.write_text(json.dumps(util.safe_json(case), indent=2), encoding="utf8")
                print(json.dumps(dict(name=name, scenario=scenario, yaw=yaw,
                                      trajectories=len(records), elapsed_seconds=time.perf_counter() - start_clock)), flush=True)
    (output / "records.json").write_text(json.dumps(util.safe_json(records), indent=2), encoding="utf8")
    metrics, truths = [], {}
    for record in records:
        if record["name"] not in truths:
            truths[record["name"]] = util.load(record["name"], duration=45.)["truth"]
        artifact = np.load(output / (record["key"] + ".npz"))
        cut = util.PROTOCOL["scenarios"][record["scenario"]]["capture_end"]
        row = {k: v for k, v in record.items() if k not in ["full_candidate", "decision"]}
        row.update({"post_" + k: v for k, v in util.score_rows(truths[record["name"]], artifact["after"], record["origin"] + cut + .05, record["origin"] + 44.95).items()})
        whole = np.vstack([artifact["before"], artifact["after"]])
        row.update({"whole_" + k: v for k, v in util.score_rows(truths[record["name"]], whole, record["origin"] + .65, record["origin"] + 44.95).items()})
        if record["decision"] is not None:
            row["heading_accepted"] = bool(record["decision"].get("accepted", False))
            row["applied_delta_deg"] = float(np.degrees(record["decision"].get("delta", 0.)))
        metrics.append(row)
    pd.DataFrame(metrics).to_csv(output / "metrics.csv", index=False)
    provenance = dict(names=args.names, yaws=args.yaws, initial_prior=args.initial_prior, native_author_event_queue=True,
        filter_radio_variance=.13, candidate_model_radio_variance="original frozen .14/.22 retained",
        sensor_only=True, policies_recomputed=True, policy_source=str(util.HERE / "run.py"),
        source_sha256=native.ours.digest(util.HERE / "run.py"),
        wrapper="Raw native acc/gyro and original event dt in recursive prefix/shadow/post. Existing 100Hz integration remains the frozen batch candidate model. Output alone is resampled to common 20Hz for scoring.")
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf8")


if __name__ == "__main__":
    main()
