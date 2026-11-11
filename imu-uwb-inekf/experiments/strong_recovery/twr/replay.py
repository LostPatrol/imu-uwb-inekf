"""Native TWR strong-branch caches for fixed restore time and gap lengths.

The original lifted capture remains intact. Its ten-second traces are reused
from this round's exact reproduction; new lengths only change radio masking.
"""

from pathlib import Path
import argparse
import json
import sys

import numpy as np

# Task-owned artifacts are independent of the manuscript and frozen Delft matrices.
CODE = Path(__file__).resolve().parents[3]
ROOT = CODE.parent
OUT = ROOT / "agent/codex/strong_twr/length_cache"
CACHE = CODE / "experiments/split_handoff/delft/results/cache"
sys.path.insert(0, str(Path(__file__).resolve().parent))
import diagnose
from cpu_replay import replay_pool

# Fixed lengths and common recovery clock were chosen before running this matrix.
LENGTHS = (2.0, 5.0, 10.0, 20.0)
RESUME = 25.0


def identifier(name, length):
    """Return a deterministic filename for a development-domain case."""
    return Path(name).stem+f"__gap{int(length)}"


def prepare_case(job):
    """Preserve original capture output and assemble only truly arrived raw TWR."""
    name, length = job
    target = OUT/(identifier(name, length)+".npz")
    if target.exists():
        return dict(sequence=name, gap_s=length, cached=True)
    data = dict(np.load(CACHE/(Path(name).stem+".npz")))
    protocol = json.loads((CODE/"results/roadmap_stage1_20261104_final/protocol.json").read_text(encoding="utf-8"))
    model = protocol["range_models"]["holdout"]
    times = data["time"]
    valid = data["changed"] & (data["ranges"] > 0)
    valid[(times >= RESUME-length) & (times < RESUME)] = False
    rows, ids = np.nonzero(valid & (times >= RESUME)[:, None])
    payload = dict(time=times, obs_time=times[rows], plus_xyz=data["anchors"][ids],
        observations=data["ranges"][rows, ids]-np.asarray(model["range_bias"])[ids],
        sigma=np.asarray(model["range_sigma"])[ids], obs_sample=rows, obs_anchor=ids,
        resume=np.array(RESUME), gap_s=np.array(length))
    if length == 10:
        # This exact original replay already passed every published metric audit.
        original = np.load(ROOT/"agent/codex/paper_baseline_check/capture_histories"/
                           (Path(name).stem+"__capture__outage1.npz"))
        assert np.array_equal(times, original["time"])
        payload.update(k_pos=original["output"][:, :3], k_vel=original["output"][:, 3:],
            preupdate_available=np.array(False), exact_previous_replay_reused=np.array(True))
    else:
        tracker = diagnose.make_capture(data, model)
        history, preupdate = np.empty((len(times), 6)), np.empty((len(times), 6))
        rotations, biases = np.empty((len(times), 3, 3)), np.empty((len(times), 6))
        last_event = None
        for k, stamp in enumerate(times):
            if k:
                tracker.predict(data["imu"][k-1], float(stamp-times[k-1]))
            preupdate[k] = np.r_[tracker.X[:3, 4], tracker.X[:3, 3]]
            events = np.flatnonzero(valid[k])
            if len(events):
                if last_event is not None and stamp-last_event > 1.0:
                    tracker.on_ranges_resumed()
                last_event = stamp
            for anchor in events:
                tracker.correct_twr(data["anchors"][anchor], data["ranges"][k, anchor], int(anchor))
            history[k] = np.r_[tracker.X[:3, 4], tracker.X[:3, 3]]
            rotations[k], biases[k] = tracker.X[:3, :3], tracker.bias
        payload.update(k_pos=history[:, :3], k_vel=history[:, 3:],
            preupdate_pos=preupdate[:, :3], preupdate_vel=preupdate[:, 3:],
            k_rotation=rotations, k_bias=biases,
            preupdate_available=np.array(True), exact_previous_replay_reused=np.array(False))
    # Previous-output states are explicitly timestamped and strictly older than each packet.
    previous = np.maximum(rows-1, 0)
    payload.update(strict_past_time=times[previous], strict_past_pos=payload["k_pos"][previous],
        strict_past_vel=payload["k_vel"][previous])
    assert np.all(payload["strict_past_time"] < payload["obs_time"])
    np.savez_compressed(target, **payload)
    return dict(sequence=name, gap_s=length, cached=False,
        reused_original_10s=bool(length == 10), packets=len(rows), samples=len(times))


def main():
    """Prepare the fixed development-domain matrix with at most four workers."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((CODE/"results/roadmap_stage1_20261104_final/protocol.json").read_text(encoding="utf-8"))
    jobs = [(name, length) for name in protocol["evaluation"] for length in LENGTHS]
    with replay_pool(len(jobs), workers=min(4, args.workers)) as pool:
        audit = list(pool.map(prepare_case, jobs))
    manifest = dict(domain="Delft development; six same-date flights, not a new domain",
        evaluation=protocol["evaluation"], lengths_s=LENGTHS, resume_s=RESUME,
        sensor_start="Unchanged static-prefix-trimmed original protocol clock",
        yaw_perturbation_deg=0, range_bias=model_value(protocol, "range_bias"),
        range_sigma=model_value(protocol, "range_sigma"),
        lever_arm="None: original native TWR antenna-as-position measurement model preserved",
        ten_second_preupdate="Not fabricated: only full updated p/v exists in the reused replay",
        strict_past="Previous sample output; never current packet posterior",
        training="Frozen original six other same-date recordings; no truth read by this script",
        audit=audit)
    (OUT/"manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


def model_value(protocol, field):
    """Expose the existing frozen calibration without fitting any new parameters."""
    return protocol["range_models"]["holdout"][field]


if __name__ == "__main__":
    main()
