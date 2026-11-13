"""Measure the original ISAS ROS anchor-list warm-up without reading pose truth."""

import argparse
import json
from pathlib import Path

import numpy as np
from rosbags.rosbag1 import Reader

# The original EstimationInterface waits for this many anchor-list messages.
ANCHOR_MESSAGE_COUNT = 20
# The common protocol supplies this amount of calibration history before time zero.
PREFIX_SECONDS = 1.5


def window_wait(timestamps, beginning_ns, epoch_ns):
    """Find the actual first and twentieth anchor arrival after a replay boundary."""
    start = int(np.searchsorted(timestamps, beginning_ns))
    selected = timestamps[start:start + ANCHOR_MESSAGE_COUNT]
    if len(selected) != ANCHOR_MESSAGE_COUNT:
        return {"available_messages": len(selected)}
    return {
        "first_arrival_ns": int(selected[0]), "twentieth_arrival_ns": int(selected[-1]),
        "first_to_twentieth_wait_s": float((selected[-1] - selected[0]) * 1e-9),
        "first_arrival_relative_episode_s": float((selected[0] - epoch_ns) * 1e-9),
        "anchor_ready_relative_episode_s": float((selected[-1] - epoch_ns) * 1e-9),
        "remaining_anchor_wait_at_episode_zero_s": float(max(0, selected[-1] - epoch_ns) * 1e-9),
    }


def audit(streams, output):
    """Read only anchor arrival timestamps and compare their availability to prefixes."""
    records = []
    for walk in (2, 3):
        start_meta = json.loads((streams / f"isas_walk{walk}_start/meta.json").read_text(encoding="utf-8-sig"))
        bag = Path(start_meta["source_bag"])
        with Reader(bag) as reader:
            connections = [connection for connection in reader.connections if connection.topic == "/anchor_list"]
            timestamps = np.asarray([stamp for _, stamp, _ in reader.messages(connections=connections)], dtype=np.int64)
            bag_start_ns = reader.start_time
            message_definition = connections[0].msgdef.data
        record = {
            "walk": walk, "source_bag": str(bag), "anchor_messages": len(timestamps),
            "anchor_message_has_header": "Header" in message_definition.split("====")[0],
            "timestamp_source": "rosbag recorded arrival timestamp; Anchorlist has no Header",
            "first_anchor_from_bag_start_s": float((timestamps[0] - bag_start_ns) * 1e-9),
            "twentieth_anchor_from_bag_start_s": float((timestamps[19] - bag_start_ns) * 1e-9),
            "first_to_twentieth_wait_s": float((timestamps[19] - timestamps[0]) * 1e-9),
            "episodes": [],
        }
        for kind in ("start", "recovery"):
            episode = f"isas_walk{walk}_{kind}"
            meta = json.loads((streams / episode / "meta.json").read_text(encoding="utf-8-sig"))
            epoch_ns = int(meta["t_epoch_ns"])
            record["episodes"].append({
                "episode": episode, "episode_epoch_ns": epoch_ns,
                "prefix_replay": window_wait(timestamps, epoch_ns - int(PREFIX_SECONDS * 1e9), epoch_ns),
                "without_prefix_replay": window_wait(timestamps, epoch_ns, epoch_ns),
            })
        records.append(record)
    result = {
        "author_anchor_message_threshold": ANCHOR_MESSAGE_COUNT,
        "common_prefix_s": PREFIX_SECONDS,
        "ground_truth_read": False, "records": records,
        "interpretation": "Twentieth anchor arrival is only a lower bound on full ROS publication. Sensor-time interpolation and computation add their own constraints. Original startup buffers IMU before anchors, so one must not simply add the 20-message wait to the 0.1 s core first state. Prefix calibration may completely overlap anchor warm-up. No claim is made that running the original filter during the prefix gives the same initialized state as the common t=0 restart.",
        "clock_caveat": "Bag arrival time and message header time share the recorded ROS epoch but are distinct timing quantities; no hardware-clock correction is inferred.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


def main():
    """Read common stream metadata and save a separate I/O timing audit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--streams", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.streams, args.output)


if __name__ == "__main__":
    main()
