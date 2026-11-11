"""Isolate downstream architecture using a shared sensor-only handoff on six fixed files."""

import argparse
import json
from pathlib import Path
import sys

# Avoid writing import caches outside this experiment's allowed directories.
sys.dont_write_bytecode = True
import run as ours
import numpy as np
import pandas as pd

# These six files were specified before reviewing any holdout accuracy scores.
PROTOCOL = json.loads((ours.HERE / "protocol.json").read_text(encoding="utf8"))


def main():
    """Generate every requested shared-handoff author continuation before scoring."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--util-output", type=Path, default=ours.UTIL / "holdout")
    parser.add_argument("--output", default="conditional")
    args = parser.parse_args()
    output = ours.HERE / args.output
    output.mkdir(parents=True, exist_ok=True)
    available = json.loads((args.util_output / "records.json").read_text(encoding="utf8"))
    selected = [r for r in available if r["name"] in PROTOCOL["conditional_fixed_recordings"] and r["policy"] == "retain_normal"]
    if len(selected) != 72:
        raise ValueError(f"Expected all 72 predefined handoffs, got {len(selected)}; finish UTIL generation first")
    records, data_cache = [], {}
    source_before = ours.digest(ours.AUTHOR)
    for record in selected:
        name = record["name"]
        if name not in data_cache:
            data_cache[name] = ours.util.load(name, duration=45.)
            data_cache[name].pop("truth")
        data = data_cache[name]
        artifact = np.load(args.util_output / (record["key"] + ".npz"))
        state, physical, bias = (artifact[k] for k in ("handoff", "handoff_physical_covariance", "bias"))
        change = ours.util.kernel.mapping(state)
        covariance = change @ physical @ change.T
        key = record["key"] + "__author_conditional"
        rows, info = ours.replay(data, state, covariance, bias, record["handoff_time"], data["times"][-1])
        np.savez_compressed(output / (key + ".npz"), rows=rows, handoff=state, physical_covariance=physical, bias=bias)
        records.append(dict(key=key, source_key=record["key"], name=name, scenario=record["scenario"],
                            yaw_offset_deg=record["yaw_offset_deg"], origin=record["origin"],
                            handoff_time=record["handoff_time"], policy="author_conditional_retain", **info))
        (output / "records.json").write_text(json.dumps(records, indent=2), encoding="utf8")
        print(json.dumps(dict(key=key, **info)), flush=True)
    metrics, truths = [], {}
    for record in records:
        if record["name"] not in truths:
            truths[record["name"]] = ours.util.load(record["name"], duration=45.)["truth"]
        rows = np.load(output / (record["key"] + ".npz"))["rows"]
        metric = dict(record)
        metric.update({"post_" + k: v for k, v in ours.util.score_rows(truths[record["name"]], rows).items()})
        metrics.append(metric)
    pd.DataFrame(metrics).to_csv(output / "metrics.csv", index=False)
    provenance = dict(source_sha256_before=source_before, source_sha256_after=ours.digest(ours.AUTHOR),
                      source_modified=source_before != ours.digest(ours.AUTHOR), records=len(records),
                      conditional_only=True, source_util_output=str(args.util_output))
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf8")


if __name__ == "__main__":
    main()
