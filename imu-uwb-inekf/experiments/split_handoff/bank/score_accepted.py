"""Score exploratory first-2/5-second effects for every accepted bank correction.

This reuses saved trajectories and the shared truth scorer without replaying filters.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

# Only these matched shared policies are compared; no outcome-based policy selection.
POLICIES = ["bank_yaw", "retain_normal", "forced_yaw_only", "full_attitude", "forced_evidence_yaw"]
# These horizons are exploratory diagnostics; the common primary metric ends at 45 s.
HORIZONS_S = [2., 5.]


def shared_module():
    """Load the existing UTIL truth scorer and loader without executing its CLI."""
    directory = Path(__file__).resolve().parents[1]/"util"
    sys.path.insert(0, str(directory))
    spec = importlib.util.spec_from_file_location("bank_shared_util_scoring", directory/"run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    """Keep every accepted bank case, including worsened heading or XY trajectories."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", default="results")
    args = parser.parse_args()
    shared = shared_module()
    output = Path(__file__).parent/args.output
    output.mkdir(exist_ok=True)
    truth_cache, rows = {}, []
    for source in args.inputs:
        directory = Path(source)
        records = json.loads((directory/"records.json").read_text(encoding="utf-8"))
        accepted = {(row["name"], row["scenario"], row["yaw_offset_deg"])
                    for row in records if row["policy"] == "bank_yaw" and row["decision"]["accepted"]}
        for record in records:
            key = (record["name"], record["scenario"], record["yaw_offset_deg"])
            if key not in accepted or record["policy"] not in POLICIES:
                continue
            name = record["name"]
            if name not in truth_cache:
                truth_cache[name] = shared.load(name, duration=45.)["truth"]
            with np.load(directory/(record["key"]+".npz")) as artifact:
                after = artifact["after"]
            for horizon in HORIZONS_S:
                subset = after[after[:, 0] <= record["handoff_time"]+horizon]
                metric = shared.score_rows(truth_cache[name], subset)
                rows.append(dict(split=directory.name, name=name, scenario=record["scenario"],
                    yaw_offset_deg=record["yaw_offset_deg"], policy=record["policy"], horizon_s=horizon,
                    exploratory_accepted_subset=True, **metric))
    frame = pd.DataFrame(rows)
    frame.to_csv(output/"accepted_short_horizon.csv", index=False)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
