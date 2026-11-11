"""Add a gyro-compensated gravity baseline without repeating the cube rotation search."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cases import make_case
from run_checks import evaluate, summarize


def run(source):
    """Evaluate one stronger gravity seed on every previously frozen condition."""
    original = pd.read_csv(source / "metrics.csv")
    assert "local_gravity" not in original.method.unique()
    protocol = json.loads((source / "protocol.json").read_text(encoding="utf-8"))
    records = []
    for seed in protocol["seeds"]:
        for motion in protocol["motions"]:
            for geometry in protocol["geometries"]:
                for tilt in protocol["tilts"]:
                    case = make_case(seed, motion, tilt, geometry)
                    records.extend(evaluate(case, include_multi=False, only_local_gravity=True))
        print(f"Completed local-gravity supplement seed {seed}", flush=True)
    frame = pd.concat((original, pd.DataFrame(records)), ignore_index=True)
    best = frame.groupby(["seed", "motion", "geometry", "tilt_deg"]).cost.transform("min")
    frame["cost_miss"] = frame.cost-best > np.maximum(.001, 1e-5*np.abs(best))
    original.to_csv(source / "base_metrics.csv", index=False)
    metadata = dict(base_metrics_sha256=hashlib.sha256((source / "base_metrics.csv").read_bytes()).hexdigest(),
        added_methods=["local_gravity"], added_fits=len(records),
        local_gravity_seed="Mean Q(t) f(t) aligned to world up; no static interval or truth",
        same_original_observations=True, local_gravity_uses_same_common_radio_path=True)
    frame.to_csv(source / "metrics.csv", index=False)
    summarize(frame).to_csv(source / "summary.csv", index=False)
    (source / "local_gravity_supplement.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(frame.groupby(["motion", "method"]).agg(cases=("acquired", "size"), acquired=("acquired", "sum"),
        rotation_median_deg=("rotation_error_deg", "median"), cost_misses=("cost_miss", "sum")).to_string())


def main():
    """Select one completed physical panel to supplement with the stronger baseline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args()
    run(args.source)


if __name__ == "__main__":
    main()
