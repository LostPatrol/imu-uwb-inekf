"""Audit complete same-data acquisition panels without using scoring truth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def audit(output):
    """Verify requested nodes, sensor pairing and measured publication queues."""
    native = pd.read_csv(output/"native.csv")
    stress = pd.read_csv(output/"stress.csv")
    trajectory = pd.read_csv(output/"trajectory.csv")
    checks = {}
    expected_native = np.arange(2., 15.001, .5)
    expected_trajectory = np.arange(2., 15.001, 1.)
    for key, panel in (("native", native), ("stress", stress), ("trajectory", trajectory)):
        assert not panel.duplicated(["episode", "case_id", "method", "arrival_s"]).any()
        assert panel.refinement_starts.eq(1).all()
        assert panel.loc[panel.status != "exception", "nfev"].le(120).all()
        # Every model/case at an epoch consumed the identical original radio packets.
        assert panel.groupby(["episode", "arrival_s"]).packet_sha256.nunique().eq(1).all()
        if key != "stress":
            assert (panel.publish_s >= panel.arrival_s).all()
            np.testing.assert_allclose(panel.publish_s, panel.compute_start_s+panel.wall_s,
                atol=1e-12, rtol=1e-12)
            for _, part in panel.groupby(["episode", "case_id", "method"]):
                part = part.sort_values("arrival_s")
                arrivals = expected_native if key == "native" else expected_trajectory
                np.testing.assert_allclose(part.arrival_s, arrivals, atol=1e-12, rtol=0)
                assert np.all(part.compute_start_s.to_numpy()[1:] >= part.publish_s.to_numpy()[:-1]-1e-12)
        else:
            assert panel.groupby(["episode", "case_id", "method"]).arrival_s.apply(
                lambda values:set(values) == {2., 10.}).all()
        checks[key] = dict(rows=len(panel), episodes=panel.episode.nunique(),
            solver_limits=int(panel.status.eq("solver_limit").sum()),
            numerical_exceptions=int(panel.status.eq("exception").sum()),
            finite_positions=int(np.isfinite(panel[["x", "y", "z"]]).all(axis=1).sum()),
            wall_median_s=float(panel.wall_s.median()), wall_p95_s=float(panel.wall_s.quantile(.95)))
    for panel in (stress, trajectory):
        # All 96 requested perturbations plus one reused native snapshot are present.
        assert panel.groupby("episode").case_id.nunique().eq(97).all()
    assert len(trajectory) == trajectory.episode.nunique()*97*14
    assert len(stress) == stress.episode.nunique()*97*2*2
    assert len(native) == native.episode.nunique()*27*2
    checks["all_requested_nodes_retained"] = True
    checks["identical_packets_across_paired_cases_and_methods"] = True
    checks["queues_include_compute_backlog_and_preserve_state_epoch"] = True
    return checks


def main():
    """Write a reproducible structural audit separately from localization scores."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checks = audit(args.output)
    (args.output/"audit.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
