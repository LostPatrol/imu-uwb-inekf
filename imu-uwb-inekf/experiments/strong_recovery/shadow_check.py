"""Numerical checks for the shared frozen shadow API, without real-data scores."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

import shadow


def example(tdoa=False, planar=False):
    """Construct a moving CV trajectory, biased K translation and raw radio packets."""
    rng = np.random.default_rng(20261112)
    time = np.arange(2400, 2801)/100.
    stamps = np.arange(25000, 28000, 5)/1000.
    anchors = np.array([[x, y, z] for x in (-4., 4.) for y in (-3., 3.) for z in (0., 3.)])
    if planar:
        anchors[:, 2] = 0.
    plus = anchors[np.arange(len(stamps)) % len(anchors)]
    minus = anchors[(np.arange(len(stamps))+3) % len(anchors)] if tdoa else None
    velocity = np.array([.8, .3, -.05])
    position = np.array([1.2, .3, 1.4])
    actual = position+(stamps-25.)[:, None]*velocity
    values, _ = shadow._radio(actual, plus, minus)
    values += rng.normal(0., .03, len(stamps))
    k_pos = position+(time-25.)[:, None]*velocity+np.array([2., -1., .1])
    k_vel = np.broadcast_to(velocity, k_pos.shape).copy()
    return dict(time=time, k_pos=k_pos, k_vel=k_vel, lever_world=None,
                obs_time=stamps, plus_xyz=plus, minus_xyz=minus, observations=values, sigma=.03, resume=25.)


def main():
    """Check domain signs, planar XY, immutability, strict past use and exact return."""
    reports = []
    for tdoa in (False, True):
        for planar in (False, True):
            values = example(tdoa, planar)
            before = values["k_pos"].copy()
            result = shadow.recover(**values)
            outside = (values["time"] < 25.) | (values["time"] >= 27.)
            assert np.array_equal(result["pos"][outside], values["k_pos"][outside])
            assert np.array_equal(result["vel"][outside], values["k_vel"][outside])
            assert np.array_equal(result["pos"][:, 2], values["k_pos"][:, 2])
            assert np.array_equal(result["vel"][:, 2], values["k_vel"][:, 2])
            assert np.array_equal(before, values["k_pos"])
            poisoned = values["observations"].copy()
            poisoned[values["obs_time"] >= 27.] = np.nan
            after = shadow.recover(**dict(values, observations=poisoned))
            assert result["diagnostics"] == after["diagnostics"]
            assert np.array_equal(result["pos"], after["pos"])
            k_poison = values["k_pos"].copy()
            current = (values["time"] >= 25.5) & (values["time"] < 25.75)
            k_poison[current] += 100.
            after = shadow.recover(**dict(values, k_pos=k_poison))
            assert result["diagnostics"]["blocks"][:3] == after["diagnostics"]["blocks"][:3]
            assert np.array_equal(result["always_c_pos"][current, :2], after["always_c_pos"][current, :2])
            radio_poison = values["observations"].copy()
            radio_poison[(values["obs_time"] >= 25.75) & (values["obs_time"] < 26.)] += 30.
            after = shadow.recover(**dict(values, observations=radio_poison))
            assert result["diagnostics"]["blocks"][:3] == after["diagnostics"]["blocks"][:3]
            assert result["diagnostics"]["blocks"][3]["fit"] == after["diagnostics"]["blocks"][3]["fit"]
            chosen = np.flatnonzero(result["use_c"])
            assert not len(chosen) or values["time"][chosen[0]] >= 26.
            reports.append(dict(tdoa=tdoa, planar=planar, selected_samples=len(chosen),
                valid_c_blocks=sum(block["fit"]["reason"] == "valid" for block in result["diagnostics"]["blocks"]),
                first_c=float(values["time"][chosen[0]]) if len(chosen) else None,
                switch_count=len(result["diagnostics"]["switches"])))
    print(json.dumps(dict(checks_passed=True, cases=reports,
        shadow_sha256=hashlib.sha256(Path(shadow.__file__).read_bytes()).hexdigest(), rules=shadow.RULES), indent=2))


if __name__ == "__main__":
    main()
