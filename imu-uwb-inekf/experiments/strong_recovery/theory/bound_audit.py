"""Independent randomized containment/contraction audit of frozen radio bounds.

Only synthetic physics and mathematical properties are tested; no dataset scores.
"""

from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path

import numpy as np

# Fixed reproducible property sample count, independent of real-data outcomes.
SEED = 20261112
CASES_PER_DOMAIN = 1000


def load_bounds():
    """Load the existing module read-only without changing its constants or rules."""
    path = Path(__file__).resolve().parents[1]/"radio_bounds.py"
    specification = importlib.util.spec_from_file_location("audited_radio_bounds", path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def unit(rng, count):
    """Draw ordinary three-dimensional unit directions for synthetic geometry."""
    values = rng.normal(size=(count, 3))
    return values/np.linalg.norm(values, axis=1)[:, None]


def property_cases(module, rng, tdoa):
    """Check sequential contraction when all body/lever/noise bounds truly hold."""
    worst_relative_excess, active, max_steps = 0., 0, 0
    for index in range(CASES_PER_DOMAIN):
        count, now = 5, 1.5
        ages = np.sort(rng.uniform(0., module.MAX_AGE_SECONDS, count))[::-1]
        truth = rng.normal(size=3)*5.
        velocity = unit(rng, 1)[0]*rng.uniform(0., module.BODY_SPEED_BOUND)
        radius = rng.uniform(0., .3)
        levers = unit(rng, count)*rng.uniform(0., radius, (count, 1))
        tag = truth-ages[:, None]*velocity+levers
        plus = rng.normal(size=(count, 3))*8.
        minus = rng.normal(size=(count, 3))*8. if tdoa else None
        if index % 4 == 0:
            plus[:, 2] = 0.
            if minus is not None:
                minus[:, 2] = 0.
        measured = np.linalg.norm(tag-plus, axis=1)
        if minus is not None:
            measured -= np.linalg.norm(tag-minus, axis=1)
        sigma = rng.uniform(.01, .3, count)
        # TWR permits arbitrarily large positive bias under its one-sided bound.
        noise = rng.uniform(-module.SIGMA_MARGIN, module.SIGMA_MARGIN, count)*sigma
        if not tdoa and index % 3 == 0:
            noise += rng.uniform(0., 100., count)*sigma
        measured += noise
        original = (truth+rng.normal(size=3)*20.)[None, :]
        before = original.copy()
        result = module.bound_xy(np.array([now]), original, radius, now-ages,
                                 plus, minus, measured, sigma, 0.)
        squared_before = float(np.sum((before[0, :2]-truth[:2])**2))
        squared_after = float(np.sum((result["pos"][0, :2]-truth[:2])**2))
        excess = (squared_after-squared_before)/max(1., squared_before)
        worst_relative_excess = max(worst_relative_excess, excess)
        assert excess <= 1e-10, (tdoa, index, squared_before, squared_after)
        assert np.array_equal(original, before)
        assert np.array_equal(result["pos"][:, 2], before[:, 2])
        active += int(result["active"][0])
        max_steps = max(max_steps, result["diagnostics"]["projection_steps"])
    return dict(cases=CASES_PER_DOMAIN, active_cases=active, max_projection_steps=max_steps,
                worst_relative_squared_error_increase=worst_relative_excess)


def height_elimination(rng):
    """Independently verify the analytic supremum over unbounded vertical position."""
    worst_supremum_excess, worst_stationary_error = 0., 0.
    for _ in range(CASES_PER_DOMAIN):
        radial_far, radial_near = rng.uniform(0., 30., 2)
        vertical = rng.uniform(-20., 20.)
        supremum = np.hypot(max(radial_far-radial_near, 0.), vertical)
        heights = np.r_[rng.normal(size=20)*100., -1e6, 1e6]
        values = np.hypot(radial_far, heights-vertical)-np.hypot(radial_near, heights)
        worst_supremum_excess = max(worst_supremum_excess, float(np.max(values)-supremum))
        assert np.max(values)-supremum <= 1e-8
        if radial_far-radial_near > 1e-5:
            stationary = -radial_near*vertical/(radial_far-radial_near)
            value = np.hypot(radial_far, stationary-vertical)-np.hypot(radial_near, stationary)
            error = abs(value-supremum)
            worst_stationary_error = max(worst_stationary_error, float(error))
            assert error <= 1e-8
    return dict(cases=CASES_PER_DOMAIN, worst_sampled_supremum_excess=worst_supremum_excess,
                worst_attained_supremum_error=worst_stationary_error)


def edge_cases(module):
    """Check skip boundaries, strictly later radio, exact return and an unsafe-bias counterexample."""
    near, far = np.array([0., 0., 0.]), np.array([4., 0., 3.])
    sigma = .01
    # d equals the anchor vertical separation: skipping is conservative, not an XY impossibility theorem.
    boundary = module.bound_xy([1.], np.array([[5., 5., 2.]]), 0., [1.], far[None], near[None],
                               [3.+module.SIGMA_MARGIN*sigma], sigma, 0.)
    assert not boundary["active"].any()
    k = np.array([[20., 0., 2.], [20., 0., 2.], [20., 0., 2.]])
    base = module.bound_xy([1.5, 2., 2.1], k, 0., [1.4], near[None], None, [2.], sigma, 0.)
    future = module.bound_xy([1.5, 2., 2.1], k, 0., [1.4, 1.6], np.vstack((near, near)),
                            None, [2., .01], sigma, 0.)
    assert np.array_equal(base["pos"], future["pos"])
    assert np.array_equal(base["pos"][1:], k[1:])
    # A large directed TDoA bias can choose the wrong branch and move an exact K truth.
    truth = np.array([[3., 0., 0.]])
    corrupted = module.bound_xy([1.], truth, 0., [1.], np.array([[4., 0., 0.]]), near[None],
                                [2.], sigma, 0.)
    false_shift = float(np.linalg.norm(corrupted["pos"]-truth))
    assert false_shift > 0.
    return dict(vertical_boundary_skipped=True, later_packet_cannot_change_earlier_output=True,
                exact_k_return=True, constructed_biased_tdoa_moves_correct_k_m=false_shift)


def main():
    """Write the single reproducible independent audit artifact beside this script."""
    module, rng = load_bounds(), np.random.default_rng(SEED)
    result = dict(seed=SEED, no_real_data_scores=True, all_checks_passed=True,
                  module_sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
                  twr=property_cases(module, rng, False), tdoa=property_cases(module, rng, True),
                  height=height_elimination(rng), edges=edge_cases(module))
    destination = Path(__file__).with_suffix(".json")
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
