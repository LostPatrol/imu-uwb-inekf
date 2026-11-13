"""Independent mathematical and implementation checks for the acquisition arm."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize._numdiff import approx_derivative
from scipy.spatial.transform import Rotation

from solver import Window, StartPrior, init, shape, solve, start_prior_terms


def verify():
    """Check physical-start derivatives, circularity and the exact published main."""
    case = shape.sim.physical_case(20261113)
    displacement, sensitivity, qend = shape.measured_physics(case, .1, 0., 0.)
    tau = case["event_times"]-shape.sim.DURATION
    values = (tau, displacement, case["anchors"], case["ranges"],
        np.full(len(tau), shape.sim.RANGE_SIGMA))
    window = Window(values, sensitivity, tau[0], displacement[0], sensitivity[0],
        case["q_true"][0], qend)
    x = np.r_[case["position"], case["velocity"], .7, [.03, -.02, .01]]
    prior = StartPrior(np.array([1., 2., 3.]), Rotation.from_euler("ZYX", [.1, .2, -.3]).as_matrix(),
        .1, np.deg2rad(5.))
    analytic = start_prior_terms(x, window, prior)[1]
    numeric = approx_derivative(lambda point:start_prior_terms(point, window, prior)[0], x)
    derivative_error = float(np.max(np.abs(analytic-numeric)))
    np.testing.assert_allclose(analytic, numeric, atol=2e-6, rtol=2e-6)
    shifted = x.copy()
    shifted[6] += 2*np.pi
    circle_error = float(np.max(np.abs(start_prior_terms(x, window, prior)[0]-
        start_prior_terms(shifted, window, prior)[0])))
    assert circle_error < 1e-10
    native = solve(window, "profile1")
    old = init.solve_window(*values, method="profile")
    native_error = float(np.max(np.abs(native["x"]-old.parameters)))
    assert native_error < 1e-8
    # Import the actual public main through its existing dependency wiring.
    published = shape.BASE.parent/"published_window_comparison"
    sys.path.insert(0, str(published))
    spec = importlib.util.spec_from_file_location("acquisition_published_main", published/"profile_main.py")
    main = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(main)
    ids = np.arange(len(tau))
    calibration = dict(static=np.zeros(len(tau)), sigma=values[-1])
    reference_x, reference_fit = main.profile_fit((*values[:4], ids), sensitivity, calibration)
    physical = solve(window, "body_bias_prior")
    main_error = float(np.max(np.abs(reference_x-physical["x"])))
    assert main_error < 1e-8
    assert reference_fit["cv_fits"] == 2 and reference_fit["raw_fits"] == 1
    assert physical["refinement_starts"] == 1
    return dict(start_prior_jacobian_max_abs=derivative_error,
        full_turn_prior_residual_max_abs=circle_error,
        paper_profile_native_parameters_max_abs=native_error,
        published_profile_main_parameters_max_abs=main_error,
        published_cv_fits=reference_fit["cv_fits"], published_raw_fits=reference_fit["raw_fits"],
        native_profile_s= native["fit_elapsed_s"], body_bias_s=physical["fit_elapsed_s"],
        body_bias_nfev=physical["nfev"], physical_model_checks=shape.verify(),
        stream_geometry_checks=verify_stream_geometry())


def verify_stream_geometry():
    """Check asynchronous held integration against a separate closed form."""
    spec = importlib.util.spec_from_file_location("profile_stream_replay", Path(__file__).with_name("run.py"))
    replay = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(replay)
    import pandas as pd
    times = np.array([0., 1., 2.])
    acceleration = np.array([.3, -.2, .1])
    rotations = np.tile(np.eye(3), (3, 1, 1))
    samples = np.tile(np.r_[acceleration-replay.GRAVITY, np.zeros(3)], (3, 1))
    events = np.linspace(.01, 2., 31)
    data = dict(time=times, samples=samples, rotations=rotations,
        displacement=.5*times[:, None]**2*acceleration,
        velocity=times[:, None]*acceleration,
        bias_position=.5*times[:, None, None]**2*rotations,
        bias_velocity=times[:, None, None]*rotations, gyro_bias=np.zeros(3),
        accel_bias=np.zeros(3), meta=dict(lever_m=[.1, .2, .3]),
        radio=pd.DataFrame(dict(t=events, anchor_id=np.zeros(len(events), dtype=int),
            range_m=np.ones(len(events)), sigma_m=np.full(len(events), .1))),
        anchors=pd.DataFrame(dict(anchor_id=[0], x=[3.], y=[2.], z=[1.])).set_index("anchor_id"))
    window, _, _, _ = replay.make_window(data, 2., 2.)
    expected = .5*(events-2.)[:, None]**2*acceleration+np.array([.1, .2, .3])
    error = float(np.max(np.abs(window.values[1]-expected)))
    sensitivity_error = float(np.max(np.abs(window.sensitivity-
        .5*(events-2.)[:, None, None]**2*np.eye(3))))
    # No tag lever belongs in the body-origin position prior at t=0.
    start_error = float(np.max(np.abs(window.start_displacement-2.*acceleration)))
    assert max(error, sensitivity_error, start_error) < 1e-12
    return dict(asynchronous_tag_shape_max_abs_m=error,
        held_bias_integral_max_abs=sensitivity_error, body_start_shape_max_abs_m=start_error)


def main():
    """Persist verification independently of experiment error outcomes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
