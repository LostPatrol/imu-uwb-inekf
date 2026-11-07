"""Check terminal recovery scoring and physical handoff covariance blocks."""

import numpy as np

from run import recovery_metrics, make_methods, initial_state, DATASET
from twr_experiment import load_real_log
from invariant15 import handoff_covariance, inverse_coordinate_map


def run_checks():
    """Check decisive score boundaries and factory propagation on a real log."""
    times = np.arange(0.0, 12.1, 0.1)
    errors = np.zeros_like(times) + 0.2
    errors[times < 2.0] = 0.8
    errors[70] = 0.8
    metrics = recovery_metrics(times, errors)
    assert abs(metrics["first_dwell_s"] - 2.0) < 1e-12
    assert not metrics["success"]
    errors[70] = 0.2
    metrics = recovery_metrics(times, errors, origin=5.0)
    assert metrics["success"] and abs(metrics["stable_recovery_s"]) < 1e-12
    errors[:] = 0.8
    assert not recovery_metrics(times, errors)["success"]
    log = load_real_log(DATASET, "2021-02-12_09_41_16_twr_triangle.csv", sensor_delay_s=0.2)
    _, state, physical, invariant, bias = initial_state(log, np.array([6., -6., 3.]), 180)
    model = dict(range_bias=np.linspace(-0.2, 0.1, 8), range_sigma=np.linspace(0.1, 0.3, 8))
    methods = make_methods(state, physical, invariant, bias, model)
    deep = methods["deep"]
    for tracker in (methods["inekf15"], methods["eskf15"], deep.tracker):
        assert tracker.gate_warmup_updates == 256 and tracker.huber_threshold == 2.0
        np.testing.assert_allclose(tracker.range_bias, model["range_bias"])
    for core in (methods["inekf15"], deep.tracker, deep.capture.imu_core):
        np.testing.assert_allclose(core.physical_covariance(), physical, atol=1e-12)
    callback_covariance = handoff_covariance(deep.capture)
    inverse = inverse_coordinate_map(deep.capture.X)
    handoff_physical = inverse @ callback_covariance @ inverse.T
    np.testing.assert_allclose(handoff_physical[3:6, 3:6], deep.capture.lifted.P[6:9, 6:9] + 0.75**2 * np.eye(3), atol=1e-12)
    np.testing.assert_allclose(handoff_physical[6:9, 6:9], deep.capture.lifted.P[3:6, 3:6] + 0.75**2 * np.eye(3), atol=1e-12)
    assert np.min(np.linalg.eigvalsh(handoff_physical)) > 0
    print("Protocol: terminal scoring, identical physical priors/policies and handoff blocks passed")


if __name__ == "__main__":
    run_checks()
