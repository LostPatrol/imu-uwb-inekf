"""Check the compiled physical/invariant model against the existing reference."""

from pathlib import Path
import json
import sys
import time
import numpy as np
from scipy.spatial.transform import Rotation

# Import the unchanged reference model, not any prior experiment outputs.
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments" / "roadmap_stage1"))
from invariant15 import Invariant15, coordinate_map
import kernel


def main():
    """Verify propagation, finite injection/reset, and native TDoA covariance math."""
    state = np.eye(5)
    state[:3, :3] = Rotation.from_euler("xyz", [.1, -.2, .3]).as_matrix()
    state[:3, 3] = [.2, -.1, .3]
    state[:3, 4] = [1., -.4, .8]
    physical = np.diag(np.array([.1] * 3 + [.7] * 3 + [.5] * 3 + [.01] * 3 + [.2] * 3) ** 2)
    change = coordinate_map(state)
    covariance = change @ physical @ change.T
    bias = np.array([.001, -.002, .003, .01, -.02, .03])
    reference = Invariant15(state, covariance, bias)
    sample = np.array([.03, -.02, 1.01, 1., -2., 3.])
    started = time.perf_counter()
    actual_state, actual_covariance = kernel.predict(state, covariance, bias, sample, .01)
    reference.predict(sample, .01)
    state_error = np.max(np.abs(reference.X - actual_state))
    covariance_error = np.max(np.abs(reference.P - actual_covariance))
    assert state_error < 1e-12, state_error
    assert covariance_error < 1e-12, covariance_error
    delta = np.array([.8, -.2, .3, .1, -.2, .3, -.2, .1, .3, .001, .002, -.001, .01, -.01, .02])
    candidate, candidate_bias = kernel.inject(state, bias, delta)
    reference = Invariant15(state, covariance, bias)
    reference.inject(delta)
    assert np.max(np.abs(reference.X - candidate)) < 1e-12
    assert np.max(np.abs(reference.bias - candidate_bias)) < 1e-12
    reset_error = np.max(np.abs(kernel.reset_jacobian(delta) - reference.reset_jacobian(delta)))
    assert reset_error < 1e-12, reset_error
    anchors = np.array([[0., 0., 0.], [4., 2., 3.]])
    lag = -.007
    tag = state[:3, 4] + lag * state[:3, 3] + state[:3, :3] @ kernel.LEVER
    da, db = tag - anchors[0], tag - anchors[1]
    gradient = db / np.linalg.norm(db) - da / np.linalg.norm(da)
    row = np.r_[-gradient @ kernel.skew(tag), lag * gradient, gradient, np.zeros(6)]
    numerical = np.zeros(15)
    for column in range(15):
        perturbation = np.zeros(15)
        perturbation[column] = 1e-6
        values = []
        for sign in [-1., 1.]:
            shifted, _ = kernel.inject(state, bias, sign * perturbation)
            antenna = shifted[:3, 4] + lag * shifted[:3, 3] + shifted[:3, :3] @ kernel.LEVER
            values.append(np.linalg.norm(antenna - anchors[1]) - np.linalg.norm(antenna - anchors[0]))
        numerical[column] = (values[1] - values[0]) / 2e-6
    observation_error = float(np.max(np.abs(row - numerical)))
    assert observation_error < 1e-7, observation_error
    result = dict(propagation_state_max_error=float(state_error), propagation_covariance_max_error=float(covariance_error),
                  reset_max_error=float(reset_error), native_tdoa_jacobian_max_error=observation_error,
                  first_compile_seconds=time.perf_counter() - started)
    folder = Path(__file__).parent / "verification"
    folder.mkdir(exist_ok=True)
    (folder / "kernel.json").write_text(json.dumps(result, indent=2), encoding="utf8")
    print(result)


if __name__ == "__main__":
    main()
