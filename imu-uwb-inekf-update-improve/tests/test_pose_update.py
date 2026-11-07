"""Focused verification for the task-04 pose propagation update."""

import importlib.util
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.linalg import expm


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPOSITORY_ROOT.parent
AUTHOR_REPOSITORY = WORKSPACE_ROOT / "imu-uwb-inekf - backup"
AUTHOR_COMMIT = "f3b664e0e8c86b62c6ed2ea2a112261810228e1c"
IMPROVED_INEKF = REPOSITORY_ROOT / "scripts" / "estimation" / "inekf.py"
AUTHOR_INEKF = AUTHOR_REPOSITORY / "scripts" / "estimation" / "inekf.py"


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


IMPROVED = _load_module("task04_improved_inekf", IMPROVED_INEKF)
AUTHOR = _load_module("task04_author_inekf", AUTHOR_INEKF)


def _reference_motion_jacobians(phi):
    """Compute J_l and Q_l from one augmented matrix exponential."""
    generator = np.zeros((9, 9))
    generator[0:3, 0:3] = IMPROVED.SO3_wedge(phi)
    generator[0:3, 3:6] = np.eye(3)
    generator[3:6, 6:9] = np.eye(3)
    motion = expm(generator)
    return motion[0:3, 3:6], motion[0:3, 6:9]


def test_motion_jacobians():
    cases = (
        np.zeros(3),
        np.array([1e-10, -2e-10, 3e-10]),
        np.array([2e-5, -4e-5, 1e-5]),
        np.array([0.2, -0.3, 0.4]),
        np.array([2.2, -0.7, 0.9]),
    )
    for phi in cases:
        actual_j, actual_q = IMPROVED.SO3_motion_jacobians(phi)
        expected_j, expected_q = _reference_motion_jacobians(phi)
        np.testing.assert_allclose(actual_j, expected_j, rtol=2e-12, atol=2e-12)
        np.testing.assert_allclose(actual_q, expected_q, rtol=2e-12, atol=2e-12)


def _make_filter(module, rotation, velocity, position):
    state = np.eye(5)
    state[0:3, 0:3] = rotation
    state[0:3, 3] = velocity
    state[0:3, 4] = position
    covariance = np.eye(15) * 1e-6
    return module.InEKF(state, covariance, np.zeros(6))


def test_exact_constant_imu_step():
    rotation = IMPROVED.SO3_exp(np.array([0.2, -0.1, 0.3]))
    velocity = np.array([0.4, -0.3, 0.2])
    position = np.array([1.0, -2.0, 10.0])
    acceleration = np.array([3.5, -1.7, 2.2])
    gyro_degrees = np.array([250.0, -120.0, 80.0])
    dt = 0.08
    measurement = np.concatenate(
        (acceleration / IMPROVED.GRAVITY_MAGNITUDE, gyro_degrees)
    )

    omega = gyro_degrees * IMPROVED.DEG_TO_RAD
    phi = omega * dt
    jacobian, second_jacobian = _reference_motion_jacobians(phi)
    gravity = np.array([0.0, 0.0, -IMPROVED.GRAVITY_MAGNITUDE])
    expected_rotation = rotation @ IMPROVED.SO3_exp(phi)
    expected_velocity = (
        velocity + (rotation @ jacobian @ acceleration + gravity) * dt
    )
    expected_position = (
        position
        + velocity * dt
        + rotation @ second_jacobian @ acceleration * dt**2
        + 0.5 * gravity * dt**2
    )

    improved_filter = _make_filter(
        IMPROVED, rotation.copy(), velocity.copy(), position.copy()
    )
    author_filter = _make_filter(
        AUTHOR, rotation.copy(), velocity.copy(), position.copy()
    )
    improved_state, _, _ = improved_filter.predict(measurement, dt)
    author_state, _, _ = author_filter.predict(measurement, dt)

    np.testing.assert_allclose(
        improved_state[0:3, 0:3], expected_rotation, rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        improved_state[0:3, 3], expected_velocity, rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        improved_state[0:3, 4], expected_position, rtol=1e-12, atol=1e-12
    )

    author_velocity_error = np.linalg.norm(
        author_state[0:3, 3] - expected_velocity
    )
    author_position_error = np.linalg.norm(
        author_state[0:3, 4] - expected_position
    )
    assert author_velocity_error > 1e-2
    assert author_position_error > 1e-4


def test_zero_rotation_matches_author_formula():
    rotation = IMPROVED.SO3_exp(np.array([0.1, 0.2, -0.1]))
    velocity = np.array([0.2, 0.1, -0.1])
    position = np.array([0.5, -0.7, 2.0])
    measurement = np.array([0.2, -0.1, 1.1, 0.0, 0.0, 0.0])
    dt = 0.01

    improved_filter = _make_filter(
        IMPROVED, rotation.copy(), velocity.copy(), position.copy()
    )
    author_filter = _make_filter(
        AUTHOR, rotation.copy(), velocity.copy(), position.copy()
    )
    improved_state, _, _ = improved_filter.predict(measurement, dt)
    author_state, _, _ = author_filter.predict(measurement, dt)
    np.testing.assert_allclose(improved_state, author_state, rtol=0.0, atol=1e-14)


def test_original_tracked_files_unchanged_except_pose_source():
    commit = subprocess.run(
        ["git", "-C", str(AUTHOR_REPOSITORY), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()
    assert commit == AUTHOR_COMMIT

    tracked_status = subprocess.run(
        [
            "git",
            "-C",
            str(AUTHOR_REPOSITORY),
            "status",
            "--short",
            "--untracked-files=no",
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()
    assert not tracked_status, tracked_status

    tracked_output = subprocess.run(
        ["git", "-C", str(AUTHOR_REPOSITORY), "ls-files", "-z"],
        check=True,
        capture_output=True,
    ).stdout
    tracked = [
        path.decode("utf-8")
        for path in tracked_output.split(b"\0")
        if path
    ]
    allowed_change = "scripts/estimation/inekf.py"
    for relative_path in tracked:
        if relative_path == allowed_change:
            continue
        author_bytes = (AUTHOR_REPOSITORY / relative_path).read_bytes()
        target_bytes = (REPOSITORY_ROOT / relative_path).read_bytes()
        assert target_bytes == author_bytes, relative_path


def main():
    tests = (
        test_motion_jacobians,
        test_exact_constant_imu_step,
        test_zero_rotation_matches_author_formula,
        test_original_tracked_files_unchanged_except_pose_source,
    )
    failures = []
    for test in tests:
        try:
            test()
            print(f"PASS {test.__name__}")
        except Exception as exc:
            failures.append((test.__name__, exc))
            print(f"FAIL {test.__name__}: {exc}", file=sys.stderr)
    if failures:
        raise SystemExit(1)
    print(f"PASS {len(tests)}/{len(tests)} task-04 tests")


if __name__ == "__main__":
    main()
