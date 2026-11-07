"""Regression checks for the compute-only author acceleration."""

import hashlib
import importlib.util
import os
from pathlib import Path
import sys
import traceback

# Match the benchmark's single-core MKL guard before importing NumPy/Numba.
os.environ["MKL_THREADING_LAYER"] = "SEQUENTIAL"

import numpy as np
from scipy.spatial.transform import Rotation


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPOSITORY_ROOT.parent
BASE_EXPERIMENTS = WORKSPACE_ROOT / "imu-uwb-inekf" / "experiments"
sys.path.insert(0, str(BASE_EXPERIMENTS))
import benchmark_author_original_vs_proposed as benchmark_base  # noqa: E402

sys.path.insert(
    0,
    str(REPOSITORY_ROOT / "scripts" / "estimation"),
)
from accelerated_replay import (  # noqa: E402
    execute_accelerated_replay,
    prepare_accelerated_replay,
)
from pipeline import Dataset  # noqa: E402


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _synthetic_dataset():
    t_imu = np.arange(0.0, 0.31, 0.01)
    # Deliberately put a UWB sample at the first event.  Some real logs begin
    # this way, and the scheduler must not silently skip that correction.
    t_uwb = np.arange(0.0, 0.305, 0.01)
    anchors = np.array(
        [
            [-2.0, -2.0, 2.0],
            [2.0, -2.0, 2.0],
            [2.0, 2.0, 2.0],
            [-2.0, 2.0, 2.0],
        ]
    )
    imu = np.zeros((len(t_imu), 6))
    imu[:, 2] = 1.0
    imu[:, 3] = 0.02
    tag_offset = np.array([-0.01245, 0.00127, 0.0908])
    distance_a = np.linalg.norm(tag_offset - anchors[0])
    distance_b = np.linalg.norm(tag_offset - anchors[1])
    measurement = distance_b - distance_a
    uwb = np.column_stack(
        (
            np.zeros(len(t_uwb)),
            np.ones(len(t_uwb)),
            measurement + 0.01 * np.sin(20.0 * t_uwb),
        )
    )
    return Dataset(
        name="synthetic_author_acceleration",
        anchors=anchors,
        t_imu=t_imu,
        imu=imu,
        t_uwb=t_uwb,
        uwb=uwb,
        t_gt=t_imu.copy(),
        p_gt=np.zeros((len(t_imu), 3)),
        q_gt_xyzw=np.tile(
            Rotation.identity().as_quat(),
            (len(t_imu), 1),
        ),
    )


def _reference_history(dataset, plan, author_module):
    adapter = benchmark_base.make_author_adapter(
        dataset,
        author_module,
        yaw_degrees=0.0,
        tdoa_variance=0.13,
        collect_diagnostics=True,
    )
    rotations = []
    velocities = []
    positions = []
    covariances = []
    biases = []
    nis = []
    rejected = []
    last_imu_index = 0

    for measurement_index in range(
        int(plan.uwb_start_at_event[0]),
        int(plan.uwb_stop_at_event[0]),
    ):
        rejected.append(
            adapter.correct(
                dataset.uwb[measurement_index],
                dataset.anchors,
            )
        )
        nis.append(adapter.last_nis)
    rotations.append(adapter.X[:3, :3].copy())
    velocities.append(adapter.X[:3, 3].copy())
    positions.append(adapter.X[:3, 4].copy())
    covariances.append(adapter.P.copy())
    biases.append(adapter.bias.copy())

    for event_index in range(1, len(plan.events)):
        event_time = float(plan.events[event_index])
        imu_index = int(plan.imu_index_at_event[event_index])
        if imu_index >= 0 and imu_index > last_imu_index:
            dt = event_time - float(dataset.t_imu[last_imu_index])
            adapter.predict(dataset.imu[last_imu_index], dt)
            last_imu_index = imu_index
        for measurement_index in range(
            int(plan.uwb_start_at_event[event_index]),
            int(plan.uwb_stop_at_event[event_index]),
        ):
            rejected.append(
                adapter.correct(
                    dataset.uwb[measurement_index],
                    dataset.anchors,
                )
            )
            nis.append(adapter.last_nis)
        rotations.append(adapter.X[:3, :3].copy())
        velocities.append(adapter.X[:3, 3].copy())
        positions.append(adapter.X[:3, 4].copy())
        covariances.append(adapter.P.copy())
        biases.append(adapter.bias.copy())

    return {
        "rotations": np.asarray(rotations),
        "velocities": np.asarray(velocities),
        "positions": np.asarray(positions),
        "covariances": np.asarray(covariances),
        "biases": np.asarray(biases),
        "nis": np.asarray(nis),
        "rejected": np.asarray(rejected, dtype=bool),
    }


def test_copied_author_source_is_byte_identical():
    backup = (
        WORKSPACE_ROOT
        / "imu-uwb-inekf - backup"
        / "scripts"
        / "estimation"
        / "inekf.py"
    )
    copied = (
        REPOSITORY_ROOT
        / "scripts"
        / "estimation"
        / "inekf.py"
    )
    assert _sha256(backup) == _sha256(copied)


def test_accelerated_replay_matches_author_equations():
    dataset = _synthetic_dataset()
    plan = benchmark_base.build_event_plan(dataset)
    author_module = benchmark_base.load_author_module()
    acc_module = _load_module(
        "test_acc_copied_inekf",
        REPOSITORY_ROOT / "scripts" / "estimation" / "inekf.py",
    )
    reference = _reference_history(dataset, plan, author_module)
    acc_adapter = benchmark_base.make_author_adapter(
        dataset,
        acc_module,
        yaw_degrees=0.0,
        tdoa_variance=0.13,
    )
    fixed_plan = (
        plan.events,
        plan.imu_index_at_event,
        plan.uwb_start_at_event,
        plan.uwb_stop_at_event,
    )
    prepared = prepare_accelerated_replay(
        dataset,
        acc_adapter.estimator,
        fixed_plan,
    )
    output = execute_accelerated_replay(prepared, record_history=True)
    (
        states,
        rotations,
        covariances,
        biases,
        _,
        nis,
        rejected,
        predict_count,
        uwb_count,
        rejected_count,
    ) = output

    np.testing.assert_allclose(
        rotations,
        reference["rotations"],
        rtol=2e-10,
        atol=2e-12,
    )
    np.testing.assert_allclose(
        states[:, 3:6],
        reference["velocities"],
        rtol=2e-10,
        atol=2e-12,
    )
    np.testing.assert_allclose(
        states[:, 6:9],
        reference["positions"],
        rtol=2e-10,
        atol=2e-12,
    )
    np.testing.assert_allclose(
        covariances,
        reference["covariances"],
        rtol=2e-9,
        atol=2e-12,
    )
    np.testing.assert_allclose(
        biases,
        reference["biases"],
        rtol=2e-10,
        atol=2e-12,
    )
    np.testing.assert_allclose(
        nis,
        reference["nis"],
        rtol=2e-10,
        atol=2e-12,
    )
    np.testing.assert_array_equal(rejected, reference["rejected"])
    assert predict_count == len(dataset.t_imu) - 1
    assert uwb_count == len(dataset.t_uwb)
    assert rejected_count == int(np.sum(reference["rejected"]))

    timing_output = execute_accelerated_replay(
        prepared,
        record_history=False,
    )
    assert timing_output[0].shape == (0, 9)
    assert timing_output[2].shape == (0, 15, 15)
    assert timing_output[5].size == 0
    assert timing_output[-3] == len(dataset.t_imu) - 1
    assert timing_output[-2] == len(dataset.t_uwb)
    assert timing_output[-1] == rejected_count

    compact_output = execute_accelerated_replay(
        prepared,
        record_history=True,
        record_covariances=False,
    )
    assert compact_output[2].shape == (0, 15, 15)
    np.testing.assert_allclose(
        compact_output[0],
        states,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        compact_output[1],
        rotations,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        compact_output[3],
        biases,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        compact_output[5],
        nis,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_array_equal(compact_output[6], rejected)


if __name__ == "__main__":
    failures = 0
    tests = sorted(
        (name, value)
        for name, value in globals().items()
        if name.startswith("test_") and callable(value)
    )
    for name, test in tests:
        try:
            test()
            print(f"PASS {name}")
        except Exception:
            failures += 1
            print(f"FAIL {name}")
            traceback.print_exc()
    print(f"{len(tests) - failures}/{len(tests)} tests passed")
    raise SystemExit(1 if failures else 0)
