"""Measure warm, sequential first-solve cost on one common recording at a time."""

import argparse
from importlib.util import module_from_spec, spec_from_file_location
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

# Numerical workers use one thread for this separate workstation timing measurement.
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
import pandas as pd

from score import DEFAULT_DATA, HERE, SCRATCH

# The two previously validated implementations are imported directly.
sys.path.insert(0, str(HERE/"profile"))
import run as window
sys.path.insert(0, str(HERE/"jiang"))
from factor_graph import BatchGraph, GraphConfig

# Load the cold native Pfeiffer replay under a distinct name from the window driver.
_pfeiffer_spec = spec_from_file_location("pfeiffer_timing", HERE/"pfeiffer/run.py")
pfeiffer = module_from_spec(_pfeiffer_spec)
_pfeiffer_spec.loader.exec_module(pfeiffer)


def timed_graph(folder):
    """Measure the same ten-second graph, including loading and preintegration."""
    before = time.perf_counter()
    meta = json.loads((folder/"meta.json").read_text(encoding="utf-8"))
    im = pd.read_csv(folder/"imu.csv").query("0 <= t <= 10")
    radio = pd.read_csv(folder/"radio.csv").query("0 <= t <= 10")
    anchors = pd.read_csv(folder/"anchors.csv").set_index("anchor_id")
    nominal = np.r_[meta["gyro_bias"], meta["accel_bias"]]
    graph = BatchGraph(im.t.to_numpy(), im[["ax", "ay", "az"]].to_numpy(),
        im[["gx", "gy", "gz"]].to_numpy(), radio.t.to_numpy(),
        anchors.loc[radio.anchor_id, ["x", "y", "z"]].to_numpy(), radio.range_m.to_numpy(),
        radio.sigma_m.to_numpy(), np.array(meta["R0"]), nominal, GraphConfig(), horizon=10.,
        lever=meta["lever_m"], gravity=meta["gravity_world_mps2"])
    first = graph.initial_state(np.zeros(3), 0.)
    sigma = np.r_[np.full(3, np.deg2rad(10.)), np.full(3, 3.), np.full(3, 20.), np.full(3, .01), np.full(3, .1)]
    res = graph.solve(first, sigma)
    return time.perf_counter()-before, res["state"][-1, 6:9]


def timed_window(folder):
    """Measure the actual first-window fit and its causal IMU integration work."""
    data = window.load_episode(folder)
    integration = sum(data["integration_work"][data["time"] <= 2])
    before = time.perf_counter()
    win, _, _, _ = window.make_window(data, 2., 2.)
    fit = window.attempt(win, "body_bias_prior")
    return time.perf_counter()-before+integration, fit["x"][:3]


def timed_spline(folder, target):
    """Time the first published optimized knot with the pinned native core."""
    build = SCRATCH/"sfuise_challenge"
    meta = json.loads((folder/"meta.json").read_text(encoding="utf-8"))
    native = pd.read_csv(build/"results"/folder.name/"native.csv").iloc[0]
    # The lossless replay duration is a clock limit, independent of the estimate values.
    end = native.arrival_s
    cmd = [str(build/"sfuise_native.exe"), "--imu", str(build/"results"/folder.name/"imu_common_calibrated.csv"),
           "--radio", str(folder/"radio.csv"), "--anchors", str(folder/"anchors.csv"),
           "--episode", folder.name, "--case", "native_restart", "--mode", "native",
           "--duration", str(end), "--output", str(target)]
    for key, value in zip(("--lx", "--ly", "--lz"), meta["lever_m"]):
        cmd.extend([key, str(value)])
    env = os.environ.copy()
    env["PATH"] = str(build/"deps/Library/bin") + os.pathsep + env["PATH"]
    subprocess.run(cmd, capture_output=True, check=True, env=env)
    first = pd.read_csv(target).iloc[0]
    return float(first.wall_s), first[["x", "y", "z"]].to_numpy(float)


def main():
    """Separate warm estimator cost from the parallel matrix's queue and intrinsic data wait."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=HERE/"evidence/native_warm_timing.csv")
    args = parser.parse_args()
    folders = sorted(p for p in (args.data/"streams").iterdir() if p.name.endswith("_start"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Load the numerical runtime and compiled kernels before the recorded pass.
    timed_graph(folders[0])
    timed_window(folders[0])
    rows = []
    with tempfile.TemporaryDirectory(dir=args.data) as scratch:
        for folder in folders:
            operations = (("SFUISE", lambda f: timed_spline(f, Path(scratch)/"spline.csv")),
                          ("body_bias_prior", timed_window), ("Jiang_independent", timed_graph))
            for method, fn in operations:
                elapsed, xyz = fn(folder)
                rows.append(dict(episode=folder.name, dataset=folder.name.split("_")[0], method=method,
                    first_work_s=elapsed, x=xyz[0], y=xyz[1], z=xyz[2],
                    timing_scope={"SFUISE":"numerical core from initialization through publication",
                    "body_bias_prior":"arrived IMU integration, window build and nonlinear fit",
                    "Jiang_independent":"CSV input, graph construction/preintegration and nonlinear solve"}[method]))
            if folder.name.startswith("delft_"):
                for kind in ("EKF", "MHE"):
                    pfeiffer.replay((folder.name, kind, "warm_timing", [0., 0., 0.], 0., None, .2, scratch))
                    first = pd.read_csv(Path(scratch)/f"{folder.name}__{kind}__warm_timing.csv").iloc[0]
                    rows.append(dict(episode=folder.name, dataset="delft", method=f"Pfeiffer_{kind}",
                        first_work_s=first.wall_s, x=first.x, y=first.y, z=first.z,
                        timing_scope="author replay from beginning of prefix through first release output"))
    frame = pd.DataFrame(rows)
    saved = []
    for row in frame.itertuples(index=False):
        if row.method == "SFUISE":
            source = SCRATCH/"sfuise_challenge/results"/row.episode/"native.csv"
        elif row.method == "body_bias_prior":
            source = SCRATCH/"profile_challenge/results"/(row.episode+"__native.csv")
        elif row.method == "Jiang_independent":
            source = HERE/"jiang/evidence"/(row.episode+"__native.csv")
        else:
            author_class = row.method.split("_")[1]
            source = SCRATCH/"pfeiffer_acquisition/traces/native"/f"{row.episode}__{author_class}__native_zero.csv"
        trace = pd.read_csv(source)
        if row.method == "body_bias_prior":
            trace = trace[trace.method == row.method]
        first = trace.sort_values("arrival_s").iloc[0]
        saved.append(np.linalg.norm(np.array([row.x, row.y, row.z]) - first[["x","y","z"]].to_numpy(float)))
    frame["replay_first_position_difference_m"] = saved
    if max(saved) > 1e-5:
        raise ValueError(f"Warm timing did not reproduce the first position, max difference {max(saved)}")
    frame.to_csv(args.out, index=False)
    print(frame.groupby(["dataset", "method"]).first_work_s.median().to_string())


if __name__ == "__main__":
    main()
