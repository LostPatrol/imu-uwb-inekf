"""Replay pristine Pfeiffer EKF/MHE kernels on frozen acquisition streams.

The author drone adapter defines thrust as body accelerometer z times mass.
No altitude, truth-dependent variance, or test truth enters this replay.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import random
import sys
import time

# Bound the numerical thread count independently of the process count.
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
import pandas as pd

# Paths, physical constants, and publication tuning are frozen across episodes.
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
STREAMS = ROOT / "agent/codex/acquisition_challenge_20261113/streams"
AUTHOR = ROOT / "third_party/pfeiffer_uwb_simulator_2021"
MASS = .03
DRAG = [.15, .15, 0.]
MHE_SETTINGS = dict(enable=True, rate=50, N_max=20, iterations=1, alpha=.015,
                    mu=10., ransac_iterations=10, ransac_threshold=1.7,
                    ransac_fraction=.4, attitude_rate=250)
EKF_SETTINGS = dict(enable=True, rate=100, outlierThreshold=1500)
SEED = 20261113
sys.path.insert(0, str(AUTHOR))
from UWBsim.estimators.mhe import MHE, MHE_Params
from UWBsim.estimators.ekf import EKF, EKF_Params
from UWBsim.utils.dataTypes import State_XVQW, TWR_meas


def replay(task):
    """Publish every original core update with sensor and cumulative wall clocks."""
    episode, method, case, position, yaw, width, horizon, output = task
    random.seed(SEED)
    np.random.seed(SEED)
    folder = STREAMS / episode
    imu = pd.read_csv(folder / "imu.csv").to_numpy()
    radio = pd.read_csv(folder / "radio.csv").to_numpy()
    anchors = pd.read_csv(folder / "anchors.csv").set_index("anchor_id")
    initial = State_XVQW(x=np.asarray(position), q=[np.cos(yaw/2), 0, 0, np.sin(yaw/2)])
    core = (MHE(MHE_Params(**MHE_SETTINGS), MASS, DRAG, initial)
            if method == "MHE" else EKF(EKF_Params(**EKF_SETTINGS), MASS, DRAG, initial))
    if width is not None and method == "EKF":
        # State injection changes only a requested prior, never author equations.
        core.P[:3, :3] = np.eye(3) * width[0]**2
        core.P[8, 8] = np.deg2rad(width[1])**2
    rate = MHE_SETTINGS["rate"] if method == "MHE" else EKF_SETTINGS["rate"]
    next_update = 1 / rate
    j = 0
    rows = []
    begun = time.perf_counter()
    status = "ok"
    failure = ""
    prior_applied = width is None
    try:
        for sample in imu:
            t = float(sample[0])
            if t > horizon + 1e-8:
                break
            clock = t + 1.5
            if not prior_applied and t >= -1e-8:
                # The sealed recovery center belongs to release, after the prefix.
                if method == "EKF":
                    core = EKF(EKF_Params(**EKF_SETTINGS), MASS, DRAG, initial)
                    # The author quaternion integrator divides by gyro*dt norm.
                    # Start at the preceding update to retain its positive dt.
                    core.t = clock - 1 / rate
                    core.P[:3, :3] = np.eye(3) * width[0]**2
                    core.P[8, 8] = np.deg2rad(width[1])**2
                else:
                    core.reset(initial)
                prior_applied = True
            acc, gyro = sample[1:4], sample[4:7]
            core.addInputs(float(acc[2]*MASS), gyro, acc, clock)
            while j < len(radio) and radio[j, 0] <= t + 1e-8:
                stamp, aid, distance, sigma = radio[j]
                core.addMeasurement(TWR_meas(anchors.loc[int(aid)].to_numpy(), int(aid),
                    float(distance), stdDev=float(sigma), timestamp=float(stamp+1.5)))
                j += 1
            if clock + 1e-8 < next_update:
                continue
            estimate = core.step(clock)
            next_update += 1 / rate
            if t < -1e-8:
                continue
            q = np.asarray([estimate.q[k] for k in range(4)])
            heading = np.arctan2(2*(q[0]*q[3]+q[1]*q[2]), 1-2*(q[2]**2+q[3]**2))
            rows.append(dict(episode=episode, case_id=case, method=f"Pfeiffer_{method}",
                arrival_s=t, state_s=t, x=estimate.x[0], y=estimate.x[1], z=estimate.x[2],
                wall_s=time.perf_counter()-begun, yaw_rad=heading, status="ok"))
    except Exception as error:
        status = "failed"
        failure = type(error).__name__ + ": " + str(error)
        rows.append(dict(episode=episode, case_id=case, method=f"Pfeiffer_{method}",
            arrival_s=t, state_s=t, x=np.nan, y=np.nan, z=np.nan,
            wall_s=time.perf_counter()-begun, yaw_rad=np.nan, status=failure))
    path = Path(output) / f"{episode}__{method}__{case}.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return dict(episode=episode, method=f"Pfeiffer_{method}", case_id=case, status=status,
        failure=failure, rows=len(rows), wall_s=time.perf_counter()-begun,
        radio_consumed=j, last_state_s=rows[-1]["state_s"] if rows else None)


def main():
    """Run native zero-seed episodes or the sealed supplied recovery priors."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--episode", default="")
    parser.add_argument("--method", default="")
    parser.add_argument("--horizon", type=float, default=15.)
    parser.add_argument("--stress", action="store_true")
    parser.add_argument("--output", type=Path, default=HERE / "native")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    episodes = [x.name for x in sorted(STREAMS.glob("delft_*")) if not args.episode or args.episode in x.name]
    tasks = []
    if args.stress:
        for episode in episodes:
            priors = STREAMS / episode / "priors.csv"
            if not priors.exists():
                continue
            for prior in pd.read_csv(priors).to_dict("records"):
                for method in ("EKF", "MHE"):
                    if args.method and method != args.method:
                        continue
                    # MHE exposes neither heading initialization nor covariance.
                    if method == "MHE" and (prior["yaw_delta_deg"] != 0 or prior["confidence"] != "wide"):
                        continue
                    tasks.append((episode, method, prior["case_id"],
                        [prior[k] for k in ("px", "py", "pz")], prior["yaw_rad"],
                        [prior["pos_std_m"], np.rad2deg(prior["yaw_std_rad"])],
                        args.horizon, str(args.output)))
    else:
        for episode in episodes:
            for method in ("EKF", "MHE"):
                if not args.method or method == args.method:
                    tasks.append((episode, method, "native_zero", [0., 0., 0.], 0., None,
                        args.horizon, str(args.output)))
    summaries = []
    with ProcessPoolExecutor(args.workers) as pool:
        for future in as_completed([pool.submit(replay, task) for task in tasks]):
            result = future.result()
            summaries.append(result)
            print(json.dumps(result), flush=True)
            pd.DataFrame(summaries).to_csv(args.output / "execution.csv", index=False)


if __name__ == "__main__":
    main()
