"""Decode genuine Delft TDoA and freeze early-record hardware calibration.

Test truth is accessible only through the separate scoring loader.
"""

from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

# Fixed local dataset and experiment protocol; no historic estimates are loaded.
HERE = Path(__file__).resolve().parent
ROOT = Path("E:/datasets/tu-delft-uwb-mhe/data_v1/UWB dataset")
PROTOCOL = json.loads((HERE / "protocol.json").read_text(encoding="utf8"))
# Calibration lag and robust scale bounds are fixed before test truth is read.
LAG_GRID = np.arange(-30, 31) / 100
SCALE_BOUNDS = (.08, .5)


def digest(path):
    """Hash a source or raw data file for exact provenance."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def anchors():
    """Load the official surveyed Cyberzoo anchor coordinates in metres."""
    obj = yaml.safe_load((ROOT / "anchor_positions.yaml").read_text(encoding="utf8"))
    return np.array([[obj[i][axis] for axis in "xyz"] for i in range(8)])


def read_frame(name, truth=False):
    """Select sensor columns independently of Optitrack validity."""
    path = ROOT / "tdoa" / (name + ".csv")
    header = pd.read_csv(path, skipinitialspace=True, nrows=0).columns
    fields = [key for key in header if key not in ("otX", "otY", "otZ")]
    if truth:
        fields = ["timeTick", "otX", "otY", "otZ"]
    return pd.read_csv(path, skipinitialspace=True, usecols=fields)


def load(name, calibration=None, duration=45.):
    """Return native-unit IMU and new TDoA transactions without loading truth."""
    frame = read_frame(name)
    time = frame.timeTick.to_numpy(float) / 1000
    imu = frame[["accX", "accY", "accZ", "gyroX", "gyroY", "gyroZ"]].to_numpy(float)
    valid = np.isfinite(imu).all(1) & (np.linalg.norm(imu[:, :3], axis=1) > .5)
    origin = float(time[np.flatnonzero(valid)[0]])
    columns = sorted(key for key in frame if key.startswith("tdoa"))
    values = frame[columns].to_numpy(float)
    changed = np.zeros_like(values, dtype=bool)
    changed[1:] = np.abs(np.diff(values, axis=0)) > 1e-12
    changed &= np.isfinite(values)
    row, channel = np.nonzero(changed & valid[:, None] & ((time-origin <= duration)[:, None]))
    pairs = np.array([[int(key[-2]), int(key[-1])] for key in columns])
    radio = np.column_stack([time[row]-origin, pairs[channel], values[row, channel]])
    if calibration is not None:
        radio[:, 3] -= np.array(calibration["pair_bias_m"])[channel]
    select = valid & (time-origin <= duration)
    return dict(name=name, times=time[select]-origin, imu=imu[select], radio=radio,
                channel=channel, columns=columns, anchors=anchors(), origin=origin,
                source_sha256=digest(ROOT/"tdoa"/(name+".csv")))


def truth(name, origin, lag):
    """Load scoring-only Optitrack in the calibrated sensor clock."""
    frame = read_frame(name, truth=True)
    position = frame[["otX", "otY", "otZ"]].to_numpy(float)
    valid = np.isfinite(position).all(1) & (np.linalg.norm(position, axis=1) > 1e-9)
    return np.column_stack([frame.timeTick.to_numpy(float)[valid]/1000-origin-lag, position[valid]])


def calibrate():
    """Fit channel bias and timestamp lag using only the first six recordings."""
    logs = []
    for name in PROTOCOL["calibration_names"]:
        data = load(name, duration=np.inf)
        reference = truth(name, data["origin"], 0.)
        logs.append((data, reference))
    objectives = []
    residual_sets = []
    for lag in LAG_GRID:
        residuals, log_objectives = [], []
        for data, reference in logs:
            radio = data["radio"]
            times = radio[:, 0] + lag
            valid = (times >= reference[0, 0]) & (times <= reference[-1, 0])
            positions = np.column_stack([np.interp(times[valid], reference[:, 0], reference[:, k]) for k in (1,2,3)])
            r = radio[valid]
            prediction = np.linalg.norm(positions-data["anchors"][r[:,2].astype(int)],axis=1)-np.linalg.norm(positions-data["anchors"][r[:,1].astype(int)],axis=1)
            residual = r[:,3]-prediction
            channels = data["channel"][valid]
            centered = residual.copy()
            for pair in range(28):
                use = channels == pair
                if use.any():
                    centered[use] -= np.median(residual[use])
            log_objectives.append(float(np.median(np.abs(centered))))
            residuals.append((channels,residual))
        objectives.append(float(np.median(log_objectives)))
        residual_sets.append(residuals)
    best = int(np.argmin(objectives))
    pieces = residual_sets[best]
    bias, sigma, count = [], [], []
    for pair in range(28):
        residual = np.concatenate([r[c==pair] for c,r in pieces])
        median = float(np.median(residual))
        bias.append(median)
        sigma.append(float(np.clip(1.4826*np.median(np.abs(residual-median)), *SCALE_BOUNDS)))
        count.append(len(residual))
    result = dict(lag_s=float(LAG_GRID[best]), lag_grid_s=LAG_GRID.tolist(), lag_objective_m=objectives,
                  pair_columns=logs[0][0]["columns"], pair_bias_m=bias, pair_sigma_m=sigma,
                  pair_count=count, calibration_names=PROTOCOL["calibration_names"],
                  calibration_sha256={d["name"]:d["source_sha256"] for d,_ in logs})
    return result


def fit_pv(radio, anchor, endpoint, seed=None):
    """Fit antenna constant-velocity motion to unique raw differences only."""
    if len(radio) < 8:
        raise ValueError("Fewer than eight distinct radio transactions")
    tau = radio[:,0]-endpoint
    plus, minus = anchor[radio[:,2].astype(int)], anchor[radio[:,1].astype(int)]
    def residual(x):
        """Evaluate the native hyperbolic range difference."""
        position = x[:3]+tau[:,None]*x[3:]
        return (np.linalg.norm(position-plus,axis=1)-np.linalg.norm(position-minus,axis=1)-radio[:,3])/.4
    if seed is None:
        seed = np.r_[anchor.mean(0), np.zeros(3)]
    result = least_squares(residual, seed, loss="huber", f_scale=2., max_nfev=80)
    return result.x, dict(success=bool(result.success),nfev=int(result.nfev),cost=float(result.cost))


def initial_state(data):
    """Use a fixed .6s prefix for gravity, gyro bias and radio-only p/v."""
    start = .6
    prefix = data["imu"][data["times"] <= start]
    mean_acc = prefix[:,:3].mean(0)
    direction = mean_acc/np.linalg.norm(mean_acc)
    cross = np.cross(direction, [0.,0.,1.])
    angle = np.arctan2(np.linalg.norm(cross), direction[2])
    rot = Rotation.from_rotvec(cross/max(np.linalg.norm(cross),1e-15)*angle).as_matrix()
    state = np.eye(5)
    state[:3,:3] = rot
    state[:3,4], state[:3,3] = np.split(fit_pv(data["radio"][data["radio"][:,0] <= start],data["anchors"],start)[0],2)
    # Bias uses the author's native acceleration-first ordering and physical SI units.
    bias = np.r_[mean_acc*9.81-rot.T@np.array([0.,0.,9.81]), np.radians(prefix[:,3:].mean(0))]
    return start, state, bias
