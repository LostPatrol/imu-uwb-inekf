"""Port the complete historical capture/tracker to one surveyed MILUV TWR tag.

Only file decoding, SI units and the supplied tag lever arm are adapted.
The original supervision, covariance, lifted priors and bad gates are retained.
"""

from dataclasses import dataclass
from functools import partial
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import yaml

# Import the existing full reference implementation without duplicating its policy.
CODE = Path(__file__).resolve().parents[3]
ROOT = CODE.parent
sys.path.insert(0, str(CODE/"experiments/roadmap_stage1"))
# Several experiments have a module named run; import this exact source by path.
LEGACY_SPEC = importlib.util.spec_from_file_location("external_legacy_twr",CODE/"experiments/roadmap_stage1/run.py")
LEGACY = importlib.util.module_from_spec(LEGACY_SPEC)
LEGACY_SPEC.loader.exec_module(LEGACY)
CaptureThenTrack, LiftedTWRInEKF = LEGACY.CaptureThenTrack, LEGACY.LiftedTWRInEKF
PRIOR_STD, SETTINGS = LEGACY.PRIOR_STD, LEGACY.SETTINGS
estimate_initial_rotation_and_bias = LEGACY.estimate_initial_rotation_and_bias
from invariant15 import Invariant15, coordinate_map, handoff_covariance
from eskf15 import skew

# One fixed robot/tag avoids treating two displaced tags as one body origin.
ROBOT = "ifo001"
TAG = 10
# The historical sensor-only prefix is unchanged and never uses motion capture.
PREFIX_S = 1.5
# The supplied MILUV single-robot benchmark uses this fixed TWR scale in metres.
RANGE_SIGMA_M = 0.30
# The original core expects acceleration in g and gyro in degrees per second.
GRAVITY = 9.81
# Only these raw IMU columns are ever accessed by the estimation side.
IMU_COLUMNS = ["timestamp"]+[f"linear_acceleration.{axis}" for axis in "xyz"]+[f"angular_velocity.{axis}" for axis in "xyz"]


@dataclass
class SensorRecord:
    """Truth-free sensor record in original-core units and one common survey frame."""

    name: str
    imu_time: np.ndarray
    imu: np.ndarray
    radio_time: np.ndarray
    radio_id: np.ndarray
    radio_range: np.ndarray
    anchors: np.ndarray
    lever: np.ndarray
    initial_rotation: np.ndarray
    initial_bias: np.ndarray
    origin_s: float
    metadata: dict


class LeverInvariant15(Invariant15):
    """Keep the full native invariant update while differentiating the actual tag."""

    def __init__(self, *args, lever=(0., 0., 0.), **kwargs):
        """Leave native covariance/noise policy intact and store the surveyed lever."""
        super().__init__(*args, **kwargs)
        self.external_lever = np.asarray(lever, dtype=float).reshape(3)

    def range_model(self, anchor):
        """Return the tag range and its full left-invariant physical Jacobian."""
        # Preserve bitwise legacy behavior when the original zero lever applies.
        if not np.any(self.external_lever):
            return super().range_model(anchor)
        tag = self.X[:3, 4]+self.X[:3, :3]@self.external_lever
        difference = tag-np.asarray(anchor)
        distance = float(np.linalg.norm(difference))
        if not np.isfinite(distance) or distance < 1e-12:
            raise ValueError("TWR direction is undefined")
        gradient = difference/distance
        row = np.zeros((1, 15))
        row[0, :3] = -gradient@skew(tag)
        row[0, 6:9] = gradient
        return distance, row


class LeverCapture(LiftedTWRInEKF):
    """Retain the native decoupled lifted capture with an estimated lever correction."""

    def __init__(self, *args, lever=(0., 0., 0.), **kwargs):
        """Construct the unmodified native capture and store its observation offset."""
        super().__init__(*args, **kwargs)
        self.external_lever = np.asarray(lever, dtype=float).reshape(3)

    def correct_twr(self, anchor, measurement, anchor_index=None):
        """Shift the anchor by R_hat*lever without changing the lifted dynamics/gate."""
        if not np.any(self.external_lever):
            return super().correct_twr(anchor, measurement, anchor_index)
        effective = np.asarray(anchor)-self.imu_core.X[:3, :3]@self.external_lever
        return super().correct_twr(effective, measurement, anchor_index)


def make_reference(state, covariance, bias, lever, range_bias=0., range_sigma=RANGE_SIGMA_M):
    """Construct the complete historical capture-reset reference with lever-aware factories."""
    return CaptureThenTrack(state, covariance, bias, reset_on_resume=True,
        tracker_factory=partial(LeverInvariant15, lever=lever),
        imu_core_factory=Invariant15,
        capture_factory=partial(LeverCapture, lever=lever),
        handoff_covariance_factory=handoff_covariance,
        range_bias=range_bias, range_sigma=range_sigma, **SETTINGS)


def active_core(reference):
    """Return the current full covariance owner for read-only audit snapshots."""
    return reference.capture.imu_core if reference.mode=="capture" else reference.tracker


def metadata_geometry(metadata, name):
    """Read immutable supplied geometry; never infer a transform from test trajectories."""
    table = pd.read_csv(metadata/"miluv_experiments.csv")
    row = table.loc[table.experiment==name].iloc[0]
    configurations = yaml.safe_load((metadata/"miluv_anchors.yaml").read_text(encoding="utf-8"))
    constellation = str(row.anchor_constellation)
    selected = configurations[constellation]
    anchors = np.array([json.loads(selected[str(i)]) for i in range(int(row.num_anchors))])
    tags = yaml.safe_load((metadata/"miluv_tags.yaml").read_text(encoding="utf-8"))
    lever = np.asarray(json.loads(tags[ROBOT][str(TAG)]))
    return anchors, lever, dict(anchor_constellation=constellation,num_anchors=len(anchors),robot=ROBOT,tag=TAG)


def read_sensors(dataset, name, metadata):
    """Read only original IMU/range columns, preserving true asynchronous arrivals."""
    directory = dataset/name/ROBOT
    # Official uwb_range.csv has test-GT row selection; require raw reconstruction.
    imu_path, radio_path = directory/"imu_px4.csv", directory/"uwb_range_unfiltered.csv"
    raw = pd.read_csv(imu_path, usecols=IMU_COLUMNS).sort_values("timestamp",kind="stable")
    raw = raw.drop_duplicates("timestamp",keep="first")
    finite = np.isfinite(raw[IMU_COLUMNS].to_numpy(float)).all(axis=1)
    raw = raw.loc[finite]
    stamps = raw.timestamp.to_numpy(float)
    if len(stamps)<20 or np.median(np.diff(stamps))<=0:
        raise ValueError("Insufficient increasing raw IMU timestamps")
    # Official cleanup merges secs+nsecs/1e9; cleaned timestamps are seconds.
    sample = raw[IMU_COLUMNS[1:]].to_numpy(float)
    sample[:, :3] /= GRAVITY
    sample[:, 3:] = np.rad2deg(sample[:, 3:])
    early = SimpleNamespace(time=stamps-stamps[0],imu=sample)
    rotation,bias = estimate_initial_rotation_and_bias(early)
    start = int(np.searchsorted(early.time,PREFIX_S))
    origin = float(stamps[start])
    anchors,lever,geometry = metadata_geometry(metadata,name)
    radio = pd.read_csv(radio_path,usecols=["timestamp","from_id","to_id","range"])
    # Only robot-initiated tag10-to-surveyed-anchor TWR is eligible.
    valid = ((radio.from_id==TAG)&radio.to_id.isin(range(len(anchors)))
             &np.isfinite(radio[["timestamp","range"]]).all(axis=1)&(radio.range>0)
             &(radio.timestamp>=origin)&(radio.timestamp<=stamps[-1]))
    radio = radio.loc[valid].sort_values("timestamp",kind="stable")
    details = dict(**geometry,source_imu_sha256=hashlib.sha256(imu_path.read_bytes()).hexdigest(),
        source_radio_sha256=hashlib.sha256(radio_path.read_bytes()).hexdigest(),
        raw_imu_columns=IMU_COLUMNS,radio_columns=["timestamp","from_id","to_id","range"],
        input_timestamp_units="seconds",origin_s=origin,prefix_s=PREFIX_S,
        imu_median_dt_s=float(np.median(np.diff(stamps))),imu_max_dt_s=float(np.max(np.diff(stamps))),
        body_frame_assumption="Official MILUV imu-only model directly uses raw PX4 body gyro/accel with supplied body-frame tag moment arms; no fitted test extrinsic.",
        gyro_bias_from_raw_prefix_rad_s=bias[:3].tolist(),accel_bias_from_raw_prefix_m_s2=bias[3:].tolist(),
        prefix_is_verified_static=False,eval_truth_in_estimator=False,
        no_official_mocap_derived_imu_bias=True,
        official_gt_filtered_uwb_range_csv_not_accepted=True)
    timeshift = dataset/name/"timeshift.yaml"
    if timeshift.exists():
        details["released_timeshift"] = yaml.safe_load(timeshift.read_text(encoding="utf-8"))
    return SensorRecord(name,stamps[start:]-origin,sample[start:],
        radio.timestamp.to_numpy(float)-origin,radio.to_id.to_numpy(int),radio.range.to_numpy(float),
        anchors,lever,rotation,bias,origin,details)


def initial_reference(record):
    """Use sensor-only tilt/bias, zero velocity and surveyed-centroid body position."""
    state = np.eye(5)
    state[:3,:3] = record.initial_rotation
    state[:3,4] = record.anchors.mean(axis=0)
    chart = coordinate_map(state)
    covariance = chart@np.diag(PRIOR_STD**2)@chart.T
    return make_reference(state,covariance,record.initial_bias,record.lever)
