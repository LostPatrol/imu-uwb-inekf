"""Decode QDrone raw sensors separately from the provider's inserted prism truth.

Raw streams keep every supplied sensor packet; test-truth correction coefficients,
height, device quaternion and all inserted truth rows are excluded from estimation.
"""

import csv
import hashlib
import json
from pathlib import Path
import re
from types import SimpleNamespace

import numpy as np

from adapter import (GRAVITY, PREFIX_S, SensorRecord,
                     estimate_initial_rotation_and_bias)

# Packet markers are defined by the official QDrone CSV description.
IMU_MARKER, RANGE_MARKER, HEIGHT_MARKER, TRUTH_MARKER = "2", "0", "3", "-1"
# The four fixed P440 anchor IDs are shared by the published site geometry.
ANCHOR_IDS = (100,101,102,103)
# Frozen before any QDrone estimator output: physical lever has not been supplied.
BOUNDS_ENGINEERING_RADIUS_M = 1.0


def csv_rows(path):
    """Ignore formatting-only empty rows without interpreting any packet payload."""
    with path.open("r",encoding="utf-8-sig",newline="") as source:
        for row in csv.reader(source):
            if row and any(part.strip() for part in row):
                yield [part.strip() for part in row]


def is_marker(row):
    """Accept both blank-padded and zero-padded official packet marker rows."""
    if row[0] not in (IMU_MARKER,RANGE_MARKER,HEIGHT_MARKER):
        return False
    try:
        return all(not x or float(x)==0 for x in row[1:])
    except ValueError:
        return False


def raw_packets(path):
    """Read only timestamped six-axis IMU and per-anchor ranges from an original raw file."""
    imu,radio=[],[]
    expected=None
    ignored_truth=0
    malformed=0
    for row in csv_rows(path):
        if row[0]==TRUTH_MARKER:
            ignored_truth+=1
            expected=None
            continue
        if expected is None:
            if is_marker(row):
                expected=row[0]
            continue
        packet=expected
        expected=None
        try:
            if packet==IMU_MARKER:
                # Quaternion values exist later on the row but are never parsed.
                values=np.array([float(value) for value in row[:7]])
                if len(values)==7 and np.isfinite(values).all():
                    imu.append(values)
                else:
                    malformed+=1
            elif packet==RANGE_MARKER:
                values=np.array([float(value) for value in row[:3]])
                if len(values)==3 and np.isfinite(values).all() and values[1] in ANCHOR_IDS and values[2]>0:
                    radio.append(values)
                else:
                    malformed+=1
        except (ValueError,IndexError):
            malformed+=1
    return np.asarray(imu),np.asarray(radio),dict(ignored_inserted_truth_rows=ignored_truth,
        malformed_or_invalid_sensor_packets=malformed)


def prism_truth(path):
    """Bind provided truth only to its preceding timestamped IMU/UWB event; never extrapolate."""
    samples=[]
    packet=None
    attached_stamp=None
    invalid=untimed=0
    for row in csv_rows(path):
        if row[0]==TRUTH_MARKER:
            if attached_stamp is None:
                untimed+=1
                continue
            try:
                value=np.array([attached_stamp]+[float(x) for x in row[1:4]])
                if len(value)==4 and np.isfinite(value[0]):
                    samples.append(value)
                    invalid+=int(not np.isfinite(value[1:]).all())
                else:
                    invalid+=1
            except ValueError:
                invalid+=1
            attached_stamp=None
        elif packet is not None:
            if packet in (IMU_MARKER,RANGE_MARKER):
                try:
                    attached_stamp=float(row[0])
                except ValueError:
                    attached_stamp=None
            else:
                attached_stamp=None
            packet=None
        elif is_marker(row):
            packet=row[0]
            attached_stamp=None
        elif attached_stamp is not None:
            # The 2018 indoor release appends bare XYZ without a -1 marker.
            try:
                value=np.array([attached_stamp]+[float(x) for x in row[:3]])
                if len(value)==4 and np.isfinite(value[0]):
                    samples.append(value)
                    invalid+=int(not np.isfinite(value[1:]).all())
                else:
                    invalid+=1
            except ValueError:
                invalid+=1
            attached_stamp=None
    values=np.asarray(samples)
    if not len(values):
        return np.empty((0,4)),dict(finite_timestamped_truth=0,untimed_truth_rows=untimed,invalid_truth_rows=invalid)
    values=values[np.argsort(values[:,0],kind="stable")]
    unique=np.r_[True,np.diff(values[:,0])>0]
    values=values[unique]
    return values,dict(finite_timestamped_truth=int(np.isfinite(values[:,1:]).all(axis=1).sum()),untimed_truth_rows=untimed,
        invalid_truth_rows=invalid,first_truth_time_s=float(values[0,0]),last_truth_time_s=float(values[-1,0]),
        published_truth_is_provider_interpolation=True,independent_raw_total_station_timestamps_available=False)


def normalize_name(name):
    """Match released raw/GT names without using data values or estimator errors."""
    return name.lower().replace("_groundtruth_addedtest","").replace("_groundtruth_added","")


def inventory(dataset):
    """Enumerate every raw recording, including source-labeled failures and missing truth."""
    gt_root=dataset/"extracted/Dataset with Ground Truth"
    gt={normalize_name(path.name):path for path in gt_root.rglob("*.csv")}
    result=[]
    raw_root=dataset/"extracted/All Datasets"
    for path in sorted(raw_root.rglob("*.csv")):
        site=path.relative_to(raw_root).parts[0]
        truth=gt.get(normalize_name(path.name))
        result.append(dict(site=site,name=path.stem,raw_path=str(path),truth_path=str(truth) if truth else None))
    return result


def make_record(entry,anchors):
    """Construct the truth-free full-reference input using only original raw measurements."""
    path=Path(entry["raw_path"])
    imu,radio,details=raw_packets(path)
    if len(imu)<20 or len(radio)<4:
        raise ValueError("Insufficient timestamped raw IMU/radio packets")
    details["imu_out_of_order_or_duplicate_rows"]=int(np.sum(np.diff(imu[:,0])<=0))
    imu=imu[np.argsort(imu[:,0],kind="stable")]
    imu=imu[np.r_[True,np.diff(imu[:,0])>0]]
    radio=radio[np.argsort(radio[:,0],kind="stable")]
    raw_common_start=float(max(imu[0,0],radio[0,0]))
    raw_common_end=float(min(imu[-1,0],radio[-1,0]))
    samples=np.c_[imu[:,4:7]/GRAVITY,np.rad2deg(imu[:,1:4])]
    initial=SimpleNamespace(time=imu[:,0]-imu[0,0],imu=samples)
    prefix=initial.time<=PREFIX_S
    prefix_norm=float(np.median(np.linalg.norm(imu[prefix,4:7],axis=1)))
    details["prefix_acceleration_norm_m_s2"]=prefix_norm
    details["specific_force_supported"]=bool(.5*GRAVITY<=prefix_norm<=1.5*GRAVITY)
    rotation,bias=estimate_initial_rotation_and_bias(initial)
    start=int(np.searchsorted(initial.time,PREFIX_S))
    # Preserve the released relative recording clock: T=25 is not T=26.5.
    origin=float(imu[0,0]) if imu[0,0]>1e8 else 0.
    use=(radio[:,0]>=imu[start,0])&(radio[:,0]<=imu[-1,0])
    radio=radio[use]
    dates=re.findall(r"\d{4}-\d{2}-\d{2}",path.name)
    details.update(site=entry["site"],source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        bounds_lever_radius_m=BOUNDS_ENGINEERING_RADIUS_M,
        nominal_lever_m=[0.,0.,0.],lever_is_unmeasured_approximation=True,
        quaternion_used_for_estimation=False,height_used=False,range_self_error_used=False,
        record_specific_gt_fitted_range_correction_used=False,
        raw_acceleration_units="m/s^2 including gravity",raw_gyro_units_assumed="rad/s",
        raw_IMU_axis_assumption="Six axes treated as body axes following official raw DJI IMU stream; no fitted test rotation.",
        orientation_initialization="Same sensor-only static 1.5s tilt/bias; startup world yaw is zero, not DJI quaternion.",
        prefix_is_verified_static=False,origin_s=origin,
        raw_common_start_s=raw_common_start-origin,raw_common_end_s=raw_common_end-origin,
        raw_common_midpoint_s=.5*(raw_common_start+raw_common_end)-origin,
        first_replay_time_s=float(imu[start,0]-origin),
        imu_median_dt_s=float(np.median(np.diff(imu[:,0]))),imu_max_dt_s=float(np.max(np.diff(imu[:,0]))),
        source_filename_calendar_date=dates[0] if dates else None)
    record=SensorRecord(entry["name"],imu[start:,0]-origin,samples[start:],
        radio[:,0]-origin,radio[:,1].astype(int)-100,radio[:,2],np.asarray(anchors),np.zeros(3),
        rotation,bias,origin,details)
    return record


def truth_at(times,truth):
    """Interpolate only inside the released truth support, without filling exterior intervals."""
    output=np.full((len(times),3),np.nan)
    if len(truth)<2:
        return output,np.zeros(len(times),dtype=bool)
    right=np.clip(np.searchsorted(truth[:,0],times),1,len(truth)-1)
    supported=(times>=truth[0,0])&(times<=truth[-1,0])&((truth[right,0]-truth[right-1,0])<=.20)
    # Keep invalid provider rows as barriers instead of interpolating across NaN holes.
    supported &= np.isfinite(truth[right,1:]).all(axis=1)&np.isfinite(truth[right-1,1:]).all(axis=1)
    for j in range(3):
        output[supported,j]=np.interp(times[supported],truth[:,0],truth[:,j+1])
    return output,supported
