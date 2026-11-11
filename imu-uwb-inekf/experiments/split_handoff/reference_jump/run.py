"""Controlled decision-time yaw-coordinate loss with matched native InEKF restarts.

No pose columns are read until all sensor-only decisions and trajectories finish.
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

# The frozen shared implementation is imported read-only under an unambiguous name.
HERE = Path(__file__).resolve().parent
UTIL = HERE.parent/'util'
sys.path.insert(0, str(UTIL))
sys.path.insert(0, str(HERE.parent))
spec = importlib.util.spec_from_file_location('reference_jump_shared_util', UTIL/'run.py')
shared = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared)
import yaw_evidence

# All scenario, availability, intervention and scoring choices are preregistered.
RULES = json.loads((HERE/'rules.json').read_text(encoding='utf-8'))
DATA = Path('E:/datasets/dataset/flight-dataset')
DT = .01


def sensor_data(name):
    """Read only sensor columns and resample causally, without constructing any pose stream."""
    columns = ['t_acc', 'acc_x', 'acc_y', 'acc_z', 't_gyro', 'gyro_x', 'gyro_y', 'gyro_z',
               't_tdoa', 'idA', 'idB', 'tdoa_meas']
    file = DATA/'csv-data'/name.split('-')[0]/(name+'.csv')
    frame = pd.read_csv(file, usecols=columns)
    def stream(names):
        """Sort independently timestamped sensor values without borrowing another stream clock."""
        rows = frame[names].dropna().to_numpy(float)
        return rows[np.argsort(rows[:, 0], kind='stable')]
    acc, gyro, radio = (stream(names) for names in (columns[:4], columns[4:8], columns[8:]))
    origin = max(acc[0, 0], gyro[0, 0], radio[0, 0])
    end = min(origin+RULES['duration_s'], acc[-1, 0], gyro[-1, 0], radio[-1, 0])
    times = np.arange(origin, end, DT)
    # A sample after a decision cannot alter its resampled endpoint measurement.
    imu = np.column_stack([raw[np.searchsorted(raw[:, 0], times, side='right')-1, 1:]
                           for raw in (acc, gyro)])
    radio = radio[(radio[:, 0] >= origin) & (radio[:, 0] <= times[-1])]
    if 'tdoa2' in name:
        ring = {(j, (j+1) % 8) for j in range(8)}
        radio = radio[np.array([tuple(pair) in ring for pair in radio[:, 1:3]])]
    anchors = np.load(DATA/'survey-results'/('anchor_'+name.split('-')[0]+'.npz'))['an_pos']
    return dict(name=name, times=times, imu=imu, radio=radio, anchors=anchors)


def write_json(path, value):
    """Save complete model diagnostics using the shared array-preserving serializer."""
    path.write_text(json.dumps(shared.safe_json(value), indent=2), encoding='utf-8')


def manifest():
    """Resolve eight preregistered slots using filenames only, retaining missing slots."""
    rows = []
    for geometry in RULES['geometries']:
        for protocol in RULES['protocols']:
            canonical = shared.record_name(geometry, RULES['trial'], protocol)
            directory = DATA/'csv-data'/f'const{geometry}'
            exact = directory/(canonical+'.csv')
            manuals = sorted(directory.glob(f'const{geometry}-trial7-tdoa{protocol}-manual*.csv'))
            chosen = exact if exact.exists() else manuals[0] if manuals else None
            rows.append(dict(geometry=geometry, protocol=protocol, canonical=canonical,
                             name=None if chosen is None else chosen.stem,
                             missing=chosen is None, available_manuals=[p.stem for p in manuals]))
    return rows


def run_scenario(data, scenario, output, verify_full):
    """Freeze one normal shadow and change only its decision-time attitude across policies."""
    origin = data['times'][0]
    rule = RULES['scenarios'][scenario]
    start, end = [data['times'][np.searchsorted(data['times'], origin+offset)] for offset in rule['window']]
    visible = shared.visible_data(data, scenario)
    t0, state, bias = shared.initial_state(visible)
    mapping = shared.kernel.mapping(state)
    covariance = mapping@shared.PHYSICAL@mapping.T
    state, covariance, bias, _, _, _, prefix_failed = shared.bounded_replay(visible, state, covariance, bias, t0, start, None)
    window_bias = bias.copy()
    state, covariance, bias, _, _, _, window_failed = shared.bounded_replay(visible, state, covariance, bias, start, end, None)
    shadow, shadow_bias = state.copy(), bias.copy()
    inverse = shared.kernel.inverse_mapping(shadow)
    physical = inverse@covariance@inverse.T
    physical = .5*(physical+physical.T)
    clipped = shared.visible_data(visible, scenario, end)
    clipped['times'] = clipped['times'][clipped['times'] <= end]
    clipped['imu'] = clipped['imu'][:len(clipped['times'])]
    path = shared.radio_path(clipped)
    inputs = shared.window_inputs(clipped, start, end, window_bias, path)
    end = inputs['endpoint']
    nominal_ref = shadow[:3, :3]@inputs['Qend'].T
    full_rotation, full_diagnostic = shared.fit_full_attitude(inputs, nominal_ref, np.r_[shadow[:3, 4], shadow[:3, 3]])
    verification = None
    if verify_full:
        jumped_ref = Rotation.from_euler('z', 90, degrees=True).as_matrix()@nominal_ref
        direct_rotation, direct_diagnostic = shared.fit_full_attitude(inputs, jumped_ref, np.r_[shadow[:3, 4], shadow[:3, 3]])
        verification = dict(relative_so3_deg=float(np.degrees(Rotation.from_matrix(full_rotation.T@direct_rotation).magnitude())),
                            nominal=full_diagnostic, direct_90=direct_diagnostic)
        write_json(output/'full_reparameterization_check.json', verification)
    records = []
    for jump in RULES['decision_time_left_yaw_jumps_deg']:
        jumped = shadow.copy()
        jumped[:3, :3] = Rotation.from_euler('z', jump, degrees=True).as_matrix()@shadow[:3, :3]
        cref = jumped[:3, :3]@inputs['Qend'].T
        decision = shared.invoke_decision(yaw_evidence, inputs, cref)
        rotations = dict(retain=jumped[:3, :3],
                         gated=Rotation.from_euler('z', decision['delta']).as_matrix()@jumped[:3, :3],
                         ungated_same_candidate=Rotation.from_euler('z', decision['candidate']).as_matrix()@jumped[:3, :3],
                         full_attitude=full_rotation)
        for policy, rotation in rotations.items():
            handoff = jumped.copy()
            handoff[:3, :3] = rotation
            mapping = shared.kernel.mapping(handoff)
            initial_cov = mapping@physical@mapping.T
            # Original world p/v, body bias and physical covariance are exactly shared.
            assert np.array_equal(handoff[:3, 3:5], shadow[:3, 3:5])
            finish = min(end+max(RULES['future_horizons_s']), visible['times'][-1])
            final, final_cov, final_bias, after, accepted, rejected, failed = shared.bounded_replay(
                visible, handoff, initial_cov, shadow_bias, end, finish, None)
            key = f"{data['name']}__{scenario}__jump{jump}__{policy}"
            np.savez_compressed(output/(key+'.npz'), after=after, handoff=handoff, shadow=shadow,
                                jumped=jumped, physical_covariance=physical, bias=shadow_bias,
                                final=final, final_covariance=final_cov, final_bias=final_bias)
            records.append(dict(key=key, name=data['name'], scenario=scenario, jump_deg=jump, policy=policy,
                                origin=origin, handoff_time=end, requested_future_horizons_s=RULES['future_horizons_s'],
                                failed=bool(prefix_failed or window_failed or failed),
                                accepted_radio=accepted, rejected_radio=rejected,
                                candidate_packets=len(inputs['y']), source_packets=inputs['source_packets'],
                                decision=decision, full_candidate=full_diagnostic,
                                full_reuse_verification=verification))
        write_json(output/(data['name']+'__'+scenario+f'__jump{jump}.json'), records[-4:])
    return records


def score(output, records):
    """Read pose columns only after all interventions and future trajectories are saved."""
    truths, rows = {}, []
    for record in records:
        name = record['name']
        if name not in truths:
            columns = ['t_pose', 'pose_x', 'pose_y', 'pose_z', 'pose_qx', 'pose_qy', 'pose_qz', 'pose_qw']
            frame = pd.read_csv(DATA/'csv-data'/name.split('-')[0]/(name+'.csv'), usecols=columns)
            truth = frame[columns].dropna().to_numpy(float)
            truths[name] = truth[np.argsort(truth[:, 0], kind='stable')]
        truth = truths[name]
        saved = np.load(output/(record['key']+'.npz'))
        row = {k:v for k,v in record.items() if k not in ('decision', 'full_candidate', 'full_reuse_verification')}
        row.update(heading_accepted=record['decision']['accepted'],
                   candidate_delta_deg=float(np.degrees(record['decision']['candidate'])),
                   gated_delta_deg=float(np.degrees(record['decision']['delta'])),
                   sd_deg=record['decision']['sd_deg'])
        for label in ('shadow', 'jumped', 'handoff'):
            state = saved[label]
            event = np.r_[record['handoff_time'], state[:3, 4], state[:3, 3], state[:3, :3].ravel(), np.zeros(3)][None]
            row.update({label+'_'+key:value for key,value in shared.score_rows(truth, event).items()})
        for horizon in RULES['future_horizons_s']:
            after = saved['after']
            selected = after[after[:, 0] <= record['handoff_time']+horizon+1e-9]
            metrics = shared.score_rows(truth, selected, record['handoff_time']+.05, record['handoff_time']+horizon)
            row.update({f'future{horizon}_'+key:value for key,value in metrics.items()})
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(output/'metrics.csv', index=False)
    fields = [f'future{horizon}_{field}' for horizon in RULES['future_horizons_s']
              for field in ('xy_rmse', 'z_rmse', 'yaw_rmse_deg')]
    frame.groupby(['scenario', 'jump_deg', 'policy'])[fields].median().to_csv(output/'summary.csv')


def main():
    """Generate every available preregistered slot before scoring any physical error."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--score-only', action='store_true')
    args = parser.parse_args()
    output = HERE/'results'
    output.mkdir(exist_ok=True)
    if args.score_only:
        score(output, json.loads((output/'records.json').read_text(encoding='utf-8')))
        return
    slots = manifest()
    write_json(output/'manifest.json', slots)
    write_json(output/'source_hashes.json', {str(file):hashlib.sha256(file.read_bytes()).hexdigest()
               for file in (HERE/'rules.json', UTIL/'run.py', UTIL/'kernel.py', HERE.parent/'yaw_evidence.py')})
    records, verified, begun = [], False, time.perf_counter()
    for slot in slots:
        if slot['missing']:
            print(json.dumps(dict(missing=slot)), flush=True)
            continue
        data = sensor_data(slot['name'])
        for scenario in RULES['scenarios']:
            required = RULES['scenarios'][scenario]['window'][-1]+max(RULES['future_horizons_s'])
            if data['times'][-1]-data['times'][0] < required:
                write_json(output/(slot['name']+'__'+scenario+'__insufficient_duration.json'),
                           dict(required_s=required, available_s=float(np.ptp(data['times']))))
                continue
            records.extend(run_scenario(data, scenario, output, not verified))
            verified = True
            write_json(output/'records.json', records)
            print(json.dumps(dict(name=slot['name'], scenario=scenario, trajectories=len(records),
                                  elapsed_s=time.perf_counter()-begun)), flush=True)
    print('All sensor-only decisions complete; now reading pose columns for scoring.', flush=True)
    if records:
        score(output, records)


if __name__ == '__main__':
    main()
