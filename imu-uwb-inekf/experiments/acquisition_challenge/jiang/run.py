"""Run independent 10 s Jiang-style initialization on isolated common inputs.

The estimator reads streams only. Truth and scoring directories are never opened.
Each finished case is checkpointed before the next optimization starts.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
import os
from pathlib import Path
import time
from dataclasses import asdict

# Every worker uses one BLAS thread to avoid oversubscribing the shared machine.
for name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[name] = '1'
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from factor_graph import BatchGraph, GraphConfig

# Result columns preserve native first-availability time and numerical termination.
FIELDS = ['episode','case_id','arrival_s','state_s','x','y','z','yaw_rad','wall_s',
          'status','nodes','ranges','initial_cost','final_cost','iterations',
          'accepted_steps','rejected_steps','residual_calls','framework_prior']


def append(path, row):
    """Immediately preserve one completed or failed case as UTF-8 CSV."""
    exists = path.exists()
    with path.open('a', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def load_graph(folder, horizon, jacobian='analytic'):
    """Load exclusively the agreed sensor stream and construct its accumulated graph."""
    meta = json.loads((folder/'meta.json').read_text(encoding='utf-8'))
    imu = pd.read_csv(folder/'imu.csv')
    radio = pd.read_csv(folder/'radio.csv')
    anchors = pd.read_csv(folder/'anchors.csv').set_index('anchor_id')
    radio = radio[(radio.t >= 0.) & (radio.t <= horizon)]
    # TDoA is outside this article's TWR likelihood; do not invent pseudo-ranges.
    if 'range_m' not in radio.columns:
        raise ValueError('Jiang ranging model requires actual TWR range_m input')
    imu = imu[(imu.t >= 0.) & (imu.t <= horizon)]
    r0 = np.asarray(meta['R0'], dtype=float)
    bias0 = np.r_[meta['gyro_bias'], meta['accel_bias']]
    ea = anchors.loc[radio.anchor_id, ['x','y','z']].to_numpy()
    graph = BatchGraph(imu.t.to_numpy(), imu[['ax','ay','az']].to_numpy(),
                       imu[['gx','gy','gz']].to_numpy(), radio.t.to_numpy(),
                       ea, radio.range_m.to_numpy(), radio.sigma_m.to_numpy(),
                       r0, bias0, GraphConfig(jacobian=jacobian), horizon=horizon, lever=meta.get('lever_m', [0.,0.,0.]),
                       gravity=meta.get('gravity_world_mps2', [0.,0.,-9.81]))
    return graph, meta


def result_row(episode, case_id, graph, result, elapsed, framework):
    """Record the current state only; smoothed pre-initialization poses are unpublished."""
    state = result['state'][-1]
    rotation = Rotation.from_rotvec(state[:3]).as_matrix()@graph.r0
    return dict(episode=episode, case_id=case_id, arrival_s=graph.times[-1],
                state_s=graph.times[-1], x=state[6], y=state[7], z=state[8],
                yaw_rad=Rotation.from_matrix(rotation).as_euler('ZYX')[0],
                wall_s=elapsed, status=result['termination'], nodes=len(graph.times),
                ranges=len(graph.observed), framework_prior=framework,
                **{k:result[k] for k in ('initial_cost','final_cost','iterations',
                    'accepted_steps','rejected_steps','residual_calls')})


def run_episode(folder_string, output_string, mode, case_limit,jacobian='analytic',native_only_first=False):
    """Share immutable preintegrations across all independent priors of one episode."""
    folder, output = Path(folder_string), Path(output_string)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    graph, meta = load_graph(folder, 10.,jacobian)
    build_s = time.perf_counter()-started
    prior_sigma = np.r_[np.full(3, np.deg2rad(10.)), np.full(3, 3.),
                        np.full(3, 20.), np.full(3, .01), np.full(3, .1)]
    if mode in ('native','both'):
        initial = graph.initial_state(np.zeros(3), 0.)
        result = graph.solve(initial, prior_sigma)
        append(output/f'{folder.name}__native.csv', result_row(folder.name, 'native', graph, result,
               build_s+result['wall_s'], False))
        np.savez_compressed(output/f'{folder.name}__native10_states.npz', times=graph.times,
                            state=result['state'], R0=graph.r0)
        # Accumulating batch continuation differs from D-iSAM2 and is named as such.
        old_graph, old_result = graph, result
        for horizon in (() if native_only_first else (11.,12.,13.,14.,15.)):
            started = time.perf_counter()
            new_graph, _ = load_graph(folder, horizon,jacobian)
            initial = new_graph.warm_state(old_graph.times,old_result['state'])
            new_result = new_graph.solve(initial, prior_sigma)
            append(output/f'{folder.name}__native.csv', result_row(folder.name, 'native', new_graph,
                   new_result, time.perf_counter()-started, False))
            if horizon == 15.:
                np.savez_compressed(output/f'{folder.name}__native15_states.npz', times=new_graph.times,
                                    state=new_result['state'], R0=new_graph.r0)
            old_graph, old_result = new_graph, new_result
    if mode in ('stress','both'):
        priors = pd.read_csv(folder/'priors.csv')
        if case_limit:
            priors = priors.iloc[:case_limit]
        checkpoint = output/f'{folder.name}__stress.csv'
        done = set(pd.read_csv(checkpoint).case_id.astype(str)) if checkpoint.exists() else set()
        for _, case in priors.iterrows():
            if str(case.case_id) in done:
                continue
            sigma = prior_sigma.copy()
            sigma[2], sigma[6:9] = float(case.yaw_std_rad), float(case.pos_std_m)
            initial = graph.initial_state(case[['px','py','pz']].to_numpy(dtype=float), float(case.yaw_rad))
            case_started = time.perf_counter()
            try:
                result = graph.solve(initial, sigma, framework=True)
                append(checkpoint, result_row(folder.name, str(case.case_id), graph, result,
                       result['wall_s']+build_s, True))
            except Exception as error:
                row = dict.fromkeys(FIELDS, '')
                row.update(episode=folder.name, case_id=str(case.case_id), arrival_s=10.,state_s=10.,
                           x=np.nan,y=np.nan,z=np.nan,yaw_rad=np.nan,framework_prior=True,
                           wall_s=time.perf_counter()-case_started+build_s,
                           status=type(error).__name__+': '+str(error),nodes=len(graph.times),ranges=len(graph.observed))
                append(checkpoint,row)
    return folder.name


def main():
    """Dispatch episodes independently while retaining all finite poor and capped fits."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--streams', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=5)
    parser.add_argument('--mode', choices=['native','stress','both'], default='both')
    parser.add_argument('--episode', action='append')
    parser.add_argument('--case-limit', type=int, default=0)
    parser.add_argument('--jacobian',choices=['analytic','numeric'],default='analytic')
    parser.add_argument('--native-only-first',action='store_true')
    args = parser.parse_args()
    folders = sorted(p.parent for p in args.streams.glob('*/meta.json'))
    if args.episode:
        folders = [p for p in folders if p.name in args.episode]
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = dict(method='independent_Jiang_style_batch_graph', author_code=False,
                    exact_D_iSAM2=False, native_seed_p=[0.,0.,0.], native_seed_yaw_rad=0.,
                    first_output_data_s=10., continuation_data_s=([] if args.native_only_first else [11.,12.,13.,14.,15.]),
                    rotation_update='left_Exp_delta_R',angle_storage='principal_rotation_vector',
                    configuration=asdict(GraphConfig(jacobian=args.jacobian)),range_gamma_over_sigma=2.,
                    range_mixture='equal_prior_max_Gaussian_positive_half_Cauchy',
                    native_pose_prior=False,stress_framework_pose_prior=True,
                    bias_prior_gyro_std=.01,bias_prior_accel_std=.1,
                    framework_tilt_prior=False,framework_velocity_prior=False,
                    workers=min(8,args.workers),blas_threads=1,random_seed=None,
                    input_directory=str(args.streams.resolve()),truth_read_by_estimator=False)
    (args.output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n',encoding='utf-8')
    with ProcessPoolExecutor(max_workers=min(8,args.workers)) as pool:
        futures = {pool.submit(run_episode, str(p), str(args.output), args.mode, args.case_limit,
                    args.jacobian,args.native_only_first):p.name for p in folders}
        for future in as_completed(futures):
            try:
                print('COMPLETED', future.result(), flush=True)
            except Exception as error:
                print('FAILED', futures[future], repr(error), flush=True)
    for mode in ('native','stress'):
        files = sorted(args.output.glob(f'*__{mode}.csv'))
        if files:
            pd.concat([pd.read_csv(p) for p in files]).to_csv(args.output/f'{mode}.csv',index=False)


if __name__ == '__main__':
    main()
