"""Parallelize independent prior cases while reusing immutable inertial graphs."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import time
import hashlib
from dataclasses import asdict
for name in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[name] = '1'
import numpy as np
import pandas as pd
from run import load_graph, result_row, append, FIELDS
from factor_graph import GraphConfig

# A process owns its own graph cache; no graph is mutated by an optimization.
GRAPHS = {}


def run_case(folder_string, case, state_output):
    """Solve one supplied prior using a process-local preintegration cache."""
    started = time.perf_counter()
    if folder_string not in GRAPHS:
        graph, _ = load_graph(Path(folder_string), 10.)
        GRAPHS[folder_string] = (graph, time.perf_counter()-started)
    graph, build_s = GRAPHS[folder_string]
    sigma = np.r_[np.full(2, np.deg2rad(10.)), float(case['yaw_std_rad']),
                  np.full(3,3.),np.full(3,float(case['pos_std_m'])),
                  np.full(3,.01),np.full(3,.1)]
    initial = graph.initial_state(np.array([case['px'],case['py'],case['pz']]),float(case['yaw_rad']))
    try:
        result = graph.solve(initial,sigma,framework=True)
        state_folder=Path(state_output)/Path(folder_string).name
        state_folder.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(state_folder/(str(case['case_id'])+'.npz'),state=result['state'])
        return result_row(Path(folder_string).name,str(case['case_id']),graph,result,
                          result['wall_s']+build_s,True)
    except Exception as error:
        row = dict.fromkeys(FIELDS,'')
        row.update(episode=Path(folder_string).name,case_id=str(case['case_id']),
                   arrival_s=10.,state_s=10.,x=np.nan,y=np.nan,z=np.nan,yaw_rad=np.nan,
                   framework_prior=True,wall_s=time.perf_counter()-started+build_s,
                   status=type(error).__name__+': '+str(error),nodes=len(graph.times),ranges=len(graph.observed))
        return row


def main():
    """Checkpoint every case in the parent process, including capped and failed fits."""
    parser=argparse.ArgumentParser()
    parser.add_argument('--streams',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--episode',action='append',required=True)
    parser.add_argument('--workers',type=int,default=5)
    parser.add_argument('--case-limit',type=int,default=0)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    manifest={}
    for episode in args.episode:
        manifest[episode]={name:hashlib.sha256((args.streams/episode/name).read_bytes()).hexdigest()
            for name in ('meta.json','imu.csv','radio.csv','anchors.csv','priors.csv')}
    protocol=dict(method='independent_Jiang_style_batch_graph',author_code=False,
                  configuration=asdict(GraphConfig()),prior_factors=['position_xyz','wrapped_physical_yaw'],
                  tilt_prior=False,velocity_prior=False,source_of_priors='common priors.csv',
                  bias_prior_gyro_std=.01,bias_prior_accel_std=.1,range_gamma_over_sigma=2.,
                  requested_episodes=args.episode,input_sha256=manifest,blas_threads=1,
                  first_output_data_s=10.,stress_trajectory_continuation=False,
                  same_full_range_imu_graph_as_native=True,truth_read_by_estimator=False)
    run_key=hashlib.sha256('|'.join(args.episode).encode('utf-8')).hexdigest()[:12]
    protocol_path=args.output/f'stress_protocol__{run_key}.json'
    protocol_path.write_text(json.dumps(protocol,indent=2)+'\n',encoding='utf-8')
    jobs=[]
    for episode in args.episode:
        folder=args.streams/episode
        priors=pd.read_csv(folder/'priors.csv')
        if args.case_limit:
            priors=priors.iloc[:args.case_limit]
        checkpoint=args.output/f'{episode}__stress.csv'
        done=set(pd.read_csv(checkpoint).case_id.astype(str)) if checkpoint.exists() else set()
        jobs.extend((str(folder),row) for row in priors.to_dict('records') if str(row['case_id']) not in done)
    started=time.perf_counter()
    with ProcessPoolExecutor(max_workers=min(8,args.workers)) as pool:
        futures={pool.submit(run_case,folder,case,str(args.output/'states10')):(Path(folder).name,case['case_id']) for folder,case in jobs}
        for i,future in enumerate(as_completed(futures),1):
            episode,case_id=futures[future]
            try:
                row=future.result()
            except Exception as error:
                row=dict.fromkeys(FIELDS,'')
                row.update(episode=episode,case_id=case_id,arrival_s=10.,state_s=10.,
                           x=np.nan,y=np.nan,z=np.nan,yaw_rad=np.nan,status=repr(error),framework_prior=True)
            append(args.output/f'{episode}__stress.csv',row)
            if i%12==0 or i==len(jobs):
                print(f'completed {i}/{len(jobs)}; elapsed {time.perf_counter()-started:.1f}s',flush=True)
    files=sorted(args.output.glob('*__stress.csv'))
    if files:
        pd.concat([pd.read_csv(p) for p in files]).to_csv(args.output/'stress.csv',index=False)
    (args.output/f'stress_run__{run_key}.json').write_text(json.dumps(dict(cases_requested=len(jobs),
         elapsed_s=time.perf_counter()-started,workers=min(8,args.workers),blas_threads=1),indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':
    main()
