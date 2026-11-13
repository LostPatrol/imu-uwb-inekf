"""Continue every frozen 10 s stress graph causally to 15 s without changing priors."""
from __future__ import annotations
import argparse
from concurrent.futures import as_completed
import gc
import json
import os
from pathlib import Path
import sys
import shutil
import time
for name in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[name]='1'
import numpy as np
import pandas as pd
from run import load_graph,result_row,append,FIELDS
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'split_handoff'))
from cpu_replay import replay_pool

# Per-process immutable graphs are reused across independent prior trajectories.
GRAPHS={}


def cached_graph(folder,horizon):
    """Cache one accumulated sensor graph and preserve its full construction cost."""
    key=(folder,horizon)
    if key not in GRAPHS:
        if GRAPHS and next(iter(GRAPHS))[0]!=folder:
            # Keep only the current episode's six horizons in each worker.
            GRAPHS.clear()
            gc.collect()
        started=time.perf_counter()
        graph,_=load_graph(Path(folder),horizon)
        GRAPHS[key]=(graph,time.perf_counter()-started)
    return GRAPHS[key]


def continue_case(folder_string,case,output_string,registered_endpoint):
    """Warm-start five causal graph updates while keeping the original supplied prior."""
    folder,output=Path(folder_string),Path(output_string)
    graph10,build10=cached_graph(folder_string,10.)
    sigma=np.r_[np.ones(2),float(case['yaw_std_rad']),np.ones(3),
                  np.full(3,float(case['pos_std_m'])),np.full(3,.01),np.full(3,.1)]
    original=graph10.initial_state(np.array([case['px'],case['py'],case['pz']]),float(case['yaw_rad']))
    prior_state=original[0].copy()
    state_path=output/'states10'/folder.name/(str(case['case_id'])+'.npz')
    restored=False
    restore_wall=0.
    if state_path.exists():
        state=np.load(state_path)['state']
    else:
        # The first ISAS run did not yet store full states; reproduce the same solve.
        restored=True
        result=graph10.solve(original,sigma,framework=True,prior_state=prior_state)
        state=result['state']
        restore_wall=build10+result['wall_s']
        state_path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(state_path,state=state)
    endpoint_delta=float(np.linalg.norm(state[-1,6:9]-np.asarray(registered_endpoint)))
    old_times=graph10.times
    rows=[]
    for horizon in (11.,12.,13.,14.,15.):
        graph,build_s=cached_graph(folder_string,horizon)
        started=time.perf_counter()
        initial=graph.warm_state(old_times,state)
        result=graph.solve(initial,sigma,framework=True,prior_state=prior_state)
        rows.append(result_row(folder.name,str(case['case_id']),graph,result,
                    build_s+time.perf_counter()-started,True))
        old_times,state=graph.times,result['state']
    endpoint_folder=output/'states15'/folder.name
    endpoint_folder.mkdir(parents=True,exist_ok=True)
    # Only endpoint state is required to audit all 768 final poses and bias estimates.
    np.savez_compressed(endpoint_folder/(str(case['case_id'])+'.npz'),state=state[-1],
                        prior_state=prior_state,resume_endpoint_delta_m=endpoint_delta)
    return rows,dict(episode=folder.name,case_id=str(case['case_id']),recomputed_10s_state=restored,
                    recompute_wall_s=restore_wall,resume_endpoint_delta_m=endpoint_delta)


def main():
    """Run all predeclared stress cases; retain divergence and cap statuses at every time."""
    parser=argparse.ArgumentParser()
    parser.add_argument('--streams',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--episode',action='append',required=True)
    parser.add_argument('--workers',type=int,default=5)
    args=parser.parse_args()
    registered=pd.read_csv(args.output/'stress.csv')
    jobs=[]
    for episode in args.episode:
        folder=args.streams/episode
        priors=pd.read_csv(folder/'priors.csv').to_dict('records')
        checkpoint=args.output/f'{episode}__continuation.csv'
        done=set(pd.read_csv(checkpoint).query('state_s == 15').case_id.astype(str)) if checkpoint.exists() else set()
        for case_index,case in enumerate(priors):
            if str(case['case_id']) in done:
                continue
            match=registered[(registered.episode==episode)&(registered.case_id==str(case['case_id']))]
            if len(match)!=1:
                raise ValueError(f'Expected one registered 10s result: {episode}/{case["case_id"]}')
            endpoint=match[['x','y','z']].iloc[0].to_numpy(dtype=float)
            jobs.append((str(folder),case,endpoint,case_index))
    # Complete a fixed head/tail subset across every episode before the remainder.
    # This allows an honest predeclared continuation subset if full runtime is excessive.
    priority_indices=list(range(12))+list(range(84,96))
    jobs.sort(key=lambda job:0 if job[3] in priority_indices else 1)
    protocol_path=args.output/'continuation_protocol.json'
    if protocol_path.exists():
        shutil.copyfile(protocol_path,args.output/f'continuation_protocol__previous_{time.time_ns()}.json')
    protocol_path.write_text(json.dumps(dict(
        episodes=args.episode,all_cases_requested=True,priority_indices=priority_indices,
        selection_uses_accuracy=False,horizons_s=[11,12,13,14,15],
        original_position_yaw_prior_retained=True,workers=args.workers,
        cache_policy='one_episode_per_worker',
        pool='split_handoff.cpu_replay.replay_pool',affinity='physical_P_core_first'),indent=2)+'\n',encoding='utf-8')
    started=time.perf_counter()
    audit_path=args.output/'continuation_resume_audit.csv'
    audit=pd.read_csv(audit_path).to_dict('records') if audit_path.exists() else []
    audit_keys={(str(row['episode']),str(row['case_id'])) for row in audit}
    for episode in args.episode:
        checkpoint=args.output/f'{episode}__continuation.csv'
        if not checkpoint.exists():
            continue
        done=pd.read_csv(checkpoint).query('state_s == 15').case_id.astype(str)
        for case_id in done:
            if (episode,case_id) in audit_keys:
                continue
            # A stop can occur between publishing a completed case and its audit row.
            state=np.load(args.output/'states10'/episode/(case_id+'.npz'))['state']
            reference=registered[(registered.episode==episode)&(registered.case_id==case_id)]
            difference=np.linalg.norm(state[-1,6:9]-reference[['x','y','z']].iloc[0].to_numpy())
            audit.append(dict(episode=episode,case_id=case_id,recomputed_10s_state=False,
                recompute_wall_s=0.,resume_endpoint_delta_m=float(difference),
                audit_recovered_after_checkpoint=True))
            audit_keys.add((episode,case_id))
    with replay_pool(len(jobs),args.workers) as pool:
        futures={pool.submit(continue_case,folder,case,str(args.output),endpoint):(Path(folder).name,case['case_id'])
                 for folder,case,endpoint,_ in jobs}
        for i,future in enumerate(as_completed(futures),1):
            episode,case_id=futures[future]
            try:
                rows,record=future.result()
                audit.append(record)
            except Exception as error:
                rows=[]
                for horizon in (11.,12.,13.,14.,15.):
                    row=dict.fromkeys(FIELDS,'')
                    row.update(episode=episode,case_id=case_id,arrival_s=horizon,state_s=horizon,
                        x=np.nan,y=np.nan,z=np.nan,yaw_rad=np.nan,status=repr(error),framework_prior=True)
                    rows.append(row)
                audit.append(dict(episode=episode,case_id=case_id,error=repr(error)))
            for row in rows:
                append(args.output/f'{episode}__continuation.csv',row)
            pd.DataFrame(audit).to_csv(args.output/'continuation_resume_audit.csv',index=False)
            if i%24==0 or i==len(jobs):
                print(f'continued {i}/{len(jobs)}; elapsed {time.perf_counter()-started:.1f}s',flush=True)
    files=sorted(args.output.glob('*__continuation.csv'))
    continuation=pd.concat([pd.read_csv(p) for p in files])
    continuation.to_csv(args.output/'stress_continuation.csv',index=False)
    pd.concat([registered,continuation]).sort_values(['episode','case_id','arrival_s']).to_csv(
        args.output/'stress_10to15.csv',index=False)
    pd.DataFrame(audit).to_csv(args.output/'continuation_resume_audit.csv',index=False)
    (args.output/'continuation_run.json').write_text(json.dumps(dict(cases_requested=len(jobs),
        elapsed_s=time.perf_counter()-started,workers=args.workers,blas_threads=1,
        horizons_s=[11,12,13,14,15],original_prior_retained=True,exact_time_node_copy=True),indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':
    main()
