"""Time fresh warm-code graphs on fixed, result-independent pressure case indices."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
for name in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[name]='1'
import numpy as np
import pandas as pd
from run import load_graph
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'split_handoff'))
from cpu_replay import replay_pool

# Chosen before observing timing or accuracy; four positions span the canonical matrix.
CASE_INDICES=(0,31,63,95)


def solve_case(graph,case):
    """Run the unchanged single-seed position/yaw-prior graph."""
    sigma=np.r_[np.ones(2),float(case.yaw_std_rad),np.ones(3),
                np.full(3,float(case.pos_std_m)),np.full(3,.01),np.full(3,.1)]
    initial=graph.initial_state(case[['px','py','pz']].to_numpy(dtype=float),float(case.yaw_rad))
    return graph.solve(initial,sigma,framework=True)


def warm_worker(folder_string):
    """Warm compiled code once inside the same P-core-preferred timing worker."""
    started=time.perf_counter()
    folder=Path(folder_string)
    graph,_=load_graph(folder,10.)
    solve_case(graph,pd.read_csv(folder/'priors.csv').iloc[0])
    return time.perf_counter()-started


def timed_case(folder_string,case,reference):
    """Measure a fresh preparation and solve on one unchanged registered case."""
    started=time.perf_counter()
    graph,_=load_graph(Path(folder_string),10.)
    prepare_s=time.perf_counter()-started
    result=solve_case(graph,case)
    elapsed=time.perf_counter()-started
    delta=np.linalg.norm(result['state'][-1,6:9]-np.asarray(reference))
    return dict(prepare_s=prepare_s,solve_s=result['wall_s'],fresh_total_s=elapsed,
        endpoint_delta_from_registered_m=float(delta),status=result['termination'],
        iterations=result['iterations'],final_cost=result['final_cost'])


def main():
    """Measure actual fresh preparation plus solve, with JIT warmup separately recorded."""
    parser=argparse.ArgumentParser()
    parser.add_argument('--streams',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--episode',action='append',required=True)
    args=parser.parse_args()
    first_folder=args.streams/args.episode[0]
    registered=pd.read_csv(args.output/'stress.csv')
    rows=[]
    with replay_pool(32,1) as pool:
        warmup_s=pool.submit(warm_worker,str(first_folder)).result()
        for episode in args.episode:
            folder=args.streams/episode
            priors=pd.read_csv(folder/'priors.csv')
            for index in CASE_INDICES:
                case=priors.iloc[index]
                reference=registered[(registered.episode==episode)&(registered.case_id==str(case.case_id))]
                result=pool.submit(timed_case,str(folder),case,reference[['x','y','z']].iloc[0].to_numpy()).result()
                rows.append(dict(episode=episode,case_id=case.case_id,case_index=index,**result))
                pd.DataFrame(rows).to_csv(args.output/'timing_warm_sample.csv',index=False)
                print(episode,case.case_id,f'{result["fresh_total_s"]:.3f}s',flush=True)
    (args.output/'timing_warm_protocol.json').write_text(json.dumps(dict(
        selection='fixed indices before observing results',case_indices=CASE_INDICES,
        episodes=args.episode,workers=1,blas_threads=1,warmup_s=warmup_s,
        full_fresh_preparation_each_timed_case=True,parameter_changes=False,
        pool='split_handoff.cpu_replay.replay_pool',affinity='P_core_first'),indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':
    main()
