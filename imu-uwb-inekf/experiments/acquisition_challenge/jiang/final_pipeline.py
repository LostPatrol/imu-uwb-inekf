"""Execute the uniform manifold repair on all predeclared episodes and pressure cases."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import pandas as pd

# All sixteen native episodes and all eight recovery matrices are fixed in advance.
NATIVE_EPISODES=tuple(f'delft_{motion}_{kind}' for motion in
    ('triangle','octagon','hourglass','random','star','square') for kind in ('start','recovery'))+(
    'isas_walk2_start','isas_walk2_recovery','isas_walk3_start','isas_walk3_recovery')
STRESS_EPISODES=('isas_walk2_recovery','isas_walk3_recovery')+tuple(
    f'delft_{motion}_recovery' for motion in ('triangle','octagon','hourglass','random','star','square'))


def execute(script,arguments,record=None):
    """Run one authorized phase using the same project interpreter."""
    command=[sys.executable,str(Path(__file__).parent/script),*map(str,arguments)]
    if record:
        result=subprocess.run(command,check=True,capture_output=True,text=True,encoding='utf-8')
        record.write_text(result.stdout,encoding='utf-8')
        print(result.stdout,flush=True)
    else:
        subprocess.run(command,check=True)


def compare_native(output,numeric_output):
    """Compare equivalent local numerical and analytic Jacobians at identical 10s seeds."""
    analytic=pd.read_csv(output/'native.csv').query('state_s == 10').copy()
    numeric=pd.read_csv(numeric_output/'native.csv').copy()
    joined=analytic.merge(numeric,on=['episode','case_id','state_s'],suffixes=('_analytic','_numeric'))
    joined['endpoint_difference_m']=np.linalg.norm(joined[['x_analytic','y_analytic','z_analytic']].to_numpy()-
                                                  joined[['x_numeric','y_numeric','z_numeric']].to_numpy(),axis=1)
    joined['yaw_difference_rad']=np.arctan2(np.sin(joined.yaw_rad_analytic-joined.yaw_rad_numeric),
                                            np.cos(joined.yaw_rad_analytic-joined.yaw_rad_numeric))
    joined['final_cost_difference']=joined.final_cost_analytic-joined.final_cost_numeric
    joined.to_csv(output/'native_jacobian_comparison.csv',index=False)


def main():
    """Prioritize native and full 10s matrices before continuation and isolated timing."""
    parser=argparse.ArgumentParser()
    parser.add_argument('--streams',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=6)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    execute('validate_jacobian.py',[],args.output/'manifold_validation.json')
    execute('smoke_check.py',[],args.output/'synthetic_factor_check.json')
    common=['--streams',args.streams,'--output',args.output,'--workers',args.workers]
    native_flags=[item for episode in NATIVE_EPISODES for item in ('--episode',episode)]
    stress_flags=[item for episode in STRESS_EPISODES for item in ('--episode',episode)]
    execute('run.py',common+['--mode','native']+native_flags)
    print('PHASE native16 finished',flush=True)
    execute('stress_parallel.py',common+stress_flags)
    print('PHASE stress768_10s finished',flush=True)
    numeric_output=args.output.parent/'native_numeric_manifold_reference'
    numeric_common=['--streams',args.streams,'--output',numeric_output,'--workers',args.workers]
    execute('run.py',numeric_common+['--mode','native','--jacobian','numeric','--native-only-first']+native_flags)
    compare_native(args.output,numeric_output)
    print('PHASE matched_native_jacobians finished',flush=True)
    execute('continue_stress.py',common+stress_flags)
    print('PHASE stress768_15s finished',flush=True)
    execute('timing_sample.py',['--streams',args.streams,'--output',args.output]+stress_flags)
    native=pd.read_csv(args.output/'native.csv')
    stress=pd.read_csv(args.output/'stress.csv')
    extended=pd.read_csv(args.output/'stress_10to15.csv')
    checks=dict(native_rows=len(native),native_episodes=native.episode.nunique(),
        stress_rows=len(stress),stress_episodes=stress.episode.nunique(),
        extended_rows=len(extended),wall_s=time.perf_counter()-started,
        expected_native_rows=96,expected_stress_rows=768,expected_extended_rows=4608,
        native_finite=int(np.isfinite(native[['x','y','z']]).all(axis=1).sum()),
        stress_finite=int(np.isfinite(stress[['x','y','z']]).all(axis=1).sum()),
        stress_termination=stress.status.value_counts().to_dict(),
        input_and_prior_selection_changed=False,truth_read_by_estimator=False)
    (args.output/'execution_checks.json').write_text(json.dumps(checks,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(checks,indent=2),flush=True)


if __name__=='__main__':
    main()
