"""Verify coverage, causal timestamps, exact radio counts, and manifold storage without truth."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    """Audit every registered output and preserve failures in the reported denominators."""
    parser=argparse.ArgumentParser()
    parser.add_argument('--streams',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    native=pd.read_csv(args.output/'native.csv')
    stress=pd.read_csv(args.output/'stress.csv')
    continuation=pd.read_csv(args.output/'stress_continuation.csv')
    extended=pd.read_csv(args.output/'stress_10to15.csv')
    rows=[]
    for episode in native.episode.unique():
        radio=pd.read_csv(args.streams/episode/'radio.csv')
        for horizon in (10.,11.,12.,13.,14.,15.):
            expected=int(((radio.t>=0.)&(radio.t<=horizon)).sum())
            selected=native[(native.episode==episode)&(native.state_s==horizon)]
            rows.append(dict(episode=episode,method='native',horizon_s=horizon,
                rows=len(selected),expected_rows=1,expected_range_count=expected,
                all_ranges_preserved=bool((selected.ranges==expected).all())))
    for episode in stress.episode.unique():
        priors=pd.read_csv(args.streams/episode/'priors.csv')
        radio=pd.read_csv(args.streams/episode/'radio.csv')
        for horizon in (10.,11.,12.,13.,14.,15.):
            selected=extended[(extended.episode==episode)&(extended.state_s==horizon)]
            expected=int(((radio.t>=0.)&(radio.t<=horizon)).sum())
            rows.append(dict(episode=episode,method='pressure',horizon_s=horizon,
                rows=len(selected),expected_rows=len(priors),expected_range_count=expected,
                all_ranges_preserved=bool((selected.ranges==expected).all())))
    coverage=pd.DataFrame(rows)
    coverage.to_csv(args.output/'coverage_and_range_audit.csv',index=False)
    audit=pd.read_csv(args.output/'continuation_resume_audit.csv')
    native_angles=[]
    for file in args.output.glob('*__native*_states.npz'):
        native_angles.append(float(np.linalg.norm(np.load(file)['state'][:,:3],axis=1).max()))
    checks=dict(native_rows=len(native),native_episodes=native.episode.nunique(),
        pressure_rows=len(stress),pressure_episodes=stress.episode.nunique(),
        continuation_rows=len(continuation),extended_rows=len(extended),
        pressure_unique=len(stress.drop_duplicates(['episode','case_id'])),
        extended_unique=len(extended.drop_duplicates(['episode','case_id','state_s'])),
        complete_registered_grid=bool((coverage.rows==coverage.expected_rows).all()),
        all_common_radio_preserved=bool(coverage.all_ranges_preserved.all()),
        no_publication_before_10s=bool((native.arrival_s>=10.).all() and (extended.arrival_s>=10.).all()),
        native_finite=int(np.isfinite(native[['x','y','z']]).all(axis=1).sum()),
        pressure_finite=int(np.isfinite(stress[['x','y','z']]).all(axis=1).sum()),
        extended_finite=int(np.isfinite(extended[['x','y','z']]).all(axis=1).sum()),
        pressure_termination=stress.status.value_counts().to_dict(),
        extended_termination=extended.status.value_counts().to_dict(),
        maximum_resume_endpoint_delta_m=float(audit.resume_endpoint_delta_m.max()),
        maximum_native_stored_rotation_rad=max(native_angles),
        truth_read_by_estimator=False)
    checks['passed']=bool(checks['complete_registered_grid'] and checks['all_common_radio_preserved'] and
        checks['no_publication_before_10s'] and len(native)==96 and len(stress)==768 and
        len(extended)==4608 and checks['extended_unique']==4608 and max(native_angles)<np.pi+1e-7)
    (args.output/'execution_checks.json').write_text(json.dumps(checks,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(checks,indent=2))
    assert checks['passed']


if __name__=='__main__':
    main()
