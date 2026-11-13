"""Persist causal-boundary and immutable sensor provenance checks for four starts."""

import hashlib
import json
import sys
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
import run


def main():
    """Check finite native output, clocks and input provenance without changing estimates."""
    checks = []
    for name in run.NAMES:
        directory = run.SCRATCH/name
        frame = pd.read_csv(directory/'native_available.csv')
        inputs = {}
        for filename in ('imu.csv', 'radio.csv', 'anchors.csv', 'truth.csv'):
            path = run.SHARED/'data'/name/filename
            inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        base = np.load(run.SHARED/'results'/(name+'__sensor_bootstrap.npz'))
        oursland = base['outputs_unbuffered'] if 'outputs_unbuffered' in base.files else base['outputs']
        checks.append(dict(name=name, published=len(frame), finite=bool(np.isfinite(frame.select_dtypes('number')).all().all()), state_not_after_arrival=bool((frame.state_s <= frame.arrival_s+1e-9).all()), available_not_before_arrival=bool((frame.available_s >= frame.arrival_s).all()), availability_monotonic=bool((np.diff(frame.available_s) > 0).all()), sensor_available_not_before_endpoint=bool((frame.sensor_available_s >= frame.arrival_s).all()), sensor_availability_monotonic=bool((np.diff(frame.sensor_available_s) > 0).all()), shared_sensor_hashes=inputs, oursland_unbuffered_array_sha256=hashlib.sha256(oursland.tobytes()).hexdigest()))
    (run.HERE/'validation.json').write_text(json.dumps(checks, indent=2), encoding='utf8')
    assert all(all(row[key] for key in ('finite', 'state_not_after_arrival', 'available_not_before_arrival', 'availability_monotonic', 'sensor_available_not_before_endpoint', 'sensor_availability_monotonic')) for row in checks)
    protocol = json.loads((run.HERE/'protocol.json').read_text(encoding='utf8'))
    assert protocol['lever_sfuise_m'] == protocol['lever_oursland_m'] == run.SHARED_LEVER
    execution = json.loads((run.HERE/'execution.json').read_text(encoding='utf8'))
    for row in execution:
        args = row['command']
        assert [float(args[args.index('--'+axis)+1]) for axis in ('lx', 'ly', 'lz')] == run.SHARED_LEVER
        directory = run.SCRATCH/row['name']
        for filename in ('imu.csv', 'radio.csv', 'anchors.csv'):
            assert (directory/filename).read_bytes() == (run.SCRATCH/'native_config'/row['name']/filename).read_bytes()
    print('PASS: four finite native streams, 149 publications each, causal monotonic clocks')


if __name__ == '__main__':
    main()
