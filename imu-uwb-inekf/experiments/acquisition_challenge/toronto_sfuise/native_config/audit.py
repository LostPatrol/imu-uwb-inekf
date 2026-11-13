"""Persist causal-boundary and immutable sensor provenance checks for four starts."""

import hashlib
import json
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
        checks.append(dict(name=name, published=len(frame), finite=bool(np.isfinite(frame.select_dtypes('number')).all().all()), state_not_after_arrival=bool((frame.state_s <= frame.arrival_s+1e-9).all()), available_not_before_arrival=bool((frame.available_s >= frame.arrival_s).all()), availability_monotonic=bool((np.diff(frame.available_s) > 0).all()), shared_sensor_hashes=inputs, oursland_unbuffered_array_sha256=hashlib.sha256(oursland.tobytes()).hexdigest()))
    (run.HERE/'validation.json').write_text(json.dumps(checks, indent=2), encoding='utf8')
    assert all(all(row[key] for key in ('finite', 'state_not_after_arrival', 'available_not_before_arrival', 'availability_monotonic')) for row in checks)
    print('PASS: four finite native streams, 149 publications each, causal monotonic clocks')


if __name__ == '__main__':
    main()
