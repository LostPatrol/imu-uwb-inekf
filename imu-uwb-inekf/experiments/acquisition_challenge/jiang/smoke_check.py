"""Check full-factor algebra on an unaligned synthetic moving trajectory."""
import json
import os
from pathlib import Path
for name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[name] = '1'
import numpy as np
from scipy.spatial.transform import Rotation
from factor_graph import BatchGraph, GraphConfig


def main():
    """Verify truth residuals and position acquisition from one wrong seed."""
    t = np.arange(0., 10.0001, .01)
    acc = np.tile([.15, .08, 9.81], (len(t), 1))
    gyro = np.zeros_like(acc)
    et = np.arange(.05, 10.0001, .05)
    anchors = np.array([[-5., -5., 0.], [5., -5., 0.], [5., 5., 3.], [-5., 5., 3.]])
    ea = anchors[np.arange(len(et)) % 4]
    position = np.array([1., -1., .5])+.5*np.outer(et**2, [.15, .08, 0.])
    ranges = np.linalg.norm(position-ea, axis=1)
    graph = BatchGraph(t, acc, gyro, et, ea, ranges, np.full(len(et), .15), np.eye(3), np.zeros(6), GraphConfig(max_iterations=60))
    initial = graph.initial_state([6., -1., .5], 0.)
    result = graph.solve(initial, np.r_[np.full(3, 2.), np.full(3, 3.), np.full(3, 20.), np.full(3, .01), np.full(3, .1)])
    expected = np.array([1., -1., .5])+.5*np.outer(graph.times**2, [.15, .08, 0.])
    error = np.linalg.norm(result['state'][:, 6:9]-expected, axis=1)
    summary = {k:v for k,v in result.items() if k not in ('state','times')}
    summary.update(position_max_m=float(error.max()), endpoint_m=float(error[-1]), nodes=len(graph.times))
    print(json.dumps(summary, indent=2))
    assert error.max() < .05
    assert np.linalg.norm(result['state'][-1, 9:15]) < .01


if __name__ == '__main__':
    main()
