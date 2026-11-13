"""Replay official SFUISE UTIL TDoA settings on the four shared raw recordings."""

from pathlib import Path
import json
import os
import subprocess
import time
import hashlib
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation, Slerp

# Independent artifacts; the sibling author core and exported sensor files are read-only.
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SHARED = HERE.parent / 'toronto_heading'
SCRATCH = ROOT / 'agent/codex/toronto_sfuise'
EXE = SCRATCH / 'sfuise_native.exe'
NAMES = [f'const{k}-trial4-tdoa2' + ('-traj1' if k == 4 else '') for k in range(1, 5)]
# Official UTIL wrapper downsampling threshold, gravity and gyro conversion constants.
IMU_INTERVAL = .01 - .0001
G = 9.81
DURATION = 15.
GRID = np.arange(0., DURATION + .001, .5)


def dump(path, value):
    """Write UTF-8 metadata with preserved failures."""
    path.write_text(json.dumps(value, indent=2, allow_nan=True), encoding='utf8')


def prepare(name):
    """Copy sensor channels only, convert SI and reproduce official IMU downsampling."""
    source = SHARED/'data'/name
    out = SCRATCH/name
    out.mkdir(parents=True, exist_ok=True)
    imu = pd.read_csv(source/'imu.csv')
    origin = float(imu.t.iloc[0])
    imu.t -= origin
    selected, last = [], -np.inf
    for i, stamp in enumerate(imu.t):
        if stamp-last > IMU_INTERVAL:
            selected.append(i)
            last = stamp
    imu = imu.iloc[selected].copy()
    imu.columns = ['t', 'ax', 'ay', 'az', 'gx', 'gy', 'gz']
    imu[['ax', 'ay', 'az']] *= G
    imu[['gx', 'gy', 'gz']] *= np.pi/180.
    imu[imu.t <= DURATION].to_csv(out/'imu.csv', index=False)
    radio = pd.read_csv(source/'radio.csv')
    radio.t -= origin
    radio[(radio.t >= 0) & (radio.t <= DURATION)].to_csv(out/'radio.csv', index=False)
    (out/'anchors.csv').write_bytes((source/'anchors.csv').read_bytes())
    meta = json.loads((source/'meta.json').read_text(encoding='utf8'))
    clocks = pd.read_csv(meta['raw_source'], usecols=['t_acc', 't_gyro'])
    acc, gyro = clocks.t_acc.dropna().to_numpy(), clocks.t_gyro.dropna().to_numpy()
    acc = acc[acc <= origin+DURATION]
    upper = np.clip(np.searchsorted(gyro, acc), 0, len(gyro)-1)
    buffer = float(np.max(np.maximum(0., gyro[upper]-acc)))
    return out, origin, buffer


def evaluate(name, origin, outputs, method):
    """Score latest computationally available held pose at common raw-stream elapsed times."""
    truth = pd.read_csv(SHARED/'data'/name/'truth.csv').to_numpy()
    truth = truth[np.r_[True, np.diff(truth[:, 0]) > 0]]
    times = origin+GRID
    gt = np.column_stack([np.interp(times, truth[:, 0], truth[:, i]) for i in (1, 2, 3)])
    attitude = Slerp(truth[:, 0], Rotation.from_quat(truth[:, 4:8]))(np.clip(times, truth[0, 0], truth[-1, 0])).as_matrix()
    gt_yaw = np.arctan2(attitude[:, 1, 0], attitude[:, 0, 0])
    idx = np.searchsorted(outputs[:, 1], GRID, side='right')-1
    valid = idx >= 0
    xy, yaw, age = [np.full(len(GRID), np.nan) for _ in range(3)]
    if np.any(valid):
        held = outputs[idx[valid]]
        xy[valid] = np.linalg.norm(held[:, 2:4]-gt[valid, :2], axis=1)
        delta = held[:, 5]-gt_yaw[valid]
        yaw[valid] = np.degrees(np.abs(np.arctan2(np.sin(delta), np.cos(delta))))
        age[valid] = GRID[valid]-held[:, 0]
    row = dict(name=name, method=method, first_state_s=float(outputs[0, 0]), first_available_s=float(outputs[0, 1]), grid_samples_available=int(valid.sum()), max_age_s=float(np.nanmax(age)))
    for threshold in (.25, .5, 1.):
        good = np.isfinite(xy) & (xy <= threshold)
        confirmed = [i for i in range(2, len(GRID)) if np.all(good[i-2:i+1])]
        row[f'capture_{threshold}m_confirm_s'] = float(GRID[confirmed[0]]) if confirmed else None
        row[f'fraction_within_{threshold}m'] = float(np.mean(good))
        row[f'continuous_after_capture_{threshold}m'] = bool(np.all(good[confirmed[0]-2:])) if confirmed else False
    for endpoint in (10, 15):
        i = int(endpoint*2)
        row[f'xy_{endpoint}s_m'], row[f'yaw_{endpoint}s_deg'] = float(xy[i]), float(yaw[i])
    pd.DataFrame(dict(elapsed_s=GRID, available=valid, xy_m=xy, yaw_deg=yaw, age_s=age)).to_csv(HERE/f'{name}__{method}_grid.csv', index=False)
    return row


def main():
    """Run one single-thread process at a time and compare existing Oursland outputs."""
    env = os.environ.copy()
    env['PATH'] = str(ROOT/'agent/codex/sfuise_challenge/deps/Library/bin') + os.pathsep + env['PATH']
    for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        env[key] = '1'
    rows, audit = [], []
    for name in NAMES:
        target, origin, buffer = prepare(name)
        command = [str(EXE)]
        args = dict(imu=target/'imu.csv', radio=target/'radio.csv', anchors=target/'anchors.csv', output=target/'native.csv', episode=name, case='native_zero', duration=DURATION, mode='native', lx=-.012, ly=.001, lz=.091)
        for key, value in args.items():
            command.extend(['--'+key, str(value)])
        before = time.perf_counter()
        process = subprocess.run(command, env=env, capture_output=True, text=True, timeout=150)
        audit.append(dict(name=name, returncode=process.returncode, stdout=process.stdout, stderr=process.stderr, wall_s=time.perf_counter()-before, gyro_buffer_s=buffer, origin_s=origin, command=command))
        dump(HERE/'execution.json', audit)
        if process.returncode:
            continue
        frame = pd.read_csv(target/'native.csv')
        # Service is serial: every optimization waits for the latest arrived packet.
        available, previous_wall, service = [], 0., 0.
        for record in frame.itertuples():
            service = max(service, record.arrival_s+buffer) + max(0., record.wall_s-previous_wall)
            available.append(service)
            previous_wall = record.wall_s
        frame['available_s'] = available
        frame.to_csv(target/'native_available.csv', index=False)
        out = frame[['state_s', 'available_s', 'x', 'y', 'z', 'yaw_rad']].to_numpy()
        rows.append(evaluate(name, origin, out, 'SFUISE_native'))
        # Parent's Oursland replay is unchanged; shift its launch clock to raw origin.
        checkpoint = SHARED/'results'/(name+'__sensor_bootstrap')
        record = json.loads(checkpoint.with_suffix('.json').read_text(encoding='utf8'))
        arrays = np.load(checkpoint.with_suffix('.npz'))
        core = arrays['outputs_unbuffered'] if 'outputs_unbuffered' in arrays.files else arrays['outputs']
        out = core[:, :6].copy()
        out[:, :2] += record['start_absolute_s']-origin
        out[:, 1] += buffer
        rotations = core[:, 8:17].reshape(-1, 3, 3)
        out[:, 5] = np.arctan2(rotations[:, 1, 0], rotations[:, 0, 0])
        rows.append(evaluate(name, origin, out, 'Oursland_sensor_prefix'))
        pd.DataFrame(rows).to_csv(HERE/'metrics.csv', index=False)
        print(name, rows[-2], flush=True)
    dump(HERE/'protocol.json', dict(common_raw_elapsed_s=True, duration_s=DURATION, oursland_prefix_s=1.5, sfuise_prefix='native early publication permitted', sfuise_config='official config_test_util.yaml', lever_sfuise_m=[-.012, .001, .091], lever_oursland_m=[-.01245, .00127, .0908], truth_read_by_native=False, gyro_buffer='fixed max future gyro wait over raw first15s', availability='serial compute queue with wall time deltas; outputs never backfilled', sfuise_portability='sole first spline endpoint repeated for absent knot[1]; audited sibling generated core copied unchanged', covariance='native zero only; no external prior factors'))
    manifest = {}
    for path in [EXE, HERE/'native_cli.cpp', HERE/'run.py', *sorted((SCRATCH/'generated').rglob('*.h'))]:
        manifest[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    dump(HERE/'hashes.json', manifest)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 4, figsize=(15, 6), sharex=True)
    for j, name in enumerate(NAMES):
        for method in ('SFUISE_native', 'Oursland_sensor_prefix'):
            grid = pd.read_csv(HERE/f'{name}__{method}_grid.csv')
            axes[0, j].plot(grid.elapsed_s, grid.xy_m, label=method)
            axes[1, j].plot(grid.elapsed_s, grid.yaw_deg)
        axes[0, j].set_title(name.split('-trial')[0])
        axes[0, j].axhline(.5, color='gray', linestyle='--')
        axes[1, j].set_xlabel('Common raw-stream elapsed (s)')
        for ax in axes[:, j]:
            ax.grid(alpha=.3)
    axes[0, 0].set_ylabel('Held current-time XY error (m)')
    axes[1, 0].set_ylabel('True yaw error (deg)')
    axes[0, 0].legend()
    fig.suptitle('Toronto native single-tag TDoA2; no pose alignment or position prior')
    fig.tight_layout()
    fig.savefig(HERE/'overview.png', dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    main()
