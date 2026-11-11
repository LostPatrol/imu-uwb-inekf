"""Render matched window errors and local conditioning from saved transfer scores."""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Output is kept beside the complete machine-readable window scores.
ROOT = Path(__file__).resolve().parent


def main():
    """Plot every preselected window without filtering failures or fitting any alignment."""
    frame = pd.DataFrame(json.loads((ROOT/'metrics.json').read_text(encoding='utf-8')))
    methods = ['identity', 'window_gravity', 'projected_wahba', 'uniform4_gravity']
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    for method in methods:
        values = frame[frame.method == method]
        axes[0].plot(np.arange(len(values)), values.position_m, marker='.', label=method)
        axes[1].plot(np.arange(len(values)), values.so3_deg, marker='.', label=method)
    radio = frame[frame.method == 'radio_only_tag']
    axes[0].plot(np.arange(len(radio)), radio.position_m, ':', label='radio only (antenna target)')
    wahba = frame[frame.method == 'projected_wahba']
    axes[2].semilogy(np.arange(len(wahba)), [x[0] for x in wahba.rotation_info_free_bias],
                     marker='o', label='free constant bias')
    axes[2].semilogy(np.arange(len(wahba)), [x[0] for x in wahba.rotation_info_bias_prior],
                     marker='.', label='bias prior sigma=0.10 m/s2')
    axes[0].set_ylabel('Endpoint error (m)')
    axes[1].set_ylabel('SO(3) error (deg)')
    axes[2].set_ylabel('Min local rotation info\n(robust curvature, rad^-2)')
    labels = [name.replace('trial3-', '').replace('-traj1', '')+' / '+str(int(start))
              for name, start in zip(wahba.name, wahba.start)]
    axes[2].set_xticks(np.arange(len(labels)), labels, rotation=60, ha='right')
    for axis in axes:
        axis.grid(alpha=.25)
        axis.legend(loc='upper left', fontsize=8)
    fig.suptitle('UTIL trial3 transfer: fixed 6 s dynamic windows, no prior attitude or static prefix')
    fig.tight_layout()
    fig.savefig(ROOT/'transfer.png', dpi=160)
    frame.drop(columns=['force_singular', 'rotation_info_free_bias', 'rotation_info_bias_prior'], errors='ignore').to_csv(ROOT/'metrics.csv', index=False)


if __name__ == '__main__':
    main()
