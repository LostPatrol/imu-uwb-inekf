"""Plot fixed synthetic full-circle challenges and optional public yaw decisions."""

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Process artifacts remain inside the authorized project directory.
OUTPUT = Path(__file__).resolve().parents[4]/'agent'/'codex'/'split_checks'


def main():
    """Keep both the weak flat cases and coherent system-error counterexample visible."""
    cases = ('strong', 'weak', 'constant_velocity', 'constant_acceleration', 'near_180',
             'coplanar_weak_z', 'rotating_bias', 'wrong_bias_model', 'tdoa_strong',
             'systematic_yaw_alias', 'link_offsets', 'tdoa_link_offsets', 'bias_only_excitation')
    fig, axes = plt.subplots(5, 3, figsize=(13, 15))
    for name, axis in zip(cases, axes.flat):
        values = np.load(OUTPUT/(name+'.npz'))
        angles = np.degrees(values['angles'])
        costs = values['costs']
        ideal = values['ideal_costs']
        axis.plot(angles, costs-costs.min(), label='noisy')
        axis.plot(angles, ideal-ideal.min(), '--', label='noise free')
        true_yaw = 179. if name == 'near_180' else np.degrees(.7)
        axis.axvline(true_yaw, color='black', alpha=.5, linewidth=1)
        axis.set_title(name)
        axis.set_xlabel('candidate yaw (deg)')
        axis.set_ylabel('profile cost above minimum')
        axis.grid(alpha=.2)
        if name.startswith('constant_') or name == 'bias_only_excitation':
            axis.set_ylim(-.01, .1)
    for axis in axes.flat[len(cases):]:
        axis.set_visible(False)
    axes[0, 0].legend()
    fig.suptitle('Fixed synthetic evidence challenges: black line is physical yaw, not supplied to decide')
    fig.tight_layout()
    fig.savefig(OUTPUT/'profiles.png', dpi=150)


if __name__ == '__main__':
    main()
