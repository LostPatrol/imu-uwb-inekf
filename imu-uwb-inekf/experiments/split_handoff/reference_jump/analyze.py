"""Summarize all controlled restart cases using actual decision-time errors and paired futures."""

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Analysis reads only this experiment's completed post-generation scores.
HERE = Path(__file__).resolve().parent
POLICIES = ('retain', 'gated', 'ungated_same_candidate', 'full_attitude')


def main():
    """Retain every nominal/perturbed case, and compare policies from identical shadow states."""
    output = HERE/'results'
    frame = pd.read_csv(output/'metrics.csv')
    keys = ['name', 'scenario', 'jump_deg']
    paired = frame.pivot(index=keys, columns='policy', values='future5_xy_rmse')
    meta = frame[frame.policy == 'retain'].set_index(keys)
    paired['actual_shadow_yaw_error_deg'] = meta.shadow_yaw_rmse_deg
    paired['actual_jumped_yaw_error_deg'] = meta.jumped_yaw_rmse_deg
    paired['accepted'] = meta.heading_accepted
    for policy in POLICIES[1:]:
        paired['retain_minus_'+policy] = paired['retain']-paired[policy]
    paired.to_csv(output/'paired_future5_xy.csv')
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for policy in POLICIES:
        rows = frame[frame.policy == policy]
        axes[0].scatter(rows.jumped_yaw_rmse_deg, rows.future5_xy_rmse, label=policy, alpha=.75)
        if policy != 'retain':
            axes[1].scatter(paired.actual_jumped_yaw_error_deg, paired['retain_minus_'+policy], label=policy, alpha=.75)
        axes[2].scatter(rows.jumped_yaw_rmse_deg, rows.future5_yaw_rmse_deg, label=policy, alpha=.75)
    axes[0].set_ylabel('Next 5 s XY RMSE (m)')
    axes[0].set_yscale('log')
    axes[1].set_ylabel('Retain minus policy XY RMSE (m)\nPositive means improvement')
    axes[1].axhline(0., color='black', linewidth=.7)
    axes[1].set_yscale('symlog', linthresh=.01)
    axes[2].set_ylabel('Next 5 s yaw RMSE (deg)')
    for axis in axes:
        axis.set_xlabel('Actual jumped yaw error at decision (deg)')
        axis.grid(alpha=.2)
        axis.legend(fontsize=7)
    fig.suptitle('Trial7 controlled coordinate loss at decision: all nominal and perturbed cases')
    fig.tight_layout()
    fig.savefig(output/'controlled_reference_jump.png', dpi=160)
    fields = ['jumped_yaw_rmse_deg', 'handoff_yaw_rmse_deg', 'future5_xy_rmse',
              'future5_z_rmse', 'future5_yaw_rmse_deg', 'future15_xy_rmse',
              'future15_z_rmse', 'future15_yaw_rmse_deg']
    print(frame.groupby(['scenario', 'jump_deg', 'policy'])[fields].median().round(3).to_string())
    print('\nAcceptance: '+str(int(meta.heading_accepted.sum()))+'/'+str(len(meta)))
    print('Failed trajectories: '+str(int(frame.failed.sum()))+'/'+str(len(frame)))


if __name__ == '__main__':
    main()
