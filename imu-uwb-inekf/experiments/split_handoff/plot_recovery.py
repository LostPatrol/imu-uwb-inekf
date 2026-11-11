"""Plot matched recovery gains by flight and by intentional heading perturbation."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Local result paths refer to the fixed schedules used by their respective experiments.
ROOT = Path(__file__).resolve().parent
# Stable colors distinguish the entire anchor geometry, never an error threshold.
COLORS = {'all8': '#007568', 'lower4': '#da743a'}


def main():
    """Write a standalone review figure without pooling repeated offsets as independent flights."""
    early = pd.read_csv(ROOT/'delft'/'early_results'/'staged_combined_paired.csv')
    jump = pd.read_csv(ROOT/'reference_jump'/'results'/'metrics.csv')
    methods = set(early.method)
    staged = next(m for m in methods if 'stag' in m and 'display' in m)
    data = early.pivot(index=['sequence','layout','yaw_offset_deg'], columns='method',values='early_xy_rmse_m')
    gain = (data['native_gate']-data[staged]).groupby(['sequence','layout']).median().unstack()
    figure, (a,b) = plt.subplots(1,2,figsize=(11,4.1), constrained_layout=True)
    y = np.arange(len(gain))
    for shift,layout,label in ((-.13,'all8','8 anchors'),(.13,'lower4','4 low anchors')):
        a.scatter(gain[layout],y+shift,label=label,c=COLORS[layout],s=47)
        a.hlines(y+shift,0,gain[layout],color=COLORS[layout],linewidth=1.4,alpha=.55)
    a.set_yticks(y,[s.split('_')[-1].split('.')[0] for s in gain.index]); a.invert_yaxis()
    a.set_xlabel('Reduction in 2 s horizontal RMSE (m)')
    a.set_title('Delft  |  early XY + later velocity',loc='left',weight='bold')
    a.legend(frameon=False,fontsize=9); a.axvline(0,color='#5f6c72',lw=.8)
    p = jump.pivot(index=['name','scenario','jump_deg'], columns='policy',
                   values=['future5_xy_rmse','future5_yaw_rmse_deg','heading_accepted'])
    dx = p['future5_yaw_rmse_deg']['retain'].astype(float)-p['future5_yaw_rmse_deg']['gated'].astype(float)
    dy = 100*(p['future5_xy_rmse']['retain'].astype(float)-p['future5_xy_rmse']['gated'].astype(float))
    mask = p['heading_accepted']['gated'].astype(str).str.lower().isin(['true','1','1.0'])
    b.axhline(0,color='#5f6c72',lw=.8); b.axvline(0,color='#5f6c72',lw=.8)
    b.scatter(dx[mask],dy[mask],marker='o',s=58,c='#007568',label=f'Updated: {mask.sum()}')
    b.scatter(dx[~mask],dy[~mask],s=60,marker='D',facecolors='none',edgecolors='#495d6b',
              label=f'Kept heading: {(~mask).sum()} at origin')
    b.set_title('UTIL  |  heading lost at handoff',loc='left',weight='bold')
    b.set_xlabel('Reduction in 5 s heading RMSE (degrees)')
    b.set_ylabel('Reduction in 5 s horizontal RMSE (cm)')
    b.legend(frameon=False,fontsize=9)
    for ax in (a,b):
        ax.grid(axis='x',color='#c9d3d5',alpha=.55)
        ax.set_axisbelow(True)
        for side in ('top','right'):
            ax.spines[side].set_visible(False)
    output = ROOT/'figures'; output.mkdir(exist_ok=True)
    figure.savefig(output/'recovery_comparison.png',dpi=180)
    plt.close(figure)


if __name__ == '__main__':
    main()
