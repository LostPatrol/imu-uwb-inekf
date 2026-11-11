"""Check matched evidence, export small review tables and draw the two main real-data outcomes."""

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# All source paths point to outputs produced by this study, never historical status reports.
HERE = Path(__file__).resolve().parent
# Report real physical windows once, without duplicating gravity-frame rotations.
KEY = ['sequence', 'layout', 'end_s', 'window_s']


def draw(logs, external, path):
    """Compare paired geometry calibration by flight with true body accuracy at the second site."""
    fig, (left,right) = plt.subplots(1,2,figsize=(10.1,3.7),gridspec_kw={'width_ratios':[1.15,1]})
    palette={'all8':'#216879','lower4':'#ba6134'}
    names=['triangle','octagon','hourglass','random','star','square']
    for layout,label in [('all8','8 anchors'),('lower4','4 low anchors')]:
        block=logs[(logs.trajectory=='BiasPrior') & (logs.layout==layout)].copy()
        block['motion']=block.sequence.str.extract(r'_twr_(\w+)\.csv$')[0]
        paired=block.pivot(index='motion',columns='radio_model',values='rmse').loc[names]
        offset=-.11 if layout=='all8' else .11
        x=np.arange(len(names))+offset
        left.vlines(x,paired.static,paired.geometry,color=palette[layout],alpha=.65,linewidth=1.7)
        left.scatter(x,paired.static,s=45,facecolor='white',edgecolor=palette[layout],zorder=3)
        left.scatter(x,paired.geometry,s=32,color=palette[layout],zorder=4,label=label)
    left.set_xticks(range(len(names)),[x.title() for x in names],rotation=20,ha='right')
    left.set_ylabel('Endpoint position RMSE (m)')
    visible=logs[(logs.trajectory=='BiasPrior')&logs.radio_model.isin(['static','geometry'])]
    left.set_ylim(0,1.27*visible.rmse.max())
    left.set_title('A   Delft  |  TWR',loc='left',fontsize=11,fontweight='bold')
    left.legend(loc='upper right',fontsize=8,frameon=False)
    left.text(.01,.97,'Open: constant bias\nFilled: direction + distance',va='top',transform=left.transAxes,fontsize=8)
    sample=external[external.method=='projected_wahba']
    for key,marker,c in [('tdoa2','o','#216879'),('tdoa3','^','#ba6134')]:
        block=sample[sample.name.str.contains(key)]
        right.scatter(block.position_m,block.so3_deg,s=35,marker=marker,color=c,alpha=.9,label=key.upper().replace('TDOA','TDoA'))
    right.axhline(15,color='#777',linewidth=.9,linestyle=(0,(3,3)))
    right.text(.98,16.9,'15°',ha='right',fontsize=8,color='#555',transform=right.get_yaxis_transform())
    right.set_xlabel('Endpoint body position error (m)'); right.set_ylabel('Full attitude error (degrees)')
    right.set(xlim=(0,None),ylim=(0,None)); right.set_title('B   UTIL  |  moving restart',loc='left',fontsize=11,fontweight='bold')
    right.legend(loc='upper right',fontsize=8,frameon=False)
    for ax in (left,right):
        ax.grid(axis='y',color='#dddddd',linewidth=.6); ax.set_axisbelow(True)
        ax.spines[['top','right']].set_visible(False); ax.tick_params(labelsize=8)
    fig.tight_layout(w_pad=2.2); fig.savefig(path,dpi=230,bbox_inches='tight'); plt.close(fig)


def main():
    """Check result counts and training separation; retain capped or poor fits in every statistic."""
    out=HERE/'evidence'; out.mkdir(exist_ok=True)
    panel=pd.read_csv(HERE/'results_windows/windows.csv')
    restart=pd.read_csv(HERE/'results_restart/windows.csv')
    transfer=pd.read_csv(HERE/'results_geometry_transfer/windows.csv')
    external=pd.read_csv(HERE/'transfer_rotation/metrics.csv')
    joint=pd.read_csv(HERE/'joint_branch/results/overall_summary.csv')
    radio=pd.read_csv(HERE/'radio_branch/results/all_models_summary.csv')
    logs=pd.read_csv(HERE/'joint_branch/results/per_log_summary.csv')
    records=pd.read_csv(HERE/'results_windows/recordings.csv')
    assert not set(records[records.split=='evaluation'].file)&set(records[records.split=='calibration'].file)
    checks=[]
    for data,keys,label in [(panel,['mode','method'],'mean shape'),(restart,['mode','method'],'moving restart'),
                            (transfer,['mode','method','geometry'],'apparent geometry')]:
        assert not data.duplicated(list(dict.fromkeys(KEY+keys))).any()
        assert np.isfinite(data.error_m).all()
        assert (data.groupby(['layout']+keys).size()==66).all()
        checks.append(dict(panel=label,rows=len(data),capped=int((~data.success).sum())))
    # Independently generated equivalent control geometries must yield the same real endpoints.
    a=restart[restart.method=='procrustes'].set_index(KEY).sort_index().error_m
    b=transfer[(transfer.method=='so3')&(transfer.geometry=='survey')].set_index(KEY).sort_index().error_m
    assert a.index.equals(b.index) and np.max(np.abs(a-b))<1e-8
    a=panel[panel.method=='radio_spline'].set_index(KEY).sort_index().error_m
    b=transfer[(transfer.method=='radio_only')&(transfer.geometry=='survey')].set_index(KEY).sort_index().error_m
    assert a.index.equals(b.index) and np.max(np.abs(a-b))<1e-8
    assert len(external[external.method=='projected_wahba'])==16
    joint.to_csv(out/'twr_joint.csv',index=False); logs.to_csv(out/'twr_by_recording.csv',index=False)
    radio.to_csv(out/'radio_ablation.csv',index=False)
    pd.read_csv(HERE/'outage_branch/results/summary.csv').to_csv(out/'outage.csv',index=False)
    for name,source in [('excluded_motion','joint_branch/novel_motion/results/overall_summary.csv'),
                        ('simulated_attitude','shape_branch/cold_rotation_checks/results/aggregate.csv')]:
        pd.read_csv(HERE/source).to_csv(out/(name+'.csv'),index=False)
    pd.DataFrame(checks).to_csv(out/'verified_panels.csv',index=False)
    draw(logs,external,HERE.parents[2]/'tex'/'range_inertial_model_comparison.png')
    print('Physical-window keys, calibration separation, two independent score paths and finite outputs: OK')


if __name__=='__main__':
    main()
