"""Render development-only heading attribution and compare distinct angle metrics."""

from pathlib import Path
import audit
import numpy as np
import pandas as pd
from scipy.interpolate import splrep,splev
from scipy.spatial.transform import Rotation,Slerp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# The plotting cohort is the fixed development pair, never independent trial2.
HERE=Path(__file__).resolve().parent
METHODS=("author_original","production_proposed","production_full_geometry","research_util")
LABELS=("Untouched author","Actual proposed class","Proposed: complete H only","Standalone research kernel")
COLORS=("#704c2d","#206e9a","#b05270","#548a54")


def main():
    """Keep log-vector component errors separate from wrapped yaw and relative SO3."""
    output=HERE/"mechanism"
    output.mkdir(parents=True,exist_ok=True)
    figure,axes=plt.subplots(2,2,figsize=(12,7))
    comparisons=[]
    for column,name in enumerate(audit.NAMES):
        data=audit.native.load_raw(name,False)
        truth=data["truth"]
        origin=data["common_origin"]
        true_R=Rotation.from_quat(truth[:,4:8])
        interpolator=Slerp(truth[:,0],true_R)
        logs=np.array([audit.native.ours.author.SO3_log(R) for R in true_R.as_matrix()])
        splines=[splrep(truth[:,0],logs[:,index],s=.5) for index in range(3)]
        for yaw in audit.YAWS:
            for method,label,color in zip(METHODS,LABELS,COLORS):
                rows=np.load(HERE/"diagnostic"/(f"{name}__yaw{yaw:g}__{method}.npz"))["rows"]
                grid=np.arange(origin+5.,origin+44.95+1e-9,.05)
                indices=np.searchsorted(rows[:,0],grid+1e-9,side="right")-1
                chosen=rows[indices].copy()
                chosen[:,0]=grid
                estimated=Rotation.from_matrix(chosen[:,7:16].reshape(-1,3,3))
                reference=interpolator(grid)
                yaw_error=estimated.as_euler("xyz")[:,2]-reference.as_euler("xyz")[:,2]
                yaw_error=np.degrees(np.arctan2(np.sin(yaw_error),np.cos(yaw_error)))
                estimated_logs=np.array([audit.native.ours.author.SO3_log(R) for R in estimated.as_matrix()])
                reference_logs=np.column_stack([splev(grid,spline) for spline in splines])
                log_error=estimated_logs-reference_logs
                comparisons.append(dict(name=name,yaw_offset_deg=yaw,method=method,
                    wrapped_yaw_rmse_deg=float(np.sqrt(np.mean(yaw_error**2))),
                    relative_so3_rmse_deg=float(np.degrees(np.sqrt(np.mean((reference.inv()*estimated).magnitude()**2)))),
                    paper_style_log_z_rms_rad=float(np.sqrt(np.mean(log_error[:,2]**2))),
                    paper_style_log_vector_rms_rad=float(np.sqrt(np.mean(np.sum(log_error**2,axis=1))))))
                if yaw==180.:
                    reference_xy=np.column_stack([np.interp(grid,truth[:,0],truth[:,index]) for index in (1,2)])
                    axes[0,column].plot(grid-origin,np.linalg.norm(chosen[:,1:3]-reference_xy,axis=1),label=label,color=color,linewidth=1)
                    axes[1,column].plot(grid-origin,yaw_error,label=label,color=color,linewidth=1)
        axes[0,column].set_title(name+"; initial yaw offset180°")
        axes[0,column].set_ylabel("World XY error (m)")
        axes[1,column].set_ylabel("Wrapped physical yaw error (deg)")
        axes[1,column].set_ylim(-185,185)
        for axis in axes[:,column]:
            axis.grid(alpha=.25)
            axis.set_xlabel("Seconds from sensor origin")
    axes[0,0].legend(fontsize=8)
    figure.suptitle("Development mechanism: one observation Jacobian switch reproduces heading loss\nSame source measurements, initial X/P and literal author native clock")
    figure.tight_layout()
    figure.savefig(output/"heading_mechanism.png",dpi=180)
    plt.close(figure)
    pd.DataFrame(comparisons).to_csv(output/"paper_style_metric_comparison.csv",index=False)
    print(pd.DataFrame(comparisons).to_string(index=False))


if __name__=="__main__":
    main()
