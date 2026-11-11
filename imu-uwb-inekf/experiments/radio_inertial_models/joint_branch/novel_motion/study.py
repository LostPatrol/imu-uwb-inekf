"""Leave one motion out of radio calibration with a common bias-window solver.

Ridge is selected inside the five retained early recordings only. Six-recording
controls reuse the five-recording CV seeds to separate calibration from seeding.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

# Match the completed joint branch's single-thread numerical execution.
for _name in ("OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","OMP_NUM_THREADS"):
    os.environ[_name]="1"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load the completed joint definitions under a unique name without modifying them.
HERE=Path(__file__).resolve().parent
JOINT=HERE.parent
spec=importlib.util.spec_from_file_location("novel_motion_joint",JOINT/"study.py")
joint=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=joint
spec.loader.exec_module(joint)
radio=joint.radio
shape=joint.shape

# These are the same predeclared settings as the completed joint experiment.
RADIO_MODELS=("static","geometry")
SCOPES=("five_other_motions","six_all_motions")


def summarize(frame,output):
    """Preserve physical per-log comparisons and distinguish historical seed controls."""
    for groups,name in ((["calibration_scope","layout","radio_model"],"overall"),
        (["calibration_scope","sequence","layout","radio_model"],"per_log"),
        (["calibration_scope","duration_s","layout","radio_model"],"duration")):
        frame.groupby(groups).position_error_m.agg(n="size",median="median",p95=lambda x:x.quantile(.95),
            rmse=lambda x:np.sqrt(np.mean(x*x)),maximum="max").to_csv(output/(name+"_summary.csv"))
    keys=["sequence","endpoint_s","duration_s","layout"]
    paired=frame.pivot(index=keys,columns=["calibration_scope","radio_model"],values="position_error_m")
    paired.to_csv(output/"paired_metrics.csv")
    rows=[]
    for (sequence,layout),part in paired.groupby(level=["sequence","layout"]):
        for scope in SCOPES:
            static=part[(scope,"static")]
            geometry=part[(scope,"geometry")]
            difference=geometry-static
            rows.append(dict(sequence=sequence,layout=layout,calibration_scope=scope,
                geometry_wins=int((difference<0).sum()),windows=len(part),
                median_geometry_minus_static_m=difference.median(),
                rmse_static_m=np.sqrt(np.mean(static**2)),rmse_geometry_m=np.sqrt(np.mean(geometry**2)),
                p95_static_m=static.quantile(.95),p95_geometry_m=geometry.quantile(.95),
                worst_geometry_degradation_m=difference.max()))
    pd.DataFrame(rows).to_csv(output/"per_log_geometry_effect.csv",index=False)
    transfer=[]
    for model in RADIO_MODELS:
        for layout,part in paired.groupby(level="layout"):
            five=part[(SCOPES[0],model)]
            six=part[(SCOPES[1],model)]
            delta=five-six
            transfer.append(dict(layout=layout,radio_model=model,windows=len(part),
                five_better_endpoints=int((delta<0).sum()),median_five_minus_six_m=delta.median(),
                rmse_five_m=np.sqrt(np.mean(five**2)),rmse_six_m=np.sqrt(np.mean(six**2)),
                p95_five_m=five.quantile(.95),p95_six_m=six.quantile(.95)))
    pd.DataFrame(transfer).to_csv(output/"five_vs_six.csv",index=False)
    historical=pd.read_csv(JOINT/"results"/"metrics.csv")
    historical=historical[(historical.trajectory=="BiasPrior")&historical.radio_model.isin(RADIO_MODELS)].copy()
    historical["calibration_scope"]="historical_six_with_six_cv_seed"
    combined=pd.concat([frame,historical],ignore_index=True)
    combined.groupby(["calibration_scope","layout","radio_model"]).position_error_m.agg(
        n="size",median="median",p95=lambda x:x.quantile(.95),rmse=lambda x:np.sqrt(np.mean(x*x))).to_csv(output/"including_historical_summary.csv")
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,(layout,part) in zip(axes,paired.groupby(level="layout")):
        for model in RADIO_MODELS:
            ax.scatter(part[(SCOPES[1],model)],part[(SCOPES[0],model)],s=18,alpha=.7,label=model)
        ax.plot([.02,3],[.02,3],color="black",linewidth=.8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(layout)
        ax.grid(alpha=.2)
    axes[0].legend()
    fig.supxlabel("Six-motion calibration endpoint error (m), common five-motion CV seed")
    fig.supylabel("Five other motions calibration endpoint error (m)")
    fig.tight_layout()
    fig.savefig(output/"five_vs_six.png",dpi=180)
    plt.close(fig)


def main():
    """Fit only early recordings of other motions and score the fixed later windows."""
    output=HERE/"results"
    output.mkdir(parents=True,exist_ok=True)
    radio_results=JOINT.parent/"radio_branch"/"results"
    protocol=json.loads((radio_results/"protocol.json").read_text())
    raw_cal=json.loads((radio_results/"calibration.json").read_text())
    six_cal={key:np.asarray(value) if isinstance(value,list) else value for key,value in raw_cal.items()}
    training=pd.read_csv(radio_results/"training_events.csv")
    assert set(training.sequence.unique())==set(protocol["training"])
    anchors=radio.load_anchors(radio.DATASET)
    metadata=dict(training=protocol["training"],evaluation=protocol["evaluation"],
        scopes=SCOPES,radio_models=RADIO_MODELS,trajectory="BiasPrior",bias_prior_std_mps2=shape.BIAS_STD,
        starts=joint.STARTS,max_nfev_per_start=joint.MAX_NFEV,
        seed_policy="Every five/six calibration variant shares CV p/v from the five-other-motion static calibration, uniform four yaw starts, zero bias",
        calibration_policy="For each late target motion, exclude its early counterpart completely; choose ridge by leaving one of the remaining five early logs out",
        unchanged_ridge_grid=radio.RIDGES,common_sensor_delay_s=.20,static_prefix_s=1.5,
        truth_usage="Early training truth fits radio calibration; late truth valid timestamps crop the inherited loader; late positions only score endpoints",
        claim_scope="Cross-motion transfer in the same arena and date; no cross-site or independent blind claim",
        source_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in (
            Path(__file__),JOINT/"study.py",radio_results/"training_events.csv",radio_results/"calibration.json")})
    radio.dump(output/"protocol.json",metadata)
    rows,calibration_rows=[],[]
    for sequence in protocol["evaluation"]:
        motion=sequence.split("_twr_")[-1].removesuffix(".csv")
        retained=training[~training.sequence.str.endswith("_twr_"+motion+".csv")].copy()
        assert retained.sequence.nunique()==5
        assert not retained.sequence.str.endswith("_twr_"+motion+".csv").any()
        local_output=output/"calibrations"/motion
        local_output.mkdir(parents=True,exist_ok=True)
        five_cal=radio.calibrate(retained,local_output)
        calibration_rows.append(dict(evaluation=sequence,excluded_motion=motion,
            retained_early_logs=list(retained.sequence.unique()),ridge=five_cal["selected_ridge"]))
        # No evaluation positions are read until the target's calibration is frozen.
        log=radio.load_real_log(radio.DATASET,sequence,sensor_delay_s=.20)
        rotation,bias=radio.estimate_initial_rotation_and_bias(log)
        selected=log.time>=1.5
        times=log.time[selected]-log.time[selected][0]
        ranges,changed=log.ranges[selected],log.changed[selected]
        rotations,d,v=radio.preintegrate(times,log.imu[selected],rotation,bias)
        for endpoint in (5.,10.,20.,30.):
            for duration in (2.,4.,8.):
                if duration>endpoint:
                    continue
                end=np.searchsorted(times,endpoint,side="right")-1
                start=np.searchsorted(times,times[end]-duration,side="left")
                for layout in ("all8","lower4"):
                    mask=changed[start:end+1]&(ranges[start:end+1]>0)
                    if layout=="lower4":
                        mask[:,4:]=False
                    indices,ids=np.nonzero(mask)
                    indices+=start
                    tau=times[indices]-times[end]
                    displacement=d[indices]-d[end]-tau[:,None]*v[end]
                    sensitivity=shape.reanchor_integral(rotations[start:end],times[start:end+1],times[indices])
                    observed=ranges[indices,ids]
                    values=(tau,displacement,anchors[ids],observed,ids)
                    cv=radio.cv_candidate(tau,anchors[ids],observed-five_cal["static"][ids],five_cal["sigma"][ids])[0]
                    seeds=[np.r_[cv,angle,np.zeros(3)] for angle in np.linspace(-np.pi,np.pi,joint.STARTS,endpoint=False)]
                    truth=log.ground_truth_position[selected][end]
                    for scope,calibration in zip(SCOPES,(five_cal,six_cal)):
                        for model in RADIO_MODELS:
                            started=time.perf_counter()
                            x,fit=joint.solve(values,sensitivity,"BiasPrior",model,calibration,seeds)
                            rows.append(dict(sequence=sequence,motion=motion,endpoint_s=endpoint,
                                actual_endpoint_s=times[end],duration_s=duration,layout=layout,
                                calibration_scope=scope,radio_model=model,events=len(tau),
                                position_error_m=np.linalg.norm(x[:3]-truth),elapsed_s=time.perf_counter()-started,
                                x=x[0],y=x[1],z=x[2],bias_norm=np.linalg.norm(x[7:10]),**fit))
        pd.DataFrame(rows).to_csv(output/"metrics.csv",index=False)
        radio.dump(output/"calibration_split_audit.json",calibration_rows)
        print(f"DONE {motion}: {len(rows)} fits",flush=True)
    frame=pd.DataFrame(rows)
    summarize(frame,output)
    radio.dump(output/"checks.json",dict(expected_rows=528,actual_rows=len(frame),
        unique_rows=not frame.duplicated(["sequence","endpoint_s","duration_s","layout","calibration_scope","radio_model"]).any(),
        finite_errors=bool(np.isfinite(frame.position_error_m).all()),maximum_nfev=int(frame.nfev.max()),
        excluded_same_motion_from_all_five_training_fits=True,no_post_evaluation_tuning=True))


if __name__=="__main__":
    main()
