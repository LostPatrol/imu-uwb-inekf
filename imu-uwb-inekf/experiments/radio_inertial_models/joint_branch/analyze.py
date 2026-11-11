"""Consolidate matched K1/K2/bias radio interactions after joint fitting ends."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

# Only newly generated branch results are read for this descriptive consolidation.
HERE=Path(__file__).resolve().parent


def main():
    """Write full model hierarchy and pairwise radio effects without retuning."""
    output=HERE/"results"
    joint=pd.read_csv(output/"metrics.csv")
    radio=HERE.parent/"radio_branch"
    original=pd.read_csv(radio/"results_ar_ablation"/"combined_metrics.csv")
    original=original[original.model.isin(joint.radio_model.unique())].rename(columns={"model":"radio_model"})
    original["trajectory"]="K1"
    frame=pd.concat([original,joint],ignore_index=True)
    frame.to_csv(output/"hierarchy_metrics.csv",index=False)
    for groups,name in ((["trajectory","layout","radio_model"],"hierarchy_summary"),
        (["trajectory","duration_s","layout","radio_model"],"hierarchy_duration")):
        frame.groupby(groups).position_error_m.agg(n="size",median="median",p95=lambda x:x.quantile(.95),
            rmse=lambda x:np.sqrt(np.mean(x*x)),maximum="max").to_csv(output/(name+".csv"))
    pairs=[]
    log_pairs=[]
    keys=["trajectory","sequence","endpoint_s","duration_s","layout"]
    pivot=frame.pivot(index=keys,columns="radio_model",values="position_error_m")
    for key,part in pivot.groupby(level=["trajectory","layout"]):
        for first,second in (("static","geometry"),("static","static_ar"),
                             ("static","geometry_ar"),("geometry","geometry_ar")):
            delta=part[second]-part[first]
            perlog=part.groupby(level="sequence").agg(lambda x:np.sqrt(np.mean(x*x)))
            pairs.append(dict(trajectory=key[0],layout=key[1],reference=first,candidate=second,
                endpoint_wins=int((delta<0).sum()),endpoints=len(delta),
                median_delta_m=delta.median(),max_degradation_m=delta.max(),
                rmse_reference_m=np.sqrt(np.mean(part[first]**2)),
                rmse_candidate_m=np.sqrt(np.mean(part[second]**2)),
                log_rmse_wins=int((perlog[second]<perlog[first]).sum()),logs=len(perlog)))
            for sequence,block in part.groupby(level="sequence"):
                local=block[second]-block[first]
                log_pairs.append(dict(trajectory=key[0],layout=key[1],sequence=sequence,
                    reference=first,candidate=second,endpoint_wins=int((local<0).sum()),
                    endpoints=len(local),median_delta_m=local.median(),max_degradation_m=local.max(),
                    rmse_reference_m=np.sqrt(np.mean(block[first]**2)),
                    rmse_candidate_m=np.sqrt(np.mean(block[second]**2))))
    pd.DataFrame(pairs).to_csv(output/"radio_effects_by_trajectory.csv",index=False)
    pd.DataFrame(log_pairs).to_csv(output/"per_log_radio_effects.csv",index=False)
    joint.groupby(["trajectory","layout","radio_model"]).nuisance_norm.agg(
        median="median",p95=lambda x:x.quantile(.95)).to_csv(output/"nuisance_norms.csv")
    checks=dict(expected_joint_fits=1056,actual_joint_fits=len(joint),
        unique_joint_fits=not joint.duplicated(keys+["radio_model"]).any(),
        finite_joint_errors=bool(np.isfinite(joint.position_error_m).all()),
        maximum_nfev=int(joint.nfev.max()),maximum_seed_count=4,
        hierarchy_rows=len(frame),gaussian_bias_prior_std_mps2=.10,
        truth_usage_clarification="The inherited loader uses valid OptiTrack timestamps to crop available intervals; actual positions only score endpoints. Training truth calibrated radio models. No evaluation truth enters residuals, seeds, online weights or priors.",
        scope="Same six chronological recording pairs; correlated overlapping windows; no blind or iid inference")
    (output/"checks.json").write_text(json.dumps(checks,indent=2),encoding="utf-8")
    protocol=json.loads((output/"protocol.json").read_text())
    # Correct the shorthand runtime flag: loader cropping also uses truth availability.
    protocol["truth_used_only_for_scoring"]=False
    protocol["truth_usage_clarification"]=checks["truth_usage_clarification"]
    (output/"protocol.json").write_text(json.dumps(protocol,indent=2),encoding="utf-8")


if __name__=="__main__":
    main()
