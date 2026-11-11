"""Consolidate descriptive radio-model results without selecting test parameters."""

import hashlib
import inspect
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import scipy

from study import (HERE, DATASET, dump, load_real_log, load_anchors,
                   estimate_initial_rotation_and_bias, preintegrate, cv_candidate)


def main():
    """Write concise reusable tables, correlation summaries and explicit limits."""
    output=HERE/"results"
    protocol=json.loads((output/"protocol.json").read_text())
    frame=pd.read_csv(HERE/"results_ar_ablation"/"combined_metrics.csv")
    frame.groupby(["layout","model"]).position_error_m.agg(
        median="median",p95=lambda x:x.quantile(.95),rmse=lambda x:np.sqrt(np.mean(x*x)),
        maximum="max").to_csv(output/"all_models_summary.csv")
    keys=["sequence","endpoint_s","duration_s","layout"]
    paired=frame.pivot(index=keys,columns="model",values="position_error_m")
    comparisons=[]
    for model in frame.model.unique():
        if model=="static":
            continue
        for layout,part in paired.groupby(level="layout"):
            diff=part[model]-part.static
            comparisons.append(dict(model=model,layout=layout,wins=int((diff<0).sum()),
                windows=len(diff),median_delta_m=diff.median(),max_degradation_m=diff.max(),
                max_improvement_m=diff.min()))
    lags=pd.read_csv(output/"lag_correlation.csv")
    cross=pd.read_csv(output/"cross_anchor_correlation_100ms.csv")
    transfer=pd.read_csv(output/"transfer_summary.csv")
    notes=dict(
        observations="Changed positive raw ranges only; six chronological training/evaluation pairs; 0.20 s common sensor correction",
        model_equation="y=d+b_a+c_a(d-dbar_a)+k_a.T*u+e; standardized e AR factor phi_a(dt)=rho_a^(dt/median_dt_a); robust Huber innovation fit",
        main_candidate="geometry_ar",
        ar_ablation="static_ar refits rho from static training residuals; same clipping and event interval",
        comparison= comparisons,
        heldout_radio_mae_by_log=transfer[transfer.split=="evaluation"].to_dict("records"),
        same_anchor_lag1_median=lags[lags.lag_events==1].groupby("split").correlation.median().to_dict(),
        same_anchor_lag32_median=lags[lags.lag_events==32].groupby("split").correlation.median().to_dict(),
        cross_anchor_100ms_median=cross.groupby("split").correlation.median().to_dict(),
        limits=["Six same-day same-room heldout logs, already development data; no independent blind claim",
            "Overlapping endpoint windows and two anchor layouts are correlated; no iid significance claim",
            "No NLOS or RF-quality labels; persistent positive residuals are not identified NLOS",
            "World-LOS mean corrections can absorb anchor survey/antenna/environment/time alignment errors; no unique physical attribution",
            "No calibrated posterior or online navigation validation; these are finite endpoint-window solves",
            "All models receive equal seed count/nfev cap, but AR residual evaluation has extra O(events) work and higher wall time",
            "Whitening can suppress smooth inertial model mismatch as well as correlated radio error; causal radio-only evidence supports correlation, not a full attribution",
            "Observed-range variance bins are only a coarse proxy for true distance bins used during training",
            "Large 8-second errors remain; radio modeling does not repair the inertial trajectory family"],
        reproducibility="study.py then analyze.py then ablate_ar.py then consolidate.py; root .venv; results/source_snapshot.py matches original protocol source hash",
        source_snapshot_sha256=hashlib.sha256((output/"source_snapshot.py").read_bytes()).hexdigest())
    dump(output/"conclusions.json",notes)
    dependencies={Path(inspect.getfile(function)) for function in (
        load_real_log,load_anchors,estimate_initial_rotation_and_bias,preintegrate,cv_candidate)}
    dependencies.update(DATASET/"twr"/name for name in protocol["training"]+protocol["evaluation"])
    dependencies.add(DATASET/"anchor_positions.yaml")
    dump(output/"integrity.json",dict(python=platform.python_version(),numpy=np.__version__,
        scipy=scipy.__version__,pandas=pd.__version__,
        snapshot_matches_protocol=notes["source_snapshot_sha256"]==protocol["source_sha256"],
        source_and_data_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest()
                                 for path in sorted(dependencies)}))


if __name__=="__main__":
    main()
