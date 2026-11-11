"""Isolate training-only correlation whitening from geometric mean calibration."""

import json

import numpy as np
import pandas as pd

from study import HERE, DATASET, DELAY, load_real_log, load_anchors, experiment, dump


def main():
    """Run the same frozen windows and seed budget with static means plus AR."""
    base=HERE/"results"
    output=HERE/"results_ar_ablation"
    output.mkdir(exist_ok=True)
    protocol=json.loads((base/"protocol.json").read_text())
    values=json.loads((base/"calibration.json").read_text())
    calibration={key:np.asarray(value) if isinstance(value,list) else value for key,value in values.items()}
    # Refit only AR coefficients from static training residuals, with identical clipping.
    train=pd.read_csv(base/"training_events.csv")
    rho=[]
    for anchor in range(8):
        pairs=[]
        for _,part in train[train.anchor==anchor].groupby("sequence"):
            e=np.clip(part.static_error.to_numpy()/calibration["sigma"][anchor],-3.,3.)
            pairs.append(np.column_stack((e[:-1],e[1:])))
        pair=np.vstack(pairs)
        rho.append(np.clip(np.sum(pair[:,0]*pair[:,1])/np.sum(pair[:,0]**2),0.,.9))
    calibration["rho"]=np.array(rho)
    dump(output/"calibration.json",calibration)
    logs={name:load_real_log(DATASET,name,sensor_delay_s=DELAY) for name in protocol["evaluation"]}
    experiment(logs,load_anchors(DATASET),calibration,output,models=("static_ar",))
    original=pd.read_csv(base/"endpoint_metrics.csv")
    ablation=pd.read_csv(output/"endpoint_metrics.csv")
    combined=pd.concat([original,ablation],ignore_index=True)
    combined.to_csv(output/"combined_metrics.csv",index=False)
    keys=["sequence","endpoint_s","duration_s","layout"]
    paired=combined.pivot(index=keys,columns="model",values="position_error_m")
    paired.to_csv(output/"paired_all_models.csv")
    dump(output/"protocol.json",dict(parent_protocol=str(base/"protocol.json"),
        purpose="Disentangle trained correlation whitening from geometry mean correction",
        same_fixed_train_test=True,same_initial_seeds=True,same_max_nfev=True,
        static_ar_rho=rho,truth_used_only_for_scoring=True))


if __name__=="__main__":
    main()
