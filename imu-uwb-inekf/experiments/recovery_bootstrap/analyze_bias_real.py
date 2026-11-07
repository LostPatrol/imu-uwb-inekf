"""Independently audit every bias-aware replay and summarize prior/data separation."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def finite_rmse(errors, mask):
    """Compute finite-output error without treating unavailability as zero error."""
    return float(np.sqrt(np.mean(errors[mask]**2))) if mask.any() else None


def main():
    """Recompute published position metrics directly from all retained NPZ samples."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result",type=Path,required=True)
    args=parser.parse_args()
    frame=pd.read_csv(args.result/"metrics.csv")
    assert len(frame)==60
    assert not frame.duplicated(["sequence","method","outage"]).any()
    differences=[]
    attempts=[]
    for row in frame.to_dict("records"):
        identifier=f"{Path(row['sequence']).stem}__{row['method']}__outage{int(row['outage'])}"
        arrays=np.load(args.result/"histories"/(identifier+".npz"))
        times,output,truth=arrays["time"],arrays["output"],arrays["truth"]
        available=np.isfinite(output).all(axis=1)
        error=np.sqrt(np.sum((output[:,:3]-truth)**2,axis=1))
        common=times>=8.1
        tail=times>=times[-1]-5.0
        metrics=dict(common_rmse_m=finite_rmse(error,common&available),
                     output_rmse_m=finite_rmse(error,available),
                     tail5_rmse_m=finite_rmse(error,tail&available),
                     availability=float(available.mean()),common_availability=float(available[common].mean()),
                     max_m=float(error[available].max()) if available.any() else None,
                     p95_m=float(np.percentile(error[available],95)) if available.any() else None)
        # Derive the stable suffix independently of the replay helper implementation.
        origin=25.0 if row["outage"] else 0.0
        after=times>=origin-1e-10
        et,tt=error[after],times[after]
        bad=np.flatnonzero(~np.isfinite(et)|(et>0.5))
        first=int(bad[-1]+1) if bad.size else 0
        success=bool(first<len(tt) and tt[-1]-tt[first]>=5.0-1e-10)
        assert success==bool(row["success"]),(identifier,"success")
        if success:
            differences.append(abs((tt[first]-origin)-row["stable_recovery_s"]))
        for name,value in metrics.items():
            if value is None:
                assert pd.isna(row[name]),(identifier,name)
            else:
                differences.append(abs(value-row[name]))
        detail=json.loads((args.result/"histories"/(identifier+".json")).read_text(encoding="utf-8"))
        for index,estimate in enumerate(detail["estimates"]):
            covariance=np.asarray(estimate["covariance"])
            assert np.linalg.eigvalsh(covariance).min()>0,(identifier,"nonpositive covariance")
            attempts.append(dict(sequence=row["sequence"],method=row["method"],outage=row["outage"],
                attempt=index,accepted=estimate["accepted"],time_s=estimate["time_s"],
                window_s=estimate["window_s"],data_information=estimate["yaw_information_data"],
                prior_conditioned_information=estimate["yaw_information_with_prior"],
                normalized_rms=estimate["normalized_rms"],
                fitted_bias_norm=float(np.linalg.norm(np.asarray(estimate["parameters"])[7:10])),
                covariance_min_eigenvalue=float(np.linalg.eigvalsh(covariance).min())))
    assert max(differences)<1e-8
    summary=frame.groupby(["outage","method"]).agg(success=("success","sum"),
        common_rmse_m=("common_rmse_m","median"),tail5_rmse_m=("tail5_rmse_m","median"),
        maximum_error_m=("max_m","max"),common_availability=("common_availability","mean"),
        startup_latency_s=("startup_unavailable_s","median"),runtime_s=("elapsed_s","median"))
    summary.to_csv(args.result/"summary.csv")
    pd.DataFrame(attempts).to_csv(args.result/"window_information.csv",index=False)
    source=Path(__file__).parent
    audit=dict(verified_rows=len(frame),verified_trajectories=len(frame),
        max_absolute_metric_difference=max(differences),all_success_flags_match=True,
        all_window_covariances_positive_definite=True,
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in [source/"bias_real.py",source/"test_bias_real.py",Path(__file__),
                                 source/"initializer.py",source/"run_research.py"]},
        note="Dependency fingerprints recorded at audit time; base cache is from the initial study.")
    # Verify that the later corrected baseline still consumes byte-identical sensor arrays.
    initial_cache=args.result.parent/"recovery_bootstrap_20261107/cache"
    final_cache=args.result.parent/"recovery_bootstrap_20261107_final/cache"
    aligned=[]
    if final_cache.exists():
        for name in sorted(frame["sequence"].unique()):
            filename=Path(name).stem+".npz"
            original=np.load(initial_cache/filename)
            corrected=np.load(final_cache/filename)
            for field in original.files:
                assert np.array_equal(original[field],corrected[field]),(name,field,"cache mismatch")
            aligned.append(name)
        audit["identical_final_baseline_cache_logs"]=aligned
    (args.result/"verification.json").write_text(json.dumps(audit,indent=2)+"\n",encoding="utf-8")
    print(summary.to_string())
    print(json.dumps(audit,indent=2))


if __name__=="__main__":
    main()
