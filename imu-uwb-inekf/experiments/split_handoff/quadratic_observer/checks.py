"""Check source-observer asymptotic convergence and real same-input replay parity."""

import json
import importlib.util
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

# Shared Delft code also imports a module named run; use a distinct module key.
SPEC = importlib.util.spec_from_file_location("quadratic_run", Path(__file__).with_name("run.py"))
RUN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUN)
HERE, paper_rhs, paper_truth, write_json = RUN.HERE, RUN.paper_rhs, RUN.paper_truth, RUN.write_json


def main():
    """Retain the original 20 s result and independently extend the same observer."""
    output = HERE/"results"
    original = np.load(output/"paper_example.npz")
    initial = np.r_[original["latent"][-1], original["estimate"][-1], original["covariance"][-1].ravel()]
    stamps = np.linspace(20., 100., 4001)
    begun = time.perf_counter()
    sol = solve_ivp(paper_rhs, (20., 100.), initial, t_eval=stamps, rtol=2e-9,
                    atol=1e-10, method="DOP853", max_step=.02)
    truth = np.array([np.r_[paper_truth(t)[0],paper_truth(t)[1]] for t in stamps])
    estimate = sol.y[3:9].T
    position_error = np.linalg.norm(estimate[:,:3]-truth[:,:3],axis=1)
    velocity_error = np.linalg.norm(estimate[:,3:]-truth[:,3:],axis=1)
    np.savez_compressed(output/"paper_continuation.npz", time=stamps, truth=truth,
                        estimate=estimate, position_error=position_error, velocity_error=velocity_error)
    results = dict(continuation_not_in_original_paper=True, interval_s=[20,100],
        same_initial_state_and_covariance_as_20s_endpoint=True, solver_success=bool(sol.success),
        elapsed_s=time.perf_counter()-begun, final_position_error_m=float(position_error[-1]),
        final_velocity_error_mps=float(velocity_error[-1]),
        finite=bool(np.all(np.isfinite(sol.y))), original_20s_retained=True,
        asymptotic_threshold_1mm_pass=bool(position_error[-1]<.001 and velocity_error[-1]<.001))
    for endpoint in (40.,60.,80.,100.):
        k=int(np.searchsorted(stamps,endpoint))
        results[str(int(endpoint))+"s_position_error_m"] = float(position_error[k])
        results[str(int(endpoint))+"s_velocity_error_mps"] = float(velocity_error[k])
    frame = pd.read_csv(output/"metrics.csv")
    pairs = frame.groupby(["sequence","layout","input_error_deg"])
    results["real_cases"] = len(pairs)
    results["same_event_count_all_pairs"] = bool((pairs.events.nunique()==1).all())
    results["real_failed_outputs"] = int(frame.failed.sum())
    results["real_complete_fixed_window_outputs"] = int((frame.fixed_availability==1).sum())
    results["no_failed_case_dropped"] = bool(len(frame)==2*len(pairs))
    # The exact p/v marginal is copied into both filters; verify every saved prior.
    prior_errors=[]
    for path in (output/"histories").glob("*.npz"):
        data=np.load(path)
        prior_errors.append(np.max(np.abs(data["initial_covariance"]-data["initial_lifted_covariance"][3:,3:])))
    results["maximum_initial_pv_covariance_mismatch"] = float(max(prior_errors))
    write_json(output/"validation.json",results)
    fig,ax=plt.subplots(figsize=(8,3.6))
    ax.semilogy(stamps,position_error,label="Position error [m]")
    ax.semilogy(stamps,velocity_error,label="Velocity error [m/s]")
    ax.axhline(.001,color="grey",linestyle=":",label="1e-3 reference")
    ax.set(xlabel="Time [s]",ylabel="Error",title="Additional continuation of the same observer; not the paper's original 20 s plot")
    ax.grid(alpha=.3); ax.legend(); fig.tight_layout()
    fig.savefig(output/"paper_continuation.png",dpi=170); plt.close(fig)
    print(json.dumps(results),flush=True)


if __name__=="__main__":
    main()
