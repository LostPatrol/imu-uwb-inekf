"""Reproduce the local quadratic-output paper and run same-input Delft modules.

Real-data estimation and scoring load separate files. The yaw labels describe
controlled errors of the shared acceleration input, never estimated yaw RMSE.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

from observer import frozen_step, lift_prior, replay

# Reuse the current raw-data preparation and sensor-only common startup.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/"delft"))
from study import solve_window, window_data, tracker_from_estimate, yaw_rotation
# The input errors and future scoring interval are fixed before running.
INPUT_ERRORS_DEG = (0., 90., -90., 180.)
LAYOUTS = ("single0", "all8", "lower4")
SCORE = (31., 36.)
# The single-absolute-range experiment always uses ID 0, independent of errors.
SINGLE_ANCHOR_ID = 0


def write_json(path, value):
    """Write portable experiment metadata without NumPy scalar serialization errors."""
    path.write_text(json.dumps(value, indent=2, default=lambda x: x.tolist()), encoding="utf-8")


def paper_truth(t):
    """Return the exact trajectory, velocity and inertial acceleration of Section 6."""
    p = np.array([20*np.cos(t)-20, 10*np.sin(2*t)+20, -4*np.cos(4*t)])
    v = np.array([-20*np.sin(t), 20*np.cos(2*t), 16*np.sin(4*t)])
    u = np.array([-20*np.cos(t), -40*np.sin(2*t), 64*np.cos(4*t)])
    return p, v, u


def paper_rhs(t, joined):
    """Integrate Theorem 3 with theta=0 and the exact Section 6 navigation lift."""
    estimate, covariance = joined[:9], joined[9:].reshape(9, 9)
    p, _, u = paper_truth(t)
    system = np.zeros((9, 9))
    system[0, 1], system[1, 2] = 1., 1.
    system[1, 3:6], system[2, 6:9] = u, 2*u
    system[3:6, 6:9] = np.eye(3)
    forcing = np.zeros(9)
    forcing[6:9] = u
    gain = covariance[:, 0]
    rate = system@estimate+forcing+gain*(.5*p@p-estimate[0])
    covariance_rate = system@covariance+covariance@system.T-np.outer(gain, gain)+.0001*np.eye(9)
    return np.r_[rate, covariance_rate.ravel()]


def numerical_example(output):
    """Check exact held-input dynamics and reproduce the noiseless 0–20 s example."""
    output.mkdir(parents=True, exist_ok=True)
    p, v, _ = paper_truth(.31)
    u, h = np.array([.7, -1.2, .4]), .013
    original = np.r_[.5*p@p, p@v, v@v, p, v]
    advanced = frozen_step(original, u, h)[0]
    p2, v2 = p+h*v+.5*h*h*u, v+h*u
    exact = np.r_[.5*p2@p2, p2@v2, v2@v2, p2, v2]
    step_error = float(np.max(np.abs(advanced-exact)))
    stamps = np.linspace(0., 20., 2001)
    start = time.perf_counter()
    sol = solve_ivp(paper_rhs, (0., 20.), np.r_[np.zeros(9), (100*np.eye(9)).ravel()],
                    t_eval=stamps, rtol=2e-9, atol=1e-10, method="DOP853", max_step=.02)
    elapsed = time.perf_counter()-start
    if not sol.success:
        raise RuntimeError(sol.message)
    truth = np.array([np.r_[paper_truth(t)[0], paper_truth(t)[1]] for t in stamps])
    estimate = sol.y[3:9].T
    errors = estimate-truth
    position_error, velocity_error = np.linalg.norm(errors[:, :3], axis=1), np.linalg.norm(errors[:, 3:], axis=1)
    np.savez_compressed(output/"paper_example.npz", time=stamps, truth=truth, estimate=estimate,
                        latent=sol.y[:3].T, covariance=sol.y[9:].T.reshape(-1, 9, 9))
    checks = dict(held_input_exact_step_max_abs=step_error, solver_success=sol.success,
        solver_nfev=sol.nfev, elapsed_s=elapsed, final_position_error_m=float(position_error[-1]),
        final_velocity_error_mps=float(velocity_error[-1]),
        last5_position_rmse_m=float(np.sqrt(np.mean(position_error[stamps>=15]**2))),
        last5_velocity_rmse_mps=float(np.sqrt(np.mean(velocity_error[stamps>=15]**2))),
        dimension_correction="Sec6 prints S0=100 I6,V=.0001 I6,W=I6 although the actual augmented state is 9 and output scalar; use M0=100 I9,V=.0001 I9,W=1.",
        unspecified_initial_estimate="All nine initial estimates set to zero; paper does not state their values.",
        original_example_is_simulation_not_real_flight=True)
    checks["noiseless_convergence_verified"] = bool(position_error[-1]<1e-3 and velocity_error[-1]<1e-3 and step_error<1e-9)
    write_json(output/"paper_checks.json", checks)
    fig = plt.figure(figsize=(11, 3.6))
    ax = fig.add_subplot(131, projection="3d")
    ax.plot(*truth[:, :3].T, label="Exact paper trajectory")
    ax.plot(*estimate[:, :3].T, label="Quadratic observer", alpha=.8)
    ax.set(xlabel="x [m]", ylabel="y [m]", zlabel="z [m]"); ax.legend(fontsize=7)
    for col, values, title in ((2, position_error, "Position error [m]"), (3, velocity_error, "Velocity error [m/s]")):
        ax = fig.add_subplot(1, 3, col)
        ax.semilogy(stamps, np.maximum(values, 1e-12)); ax.set(xlabel="Time [s]", ylabel=title)
        ax.grid(alpha=.3)
    fig.suptitle("Theodosis et al., Section 6: known inertial acceleration, one absolute range")
    fig.tight_layout(); fig.savefig(output/"paper_example.png", dpi=170); plt.close(fig)
    print(json.dumps(checks), flush=True)


def metrics(times, truth, output):
    """Report whole-window errors; any missing output invalidates that window's RMSE."""
    error = output[:, :3]-truth
    result = {}
    for label, mask in dict(prefix=(times>=8.1)&(times<15), gap=(times>=15)&(times<25),
            post_radio=(times>=25), fixed=(times>=SCORE[0])&(times<SCORE[1])).items():
        selected = error[mask]
        valid = np.all(np.isfinite(selected), axis=1)
        result[label+"_availability"] = float(np.mean(valid))
        result[label+"_xy_rmse_m"] = float(np.sqrt(np.mean(np.sum(selected[:, :2]**2, axis=1)))) if valid.all() else float("nan")
        result[label+"_z_rmse_m"] = float(np.sqrt(np.mean(selected[:, 2]**2))) if valid.all() else float("nan")
    return result


def real_data(cache, output, only_sequence=""):
    """Freeze one input/early prior and compare every actual asynchronous radio event."""
    output.mkdir(parents=True, exist_ok=True)
    histories = output/"histories"
    histories.mkdir(exist_ok=True)
    protocol = json.loads((cache/"protocol.json").read_text(encoding="utf-8"))
    model = protocol["model"]
    manifest = dict(source_paper="Local 2006.02688v2.pdf, Section 6 and Theorem 3",
        evaluation=protocol["evaluation"], training=protocol["training"], model=model,
        dataset_sha256=protocol["dataset_sha256"], sensor_delay_s=.2, static_prefix_s=1.5,
        estimator_truth_input=False, truth_only_for_scoring_and_common_raw_valid_crop=True,
        training_truth_for_radio_calibration=True, independent_test_date=False,
        initialization="Shared sensor-only all8 Profile-1 at first t>=2 s; common [p,v] marginal and exact Gaussian quadratic moments.",
        input="Sensor-only gyro integration, startup Profile-1 yaw, constant sensor-derived IMU bias; no radio/attitude update after startup. Error rotation starts at 15 s.",
        input_error_degrees=INPUT_ERRORS_DEG, input_error_is_not_yaw_estimation=True,
        single_anchor_id=SINGLE_ANCHOR_ID, layouts=LAYOUTS, outage_s=[15,25], fixed_score_s=SCORE,
        no_post_outage_resets=True, every_arrived_range_used_once=True,
        radio_policy="No NIS gate/Huber in either module; same calibrated raw positive changed arrivals.",
        quadratic_multi_anchor_is_extension=True,
        quadratic_measurement=".5*(rho_calibrated^2-sigma^2)-.5*a^T a; H=[1,0,0,-a^T,0]; R=||p_hat-a||^2*sigma^2+.5*sigma^4.",
        process="Same p/v noise marginal: held independent acceleration sample variance 4I and 1e-4 I6 per second; lifted rows use exact held-step input Jacobian with additional 1e-4 I3 latent forcing.",
        comparison_limit="Conditional same-input p/v module experiment, not a full attitude+IMU-bias InEKF. Source observer has no yaw output.")
    write_json(output/"protocol.json", manifest)
    rows = []
    for name in protocol["evaluation"]:
        if only_sequence and only_sequence not in name:
            continue
        data = dict(np.load(cache/(Path(name).stem+".npz")))
        stamps = data["time"]
        first = int(np.searchsorted(stamps, 2.))
        estimate = solve_window(*window_data(data, 0, first, model), method="profile")
        core = tracker_from_estimate(estimate, data["rotations"][first], data["bias"], model)
        mean = np.r_[core.X[:3,4], core.X[:3,3]]
        physical = core.physical_covariance()
        order = np.r_[6:9,3:6]
        prior = physical[np.ix_(order,order)]
        lifted, lifted_covariance = lift_prior(mean, prior)
        rotation = np.einsum("ij,njk->nik", yaw_rotation(estimate.parameters[6]), data["rotations"])
        force = data["imu"][:,:3]*9.81-data["bias"][3:]
        world_force = np.einsum("nij,nj->ni", rotation, force)
        for yaw in INPUT_ERRORS_DEG:
            acceleration = world_force.copy()
            perturb = stamps>=15.
            acceleration[perturb] = acceleration[perturb]@yaw_rotation(np.deg2rad(yaw)).T
            acceleration += np.array([0.,0.,-9.81])
            for layout in LAYOUTS:
                active = np.zeros(len(data["anchors"]), dtype=bool)
                ids = [SINGLE_ANCHOR_ID] if layout=="single0" else (range(len(active)) if layout=="all8" else np.argsort(data["anchors"][:,2])[:4])
                active[list(ids)] = True
                begun = time.perf_counter()
                lifted_history, base_history, latent, counts, failed, failure_time = replay(stamps, acceleration,
                    data["ranges"], data["changed"], data["anchors"], np.array(model["range_bias"]),
                    np.array(model["range_sigma"]), first, lifted, lifted_covariance, active)
                elapsed = time.perf_counter()-begun
                # Evaluation truth is deliberately accessed only after the estimator completes.
                truth = np.load(cache/(Path(name).stem+"_scoring.npz"))["position"]
                identifier = Path(name).stem+"__"+layout+"__yaw"+str(int(yaw))
                np.savez_compressed(histories/(identifier+".npz"), time=stamps, truth=truth,
                    quadratic=lifted_history, range_ekf=base_history, latent=latent, world_acceleration=acceleration,
                    initial_mean=mean, initial_covariance=prior, initial_lifted=lifted,
                    initial_lifted_covariance=lifted_covariance, active_anchors=np.flatnonzero(active))
                for j,(method,history) in enumerate((("quadratic",lifted_history),("range_ekf",base_history))):
                    row = dict(sequence=name,layout=layout,input_error_deg=yaw,method=method,
                        events=int(counts[j]),failed=bool(failed[j]),failure_time_s=float(failure_time[j]),
                        elapsed_pair_s=elapsed,initial_time_s=float(stamps[first]),yaw_truth_available=False)
                    row.update(metrics(stamps,truth,history)); rows.append(row)
                print(f"DONE {identifier} quadXY={rows[-2]['fixed_xy_rmse_m']:.4g} EKFXY={rows[-1]['fixed_xy_rmse_m']:.4g} {elapsed:.2f}s",flush=True)
        pd.DataFrame(rows).to_csv(output/"metrics.csv", index=False)
    frame = pd.DataFrame(rows)
    summary = frame.groupby(["layout","input_error_deg","method"])[["fixed_xy_rmse_m","fixed_z_rmse_m","post_radio_xy_rmse_m","post_radio_z_rmse_m"]].median()
    summary.to_csv(output/"summary.csv")
    frame.groupby(["layout","method"])[["fixed_xy_rmse_m","fixed_z_rmse_m"]].agg(["median","max"]).to_csv(output/"aggregate.csv")
    plot_results(frame, output)


def plot_results(frame, output):
    """Show native single-anchor and explicitly extended multi-anchor paired results."""
    fig, axes = plt.subplots(2,3,figsize=(12,6),sharex=True)
    for col,layout in enumerate(LAYOUTS):
        for row,metric in enumerate(("fixed_xy_rmse_m","fixed_z_rmse_m")):
            ax = axes[row,col]
            for method,style in (("quadratic","o-"),("range_ekf","s--")):
                selected=frame[(frame.layout==layout)&(frame.method==method)]
                stats=selected.groupby("input_error_deg")[metric].median().reindex(INPUT_ERRORS_DEG)
                ax.plot(range(4),stats,style,label=method)
            ax.set_yscale("log"); ax.grid(alpha=.3)
            ax.set_xticks(range(4),["0","+90","-90","180"])
            if col==0: ax.set_ylabel("XY RMSE [m]" if row==0 else "Z RMSE [m]")
            if row==0: ax.set_title("Native single anchor ID0" if col==0 else "Multi-anchor extension: "+layout)
            else: ax.set_xlabel("Shared input heading error [deg]")
            ax.legend(fontsize=8)
    fig.suptitle("Real Delft TWR, median across six late recordings, fixed 31–36 s\nSame sensor-only world input + same early p/v prior; no yaw is estimated by either module")
    fig.tight_layout(); fig.savefig(output/"delft_panels.png",dpi=170); plt.close(fig)
    # A predetermined recording provides full time histories, including the outage.
    first=frame.sequence.iloc[0]
    fig,axes=plt.subplots(2,3,figsize=(12,6),sharex=True)
    for col,layout in enumerate(LAYOUTS):
        data=np.load(output/"histories"/(Path(first).stem+"__"+layout+"__yaw90.npz"))
        for method in ("quadratic","range_ekf"):
            error=data[method][:,:3]-data["truth"]
            axes[0,col].plot(data["time"],np.linalg.norm(error[:,:2],axis=1),label=method)
            axes[1,col].plot(data["time"],np.abs(error[:,2]),label=method)
        for row in range(2):
            axes[row,col].axvspan(15,25,color="grey",alpha=.15)
            axes[row,col].axvspan(31,36,color="green",alpha=.08)
            axes[row,col].grid(alpha=.3); axes[row,col].set_yscale("log"); axes[row,col].legend(fontsize=8)
        axes[0,col].set_title(layout); axes[1,col].set_xlabel("Time [s]")
    axes[0,0].set_ylabel("XY error [m]"); axes[1,0].set_ylabel("Absolute Z error [m]")
    fig.suptitle(Path(first).stem+": shared +90 deg input error from 15 s")
    fig.tight_layout(); fig.savefig(output/"delft_example.png",dpi=170); plt.close(fig)


def main():
    """Run reproducible source-paper checks followed by the frozen real-data panels."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache",type=Path,default=HERE.parent/"delft"/"results"/"cache")
    parser.add_argument("--output",type=Path,default=HERE/"results")
    parser.add_argument("--simulation-only",action="store_true")
    parser.add_argument("--sequence",default="")
    args=parser.parse_args()
    numerical_example(args.output)
    if not args.simulation_only:
        real_data(args.cache,args.output,args.sequence)
    source={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob("*.py")}
    write_json(args.output/"source_sha256.json",source)


if __name__=="__main__":
    main()
