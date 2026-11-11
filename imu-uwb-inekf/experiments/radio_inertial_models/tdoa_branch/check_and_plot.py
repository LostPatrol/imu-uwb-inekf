"""Verify core Jacobians and visualize the fixed real-recording comparison."""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from study import LEVER, group_exp, skew


def check_jacobian():
    """Check the complete lever-arm TDoA invariant Jacobian by central differences."""
    state = group_exp(np.array([.2, -.1, .3, .5, -.3, .1, 2., -1., 1.]))
    anchor_a, anchor_b = np.array([-2., 1., 3.]), np.array([3., -3., .2])
    lag = -.007
    def prediction(value):
        """Evaluate the asynchronous antenna observation at a perturbed state."""
        tag = value[:3, 4] + lag * value[:3, 3] + value[:3, :3] @ LEVER
        return np.linalg.norm(tag - anchor_b) - np.linalg.norm(tag - anchor_a)
    tag = state[:3, 4] + lag * state[:3, 3] + state[:3, :3] @ LEVER
    gradient = (tag - anchor_b) / np.linalg.norm(tag - anchor_b) - (tag - anchor_a) / np.linalg.norm(tag - anchor_a)
    analytic = np.r_[-gradient @ skew(tag), lag * gradient, gradient]
    numeric = np.zeros(9)
    step = 1e-6
    for k in range(9):
        direction = np.zeros(9)
        direction[k] = step
        numeric[k] = (prediction(group_exp(direction) @ state) - prediction(group_exp(-direction) @ state)) / (2 * step)
    error = np.max(np.abs(analytic - numeric))
    assert error < 1e-7, error
    print(f"TDoA invariant Jacobian max error: {error:.3g}")


def check_covariance():
    """Verify OU augmentation matches the dense correlated Gaussian likelihood."""
    times = np.array([0., .01, .04, .12, .18, .21])
    residual = np.array([.1, -.2, .3, .05, .18, -.1])
    sigma, tau, white = .3, 2., .05
    kernel = sigma**2 * np.exp(-np.abs(times[:, None] - times[None]) / tau) + white * np.eye(len(times))
    mean, variance, objective = 0., sigma**2, 0.
    for k in range(len(times)):
        if k:
            phi = np.exp(-(times[k] - times[k - 1]) / tau)
            mean *= phi
            variance = phi**2 * variance + sigma**2 * (1 - phi**2)
        innovation = residual[k] - mean
        total = variance + white
        objective += np.log(total) + innovation**2 / total
        gain = variance / total
        mean += gain * innovation
        variance *= 1 - gain
    dense = np.linalg.slogdet(kernel)[1] + residual @ np.linalg.solve(kernel, residual)
    error = abs(objective - dense)
    assert error < 1e-10, error
    print(f"OU vs dense likelihood error: {error:.3g}")


def plot_results():
    """Plot real RMSE across recording geometries without hidden alignment."""
    folder = Path(__file__).parent / "results_validation"
    frame = pd.read_csv(folder / "metrics.csv")
    names = frame.name.unique()
    modes = ["native", "white_inflated", "pair_ou"]
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    labels = ["Native TDoA + Huber", "Native + marginal white variance", "Pair OU + Huber"]
    width = .24
    for axis, field, ylabel in zip(axes, ["position_rmse", "yaw_rmse_deg"], ["Absolute body position RMSE (m)", "Absolute yaw RMSE (degrees)"]):
        for index, (mode, label) in enumerate(zip(modes, labels)):
            values = frame[frame["mode"] == mode].set_index("name").reindex(names)[field]
            axis.bar(np.arange(len(names)) + (index - 1) * width, values, width=width, label=label)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=.25)
    axes[0].legend(fontsize=9)
    axes[1].set_xticks(np.arange(len(names)), [name.replace("-trial2", "\ntrial2") for name in names], fontsize=8)
    fig.suptitle("Real UTIAS flight data: fixed 2-60 s evaluation; no truth initialization")
    fig.tight_layout()
    fig.savefig(folder / "comparison.png", dpi=160)
    plt.close(fig)
    print(frame.groupby("mode")[["position_rmse", "yaw_rmse_deg", "runtime"]].median().to_string())


if __name__ == "__main__":
    check_jacobian()
    check_covariance()
    plot_results()
